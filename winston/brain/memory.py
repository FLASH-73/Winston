"""
WINSTON Three-Tier Memory System

Tier 1 — Working Memory (in-process, dies with session)
    Last N observations, conversation turns, session narrative.
    Zero-cost reads, no DB or API calls.

Tier 2 — Episodic Memory (ChromaDB)
    All observations/conversations with importance scoring (1-10).
    Enhanced recall with time-decay weighting + importance boosting.
    Deduplication to prevent identical observations piling up.
    Session summaries generated at shutdown via Claude Haiku.

Tier 3 — Semantic Memory (JSON file + ChromaDB)
    Structured facts about Roberto: preferences, equipment, projects, habits.
    Extracted via Claude Haiku after conversations and at session end.
    Always included in every API call (~100-300 tokens, negligible cost).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import chromadb

from config import (
    MEMORY_CONSOLIDATE_AFTER_DAYS,
    MEMORY_CONSOLIDATE_MIN_IMPORTANCE,
    MEMORY_CONTEXT_BUDGET_CONVERSATION,
    MEMORY_CONTEXT_BUDGET_LIGHTWEIGHT,
    MEMORY_CONTEXT_BUDGET_NARRATIVE,
    MEMORY_CONTEXT_BUDGET_PROACTIVE,
    MEMORY_DB_PATH,
    MEMORY_DEDUP_THRESHOLD,
    MEMORY_DEDUP_WINDOW_SECONDS,
    MEMORY_EPISODES_COLLECTION,
    MEMORY_SEMANTIC_COLLECTION,
    MEMORY_SUMMARIES_COLLECTION,
    MEMORY_USER_PROFILE_FILE,
    SYSTEM_PROMPT_FACT_EXTRACTION,
    SYSTEM_PROMPT_SESSION_SUMMARY,
    WORKING_MEMORY_MAX_CONVERSATIONS,
    WORKING_MEMORY_MAX_OBSERVATIONS,
)

logger = logging.getLogger("winston.memory")

_TEMPORAL_QUERY_PATTERNS = re.compile(
    r"yesterday|last session|last time|this morning|this afternoon|"
    r"how long|when did|earlier today|"
    r"gestern|letzte sitzung|letztes mal|heute morgen|wie lange|wann hast",
    re.IGNORECASE,
)

_VAGUE_ENTITY_PATTERNS = re.compile(
    r"^(unknown_?\w*|unidentified_?\w*|unnamed_?\w*"
    r"|he|she|they|it|him|her|them"
    r"|the person|someone|a friend|a person|some guy|some girl"
    r"|a man|a woman|the man|the woman)$",
    re.IGNORECASE,
)

# Fact attributes that define entity aliases/relationships
_RELATIONSHIP_ATTRS = frozenset({
    "relationship", "relation", "role", "alias", "also_known_as", "name",
})


# ---------------------------------------------------------------------------
# Tier 1: Working Memory (in-process, ephemeral)
# ---------------------------------------------------------------------------


class WorkingMemory:
    """Fast, in-process memory for the current session. No DB or API calls."""

    def __init__(self):
        self.observations: deque[dict] = deque(maxlen=WORKING_MEMORY_MAX_OBSERVATIONS)
        self.conversations: list[dict] = []
        self.proactive_spoken: list[dict] = []
        self.session_start = datetime.now()
        self._max_conversations = WORKING_MEMORY_MAX_CONVERSATIONS

    def add_observation(self, text: str, activity: str = "") -> None:
        """Record a scene observation with timestamp."""
        self.observations.append(
            {
                "text": text,
                "activity": activity,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def add_conversation(self, question: str, answer: str) -> None:
        """Record a Q/A exchange. Bounded to last N turns."""
        self.conversations.append(
            {
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
            }
        )
        # Keep bounded
        if len(self.conversations) > self._max_conversations:
            self.conversations = self.conversations[-self._max_conversations :]

    def add_proactive(self, message: str) -> None:
        """Record a proactive comment Winston made."""
        self.proactive_spoken.append(
            {
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_recent_observations(self, n: int = 5) -> list[str]:
        """Return the text of the last N observations."""
        return [o["text"] for o in list(self.observations)[-n:]]

    def get_conversation_context(self) -> str:
        """Format recent conversation turns as readable text."""
        if not self.conversations:
            return ""
        lines = []
        for turn in self.conversations[-5:]:
            lines.append(f"Roberto: {turn['question']}")
            lines.append(f"Winston: {turn['answer']}")
        return "\n".join(lines)

    def get_session_narrative(self) -> str:
        """Build a concise text summary of what happened this session."""
        parts = []
        duration = datetime.now() - self.session_start
        minutes = int(duration.total_seconds() / 60)
        parts.append(f"Session duration: {minutes} minutes.")

        if self.observations:
            recent = [o["text"] for o in list(self.observations)[-3:]]
            parts.append("Recent observations: " + "; ".join(recent))

        if self.conversations:
            parts.append(f"Conversations this session: {len(self.conversations)}.")
            last = self.conversations[-1]
            parts.append(f"Last exchange: Q: {last['question'][:80]} A: {last['answer'][:80]}")

        if self.proactive_spoken:
            parts.append(f"Proactive comments made: {len(self.proactive_spoken)}.")

        return " ".join(parts)

    def get_all_text_for_summary(self) -> str:
        """Return all session content as text for summarization."""
        lines = []
        for obs in self.observations:
            lines.append(f"[Observation] {obs['text']}")
        for conv in self.conversations:
            lines.append(f"[Conversation] Q: {conv['question']} A: {conv['answer']}")
        for p in self.proactive_spoken:
            lines.append(f"[Proactive] {p['message']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier 2: Episodic Memory (ChromaDB — persistent across sessions)
# ---------------------------------------------------------------------------


class EpisodicMemory:
    """Persistent memory for observations and conversations with importance scoring."""

    # Keyword patterns for importance scoring
    _SAFETY_KEYWORDS = re.compile(
        r"safety|danger|warning|careful|hot|burn|sharp|eye protection|glasses|fire|shock|"
        r"soldering iron left|unplug|hazard",
        re.IGNORECASE,
    )
    _MEASUREMENT_KEYWORDS = re.compile(
        r"\d+\s*(mm|cm|m|inch|V|mV|A|mA|ohm|kg|g|lb|rpm|Hz|kHz|MHz|"
        r"degrees|°|N[·.]?m|torque)",
        re.IGNORECASE,
    )
    _MILESTONE_KEYWORDS = re.compile(
        r"fixed|solved|completed|finished|works now|success|done|resolved|"
        r"assembled|printed|calibrated",
        re.IGNORECASE,
    )
    _PREFERENCE_KEYWORDS = re.compile(
        r"prefer|decided|switched to|always use|better with|I like|I use|"
        r"my go-to|I usually",
        re.IGNORECASE,
    )
    _CORRECTION_KEYWORDS = re.compile(
        r"wrong|mistake|should have|shouldn't|correct way|actually|"
        r"not that|the other|I meant",
        re.IGNORECASE,
    )

    def __init__(self):
        self._client: Optional[chromadb.ClientAPI] = None
        self._episodes: Optional[chromadb.Collection] = None
        self._summaries: Optional[chromadb.Collection] = None
        self._session_id = str(uuid.uuid4())[:8]
        self._counter = 0

    def initialize(self, client: chromadb.ClientAPI) -> None:
        self._client = client
        self._episodes = client.get_or_create_collection(name=MEMORY_EPISODES_COLLECTION)
        self._summaries = client.get_or_create_collection(name=MEMORY_SUMMARIES_COLLECTION)
        logger.info("Episodic memory: %d episodes, %d summaries", self._episodes.count(), self._summaries.count())

    def store(self, text: str, entry_type: str = "observation", activity: str = "") -> None:
        """Store an entry with auto-computed importance. Skips duplicates."""
        if not self._episodes or not text.strip():
            return

        # Deduplication check
        if self._is_duplicate(text):
            logger.debug("Skipping duplicate memory entry")
            return

        importance = self._compute_importance(text, entry_type)
        self._counter += 1
        timestamp = datetime.now()
        doc_id = f"{self._session_id}_{self._counter:06d}"

        formatted = f"{timestamp.strftime('%Y-%m-%d %H:%M')} — {text}"
        meta = {
            "type": entry_type,
            "timestamp": timestamp.isoformat(),
            "timestamp_unix": time.time(),
            "session_id": self._session_id,
            "activity": activity,
            "importance": importance,
        }

        try:
            self._episodes.add(documents=[formatted], ids=[doc_id], metadatas=[meta])
            logger.debug("Stored episode (importance=%d): %.60s", importance, text)
        except Exception as e:
            logger.error("Failed to store episode: %s", e)

    def recall(self, query: str, n_results: int = 5) -> list[str]:
        """Semantic search with time-decay and importance re-ranking."""
        if not self._episodes or self._episodes.count() == 0:
            return []

        try:
            # Over-fetch 3x to allow re-ranking
            fetch_n = min(n_results * 3, self._episodes.count())
            results = self._episodes.query(
                query_texts=[query],
                n_results=fetch_n,
                include=["documents", "metadatas", "distances"],
            )

            if not results or not results["documents"] or not results["documents"][0]:
                return []

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            # Re-rank with composite scoring
            scored = []
            now = time.time()
            for doc, meta, dist in zip(docs, metas, distances):
                # Semantic similarity (ChromaDB distance → similarity)
                semantic_score = max(0.0, 1.0 - dist)

                # Time decay: halve score every 7 days
                age_days = (now - meta.get("timestamp_unix", now)) / 86400.0
                time_decay = 0.5 ** (age_days / 7.0)

                # Importance factor
                imp = meta.get("importance", 5)
                importance_factor = 0.5 + (imp / 10.0)

                composite = semantic_score * time_decay * importance_factor
                scored.append((doc, composite))

            # Sort by composite score descending, return top n
            scored.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored[:n_results]]

        except Exception as e:
            logger.error("Episodic recall failed: %s", e)
            return []

    def get_recent(self, n: int = 10) -> list[str]:
        """Get last N entries by insertion order."""
        if not self._episodes:
            return []
        try:
            total = self._episodes.count()
            if total == 0:
                return []
            results = self._episodes.get(limit=n, offset=max(0, total - n))
            if results and results["documents"]:
                return results["documents"]
            return []
        except Exception as e:
            logger.error("Failed to get recent episodes: %s", e)
            return []

    def generate_session_summary(self, llm_client, session_text: str) -> Optional[str]:
        """Generate and store a session summary using Claude Haiku."""
        if not session_text.strip():
            return None
        if hasattr(llm_client, "is_available") and not llm_client.is_available():
            logger.warning("[memory] Skipping session summary: API circuit breaker open")
            return None

        try:
            summary = llm_client.text_only_chat(
                prompt=f"Session content:\n{session_text}",
                system_prompt=SYSTEM_PROMPT_SESSION_SUMMARY,
                max_tokens=300,
            )
            if summary:
                self._store_summary(summary)
                logger.info("Session summary generated and stored")
                return summary
        except Exception as e:
            logger.error("Failed to generate session summary: %s", e)
        return None

    def _store_summary(self, summary: str) -> None:
        """Store a session summary in the summaries collection."""
        if not self._summaries:
            return
        doc_id = f"summary_{self._session_id}"
        meta = {
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "timestamp_unix": time.time(),
        }
        try:
            self._summaries.add(documents=[summary], ids=[doc_id], metadatas=[meta])
        except Exception as e:
            logger.error("Failed to store session summary: %s", e)

    def get_past_summaries(self, n: int = 3) -> list[str]:
        """Get recent session summaries for context."""
        if not self._summaries or self._summaries.count() == 0:
            return []
        try:
            total = self._summaries.count()
            results = self._summaries.get(limit=n, offset=max(0, total - n))
            if results and results["documents"]:
                return results["documents"]
            return []
        except Exception as e:
            logger.error("Failed to get past summaries: %s", e)
            return []

    def consolidate(self) -> int:
        """Delete low-importance old entries from sessions that have summaries.

        Returns the number of entries deleted.
        """
        if not self._episodes or not self._summaries:
            return 0

        cutoff = time.time() - (MEMORY_CONSOLIDATE_AFTER_DAYS * 86400)
        deleted = 0

        try:
            # Get all session IDs that have summaries
            all_summaries = self._summaries.get(include=["metadatas"])
            if not all_summaries or not all_summaries["metadatas"]:
                return 0
            summarized_sessions = {m["session_id"] for m in all_summaries["metadatas"]}

            # Get old episodes
            all_episodes = self._episodes.get(include=["metadatas"])
            if not all_episodes or not all_episodes["ids"]:
                return 0

            ids_to_delete = []
            for doc_id, meta in zip(all_episodes["ids"], all_episodes["metadatas"]):
                ts = meta.get("timestamp_unix", time.time())
                imp = meta.get("importance", 5)
                sid = meta.get("session_id", "")

                # Only delete if: old enough, session is summarized, low importance
                if ts < cutoff and sid in summarized_sessions and imp < MEMORY_CONSOLIDATE_MIN_IMPORTANCE:
                    ids_to_delete.append(doc_id)

            if ids_to_delete:
                # ChromaDB delete in batches
                batch_size = 100
                for i in range(0, len(ids_to_delete), batch_size):
                    batch = ids_to_delete[i : i + batch_size]
                    self._episodes.delete(ids=batch)
                deleted = len(ids_to_delete)
                logger.info("Consolidated %d old low-importance entries", deleted)

        except Exception as e:
            logger.error("Consolidation failed: %s", e)

        return deleted

    @property
    def episode_count(self) -> int:
        if self._episodes:
            try:
                return self._episodes.count()
            except Exception:
                return 0
        return 0

    def _compute_importance(self, text: str, entry_type: str) -> int:
        """Compute importance score 1-10 using keyword matching."""
        score = 4 if entry_type == "observation" else 5

        if self._SAFETY_KEYWORDS.search(text):
            score = max(score, 8)
        if self._CORRECTION_KEYWORDS.search(text):
            score = max(score, 9)
        if self._MILESTONE_KEYWORDS.search(text):
            score = max(score, 7)
        if self._PREFERENCE_KEYWORDS.search(text):
            score = max(score, 7)
        if self._MEASUREMENT_KEYWORDS.search(text):
            score = max(score, 6)

        # Conversations are inherently more important
        if entry_type == "conversation":
            score = max(score, 5)

        return min(score, 10)

    def _is_duplicate(self, text: str) -> bool:
        """Check if a very similar entry was stored recently."""
        if not self._episodes or self._episodes.count() == 0:
            return False
        try:
            results = self._episodes.query(
                query_texts=[text],
                n_results=1,
                include=["metadatas", "distances"],
            )
            if (
                results
                and results["distances"]
                and results["distances"][0]
                and results["distances"][0][0] < MEMORY_DEDUP_THRESHOLD
            ):
                # Check if within time window
                meta = results["metadatas"][0][0]
                ts = meta.get("timestamp_unix", 0)
                if (time.time() - ts) < MEMORY_DEDUP_WINDOW_SECONDS:
                    return True
            return False
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Tier 3: Semantic Memory (JSON + ChromaDB — learned facts about Roberto)
# ---------------------------------------------------------------------------


class SemanticMemory:
    """Persistent facts about the user, extracted from conversations."""

    def __init__(self):
        self._profile_path: str = ""
        self._facts: list[dict] = []
        self._chromadb_collection: Optional[chromadb.Collection] = None
        self._facts_text_cache: str = ""
        self._facts_cache_valid: bool = False
        self._alias_map: dict[str, str] = {}
        self._alias_map_valid: bool = False

    def initialize(self, client: chromadb.ClientAPI, db_path: str) -> None:
        self._profile_path = os.path.join(db_path, MEMORY_USER_PROFILE_FILE)
        self._chromadb_collection = client.get_or_create_collection(name=MEMORY_SEMANTIC_COLLECTION)
        self._load_profile()
        logger.info("Semantic memory: %d facts loaded", len(self._facts))

    def _load_profile(self) -> None:
        """Load facts from JSON file."""
        if os.path.exists(self._profile_path):
            try:
                with open(self._profile_path, "r") as f:
                    data = json.load(f)
                self._facts = data.get("facts", [])
            except Exception as e:
                logger.error("Failed to load user profile: %s", e)
                self._facts = []
        else:
            self._facts = []
        self._facts_cache_valid = False

    def _save_profile(self) -> None:
        """Atomically save facts to JSON file."""
        self._facts_cache_valid = False  # Invalidate formatted text cache
        self._alias_map_valid = False  # Invalidate entity alias cache
        data = {"facts": self._facts, "updated": datetime.now().isoformat()}
        dir_path = os.path.dirname(self._profile_path)
        os.makedirs(dir_path, exist_ok=True)

        try:
            # Write to temp file first, then rename (atomic on POSIX)
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".json")
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self._profile_path)
        except Exception as e:
            logger.error("Failed to save user profile: %s", e)

    def add_fact(
        self, entity: str, attribute: str, value: str, confidence: float = 0.8, category: str = "personal"
    ) -> None:
        """Add or update a fact. Deduplicates by entity+attribute."""
        # Check for existing fact with same entity+attribute
        for fact in self._facts:
            if fact["entity"].lower() == entity.lower() and fact["attribute"].lower() == attribute.lower():
                # Update if newer has higher or equal confidence
                if confidence >= fact.get("confidence", 0):
                    fact["value"] = value
                    fact["confidence"] = confidence
                    fact["category"] = category
                    fact["last_confirmed"] = datetime.now().isoformat()
                    self._save_profile()
                    self._update_chromadb_fact(fact)
                    logger.debug("Updated fact: %s.%s = %s", entity, attribute, value)
                return

        # New fact
        fact = {
            "entity": entity,
            "attribute": attribute,
            "value": value,
            "confidence": confidence,
            "category": category,
            "first_learned": datetime.now().isoformat(),
            "last_confirmed": datetime.now().isoformat(),
        }
        self._facts.append(fact)
        self._save_profile()
        self._add_chromadb_fact(fact)
        logger.info("New fact learned: %s.%s = %s", entity, attribute, value)

    def add_facts_from_json(self, facts_json: list[dict]) -> int:
        """Add multiple facts from a JSON array (as returned by the LLM)."""
        added = 0
        for f in facts_json:
            entity = f.get("entity", "")
            attribute = f.get("attribute", "")
            value = f.get("value", "")
            if not entity or not attribute or not value:
                continue

            # Resolve entity to canonical name (e.g. "Roberto's girlfriend" → "Marisa")
            entity = self._resolve_entity(entity)

            # Skip unresolvable vague entities (unknown_*, pronouns, etc.)
            if self._is_vague_entity(entity):
                logger.debug("[facts] Skipping vague entity: %s", entity)
                continue

            self.add_fact(
                entity=entity,
                attribute=attribute,
                value=value,
                confidence=f.get("confidence", 0.7),
                category=f.get("category", "personal"),
            )
            added += 1
        return added

    def get_all_facts_as_text(self) -> str:
        """Format all facts as a readable text block for system prompts.

        Cached — reformatted only when facts change.
        """
        if self._facts_cache_valid:
            return self._facts_text_cache

        if not self._facts:
            self._facts_text_cache = ""
            self._facts_cache_valid = True
            return ""

        lines = ["Known facts about Roberto and the workshop:"]
        by_category: dict[str, list[str]] = {}
        for fact in self._facts:
            cat = fact.get("category", "other")
            text = f"{fact['entity']}: {fact['attribute']} = {fact['value']}"
            by_category.setdefault(cat, []).append(text)

        for cat, items in sorted(by_category.items()):
            lines.append(f"\n[{cat.title()}]")
            for item in items:
                lines.append(f"- {item}")

        self._facts_text_cache = "\n".join(lines)
        self._facts_cache_valid = True
        return self._facts_text_cache

    def search_facts(self, query: str, n_results: int = 5) -> list[str]:
        """Semantic search through stored facts."""
        if not self._chromadb_collection or self._chromadb_collection.count() == 0:
            return []
        try:
            results = self._chromadb_collection.query(
                query_texts=[query],
                n_results=min(n_results, self._chromadb_collection.count()),
            )
            if results and results["documents"] and results["documents"][0]:
                return results["documents"][0]
            return []
        except Exception as e:
            logger.error("Semantic fact search failed: %s", e)
            return []

    def get_related_facts(self, text: str, n: int = 8) -> str:
        """Return existing facts relevant to the given text, formatted for the extraction prompt."""
        facts = self.search_facts(text, n_results=n)
        if not facts:
            return ""
        return "Known facts (use these entity names):\n" + "\n".join(f"- {f}" for f in facts)

    # -- Entity resolution -------------------------------------------------

    def _build_alias_map(self) -> dict[str, str]:
        """Build a lowercase mapping from aliases → canonical entity name.

        Uses relationship-type facts to create reverse lookups. E.g. if
        ``Marisa.relationship = "Roberto's girlfriend"`` exists, then
        ``"roberto's girlfriend" → "Marisa"``.
        """
        alias_map: dict[str, str] = {}
        for fact in self._facts:
            attr = fact.get("attribute", "").lower()
            if attr in _RELATIONSHIP_ATTRS:
                alias = fact["value"].lower().strip()
                canonical = fact["entity"]
                if alias and canonical:
                    alias_map[alias] = canonical
        return alias_map

    def _resolve_entity(self, entity: str) -> str:
        """Resolve an entity name to its canonical form using known facts.

        Examples:
            "Roberto's girlfriend" → "Marisa"  (via alias map)
            "unknown_female" → kept as-is (handled by _is_vague_entity later)
            "John" (unknown) → "John" (no change)
        """
        if not self._alias_map_valid:
            self._alias_map = self._build_alias_map()
            self._alias_map_valid = True

        key = entity.lower().strip()

        # Direct alias match
        if key in self._alias_map:
            resolved = self._alias_map[key]
            logger.debug("[facts] Resolved entity '%s' → '%s'", entity, resolved)
            return resolved

        # Possessive/descriptor match — only for multi-word keys to avoid
        # false positives (e.g. "roberto" matching "roberto's girlfriend")
        if " " in key or "'s" in key:
            for alias, canonical in self._alias_map.items():
                if key in alias or alias in key:
                    logger.debug("[facts] Fuzzy-resolved entity '%s' → '%s'", entity, canonical)
                    return canonical

        return entity

    @staticmethod
    def _is_vague_entity(entity: str) -> bool:
        """Return True if the entity is too vague to store (pronouns, unknown_*, etc.)."""
        return bool(_VAGUE_ENTITY_PATTERNS.match(entity.strip()))

    # -- Consolidation -----------------------------------------------------

    def _delete_chromadb_fact(self, fact: dict) -> None:
        """Remove a fact from ChromaDB by its computed ID."""
        if not self._chromadb_collection:
            return
        try:
            self._chromadb_collection.delete(ids=[self._fact_id(fact)])
        except Exception as e:
            logger.debug("Failed to delete fact from ChromaDB: %s", e)

    def consolidate_entities(self) -> int:
        """Merge facts stored under different names for the same entity.

        Uses relationship facts to build an alias graph, then merges all facts
        to the canonical (proper-noun) entity name.  Returns number of facts merged.
        """
        alias_map = self._build_alias_map()
        if not alias_map:
            return 0

        merged = 0
        facts_to_remove: list[int] = []

        for i, fact in enumerate(self._facts):
            entity_lower = fact["entity"].lower().strip()
            if entity_lower not in alias_map:
                continue
            canonical = alias_map[entity_lower]
            if fact["entity"] == canonical:
                continue  # already canonical

            attr_lower = fact["attribute"].lower()
            existing = next(
                (f for f in self._facts
                 if f["entity"] == canonical and f["attribute"].lower() == attr_lower),
                None,
            )
            if existing:
                # Keep higher confidence or more recent
                if fact.get("confidence", 0) > existing.get("confidence", 0):
                    existing["value"] = fact["value"]
                    existing["confidence"] = fact["confidence"]
                    existing["last_confirmed"] = fact.get("last_confirmed", existing.get("last_confirmed"))
                    self._update_chromadb_fact(existing)
                self._delete_chromadb_fact(fact)
                facts_to_remove.append(i)
            else:
                # Move fact to canonical entity
                self._delete_chromadb_fact(fact)
                fact["entity"] = canonical
                self._add_chromadb_fact(fact)

            merged += 1

        # Also remove vague entities that couldn't be resolved
        for i, fact in enumerate(self._facts):
            if i not in facts_to_remove and self._is_vague_entity(fact["entity"]):
                self._delete_chromadb_fact(fact)
                facts_to_remove.append(i)
                merged += 1

        # Remove merged/vague facts (reverse order to preserve indices)
        for i in sorted(facts_to_remove, reverse=True):
            self._facts.pop(i)

        if merged > 0:
            self._save_profile()
            logger.info("[facts] Consolidated %d fragmented facts", merged)

        return merged

    @property
    def fact_count(self) -> int:
        """Number of structured facts in the JSON profile."""
        return len(self._facts)

    def _fact_id(self, fact: dict) -> str:
        """Generate a stable ID for a fact."""
        key = f"{fact['entity']}_{fact['attribute']}".lower().replace(" ", "_")
        return f"fact_{key}"

    def _fact_text(self, fact: dict) -> str:
        return f"{fact['entity']}: {fact['attribute']} — {fact['value']}"

    def _add_chromadb_fact(self, fact: dict) -> None:
        if not self._chromadb_collection:
            return
        try:
            self._chromadb_collection.upsert(
                documents=[self._fact_text(fact)],
                ids=[self._fact_id(fact)],
                metadatas=[{"category": fact.get("category", "personal")}],
            )
        except Exception as e:
            logger.debug("Failed to add fact to ChromaDB: %s", e)

    def _update_chromadb_fact(self, fact: dict) -> None:
        self._add_chromadb_fact(fact)  # upsert handles updates


# ---------------------------------------------------------------------------
# Memory Facade (backward-compatible interface)
# ---------------------------------------------------------------------------


class Memory:
    """Main memory interface — routes to the three tiers.

    Public API (backward-compatible):
        start(), store(), recall(), get_recent(), entry_count, get_session_summary()

    New API:
        assemble_context(), extract_facts_from_text(), shutdown_session(),
        consolidate(), get_user_profile()
    """

    def __init__(self):
        self._client: Optional[chromadb.ClientAPI] = None
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()

    def start(self) -> bool:
        """Initialize all memory tiers."""
        import chromadb

        try:
            self._client = chromadb.PersistentClient(path=MEMORY_DB_PATH)
            self.episodic.initialize(self._client)
            self.semantic.initialize(self._client, MEMORY_DB_PATH)

            # Clean up entity fragmentation from previous sessions
            consolidated = self.semantic.consolidate_entities()
            if consolidated:
                logger.info("[memory] Consolidated %d fragmented entity facts on startup", consolidated)

            logger.info(
                "Memory system online — %d episodes, %d facts", self.episodic.episode_count, self.semantic.fact_count
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize memory system: %s", e)
            return False

    # ------ Backward-compatible interface ------

    def store(
        self, text: str, entry_type: str = "observation", activity: str = "", metadata: Optional[dict] = None
    ) -> None:
        """Store to both working memory and episodic memory."""
        if entry_type == "observation":
            self.working.add_observation(text, activity)
        elif entry_type == "conversation":
            # Parse Q/A format if present
            if text.startswith("Q:") and "\nA:" in text:
                parts = text.split("\nA:", 1)
                question = parts[0][2:].strip()
                answer = parts[1].strip()
                self.working.add_conversation(question, answer)
            else:
                self.working.add_observation(text, activity)

        self.episodic.store(text, entry_type, activity)

    def recall(self, query: str, n_results: int = 5) -> list[str]:
        """Semantic search across episodic memory."""
        return self.episodic.recall(query, n_results)

    def get_recent(self, n: int = 10) -> list[str]:
        """Get recent entries from episodic memory."""
        return self.episodic.get_recent(n)

    @property
    def entry_count(self) -> int:
        """Total episodic memory entries in ChromaDB."""
        return self.episodic.episode_count

    def get_session_summary(self) -> str:
        """Return working memory's session narrative."""
        return self.working.get_session_narrative()

    # ------ New three-tier interface ------

    def assemble_context(
        self,
        query: str = "",
        purpose: str = "conversation",
        temporal_narrative: str = "",
    ) -> str:
        """Build optimal context string from all memory tiers.

        Args:
            query: The user's query (for semantic search). Empty for proactive.
            purpose: "conversation", "proactive", or "lightweight" — determines token budget.
                     "lightweight" skips ChromaDB search and session narrative (fast path).
            temporal_narrative: Rolling timeline from the visual cortex (read-only).
                     Injected as the first context section so Claude has temporal awareness.

        Returns:
            A formatted context string ready to include in prompts.
        """
        if purpose == "lightweight":
            budget = MEMORY_CONTEXT_BUDGET_LIGHTWEIGHT
        elif purpose == "conversation":
            budget = MEMORY_CONTEXT_BUDGET_CONVERSATION
        else:
            budget = MEMORY_CONTEXT_BUDGET_PROACTIVE

        sections = []
        tokens_used = 0

        # 0. Temporal narrative (visual cortex timeline — highest priority, skip for lightweight)
        if temporal_narrative and purpose != "lightweight":
            narrative_tokens = self._estimate_tokens(temporal_narrative)
            if narrative_tokens > MEMORY_CONTEXT_BUDGET_NARRATIVE:
                # Truncate from the left — keep the most recent entries
                max_chars = MEMORY_CONTEXT_BUDGET_NARRATIVE * 4
                temporal_narrative = "..." + temporal_narrative[-(max_chars - 3):]
                narrative_tokens = MEMORY_CONTEXT_BUDGET_NARRATIVE
            section = f"## Recent Workshop Timeline\n{temporal_narrative}"
            sections.append(section)
            tokens_used += narrative_tokens

        # 1. Semantic facts (ALWAYS included, highest priority)
        facts_text = self.semantic.get_all_facts_as_text()
        if facts_text:
            sections.append(facts_text)
            tokens_used += self._estimate_tokens(facts_text)

        # Promote query-relevant facts to a highlighted section (helps pronoun resolution)
        if purpose == "conversation" and query:
            relevant = self.semantic.search_facts(query, n_results=5)
            if relevant:
                focused = "\n".join(f"- {r}" for r in relevant)
                sections.append(f"\n## Relevant to this query:\n{focused}")
                tokens_used += self._estimate_tokens(focused)

        # Lightweight: facts only, skip everything else (no DB calls)
        if purpose == "lightweight":
            return "\n".join(sections) if sections else ""

        # 2. Session narrative
        narrative = self.working.get_session_narrative()
        if narrative and tokens_used < budget:
            sections.append(f"\nCurrent session: {narrative}")
            tokens_used += self._estimate_tokens(narrative)

        # 3. Conversation thread (only for proactive — conversation uses multi-turn messages)
        if purpose == "proactive":
            conv_context = self.working.get_conversation_context()
            if conv_context and tokens_used < budget:
                sections.append(f"\nRecent conversation:\n{conv_context}")
                tokens_used += self._estimate_tokens(conv_context)

        # 4. Episodic recall (semantic search for relevant memories)
        is_temporal_query = bool(query and _TEMPORAL_QUERY_PATTERNS.search(query))
        if query and tokens_used < budget:
            recall_n = 8 if is_temporal_query else 5
            memories = self.episodic.recall(query, n_results=recall_n)
            if memories:
                mem_lines = []
                for mem in memories:
                    line = f"- {mem}"
                    line_tokens = self._estimate_tokens(line)
                    if tokens_used + line_tokens > budget:
                        break
                    mem_lines.append(line)
                    tokens_used += line_tokens
                if mem_lines:
                    sections.append("\nRelevant memories:\n" + "\n".join(mem_lines))

        # 5. Past session summaries (if budget allows)
        if tokens_used < budget:
            summary_n = 4 if is_temporal_query else 2
            summaries = self.episodic.get_past_summaries(n=summary_n)
            if summaries:
                for s in summaries:
                    line = f"- {s}"
                    line_tokens = self._estimate_tokens(line)
                    if tokens_used + line_tokens > budget:
                        break
                    sections.append(f"\nPrevious session: {s}")
                    tokens_used += line_tokens

        if not sections:
            return ""

        return "\n".join(sections)

    def extract_facts_from_text(
        self, text: str, llm_client, recent_context: str = "", existing_facts: str = ""
    ) -> int:
        """Use Claude Haiku to extract structured facts from text.

        Args:
            text: Current Q/A exchange to analyze.
            llm_client: LLM client for API calls.
            recent_context: Optional previous exchange(s) for relationship inference.
            existing_facts: Pre-formatted relevant facts so the LLM knows canonical entity names.

        Returns the number of facts extracted.
        """
        if hasattr(llm_client, "is_available") and not llm_client.is_available():
            logger.debug("[facts] Skipping extraction: API circuit breaker open")
            return 0
        try:
            logger.info("[facts] Attempting extraction from: %s", text[:100])
            parts: list[str] = []
            if existing_facts:
                parts.append(existing_facts)
            if recent_context:
                parts.append(f"Previous conversation:\n{recent_context}")
            parts.append(text)
            analysis = "\n\n".join(parts)
            response = llm_client.text_only_chat(
                prompt=f"Text to analyze:\n{analysis}",
                system_prompt=SYSTEM_PROMPT_FACT_EXTRACTION,
                max_tokens=500,
            )
            if not response:
                logger.info("[facts] LLM returned empty response")
                return 0

            logger.info("[facts] LLM returned: %s", response[:200])
            facts = self._parse_facts_json(response)
            if facts:
                count = self.semantic.add_facts_from_json(facts)
                logger.info("[facts] Parsed %d facts, added %d", len(facts), count)
                return count
            return 0
        except Exception as e:
            logger.error("[facts] Extraction failed: %s", e, exc_info=True)
            return 0

    def shutdown_session(self, llm_client, temporal_narrative: str = "") -> None:
        """End-of-session: generate summary, extract facts, save everything."""
        session_text = self.working.get_all_text_for_summary()

        # Include visual cortex observations in the session summary
        if temporal_narrative:
            session_text = f"Visual observations:\n{temporal_narrative}\n\n{session_text}"

        if not session_text.strip():
            logger.info("No session content to summarize")
            return

        # Generate session summary
        summary = self.episodic.generate_session_summary(llm_client, session_text)
        if summary:
            logger.info("Session summary: %s", summary[:100])

        # Extract any remaining facts from the full session
        self.extract_facts_from_text(session_text, llm_client)

        # Merge any fragmented entities created during this session
        self.semantic.consolidate_entities()

        logger.info(
            "Session shutdown complete — %d episodes, %d facts total",
            self.episodic.episode_count,
            self.semantic.fact_count,
        )

    def consolidate(self) -> int:
        """Clean up old, low-importance entries. Run at startup in background."""
        return self.episodic.consolidate()

    def get_user_profile(self) -> str:
        """Return all semantic facts as text."""
        return self.semantic.get_all_facts_as_text()

    def store_proactive(self, message: str, reasoning: str = "") -> None:
        """Store a proactive comment in both working and episodic memory."""
        self.working.add_proactive(message)
        self.episodic.store(f"[Proactive] {message}", entry_type="observation", activity=reasoning)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4

    @staticmethod
    def _parse_facts_json(text: str) -> list[dict]:
        """Parse JSON array from the LLM's fact extraction response."""
        if not text:
            return []
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n", 1)
            if len(lines) > 1:
                text = lines[1]
            text = text.rsplit("```", 1)[0].strip()
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            return []
        except json.JSONDecodeError:
            logger.warning("[facts] Failed to parse JSON: %.200s", text)
            return []
