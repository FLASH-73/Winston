"""
Winston Curiosity Engine — autonomous background thinking loop.

Runs as a daemon thread. Three phases per cycle:
  Phase 1 (REFLECT): Haiku reviews memory/context, picks interesting topics
  Phase 2 (EXPLORE): DuckDuckGo web search, Haiku extracts findings
  Phase 3 (CRAFT):   Sonnet crafts a message if the finding is genuinely interesting

Guards: quiet hours, daily cap, topic deduplication, budget check, absence check-in.
"""

import json
import logging
import os
import random
import tempfile
import threading
import time
from datetime import date, datetime, timedelta
from typing import Optional

from config import (
    CURIOSITY_ABSENCE_HOURS,
    CURIOSITY_COMPANION_PROMPT,
    CURIOSITY_DAILY_CAP,
    CURIOSITY_MAX_INTERVAL,
    CURIOSITY_MIN_INTERVAL,
    CURIOSITY_QUIET_END,
    CURIOSITY_QUIET_START,
    CURIOSITY_SEARCH_ENABLED,
    CURIOSITY_STATE_FILE,
    SMART_MODEL,
)

logger = logging.getLogger("winston.curiosity")

# Recent topics within this window are considered duplicates
_TOPIC_DEDUP_HOURS = 48


class CuriosityEngine:
    """Autonomous background loop: reflect, explore, share."""

    def __init__(self, llm, memory, temporal_memory, telegram_notifier, cost_tracker, presence_tracker=None):
        self._llm = llm
        self._memory = memory
        self._temporal_memory = temporal_memory
        self._notifier = telegram_notifier
        self._cost_tracker = cost_tracker
        self._presence = presence_tracker

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # State (persisted to disk)
        self._sends_today: list[dict] = []       # [{topic, timestamp, message}]
        self._recent_topics: list[dict] = []     # [{topic, timestamp}]
        self._last_activity_time: float = time.time()
        self._today: str = date.today().isoformat()

        self._load_state()

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self):
        """Start the curiosity background thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="curiosity-engine"
        )
        self._thread.start()
        logger.info(
            "Curiosity engine started (interval: %d-%ds, cap: %d/day)",
            CURIOSITY_MIN_INTERVAL, CURIOSITY_MAX_INTERVAL, CURIOSITY_DAILY_CAP,
        )

    def stop(self):
        """Stop the curiosity engine gracefully."""
        self._running = False
        self._save_state()
        logger.info("Curiosity engine stopped")

    def record_activity(self):
        """Called when Roberto interacts (voice/telegram). Resets absence timer."""
        self._last_activity_time = time.time()
        if self._presence:
            self._presence.signal()

    # ── Main loop ─────────────────────────────────────────────────

    def _loop(self):
        # Initial delay: let the system fully warm up
        initial_delay = random.randint(120, 300)
        logger.info("Curiosity engine: initial delay %ds", initial_delay)
        self._sleep_interruptible(initial_delay)

        while self._running:
            try:
                self._cycle()
            except Exception as e:
                logger.error("Curiosity cycle error: %s", e, exc_info=True)

            # Random interval for next cycle
            interval = random.randint(CURIOSITY_MIN_INTERVAL, CURIOSITY_MAX_INTERVAL)
            logger.debug("Curiosity engine: next cycle in %ds", interval)
            self._sleep_interruptible(interval)

    def _sleep_interruptible(self, seconds: int):
        """Sleep in 10s chunks so stop() takes effect quickly."""
        for _ in range(seconds // 10):
            if not self._running:
                return
            time.sleep(10)
        remaining = seconds % 10
        if remaining and self._running:
            time.sleep(remaining)

    def _cycle(self):
        """One curiosity cycle: check guards, then reflect/explore/craft."""
        self._rotate_day()

        # Guard: quiet hours
        hour = datetime.now().hour
        if CURIOSITY_QUIET_START <= hour < CURIOSITY_QUIET_END:
            logger.debug("Curiosity: quiet hours, skipping")
            return

        # Guard: daily cap
        if len(self._sends_today) >= CURIOSITY_DAILY_CAP:
            logger.debug("Curiosity: daily cap reached (%d/%d)",
                         len(self._sends_today), CURIOSITY_DAILY_CAP)
            return

        # Guard: budget
        if not self._cost_tracker.check_budget():
            logger.debug("Curiosity: daily budget exceeded, skipping")
            return

        # Guard: tiered budget — skip non-essential when critical or exhausted
        budget_state = self._cost_tracker.get_budget_state()
        if budget_state in ("critical", "exhausted"):
            logger.debug("Curiosity: budget state '%s', skipping", budget_state)
            return

        # Guard: circuit breaker — skip if API is down
        if not self._llm.is_available():
            logger.debug("Curiosity: API unavailable (circuit breaker), skipping")
            return

        # Check: absence check-in (priority over normal curiosity)
        if self._presence:
            absence_hours = self._presence.minutes_idle / 60.0
        else:
            absence_hours = (time.time() - self._last_activity_time) / 3600.0
        if absence_hours >= CURIOSITY_ABSENCE_HOURS:
            if not self._is_topic_recent("__absence_checkin__"):
                self._absence_checkin(absence_hours)
                return

        # Phase 1: REFLECT
        topics = self._phase_reflect()
        if not topics:
            logger.debug("Curiosity: reflection produced no topics")
            return

        # Phase 2 + 3: EXPLORE then CRAFT (for each topic until one succeeds)
        for topic_data in topics:
            topic = topic_data.get("topic", "")
            search_query = topic_data.get("search_query", "")
            why_interesting = topic_data.get("why_interesting", "")

            if not topic or not search_query:
                continue

            # Dedup check
            if self._is_topic_recent(topic):
                logger.debug("Curiosity: topic already covered recently: %s", topic[:60])
                continue

            finding = self._phase_explore(search_query, why_interesting)
            if not finding or "nothing interesting" in finding.lower():
                logger.debug("Curiosity: exploration found nothing for: %s", topic[:60])
                continue

            # Phase 3: CRAFT + DECIDE
            message = self._phase_craft(topic, finding)
            if not message or "NOT_WORTH_SENDING" in message:
                logger.info("Curiosity: decided not to send for: %s", topic[:60])
                # Still record the topic to avoid re-exploring it
                self._recent_topics.append({
                    "topic": topic,
                    "timestamp": datetime.now().isoformat(),
                })
                continue

            self._send(message, topic)
            return  # One message per cycle max

        logger.debug("Curiosity: no topics survived the pipeline")

    # ── Phase 1: REFLECT ──────────────────────────────────────────

    def _phase_reflect(self) -> list[dict]:
        """Use Haiku to reflect on Roberto's world and suggest topics."""
        from personality import get_personality

        p = get_personality()
        if not p.companion.reflect_prompt:
            return []

        # Assemble context
        context_parts = []

        facts = self._memory.semantic.get_all_facts_as_text()
        if facts:
            context_parts.append(f"What I know about Roberto:\n{facts}")

        if self._temporal_memory:
            narrative = self._temporal_memory.get_narrative(hours=4.0)
            if narrative:
                context_parts.append(f"Recent workshop activity (last 4 hours):\n{narrative}")

        session = self._memory.working.get_session_narrative()
        if session:
            context_parts.append(f"Current session:\n{session}")

        # What we already covered recently (avoid repetition)
        recent_topic_names = [e["topic"] for e in self._recent_topics[-20:]]
        if recent_topic_names:
            context_parts.append(f"Topics already explored recently: {', '.join(recent_topic_names)}")

        if self._sends_today:
            sent_summaries = [s["topic"] for s in self._sends_today]
            context_parts.append(f"Already messaged about today: {', '.join(sent_summaries)}")

        context_parts.append(f"Current time: {datetime.now().strftime('%A %H:%M')}")

        context = "\n\n".join(context_parts) if context_parts else "No recent context available."
        prompt = f"{p.companion.reflect_prompt}\n\nContext:\n{context}"

        response = self._llm.text_only_chat(prompt=prompt, max_tokens=400)
        if not response:
            return []

        try:
            data = json.loads(_strip_code_fences(response))
            topics = data.get("topics", [])
            if isinstance(topics, list):
                return topics[:3]  # Cap at 3
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.debug("Curiosity reflect: non-JSON response, skipping")

        return []

    # ── Phase 2: EXPLORE ──────────────────────────────────────────

    def _phase_explore(self, search_query: str, why_interesting: str) -> Optional[str]:
        """Search the web and extract the interesting bit."""
        if not CURIOSITY_SEARCH_ENABLED:
            return None

        from brain.agent_tools import _web_search
        from personality import get_personality

        results = _web_search({"query": search_query, "max_results": 5})
        if not results or results.startswith("Error") or results.startswith("No results"):
            return None

        p = get_personality()
        if not p.companion.explore_prompt:
            return None

        prompt = p.companion.explore_prompt.format(
            query=search_query,
            results=results[:3000],
            why_interesting=why_interesting,
        )

        return self._llm.text_only_chat(prompt=prompt, max_tokens=200)

    # ── Phase 3: CRAFT ────────────────────────────────────────────

    def _phase_craft(self, topic: str, finding: str) -> Optional[str]:
        """Use Sonnet to craft a natural Telegram message."""
        from personality import get_personality

        p = get_personality()
        if not p.companion.craft_prompt:
            return None

        context = self._memory.semantic.get_all_facts_as_text() or "No stored facts."

        prompt = p.companion.craft_prompt.format(
            context=context[:1000],
            topic=topic,
            finding=finding,
        )

        message = self._llm.text_only_chat(
            prompt=prompt,
            system_prompt=CURIOSITY_COMPANION_PROMPT,
            max_tokens=300,
            model=SMART_MODEL,
        )
        return message.strip() if message else None

    # ── Absence check-in ──────────────────────────────────────────

    def _absence_checkin(self, hours: float):
        """Send a check-in message after extended absence."""
        from personality import get_personality

        p = get_personality()
        if not p.companion.absence_prompt:
            return

        context = self._memory.semantic.get_all_facts_as_text() or ""
        if self._temporal_memory:
            latest = self._temporal_memory.get_latest_state()
            if latest:
                context += f"\nLast observation: {latest}"

        prompt = p.companion.absence_prompt.format(
            hours=f"{hours:.0f}",
            context=context[:1000],
        )

        message = self._llm.text_only_chat(
            prompt=prompt,
            system_prompt=CURIOSITY_COMPANION_PROMPT,
            max_tokens=200,
            model=SMART_MODEL,
        )

        if message and "NOT_WORTH_SENDING" not in message:
            self._send(message.strip(), "__absence_checkin__")

    # ── Send + state management ───────────────────────────────────

    def _send(self, message: str, topic: str):
        """Send message via Telegram and record in memory + state."""
        self._notifier.send_curiosity_message(message)

        # Record in memory so Winston remembers what he shared
        self._memory.store_proactive(message, reasoning=f"Curiosity: {topic}")

        # Record in state
        now = datetime.now().isoformat()
        self._sends_today.append({"topic": topic, "timestamp": now, "message": message[:500]})
        self._recent_topics.append({"topic": topic, "timestamp": now})

        self._save_state()
        logger.info("Curiosity message sent (%d/%d today): %s",
                     len(self._sends_today), CURIOSITY_DAILY_CAP, message[:100])

    def _is_topic_recent(self, topic: str) -> bool:
        """Check if a similar topic was already covered within the dedup window."""
        cutoff = datetime.now() - timedelta(hours=_TOPIC_DEDUP_HOURS)
        topic_lower = topic.lower()
        for entry in self._recent_topics:
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
                if ts >= cutoff and entry["topic"].lower() == topic_lower:
                    return True
            except (ValueError, KeyError):
                continue
        return False

    def _rotate_day(self):
        """Reset daily counters if it's a new day."""
        today = date.today().isoformat()
        if today != self._today:
            self._today = today
            self._sends_today = []
            # Prune old topics beyond dedup window
            cutoff = datetime.now() - timedelta(hours=_TOPIC_DEDUP_HOURS)
            self._recent_topics = [
                e for e in self._recent_topics
                if _parse_timestamp(e.get("timestamp")) and _parse_timestamp(e["timestamp"]) >= cutoff
            ]
            self._save_state()

    # ── Persistence ───────────────────────────────────────────────

    def _save_state(self):
        """Persist curiosity state to JSON (atomic write)."""
        data = {
            "today": self._today,
            "sends_today": self._sends_today,
            "recent_topics": self._recent_topics[-50:],
            "last_activity_time": self._last_activity_time,
        }
        try:
            dir_path = os.path.dirname(CURIOSITY_STATE_FILE)
            os.makedirs(dir_path, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".json")
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, CURIOSITY_STATE_FILE)
        except Exception as e:
            logger.error("Failed to save curiosity state: %s", e)

    def _load_state(self):
        """Load curiosity state from JSON."""
        if not os.path.exists(CURIOSITY_STATE_FILE):
            return
        try:
            with open(CURIOSITY_STATE_FILE) as f:
                data = json.load(f)
            saved_today = data.get("today", "")
            if saved_today == date.today().isoformat():
                self._sends_today = data.get("sends_today", [])
            self._recent_topics = data.get("recent_topics", [])
            self._last_activity_time = data.get("last_activity_time", time.time())
            self._today = date.today().isoformat()
            logger.info("Curiosity state loaded: %d sends today, %d recent topics",
                        len(self._sends_today), len(self._recent_topics))
        except Exception as e:
            logger.error("Failed to load curiosity state: %s", e)


# ── Helpers ───────────────────────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n", 1)
        if len(lines) > 1:
            text = lines[1]
        text = text.rsplit("```", 1)[0].strip()
    return text


def _parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse an ISO timestamp, returning None on failure."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
