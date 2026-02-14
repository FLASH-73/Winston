import base64
import hashlib
import json
import logging
import time
from typing import Callable, Optional

from config import (
    ANTHROPIC_API_KEY,
    FAST_MODEL,
    SMART_MODEL,
    SYSTEM_PROMPT_PERCEPTION,
    SYSTEM_PROMPT_CONVERSATION,
)
from utils.cost_tracker import CostTracker

logger = logging.getLogger("winston.claude")

MAX_RETRIES = 2
BASE_BACKOFF = 1.0  # max ~3s total (1s + 2s) — voice can't wait longer

# Routing tools for process_user_input() — Haiku decides what to do
ROUTING_TOOLS = [
    {
        "name": "delegate_to_agent",
        "description": (
            "Delegate a task to the autonomous agent that can control the computer. "
            "Use when the user wants you to DO something: open files/URLs/apps, "
            "search the web and display results, navigate documents, investigate code, "
            "control applications, or any action requiring computer interaction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Clear description of what to do, including any specifics the user mentioned",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "save_note",
        "description": (
            "Save a note or reminder when the user asks to remember, write down, "
            "or note something for later."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to save (extract the actual note, not the command)",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "shutdown_system",
        "description": (
            "Shut down Winston when the user says goodbye, go offline, go to sleep, "
            "or otherwise indicates they want to end the session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_current_time",
        "description": (
            "Get the current date and time. Use when the user asks what time it is, "
            "what the date is, or anything related to current date/time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]

INTENT_CLASSIFICATION_PROMPT = """Is the speaker talking to you (Winston, an AI voice assistant)?
They are alone in a robotics workshop. Speech may be in English or German.
If the speech seems directed at an assistant, contains a question or request, or expects a response, say "yes".
Only say "no" for: obvious self-talk with no expectation of response, phone calls, conversations with other people present, or meaningless utterances like "hmm" or "okay".

Speech: "{text}"

Answer "yes" or "no" only."""

COMBINED_INTENT_ROUTING_PROMPT = """Is the speaker talking to you (Winston, an AI voice assistant)? If so, does the request need deep analysis?
They are alone in a robotics workshop. Speech may be in English or German.

Speech: "{text}"

Answer ONLY one of: "yes_fast", "yes_smart", "no"."""

INTENT_CLASSIFICATION_FULL_PROMPT = """Classify the user's intent. They are talking to Winston, an AI workshop assistant. Speech may be in any language.

Speech: "{text}"

Respond with ONLY valid JSON, no markdown:
{{"intent": "conversation", "note_content": null}}

Possible intents:
- "shutdown": User wants to stop/quit/shutdown Winston or the system (e.g. "go offline", "shut down", "mach Feierabend", "geh schlafen", "shut up and stop", "goodbye winston")
- "agent": User wants Winston to autonomously investigate their CODE or GITHUB REPOSITORY in the background (e.g. "check my code for bugs", "look at the repo for issues", "schau dir den Code an", "find the bug in my project"). This is ONLY for code/repository investigation tasks, NOT for general questions, web lookups, or information requests.
- "note": User wants to save/remember/write down something. Extract ONLY the content to save into "note_content" (e.g. "save this: buy M3 bolts" → note_content="buy M3 bolts", "merk dir ich brauche Schrauben" → note_content="ich brauche Schrauben")
- "conversation": Everything else — questions, chat, requests, commands, web lookups, time checks, general knowledge, opinions. This is the default. When in doubt, use this."""

# Keywords that trigger the Smart (Sonnet) model
PRO_TRIGGER_PATTERNS = [
    "look closely", "inspect", "analyze", "what's wrong",
    "check this", "debug", "diagnose", "compare",
    "explain in detail", "zoom in", "carefully",
    "help me figure out", "what do you think about",
    "look at this", "examine", "what went wrong",
]

# Keywords that indicate speech is addressed to Winston (English + German)
ADDRESSED_KEYWORDS = [
    "winston", "look at", "help me", "tell me", "show me",
    "check this", "remember", "what do you think",
    "can you", "could you", "please",
    # German
    "schau", "hilf mir", "sag mir", "zeig mir",
    "kannst du", "könntest du", "bitte", "was denkst du",
    "merk dir", "überprüf",
]

QUESTION_WORDS = {
    "who", "what", "where", "when", "why", "how",
    "can", "could", "should", "is", "are", "do", "does",
    "did", "will", "would", "have", "has",
    # German
    "wer", "was", "wo", "wann", "warum", "wie",
    "kann", "soll", "ist", "sind", "hat", "hatte",
}

# Single-word conversational responses that count as addressed during active conversation
CONVERSATIONAL_RESPONSES = {
    "yeah", "yes", "no", "nah", "sure", "okay", "ok", "thanks",
    "right", "exactly", "correct", "yep", "nope", "absolutely",
    "definitely", "alright",
    # German
    "ja", "nein", "klar", "genau", "richtig", "stimmt",
    "danke", "natürlich",
}

# Minimum sentence length before flushing to TTS during streaming
SENTENCE_MIN_LENGTH = 10


class ClaudeClient:
    def __init__(self, cost_tracker: CostTracker):
        self._client = None
        self._cost_tracker = cost_tracker
        self._scene_cache: dict[str, tuple[dict, float]] = {}
        self._cache_ttl = 30.0

    @property
    def client(self):
        """Raw Anthropic client for agent executor."""
        return self._client

    def start(self) -> bool:
        """Initialize the Anthropic client and verify connectivity."""
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_key_here":
            logger.error("ANTHROPIC_API_KEY not set. Add it to your .env file.")
            return False
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=0)
            response = self._call_with_retry(
                model=FAST_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "Respond with just the word: ready"}],
            )
            if response is None:
                logger.warning("Claude API verification failed (transient), continuing anyway")
            else:
                text = response.content[0].text if response.content else "ok"
                logger.info("Claude API connected (%s)", text.strip())
            return True  # Client is initialized — API may recover by first real call
        except Exception as e:
            logger.error("Failed to connect to Claude API: %s", e)
            return False

    def analyze_frame(
        self,
        frame_bytes: bytes,
        prompt: str = None,
        system_prompt: str = SYSTEM_PROMPT_PERCEPTION,
        use_pro: bool = False,
        max_output_tokens: int = 300,
    ) -> Optional[dict]:
        """Send an image + text prompt to Claude for analysis.

        Returns parsed JSON dict, or None on failure.
        """
        frame_hash = hashlib.md5(frame_bytes).hexdigest()
        cached = self._scene_cache.get(frame_hash)
        if cached and (time.time() - cached[1]) < self._cache_ttl:
            logger.debug("Returning cached scene analysis")
            return cached[0]

        model = SMART_MODEL if use_pro else FAST_MODEL
        prompt_text = prompt or "Analyze this frame."

        image_b64 = base64.standard_b64encode(frame_bytes).decode("utf-8")
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            },
            {"type": "text", "text": prompt_text},
        ]

        response = self._call_with_retry(
            model=model,
            max_tokens=max_output_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        if response is None:
            return None

        self._track_usage(response, model)

        text = self._safe_response_text(response)
        if not text:
            logger.warning("Claude analyze_frame returned empty response (model=%s)", model)
            return None
        result = self._parse_json(text)
        if result is not None:
            self._scene_cache[frame_hash] = (result, time.time())
        return result

    def chat(
        self,
        message: str,
        frame_bytes: Optional[bytes] = None,
        use_pro: bool = False,
        context: Optional[str] = None,
        max_output_tokens: int = 500,
        language: str = "en",
    ) -> Optional[str]:
        """Conversational response. Optionally includes current frame and memory context.

        Returns response text string, or None on failure.
        """
        model = SMART_MODEL if use_pro else FAST_MODEL

        prompt = message
        if context:
            prompt = f"{message}\n\n{context}"

        content = []
        if frame_bytes:
            image_b64 = base64.standard_b64encode(frame_bytes).decode("utf-8")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            })
        content.append({"type": "text", "text": prompt})

        system = SYSTEM_PROMPT_CONVERSATION
        if language != "en":
            system = f"IMPORTANT: Respond entirely in German (Deutsch). The user is speaking German.\n\n{system}"

        response = self._call_with_retry(
            model=model,
            max_tokens=max_output_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        if response is None:
            return None
        logger.info("Claude response received, extracting text...")

        self._track_usage(response, model)
        text = self._safe_response_text(response)
        logger.info("Text extracted: %s", "None" if not text else f"{len(text)} chars")

        if not text:
            logger.warning(
                "Claude chat returned empty response (model=%s, message=%.100s)",
                model, message,
            )
            return None
        return text.strip()

    def text_only_chat(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 300,
    ) -> Optional[str]:
        """Text-only call for internal tasks (summarization, fact extraction).

        No image, cheap. Used by the memory system.
        """
        response = self._call_with_retry(
            model=FAST_MODEL,
            max_tokens=max_tokens,
            system=system_prompt if system_prompt else None,
            messages=[{"role": "user", "content": prompt}],
        )
        if response is None:
            return None

        self._track_usage(response, FAST_MODEL)
        text = self._safe_response_text(response)
        if not text:
            logger.warning(
                "Claude text_only_chat returned empty response (prompt=%.100s)",
                prompt,
            )
            return None
        return text.strip()

    def route_request(self, text: str) -> str:
        """Use fast model to intelligently decide whether smart model is needed."""
        prompt = ROUTING_PROMPT.format(request=text)
        response = self._call_with_retry(
            model=FAST_MODEL,
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        if response:
            resp_text = self._safe_response_text(response)
            if resp_text and "smart" in resp_text.strip().lower():
                logger.info("Routing decision: Smart (Sonnet)")
                return SMART_MODEL
        logger.info("Routing decision: Fast (Haiku)")
        return FAST_MODEL

    def classify_intent(self, text: str) -> bool:
        """Classify whether transcribed speech is addressed to Winston.

        Uses Haiku with max_tokens=3 for minimal latency (~200-400ms).
        Returns True if the user appears to be talking to Winston.
        Defaults to True on failure (better to respond than miss a command).
        """
        prompt = INTENT_CLASSIFICATION_PROMPT.format(text=text)
        response = self._call_with_retry(
            model=FAST_MODEL,
            max_tokens=3,
            messages=[{"role": "user", "content": prompt}],
        )
        if response:
            self._track_usage(response, FAST_MODEL)
            resp_text = self._safe_response_text(response)
            if resp_text:
                result = "yes" in resp_text.strip().lower()
                logger.info("Intent classification: '%s' -> %s",
                            text[:60], "addressed" if result else "not addressed")
                return result
        logger.warning("Intent classification failed, defaulting to True")
        return True

    def classify_user_intent(self, text: str) -> dict:
        """Classify user intent via Haiku. Returns {"intent": str, "note_content": str|None}.

        Intents: "conversation", "shutdown", "agent", "note".
        Used to replace keyword-based trigger matching with intelligent classification.
        """
        default = {"intent": "conversation", "note_content": None}
        prompt = INTENT_CLASSIFICATION_FULL_PROMPT.format(text=text)
        response = self._call_with_retry(
            model=FAST_MODEL,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        if response:
            self._track_usage(response, FAST_MODEL)
            resp_text = self._safe_response_text(response)
            if resp_text:
                result = self._parse_json(resp_text)
                if result and "intent" in result:
                    logger.info("Intent classification: '%s' -> %s", text[:60], result.get("intent"))
                    return result
        logger.warning("Intent classification failed, defaulting to 'conversation'")
        return default

    # ── Fast local classifiers (no API call) ──────────────────────────

    @staticmethod
    def classify_intent_local(text: str, conversation_active: bool = False) -> Optional[bool]:
        """Classify intent using local heuristics. Returns True/False/None.

        True  = definitely addressed to Winston (skip API)
        False = definitely NOT addressed (skip API)
        None  = ambiguous, needs API call

        When conversation_active=True (Winston spoke within last 30s),
        be very permissive — almost everything is a response to Winston.
        """
        text_lower = text.lower().strip()
        words = text_lower.split()

        if not words:
            return False

        # Contains "winston" → definitely addressed
        if "winston" in text_lower:
            return True

        # Matches command patterns → addressed
        for kw in ADDRESSED_KEYWORDS:
            if kw in text_lower:
                return True

        # Starts with a question word → addressed
        if words and words[0] in QUESTION_WORDS:
            return True

        # In active conversation: be very permissive
        if conversation_active:
            # Single-word grunts not addressed even in conversation
            if len(words) <= 1 and text_lower not in CONVERSATIONAL_RESPONSES:
                return False
            return True  # Everything 2+ words in conversation → addressed

        # Question mark WITHOUT other signals → ambiguous, let API decide
        # (prevents "Hey [random name], what's the time?" from being misrouted)
        if text.rstrip().endswith("?"):
            return None

        # Not in conversation: short utterances → not addressed
        if len(words) <= 2:
            return False

        # Ambiguous — needs API
        return None

    @staticmethod
    def route_request_local(text: str) -> str:
        """Route to Smart/Fast model using local keyword matching. Instant, no API."""
        text_lower = text.lower()
        for trigger in PRO_TRIGGER_PATTERNS:
            if trigger in text_lower:
                logger.info("Routing decision: Smart/Sonnet (local, keyword='%s')", trigger)
                return SMART_MODEL
        logger.info("Routing decision: Fast/Haiku (local)")
        return FAST_MODEL

    def classify_and_route(self, text: str) -> tuple[bool, str]:
        """Combined intent + routing in a single API call.

        Returns (is_addressed, model_name). Defaults to (True, FAST_MODEL) on failure.
        """
        prompt = COMBINED_INTENT_ROUTING_PROMPT.format(text=text)
        response = self._call_with_retry(
            model=FAST_MODEL,
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        if response:
            self._track_usage(response, FAST_MODEL)
            resp_text = self._safe_response_text(response)
            if resp_text:
                resp_lower = resp_text.strip().lower()
                if "no" in resp_lower and "yes" not in resp_lower:
                    logger.info("Combined classify+route: not addressed (API)")
                    return (False, FAST_MODEL)
                elif "smart" in resp_lower:
                    logger.info("Combined classify+route: addressed, Smart (API)")
                    return (True, SMART_MODEL)
                else:
                    logger.info("Combined classify+route: addressed, Fast (API)")
                    return (True, FAST_MODEL)
        logger.warning("Combined classify+route failed, defaulting to (True, Fast)")
        return (True, FAST_MODEL)

    # ── Streaming chat ────────────────────────────────────────────────

    def chat_streaming(
        self,
        message: str,
        frame_bytes: Optional[bytes] = None,
        use_pro: bool = False,
        context: Optional[str] = None,
        max_output_tokens: int = 500,
        on_sentence: Optional[Callable[[str], None]] = None,
        abort_event=None,
        language: str = "en",
    ) -> Optional[str]:
        """Streaming conversational response. Fires on_sentence() for each sentence.

        Returns the full response text, or None on failure.
        Falls back to non-streaming chat() if streaming fails.
        If abort_event is set, stops streaming early (e.g. on barge-in).
        """
        model = SMART_MODEL if use_pro else FAST_MODEL

        prompt = message
        if context:
            prompt = f"{message}\n\n{context}"

        content = []
        if frame_bytes:
            image_b64 = base64.standard_b64encode(frame_bytes).decode("utf-8")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            })
        content.append({"type": "text", "text": prompt})

        system = SYSTEM_PROMPT_CONVERSATION
        if language != "en":
            system = f"IMPORTANT: Respond entirely in German (Deutsch). The user is speaking German.\n\n{system}"

        kwargs = dict(
            model=model,
            max_tokens=max_output_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )

        try:
            stream = self._client.messages.stream(**kwargs)
            full_text = ""
            sentence_buf = ""
            pending_boundary = False
            pending_is_clause = False

            CLAUSE_MIN_LENGTH = 35  # Higher threshold for comma-based flushing

            with stream as s:
                for delta in s.text_stream:
                    # Check abort (barge-in)
                    if abort_event and abort_event.is_set():
                        logger.info("Streaming aborted (barge-in)")
                        break

                    full_text += delta

                    # Sentence boundary detection
                    if on_sentence is None:
                        continue

                    for ch in delta:
                        sentence_buf += ch
                        if pending_boundary:
                            if ch in (" ", "\n"):
                                min_len = CLAUSE_MIN_LENGTH if pending_is_clause else SENTENCE_MIN_LENGTH
                                if len(sentence_buf) >= min_len:
                                    on_sentence(sentence_buf.strip())
                                    sentence_buf = ""
                                pending_boundary = False
                                pending_is_clause = False
                            elif ch.isupper():
                                # Split: flush everything before this char
                                min_len = CLAUSE_MIN_LENGTH if pending_is_clause else SENTENCE_MIN_LENGTH
                                to_flush = sentence_buf[:-1].strip()
                                if to_flush and len(to_flush) >= min_len:
                                    on_sentence(to_flush)
                                    sentence_buf = ch
                                pending_boundary = False
                                pending_is_clause = False
                            else:
                                pending_boundary = False
                                pending_is_clause = False
                        elif ch in ".!?":
                            pending_boundary = True
                            pending_is_clause = False
                        elif ch == "," and len(sentence_buf.strip()) >= CLAUSE_MIN_LENGTH:
                            pending_boundary = True
                            pending_is_clause = True
                        elif ch == "\n" and len(sentence_buf.strip()) >= SENTENCE_MIN_LENGTH:
                            on_sentence(sentence_buf.strip())
                            sentence_buf = ""

                # Flush remainder
                if on_sentence and sentence_buf.strip():
                    on_sentence(sentence_buf.strip())

                # Cost tracking from final message
                final_msg = s.get_final_message()
                self._track_usage(final_msg, model)

            logger.info("Streaming response complete: %d chars", len(full_text))
            return full_text.strip() if full_text else None

        except Exception as e:
            logger.warning("Streaming failed (%s), falling back to non-streaming", e)
            return self.chat(
                message=message,
                frame_bytes=frame_bytes,
                use_pro=use_pro,
                context=context,
                max_output_tokens=max_output_tokens,
                language=language,
            )

    # ── End-to-end routing via tool use ─────────────────────────────

    def process_user_input(
        self,
        text: str,
        frame_bytes: Optional[bytes] = None,
        context: Optional[str] = None,
        language: str = "en",
        conversation_history: Optional[list[dict]] = None,
    ) -> tuple[str, dict]:
        """Process user input end-to-end. Haiku decides: answer, delegate, note, or shutdown.

        Returns (action_type, data) where action_type is one of:
        - "conversation": data = {"text": response_text}
        - "agent": data = {"task": task_description}
        - "note": data = {"content": note_text}
        - "shutdown": data = {}

        conversation_history: list of {"role": "user"/"assistant", "content": str}
            for multi-turn context. Claude sees the full conversation flow.
        """
        from config import SYSTEM_PROMPT_ROUTING

        # Build messages array with conversation history
        messages = []
        if conversation_history:
            for msg in conversation_history:
                messages.append(msg)

        # Build current user message (with image + context)
        prompt = text
        if context:
            prompt = f"{text}\n\n{context}"

        content = []
        if frame_bytes:
            image_b64 = base64.standard_b64encode(frame_bytes).decode("utf-8")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            })
        content.append({"type": "text", "text": prompt})

        # Ensure proper alternation: if last history message is "user", merge
        if messages and messages[-1]["role"] == "user":
            if isinstance(messages[-1]["content"], str):
                messages[-1]["content"] = [{"type": "text", "text": messages[-1]["content"]}]
            messages[-1]["content"].extend(content)
        else:
            messages.append({"role": "user", "content": content})

        system = SYSTEM_PROMPT_ROUTING
        if language != "en":
            system = f"IMPORTANT: Respond entirely in German (Deutsch). The user is speaking German.\n\n{system}"

        response = self._call_with_retry(
            model=FAST_MODEL,
            max_tokens=500,
            system=system,
            tools=ROUTING_TOOLS,
            messages=messages,
        )
        if response is None:
            return ("conversation", {"text": "Sorry, I couldn't process that. Try again?"})

        self._track_usage(response, FAST_MODEL)

        # Check for tool use in response
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                if tool_name == "delegate_to_agent":
                    task = tool_input.get("task", text)
                    logger.info("Routing: delegate_to_agent('%s')", task[:100])
                    return ("agent", {"task": task})

                elif tool_name == "save_note":
                    note_content = tool_input.get("content", text)
                    logger.info("Routing: save_note('%s')", note_content[:100])
                    return ("note", {"content": note_content})

                elif tool_name == "shutdown_system":
                    logger.info("Routing: shutdown_system")
                    return ("shutdown", {})

                elif tool_name == "get_current_time":
                    from datetime import datetime
                    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
                    logger.info("Routing: get_current_time → %s", now)
                    return ("conversation", {"text": f"It's {now}."})

        # No tool use — extract text response
        texts = []
        for block in response.content:
            if hasattr(block, "text") and block.text:
                texts.append(block.text)
        response_text = " ".join(texts).strip()

        if not response_text:
            logger.warning("process_user_input: empty response for '%s'", text[:60])
            return ("conversation", {"text": "Sorry, I didn't catch that."})

        logger.info("Routing: conversation (%d chars)", len(response_text))
        return ("conversation", {"text": response_text})

    def _call_with_retry(self, **kwargs):
        """Call Claude API with exponential backoff on failure."""
        for attempt in range(MAX_RETRIES):
            try:
                # Remove None system prompts (Anthropic doesn't accept None)
                if kwargs.get("system") is None:
                    kwargs.pop("system", None)
                response = self._client.messages.create(**kwargs)
                return response
            except Exception as e:
                wait = BASE_BACKOFF * (2 ** attempt)
                logger.warning("API call failed (attempt %d/%d): %s. Retrying in %.1fs",
                               attempt + 1, MAX_RETRIES, e, wait)
                time.sleep(wait)

        logger.error("API call failed after %d attempts", MAX_RETRIES)
        return None

    def _track_usage(self, response, model: str) -> None:
        """Extract token counts from response and record in cost tracker."""
        try:
            usage = response.usage
            model_type = "fast" if model == FAST_MODEL else "smart"
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            self._cost_tracker.record(model_type, input_tokens, output_tokens)
            logger.debug("Tokens: %d in / %d out (%s)", input_tokens, output_tokens, model_type)
        except Exception as e:
            logger.debug("Could not track usage: %s", e)

    @staticmethod
    def _safe_response_text(response) -> Optional[str]:
        """Safely extract text from a Claude response."""
        try:
            if response.content and len(response.content) > 0:
                block = response.content[0]
                if hasattr(block, "text") and block.text:
                    return block.text
            logger.warning(
                "Claude response has no text content (stop_reason=%s)",
                getattr(response, "stop_reason", "N/A"),
            )
            return None
        except Exception as e:
            logger.error("Failed to extract text from Claude response: %s", e)
            return None

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """Parse JSON from Claude response, handling markdown code fences."""
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n", 1)
            if len(lines) > 1:
                text = lines[1]
            text = text.rsplit("```", 1)[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from response: %.100s...", text)
            return None
