import base64
import hashlib
import json
import logging
import threading
import time
from typing import Callable, Optional, Union

# Type for frame data: single frame or labeled multi-frame sequence
FrameInput = Union[bytes, list[tuple[str, bytes]]]

from config import (
    ANTHROPIC_API_KEY,
    FAST_MODEL,
    SMART_MODEL,
    SYSTEM_PROMPT_PERCEPTION,
    get_conversation_prompt,
    get_routing_prompt,
)
from utils.cost_tracker import CostTracker

logger = logging.getLogger("winston.claude")

MAX_RETRIES = 2
BASE_BACKOFF = 1.0  # max ~3s total (1s + 2s) — voice can't wait longer

# Circuit breaker: stop hammering the API on fatal errors (credit exhaustion, bad key)
FATAL_ERROR_PATTERNS = [
    "credit balance is too low",
    "invalid x-api-key",
    "api key not found",
    "account has been disabled",
]
_CB_INITIAL_RETRY_SECS = 300   # 5 minutes between probes
_CB_MAX_RETRY_SECS = 1800      # cap at 30 minutes

# Routing tools for process_user_input() — Haiku decides what to do
ROUTING_TOOLS = [
    {
        "name": "delegate_to_agent",
        "description": (
            "Delegate a task to the autonomous agent that can control the computer. "
            "Use ONLY when the user wants you to physically interact with the screen: "
            "open apps, click UI elements, type into fields, navigate documents, "
            "manipulate files on their machine. Do NOT use for web research or reading repos — "
            "use do_research instead."
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
        "name": "do_research",
        "description": (
            "Research a topic in the background. Use for: looking up information, "
            "investigating repos, finding documentation, web research, checking "
            "what's new with a project. Runs in the background — user can keep chatting. "
            "Do NOT use delegate_to_agent for research tasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What to research — be specific about what info to find",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "save_note",
        "description": (
            "Save a note or reminder when the user asks to remember, write down, or note something for later."
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
    {
        "name": "send_voice_response",
        "description": (
            "Send the response as a Telegram voice message instead of text. "
            "ONLY use this when the user explicitly asks for audio, a voice message, "
            "a voice note, or to 'speak' / 'say it out loud' / 'sprich mit mir'. "
            "Default is ALWAYS text — never use this unless explicitly requested."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The response text to convert to speech and send as voice",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "camera_request",
        "description": (
            "Send workshop camera content via Telegram: a live photo, a short video clip "
            "of recent seconds, or a sped-up timelapse of recent hours. "
            "Use when the user wants to see the workshop, asks for a photo/snapshot, "
            "a clip/video, or a timelapse. Works in English and German."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "request_type": {
                    "type": "string",
                    "enum": ["snapshot", "clip", "timelapse"],
                    "description": (
                        "snapshot = single photo, "
                        "clip = short video of last N seconds, "
                        "timelapse = sped-up video of last N hours"
                    ),
                },
                "parameter": {
                    "type": "number",
                    "description": (
                        "For clip: duration in seconds (default 15, max 60). "
                        "For timelapse: hours to cover (default 2, max 8). "
                        "Omit for snapshot."
                    ),
                },
            },
            "required": ["request_type"],
        },
    },
]

INTENT_CLASSIFICATION_PROMPT = """Is the speaker talking to you (Winston, an AI voice assistant)?
They are alone in a robotics workshop. Speech may be in English or German.
If the speech seems directed at an assistant, contains a question or request, or expects a response, say "yes".
Only say "no" for: obvious self-talk with no expectation of response, phone calls, conversations with other people present, or meaningless utterances like "hmm" or "okay".

Speech: "{text}"

Answer "yes" or "no" only."""

# Keywords that indicate speech is addressed to Winston (English + German)
ADDRESSED_KEYWORDS = [
    "winston",
    "look at",
    "help me",
    "tell me",
    "show me",
    "check this",
    "remember",
    "what do you think",
    "can you",
    "could you",
    "please",
    # German
    "schau",
    "hilf mir",
    "sag mir",
    "zeig mir",
    "kannst du",
    "könntest du",
    "bitte",
    "was denkst du",
    "merk dir",
    "überprüf",
]

QUESTION_WORDS = {
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "can",
    "could",
    "should",
    "is",
    "are",
    "do",
    "does",
    "did",
    "will",
    "would",
    "have",
    "has",
    # German
    "wer",
    "was",
    "wo",
    "wann",
    "warum",
    "wie",
    "kann",
    "soll",
    "ist",
    "sind",
    "hat",
    "hatte",
}

# Single-word conversational responses that count as addressed during active conversation
CONVERSATIONAL_RESPONSES = {
    "yeah",
    "yes",
    "no",
    "nah",
    "sure",
    "okay",
    "ok",
    "thanks",
    "right",
    "exactly",
    "correct",
    "yep",
    "nope",
    "absolutely",
    "definitely",
    "alright",
    # German
    "ja",
    "nein",
    "klar",
    "genau",
    "richtig",
    "stimmt",
    "danke",
    "natürlich",
}

# Keywords indicating the query needs camera/visual context (bilingual)
VISUAL_KEYWORDS = [
    # Explicit vision requests
    "look at", "what do you see", "what's on", "show me", "identify",
    "what is this", "what are these", "can you see", "check this",
    "what's happening", "what happened",
    # Spatial/object references
    "on my desk", "on the table", "on my screen", "on the screen",
    "in front of", "next to", "the printer", "over there",
    # Workshop observation
    "is the", "did i leave", "is it still", "does it look",
    # German equivalents
    "schau dir", "was siehst du", "was ist das", "zeig mir",
    "erkennst du", "auf dem tisch", "auf meinem schreibtisch",
    "was passiert", "ist das", "hab ich", "auf dem bildschirm",
]

# Short demonstrative pronouns that suggest pointing at something physical
_DEMONSTRATIVE_WORDS = {"this", "that", "these", "those", "dies", "das", "diese"}

# Minimum sentence length before flushing to TTS during streaming
SENTENCE_MIN_LENGTH = 10

# Minimum clause length for comma-based flushing
CLAUSE_MIN_LENGTH = 35


class SentenceChunker:
    """Accumulates streaming text and fires a callback on sentence boundaries.

    Boundaries: .!? followed by space/newline/uppercase (min SENTENCE_MIN_LENGTH chars),
                , at CLAUSE_MIN_LENGTH+ chars, \\n at SENTENCE_MIN_LENGTH+ chars.
    """

    def __init__(self, on_sentence: Callable[[str], None]):
        self._on_sentence = on_sentence
        self._buf = ""
        self._pending_boundary = False
        self._pending_is_clause = False

    def feed(self, delta: str) -> None:
        """Feed a text delta from the stream. May fire on_sentence zero or more times."""
        for ch in delta:
            self._buf += ch
            if self._pending_boundary:
                if ch in (" ", "\n"):
                    min_len = CLAUSE_MIN_LENGTH if self._pending_is_clause else SENTENCE_MIN_LENGTH
                    if len(self._buf) >= min_len:
                        self._on_sentence(self._buf.strip())
                        self._buf = ""
                    self._pending_boundary = False
                    self._pending_is_clause = False
                elif ch.isupper():
                    min_len = CLAUSE_MIN_LENGTH if self._pending_is_clause else SENTENCE_MIN_LENGTH
                    to_flush = self._buf[:-1].strip()
                    if to_flush and len(to_flush) >= min_len:
                        self._on_sentence(to_flush)
                        self._buf = ch
                    self._pending_boundary = False
                    self._pending_is_clause = False
                else:
                    self._pending_boundary = False
                    self._pending_is_clause = False
            elif ch in ".!?":
                self._pending_boundary = True
                self._pending_is_clause = False
            elif ch == "," and len(self._buf.strip()) >= CLAUSE_MIN_LENGTH:
                self._pending_boundary = True
                self._pending_is_clause = True
            elif ch == "\n" and len(self._buf.strip()) >= SENTENCE_MIN_LENGTH:
                self._on_sentence(self._buf.strip())
                self._buf = ""

    def flush(self) -> None:
        """Flush any remaining text as a final sentence."""
        if self._buf.strip():
            self._on_sentence(self._buf.strip())
            self._buf = ""


class ClaudeClient:
    """Central Claude API client. Handles routing (Haiku tool-use),
    conversation, frame analysis, intent classification, and streaming.
    Includes retry with exponential backoff and per-response cost tracking."""

    def __init__(self, cost_tracker: CostTracker):
        self._client = None
        self._cost_tracker = cost_tracker
        self._scene_cache: dict[str, tuple[dict, float]] = {}
        self._cache_ttl = 30.0

        # Circuit breaker state
        self._circuit_open = False
        self._circuit_opened_at: float = 0.0
        self._circuit_retry_interval: float = _CB_INITIAL_RETRY_SECS
        self._circuit_lock = threading.Lock()
        self._on_circuit_open: Optional[Callable[[str], None]] = None

    @property
    def client(self):
        """Raw Anthropic client for agent executor."""
        return self._client

    def set_circuit_breaker_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback invoked once when the circuit breaker opens."""
        self._on_circuit_open = callback

    def is_available(self) -> bool:
        """Check if the API is believed to be available.

        Returns True if the circuit is closed or the retry interval has elapsed
        (time to probe). Background subsystems should check this before calling.
        """
        with self._circuit_lock:
            if not self._circuit_open:
                return True
            return (time.time() - self._circuit_opened_at) >= self._circuit_retry_interval

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
                caller="startup_verify",
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
            caller="camera_analysis",
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
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        system = get_conversation_prompt()
        if language != "en":
            system = f"IMPORTANT: Respond entirely in German (Deutsch). The user is speaking German.\n\n{system}"

        response = self._call_with_retry(
            model=model,
            max_tokens=max_output_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            caller="conversation",
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
                model,
                message,
            )
            return None
        return text.strip()

    def text_only_chat(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 300,
        model: str = None,
    ) -> Optional[str]:
        """Text-only call for internal tasks (summarization, fact extraction).

        No image, cheap by default (Haiku). Pass model=SMART_MODEL for higher quality.
        """
        use_model = model or FAST_MODEL
        response = self._call_with_retry(
            model=use_model,
            max_tokens=max_tokens,
            system=system_prompt if system_prompt else None,
            messages=[{"role": "user", "content": prompt}],
            caller="text_only",
        )
        if response is None:
            return None

        self._track_usage(response, use_model)
        text = self._safe_response_text(response)
        if not text:
            logger.warning(
                "Claude text_only_chat returned empty response (prompt=%.100s)",
                prompt,
            )
            return None
        return text.strip()

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
            caller="intent_classification",
        )
        if response:
            self._track_usage(response, FAST_MODEL)
            resp_text = self._safe_response_text(response)
            if resp_text:
                result = "yes" in resp_text.strip().lower()
                logger.info("Intent classification: '%s' -> %s", text[:60], "addressed" if result else "not addressed")
                return result
        logger.warning("Intent classification failed, defaulting to True")
        return True

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
    def needs_visual_context(text: str) -> bool:
        """Fast heuristic: does this query benefit from the camera frame?

        Returns True if visual context is likely needed.
        No API call — pure keyword matching.
        """
        text_lower = text.lower().strip()

        for kw in VISUAL_KEYWORDS:
            if kw in text_lower:
                return True

        # Short utterances with demonstrative pronouns → likely pointing at something
        words = text_lower.split()
        if len(words) <= 6 and _DEMONSTRATIVE_WORDS & set(words):
            return True

        return False

    @staticmethod
    def _build_image_content(frame_input: "FrameInput") -> tuple[list[dict], int]:
        """Convert frame input to Claude API content blocks.

        Accepts either raw bytes (single frame) or a list of (label, bytes)
        tuples (multi-frame with temporal labels).

        Returns (content_blocks, image_count).
        """
        blocks: list[dict] = []

        if isinstance(frame_input, bytes):
            image_b64 = base64.standard_b64encode(frame_input).decode("utf-8")
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            })
            return blocks, 1

        if isinstance(frame_input, list):
            for label, frame_bytes in frame_input:
                blocks.append({"type": "text", "text": label})
                image_b64 = base64.standard_b64encode(frame_bytes).decode("utf-8")
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                })
            return blocks, len(frame_input)

        return [], 0

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
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        system = get_conversation_prompt()
        if language != "en":
            system = f"IMPORTANT: Respond entirely in German (Deutsch). The user is speaking German.\n\n{system}"

        kwargs = dict(
            model=model,
            max_tokens=max_output_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )

        try:
            full_text = ""
            chunker = SentenceChunker(on_sentence) if on_sentence else None

            with self._client.messages.stream(**kwargs) as s:
                for delta in s.text_stream:
                    # Check abort (barge-in)
                    if abort_event and abort_event.is_set():
                        logger.info("Streaming aborted (barge-in)")
                        break

                    full_text += delta
                    if chunker:
                        chunker.feed(delta)

                # Flush remainder
                if chunker:
                    chunker.flush()

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
        frame_bytes: Optional[FrameInput] = None,
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
        image_count = 0
        if frame_bytes:
            image_blocks, image_count = self._build_image_content(frame_bytes)
            content.extend(image_blocks)
        content.append({"type": "text", "text": prompt})

        # Ensure proper alternation: if last history message is "user", merge
        if messages and messages[-1]["role"] == "user":
            if isinstance(messages[-1]["content"], str):
                messages[-1]["content"] = [{"type": "text", "text": messages[-1]["content"]}]
            messages[-1]["content"].extend(content)
        else:
            messages.append({"role": "user", "content": content})

        system = get_routing_prompt()
        if language != "en":
            system = f"IMPORTANT: Respond entirely in German (Deutsch). The user is speaking German.\n\n{system}"

        self._log_prompt_audit(messages, has_image=frame_bytes is not None, system_len=len(system), image_count=image_count)

        t_llm_start = time.time()
        response = self._call_with_retry(
            model=FAST_MODEL,
            max_tokens=500,
            system=system,
            tools=ROUTING_TOOLS,
            messages=messages,
            caller="routing",
        )
        llm_ms = (time.time() - t_llm_start) * 1000
        logger.info("[latency] LLM first-token: %dms", llm_ms)
        logger.info("[latency] LLM total: %dms", llm_ms)
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

                elif tool_name == "do_research":
                    task = tool_input.get("task", text)
                    logger.info("Routing: do_research('%s')", task[:100])
                    return ("research", {"task": task})

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

                elif tool_name == "send_voice_response":
                    voice_text = tool_input.get("text", "")
                    logger.info("Routing: send_voice_response (%d chars)", len(voice_text))
                    return ("conversation_voice", {"text": voice_text})

                elif tool_name == "camera_request":
                    req_type = tool_input.get("request_type", "snapshot")
                    param = tool_input.get("parameter")
                    logger.info("Routing: camera_request(%s, param=%s)", req_type, param)
                    return ("camera_request", {"request_type": req_type, "parameter": param})

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

    def process_user_input_streaming(
        self,
        text: str,
        frame_bytes: Optional[FrameInput] = None,
        context: Optional[str] = None,
        language: str = "en",
        conversation_history: Optional[list[dict]] = None,
        on_sentence: Optional[Callable[[str], None]] = None,
        abort_event=None,
    ) -> tuple[str, dict]:
        """Streaming version of process_user_input().

        Fires on_sentence() for each sentence of a conversation response while
        the LLM is still generating. For tool calls (agent/note/shutdown),
        on_sentence may not fire — the tool result is returned as usual.

        Falls back to process_user_input() on any streaming error.

        Returns (action_type, data) — same contract as process_user_input().
        """

        # Build messages (identical to process_user_input)
        messages = []
        if conversation_history:
            for msg in conversation_history:
                messages.append(msg)

        prompt = text
        if context:
            prompt = f"{text}\n\n{context}"

        content = []
        image_count = 0
        if frame_bytes:
            image_blocks, image_count = self._build_image_content(frame_bytes)
            content.extend(image_blocks)
        content.append({"type": "text", "text": prompt})

        # Ensure proper alternation: if last history message is "user", merge
        if messages and messages[-1]["role"] == "user":
            if isinstance(messages[-1]["content"], str):
                messages[-1]["content"] = [{"type": "text", "text": messages[-1]["content"]}]
            messages[-1]["content"].extend(content)
        else:
            messages.append({"role": "user", "content": content})

        system = get_routing_prompt()
        if language != "en":
            system = f"IMPORTANT: Respond entirely in German (Deutsch). The user is speaking German.\n\n{system}"

        kwargs = dict(
            model=FAST_MODEL,
            max_tokens=500,
            system=system,
            tools=ROUTING_TOOLS,
            messages=messages,
        )

        full_text = ""
        try:
            t_start = time.time()
            first_token_logged = False
            chunker = SentenceChunker(on_sentence) if on_sentence else None

            self._log_prompt_audit(messages, has_image=frame_bytes is not None, system_len=len(system), image_count=image_count)
            logger.info("[api] routing_streaming: calling %s", FAST_MODEL)
            with self._client.messages.stream(**kwargs) as stream:
                for delta in stream.text_stream:
                    # Log first-token latency
                    if not first_token_logged:
                        first_token_ms = (time.time() - t_start) * 1000
                        logger.info("[latency] LLM first-token: %dms", first_token_ms)
                        first_token_logged = True

                    # Check abort (barge-in)
                    if abort_event and abort_event.is_set():
                        logger.info("Streaming routing aborted (barge-in)")
                        break

                    full_text += delta
                    if chunker:
                        chunker.feed(delta)

                # Flush remaining buffered text
                if chunker:
                    chunker.flush()

                # Get final message for tool-call detection and cost tracking
                final_msg = stream.get_final_message()
                llm_ms = (time.time() - t_start) * 1000
                logger.info("[latency] LLM total: %dms", llm_ms)

                self._track_usage(final_msg, FAST_MODEL)

                # Check for tool use in the final message
                for block in final_msg.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        if tool_name == "delegate_to_agent":
                            task = tool_input.get("task", text)
                            logger.info("Routing: delegate_to_agent('%s')", task[:100])
                            return ("agent", {"task": task, "_streamed_text": full_text})

                        elif tool_name == "do_research":
                            task = tool_input.get("task", text)
                            logger.info("Routing: do_research('%s')", task[:100])
                            return ("research", {"task": task, "_streamed_text": full_text})

                        elif tool_name == "save_note":
                            note_content = tool_input.get("content", text)
                            logger.info("Routing: save_note('%s')", note_content[:100])
                            return ("note", {"content": note_content, "_streamed_text": full_text})

                        elif tool_name == "shutdown_system":
                            logger.info("Routing: shutdown_system")
                            return ("shutdown", {"_streamed_text": full_text})

                        elif tool_name == "get_current_time":
                            from datetime import datetime

                            now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
                            logger.info("Routing: get_current_time → %s", now)
                            return ("conversation", {"text": f"It's {now}."})

                        elif tool_name == "send_voice_response":
                            voice_text = tool_input.get("text", "")
                            logger.info("Routing: send_voice_response (%d chars)", len(voice_text))
                            return ("conversation_voice", {"text": voice_text})

            # No tool use — pure conversation response (already streamed)
            if not full_text.strip():
                logger.warning("process_user_input_streaming: empty response for '%s'", text[:60])
                return ("conversation", {"text": "Sorry, I didn't catch that."})

            logger.info("Routing: conversation (%d chars, streamed)", len(full_text))
            return ("conversation", {"text": full_text.strip()})

        except Exception as e:
            logger.warning("Streaming routing failed (%s), falling back to non-streaming", e)
            if full_text.strip():
                # Partial response was already streamed — use what we have
                logger.info("Returning partial streamed response (%d chars)", len(full_text))
                return ("conversation", {"text": full_text.strip()})
            # Nothing was streamed — safe to fall back
            return self.process_user_input(
                text=text,
                frame_bytes=frame_bytes,
                context=context,
                language=language,
                conversation_history=conversation_history,
            )

    def _call_with_retry(self, **kwargs):
        """Call Claude API with exponential backoff and circuit breaker."""
        caller = kwargs.pop("caller", "unknown")

        # Circuit breaker check
        with self._circuit_lock:
            if self._circuit_open:
                elapsed = time.time() - self._circuit_opened_at
                if elapsed < self._circuit_retry_interval:
                    logger.debug(
                        "Circuit breaker open, skipping API call [%s] (%ds remaining)",
                        caller,
                        int(self._circuit_retry_interval - elapsed),
                    )
                    return None
                else:
                    logger.info(
                        "Circuit breaker: attempting probe call [%s] after %ds",
                        caller,
                        int(elapsed),
                    )
                    self._circuit_open = False

        for attempt in range(MAX_RETRIES):
            try:
                # Remove None system prompts (Anthropic doesn't accept None)
                if kwargs.get("system") is None:
                    kwargs.pop("system", None)
                logger.info("[api] %s: calling %s", caller, kwargs.get("model", "?"))
                response = self._client.messages.create(**kwargs)

                # Success — reset circuit breaker if it was tripped
                self._reset_circuit_breaker()
                return response
            except Exception as e:
                error_str = str(e).lower()
                if any(pattern in error_str for pattern in FATAL_ERROR_PATTERNS):
                    logger.error(
                        "Fatal API error [%s]: %s — opening circuit breaker",
                        caller, e,
                    )
                    self._open_circuit_breaker(str(e))
                    return None

                wait = BASE_BACKOFF * (2**attempt)
                logger.warning(
                    "API call failed [%s] (attempt %d/%d): %s. Retrying in %.1fs",
                    caller, attempt + 1, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)

        logger.error("API call failed after %d attempts [%s]", MAX_RETRIES, caller)
        return None

    def _open_circuit_breaker(self, error_msg: str) -> None:
        """Open the circuit breaker and fire the notification callback once."""
        fire_callback = False
        with self._circuit_lock:
            if not self._circuit_open:
                fire_callback = True
                self._circuit_retry_interval = _CB_INITIAL_RETRY_SECS
            else:
                # Already open — double the probe interval (exponential backoff)
                self._circuit_retry_interval = min(
                    self._circuit_retry_interval * 2, _CB_MAX_RETRY_SECS,
                )
            self._circuit_open = True
            self._circuit_opened_at = time.time()
            logger.warning(
                "CIRCUIT BREAKER OPEN: %s (next probe in %ds)",
                error_msg[:200],
                int(self._circuit_retry_interval),
            )

        if fire_callback and self._on_circuit_open:
            try:
                self._on_circuit_open(error_msg)
            except Exception as cb_err:
                logger.error("Circuit breaker callback failed: %s", cb_err)

    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker state after a successful API call."""
        with self._circuit_lock:
            if self._circuit_open:
                logger.info("Circuit breaker reset: API recovered")
            self._circuit_open = False
            self._circuit_opened_at = 0.0
            self._circuit_retry_interval = _CB_INITIAL_RETRY_SECS

    @staticmethod
    def _log_prompt_audit(messages: list, has_image: bool, system_len: int, image_count: int = 1) -> None:
        """Log approximate token breakdown of the outgoing prompt."""
        history_tokens = 0
        current_text_tokens = 0

        for msg in messages[:-1]:  # All but last = history
            content = msg.get("content", "")
            if isinstance(content, str):
                history_tokens += len(content) // 4

        # Last message = current user message
        if messages:
            content = messages[-1].get("content", "")
            if isinstance(content, str):
                current_text_tokens += len(content) // 4
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        current_text_tokens += len(block.get("text", "")) // 4

        image_tokens = 1280 * image_count if has_image else 0
        system_tokens = system_len // 4
        total = system_tokens + history_tokens + current_text_tokens + image_tokens

        logger.info(
            "[prompt_audit] ~%d tokens (system=%d, history=%d, text=%d, image=%d)",
            total, system_tokens, history_tokens, current_text_tokens, image_tokens,
        )

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
