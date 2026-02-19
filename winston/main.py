"""
WINSTON Workshop Assistant — Main Orchestrator

Architecture:
                                    +----------------+
                         +--------->| Claude Sonnet  | (on-demand, expensive)
                         |          +----------------+
+----------+    +--------+-------+  +----------------+
| Camera   |--->| Orchestrator   |->| Claude Haiku   | (always-on, cheap)
| Mic/STT  |--->| (main loop)    |  +----------------+
| Memory   |<-->|                |  +----------------+
+----------+    +--------+-------+  | TTS Output     |
                         |          +----------------+
                         +--------->| ChromaDB       |
                                    +----------------+

Startup: Dashboard first (instant UI), then Camera, Audio (fast),
         Memory+TTS+Claude in parallel.

All heavy imports (cv2, anthropic, fastapi, sounddevice, chromadb) are
deferred to start() so python main.py shows output instantly.
"""

import logging
import os
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

# Configure logging EARLY so import progress is visible
logging.basicConfig(
    level=logging.INFO,
    format="[Winston] %(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("winston")


# Catch unhandled exceptions in daemon threads so they don't die silently
def _daemon_thread_exception_hook(args):
    logger.error(
        "Unhandled exception in thread '%s': %s",
        args.thread.name if args.thread else "unknown",
        args.exc_value,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


threading.excepthook = _daemon_thread_exception_hook

logger.info("Starting up...")

# Only import lightweight modules at top level (config is just stdlib + dotenv)
from config import (
    CAMERA_ANALYSIS_INTERVAL,
    CAPTURE_INTERVAL,
    STREAMING_ENABLED,
    SYSTEM_PROMPT_PERCEPTION,
)
from dashboard.state import WinstonState  # stdlib only, instant


class Winston:
    def __init__(self):
        self._running = False
        self._capture_interval = CAPTURE_INTERVAL
        self.state = WinstonState()
        # Conversational context: track when Winston last responded
        self._last_winston_response_time = 0.0
        # Continuation tracking: allow split utterances to keep context
        self._last_addressed_time: float = 0.0
        self._last_user_utterance: str = ""
        # Abort event for canceling Claude streaming on barge-in
        self._streaming_abort = threading.Event()
        # Guard against concurrent agent tasks (rate limit protection)
        self._agent_lock = threading.Lock()
        # Persistent stores (initialized in start())
        self._agent_store = None
        self._notes_store = None
        self._telegram_bot = None
        self._telegram_notifier = None
        # Context cache for continuation window (avoid re-assembling within 8s)
        self._cached_context: str = ""
        self._cached_context_time: float = 0.0
        self._cached_frame_bytes: bytes | None = None
        # Camera analysis throttling + concern dedup
        self._last_camera_analysis_time: float = 0.0
        self._last_concern: str = ""
        self._last_concern_time: float = 0.0
        # All other modules created lazily in start()

    def start(self) -> bool:
        """Initialize all modules. Each import is timed for diagnostics."""
        self._log_banner("Initializing WINSTON Workshop Assistant")
        self.state.set_status("initializing")

        # ── 0. Dashboard FIRST (imports fastapi) ──────────────────────
        self._log_step("Starting dashboard")
        t = time.time()
        from dashboard.server import (
            create_app,
            set_audio,
            set_camera,
            set_cost_tracker,
            set_memory,
            set_notes_store,
            start_server,
        )

        logger.info("  dashboard module loaded (%.1fs)", time.time() - t)
        dashboard_app = create_app(self.state, None)
        start_server(dashboard_app, port=8420)
        logger.info("Dashboard live: http://localhost:8420")

        # ── 1. Camera (imports cv2) ───────────────────────────────────
        self._log_step("Initializing camera")
        t = time.time()
        from config import CAMERA_SOURCE
        from perception.camera import Camera

        logger.info("  camera module loaded (%.1fs)", time.time() - t)
        self.camera = Camera()
        self.state.set_module_status("camera", "loading")
        t = time.time()
        if self.camera.start(source=CAMERA_SOURCE):
            self.state.set_module_status("camera", "online", (time.time() - t) * 1000)
        else:
            logger.warning("Camera initialization failed. Continuing without camera.")
            self.state.set_module_status("camera", "error")
        set_camera(self.camera)

        # ── 2. Audio (imports sounddevice, faster-whisper deferred) ───
        self._log_step("Loading speech recognition")
        t = time.time()
        from perception.audio import AudioListener

        logger.info("  audio module loaded (%.1fs)", time.time() - t)
        self.audio = AudioListener()
        self.state.set_module_status("audio", "loading")
        t = time.time()
        if not self.audio.start():
            logger.error("Audio initialization failed.")
            self.state.set_module_status("audio", "error")
            return False
        self.state.set_module_status("audio", "online", (time.time() - t) * 1000)
        self.audio.set_callbacks(
            on_transcription=self._on_transcription,
            on_bargein=self._on_bargein,
            tts_is_speaking=lambda: self.tts.is_speaking,
            get_last_tts_text=lambda: self.tts.last_spoken_text,
            on_ambient_transcription=self._on_ambient_transcription,
        )
        set_audio(self.audio)

        # ── 3. Memory + TTS + Claude (imports anthropic, chromadb) ──
        self._log_step("Initializing Memory, TTS, and Claude in parallel")
        t = time.time()
        from brain.claude_client import ClaudeClient
        from brain.memory import Memory
        from perception.tts import TTSEngine
        from utils.cost_tracker import CostTracker

        logger.info("  brain modules loaded (%.1fs)", time.time() - t)

        self.cost_tracker = CostTracker()
        self.tts = TTSEngine()
        self.memory = Memory()
        self.llm = ClaudeClient(self.cost_tracker)

        def _init_memory():
            self.state.set_module_status("memory", "loading")
            t0 = time.time()
            self.memory.start()
            self.state.set_module_status("memory", "online", (time.time() - t0) * 1000)
            self.state.set_memory_stats(self.memory.entry_count, self.memory.semantic.fact_count)

        def _init_tts():
            self.state.set_module_status("tts", "loading")
            t0 = time.time()
            self.tts.start()
            self.state.set_module_status("tts", "online", (time.time() - t0) * 1000)

        def _init_llm():
            self.state.set_module_status("llm", "loading")
            t0 = time.time()
            ok = self.llm.start()
            if ok:
                self.state.set_module_status("llm", "online", (time.time() - t0) * 1000)
            else:
                self.state.set_module_status("llm", "error")
            return ok

        llm_ok = True
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_init_memory): "memory",
                executor.submit(_init_tts): "tts",
                executor.submit(_init_llm): "llm",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if name == "llm" and result is False:
                        llm_ok = False
                except Exception as e:
                    logger.error("Module '%s' init failed: %s", name, e)
                    return False

        if not llm_ok:
            logger.error("Claude API initialization failed.")
            return False

        # Wire TTS state changes → AudioListener energy-delta detector
        self.tts.set_state_callbacks(
            on_speaking_start=self.audio.on_tts_start,
            on_speaking_stop=self.audio.on_tts_stop,
        )

        set_memory(self.memory)
        set_cost_tracker(self.cost_tracker)

        # ── 3b. Persistent stores (agent tasks + notes) ────────────────
        from config import AGENT_TASKS_FILE, NOTES_FILE
        from utils.persistent_store import PersistentStore

        self._agent_store = PersistentStore(AGENT_TASKS_FILE, {"tasks": []})
        self._notes_store = PersistentStore(NOTES_FILE, {"notes": []})
        set_notes_store(self._notes_store)

        # Load saved tasks + notes into dashboard state
        saved_tasks = self._agent_store.get("tasks", [])
        if saved_tasks:
            self.state.set_agent_tasks(saved_tasks)
        saved_notes = self._notes_store.get("notes", [])
        if saved_notes:
            self.state.set_notes(saved_notes)

        # ── 4. Proactive engine (imports cv2 again — cached, instant) ─
        from brain.proactive import ProactiveEngine

        self.proactive = ProactiveEngine(self.camera, self.llm, self.memory, self.tts)

        # Background tasks
        threading.Thread(target=self._consolidate_memory, daemon=True, name="memory-consolidate").start()

        # Quick end-to-end test: verify Claude chat returns text
        try:
            test_response = self.llm.chat(message="Say 'hello' in one word.", max_output_tokens=10)
            if test_response:
                logger.info("Claude chat self-test passed: '%s'", test_response[:50])
            else:
                logger.error(
                    "Claude chat self-test FAILED — chat() returned None. "
                    "Run 'python test_claude.py' to diagnose API issues."
                )
        except Exception as e:
            logger.error("Claude chat self-test EXCEPTION: %s", e, exc_info=True)

        # Check for agent results from previous sessions
        self._check_pending_agent_results()

        # Announce ready
        self.state.set_status("idle")
        facts_count = self.memory.semantic.fact_count
        self._log_banner(
            f"System online. Memory: {self.memory.entry_count} episodes, {facts_count} facts. Mode: always-listening."
        )
        from personality import get_catchphrase

        self.tts.speak_async(get_catchphrase("greeting", "en"))

        # ── 5. Telegram bot (optional) ──────────────────────────────
        from config import TELEGRAM_ENABLED

        if TELEGRAM_ENABLED:
            from config import TELEGRAM_ALLOWED_USERS, TELEGRAM_BOT_TOKEN

            if TELEGRAM_BOT_TOKEN:
                self._log_step("Starting Telegram bot")
                from telegram_bot import TelegramBot

                # Pass camera, STT provider, and TTS engine for new features
                stt_provider = getattr(self.audio, "_stt_provider", None)

                self._telegram_bot = TelegramBot(
                    token=TELEGRAM_BOT_TOKEN,
                    allowed_users=TELEGRAM_ALLOWED_USERS,
                    llm=self.llm,
                    memory=self.memory,
                    cost_tracker=self.cost_tracker,
                    notes_store=self._notes_store,
                    agent_store=self._agent_store,
                    state=self.state,
                    agent_lock=self._agent_lock,
                    camera=self.camera,
                    stt_provider=stt_provider,
                    tts_engine=self.tts,
                )
                threading.Thread(
                    target=self._telegram_bot.start,
                    daemon=True,
                    name="telegram-bot",
                ).start()
                logger.info("Telegram bot thread launched")

                # Set up proactive Telegram notifications
                from config import TELEGRAM_NOTIFY_PROACTIVE

                if TELEGRAM_NOTIFY_PROACTIVE:
                    from telegram_bot import TelegramNotifier

                    self._telegram_notifier = TelegramNotifier(self._telegram_bot)
                    logger.info("Telegram proactive notifier enabled")

                # Register bot with dashboard for webhook support
                from dashboard.server import set_telegram_bot

                set_telegram_bot(self._telegram_bot)
            else:
                logger.warning("TELEGRAM_ENABLED=true but TELEGRAM_BOT_TOKEN not set")

        return True

    def run(self):
        """Main event loop."""
        self._running = True
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        while self._running:
            loop_start = time.time()

            try:
                # Update audio level for dashboard
                if hasattr(self.audio, "_current_audio_level"):
                    self.state.set_audio_level(self.audio._current_audio_level)

                # Update always-listen state for dashboard
                self.state.set_always_listen_state(self.audio.always_listen_state)

                # Reset to idle when TTS finishes speaking
                if self.state.status == "speaking" and not self.tts.is_speaking:
                    self.state.set_status("idle")

                # Safety: detect stuck TTS state (is_speaking=True but no audio for too long)
                if self.tts.is_speaking:
                    stuck_time = time.monotonic() - self.tts.speaking_start_time
                    if stuck_time > 45.0:
                        logger.warning(
                            "TTS stuck for %.1fs (audio pipeline: %s) — force resetting",
                            stuck_time,
                            self.audio.pipeline_state,
                        )
                        self.tts.interrupt()
                        self.state.set_status("idle")

                # 1. Capture frame
                frame_bytes = self.camera.get_frame_bytes()

                # 2. Scene change detection -> Flash analysis (throttled)
                conversation_active = self.tts.is_speaking or self.audio.pipeline_state == "recording"
                analysis_due = (time.monotonic() - self._last_camera_analysis_time) >= CAMERA_ANALYSIS_INTERVAL
                if frame_bytes and analysis_due and not conversation_active and self.camera.scene_changed():
                    self._last_camera_analysis_time = time.monotonic()
                    result = self.llm.analyze_frame(
                        frame_bytes,
                        system_prompt=SYSTEM_PROMPT_PERCEPTION,
                    )
                    if result:
                        desc = result.get("scene_description", "")
                        self.memory.store(
                            desc,
                            entry_type="observation",
                            activity=result.get("activity", ""),
                        )
                        if desc:
                            self.state.add_observation(desc)
                        self.state.set_cost(self.cost_tracker.get_daily_cost())
                        concerns = result.get("concerns")
                        if concerns:
                            concern_text = str(concerns)
                            is_repeat = (
                                concern_text == self._last_concern
                                and (time.monotonic() - self._last_concern_time) < 120.0
                            )
                            if not is_repeat:
                                logger.warning("Concern detected: %s", concerns)
                                self._last_concern = concern_text
                                self._last_concern_time = time.monotonic()

                # 3. Proactive check
                if frame_bytes and self.proactive.should_check():
                    context = self.memory.assemble_context(purpose="proactive")
                    proactive_msg = self.proactive.check(frame_bytes, recent_context=context)

                    # Send high-importance proactive messages to Telegram
                    if proactive_msg and self._telegram_notifier:
                        from config import TELEGRAM_NOTIFY_THRESHOLD

                        if self.proactive.last_usefulness_score >= TELEGRAM_NOTIFY_THRESHOLD:
                            self._telegram_notifier.notify(proactive_msg, frame_bytes)

                # 4. Budget check
                if not self.cost_tracker.check_budget():
                    self._enter_low_budget_mode()

                # 5. Update latency stats for dashboard
                from utils.latency_tracker import LatencyTracker

                lat_tracker = LatencyTracker.get()
                lat_stats = lat_tracker.get_stats()
                if lat_stats.get("interaction_count", 0) > 0:
                    self.state.set_latency_stats(lat_stats)
                lat_tracker.cleanup_stale()

            except Exception as e:
                logger.error("Main loop error: %s", e, exc_info=True)

            # Sleep for remainder of interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, self._capture_interval - elapsed)
            time.sleep(sleep_time)

    def _build_conversation_history(self, max_turns: int = 10) -> list[dict]:
        """Build Claude-compatible message history from recent conversation.

        Returns list of {"role": "user"/"assistant", "content": str} dicts.
        Merges consecutive same-role messages. Skips the current (pending) message.
        """
        with self.state._lock:
            recent = list(self.state.conversation[-max_turns:])

        if not recent:
            return []

        messages = []
        for turn in recent:
            role = "user" if turn["role"] == "user" else "assistant"
            text = turn["text"]

            # Merge consecutive same-role messages (Claude requires alternation)
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += f" {text}"
            else:
                messages.append({"role": role, "content": text})

        # Ensure it starts with "user" (Claude requirement)
        while messages and messages[0]["role"] != "user":
            messages.pop(0)

        return messages

    @staticmethod
    def _is_simple_query(text: str) -> bool:
        """Heuristic: is this a simple query that doesn't need full memory context?

        Simple queries use lightweight context (semantic facts only, no ChromaDB search).
        """
        words = text.split()
        if len(words) > 12:
            return False

        text_lower = text.lower()

        # References to past context need full memory recall
        memory_triggers = [
            "remember", "last time", "earlier", "yesterday", "before",
            "we talked", "you said", "you told me", "that project",
            "the motor", "the printer", "the arm", "the robot",
            # German
            "erinnerst", "letztes mal", "gestern", "vorher", "wir haben",
            "du hast gesagt", "das projekt",
        ]
        for trigger in memory_triggers:
            if trigger in text_lower:
                return False

        return True

    def _on_bargein(self):
        """Called when barge-in (user speech during TTS) is detected."""
        logger.info("Barge-in detected — interrupting speech")
        self._streaming_abort.set()
        self.tts.interrupt()
        self._streaming_abort.clear()

    def _on_transcription(self, text: str, lang: str = "en", model: str = None, interaction_id: str = None):
        """Called when user speech is transcribed. Runs in a daemon thread.

        Uses Claude Haiku with tool-use routing: Haiku decides whether to answer
        directly, delegate to the agent, save a note, or shut down.
        No keyword matching — end-to-end intelligence.
        """
        from utils.latency_tracker import LatencyTracker

        tracker = LatencyTracker.get()

        try:
            logger.info("[transcription] User said [%s]: '%s'", lang, text)

            if not text.strip():
                logger.info("[transcription] Empty text, ignoring")
                self.state.set_status("idle")
                tracker.discard(interaction_id)
                return

            # Quick local check for music mode toggle (no API call needed)
            text_lower = text.strip().lower()
            if any(phrase in text_lower for phrase in [
                "music mode on", "musik modus an", "playing music", "ich höre musik",
                "winston music on", "winston musik an",
            ]):
                self.audio.set_music_mode(True)
                self.state.set_music_mode(True)
                msg = "Music mode on. I'll only listen for louder, direct speech." if lang != "de" else "Musikmodus an. Ich höre nur auf lautere, direkte Sprache."
                self.tts.speak_async(msg)
                self.state.set_status("idle")
                tracker.discard(interaction_id)
                return

            if any(phrase in text_lower for phrase in [
                "music mode off", "musik modus aus", "music off", "musik aus",
                "winston music off", "winston musik aus",
            ]):
                self.audio.set_music_mode(False)
                self.state.set_music_mode(False)
                msg = "Music mode off. Listening normally." if lang != "de" else "Musikmodus aus. Ich höre normal."
                self.tts.speak_async(msg)
                self.state.set_status("idle")
                tracker.discard(interaction_id)
                return

            self.state.set_status("thinking")

            # Decide what context is needed for this query
            needs_image = self.llm.needs_visual_context(text)
            context_purpose = "lightweight" if self._is_simple_query(text) else "conversation"

            # Check if we can reuse cached context (continuation window)
            from config import ALWAYS_LISTEN_CONTINUATION_WINDOW
            in_continuation = (time.monotonic() - self._last_addressed_time) < ALWAYS_LISTEN_CONTINUATION_WINDOW
            context_fresh = (time.monotonic() - self._cached_context_time) < ALWAYS_LISTEN_CONTINUATION_WINDOW

            t0 = time.time()
            if in_continuation and context_fresh and self._cached_context:
                # Reuse context from the previous utterance in this conversation burst
                context = self._cached_context
                frame_bytes = self._cached_frame_bytes if needs_image else None
                logger.info("[transcription] Reusing cached context (%.1fs old)", time.monotonic() - self._cached_context_time)
            else:
                # Fetch frame (only if visual) + context in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    if needs_image:
                        frame_future = executor.submit(self.camera.get_frame_bytes)
                    context_future = executor.submit(self.memory.assemble_context, query=text, purpose=context_purpose)
                    frame_bytes = frame_future.result(timeout=5) if needs_image else None
                    context = context_future.result(timeout=5)

                # Cache for potential continuation
                self._cached_context = context
                self._cached_context_time = time.monotonic()
                self._cached_frame_bytes = frame_bytes

            # Inject recent agent result so Haiku knows about follow-ups
            recent_agent = self._get_recent_agent_result()
            if recent_agent:
                context = (context or "") + (
                    f"\n\n[Recent computer task — completed moments ago]\n"
                    f"Request: {recent_agent['query']}\n"
                    f"Result: {recent_agent['result'][:500]}"
                )

            # Inject last user utterance for speech continuations
            if self._last_user_utterance:
                context = (context or "") + (
                    f'\n\n[Previous speech fragment from Roberto (moments ago)]\n"{self._last_user_utterance}"'
                )

            logger.info("[transcription] Frame: %s, context: %s (%.1fms)",
                        "included" if frame_bytes else "skipped", context_purpose, (time.time() - t0) * 1000)

            # Mark context ready for latency tracking
            if interaction_id:
                tracker.mark(interaction_id, "context_ready")

            # Build conversation history for multi-turn context
            from config import VOICE_CONVERSATION_HISTORY_TURNS
            conversation_history = self._build_conversation_history(max_turns=VOICE_CONVERSATION_HISTORY_TURNS)

            # Intelligent routing — Claude Haiku decides what to do
            t0 = time.time()

            if STREAMING_ENABLED:
                # Streaming path: sentences are sent to TTS as they arrive
                sentences_streamed = []
                first_sentence_sent = False

                def _on_sentence(sentence_text: str):
                    nonlocal first_sentence_sent
                    iid = None
                    if not first_sentence_sent:
                        first_sentence_sent = True
                        iid = interaction_id
                        if interaction_id:
                            tracker.mark(interaction_id, "llm_first_token")
                        self.tts.begin_streaming_response()
                        self.state.set_status("speaking")
                    sentences_streamed.append(sentence_text)
                    self.tts.speak_async(sentence_text, interaction_id=iid)

                self._streaming_abort.clear()
                action, data = self.llm.process_user_input_streaming(
                    text=text,
                    frame_bytes=frame_bytes,
                    context=context,
                    language=lang,
                    conversation_history=conversation_history,
                    on_sentence=_on_sentence,
                    abort_event=self._streaming_abort,
                )
            else:
                # Non-streaming fallback
                action, data = self.llm.process_user_input(
                    text=text,
                    frame_bytes=frame_bytes,
                    context=context,
                    language=lang,
                    conversation_history=conversation_history,
                )
                sentences_streamed = []
                first_sentence_sent = False

            routing_ms = (time.time() - t0) * 1000
            logger.info("[transcription] Routed to '%s' (%.0fms)", action, routing_ms)

            # Mark LLM done for latency tracking
            if interaction_id:
                tracker.mark(interaction_id, "llm_done")

            if action == "shutdown":
                if sentences_streamed:
                    self.tts.interrupt()
                logger.info("[transcription] Shutdown command: '%s'", text)
                from personality import get_catchphrase

                msg = get_catchphrase("farewell", lang)
                self.state.set_status("speaking")
                tracker.finish(interaction_id)
                self.tts.speak(msg)
                self._running = False
                return

            if action == "agent":
                if sentences_streamed:
                    self.tts.interrupt()
                self._spawn_agent_task(data.get("task", text), lang)
                tracker.finish(interaction_id)
                return

            if action == "note":
                if sentences_streamed:
                    self.tts.interrupt()
                self._handle_note_from_intent(data.get("content", text), lang)
                tracker.finish(interaction_id)
                return

            # Conversation — response already streamed (or available in data)
            response = data.get("text", "")
            if response:
                # End streaming response (signals TTS worker to fire on_speaking_stop)
                if sentences_streamed:
                    self.tts.end_streaming_response()
                else:
                    # Non-streaming path, or response too short for sentence boundary
                    self.state.set_status("speaking")
                    self.tts.speak_async(response, interaction_id=interaction_id)

                self._last_winston_response_time = time.monotonic()
                self._last_user_utterance = text
                self.state.add_conversation("user", text)
                self.state.add_conversation("winston", response)

                conversation_text = f"Q: {text}\nA: {response}"
                self.memory.store(conversation_text, entry_type="conversation")
                self.state.set_cost(self.cost_tracker.get_daily_cost())
                self.state.set_memory_stats(self.memory.entry_count, self.memory.semantic.fact_count)

                # Extract facts from genuine conversations (skip error fallbacks)
                if not response.startswith("Sorry"):
                    threading.Thread(
                        target=self.memory.extract_facts_from_text,
                        args=(conversation_text, self.llm),
                        daemon=True,
                        name="fact-extraction",
                    ).start()
            else:
                if sentences_streamed:
                    self.tts.end_streaming_response()
                from personality import get_catchphrase

                self.tts.speak_async(get_catchphrase("error_process", "en"))
                self.state.set_status("idle")
                tracker.discard(interaction_id)

        except Exception as e:
            logger.error("[transcription] Error: %s", e, exc_info=True)
            tracker.discard(interaction_id)
            try:
                from personality import get_catchphrase

                self.tts.speak_async(get_catchphrase("error_generic", "en"))
            except Exception:
                pass
            self.state.set_status("idle")

    def _on_ambient_transcription(self, text: str, lang: str = "en", interaction_id: str = None) -> None:
        """Called when always-listening detects speech.

        Uses local heuristic to decide if speech is addressed to Winston.
        If addressed, routes through _on_transcription() which uses Claude for routing.
        Supports continuations: if a recent fragment was addressed to Winston,
        subsequent fragments within the continuation window skip intent classification.
        """
        from utils.latency_tracker import LatencyTracker

        tracker = LatencyTracker.get()

        try:
            if not text.strip():
                tracker.discard(interaction_id)
                return

            logger.info("[ambient] Heard [%s]: '%s'", lang, text)

            # ━━━━ CONTINUATION CHECK ━━━━
            # If a recent fragment was addressed to Winston, treat this as continuation
            from config import ALWAYS_LISTEN_CONTINUATION_WINDOW

            if (time.monotonic() - self._last_addressed_time) < ALWAYS_LISTEN_CONTINUATION_WINDOW:
                logger.info(
                    "[ambient] Intent: continuation (%.1fs since last addressed)",
                    time.monotonic() - self._last_addressed_time,
                )
                self._last_addressed_time = time.monotonic()
                if interaction_id:
                    tracker.mark(interaction_id, "intent_done")
                self._on_transcription(text, lang, interaction_id=interaction_id)
                return

            # ━━━━ NORMAL CLASSIFICATION ━━━━
            conversation_window = 30.0
            recently_spoke = (time.monotonic() - self._last_winston_response_time) < conversation_window

            t0 = time.time()
            local_result = self.llm.classify_intent_local(text, conversation_active=recently_spoke)

            if local_result is True:
                source = "conversational context" if recently_spoke else "local"
                logger.info("[ambient] Intent: for Winston (%s, %.0fms)", source, (time.time() - t0) * 1000)
                self._last_addressed_time = time.monotonic()
                if interaction_id:
                    tracker.mark(interaction_id, "intent_done")
                self._on_transcription(text, lang, interaction_id=interaction_id)

            elif local_result is False:
                logger.info("[ambient] Intent: not for Winston (local, %.0fms)", (time.time() - t0) * 1000)
                tracker.discard(interaction_id)
                from config import ALWAYS_LISTEN_STORE_REJECTED

                if ALWAYS_LISTEN_STORE_REJECTED:
                    self.memory.store(
                        f'[Ambient] Roberto said (not to Winston): "{text}"',
                        entry_type="observation",
                        activity="ambient speech",
                    )

            else:
                # Ambiguous — single API call to check if addressed
                t0 = time.time()
                is_addressed = self.llm.classify_intent(text)
                logger.info(
                    "[ambient] Intent: %s (API, %.0fms)",
                    "for Winston" if is_addressed else "not for Winston",
                    (time.time() - t0) * 1000,
                )

                if is_addressed:
                    self._last_addressed_time = time.monotonic()
                    if interaction_id:
                        tracker.mark(interaction_id, "intent_done")
                    self._on_transcription(text, lang, interaction_id=interaction_id)
                else:
                    tracker.discard(interaction_id)
                    from config import ALWAYS_LISTEN_STORE_REJECTED

                    if ALWAYS_LISTEN_STORE_REJECTED:
                        self.memory.store(
                            f'[Ambient] Roberto said (not to Winston): "{text}"',
                            entry_type="observation",
                            activity="ambient speech",
                        )

        except Exception as e:
            logger.error("[ambient] Error: %s", e, exc_info=True)
            tracker.discard(interaction_id)

    def _get_recent_agent_result(self, max_age_seconds: float = 300) -> Optional[dict]:
        """Get the most recent completed agent task within max_age_seconds.

        Returns the task dict (with 'query' and 'result') or None.
        Used to give follow-up agent tasks context about what was just done.
        """
        tasks = self._agent_store.get("tasks", [])
        now = datetime.now()
        for task in reversed(tasks):
            if task.get("status") != "completed" or not task.get("result"):
                continue
            completed_at = task.get("completed_at")
            if completed_at:
                try:
                    age = (now - datetime.fromisoformat(completed_at)).total_seconds()
                    if age <= max_age_seconds:
                        return task
                except (ValueError, TypeError):
                    continue
        return None

    def _spawn_agent_task(self, text: str, lang: str = "en"):
        """Spawn a background agent to investigate a code problem."""
        if not self._agent_lock.acquire(blocking=False):
            from personality import get_catchphrase

            msg = get_catchphrase("agent_busy", lang)
            self.tts.speak_async(msg)
            return
        task_id = str(uuid.uuid4())
        logger.info("[agent] Spawning agent task %s: '%s'", task_id[:8], text)

        self.state.add_conversation("user", text)
        self.state.set_status("speaking")
        from personality import get_catchphrase

        msg = get_catchphrase("agent_starting", lang)
        self.tts.speak_async(msg)

        # Persist task to disk (survives crashes)
        task_record = {
            "id": task_id,
            "query": text,
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "result": None,
            "reported": False,
        }
        self._agent_store.append_to_list("tasks", task_record, max_items=50)
        self.state.add_agent_task(task_record)

        # Assemble context for the agent (memory + recent agent result for follow-ups)
        context = self.memory.assemble_context(query=text, purpose="conversation")
        recent = self._get_recent_agent_result()
        if recent:
            context += (
                f"\n\nPrevious agent task (completed recently):\n"
                f"  Request: {recent['query']}\n"
                f"  Result: {recent['result'][:500]}"
            )

        threading.Thread(
            target=self._run_agent_task,
            args=(task_id, text, context),
            daemon=True,
            name="agent-task",
        ).start()

    def _run_agent_task(self, task_id: str, task: str, context: str = None):
        """Background thread: run agent with Computer Use and speak findings."""
        try:  # outer try for _agent_lock release
            from brain.agent_executor import AgentExecutor
            from brain.agent_tools import create_default_registry
            from config import COMPUTER_USE_DISPLAY_HEIGHT, COMPUTER_USE_DISPLAY_WIDTH, COMPUTER_USE_ENABLED

            registry = create_default_registry()

            computer = None
            if COMPUTER_USE_ENABLED:
                from brain.computer_use import MacOSComputerController

                computer = MacOSComputerController(
                    display_width=COMPUTER_USE_DISPLAY_WIDTH,
                    display_height=COMPUTER_USE_DISPLAY_HEIGHT,
                )

            executor = AgentExecutor(
                self.llm.client,
                self.cost_tracker,
                registry,
                computer_controller=computer,
            )

            full_task = task
            if context:
                full_task = f"{task}\n\nContext from memory:\n{context}"

            findings = executor.run(full_task)

            if findings:
                # Persist result
                self._agent_store.update_in_list(
                    "tasks",
                    task_id,
                    {
                        "status": "completed",
                        "completed_at": datetime.now().isoformat(),
                        "result": findings,
                        "reported": True,
                    },
                )
                self.state.update_agent_task(task_id, "completed", findings)

                # Summarize if too long for TTS
                if len(findings) > 500:
                    summary = self.llm.text_only_chat(
                        f"Summarize these findings in 2-3 spoken sentences:\n\n{findings}",
                        max_tokens=150,
                    )
                    findings_spoken = summary or findings[:500]
                else:
                    findings_spoken = findings

                self.state.add_conversation("agent", findings_spoken)
                self.state.set_status("speaking")
                self.tts.speak_async(findings_spoken)
                self.memory.store(
                    f"[Agent Investigation]\nQ: {task}\nFindings: {findings}",
                    entry_type="conversation",
                )
                self.state.set_cost(self.cost_tracker.get_daily_cost())
            else:
                self._agent_store.update_in_list(
                    "tasks",
                    task_id,
                    {
                        "status": "completed",
                        "completed_at": datetime.now().isoformat(),
                        "result": "No conclusive findings.",
                        "reported": True,
                    },
                )
                self.state.update_agent_task(task_id, "completed", "No conclusive findings.")
                self.state.set_status("speaking")
                from personality import get_catchphrase

                self.tts.speak_async(get_catchphrase("agent_no_findings", "en"))

        except Exception as e:
            logger.error("[agent] Task failed: %s", e, exc_info=True)
            self._agent_store.update_in_list(
                "tasks",
                task_id,
                {
                    "status": "failed",
                    "completed_at": datetime.now().isoformat(),
                    "result": f"Error: {e}",
                    "reported": True,
                },
            )
            self.state.update_agent_task(task_id, "failed", f"Error: {e}")
            self.state.set_status("speaking")
            from personality import get_catchphrase

            self.tts.speak_async(get_catchphrase("error_investigation", "en"))
        finally:
            self._agent_lock.release()

    def _handle_note_from_intent(self, note_text: str, lang: str = "en"):
        """Save note content (already extracted by intent classifier) to persistent store + dashboard."""
        from personality import get_catchphrase

        if not note_text or len(note_text.strip()) < 3:
            self.tts.speak_async(get_catchphrase("note_prompt", lang))
            return

        note_text = note_text.strip()
        note = {
            "id": str(uuid.uuid4()),
            "text": note_text,
            "created_at": datetime.now().isoformat(),
            "source": "voice",
            "done": False,
        }
        self._notes_store.append_to_list("notes", note, max_items=100)
        self.state.add_note(note)
        logger.info("[notes] Saved note: '%s'", note_text)
        self.state.add_conversation("user", note_text)
        label = get_catchphrase("note_label", lang)
        self.state.add_conversation("winston", f"{label}: {note_text}")
        self.state.set_status("speaking")
        confirm = get_catchphrase("note_confirm", lang).format(text=note_text)
        self.tts.speak_async(confirm)
        self._last_winston_response_time = time.monotonic()

    def _check_pending_agent_results(self):
        """On startup, report any agent results from previous sessions."""
        if self._agent_store is None:
            return
        tasks = self._agent_store.get("tasks", [])
        changed = False

        # Mark interrupted tasks as failed
        for task in tasks:
            if task.get("status") == "running":
                task["status"] = "failed"
                task["result"] = "Interrupted by shutdown"
                task["reported"] = False
                changed = True

        # Report unreported results
        unreported = [t for t in tasks if not t.get("reported") and t["status"] in ("completed", "failed")]
        for task in unreported:
            if task["status"] == "completed" and task.get("result"):
                result = task["result"]
                if len(result) > 500:
                    summary = self.llm.text_only_chat(
                        f"Summarize briefly: {result}",
                        max_tokens=150,
                    )
                    result = summary or result[:500]
                from personality import get_catchphrase

                self.tts.speak_async(get_catchphrase("agent_previous", "en").format(result=result))
                self.state.add_conversation("agent", f"[Previous investigation] {result}")
            elif task["status"] == "failed":
                from personality import get_catchphrase

                self.tts.speak_async(
                    get_catchphrase("agent_interrupted", "en").format(query=task.get("query", "unknown task"))
                )
            task["reported"] = True
            changed = True

        if changed:
            self._agent_store.update("tasks", tasks)
            self.state.set_agent_tasks(tasks)

    def _enter_low_budget_mode(self):
        """Reduce API usage when daily budget is hit."""
        if self._capture_interval < 30.0:
            logger.warning("Daily budget exceeded. Entering low-budget mode.")
            self._capture_interval = 30.0
            self.proactive.update_intervals(60.0)
            from personality import get_catchphrase

            self.tts.speak_async(get_catchphrase("budget_warning", "en"))

    def _shutdown_handler(self, signum, frame):
        """Handle Ctrl+C gracefully. Second Ctrl+C forces immediate exit."""
        if not self._running:
            # Second signal — force exit
            logger.info("Force shutdown (second signal).")
            sys.exit(1)
        logger.info("Shutdown signal received. Press Ctrl+C again to force quit.")
        self._running = False

    def shutdown(self):
        """Clean up all resources. Non-blocking — exits within ~3 seconds."""
        logger.info("Shutting down WINSTON...")

        # Non-blocking goodbye
        from personality import get_catchphrase

        self.tts.speak_async(get_catchphrase("farewell", "en"))

        # Memory session summary with timeout (don't block shutdown)
        mem_thread = threading.Thread(target=self._safe_shutdown_memory, daemon=True, name="shutdown-memory")
        mem_thread.start()
        mem_thread.join(timeout=3.0)
        if mem_thread.is_alive():
            logger.warning("Memory shutdown timed out (3s), skipping")

        # Brief pause for goodbye TTS to play
        time.sleep(1.5)

        if hasattr(self, '_telegram_bot'):
            self._telegram_bot.stop()

        self.audio.stop()
        self.camera.stop()
        self.tts.stop()

        report = self.cost_tracker.get_daily_report()
        logger.info("\n%s", report)
        logger.info("WINSTON offline.")

    def _safe_shutdown_memory(self):
        """Run memory shutdown in a thread so it can be timed out."""
        try:
            self.memory.shutdown_session(self.llm)
        except Exception as e:
            logger.error("Session shutdown error: %s", e)

    def _consolidate_memory(self):
        """Run memory consolidation in background at startup."""
        try:
            deleted = self.memory.consolidate()
            if deleted > 0:
                logger.info("Memory consolidation: removed %d old entries", deleted)
        except Exception as e:
            logger.error("Memory consolidation failed: %s", e)

    def _log_step(self, message: str):
        logger.info("%s...", message)

    def _log_banner(self, message: str):
        separator = "=" * 50
        logger.info(separator)
        logger.info(message)
        logger.info(separator)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Winston Workshop Assistant")
    parser.add_argument("--personality", "-p", default=None, help="Personality preset name or path to YAML file")
    args = parser.parse_args()

    # Load personality before anything else — other modules read it via get_personality()
    from personality import load_personality, set_personality

    personality_name = args.personality or os.getenv("WINSTON_PERSONALITY", "default")
    try:
        set_personality(load_personality(personality_name))
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    winston = Winston()

    if not winston.start():
        logger.error("Failed to initialize. Exiting.")
        sys.exit(1)

    try:
        winston.run()
    except KeyboardInterrupt:
        pass
    finally:
        winston.shutdown()


if __name__ == "__main__":
    main()
