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
import signal
import sys
import time
import threading
import uuid
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    CAPTURE_INTERVAL,
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
        self._agent_running = False
        # Persistent stores (initialized in start())
        self._agent_store = None
        self._notes_store = None
        # All other modules created lazily in start()

    def start(self) -> bool:
        """Initialize all modules. Each import is timed for diagnostics."""
        self._log_banner("Initializing WINSTON Workshop Assistant")
        self.state.set_status("initializing")

        # ── 0. Dashboard FIRST (imports fastapi) ──────────────────────
        self._log_step("Starting dashboard")
        t = time.time()
        from dashboard.server import (
            create_app, start_server, set_audio,
            set_camera, set_memory, set_cost_tracker,
            set_notes_store,
        )
        logger.info("  dashboard module loaded (%.1fs)", time.time() - t)
        dashboard_app = create_app(self.state, None)
        start_server(dashboard_app, port=8420)
        logger.info("Dashboard live: http://localhost:8420")

        # ── 1. Camera (imports cv2) ───────────────────────────────────
        self._log_step("Initializing camera")
        t = time.time()
        from perception.camera import Camera
        from config import CAMERA_SOURCE
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
        from perception.tts import TTSEngine
        from brain.claude_client import ClaudeClient
        from brain.memory import Memory
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
        threading.Thread(
            target=self._consolidate_memory, daemon=True, name="memory-consolidate"
        ).start()

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
            f"System online. Memory: {self.memory.entry_count} episodes, "
            f"{facts_count} facts. Mode: always-listening."
        )
        self.tts.speak_async("I'm online. I can see your workshop.")

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
                if hasattr(self.audio, '_current_audio_level'):
                    self.state.set_audio_level(self.audio._current_audio_level)

                # Update always-listen state for dashboard
                self.state.set_always_listen_state(self.audio.always_listen_state)

                # Reset to idle when TTS finishes speaking
                if self.state.status == "speaking" and not self.tts.is_speaking:
                    self.state.set_status("idle")

                # 1. Capture frame
                frame_bytes = self.camera.get_frame_bytes()

                # 2. Scene change detection -> Flash analysis
                if frame_bytes and self.camera.scene_changed():
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
                            logger.warning("Concern detected: %s", concerns)

                # 3. Proactive check
                if frame_bytes and self.proactive.should_check():
                    context = self.memory.assemble_context(purpose="proactive")
                    self.proactive.check(frame_bytes, recent_context=context)

                # 4. Budget check
                if not self.cost_tracker.check_budget():
                    self._enter_low_budget_mode()

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

    def _on_bargein(self):
        """Called when barge-in (user speech during TTS) is detected."""
        logger.info("Barge-in detected — interrupting speech")
        self._streaming_abort.set()
        self.tts.interrupt()

    def _on_transcription(self, text: str, lang: str = "en", model: str = None):
        """Called when user speech is transcribed. Runs in a daemon thread.

        Uses Claude Haiku with tool-use routing: Haiku decides whether to answer
        directly, delegate to the agent, save a note, or shut down.
        No keyword matching — end-to-end intelligence.
        """
        try:
            logger.info("[transcription] User said [%s]: '%s'", lang, text)

            if not text.strip():
                logger.info("[transcription] Empty text, ignoring")
                self.state.set_status("idle")
                return

            self.state.set_status("thinking")

            # Get frame + context in parallel
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=2) as executor:
                frame_future = executor.submit(self.camera.get_frame_bytes)
                context_future = executor.submit(
                    self.memory.assemble_context, query=text, purpose="conversation"
                )
                frame_bytes = frame_future.result(timeout=5)
                context = context_future.result(timeout=5)
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
                    f"\n\n[Previous speech fragment from Roberto (moments ago)]\n"
                    f"\"{self._last_user_utterance}\""
                )

            logger.info("[transcription] Frame + context ready (%.1fms)", (time.time() - t0) * 1000)

            # Build conversation history for multi-turn context
            conversation_history = self._build_conversation_history()

            # Intelligent routing — Claude Haiku decides what to do
            t0 = time.time()
            action, data = self.llm.process_user_input(
                text=text,
                frame_bytes=frame_bytes,
                context=context,
                language=lang,
                conversation_history=conversation_history,
            )
            logger.info("[transcription] Routed to '%s' (%.0fms)", action, (time.time() - t0) * 1000)

            if action == "shutdown":
                logger.info("[transcription] Shutdown command: '%s'", text)
                msg = "Gehe offline. Tschüss." if lang == "de" else "Going offline. Goodbye."
                self.state.set_status("speaking")
                self.tts.speak(msg)
                self._running = False
                return

            if action == "agent":
                self._spawn_agent_task(data.get("task", text), lang)
                return

            if action == "note":
                self._handle_note_from_intent(data.get("content", text), lang)
                return

            # Conversation — Haiku already generated the response
            response = data.get("text", "")
            if response:
                self._last_winston_response_time = time.monotonic()
                self._last_user_utterance = text
                self.state.add_conversation("user", text)
                self.state.add_conversation("winston", response)
                self.state.set_status("speaking")
                self.tts.speak_async(response)

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
                self.tts.speak_async("Sorry, I couldn't process that. Try again?")
                self.state.set_status("idle")

        except Exception as e:
            logger.error("[transcription] Error: %s", e, exc_info=True)
            try:
                self.tts.speak_async("Sorry, something went wrong. Try again?")
            except Exception:
                pass
            self.state.set_status("idle")

    def _on_ambient_transcription(self, text: str, lang: str = "en") -> None:
        """Called when always-listening detects speech.

        Uses local heuristic to decide if speech is addressed to Winston.
        If addressed, routes through _on_transcription() which uses Claude for routing.
        Supports continuations: if a recent fragment was addressed to Winston,
        subsequent fragments within the continuation window skip intent classification.
        """
        try:
            if not text.strip():
                return

            logger.info("[ambient] Heard [%s]: '%s'", lang, text)

            # ━━━━ CONTINUATION CHECK ━━━━
            # If a recent fragment was addressed to Winston, treat this as continuation
            from config import ALWAYS_LISTEN_CONTINUATION_WINDOW
            if (time.monotonic() - self._last_addressed_time) < ALWAYS_LISTEN_CONTINUATION_WINDOW:
                logger.info("[ambient] Intent: continuation (%.1fs since last addressed)",
                            time.monotonic() - self._last_addressed_time)
                self._last_addressed_time = time.monotonic()
                self._on_transcription(text, lang)
                return

            # ━━━━ NORMAL CLASSIFICATION ━━━━
            conversation_window = 30.0
            recently_spoke = (time.monotonic() - self._last_winston_response_time) < conversation_window

            t0 = time.time()
            local_result = self.llm.classify_intent_local(text, conversation_active=recently_spoke)

            if local_result is True:
                source = "conversational context" if recently_spoke else "local"
                logger.info("[ambient] Intent: for Winston (%s, %.0fms)",
                            source, (time.time() - t0) * 1000)
                self._last_addressed_time = time.monotonic()
                self._on_transcription(text, lang)

            elif local_result is False:
                logger.info("[ambient] Intent: not for Winston (local, %.0fms)",
                            (time.time() - t0) * 1000)
                from config import ALWAYS_LISTEN_STORE_REJECTED
                if ALWAYS_LISTEN_STORE_REJECTED:
                    self.memory.store(
                        f"[Ambient] Roberto said (not to Winston): \"{text}\"",
                        entry_type="observation",
                        activity="ambient speech",
                    )

            else:
                # Ambiguous — single API call to check if addressed
                t0 = time.time()
                is_addressed = self.llm.classify_intent(text)
                logger.info("[ambient] Intent: %s (API, %.0fms)",
                            "for Winston" if is_addressed else "not for Winston",
                            (time.time() - t0) * 1000)

                if is_addressed:
                    self._last_addressed_time = time.monotonic()
                    self._on_transcription(text, lang)
                else:
                    from config import ALWAYS_LISTEN_STORE_REJECTED
                    if ALWAYS_LISTEN_STORE_REJECTED:
                        self.memory.store(
                            f"[Ambient] Roberto said (not to Winston): \"{text}\"",
                            entry_type="observation",
                            activity="ambient speech",
                        )

        except Exception as e:
            logger.error("[ambient] Error: %s", e, exc_info=True)

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
        if self._agent_running:
            msg = ("Ich arbeite noch an der vorherigen Aufgabe."
                   if lang == "de" else "I'm still working on the previous task.")
            self.tts.speak_async(msg)
            return
        self._agent_running = True
        task_id = str(uuid.uuid4())
        logger.info("[agent] Spawning agent task %s: '%s'", task_id[:8], text)

        self.state.add_conversation("user", text)
        self.state.set_status("speaking")
        msg = ("Ich schaue mir das an. Ich melde mich." if lang == "de"
               else "Let me investigate that. I'll let you know what I find.")
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
        try:  # noqa: outer try for _agent_running reset
            from brain.agent_executor import AgentExecutor
            from brain.agent_tools import create_default_registry
            from config import COMPUTER_USE_ENABLED, COMPUTER_USE_DISPLAY_WIDTH, COMPUTER_USE_DISPLAY_HEIGHT

            registry = create_default_registry()

            computer = None
            if COMPUTER_USE_ENABLED:
                from brain.computer_use import MacOSComputerController
                computer = MacOSComputerController(
                    display_width=COMPUTER_USE_DISPLAY_WIDTH,
                    display_height=COMPUTER_USE_DISPLAY_HEIGHT,
                )

            executor = AgentExecutor(
                self.llm.client, self.cost_tracker, registry,
                computer_controller=computer,
            )

            full_task = task
            if context:
                full_task = f"{task}\n\nContext from memory:\n{context}"

            findings = executor.run(full_task)

            if findings:
                # Persist result
                self._agent_store.update_in_list("tasks", task_id, {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "result": findings,
                    "reported": True,
                })
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

                self.state.add_conversation("winston", findings_spoken)
                self.state.set_status("speaking")
                self.tts.speak_async(findings_spoken)
                self.memory.store(
                    f"[Agent Investigation]\nQ: {task}\nFindings: {findings}",
                    entry_type="conversation",
                )
                self.state.set_cost(self.cost_tracker.get_daily_cost())
            else:
                self._agent_store.update_in_list("tasks", task_id, {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "result": "No conclusive findings.",
                    "reported": True,
                })
                self.state.update_agent_task(task_id, "completed", "No conclusive findings.")
                self.state.set_status("speaking")
                self.tts.speak_async("I couldn't find anything conclusive. Can you give me more details?")

        except Exception as e:
            logger.error("[agent] Task failed: %s", e, exc_info=True)
            self._agent_store.update_in_list("tasks", task_id, {
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "result": f"Error: {e}",
                "reported": True,
            })
            self.state.update_agent_task(task_id, "failed", f"Error: {e}")
            self.state.set_status("speaking")
            self.tts.speak_async("Sorry, the investigation ran into an error.")
        finally:
            self._agent_running = False

    def _handle_note_from_intent(self, note_text: str, lang: str = "en"):
        """Save note content (already extracted by intent classifier) to persistent store + dashboard."""
        if not note_text or len(note_text.strip()) < 3:
            msg = "Was soll ich notieren?" if lang == "de" else "What should I write down?"
            self.tts.speak_async(msg)
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
        label = "Notiert" if lang == "de" else "Noted"
        self.state.add_conversation("winston", f"{label}: {note_text}")
        self.state.set_status("speaking")
        confirm = f"Notiert: {note_text}" if lang == "de" else f"Got it. Noted: {note_text}"
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
                        f"Summarize briefly: {result}", max_tokens=150,
                    )
                    result = summary or result[:500]
                self.tts.speak_async(f"From a previous investigation: {result}")
                self.state.add_conversation("winston", f"[Previous investigation] {result}")
            elif task["status"] == "failed":
                self.tts.speak_async(f"A previous investigation was interrupted: {task.get('query', 'unknown task')}")
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
            self.tts.speak_async("Budget limit approaching. I'll reduce my observation frequency.")

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
        self.tts.speak_async("Going offline. Goodbye.")

        # Memory session summary with timeout (don't block shutdown)
        mem_thread = threading.Thread(
            target=self._safe_shutdown_memory, daemon=True, name="shutdown-memory"
        )
        mem_thread.start()
        mem_thread.join(timeout=3.0)
        if mem_thread.is_alive():
            logger.warning("Memory shutdown timed out (3s), skipping")

        # Brief pause for goodbye TTS to play
        time.sleep(1.5)

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
