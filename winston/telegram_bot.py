"""Telegram bot interface for Winston — runs as a daemon thread alongside voice.

Features: Haiku routing, agent tasks with live progress, inline keyboards,
MarkdownV2 formatting, message queueing, camera bridge, voice notes,
session persistence, webhook support, and security hardening.
"""

import asyncio
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime
from io import BytesIO
from typing import Optional

import numpy as np
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logger = logging.getLogger("winston.telegram")

# ── Audit Logger ──────────────────────────────────────────────────
_audit_logger = logging.getLogger("winston.telegram.audit")
_audit_logger.setLevel(logging.INFO)
_audit_logger.propagate = False
os.makedirs("winston_memory", exist_ok=True)
_audit_handler = logging.FileHandler("winston_memory/telegram_audit.log")
_audit_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
_audit_logger.addHandler(_audit_handler)

# Common German words for lightweight language detection
_GERMAN_WORDS = frozenset({
    "der", "die", "das", "und", "ist", "ich", "ein", "eine", "nicht",
    "auf", "mit", "haben", "werden", "kann", "sind", "auch", "aber",
    "noch", "nur", "schon", "wenn", "oder", "wie", "mehr", "nach",
    "zum", "zur", "den", "dem", "des", "von", "aus", "mir", "mich",
    "dir", "dich", "wir", "sie", "was", "wer", "wo", "hier",
})

# MarkdownV2 special chars that must be escaped
_MDV2_ESCAPE_CHARS = set(r"_*[]()~`>#+-=|{}.!")

# Tool-name-to-emoji mapping for progress messages
_TOOL_EMOJI = {
    "web_search": "\U0001f50d",
    "fetch_webpage": "\U0001f310",
    "read_local_file": "\U0001f4c1",
    "search_local_files": "\U0001f4c1",
    "list_local_directory": "\U0001f4c1",
    "run_shell_command": "\u2699\ufe0f",
    "computer": "\U0001f5a5\ufe0f",
    "get_current_time": "\U0001f552",
    "open_url": "\U0001f310",
}

# ── Control character regex (strip \x00-\x08, \x0b, \x0c, \x0e-\x1f, \x7f) ─
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_input(text: str) -> str:
    """Strip control characters and limit length to 4000 chars."""
    cleaned = _CONTROL_CHAR_RE.sub("", text)
    return cleaned[:4000]


def _escape_markdown_v2(text: str) -> str:
    """Escape all MarkdownV2 special characters."""
    result = []
    for ch in text:
        if ch in _MDV2_ESCAPE_CHARS:
            result.append("\\")
        result.append(ch)
    return "".join(result)


def _format_as_markdown_v2(text: str) -> str:
    """Format text for Telegram MarkdownV2, preserving code blocks."""
    parts = text.split("```")
    formatted = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            formatted.append(_escape_markdown_v2(part))
        else:
            # Code block — preserve content, wrap in ```
            formatted.append(f"```{part}```")
    return "".join(formatted)


def _tool_emoji(name: str) -> str:
    """Get emoji for a tool name."""
    if name in _TOOL_EMOJI:
        return _TOOL_EMOJI[name]
    if name.startswith("github_"):
        return "\U0001f419"
    return "\U0001f527"


def _split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split text into chunks that fit Telegram's message size limit."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, max_len)
        if split_at < max_len // 2:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


class _RateLimiter:
    """Sliding-window rate limiter (per-user, thread-safe)."""

    def __init__(self, max_per_hour: int = 30):
        self._max = max_per_hour
        self._timestamps: dict[int, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, user_id: int) -> bool:
        now = time.time()
        cutoff = now - 3600
        with self._lock:
            ts = [t for t in self._timestamps.get(user_id, []) if t > cutoff]
            if len(ts) >= self._max:
                self._timestamps[user_id] = ts
                return False
            ts.append(now)
            self._timestamps[user_id] = ts
            return True

    def remaining(self, user_id: int) -> int:
        now = time.time()
        cutoff = now - 3600
        with self._lock:
            ts = [t for t in self._timestamps.get(user_id, []) if t > cutoff]
            return max(0, self._max - len(ts))


class TelegramBot:
    """Telegram interface for Winston, bridging into the existing voice pipeline."""

    def __init__(
        self,
        token: str,
        allowed_users: set[int],
        llm,
        memory,
        cost_tracker,
        notes_store,
        agent_store,
        state,
        agent_lock: threading.Lock,
        camera=None,
        stt_provider=None,
        tts_engine=None,
    ):
        self._token = token
        self._allowed_users = allowed_users
        self._llm = llm
        self._memory = memory
        self._cost_tracker = cost_tracker
        self._notes_store = notes_store
        self._agent_store = agent_store
        self._state = state
        self._agent_lock = agent_lock  # kept for backward compat, not used by Telegram
        self._camera = camera
        self._stt_provider = stt_provider
        self._tts_engine = tts_engine

        from config import (
            TELEGRAM_AGENT_RATE_LIMIT_PER_HOUR,
            TELEGRAM_HISTORY_FILE,
            TELEGRAM_MAX_CONCURRENT_AGENTS,
            TELEGRAM_RATE_LIMIT_PER_HOUR,
        )

        self._rate_limiter = _RateLimiter(TELEGRAM_RATE_LIMIT_PER_HOUR)
        self._agent_rate_limiter = _RateLimiter(TELEGRAM_AGENT_RATE_LIMIT_PER_HOUR)
        self._history_lock = threading.Lock()
        self._start_time = time.time()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._application = None

        # Agent concurrency (Telegram-specific, separate from voice pipeline)
        self._telegram_agent_sem = threading.Semaphore(TELEGRAM_MAX_CONCURRENT_AGENTS)

        # Per-user message queues
        self._message_queues: dict[int, asyncio.Queue] = {}
        self._user_active_agents: dict[int, int] = {}
        self._queue_lock = threading.Lock()

        # Task result cache for inline keyboard callbacks (1hr TTL)
        self._task_results: dict[str, dict] = {}
        self._task_results_lock = threading.Lock()

        # Followup tracking
        self._awaiting_followup: dict[int, str] = {}

        # Progress message state per running task
        self._progress_messages: dict[str, dict] = {}

        # Watch tasks (per-user camera monitoring)
        self._watch_tasks: dict[int, asyncio.Task] = {}

        # Blocked users (runtime, persisted in history store)
        self._blocked_users: set[int] = set()

        # Session persistence
        from utils.persistent_store import PersistentStore

        self._history_store = PersistentStore(
            TELEGRAM_HISTORY_FILE,
            default_data={"conversations": {}, "blocked_users": []},
        )
        # Load persisted conversation history (last 20 messages per user)
        saved = self._history_store.get("conversations", {})
        self._conversation_history: dict[int, list[dict]] = {}
        for uid_str, messages in saved.items():
            try:
                self._conversation_history[int(uid_str)] = messages[-20:]
            except (ValueError, TypeError):
                pass
        # Load blocked users
        blocked = self._history_store.get("blocked_users", [])
        self._blocked_users = set(blocked) if isinstance(blocked, list) else set()
        logger.info(
            "[telegram] Loaded %d conversation histories, %d blocked users",
            len(self._conversation_history),
            len(self._blocked_users),
        )

    # ── Audit ─────────────────────────────────────────────────

    @staticmethod
    def _audit(action: str, user_id: int, detail: str = ""):
        _audit_logger.info("user=%d action=%s %s", user_id, action, detail)

    # ── Lifecycle ────────────────────────────────────────────

    def _setup_application(self):
        """Build the Application and register all handlers."""
        app = ApplicationBuilder().token(self._token).build()

        # Core commands
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("note", self._cmd_note))
        app.add_handler(CommandHandler("notes", self._cmd_notes))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("cost", self._cmd_cost))
        app.add_handler(CommandHandler("search", self._cmd_search))
        app.add_handler(CommandHandler("ask", self._cmd_ask))
        app.add_handler(CommandHandler("memory", self._cmd_memory))
        app.add_handler(CommandHandler("forget", self._cmd_forget))
        app.add_handler(CommandHandler("personality", self._cmd_personality))

        # Camera commands
        app.add_handler(CommandHandler(["camera", "cam"], self._cmd_camera))
        app.add_handler(CommandHandler("watch", self._cmd_watch))
        app.add_handler(CommandHandler("stop", self._cmd_stop))

        # Session & security commands
        app.add_handler(CommandHandler("clear", self._cmd_clear))
        app.add_handler(CommandHandler("block", self._cmd_block))
        app.add_handler(CommandHandler("unblock", self._cmd_unblock))

        # Inline keyboard callbacks
        app.add_handler(CallbackQueryHandler(self._handle_callback))

        # Message handlers
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))
        app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._handle_voice))

        # Edited message handler
        app.add_handler(MessageHandler(
            filters.UpdateType.EDITED_MESSAGE & filters.TEXT, self._handle_edit
        ))

        self._application = app
        return app

    def start(self):
        """Blocking — called from the daemon thread. Runs polling or webhook."""
        self._setup_application()

        from config import TELEGRAM_WEBHOOK_URL

        if TELEGRAM_WEBHOOK_URL:
            self._start_webhook(TELEGRAM_WEBHOOK_URL)
        else:
            async def _post_init(_app):
                self._loop = asyncio.get_running_loop()

            self._application.post_init = _post_init
            logger.info("Telegram bot polling started")
            self._application.run_polling(drop_pending_updates=True)

    def _start_webhook(self, webhook_url: str):
        """Register webhook and keep event loop alive for processing updates."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _setup():
            await self._application.initialize()
            await self._application.start()
            from config import TELEGRAM_WEBHOOK_SECRET

            await self._application.bot.set_webhook(
                url=f"{webhook_url}/telegram/webhook",
                secret_token=TELEGRAM_WEBHOOK_SECRET or None,
            )
            logger.info(
                "Telegram webhook registered: %s/telegram/webhook", webhook_url
            )

        loop.run_until_complete(_setup())
        loop.run_forever()  # Keep event loop alive for processing

    # ── Auth & Rate Limiting ─────────────────────────────────

    async def _check_user(self, update: Update) -> bool:
        uid = update.effective_user.id
        if uid in self._blocked_users:
            self._audit("blocked_access", uid)
            return False
        if uid not in self._allowed_users:
            self._audit("unauthorized", uid)
            msg = update.message or (update.callback_query and update.callback_query.message)
            if msg:
                await msg.reply_text("Sorry, I'm not authorized to chat with you.")
            return False
        return True

    async def _check_rate_limit(self, update: Update) -> bool:
        uid = update.effective_user.id
        if not self._rate_limiter.check(uid):
            remaining_min = 60 - int((time.time() % 3600) // 60)
            await update.message.reply_text(
                f"You've reached the message limit (30/hour). Try again in ~{remaining_min} minutes."
            )
            return False
        return True

    # ── Language Detection ───────────────────────────────────

    @staticmethod
    def _detect_language(text: str) -> str:
        words = set(text.lower().split())
        german_matches = words & _GERMAN_WORDS
        if len(german_matches) >= 2:
            return "de"
        if german_matches and any(c in text for c in "\u00e4\u00f6\u00fc\u00df"):
            return "de"
        return "en"

    # ── Conversation History (per-user, persisted) ────────────

    def _build_history(self, user_id: int) -> list[dict]:
        with self._history_lock:
            return list(self._conversation_history.get(user_id, []))

    def _append_history(self, user_id: int, role: str, text: str):
        with self._history_lock:
            history = self._conversation_history.setdefault(user_id, [])
            if history and history[-1]["role"] == role:
                history[-1]["content"] += f" {text}"
            else:
                history.append({"role": role, "content": text})
            while len(history) > 10:
                history.pop(0)
            while history and history[0]["role"] != "user":
                history.pop(0)
            # Persist to disk
            self._history_store.update(
                "conversations",
                {str(uid): msgs for uid, msgs in self._conversation_history.items()},
            )

    # ── Command Handlers ─────────────────────────────────────

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            self._audit("command", update.effective_user.id, "/start")
            await update.message.reply_text(
                "Hello! I'm Winston, your workshop assistant.\n\n"
                "Send me a message and I'll help with whatever you need. "
                "I can answer questions, run computer tasks, and take notes.\n\n"
                "Type /help for all available commands."
            )
        except Exception as e:
            logger.error("Telegram /start error: %s", e, exc_info=True)

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            self._audit("command", update.effective_user.id, "/help")
            await update.message.reply_text(
                "Winston Commands:\n\n"
                "/start \u2014 Welcome message\n"
                "/help \u2014 This help message\n"
                "/note <text> \u2014 Save a note\n"
                "/notes \u2014 List all notes\n"
                "/search <query> \u2014 Web search via agent\n"
                "/ask <question> \u2014 Direct Sonnet response (higher quality)\n"
                "/memory \u2014 Show known facts about you\n"
                "/forget <number> \u2014 Remove a fact by number\n"
                "/personality <name> \u2014 Switch preset (affects voice too)\n"
                "/camera or /cam \u2014 Capture workshop snapshot\n"
                "/watch \u2014 Monitor workshop (30s intervals, 5min)\n"
                "/stop \u2014 Stop watching\n"
                "/clear \u2014 Clear conversation history\n"
                "/status \u2014 System status\n"
                "/cost \u2014 Today's API cost breakdown\n"
                "/block <id> \u2014 Block user (admin)\n"
                "/unblock <id> \u2014 Unblock user (admin)\n\n"
                "Send voice messages for transcription + response.\n"
                "Or send a text/photo for conversation."
            )
        except Exception as e:
            logger.error("Telegram /help error: %s", e, exc_info=True)

    async def _cmd_note(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            if not await self._check_rate_limit(update):
                return
            uid = update.effective_user.id
            self._audit("command", uid, "/note")
            note_text = " ".join(context.args) if context.args else ""
            if not note_text.strip():
                await update.message.reply_text("Usage: /note <text>\nExample: /note buy 10mm bolts")
                return
            note = {
                "id": str(uuid.uuid4()),
                "text": note_text.strip(),
                "created_at": datetime.now().isoformat(),
                "source": "telegram",
                "done": False,
            }
            self._notes_store.append_to_list("notes", note, max_items=100)
            self._state.add_note(note)
            logger.info("[telegram] Saved note: '%s'", note_text.strip())
            await update.message.reply_text(f"Noted: {note_text.strip()}")
        except Exception as e:
            logger.error("Telegram /note error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _cmd_notes(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            self._audit("command", update.effective_user.id, "/notes")
            notes = self._notes_store.get("notes", [])
            if not notes:
                await update.message.reply_text("No notes saved yet.")
                return
            lines = []
            for i, n in enumerate(notes, 1):
                mark = "\u2705" if n.get("done") else "\u2b1c"
                lines.append(f"{mark} {i}. {n.get('text', '?')}")
            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            logger.error("Telegram /notes error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            self._audit("command", update.effective_user.id, "/status")
            uptime_s = time.time() - self._start_time
            hours, remainder = divmod(int(uptime_s), 3600)
            minutes, seconds = divmod(remainder, 60)

            episodes = getattr(self._memory, "entry_count", 0)
            facts = 0
            if hasattr(self._memory, "semantic"):
                facts = getattr(self._memory.semantic, "fact_count", 0)

            daily_cost = self._cost_tracker.get_daily_cost()
            tasks = self._agent_store.get("tasks", [])
            running = sum(1 for t in tasks if t.get("status") == "running")
            completed = sum(1 for t in tasks if t.get("status") == "completed")
            failed = sum(1 for t in tasks if t.get("status") == "failed")

            await update.message.reply_text(
                f"Winston Status\n"
                f"{'=' * 20}\n"
                f"Uptime: {hours}h {minutes}m {seconds}s\n"
                f"Memory: {episodes} episodes, {facts} facts\n"
                f"Cost today: ${daily_cost:.4f}\n"
                f"Agent tasks: {running} running, {completed} done, {failed} failed"
            )
        except Exception as e:
            logger.error("Telegram /status error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _cmd_cost(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            self._audit("command", update.effective_user.id, "/cost")
            report = self._cost_tracker.get_daily_report()
            await update.message.reply_text(report)
        except Exception as e:
            logger.error("Telegram /cost error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _cmd_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            if not await self._check_rate_limit(update):
                return
            uid = update.effective_user.id
            self._audit("command", uid, "/search")
            query = " ".join(context.args) if context.args else ""
            if not query.strip():
                await update.message.reply_text("Usage: /search <query>")
                return
            task = (
                f"Search the web for: {query}. "
                "Use the web_search tool, then summarize the top results clearly."
            )
            chat_id = update.effective_chat.id
            lang = self._detect_language(query)
            await self._spawn_agent_with_progress(task, chat_id, uid, query, lang)
        except Exception as e:
            logger.error("Telegram /search error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _cmd_ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            if not await self._check_rate_limit(update):
                return
            uid = update.effective_user.id
            self._audit("command", uid, "/ask")
            question = " ".join(context.args) if context.args else ""
            if not question.strip():
                await update.message.reply_text("Usage: /ask <question>")
                return

            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

            from config import SMART_MODEL, get_conversation_prompt

            lang = self._detect_language(question)
            system = get_conversation_prompt()
            if lang == "de":
                system = f"IMPORTANT: Respond in German.\n\n{system}"

            history = self._build_history(uid)
            messages = list(history) + [{"role": "user", "content": question}]

            def _call_sonnet():
                response = self._llm.client.messages.create(
                    model=SMART_MODEL,
                    max_tokens=1000,
                    system=system,
                    messages=messages,
                )
                usage = response.usage
                self._cost_tracker.record(
                    "smart",
                    getattr(usage, "input_tokens", 0),
                    getattr(usage, "output_tokens", 0),
                )
                texts = [b.text for b in response.content if hasattr(b, "text") and b.text]
                return " ".join(texts).strip()

            response_text = await asyncio.to_thread(_call_sonnet)
            if response_text:
                await self._send_formatted_message(update, response_text)
                self._store_conversation(uid, question, response_text)
            else:
                await update.message.reply_text("Sorry, I couldn't generate a response.")
        except Exception as e:
            logger.error("Telegram /ask error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            self._audit("command", update.effective_user.id, "/memory")
            if not hasattr(self._memory, "semantic"):
                await update.message.reply_text("Memory system not available.")
                return

            facts = self._memory.semantic._facts
            if not facts:
                await update.message.reply_text("No facts stored yet.")
                return

            by_category: dict[str, list[str]] = {}
            for i, fact in enumerate(facts):
                cat = fact.get("category", "other")
                text = f"{i + 1}. {fact['entity']}: {fact['attribute']} = {fact['value']}"
                by_category.setdefault(cat, []).append(text)

            lines = [f"Known facts ({len(facts)} total):\n"]
            for cat, items in sorted(by_category.items()):
                lines.append(f"\n[{cat.title()}]")
                lines.extend(items)
            lines.append("\nUse /forget <number> to remove a fact.")

            await self._send_long_message(update, "\n".join(lines))
        except Exception as e:
            logger.error("Telegram /memory error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _cmd_forget(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            self._audit("command", update.effective_user.id, "/forget")
            if not context.args or not context.args[0].isdigit():
                await update.message.reply_text(
                    "Usage: /forget <number>\nUse /memory to see fact numbers."
                )
                return

            idx = int(context.args[0]) - 1  # 1-based to 0-based
            facts = self._memory.semantic._facts

            if idx < 0 or idx >= len(facts):
                await update.message.reply_text(
                    f"Invalid number. Use /memory to see {len(facts)} facts."
                )
                return

            fact = facts[idx]
            fact_desc = f"{fact['entity']}: {fact['attribute']} = {fact['value']}"
            facts.pop(idx)

            def _delete():
                self._memory.semantic._save_profile()
                if self._memory.semantic._chromadb_collection:
                    try:
                        fact_id = self._memory.semantic._fact_id(fact)
                        self._memory.semantic._chromadb_collection.delete(ids=[fact_id])
                    except Exception as e:
                        logger.warning("Failed to delete fact from ChromaDB: %s", e)

            await asyncio.to_thread(_delete)
            await update.message.reply_text(f"Removed: {fact_desc}")
        except Exception as e:
            logger.error("Telegram /forget error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _cmd_personality(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            self._audit("command", update.effective_user.id, "/personality")

            from personality import PERSONALITIES_DIR, load_personality, set_personality

            available = [f.stem for f in PERSONALITIES_DIR.glob("*.yaml")]

            if not context.args:
                await update.message.reply_text(
                    f"Available personalities: {', '.join(available)}\n\n"
                    "Usage: /personality <name>\n"
                    "Note: this changes personality globally (voice + Telegram)."
                )
                return

            name = context.args[0].lower()
            if name not in available:
                await update.message.reply_text(
                    f"Unknown personality '{name}'. Available: {', '.join(available)}"
                )
                return

            config = await asyncio.to_thread(load_personality, name)
            set_personality(config)
            await update.message.reply_text(f"Personality switched to: {name}")
        except Exception as e:
            logger.error("Telegram /personality error: %s", e, exc_info=True)
            await self._send_error(update)

    # ── Camera Commands ──────────────────────────────────────

    async def _cmd_camera(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            uid = update.effective_user.id
            self._audit("command", uid, "/camera")

            if not self._camera or not self._camera.is_open:
                await update.message.reply_text("\U0001f4f7 Camera is offline")
                return

            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO
            )

            frame_bytes = self._camera.get_frame_bytes(quality=85)
            if not frame_bytes:
                await update.message.reply_text("\U0001f4f7 Could not capture frame")
                return

            # Analyze with Haiku
            result = await asyncio.to_thread(
                self._llm.analyze_frame,
                frame_bytes,
                system_prompt="Describe what you see concisely in 1-2 sentences.",
            )
            caption = (
                result.get("scene_description", "Workshop view") if result else "Workshop view"
            )
            await update.message.reply_photo(
                photo=BytesIO(frame_bytes), caption=caption[:1024]
            )
        except Exception as e:
            logger.error("Telegram /camera error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _cmd_watch(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            uid = update.effective_user.id
            self._audit("command", uid, "/watch")

            if not self._camera or not self._camera.is_open:
                await update.message.reply_text("\U0001f4f7 Camera is offline")
                return

            if uid in self._watch_tasks:
                await update.message.reply_text(
                    "Already watching. Send /stop to end."
                )
                return

            await update.message.reply_text(
                "\U0001f4f7 Starting workshop watch (30s intervals, 5 minutes). Send /stop to end."
            )
            task = asyncio.create_task(
                self._watch_loop(uid, update.effective_chat.id)
            )
            self._watch_tasks[uid] = task
        except Exception as e:
            logger.error("Telegram /watch error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _watch_loop(self, uid: int, chat_id: int):
        """Send a camera frame every 30s for 5 minutes (10 frames)."""
        try:
            for i in range(10):
                if not self._camera or not self._camera.is_open:
                    await self._application.bot.send_message(
                        chat_id=chat_id,
                        text="\U0001f4f7 Camera offline, stopping watch.",
                    )
                    return
                frame_bytes = self._camera.get_frame_bytes(quality=85)
                if not frame_bytes:
                    await self._application.bot.send_message(
                        chat_id=chat_id,
                        text="\U0001f4f7 Could not capture frame, stopping watch.",
                    )
                    return
                result = await asyncio.to_thread(
                    self._llm.analyze_frame, frame_bytes
                )
                caption = (
                    result.get("scene_description", f"Frame {i + 1}/10")
                    if result
                    else f"Frame {i + 1}/10"
                )
                await self._application.bot.send_photo(
                    chat_id=chat_id,
                    photo=BytesIO(frame_bytes),
                    caption=caption[:1024],
                )
                if i < 9:  # Don't sleep after the last frame
                    await asyncio.sleep(30)
        except asyncio.CancelledError:
            logger.info("[telegram] Watch cancelled for user %d", uid)
        except Exception as e:
            logger.error("[telegram] Watch loop error for user %d: %s", uid, e)
        finally:
            self._watch_tasks.pop(uid, None)
            try:
                await self._application.bot.send_message(
                    chat_id=chat_id, text="\U0001f4f7 Watch ended."
                )
            except Exception:
                pass

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            uid = update.effective_user.id
            self._audit("command", uid, "/stop")
            task = self._watch_tasks.pop(uid, None)
            if task:
                task.cancel()
                await update.message.reply_text("Watch stopped.")
            else:
                await update.message.reply_text("No active watch to stop.")
        except Exception as e:
            logger.error("Telegram /stop error: %s", e, exc_info=True)
            await self._send_error(update)

    # ── Session Commands ─────────────────────────────────────

    async def _cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            uid = update.effective_user.id
            self._audit("command", uid, "/clear")
            with self._history_lock:
                self._conversation_history.pop(uid, None)
                self._history_store.update(
                    "conversations",
                    {str(k): v for k, v in self._conversation_history.items()},
                )
            await update.message.reply_text("Conversation history cleared.")
        except Exception as e:
            logger.error("Telegram /clear error: %s", e, exc_info=True)
            await self._send_error(update)

    # ── Security Commands (admin only) ────────────────────────

    async def _cmd_block(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            uid = update.effective_user.id
            from config import TELEGRAM_ADMIN_USER

            if uid != TELEGRAM_ADMIN_USER:
                await update.message.reply_text("Admin only.")
                return
            if not context.args or not context.args[0].isdigit():
                await update.message.reply_text("Usage: /block <user_id>")
                return
            target = int(context.args[0])
            self._blocked_users.add(target)
            self._history_store.update("blocked_users", list(self._blocked_users))
            self._audit("block", uid, f"target={target}")
            await update.message.reply_text(f"Blocked user {target}")
        except Exception as e:
            logger.error("Telegram /block error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _cmd_unblock(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            uid = update.effective_user.id
            from config import TELEGRAM_ADMIN_USER

            if uid != TELEGRAM_ADMIN_USER:
                await update.message.reply_text("Admin only.")
                return
            if not context.args or not context.args[0].isdigit():
                await update.message.reply_text("Usage: /unblock <user_id>")
                return
            target = int(context.args[0])
            self._blocked_users.discard(target)
            self._history_store.update("blocked_users", list(self._blocked_users))
            self._audit("unblock", uid, f"target={target}")
            await update.message.reply_text(f"Unblocked user {target}")
        except Exception as e:
            logger.error("Telegram /unblock error: %s", e, exc_info=True)
            await self._send_error(update)

    # ── Message Handlers ─────────────────────────────────────

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            if not await self._check_rate_limit(update):
                return

            text = _sanitize_input(update.message.text)
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            lang = self._detect_language(text)
            self._audit("text_message", user_id, f"len={len(text)}")

            # Check for follow-up from inline button
            if user_id in self._awaiting_followup:
                task_id = self._awaiting_followup.pop(user_id)
                with self._task_results_lock:
                    cached = self._task_results.get(task_id)
                prev_context = ""
                if cached:
                    prev_context = (
                        f"Previous task: {cached['query']}\n"
                        f"Result: {cached['result'][:500]}"
                    )
                followup_task = f"{text}\n\nContext from previous agent task:\n{prev_context}"
                await self._spawn_agent_with_progress(followup_task, chat_id, user_id, text, lang)
                return

            # Check if user has active agents — queue if so
            with self._queue_lock:
                active = self._user_active_agents.get(user_id, 0)
            if active > 0:
                self._enqueue_message(user_id, ("text", text, user_id, chat_id, lang))
                await update.message.reply_text(
                    "\u23f3 Queued \u2014 I'll process this after the current task."
                )
                return

            # Send typing indicator
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

            await self._process_message(text, user_id, chat_id, lang, update)
        except Exception as e:
            logger.error("Telegram text handler error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not await self._check_user(update):
                return
            if not await self._check_rate_limit(update):
                return

            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            caption = update.message.caption or "What do you see in this image?"
            lang = self._detect_language(caption)
            self._audit("photo_message", user_id)

            # Download the largest available photo
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            buf = BytesIO()
            await file.download_to_memory(buf)
            frame_bytes = buf.getvalue()

            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

            await self._process_message(
                caption, user_id, chat_id, lang, update, frame_bytes=frame_bytes
            )
        except Exception as e:
            logger.error("Telegram photo handler error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming voice messages and audio files."""
        try:
            if not await self._check_user(update):
                return
            if not await self._check_rate_limit(update):
                return

            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            self._audit("voice_message", user_id)

            if not self._stt_provider:
                await update.message.reply_text(
                    "Voice transcription is not available (STT provider not configured)."
                )
                return

            # Download voice file (OGG/Opus or other format)
            voice = update.message.voice or update.message.audio
            if not voice:
                await update.message.reply_text("Could not read audio.")
                return
            file = await voice.get_file()
            audio_bytes = await file.download_as_bytearray()

            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

            # Convert to float32 PCM at 16kHz via ffmpeg
            audio_array = await asyncio.to_thread(
                self._convert_voice_to_array, bytes(audio_bytes)
            )
            if audio_array is None:
                await update.message.reply_text(
                    "Could not process audio. Is ffmpeg installed?"
                )
                return

            # Transcribe with existing STT provider
            result = await asyncio.to_thread(
                self._stt_provider.transcribe, audio_array, 16000
            )
            if not result or not result.text.strip():
                await update.message.reply_text(
                    "Could not transcribe the voice message."
                )
                return

            text = _sanitize_input(result.text.strip())
            lang = getattr(result, "language", None) or self._detect_language(text)
            await update.message.reply_text(f"\U0001f3a4 \"{text}\"")

            # Process as text through the normal routing pipeline
            await self._process_message(text, user_id, chat_id, lang, update)

            # Optionally send voice response
            from config import TELEGRAM_VOICE_RESPONSES

            if TELEGRAM_VOICE_RESPONSES:
                # Get the last response we sent for this user
                history = self._build_history(user_id)
                if history and history[-1]["role"] == "assistant":
                    response_text = history[-1]["content"]
                    await self._send_voice_response(chat_id, response_text)
        except Exception as e:
            logger.error("Telegram voice handler error: %s", e, exc_info=True)
            await self._send_error(update)

    async def _handle_edit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Log edited messages. Full edit-swap would require per-message state tracking."""
        if update.edited_message and update.edited_message.text:
            logger.info(
                "[telegram] User %d edited message: %s",
                update.effective_user.id,
                update.edited_message.text[:100],
            )

    # ── Inline Keyboard Callback ─────────────────────────────

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        user_id = query.from_user.id
        if user_id not in self._allowed_users:
            return

        data = query.data

        if data.startswith("followup:"):
            task_id = data.split(":", 1)[1]
            self._awaiting_followup[user_id] = task_id
            await query.edit_message_reply_markup(reply_markup=None)
            await self._application.bot.send_message(
                chat_id=query.message.chat_id,
                text="What would you like to follow up on?",
            )

        elif data.startswith("savenote:"):
            task_id = data.split(":", 1)[1]
            with self._task_results_lock:
                cached = self._task_results.get(task_id)
            if cached:
                note_text = cached["result"][:200].strip()
                note = {
                    "id": str(uuid.uuid4()),
                    "text": f"[Agent] {note_text}",
                    "created_at": datetime.now().isoformat(),
                    "source": "telegram-agent",
                    "done": False,
                }
                self._notes_store.append_to_list("notes", note, max_items=100)
                self._state.add_note(note)
                await query.edit_message_reply_markup(reply_markup=None)
                await self._application.bot.send_message(
                    chat_id=query.message.chat_id,
                    text=f"Saved as note: {note_text}",
                )
            else:
                await self._application.bot.send_message(
                    chat_id=query.message.chat_id,
                    text="Task result expired. Cannot save.",
                )

    # ── Core Processing ──────────────────────────────────────

    async def _process_message(
        self,
        text: str,
        user_id: int,
        chat_id: int,
        lang: str,
        update: Update,
        frame_bytes: Optional[bytes] = None,
    ):
        """Route a message through ClaudeClient and handle the result."""
        mem_context = await asyncio.to_thread(
            self._memory.assemble_context, query=text, purpose="conversation"
        )
        history = self._build_history(user_id)

        action, data = await asyncio.to_thread(
            self._llm.process_user_input,
            text=text,
            frame_bytes=frame_bytes,
            context=mem_context,
            language=lang,
            conversation_history=history,
        )

        if action == "conversation":
            response = data.get("text", "")
            if response:
                await self._send_formatted_message(update, response)
                self._store_conversation(user_id, text, response)
        elif action == "agent":
            task = data.get("task", text)
            await self._spawn_agent_with_progress(task, chat_id, user_id, text, lang)
        elif action == "note":
            content = data.get("content", text)
            note = {
                "id": str(uuid.uuid4()),
                "text": content.strip(),
                "created_at": datetime.now().isoformat(),
                "source": "telegram",
                "done": False,
            }
            self._notes_store.append_to_list("notes", note, max_items=100)
            self._state.add_note(note)
            await update.message.reply_text(f"Noted: {content.strip()}")
            self._store_conversation(user_id, text, f"Noted: {content.strip()}")
        elif action == "shutdown":
            await update.message.reply_text("Going offline.")
            self._store_conversation(user_id, text, "Going offline.")

    def _store_conversation(self, user_id: int, user_text: str, response: str):
        """Store conversation in shared memory, dashboard state, and per-user history."""
        self._append_history(user_id, "user", user_text)
        self._append_history(user_id, "assistant", response)
        self._state.add_conversation("user", user_text)
        self._state.add_conversation("winston", response)
        conversation_text = f"Q: {user_text}\nA: {response}"
        try:
            self._memory.store(conversation_text, entry_type="conversation")
        except Exception as e:
            logger.error("Failed to store conversation in memory: %s", e)
        self._state.set_cost(self._cost_tracker.get_daily_cost())

    # ── Voice Note Helpers ────────────────────────────────────

    @staticmethod
    def _convert_voice_to_array(audio_bytes: bytes) -> Optional[np.ndarray]:
        """Convert audio bytes (OGG/Opus/etc.) to float32 numpy array at 16kHz via ffmpeg."""
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-i", "pipe:0",
                    "-f", "s16le", "-ar", "16000", "-ac", "1",
                    "pipe:1",
                ],
                input=audio_bytes,
                capture_output=True,
                timeout=10,
            )
            if proc.returncode != 0:
                logger.error(
                    "[telegram] ffmpeg conversion failed: %s", proc.stderr[:200]
                )
                return None
            pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            return pcm
        except FileNotFoundError:
            logger.error("[telegram] ffmpeg not found — install it for voice support")
            return None
        except subprocess.TimeoutExpired:
            logger.error("[telegram] ffmpeg conversion timed out")
            return None

    async def _send_voice_response(self, chat_id: int, text: str) -> bool:
        """Generate TTS audio and send as Telegram voice message."""
        pcm_bytes = await asyncio.to_thread(self._generate_tts_bytes, text)
        if not pcm_bytes:
            return False
        ogg_bytes = await asyncio.to_thread(self._convert_pcm_to_ogg, pcm_bytes)
        if ogg_bytes:
            buf = BytesIO(ogg_bytes)
            buf.name = "response.ogg"
            await self._application.bot.send_voice(chat_id=chat_id, voice=buf)
            return True
        return False

    def _generate_tts_bytes(self, text: str) -> Optional[bytes]:
        """Get raw PCM bytes from ElevenLabs without playing."""
        if not self._tts_engine:
            return None
        backend = getattr(self._tts_engine, "_primary", None)
        if not backend or not getattr(backend, "_client", None):
            return None
        try:
            from elevenlabs import VoiceSettings

            stream = backend._client.text_to_speech.stream(
                text=text,
                voice_id=backend._voice_id,
                model_id=backend._model_id,
                output_format=backend._output_format,
                voice_settings=VoiceSettings(
                    stability=backend._stability,
                    similarity_boost=backend._similarity,
                    style=backend._style,
                    use_speaker_boost=True,
                ),
            )
            chunks = [c for c in stream if isinstance(c, bytes) and len(c) > 0]
            return b"".join(chunks) if chunks else None
        except Exception as e:
            logger.error("[telegram] TTS generation failed: %s", e)
            return None

    @staticmethod
    def _convert_pcm_to_ogg(pcm_bytes: bytes) -> Optional[bytes]:
        """Convert raw PCM S16LE 24kHz mono to OGG/Opus via ffmpeg."""
        try:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
                    "-c:a", "libopus", "-b:a", "64k", "-f", "ogg", "pipe:1",
                ],
                input=pcm_bytes,
                capture_output=True,
                timeout=15,
            )
            return proc.stdout if proc.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    # ── Agent Handling with Progress ─────────────────────────

    async def _spawn_agent_with_progress(
        self, task: str, chat_id: int, user_id: int, user_text: str, lang: str
    ):
        """Spawn agent with live progress message editing."""
        # Check agent-specific rate limit
        if not self._agent_rate_limiter.check(user_id):
            await self._application.bot.send_message(
                chat_id=chat_id,
                text="Agent task limit reached (10/hour). Try again later.",
            )
            return

        if not self._telegram_agent_sem.acquire(blocking=False):
            # At capacity — queue the request
            self._enqueue_message(
                user_id, ("agent", task, chat_id, user_id, user_text, lang)
            )
            await self._application.bot.send_message(
                chat_id=chat_id,
                text="\u23f3 I'm at capacity (2 tasks running). Your request is queued.",
            )
            return

        # Send initial progress message
        msg = await self._application.bot.send_message(
            chat_id=chat_id,
            text="\u23f3 Investigating...",
        )

        task_id = str(uuid.uuid4())
        logger.info("[telegram-agent] Spawning task %s: '%s'", task_id[:8], task)
        self._audit("agent_spawn", user_id, f"task_id={task_id[:8]}")

        # Store progress message info
        self._progress_messages[task_id] = {
            "chat_id": chat_id,
            "message_id": msg.message_id,
            "last_edit": time.time(),
            "lines": ["\u23f3 Investigating..."],
        }

        # Track user active agents
        with self._queue_lock:
            self._user_active_agents[user_id] = (
                self._user_active_agents.get(user_id, 0) + 1
            )

        # Create persistent task record
        task_record = {
            "id": task_id,
            "query": task,
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "result": None,
            "reported": False,
        }
        self._agent_store.append_to_list("tasks", task_record, max_items=50)
        self._state.add_agent_task(task_record)
        self._state.add_conversation("user", user_text)

        context = self._memory.assemble_context(query=task, purpose="conversation")

        threading.Thread(
            target=self._run_agent_with_progress,
            args=(task_id, task, context, chat_id, user_id, lang),
            daemon=True,
            name=f"telegram-agent-{task_id[:8]}",
        ).start()

    def _run_agent_with_progress(
        self,
        task_id: str,
        task: str,
        context: str,
        chat_id: int,
        user_id: int,
        lang: str,
    ):
        """Background thread: run AgentExecutor with progress callbacks."""
        try:
            from brain.agent_executor import AgentExecutor
            from brain.agent_tools import create_default_registry
            from config import (
                COMPUTER_USE_DISPLAY_HEIGHT,
                COMPUTER_USE_DISPLAY_WIDTH,
                COMPUTER_USE_ENABLED,
                TELEGRAM_PROGRESS_DEBOUNCE,
            )

            registry = create_default_registry()
            computer = None
            if COMPUTER_USE_ENABLED:
                from brain.computer_use import MacOSComputerController

                computer = MacOSComputerController(
                    display_width=COMPUTER_USE_DISPLAY_WIDTH,
                    display_height=COMPUTER_USE_DISPLAY_HEIGHT,
                )

            executor = AgentExecutor(
                self._llm.client,
                self._cost_tracker,
                registry,
                computer_controller=computer,
            )

            full_task = task
            if context:
                full_task = f"{task}\n\nContext from memory:\n{context}"

            # Progress callback with debouncing
            def on_progress(msg: str):
                info = self._progress_messages.get(task_id)
                if not info or not self._loop:
                    return
                emoji = _tool_emoji(msg.replace("Using ", "").rstrip("..."))
                info["lines"].append(f"{emoji} {msg}")
                now = time.time()
                if now - info["last_edit"] < TELEGRAM_PROGRESS_DEBOUNCE:
                    return  # debounce — skip edit
                info["last_edit"] = now
                text = "\n".join(info["lines"])
                asyncio.run_coroutine_threadsafe(
                    self._safe_edit_message(info["chat_id"], info["message_id"], text),
                    self._loop,
                )

            # Screenshot callback
            def on_screenshot(img_bytes: bytes):
                if not self._loop or not self._application:
                    return
                buf = BytesIO(img_bytes)
                buf.name = "screenshot.png"
                asyncio.run_coroutine_threadsafe(
                    self._application.bot.send_photo(
                        chat_id=chat_id,
                        photo=buf,
                        caption="\U0001f5a5\ufe0f Screenshot",
                    ),
                    self._loop,
                )

            findings = executor.run(
                full_task, on_progress=on_progress, on_screenshot=on_screenshot
            )

            # Update task store
            result = findings or "No conclusive findings."
            self._agent_store.update_in_list(
                "tasks",
                task_id,
                {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "result": result,
                    "reported": True,
                },
            )
            self._state.update_agent_task(task_id, "completed", result)

            if findings:
                self._state.add_conversation("agent", findings[:500])
                self._memory.store(
                    f"[Agent Investigation]\nQ: {task}\nFindings: {findings}",
                    entry_type="conversation",
                )
                self._append_history(user_id, "assistant", findings[:500])

            # Cache result for inline buttons
            with self._task_results_lock:
                self._task_results[task_id] = {
                    "result": result,
                    "timestamp": time.time(),
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "query": task,
                }

            # Send final result with inline keyboard
            if self._loop and self._application:
                asyncio.run_coroutine_threadsafe(
                    self._send_agent_result_with_buttons(task_id, chat_id, result),
                    self._loop,
                )

            self._state.set_cost(self._cost_tracker.get_daily_cost())

        except Exception as e:
            logger.error("[telegram-agent] Task %s failed: %s", task_id[:8], e, exc_info=True)
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
            self._state.update_agent_task(task_id, "failed", f"Error: {e}")
            if self._loop and self._application:
                info = self._progress_messages.get(task_id, {})
                asyncio.run_coroutine_threadsafe(
                    self._safe_edit_message(
                        info.get("chat_id", chat_id),
                        info.get("message_id"),
                        "\u274c Task failed. Please try again.",
                    ),
                    self._loop,
                )
        finally:
            self._telegram_agent_sem.release()
            self._progress_messages.pop(task_id, None)
            # Decrement user active agents
            with self._queue_lock:
                count = self._user_active_agents.get(user_id, 1) - 1
                if count <= 0:
                    self._user_active_agents.pop(user_id, None)
                else:
                    self._user_active_agents[user_id] = count
            # Drain queued messages
            if self._loop:
                asyncio.run_coroutine_threadsafe(self._drain_queue(user_id), self._loop)
            # Clean up expired task results
            self._cleanup_task_cache()

    async def _send_agent_result_with_buttons(
        self, task_id: str, chat_id: int, findings: str
    ):
        """Edit progress message with final result + inline keyboard buttons."""
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "\U0001f504 Follow up", callback_data=f"followup:{task_id}"
                    ),
                    InlineKeyboardButton(
                        "\U0001f4cb Save as note", callback_data=f"savenote:{task_id}"
                    ),
                ]
            ]
        )

        info = self._progress_messages.get(task_id)
        header = "\U0001f4cb Agent Results:\n\n"
        text = header + findings

        if info and info.get("message_id"):
            try:
                if len(text) <= 4096:
                    await self._application.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=info["message_id"],
                        text=text,
                        reply_markup=keyboard,
                    )
                else:
                    # Edit progress message with first chunk
                    await self._application.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=info["message_id"],
                        text=text[:4096],
                    )
                    # Send remaining chunks
                    remaining = text[4096:]
                    for chunk in _split_message(remaining):
                        await self._application.bot.send_message(
                            chat_id=chat_id, text=chunk
                        )
                    # Send buttons as final message
                    await self._application.bot.send_message(
                        chat_id=chat_id,
                        text="Task complete.",
                        reply_markup=keyboard,
                    )
            except Exception as e:
                logger.error("Failed to edit agent result: %s", e)
                # Fallback: send as new messages
                for chunk in _split_message(text):
                    await self._application.bot.send_message(
                        chat_id=chat_id, text=chunk
                    )
        else:
            # No progress message — send as new
            for chunk in _split_message(text):
                await self._application.bot.send_message(chat_id=chat_id, text=chunk)

    async def _safe_edit_message(
        self, chat_id: int, message_id: Optional[int], text: str
    ):
        """Edit a message, ignoring errors (deleted, not modified, etc.)."""
        if not message_id:
            return
        try:
            await self._application.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text[:4096],
            )
        except Exception as e:
            logger.debug("Failed to edit message %s: %s", message_id, e)

    # ── Message Queueing ─────────────────────────────────────

    def _enqueue_message(self, user_id: int, message_data: tuple):
        """Add a message to the user's queue."""
        from config import TELEGRAM_QUEUE_MAX_SIZE

        with self._queue_lock:
            if user_id not in self._message_queues:
                self._message_queues[user_id] = asyncio.Queue(
                    maxsize=TELEGRAM_QUEUE_MAX_SIZE
                )
        q = self._message_queues[user_id]
        try:
            q.put_nowait(message_data)
        except asyncio.QueueFull:
            logger.warning("Queue full for user %d, dropping message", user_id)

    async def _drain_queue(self, user_id: int):
        """Process all queued messages for a user after agent completes."""
        q = self._message_queues.get(user_id)
        if not q:
            return
        while not q.empty():
            try:
                item = q.get_nowait()
                msg_type = item[0]
                if msg_type == "agent":
                    # (type, task, chat_id, user_id, user_text, lang)
                    _, task, chat_id, uid, user_text, lang = item
                    await self._spawn_agent_with_progress(
                        task, chat_id, uid, user_text, lang
                    )
                elif msg_type == "text":
                    # (type, text, user_id, chat_id, lang)
                    _, text, uid, chat_id, lang = item
                    # Create a minimal Update-less processing path
                    await self._process_message_headless(text, uid, chat_id, lang)
            except Exception as e:
                logger.error("Error draining queue for user %d: %s", user_id, e)

    async def _process_message_headless(
        self, text: str, user_id: int, chat_id: int, lang: str
    ):
        """Process a queued message without an Update object."""
        mem_context = await asyncio.to_thread(
            self._memory.assemble_context, query=text, purpose="conversation"
        )
        history = self._build_history(user_id)

        action, data = await asyncio.to_thread(
            self._llm.process_user_input,
            text=text,
            context=mem_context,
            language=lang,
            conversation_history=history,
        )

        if action == "conversation":
            response = data.get("text", "")
            if response:
                for chunk in _split_message(response):
                    await self._application.bot.send_message(
                        chat_id=chat_id, text=chunk
                    )
                self._store_conversation(user_id, text, response)
        elif action == "agent":
            task = data.get("task", text)
            await self._spawn_agent_with_progress(task, chat_id, user_id, text, lang)
        elif action == "note":
            content = data.get("content", text)
            note = {
                "id": str(uuid.uuid4()),
                "text": content.strip(),
                "created_at": datetime.now().isoformat(),
                "source": "telegram",
                "done": False,
            }
            self._notes_store.append_to_list("notes", note, max_items=100)
            self._state.add_note(note)
            await self._application.bot.send_message(
                chat_id=chat_id, text=f"Noted: {content.strip()}"
            )
            self._store_conversation(user_id, text, f"Noted: {content.strip()}")

    # ── Rich Message Formatting ──────────────────────────────

    async def _send_formatted_message(self, update: Update, text: str):
        """Send with MarkdownV2 formatting, falling back to plain text."""
        from config import TELEGRAM_LONG_MESSAGE_THRESHOLD

        if len(text) > TELEGRAM_LONG_MESSAGE_THRESHOLD:
            # Send as document
            buf = BytesIO(text.encode("utf-8"))
            buf.name = "response.txt"
            caption = text[:200] + "..." if len(text) > 200 else text
            await update.message.reply_document(document=buf, caption=caption)
            return

        try:
            formatted = _format_as_markdown_v2(text)
            for chunk in _split_message(formatted):
                await update.message.reply_text(
                    chunk, parse_mode=ParseMode.MARKDOWN_V2
                )
        except Exception:
            # Fallback to plain text on MarkdownV2 parse failure
            for chunk in _split_message(text):
                await update.message.reply_text(chunk)

    # ── Utilities ────────────────────────────────────────────

    async def _send_long_message(self, update: Update, text: str):
        """Reply with text, splitting into chunks if it exceeds Telegram's limit."""
        for chunk in _split_message(text):
            await update.message.reply_text(chunk)

    async def _send_error(self, update: Update):
        try:
            msg = update.message or (
                update.callback_query and update.callback_query.message
            )
            if msg:
                await msg.reply_text(
                    "Sorry, something went wrong processing your message. Please try again."
                )
        except Exception:
            pass

    def _cleanup_task_cache(self):
        """Remove task results older than 1 hour."""
        cutoff = time.time() - 3600
        with self._task_results_lock:
            expired = [k for k, v in self._task_results.items() if v["timestamp"] < cutoff]
            for k in expired:
                del self._task_results[k]


class TelegramNotifier:
    """Sends proactive alerts to all allowed Telegram users."""

    def __init__(self, bot: TelegramBot):
        self._bot = bot

    def notify(self, message: str, frame_bytes: Optional[bytes] = None):
        """Thread-safe: schedule async send on the bot's event loop."""
        if not self._bot._loop or not self._bot._application:
            return
        asyncio.run_coroutine_threadsafe(
            self._send_to_all(message, frame_bytes),
            self._bot._loop,
        )

    async def _send_to_all(self, message: str, frame_bytes: Optional[bytes] = None):
        bot_api = self._bot._application.bot
        for uid in self._bot._allowed_users:
            try:
                if frame_bytes:
                    await bot_api.send_photo(
                        chat_id=uid,
                        photo=BytesIO(frame_bytes),
                        caption=message[:1024],
                    )
                else:
                    await bot_api.send_message(chat_id=uid, text=message)
            except Exception as e:
                logger.error(
                    "[telegram] Failed to send proactive notification to %d: %s",
                    uid, e,
                )
