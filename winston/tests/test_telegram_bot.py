"""Tests for Telegram bot: auth, rate limiting, command parsing, formatting, persistence."""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Async test helper (avoids pytest-asyncio dependency) ─────────

def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Module-level helpers ──────────────────────────────────────────


def _make_bot(tmp_path, allowed_users=None, camera=None, stt_provider=None, tts_engine=None):
    """Create a TelegramBot with mocked dependencies."""
    from telegram_bot import TelegramBot

    if allowed_users is None:
        allowed_users = {111, 222}

    bot = TelegramBot(
        token="test-token",
        allowed_users=allowed_users,
        llm=MagicMock(),
        memory=MagicMock(),
        cost_tracker=MagicMock(),
        notes_store=MagicMock(),
        agent_store=MagicMock(),
        state=MagicMock(),
        agent_lock=threading.Lock(),
        camera=camera,
        stt_provider=stt_provider,
        tts_engine=tts_engine,
    )
    return bot


def _make_update(user_id=111, text="hello", is_command=False):
    """Create a mock Telegram Update object."""
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat.id = 9999
    update.message.text = text
    update.message.reply_text = AsyncMock()
    update.message.reply_photo = AsyncMock()
    update.message.reply_document = AsyncMock()
    update.message.voice = None
    update.message.audio = None
    update.message.photo = None
    update.callback_query = None
    return update


# ── Input Sanitization ───────────────────────────────────────────


class TestInputSanitization:
    def test_strip_control_characters(self):
        from telegram_bot import _sanitize_input

        dirty = "hello\x00world\x07test\x1fend"
        assert _sanitize_input(dirty) == "helloworldtestend"

    def test_preserves_newlines_and_tabs(self):
        from telegram_bot import _sanitize_input

        text = "hello\nworld\ttab"
        assert _sanitize_input(text) == "hello\nworld\ttab"

    def test_length_limit(self):
        from telegram_bot import _sanitize_input

        long_text = "a" * 5000
        result = _sanitize_input(long_text)
        assert len(result) == 4000

    def test_normal_text_unchanged(self):
        from telegram_bot import _sanitize_input

        text = "Hello, how are you? I'm fine."
        assert _sanitize_input(text) == text


# ── Auth & Security ──────────────────────────────────────────────


class TestAuthAndSecurity:
    @pytest.fixture
    def bot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.TELEGRAM_HISTORY_FILE", str(tmp_path / "hist.json"))
        return _make_bot(tmp_path, allowed_users={111, 222})

    def test_unauthorized_user_rejected(self, bot):
        update = _make_update(user_id=999)
        result = _run(bot._check_user(update))
        assert result is False
        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "not authorized" in call_text.lower()

    def test_authorized_user_passes(self, bot):
        update = _make_update(user_id=111)
        result = _run(bot._check_user(update))
        assert result is True

    def test_blocked_user_rejected(self, bot):
        bot._blocked_users.add(111)
        update = _make_update(user_id=111)
        result = _run(bot._check_user(update))
        assert result is False
        # Blocked users get silently rejected (no message)
        update.message.reply_text.assert_not_called()

    def test_admin_can_block(self, bot, monkeypatch):
        monkeypatch.setattr("config.TELEGRAM_ADMIN_USER", 111)
        update = _make_update(user_id=111, text="/block 333")
        context = MagicMock()
        context.args = ["333"]

        bot._application = MagicMock()
        _run(bot._cmd_block(update, context))

        assert 333 in bot._blocked_users
        update.message.reply_text.assert_called()
        call_text = update.message.reply_text.call_args[0][0]
        assert "Blocked" in call_text

    def test_non_admin_cannot_block(self, bot, monkeypatch):
        monkeypatch.setattr("config.TELEGRAM_ADMIN_USER", 111)
        update = _make_update(user_id=222, text="/block 333")
        context = MagicMock()
        context.args = ["333"]

        _run(bot._cmd_block(update, context))

        assert 333 not in bot._blocked_users
        update.message.reply_text.assert_called()
        call_text = update.message.reply_text.call_args[0][0]
        assert "Admin only" in call_text


# ── Rate Limiting ────────────────────────────────────────────────


class TestRateLimiting:
    def test_message_rate_limit(self):
        from telegram_bot import _RateLimiter

        limiter = _RateLimiter(max_per_hour=3)
        assert limiter.check(1) is True
        assert limiter.check(1) is True
        assert limiter.check(1) is True
        assert limiter.check(1) is False  # 4th should fail

    def test_rate_limit_per_user(self):
        from telegram_bot import _RateLimiter

        limiter = _RateLimiter(max_per_hour=2)
        assert limiter.check(1) is True
        assert limiter.check(1) is True
        assert limiter.check(1) is False  # User 1 exhausted
        assert limiter.check(2) is True   # User 2 still has quota

    def test_rate_limit_sliding_window(self):
        from telegram_bot import _RateLimiter

        limiter = _RateLimiter(max_per_hour=1)
        # Manually inject an old timestamp
        limiter._timestamps[1] = [time.time() - 3700]  # Older than 1 hour
        assert limiter.check(1) is True  # Old one expired, new one allowed

    def test_remaining_count(self):
        from telegram_bot import _RateLimiter

        limiter = _RateLimiter(max_per_hour=5)
        assert limiter.remaining(1) == 5
        limiter.check(1)
        limiter.check(1)
        assert limiter.remaining(1) == 3

    def test_agent_rate_limit_separate(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.TELEGRAM_HISTORY_FILE", str(tmp_path / "hist.json"))
        bot = _make_bot(tmp_path)
        # Message limiter and agent limiter are independent
        assert bot._rate_limiter._max == 30
        assert bot._agent_rate_limiter._max == 10


# ── Message Formatting ───────────────────────────────────────────


class TestMessageFormatting:
    def test_escape_markdown_v2(self):
        from telegram_bot import _escape_markdown_v2

        text = "Hello *bold* and _italic_ (test)"
        escaped = _escape_markdown_v2(text)
        assert "\\*" in escaped
        assert "\\_" in escaped
        assert "\\(" in escaped
        assert "\\)" in escaped

    def test_split_long_message(self):
        from telegram_bot import _split_message

        # Short message should not be split
        short = "Hello world"
        assert _split_message(short) == [short]

        # Long message should be split
        long_text = "Line\n" * 2000  # ~10000 chars
        chunks = _split_message(long_text, max_len=4096)
        assert len(chunks) > 1
        assert all(len(c) <= 4096 for c in chunks)

    def test_split_message_preserves_content(self):
        from telegram_bot import _split_message

        text = "A" * 8000
        chunks = _split_message(text, max_len=4096)
        reassembled = "".join(chunks)
        assert reassembled == text


# ── Command Parsing ──────────────────────────────────────────────


class TestCommandParsing:
    @pytest.fixture
    def bot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.TELEGRAM_HISTORY_FILE", str(tmp_path / "hist.json"))
        return _make_bot(tmp_path)

    def test_camera_offline(self, bot):
        """Camera not available -> offline message."""
        update = _make_update(user_id=111)
        context = MagicMock()
        context.bot = MagicMock()
        context.bot.send_chat_action = AsyncMock()

        _run(bot._cmd_camera(update, context))

        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "offline" in call_text.lower()

    def test_camera_with_mock(self, bot):
        """Camera available -> sends photo with description."""
        mock_camera = MagicMock()
        mock_camera.is_open = True
        mock_camera.get_frame_bytes.return_value = b"\xff\xd8\xff\xe0JPEG"
        bot._camera = mock_camera
        bot._llm.analyze_frame.return_value = {"scene_description": "A workbench with tools"}

        update = _make_update(user_id=111)
        context = MagicMock()
        context.bot = MagicMock()
        context.bot.send_chat_action = AsyncMock()

        _run(bot._cmd_camera(update, context))

        update.message.reply_photo.assert_called_once()
        call_kwargs = update.message.reply_photo.call_args
        assert "workbench" in call_kwargs.kwargs.get("caption", call_kwargs[1].get("caption", "")).lower()

    def test_clear_resets_history(self, bot):
        """The /clear command empties per-user history."""
        bot._conversation_history[111] = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        update = _make_update(user_id=111)
        context = MagicMock()

        _run(bot._cmd_clear(update, context))

        assert 111 not in bot._conversation_history
        update.message.reply_text.assert_called_once()
        assert "cleared" in update.message.reply_text.call_args[0][0].lower()

    def test_stop_without_watch(self, bot):
        """/stop without active watch -> message."""
        update = _make_update(user_id=111)
        context = MagicMock()

        _run(bot._cmd_stop(update, context))

        update.message.reply_text.assert_called_once()
        assert "no active" in update.message.reply_text.call_args[0][0].lower()


# ── Session Persistence ──────────────────────────────────────────


class TestPersistence:
    @pytest.fixture
    def bot(self, tmp_path, monkeypatch):
        hist_file = str(tmp_path / "telegram_history.json")
        monkeypatch.setattr("config.TELEGRAM_HISTORY_FILE", hist_file)
        return _make_bot(tmp_path)

    def test_history_persisted_to_disk(self, bot):
        """Appending history writes to the persistent store."""
        bot._append_history(111, "user", "hello")
        bot._append_history(111, "assistant", "hi there")

        # Read directly from the store
        saved = bot._history_store.get("conversations", {})
        assert "111" in saved
        assert len(saved["111"]) == 2
        assert saved["111"][0]["role"] == "user"
        assert saved["111"][1]["role"] == "assistant"

    def test_history_loaded_on_init(self, tmp_path, monkeypatch):
        """Pre-populated store loads into memory on bot creation."""
        from utils.persistent_store import PersistentStore

        hist_file = str(tmp_path / "telegram_history.json")
        monkeypatch.setattr("config.TELEGRAM_HISTORY_FILE", hist_file)

        # Pre-populate the store
        store = PersistentStore(hist_file, default_data={"conversations": {}, "blocked_users": []})
        store.update("conversations", {
            "111": [
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": "previous answer"},
            ]
        })

        # Create bot — should load the history
        bot = _make_bot(tmp_path)
        assert 111 in bot._conversation_history
        assert len(bot._conversation_history[111]) == 2
        assert bot._conversation_history[111][0]["content"] == "previous question"

    def test_clear_removes_persisted(self, bot, monkeypatch):
        """The /clear command also clears the persistent store."""
        bot._append_history(111, "user", "hello")

        # Verify it was saved
        assert "111" in bot._history_store.get("conversations", {})

        # Clear
        with bot._history_lock:
            bot._conversation_history.pop(111, None)
            bot._history_store.update(
                "conversations",
                {str(k): v for k, v in bot._conversation_history.items()},
            )

        # Verify it was removed
        saved = bot._history_store.get("conversations", {})
        assert "111" not in saved

    def test_blocked_users_persisted(self, bot):
        """Blocked users survive through the persistent store."""
        bot._blocked_users.add(999)
        bot._history_store.update("blocked_users", list(bot._blocked_users))

        # Read back
        blocked = bot._history_store.get("blocked_users", [])
        assert 999 in blocked

    def test_history_max_20_on_load(self, tmp_path, monkeypatch):
        """Only last 20 messages loaded from persisted store."""
        from utils.persistent_store import PersistentStore

        hist_file = str(tmp_path / "telegram_history.json")
        monkeypatch.setattr("config.TELEGRAM_HISTORY_FILE", hist_file)

        # Pre-populate with 30 messages
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "msg %d" % i}
            for i in range(30)
        ]
        store = PersistentStore(hist_file, default_data={"conversations": {}, "blocked_users": []})
        store.update("conversations", {"111": messages})

        bot = _make_bot(tmp_path)
        assert len(bot._conversation_history[111]) == 20


# ── Voice Conversion ─────────────────────────────────────────────


class TestVoiceConversion:
    def test_convert_voice_to_array_ffmpeg_not_found(self):
        """Graceful failure when ffmpeg is not available."""
        from telegram_bot import TelegramBot

        with patch("telegram_bot.subprocess.run", side_effect=FileNotFoundError):
            result = TelegramBot._convert_voice_to_array(b"fake audio")
            assert result is None

    def test_convert_voice_to_array_bad_data(self):
        """Graceful failure when ffmpeg returns non-zero exit code."""
        from telegram_bot import TelegramBot

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = b"error decoding"

        with patch("telegram_bot.subprocess.run", return_value=mock_proc):
            result = TelegramBot._convert_voice_to_array(b"not audio")
            assert result is None

    def test_convert_pcm_to_ogg_success(self):
        """PCM -> OGG conversion returns bytes on success."""
        from telegram_bot import TelegramBot

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b"OggS\x00fake_ogg_data"

        with patch("telegram_bot.subprocess.run", return_value=mock_proc):
            result = TelegramBot._convert_pcm_to_ogg(b"\x00" * 48000)
            assert result is not None
            assert result.startswith(b"OggS")


# ── TelegramNotifier ─────────────────────────────────────────────


class TestTelegramNotifier:
    def test_notify_without_loop_is_noop(self, tmp_path, monkeypatch):
        """Calling notify when bot has no event loop does nothing (no crash)."""
        monkeypatch.setattr("config.TELEGRAM_HISTORY_FILE", str(tmp_path / "hist.json"))
        from telegram_bot import TelegramNotifier

        bot = _make_bot(tmp_path)
        bot._loop = None
        notifier = TelegramNotifier(bot)
        notifier.notify("test message")  # Should not raise

    def test_notify_schedules_coroutine(self, tmp_path, monkeypatch):
        """Calling notify with an active loop schedules the send coroutine."""
        monkeypatch.setattr("config.TELEGRAM_HISTORY_FILE", str(tmp_path / "hist.json"))
        from telegram_bot import TelegramNotifier

        bot = _make_bot(tmp_path)
        bot._loop = MagicMock()
        bot._application = MagicMock()
        notifier = TelegramNotifier(bot)

        with patch("telegram_bot.asyncio.run_coroutine_threadsafe") as mock_schedule:
            notifier.notify("proactive alert", frame_bytes=b"jpeg_data")
            mock_schedule.assert_called_once()
