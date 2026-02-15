"""Tests for STT provider abstraction."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure winston/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.stt import (
    FallbackSTTProvider,
    GroqWhisperProvider,
    LocalWhisperProvider,
    STTResult,
    _audio_to_wav,
    create_stt_provider,
)

# ── Unit helpers ──────────────────────────────────────────────────────


def _silent_audio(duration_s: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate near-silent float32 audio."""
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


def _make_result(text: str = "hello", lang: str = "en") -> STTResult:
    return STTResult(text=text, language=lang, confidence=0.0, duration_ms=50.0)


# ── _audio_to_wav ────────────────────────────────────────────────────


class TestAudioToWav:
    def test_produces_valid_wav_header(self):
        audio = np.zeros(16000, dtype=np.float32)  # 1 second
        wav = _audio_to_wav(audio, 16000)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        assert wav[36:40] == b"data"

    def test_correct_byte_length(self):
        samples = 8000
        audio = np.zeros(samples, dtype=np.float32)
        wav = _audio_to_wav(audio, 16000)
        # 44 header bytes + 2 bytes per sample
        assert len(wav) == 44 + samples * 2


# ── GroqWhisperProvider ──────────────────────────────────────────────


class TestGroqWhisperProvider:
    @patch("perception.stt.Groq", create=True)
    def test_transcribe_success(self, mock_groq_cls):
        """Mocked Groq client returns a valid response."""
        # Set up mock
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "  open the gcode  "
        mock_response.language = "en"
        mock_client.audio.transcriptions.create.return_value = mock_response

        # Patch the import inside GroqWhisperProvider.__init__
        with patch.dict("sys.modules", {"groq": MagicMock(Groq=mock_groq_cls)}):
            provider = GroqWhisperProvider(api_key="test-key", model="whisper-large-v3-turbo")

        provider._client = mock_client  # Ensure our mock is used

        audio = _silent_audio(1.0)
        result = provider.transcribe(audio)

        assert result is not None
        assert result.text == "open the gcode"
        assert result.language == "en"
        assert result.duration_ms >= 0
        mock_client.audio.transcriptions.create.assert_called_once()

    @patch("perception.stt.Groq", create=True)
    def test_transcribe_error_returns_none(self, mock_groq_cls):
        """Groq API error should return None (not crash)."""
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = Exception("rate limit")

        with patch.dict("sys.modules", {"groq": MagicMock(Groq=mock_groq_cls)}):
            provider = GroqWhisperProvider(api_key="test-key")

        provider._client = mock_client

        result = provider.transcribe(_silent_audio())
        assert result is None

    @patch("perception.stt.Groq", create=True)
    def test_transcribe_empty_text(self, mock_groq_cls):
        """Empty response text should return STTResult with empty string."""
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.language = "en"
        mock_client.audio.transcriptions.create.return_value = mock_response

        with patch.dict("sys.modules", {"groq": MagicMock(Groq=mock_groq_cls)}):
            provider = GroqWhisperProvider(api_key="test-key")

        provider._client = mock_client

        result = provider.transcribe(_silent_audio())
        assert result is not None
        assert result.text == ""


# ── LocalWhisperProvider ─────────────────────────────────────────────


class TestLocalWhisperProvider:
    def test_lazy_init_no_model_on_creation(self):
        """Model should not load in __init__."""
        provider = LocalWhisperProvider(model_name="tiny")
        assert provider._model is None

    @patch("perception.stt.WhisperModel", create=True)
    def test_transcribe_with_mocked_model(self, _mock_cls):
        """Mocked faster-whisper returns segments."""
        provider = LocalWhisperProvider(model_name="medium")

        # Create mock model
        mock_model = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = "hello world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        # Inject mock model (skip lazy loading)
        provider._model = mock_model

        result = provider.transcribe(_silent_audio())
        assert result is not None
        assert result.text == "hello world"
        assert result.language == "en"
        mock_model.transcribe.assert_called_once()

    def test_transcribe_returns_none_on_model_load_failure(self):
        """If the model can't load, transcribe returns None."""
        provider = LocalWhisperProvider(model_name="nonexistent-model")
        # _ensure_model will try to import faster_whisper — may fail if not installed
        # Either way, a failure path should return None
        with patch("perception.stt.WhisperModel", create=True, side_effect=Exception("no model")):
            # Force fresh load attempt
            provider._model = None
            result = provider.transcribe(_silent_audio())
            assert result is None


# ── FallbackSTTProvider ──────────────────────────────────────────────


class TestFallbackSTTProvider:
    def test_uses_primary_on_success(self):
        """When primary succeeds, fallback is never called."""
        primary = MagicMock()
        fallback = MagicMock()
        primary.transcribe.return_value = _make_result("primary text")

        provider = FallbackSTTProvider(primary, fallback)
        result = provider.transcribe(_silent_audio())

        assert result.text == "primary text"
        primary.transcribe.assert_called_once()
        fallback.transcribe.assert_not_called()

    def test_uses_fallback_on_primary_failure(self):
        """When primary returns None, fallback is tried."""
        primary = MagicMock()
        fallback = MagicMock()
        primary.transcribe.return_value = None
        fallback.transcribe.return_value = _make_result("fallback text")

        provider = FallbackSTTProvider(primary, fallback)
        result = provider.transcribe(_silent_audio())

        assert result.text == "fallback text"
        primary.transcribe.assert_called_once()
        fallback.transcribe.assert_called_once()

    def test_returns_none_when_both_fail(self):
        """When both providers fail, returns None."""
        primary = MagicMock()
        fallback = MagicMock()
        primary.transcribe.return_value = None
        fallback.transcribe.return_value = None

        provider = FallbackSTTProvider(primary, fallback)
        result = provider.transcribe(_silent_audio())

        assert result is None


# ── create_stt_provider factory ──────────────────────────────────────


class TestFactory:
    @patch("perception.stt.GroqWhisperProvider")
    @patch("perception.stt.LocalWhisperProvider")
    def test_groq_with_fallback(self, mock_local_cls, mock_groq_cls):
        """With Groq + fallback enabled, returns FallbackSTTProvider."""
        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(
                    STT_PROVIDER="groq",
                    GROQ_API_KEY="test-key",
                    GROQ_WHISPER_MODEL="whisper-large-v3-turbo",
                    WHISPER_MODEL="medium",
                    NOISE_REDUCE_ENABLED=False,
                    STT_FALLBACK_ENABLED=True,
                ),
            },
        ):
            provider = create_stt_provider()
            assert isinstance(provider, FallbackSTTProvider)

    @patch("perception.stt.LocalWhisperProvider")
    def test_local_only(self, mock_local_cls):
        """With STT_PROVIDER=local, returns LocalWhisperProvider."""
        mock_instance = MagicMock()
        mock_local_cls.return_value = mock_instance

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(
                    STT_PROVIDER="local",
                    GROQ_API_KEY=None,
                    GROQ_WHISPER_MODEL="whisper-large-v3-turbo",
                    WHISPER_MODEL="medium",
                    NOISE_REDUCE_ENABLED=False,
                    STT_FALLBACK_ENABLED=True,
                ),
            },
        ):
            provider = create_stt_provider()
            assert provider is mock_instance

    @patch("perception.stt.GroqWhisperProvider")
    @patch("perception.stt.LocalWhisperProvider")
    def test_groq_without_fallback(self, mock_local_cls, mock_groq_cls):
        """With fallback disabled, returns bare GroqWhisperProvider."""
        mock_groq_instance = MagicMock()
        mock_groq_cls.return_value = mock_groq_instance

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(
                    STT_PROVIDER="groq",
                    GROQ_API_KEY="test-key",
                    GROQ_WHISPER_MODEL="whisper-large-v3-turbo",
                    WHISPER_MODEL="medium",
                    NOISE_REDUCE_ENABLED=False,
                    STT_FALLBACK_ENABLED=False,
                ),
            },
        ):
            provider = create_stt_provider()
            assert provider is mock_groq_instance

    @patch("perception.stt.GroqWhisperProvider", side_effect=Exception("no groq"))
    @patch("perception.stt.LocalWhisperProvider")
    def test_groq_init_failure_falls_back(self, mock_local_cls, mock_groq_cls):
        """If Groq init fails, falls back to local."""
        mock_local_instance = MagicMock()
        mock_local_cls.return_value = mock_local_instance

        with patch.dict(
            "sys.modules",
            {
                "config": MagicMock(
                    STT_PROVIDER="groq",
                    GROQ_API_KEY="test-key",
                    GROQ_WHISPER_MODEL="whisper-large-v3-turbo",
                    WHISPER_MODEL="medium",
                    NOISE_REDUCE_ENABLED=False,
                    STT_FALLBACK_ENABLED=True,
                ),
            },
        ):
            provider = create_stt_provider()
            assert provider is mock_local_instance
