"""STT (Speech-to-Text) provider abstraction.

Supports multiple backends (Groq Whisper API, local faster-whisper) with
automatic per-request fallback when the primary provider fails.
"""

import io
import logging
import struct
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

logger = logging.getLogger("winston.stt")

DEFAULT_SAMPLE_RATE = 16000
WHISPER_PROMPT = "Winston, workshop assistant. Bilingual: English and German."


@dataclass
class STTResult:
    """Result from a speech-to-text transcription."""

    text: str
    language: str
    confidence: float
    duration_ms: float  # how long transcription took


class STTProvider(Protocol):
    """Protocol for speech-to-text providers."""

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Optional[STTResult]:
        """Transcribe audio. Returns STTResult or None on failure."""
        ...


class LocalWhisperProvider:
    """Local faster-whisper STT provider with lazy model loading."""

    def __init__(self, model_name: str = "medium", noise_reduce: bool = False):
        self._model_name = model_name
        self._noise_reduce = noise_reduce
        self._model = None  # Lazy-loaded on first transcribe()

    def _ensure_model(self) -> bool:
        """Load the Whisper model if not already loaded. Returns True on success."""
        if self._model is not None:
            return True
        try:
            from faster_whisper import WhisperModel

            logger.info("Loading Whisper model (%s) via faster-whisper...", self._model_name)
            self._model = WhisperModel(
                self._model_name,
                device="cpu",
                compute_type="int8",
            )
            logger.info("Whisper model loaded (faster-whisper, int8)")
            return True
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)
            return False

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Optional[STTResult]:
        if not self._ensure_model():
            return None

        start = time.monotonic()
        try:
            audio = audio_data
            if self._noise_reduce:
                try:
                    import noisereduce as nr

                    audio = nr.reduce_noise(y=audio, sr=sample_rate)
                except ImportError:
                    pass

            segments, info = self._model.transcribe(
                audio,
                beam_size=5,
                language=None,
                vad_filter=True,
                initial_prompt=WHISPER_PROMPT,
            )
            text = " ".join(seg.text for seg in segments).strip()
            lang = info.language if info else "en"
            duration_ms = (time.monotonic() - start) * 1000
            logger.info("Transcription [%s] (local, %.0fms): %s", lang, duration_ms, text)
            return STTResult(text=text, language=lang, confidence=0.0, duration_ms=duration_ms)
        except Exception as e:
            logger.error("Local transcription failed: %s", e)
            return None


class GroqWhisperProvider:
    """Groq Whisper API STT provider."""

    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo"):
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model
        logger.info("STT: Groq API (model: %s)", model)

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Optional[STTResult]:
        start = time.monotonic()
        try:
            wav_bytes = _audio_to_wav(audio_data, sample_rate)

            response = self._client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes),
                model=self._model,
                response_format="verbose_json",
                prompt=WHISPER_PROMPT,
            )
            text = response.text.strip() if response.text else ""
            lang = getattr(response, "language", "en") or "en"
            duration_ms = (time.monotonic() - start) * 1000
            logger.info("Transcription [%s] (groq, %.0fms): %s", lang, duration_ms, text)
            return STTResult(text=text, language=lang, confidence=0.0, duration_ms=duration_ms)
        except Exception as e:
            logger.error("Groq transcription failed: %s", e)
            return None


class FallbackSTTProvider:
    """Wraps a primary provider with automatic fallback to a secondary."""

    def __init__(self, primary: STTProvider, fallback: STTProvider):
        self._primary = primary
        self._fallback = fallback

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Optional[STTResult]:
        result = self._primary.transcribe(audio_data, sample_rate)
        if result is not None:
            return result
        logger.warning("Primary STT failed, falling back to secondary provider")
        return self._fallback.transcribe(audio_data, sample_rate)


def create_stt_provider() -> STTProvider:
    """Factory: create the appropriate STT provider based on config."""
    from config import GROQ_API_KEY, GROQ_WHISPER_MODEL, NOISE_REDUCE_ENABLED, STT_PROVIDER, WHISPER_MODEL

    # Import fallback setting with default
    try:
        from config import STT_FALLBACK_ENABLED
    except ImportError:
        STT_FALLBACK_ENABLED = True

    if STT_PROVIDER == "groq" and GROQ_API_KEY:
        try:
            primary = GroqWhisperProvider(api_key=GROQ_API_KEY, model=GROQ_WHISPER_MODEL)
            if STT_FALLBACK_ENABLED:
                fallback = LocalWhisperProvider(model_name=WHISPER_MODEL, noise_reduce=NOISE_REDUCE_ENABLED)
                return FallbackSTTProvider(primary, fallback)
            return primary
        except Exception as e:
            logger.error("Failed to initialize Groq STT: %s. Falling back to local.", e)

    return LocalWhisperProvider(model_name=WHISPER_MODEL, noise_reduce=NOISE_REDUCE_ENABLED)


def _audio_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy audio to 16-bit PCM WAV bytes."""
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    data_size = len(pcm) * 2  # 2 bytes per int16 sample
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # fmt chunk size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", 1))  # mono
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))  # byte rate
    buf.write(struct.pack("<H", 2))  # block align
    buf.write(struct.pack("<H", 16))  # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()
