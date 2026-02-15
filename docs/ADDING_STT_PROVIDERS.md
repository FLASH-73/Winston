# Adding STT Providers

Winston's speech-to-text system uses a provider abstraction defined in `winston/perception/stt.py`. This guide explains how to add a new STT backend.

## Architecture

```
                 ┌──────────────┐
                 │ STTProvider   │  (Protocol — duck typing)
                 │ .transcribe() │
                 └──────┬───────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
┌────────┴───┐  ┌───────┴────┐  ┌──────┴───────┐
│ LocalWhisper│  │ GroqWhisper │  │ YourProvider  │
│ Provider    │  │ Provider    │  │               │
└─────────────┘  └────────────┘  └───────────────┘

         ┌────────────────────┐
         │ FallbackSTTProvider │  (Composite: tries primary, then fallback)
         └────────────────────┘

         ┌────────────────────┐
         │ create_stt_provider │  (Factory: reads config, returns provider)
         └────────────────────┘
```

### The Protocol

```python
class STTProvider(Protocol):
    def transcribe(
        self, audio_data: np.ndarray, sample_rate: int = 16000
    ) -> Optional[STTResult]:
        ...
```

- **Input**: `audio_data` is a float32 numpy array, mono, typically 16kHz. `sample_rate` defaults to 16000.
- **Output**: `STTResult` on success, `None` on failure.
- **Contract**: Must not raise exceptions. Catch internally and return `None`.

### STTResult

```python
@dataclass
class STTResult:
    text: str           # Transcribed text
    language: str       # ISO 639-1 code (e.g. "en", "de")
    confidence: float   # 0.0–1.0 (use 0.0 if provider doesn't report it)
    duration_ms: float  # How long transcription took (for latency tracking)
```

---

## Step-by-Step: Adding a Provider

This example adds a Deepgram provider.

### Step 1: Implement the Provider Class

In `winston/perception/stt.py`, add a new class:

```python
class DeepgramProvider:
    """Deepgram STT API provider."""

    def __init__(self, api_key: str, model: str = "nova-2"):
        from deepgram import DeepgramClient
        self._client = DeepgramClient(api_key)
        self._model = model
        logger.info("STT: Deepgram API (model: %s)", model)

    def transcribe(
        self, audio_data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> Optional[STTResult]:
        start = time.monotonic()
        try:
            wav_bytes = _audio_to_wav(audio_data, sample_rate)

            response = self._client.listen.prerecorded.v("1").transcribe_file(
                {"buffer": wav_bytes, "mimetype": "audio/wav"},
                {"model": self._model, "detect_language": True},
            )

            alt = response.results.channels[0].alternatives[0]
            text = alt.transcript.strip()
            lang = response.results.channels[0].detected_language or "en"
            confidence = alt.confidence
            duration_ms = (time.monotonic() - start) * 1000

            logger.info("Transcription [%s] (deepgram, %.0fms): %s", lang, duration_ms, text)
            return STTResult(text=text, language=lang, confidence=confidence, duration_ms=duration_ms)

        except Exception as e:
            logger.error("Deepgram transcription failed: %s", e)
            return None
```

### Step 2: Add Config Values

In `winston/config.py`:

```python
# Deepgram STT
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_MODEL = "nova-2"
```

### Step 3: Register in the Factory

Update `create_stt_provider()` in `winston/perception/stt.py`:

```python
def create_stt_provider() -> STTProvider:
    from config import STT_PROVIDER, ...

    if STT_PROVIDER == "deepgram" and DEEPGRAM_API_KEY:
        try:
            primary = DeepgramProvider(api_key=DEEPGRAM_API_KEY, model=DEEPGRAM_MODEL)
            if STT_FALLBACK_ENABLED:
                fallback = LocalWhisperProvider(...)
                return FallbackSTTProvider(primary, fallback)
            return primary
        except Exception as e:
            logger.error("Deepgram init failed: %s. Falling back.", e)

    # ... existing groq/local logic ...
```

### Step 4: Write Tests

Follow the patterns in `winston/tests/test_stt.py`:

```python
from unittest.mock import patch, MagicMock
import numpy as np


def _silent_audio(duration_s=1.0, sample_rate=16000):
    """Generate silent audio for testing."""
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


class TestDeepgramProvider:
    def test_transcribe_success(self):
        with patch.dict("sys.modules", {"deepgram": MagicMock()}):
            from perception.stt import DeepgramProvider

            provider = DeepgramProvider(api_key="test-key")

            # Mock the API response
            mock_alt = MagicMock()
            mock_alt.transcript = "hello world"
            mock_alt.confidence = 0.95
            mock_channel = MagicMock()
            mock_channel.alternatives = [mock_alt]
            mock_channel.detected_language = "en"
            mock_response = MagicMock()
            mock_response.results.channels = [mock_channel]
            provider._client.listen.prerecorded.v.return_value.transcribe_file.return_value = mock_response

            result = provider.transcribe(_silent_audio())
            assert result is not None
            assert result.text == "hello world"
            assert result.language == "en"

    def test_transcribe_error_returns_none(self):
        with patch.dict("sys.modules", {"deepgram": MagicMock()}):
            from perception.stt import DeepgramProvider

            provider = DeepgramProvider(api_key="test-key")
            provider._client.listen.prerecorded.v.side_effect = Exception("API error")

            result = provider.transcribe(_silent_audio())
            assert result is None
```

---

## Key Conventions

1. **Reuse `_audio_to_wav()`** — Converts float32 numpy to 16-bit PCM WAV bytes. Already in `stt.py`. Use it for any provider that needs WAV input.

2. **Log transcription results at INFO** — Format: `Transcription [lang] (provider, Xms): text`. This feeds into latency tracking.

3. **Never raise exceptions** — `transcribe()` must catch all errors and return `None`. The `FallbackSTTProvider` relies on this to trigger fallback.

4. **Auto-detect language** — Do not hardcode `language="en"`. Winston is bilingual (English + German). Use the provider's language detection feature.

5. **Use `WHISPER_PROMPT`** — If the provider supports a prompt/context hint, pass `WHISPER_PROMPT` (`"Winston, workshop assistant. Bilingual: English and German."`). It improves language detection and reduces hallucinations.

6. **Config pattern** — API key from env var, model name as config constant, provider name in `STT_PROVIDER` string.

---

## Checklist

- [ ] Class implements `transcribe(audio_data, sample_rate) -> Optional[STTResult]`
- [ ] Catches all exceptions internally, returns `None` on failure
- [ ] Logs at INFO with provider name and timing
- [ ] Config values added to `config.py` (API key from env, model as constant)
- [ ] Factory function updated in `create_stt_provider()`
- [ ] Fallback wrapping supported via `FallbackSTTProvider`
- [ ] Tests in `winston/tests/test_stt.py`
- [ ] Tests work offline (mocked SDK, no API calls)
