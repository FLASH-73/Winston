import logging
import time
import threading
from abc import ABC, abstractmethod
from queue import Queue, Empty
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger("winston.tts")


class TTSBackend(ABC):
    """Abstract interface for a TTS backend."""

    @abstractmethod
    def initialize(self) -> bool:
        ...

    @abstractmethod
    def synthesize_and_play(self, text: str, on_playback_start: Optional[Callable] = None) -> None:
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...


class ElevenLabsBackend(TTSBackend):
    """ElevenLabs streaming TTS backend using sounddevice for playback."""

    def __init__(self):
        self._client = None
        self._voice_id = None
        self._model_id = None
        self._output_format = None
        self._sample_rate = None
        self._stability = None
        self._similarity = None
        self._style = None
        self._interrupt_flag = threading.Event()
        self._active_stream: Optional[sd.OutputStream] = None

    def initialize(self) -> bool:
        try:
            from elevenlabs.client import ElevenLabs
            from config import (
                ELEVENLABS_API_KEY,
                ELEVENLABS_VOICE_ID,
                ELEVENLABS_MODEL,
                ELEVENLABS_OUTPUT_FORMAT,
                ELEVENLABS_SAMPLE_RATE,
                ELEVENLABS_VOICE_STABILITY,
                ELEVENLABS_VOICE_SIMILARITY,
                ELEVENLABS_VOICE_STYLE,
            )

            if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "your_elevenlabs_key_here":
                logger.info("ELEVENLABS_API_KEY not set")
                return False

            self._client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            self._voice_id = ELEVENLABS_VOICE_ID
            self._model_id = ELEVENLABS_MODEL
            self._output_format = ELEVENLABS_OUTPUT_FORMAT
            self._sample_rate = ELEVENLABS_SAMPLE_RATE
            self._stability = ELEVENLABS_VOICE_STABILITY
            self._similarity = ELEVENLABS_VOICE_SIMILARITY
            self._style = ELEVENLABS_VOICE_STYLE

            logger.info("ElevenLabs backend initialized (voice=%s, model=%s)",
                        self._voice_id, self._model_id)
            return True

        except ImportError:
            logger.info("elevenlabs package not installed")
            return False
        except Exception as e:
            logger.error("ElevenLabs initialization failed: %s", e)
            return False

    def synthesize_and_play(self, text: str, on_playback_start: Optional[Callable] = None) -> None:
        """Stream audio from ElevenLabs and play chunks as they arrive."""
        from config import TTS_STREAMING_PLAYBACK
        from elevenlabs import VoiceSettings

        audio_stream = self._client.text_to_speech.stream(
            text=text,
            voice_id=self._voice_id,
            model_id=self._model_id,
            output_format=self._output_format,
            voice_settings=VoiceSettings(
                stability=self._stability,
                similarity_boost=self._similarity,
                style=self._style,
                use_speaker_boost=True,
            ),
        )

        if TTS_STREAMING_PLAYBACK:
            self._play_streaming(audio_stream, on_playback_start)
        else:
            self._play_buffered(audio_stream, on_playback_start)

    def _play_streaming(self, audio_stream, on_playback_start: Optional[Callable] = None) -> None:
        """Play audio chunks as they arrive via sd.OutputStream."""
        self._interrupt_flag.clear()
        stream_out = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
        )
        stream_out.start()
        self._active_stream = stream_out
        playback_started = False

        try:
            for chunk in audio_stream:
                if self._interrupt_flag.is_set():
                    break
                if not isinstance(chunk, bytes) or len(chunk) == 0:
                    continue
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                if not playback_started:
                    playback_started = True
                    if on_playback_start:
                        on_playback_start()
                stream_out.write(samples.reshape(-1, 1))
        finally:
            if not self._interrupt_flag.is_set():
                time.sleep(0.1)  # Let last chunk drain
            stream_out.stop()
            stream_out.close()
            self._active_stream = None

    def _play_buffered(self, audio_stream, on_playback_start: Optional[Callable] = None) -> None:
        """Original approach: collect all chunks, then play at once."""
        audio_chunks = []
        for chunk in audio_stream:
            if isinstance(chunk, bytes) and len(chunk) > 0:
                audio_chunks.append(chunk)
        if not audio_chunks:
            raise RuntimeError("No audio data received from ElevenLabs")
        pcm_data = b"".join(audio_chunks)
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        if on_playback_start:
            on_playback_start()
        sd.play(samples, samplerate=self._sample_rate)
        sd.wait()

    def interrupt(self) -> None:
        """Interrupt streaming playback. Safe to call from any thread."""
        self._interrupt_flag.set()
        stream = self._active_stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass

    def shutdown(self) -> None:
        self.interrupt()
        self._client = None


class Pyttsx3Backend(TTSBackend):
    """Offline pyttsx3 TTS backend (fallback)."""

    def __init__(self, rate: int = 175, volume: float = 0.9):
        self._rate = rate
        self._volume = volume
        self._engine = None

    def initialize(self) -> bool:
        try:
            import pyttsx3
            try:
                self._engine = pyttsx3.init("nsss")  # macOS NSSpeechSynthesizer
            except Exception:
                self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
            self._select_voice()
            logger.info("pyttsx3 fallback backend initialized")
            return True
        except Exception as e:
            logger.error("pyttsx3 initialization failed: %s", e)
            return False

    def synthesize_and_play(self, text: str, on_playback_start: Optional[Callable] = None) -> None:
        if self._engine is None:
            return
        # pyttsx3 plays immediately (no download delay)
        if on_playback_start:
            on_playback_start()
        self._engine.say(text)
        self._engine.runAndWait()

    def stop_speaking(self) -> None:
        """Stop pyttsx3 mid-utterance. Must be called from the same thread."""
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    def shutdown(self) -> None:
        self._engine = None

    def _select_voice(self):
        if self._engine is None:
            return
        voices = self._engine.getProperty("voices")
        preferred = ["daniel", "alex", "samantha"]
        for pref in preferred:
            for voice in voices:
                if pref in voice.name.lower():
                    self._engine.setProperty("voice", voice.id)
                    logger.info("pyttsx3 voice: %s", voice.name)
                    return


class TTSEngine:
    """Main TTS engine with ElevenLabs primary + pyttsx3 fallback.

    Public interface (unchanged from original):
      - start() -> bool
      - speak(text) -> None          (blocking)
      - speak_async(text) -> None    (non-blocking, queued)
      - stop() -> None
    """

    def __init__(self, rate: int = 175, volume: float = 0.9):
        self._rate = rate
        self._volume = volume
        self._queue: Queue = Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._primary: Optional[TTSBackend] = None
        self._fallback: Optional[TTSBackend] = None
        self._consecutive_primary_failures = 0
        self._max_consecutive_failures = 3

        # Barge-in / interrupt support
        self._interrupt_event = threading.Event()
        self._is_speaking = False
        self._is_speaking_lock = threading.Lock()
        self._speaking_start_time: float = 0.0

        # Echo rejection: track what TTS is currently saying
        self._last_spoken_text: str = ""
        self._last_spoken_lock = threading.Lock()

        # Streaming response: keep is_speaking=True between sentences
        self._streaming_response_active = False

        # Callbacks for TTS state changes (notifies AudioListener)
        self._on_speaking_start: Optional[Callable[[], None]] = None
        self._on_speaking_stop: Optional[Callable[[], None]] = None

    def start(self) -> bool:
        """Start the TTS worker thread."""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="tts-worker")
        self._thread.start()
        logger.info("TTS engine started")
        return True

    def speak(self, text: str) -> None:
        """Speak text and block until done."""
        if not text or not text.strip():
            return
        done_event = threading.Event()
        self._queue.put((text, done_event))
        done_event.wait(timeout=30)

    def speak_async(self, text: str) -> None:
        """Queue text for async speech (non-blocking)."""
        if not text or not text.strip():
            return
        self._queue.put((text, None))

    @property
    def is_speaking(self) -> bool:
        """Thread-safe check whether TTS is currently playing audio."""
        with self._is_speaking_lock:
            return self._is_speaking

    @property
    def speaking_start_time(self) -> float:
        """Monotonic time when current speech started. 0.0 if not speaking."""
        with self._is_speaking_lock:
            return self._speaking_start_time if self._is_speaking else 0.0

    @property
    def last_spoken_text(self) -> str:
        """The most recent text that TTS spoke or is speaking."""
        with self._last_spoken_lock:
            return self._last_spoken_text

    def set_state_callbacks(
        self,
        on_speaking_start: Optional[Callable[[], None]] = None,
        on_speaking_stop: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register callbacks for TTS state transitions."""
        if on_speaking_start is not None:
            self._on_speaking_start = on_speaking_start
        if on_speaking_stop is not None:
            self._on_speaking_stop = on_speaking_stop

    def begin_streaming_response(self) -> None:
        """Mark start of a multi-sentence streaming response.

        While active, the worker keeps is_speaking=True between sentences
        and doesn't fire _on_speaking_stop until end_streaming_response().
        """
        self._streaming_response_active = True

    def end_streaming_response(self) -> None:
        """Mark end of streaming response. Queues a boundary marker.

        The worker will fire _on_speaking_stop when it processes the marker.
        """
        self._streaming_response_active = False
        self._queue.put(("__RESPONSE_BOUNDARY__", None))

    def interrupt(self) -> None:
        """Interrupt current speech immediately. Safe to call from any thread."""
        self._streaming_response_active = False
        self._interrupt_event.set()
        # Stop sounddevice playback (works from any thread)
        try:
            sd.stop()
        except Exception:
            pass
        # Interrupt backend streaming if active
        if self._primary and hasattr(self._primary, "interrupt"):
            try:
                self._primary.interrupt()
            except Exception:
                pass
        # Clear pending items so nothing else plays
        self._clear_queue()
        logger.info("TTS interrupted")

    def _clear_queue(self) -> None:
        """Drain the queue, signaling any blocking speak() callers."""
        while True:
            try:
                item = self._queue.get_nowait()
                if item is not None:
                    _, done_event = item
                    if done_event is not None:
                        done_event.set()
            except Empty:
                break

    def _worker(self):
        """Background thread: initializes backends and processes queue."""
        # Initialize backends on the worker thread (pyttsx3 requires same-thread init/use)
        elevenlabs_backend = ElevenLabsBackend()
        pyttsx3_backend = Pyttsx3Backend(rate=self._rate, volume=self._volume)

        if elevenlabs_backend.initialize():
            self._primary = elevenlabs_backend
            logger.info("Using ElevenLabs as primary TTS")
        else:
            logger.info("ElevenLabs unavailable, using pyttsx3 only")

        if pyttsx3_backend.initialize():
            self._fallback = pyttsx3_backend

        if self._primary is None and self._fallback is None:
            logger.error("No TTS backend available!")
            return

        # Accumulate sentences from streaming responses for echo cancellation.
        # When sentences arrive rapidly (streaming), they're part of the same response.
        # On queue timeout (gap), the response is complete — reset accumulator.
        response_accumulator = ""

        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                # Gap in queue = response complete, reset accumulator
                if response_accumulator and not self._streaming_response_active:
                    response_accumulator = ""
                continue

            if item is None:
                break

            # Check if interrupted before starting speech
            if self._interrupt_event.is_set():
                self._interrupt_event.clear()
                response_accumulator = ""
                text_item, done_ev = item
                if done_ev is not None:
                    done_ev.set()
                continue

            text, done_event = item

            # Handle response boundary marker (end of streaming response)
            if text == "__RESPONSE_BOUNDARY__":
                with self._is_speaking_lock:
                    self._is_speaking = False
                if self._on_speaking_stop:
                    try:
                        self._on_speaking_stop()
                    except Exception:
                        pass
                response_accumulator = ""
                if done_event is not None:
                    done_event.set()
                continue

            try:
                # Accumulate for echo cancellation (sentences → full response)
                response_accumulator = (response_accumulator + " " + text).strip()
                with self._last_spoken_lock:
                    self._last_spoken_text = response_accumulator

                with self._is_speaking_lock:
                    self._is_speaking = True
                    self._speaking_start_time = time.monotonic()

                # _on_speaking_start is deferred to on_playback_start callback
                # so it fires AFTER HTTP download, right before sd.play()
                def _notify_playback_start():
                    with self._is_speaking_lock:
                        self._speaking_start_time = time.monotonic()
                    if self._on_speaking_start:
                        try:
                            self._on_speaking_start()
                        except Exception:
                            pass

                self._speak_with_fallback(text, on_playback_start=_notify_playback_start)
            except Exception as e:
                logger.error("TTS error: %s", e)
            finally:
                if not self._streaming_response_active:
                    # Standalone utterance or last sentence — fire stop
                    with self._is_speaking_lock:
                        self._is_speaking = False
                    if self._on_speaking_stop:
                        try:
                            self._on_speaking_stop()
                        except Exception:
                            pass
                # else: streaming response active — keep is_speaking=True
                self._interrupt_event.clear()
                if done_event is not None:
                    done_event.set()

    def _speak_with_fallback(self, text: str, on_playback_start: Optional[Callable] = None) -> None:
        """Try primary backend, fall back to secondary on failure."""
        if self._primary is not None and self._consecutive_primary_failures < self._max_consecutive_failures:
            try:
                self._primary.synthesize_and_play(text, on_playback_start=on_playback_start)
                self._consecutive_primary_failures = 0
                return
            except Exception as e:
                self._consecutive_primary_failures += 1
                logger.warning(
                    "ElevenLabs failed (%d/%d): %s — falling back to pyttsx3",
                    self._consecutive_primary_failures,
                    self._max_consecutive_failures,
                    e,
                )

        if self._fallback is not None:
            self._fallback.synthesize_and_play(text, on_playback_start=on_playback_start)
        else:
            logger.error("No fallback TTS available, speech dropped: %.50s", text)

    def stop(self):
        """Stop the TTS engine."""
        self._running = False
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._primary:
            self._primary.shutdown()
        if self._fallback:
            self._fallback.shutdown()
        logger.info("TTS engine stopped")
