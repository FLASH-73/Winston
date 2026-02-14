import logging
import os
import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger("winston.audio")

# Known Whisper hallucinations on near-silent audio
WHISPER_HALLUCINATIONS = {
    "thank you for joining us",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "see you next time",
    "bye bye",
    "you",
    "the end",
    "thanks for listening",
    "subtitles by",
}

# Whisper prompt words — used to detect prompt-leak transcriptions.
# When Whisper gets ambiguous audio, it echoes back the prompt text.
WHISPER_PROMPT_WORDS = set("winston workshop assistant bilingual english and german".split())

SAMPLE_RATE = 16000  # Both openwakeword and Whisper need 16kHz
CHUNK_SAMPLES = 1280  # 80ms at 16kHz — openwakeword's expected frame size
SILENCE_DURATION = 1.0  # Seconds of silence before ending recording
SILENCE_CHECK_SAMPLES = int(SAMPLE_RATE * 0.5)  # 0.5s chunks for silence detection


class AudioListener:
    def __init__(self):
        self._whisper_model = None
        self._oww_model = None
        self._vad_model = None
        self._stream: Optional[sd.InputStream] = None
        self._running = False

        # Rolling audio buffer (30 seconds)
        self._audio_buffer: deque[np.ndarray] = deque(maxlen=int(SAMPLE_RATE * 30 / CHUNK_SAMPLES))

        # Post-wake-word recording state
        self._is_listening = False
        self._listen_buffer: list[np.ndarray] = []
        self._listen_start_time = 0.0

        # Callbacks
        self._on_wake_word: Optional[Callable] = None
        self._on_transcription: Optional[Callable[[str], None]] = None
        self._on_bargein: Optional[Callable] = None
        self._tts_is_speaking: Optional[Callable[[], bool]] = None

        # Barge-in state
        self._bargein_cooldown_until = 0.0  # Suppress barge-in after false positives

        # Energy-delta barge-in detector (replaces Silero VAD during TTS)
        from utils.echo_cancel import EnergyBargeInDetector
        from config import BARGEIN_ENERGY_THRESHOLD, BARGEIN_THRESHOLD_FACTOR, BARGEIN_CONSECUTIVE_FRAMES
        self._energy_detector = EnergyBargeInDetector(
            threshold_factor=BARGEIN_THRESHOLD_FACTOR,
            consecutive_trigger=BARGEIN_CONSECUTIVE_FRAMES,
            calibration_frames=5,
            min_energy=BARGEIN_ENERGY_THRESHOLD,
        )

        # Echo text rejection
        self._get_last_tts_text: Optional[Callable[[], str]] = None

        # Audio level for dashboard (updated every callback)
        self._current_audio_level = 0.0

        # OWW deferred loading state
        self._oww_ready = False
        self._oww_loading = False

        # Always-listening state machine
        self._always_listen_enabled = False
        self._al_state = "disabled"  # "disabled" | "idle" | "accumulating" | "cooldown"
        self._al_buffer: list[np.ndarray] = []
        self._al_speech_start_time = 0.0
        self._al_silence_counter = 0.0
        self._al_last_check_time = 0.0
        self._al_cooldown_until = 0.0
        self._al_active_threads = 0
        self._al_active_threads_lock = threading.Lock()
        self._al_last_tts_stop_time = 0.0
        self._on_ambient_transcription: Optional[Callable[[str], None]] = None

        # STT provider state (initialized in start())
        self._stt_provider = "local"  # "groq" or "local"
        self._groq_client = None
        self._groq_whisper_model = None

        # Import-time config
        from config import WHISPER_MODEL, SILENCE_THRESHOLD, LISTEN_TIMEOUT
        from config import BARGEIN_ENABLED
        from config import NOISE_REDUCE_ENABLED, MIN_AUDIO_DURATION
        from config import (
            ALWAYS_LISTEN_ENABLED, ALWAYS_LISTEN_ENERGY_THRESHOLD,
            ALWAYS_LISTEN_SILENCE_DURATION, ALWAYS_LISTEN_TIMEOUT,
            ALWAYS_LISTEN_MIN_SPEECH_DURATION, ALWAYS_LISTEN_COOLDOWN_AFTER_TTS,
            ALWAYS_LISTEN_COOLDOWN_AFTER_RESPONSE, ALWAYS_LISTEN_STORE_REJECTED,
        )
        self._whisper_model_name = WHISPER_MODEL
        self._silence_threshold = SILENCE_THRESHOLD
        self._listen_timeout = LISTEN_TIMEOUT
        self._bargein_enabled = BARGEIN_ENABLED
        self._noise_reduce_enabled = NOISE_REDUCE_ENABLED
        self._min_audio_duration = MIN_AUDIO_DURATION
        self._al_config_enabled = ALWAYS_LISTEN_ENABLED
        self._al_energy_threshold = ALWAYS_LISTEN_ENERGY_THRESHOLD
        self._al_silence_duration = ALWAYS_LISTEN_SILENCE_DURATION
        self._al_timeout = ALWAYS_LISTEN_TIMEOUT
        self._al_min_speech_duration = ALWAYS_LISTEN_MIN_SPEECH_DURATION
        self._al_cooldown_after_tts = ALWAYS_LISTEN_COOLDOWN_AFTER_TTS
        self._al_cooldown_after_response = ALWAYS_LISTEN_COOLDOWN_AFTER_RESPONSE
        self._al_store_rejected = ALWAYS_LISTEN_STORE_REJECTED

    def start(self) -> bool:
        """Initialize STT provider, defer OWW to background, open audio stream."""
        # 1. Initialize STT provider
        from config import STT_PROVIDER
        if STT_PROVIDER == "groq":
            try:
                from groq import Groq
                from config import GROQ_API_KEY, GROQ_WHISPER_MODEL
                self._groq_client = Groq(api_key=GROQ_API_KEY)
                self._groq_whisper_model = GROQ_WHISPER_MODEL
                self._stt_provider = "groq"
                logger.info("STT: Groq API (model: %s)", GROQ_WHISPER_MODEL)
            except Exception as e:
                logger.error("Failed to initialize Groq STT: %s. Falling back to local.", e)
                STT_PROVIDER = "local"

        if STT_PROVIDER != "groq":
            try:
                from faster_whisper import WhisperModel
                logger.info("Loading Whisper model (%s) via faster-whisper...", self._whisper_model_name)
                self._whisper_model = WhisperModel(
                    self._whisper_model_name,
                    device="cpu",
                    compute_type="int8",
                )
                self._stt_provider = "local"
                logger.info("Whisper model loaded (faster-whisper, int8)")
            except Exception as e:
                logger.error("Failed to load Whisper model: %s", e)
                return False

        # 2. Open audio stream
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("Audio stream opened (16kHz mono)")
        except Exception as e:
            logger.error("Failed to open audio stream: %s", e)
            return False

        # 3. Start processing thread
        self._running = True
        self._thread = threading.Thread(target=self._processing_loop, daemon=True, name="audio-processor")
        self._thread.start()

        # 4. Enable always-listening if configured
        if self._al_config_enabled:
            self._always_listen_enabled = True
            self._al_state = "idle"
            logger.info("Always-listening mode enabled")

        return True

    def _load_oww_background(self):
        """Load openwakeword + VAD in background thread. Sets _oww_ready when done."""
        try:
            import openwakeword
            from openwakeword.model import Model as OWWModel

            # Download models on first run if they don't exist yet
            model_paths = openwakeword.get_pretrained_model_paths("onnx")
            if not any(os.path.exists(p) for p in model_paths):
                logger.info("Downloading openwakeword models (first time)...")
                from openwakeword.utils import download_models
                download_models()

            logger.info("Loading wake word model...")
            self._oww_model = OWWModel(
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx",
            )
            logger.info("Wake word model loaded")

            # Load Silero VAD for barge-in (depends on openwakeword)
            if self._bargein_enabled:
                try:
                    from openwakeword.vad import VAD as SileroVAD
                    self._vad_model = SileroVAD()
                    logger.info("Silero VAD loaded for barge-in detection (ONNX)")
                except Exception as e:
                    logger.warning("Silero VAD failed to load, barge-in disabled: %s", e)
                    self._bargein_enabled = False

            self._oww_ready = True
            logger.info("Wake word detection now active")

        except Exception as e:
            logger.error("Failed to load wake word model: %s", e)
        finally:
            self._oww_loading = False

    @property
    def wake_word_ready(self) -> bool:
        """Whether openwakeword has finished loading."""
        return self._oww_ready

    def set_callbacks(
        self,
        on_wake_word: Optional[Callable] = None,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_bargein: Optional[Callable] = None,
        tts_is_speaking: Optional[Callable[[], bool]] = None,
        get_last_tts_text: Optional[Callable[[], str]] = None,
        on_ambient_transcription: Optional[Callable[[str], None]] = None,
    ):
        """Register callback functions."""
        if on_wake_word is not None:
            self._on_wake_word = on_wake_word
        if on_transcription is not None:
            self._on_transcription = on_transcription
        if on_bargein is not None:
            self._on_bargein = on_bargein
        if tts_is_speaking is not None:
            self._tts_is_speaking = tts_is_speaking
        if get_last_tts_text is not None:
            self._get_last_tts_text = get_last_tts_text
        if on_ambient_transcription is not None:
            self._on_ambient_transcription = on_ambient_transcription

    def on_tts_start(self) -> None:
        """Called when TTS starts playing. Starts energy-delta calibration.

        Only calibrates if detector is not already active — avoids resetting
        mid-response during streaming multi-sentence playback.
        """
        if not self._energy_detector.is_active:
            self._energy_detector.start_calibration()

    def on_tts_stop(self) -> None:
        """Called when TTS stops playing. Resets energy-delta detector."""
        self._energy_detector.reset()
        self._al_last_tts_stop_time = time.monotonic()

    def trigger_listen(self):
        """Manually trigger listen mode (dashboard button fallback).

        Runs the same flow as wake word detection: acknowledge, record, transcribe.
        Safe to call from any thread.
        """
        if self._is_listening:
            logger.info("Already listening, ignoring manual trigger")
            return

        logger.info("Manual listen triggered (dashboard button)")
        threading.Thread(
            target=self._handle_wake_word,
            daemon=True,
            name="manual-listen",
        ).start()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """sounddevice callback — runs on PortAudio thread. Keep minimal."""
        if status:
            logger.debug("Audio status: %s", status)
        chunk = indata[:, 0].copy()  # Mono float32
        self._audio_buffer.append(chunk)
        if self._is_listening:
            self._listen_buffer.append(chunk)
        if self._al_state == "accumulating":
            self._al_buffer.append(chunk)
        # Publish audio level for dashboard
        self._current_audio_level = float(np.sqrt(np.mean(chunk ** 2)))

    def _processing_loop(self):
        """Main processing loop running in background thread."""
        # Buffer to accumulate audio for wake word detection
        oww_buffer = np.zeros(0, dtype=np.float32)

        while self._running:
            # Collect available audio chunks
            chunks = []
            while self._audio_buffer:
                try:
                    chunks.append(self._audio_buffer.popleft())
                except IndexError:
                    break

            if not chunks:
                time.sleep(0.05)
                continue

            audio = np.concatenate(chunks)
            oww_buffer = np.concatenate([oww_buffer, audio])

            # Process in CHUNK_SAMPLES-sized frames
            while len(oww_buffer) >= CHUNK_SAMPLES:
                frame = oww_buffer[:CHUNK_SAMPLES]
                oww_buffer = oww_buffer[CHUNK_SAMPLES:]

                # Barge-in detection during TTS playback
                if (self._bargein_enabled
                        and not self._is_listening
                        and self._tts_is_speaking is not None
                        and self._tts_is_speaking()
                        and time.monotonic() >= self._bargein_cooldown_until):
                    self._check_bargein(frame)

                # Always-listen: detect ambient speech (non-blocking)
                if not self._is_listening:
                    self._al_process_frame(frame)

    def _handle_wake_word(self):
        """Handle wake word detection: acknowledge and record immediately."""
        # Notify via callback (this queues "Yes?" TTS)
        if self._on_wake_word:
            self._on_wake_word()

        # Record immediately — echo text rejection handles any TTS contamination
        self._record_and_transcribe()

    def _handle_bargein(self):
        """Handle barge-in: interrupt TTS, then record and transcribe."""
        logger.info("Barge-in triggered — interrupting TTS")
        if self._on_bargein:
            self._on_bargein()

        # Explicitly deactivate detector (don't rely on async TTS callback chain)
        self._energy_detector.reset()

        # Small delay to let TTS audio stop cleanly
        time.sleep(0.05)

        self._record_and_transcribe(is_bargein=True)

    def _record_and_transcribe(self, is_bargein: bool = False):
        """Shared logic: record audio until silence/timeout, then transcribe.

        Args:
            is_bargein: If True, set a cooldown on empty transcription to prevent
                        barge-in loops from ambient noise (3D printer, etc.).
        """
        # Enter listening mode
        self._listen_buffer = []
        self._is_listening = True
        self._listen_start_time = time.time()

        logger.info("Listening for speech...")

        # Record until silence or timeout
        consecutive_silence = 0.0
        last_check = time.time()

        while self._is_listening:
            time.sleep(0.1)
            elapsed = time.time() - self._listen_start_time

            # Timeout check
            if elapsed > self._listen_timeout:
                logger.info("Listen timeout reached (%.0fs)", self._listen_timeout)
                break

            # Silence detection — check accumulated audio in chunks
            now = time.time()
            if now - last_check >= 0.5 and self._listen_buffer:
                last_check = now
                # Check the last 0.5s of audio
                recent = self._listen_buffer[-int(SAMPLE_RATE * 0.5 / CHUNK_SAMPLES):]
                if recent:
                    recent_audio = np.concatenate(recent)
                    if self._detect_silence(recent_audio):
                        consecutive_silence += 0.5
                        if consecutive_silence >= SILENCE_DURATION:
                            logger.info("Silence detected, stopping recording")
                            break
                    else:
                        consecutive_silence = 0.0

        self._is_listening = False

        # Transcribe
        if self._listen_buffer:
            audio = np.concatenate(self._listen_buffer)
            self._listen_buffer = []

            # Skip if very short (probably just noise after wake word)
            if len(audio) < SAMPLE_RATE * self._min_audio_duration:
                logger.info("Audio too short (%.2fs), skipping transcription",
                            len(audio) / SAMPLE_RATE)
                if is_bargein:
                    self._bargein_cooldown_until = time.monotonic() + 3.0
                    logger.info("Barge-in false positive (short audio) — cooldown 3s")
                return

            text, lang = self._transcribe(audio)

            # Barge-in false positive detection: if we got no speech, suppress
            # further barge-in for a cooldown period to break infinite loops.
            if is_bargein and (not text or not text.strip()):
                self._bargein_cooldown_until = time.monotonic() + 3.0
                logger.info("Barge-in false positive (empty transcription) — cooldown 3s")
                return

            if text and self._on_transcription:
                # Echo text handling: strip TTS prefix from transcription
                if self._get_last_tts_text is not None:
                    from utils.echo_cancel import strip_echo_prefix, echo_text_overlap
                    last_tts = self._get_last_tts_text()
                    if last_tts:
                        # First try prefix stripping ("Yes, how are you?" -> "how are you?")
                        cleaned = strip_echo_prefix(text, last_tts)
                        if not cleaned:
                            # All words were TTS echo — discard entirely
                            logger.info("Echo rejected (all words matched TTS): '%s'", text)
                            return
                        if cleaned != text:
                            text = cleaned
                        else:
                            # No prefix match — fall back to overlap check
                            overlap = echo_text_overlap(text, last_tts)
                            if overlap > 0.6:
                                logger.info(
                                    "Echo rejected (%.0f%% overlap with TTS): '%s'",
                                    overlap * 100, text,
                                )
                                return

                # Run callback in separate thread to avoid blocking audio processing
                threading.Thread(
                    target=self._on_transcription,
                    args=(text, lang),
                    daemon=True,
                    name="transcription-handler",
                ).start()
        else:
            logger.info("No audio captured")

    def _check_bargein(self, frame: np.ndarray):
        """Detect user speech during TTS using energy spike detection."""
        if self._energy_detector.process(frame):
            self._handle_bargein()

    def _detect_silence(self, audio_chunk: np.ndarray) -> bool:
        """Return True if audio chunk is below silence threshold."""
        rms = np.sqrt(np.mean(audio_chunk**2)) * 32768  # Convert float32 to int16 scale
        return rms < self._silence_threshold

    def _transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """Transcribe audio. Routes to Groq API or local faster-whisper."""
        if self._stt_provider == "groq":
            return self._transcribe_groq(audio)
        return self._transcribe_local(audio)

    def _transcribe_groq(self, audio: np.ndarray) -> tuple[str, str]:
        """Transcribe via Groq Whisper API. Returns (text, detected_language_code)."""
        import io
        import struct

        try:
            # Convert float32 numpy → 16-bit PCM WAV bytes
            pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            buf = io.BytesIO()
            # Write WAV header (44 bytes) for 16kHz mono 16-bit PCM
            data_size = len(pcm) * 2  # 2 bytes per int16 sample
            buf.write(b"RIFF")
            buf.write(struct.pack("<I", 36 + data_size))
            buf.write(b"WAVE")
            buf.write(b"fmt ")
            buf.write(struct.pack("<I", 16))       # fmt chunk size
            buf.write(struct.pack("<H", 1))        # PCM format
            buf.write(struct.pack("<H", 1))        # mono
            buf.write(struct.pack("<I", SAMPLE_RATE))
            buf.write(struct.pack("<I", SAMPLE_RATE * 2))  # byte rate
            buf.write(struct.pack("<H", 2))        # block align
            buf.write(struct.pack("<H", 16))       # bits per sample
            buf.write(b"data")
            buf.write(struct.pack("<I", data_size))
            buf.write(pcm.tobytes())
            buf.seek(0)

            response = self._groq_client.audio.transcriptions.create(
                file=("audio.wav", buf.read()),
                model=self._groq_whisper_model,
                response_format="verbose_json",
                prompt="Winston, workshop assistant. Bilingual: English and German.",
            )
            text = response.text.strip() if response.text else ""
            lang = getattr(response, "language", "en") or "en"
            logger.info("Transcription [%s] (groq): %s", lang, text)
            return text, lang
        except Exception as e:
            logger.error("Groq transcription failed: %s", e)
            return "", "en"

    def _transcribe_local(self, audio: np.ndarray) -> tuple[str, str]:
        """Transcribe via local faster-whisper. Returns (text, detected_language_code)."""
        try:
            if self._noise_reduce_enabled:
                try:
                    import noisereduce as nr
                    audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE)
                except ImportError:
                    pass

            segments, info = self._whisper_model.transcribe(
                audio,
                beam_size=5,
                language=None,
                vad_filter=True,
                initial_prompt="Winston, workshop assistant. Bilingual: English and German.",
            )
            text = " ".join(seg.text for seg in segments).strip()
            lang = info.language if info else "en"
            logger.info("Transcription [%s] (local): %s", lang, text)
            return text, lang
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return "", "en"

    # ── Always-listen state machine ───────────────────────────────────

    def _al_process_frame(self, frame: np.ndarray) -> None:
        """Process a single audio frame through the always-listen state machine.

        Called from _processing_loop. MUST NOT block.
        """
        if not self._always_listen_enabled or self._al_state == "disabled":
            return

        # Suppress during TTS playback
        if self._tts_is_speaking is not None and self._tts_is_speaking():
            if self._al_state == "accumulating":
                self._al_cancel("TTS started playing")
            return

        # Cooldown check
        if self._al_state == "cooldown":
            if time.monotonic() >= self._al_cooldown_until:
                self._al_state = "idle"
            return

        energy = float(np.sqrt(np.mean(frame ** 2)))

        if self._al_state == "idle":
            if energy >= self._al_energy_threshold:
                # Post-TTS cooldown: don't start accumulating too soon after TTS
                if (self._al_last_tts_stop_time > 0 and
                        (time.monotonic() - self._al_last_tts_stop_time) < self._al_cooldown_after_tts):
                    return
                # Concurrent thread limit
                with self._al_active_threads_lock:
                    if self._al_active_threads >= 1:
                        return
                # Transition to accumulating
                self._al_state = "accumulating"
                self._al_buffer = [frame.copy()]
                self._al_speech_start_time = time.monotonic()
                self._al_silence_counter = 0.0
                self._al_last_check_time = time.monotonic()
                logger.debug("Always-listen: speech onset (energy=%.4f)", energy)

        elif self._al_state == "accumulating":
            now = time.monotonic()
            elapsed = now - self._al_speech_start_time

            # Timeout
            if elapsed >= self._al_timeout:
                logger.info("Always-listen: timeout (%.1fs), dispatching", elapsed)
                self._al_dispatch()
                return

            # Check silence periodically (~every 0.3s worth of frames)
            if (now - self._al_last_check_time) >= 0.3:
                self._al_last_check_time = now
                if energy < self._al_energy_threshold:
                    self._al_silence_counter += 0.3
                    if self._al_silence_counter >= self._al_silence_duration:
                        logger.info("Always-listen: silence after %.1fs, dispatching",
                                    elapsed)
                        self._al_dispatch()
                else:
                    self._al_silence_counter = 0.0

    def _al_dispatch(self) -> None:
        """Dispatch accumulated audio for transcription + classification.

        Non-blocking: spawns a daemon thread. Sets state to cooldown.
        """
        audio_chunks = self._al_buffer
        self._al_buffer = []
        self._al_state = "cooldown"
        self._al_cooldown_until = time.monotonic() + self._al_cooldown_after_response

        if not audio_chunks:
            self._al_state = "idle"
            return

        audio = np.concatenate(audio_chunks)
        duration = len(audio) / SAMPLE_RATE

        if duration < self._al_min_speech_duration:
            logger.debug("Always-listen: too short (%.2fs), skipping", duration)
            self._al_state = "idle"
            return

        with self._al_active_threads_lock:
            self._al_active_threads += 1

        threading.Thread(
            target=self._al_transcribe_and_classify,
            args=(audio,),
            daemon=True,
            name="al-transcribe",
        ).start()

    def _al_transcribe_and_classify(self, audio: np.ndarray) -> None:
        """Background thread: transcribe audio, apply echo rejection, fire callback."""
        try:
            # Pre-check: skip Whisper if average energy is too low (mostly silence)
            avg_energy = float(np.sqrt(np.mean(audio ** 2)))
            if avg_energy < self._al_energy_threshold * 0.5:
                logger.debug("Always-listen: skipping transcription (avg energy %.4f < %.4f)",
                            avg_energy, self._al_energy_threshold * 0.5)
                return

            text, lang = self._transcribe(audio)
            if not text or not text.strip():
                return

            # Filter known Whisper hallucinations (phantom phrases on near-silent audio)
            if text.strip().lower().rstrip(".!?,") in WHISPER_HALLUCINATIONS:
                logger.info("Always-listen: filtered Whisper hallucination: '%s'", text)
                return

            # Filter prompt leaks: Whisper echoes its prompt on ambiguous audio
            import re
            text_words = set(re.sub(r'[^\w\s]', '', text.lower()).split())
            if text_words and text_words.issubset(WHISPER_PROMPT_WORDS):
                logger.info("Always-listen: filtered prompt leak: '%s'", text)
                return

            # Echo rejection: check for TTS echo contamination
            if self._get_last_tts_text is not None:
                from utils.echo_cancel import strip_echo_prefix, echo_text_overlap
                last_tts = self._get_last_tts_text()
                if last_tts:
                    overlap = echo_text_overlap(text, last_tts)
                    if overlap > 0.6:
                        logger.info("Always-listen: echo rejected (%.0f%% overlap): '%s'",
                                    overlap * 100, text)
                        return
                    cleaned = strip_echo_prefix(text, last_tts)
                    if not cleaned:
                        logger.info("Always-listen: echo rejected (all prefix): '%s'", text)
                        return
                    if cleaned != text:
                        text = cleaned

            logger.info("Always-listen transcription: '%s'", text)

            # Fire ambient callback for intent classification in main.py
            if self._on_ambient_transcription:
                self._on_ambient_transcription(text, lang)
            elif self._on_transcription:
                # Fallback: route directly if no ambient callback
                threading.Thread(
                    target=self._on_transcription,
                    args=(text, lang),
                    daemon=True,
                    name="transcription-handler",
                ).start()

        except Exception as e:
            logger.error("Always-listen error: %s", e, exc_info=True)
        finally:
            with self._al_active_threads_lock:
                self._al_active_threads = max(0, self._al_active_threads - 1)

    def _al_cancel(self, reason: str) -> None:
        """Cancel in-progress always-listen accumulation."""
        if self._al_state == "accumulating":
            logger.debug("Always-listen: cancelled (%s)", reason)
        self._al_buffer = []
        self._al_state = "idle"
        self._al_silence_counter = 0.0

    def set_always_listen(self, enabled: bool) -> None:
        """Enable or disable always-listening at runtime."""
        if enabled and not self._al_config_enabled:
            logger.warning("Always-listen not available (disabled in config)")
            return
        self._always_listen_enabled = enabled
        if enabled:
            self._al_state = "idle"
            logger.info("Always-listen enabled")
        else:
            self._al_cancel("disabled by user")
            self._al_state = "disabled"
            logger.info("Always-listen disabled")

    @property
    def always_listen_active(self) -> bool:
        return self._always_listen_enabled and self._al_state in ("idle", "accumulating")

    @property
    def always_listen_state(self) -> str:
        return self._al_state

    def stop(self):
        """Stop audio stream and cleanup."""
        self._running = False
        self._is_listening = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Audio listener stopped")
