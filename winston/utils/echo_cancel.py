"""Energy-delta barge-in detection and echo text rejection.

During TTS playback, the microphone picks up the speaker output (echo).
Instead of using VAD (which can't distinguish user speech from echo),
we detect user speech as an energy spike ABOVE the steady echo baseline.

TTS echo → steady mic energy level.
User speaking over TTS → energy spike above that level.
"""

import logging
import time

import numpy as np

logger = logging.getLogger("winston.echo")


class EnergyBargeInDetector:
    """Detect user speech during TTS by monitoring energy spikes above echo baseline.

    Simple design: brief calibration sets a fixed threshold, then hard consecutive
    counting triggers barge-in. No EMA drift, no slow decay, no grace period.
    False positives are handled downstream by echo text rejection.

    Usage:
        detector = EnergyBargeInDetector()
        detector.start_calibration()  # Call when TTS starts
        triggered = detector.process(mic_frame)  # Call for each mic frame
        detector.reset()  # Call when TTS stops
    """

    def __init__(
        self,
        threshold_factor: float = 2.0,
        consecutive_trigger: int = 3,
        calibration_frames: int = 5,
        min_energy: float = 0.03,
    ):
        self._threshold_factor = threshold_factor
        self._consecutive_trigger = consecutive_trigger
        self._calibration_count = calibration_frames
        self._min_energy = min_energy  # absolute floor for threshold

        self._active: bool = False
        self._calibrating: bool = False
        self._frames_seen: int = 0
        self._peak_energy: float = 0.0
        self._threshold: float = 0.0
        self._consecutive_above: int = 0

    def start_calibration(self) -> None:
        """Begin calibration phase. Call when TTS starts playing."""
        self._calibrating = True
        self._active = True
        self._frames_seen = 0
        self._peak_energy = 0.0
        self._threshold = 0.0
        self._consecutive_above = 0

    def reset(self) -> None:
        """Reset detector. Call when TTS stops."""
        self._active = False
        self._calibrating = False
        self._consecutive_above = 0

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    def process(self, mic_frame: np.ndarray) -> bool:
        """Process a mic frame. Returns True if barge-in should trigger."""
        if not self._active:
            return False

        energy = float(np.sqrt(np.mean(mic_frame ** 2)))

        if self._calibrating:
            # Measure TTS echo level during calibration
            self._peak_energy = max(self._peak_energy, energy)
            self._frames_seen += 1
            if self._frames_seen >= self._calibration_count:
                self._calibrating = False
                # Threshold = 2x echo peak, but at least min_energy
                self._threshold = max(
                    self._peak_energy * self._threshold_factor,
                    self._min_energy,
                )
                logger.info(
                    "Barge-in ready: echo_peak=%.4f, threshold=%.4f",
                    self._peak_energy, self._threshold,
                )
            return False

        # Detection: is this frame above threshold?
        if energy > self._threshold:
            self._consecutive_above += 1
            if self._consecutive_above <= self._consecutive_trigger + 2:
                logger.debug(
                    "Barge-in energy: %.4f > %.4f (%d/%d)",
                    energy, self._threshold,
                    self._consecutive_above, self._consecutive_trigger,
                )
        else:
            self._consecutive_above = 0  # Hard reset — no slow decay

        if self._consecutive_above >= self._consecutive_trigger:
            logger.info(
                "BARGE-IN TRIGGERED (energy=%.4f, threshold=%.4f, ratio=%.1fx)",
                energy, self._threshold,
                energy / max(self._threshold, 1e-8),
            )
            self._consecutive_above = 0
            return True

        return False


def strip_echo_prefix(transcription: str, tts_text: str) -> str:
    """Remove leading words from transcription that match the TTS echo.

    When the mic picks up both TTS output ("Yes?") and user speech
    ("how are you?"), Whisper transcribes "Yes, how are you?".
    This strips the TTS words from the beginning of the transcription,
    returning only the user's actual speech.

    Returns the cleaned transcription with TTS prefix removed.
    If nothing remains after stripping, returns empty string.
    """
    if not transcription or not tts_text:
        return transcription or ""

    import re

    def _normalize_words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    trans_words_raw = transcription.split()
    trans_words_norm = _normalize_words(transcription)
    tts_words_norm = set(_normalize_words(tts_text))

    if not trans_words_norm or not tts_words_norm:
        return transcription

    # Find how many leading words in the transcription match TTS words
    strip_count = 0
    for word in trans_words_norm:
        if word in tts_words_norm:
            strip_count += 1
        else:
            break  # Stop at first non-matching word

    if strip_count == 0:
        return transcription

    # Strip that many words from the original (preserving casing)
    remaining = trans_words_raw[strip_count:]
    result = " ".join(remaining).strip()

    # Strip leading punctuation artifacts
    result = result.lstrip(",.;:!? ")

    if result:
        logger.info(
            "Echo prefix stripped (%d words): '%s' -> '%s'",
            strip_count, transcription, result,
        )

    return result


def echo_text_overlap(transcription: str, tts_text: str) -> float:
    """Compute word overlap ratio between a transcription and what TTS was saying.

    Returns a float 0.0-1.0 where 1.0 means the transcription is entirely
    contained in the TTS text (i.e., it's just echo).
    """
    if not transcription or not tts_text:
        return 0.0

    trans_words = set(transcription.lower().split())
    tts_words = set(tts_text.lower().split())

    if not trans_words:
        return 0.0

    overlap = trans_words & tts_words
    return len(overlap) / len(trans_words)
