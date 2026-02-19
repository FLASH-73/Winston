"""Tests for utils/echo_cancel.py — barge-in detection and echo text rejection."""

import threading
import time

import numpy as np
from utils.echo_cancel import EnergyBargeInDetector, echo_text_overlap, strip_echo_prefix


class TestEnergyBargeInDetector:
    def test_inactive_by_default(self):
        detector = EnergyBargeInDetector()
        assert not detector.is_active
        # Should never trigger when inactive
        frame = np.ones(1280, dtype=np.float32) * 0.5
        assert detector.process(frame) is False

    def test_calibration_phase(self):
        detector = EnergyBargeInDetector(calibration_frames=3)
        detector.start_calibration()
        assert detector.is_active
        assert detector.is_calibrating

        # Feed 3 calibration frames (simulating TTS echo)
        echo_frame = np.ones(1280, dtype=np.float32) * 0.05
        for _ in range(3):
            result = detector.process(echo_frame)
            assert result is False  # Never triggers during calibration

        assert not detector.is_calibrating  # Calibration done

    def test_triggers_on_consecutive_loud_frames(self):
        detector = EnergyBargeInDetector(
            threshold_factor=2.0,
            consecutive_trigger=3,
            calibration_frames=3,
            min_energy=0.01,
        )
        detector.start_calibration()

        # Calibrate with quiet echo
        echo_frame = np.ones(1280, dtype=np.float32) * 0.02
        for _ in range(3):
            detector.process(echo_frame)

        # Now send loud frames (user speaking over TTS)
        loud_frame = np.ones(1280, dtype=np.float32) * 0.2
        assert detector.process(loud_frame) is False  # 1/3
        assert detector.process(loud_frame) is False  # 2/3
        assert detector.process(loud_frame) is True  # 3/3 — triggered!

    def test_resets_consecutive_on_quiet_frame(self):
        detector = EnergyBargeInDetector(
            threshold_factor=2.0,
            consecutive_trigger=3,
            calibration_frames=2,
            min_energy=0.01,
        )
        detector.start_calibration()

        echo_frame = np.ones(1280, dtype=np.float32) * 0.02
        for _ in range(2):
            detector.process(echo_frame)

        loud = np.ones(1280, dtype=np.float32) * 0.2
        quiet = np.ones(1280, dtype=np.float32) * 0.01

        detector.process(loud)  # 1/3
        detector.process(loud)  # 2/3
        detector.process(quiet)  # reset!
        detector.process(loud)  # 1/3 again
        detector.process(loud)  # 2/3
        assert detector.process(loud) is True  # 3/3

    def test_min_energy_floor(self):
        detector = EnergyBargeInDetector(
            threshold_factor=2.0,
            calibration_frames=2,
            min_energy=0.1,
        )
        detector.start_calibration()

        # Very quiet calibration — threshold should be at min_energy floor
        silent = np.zeros(1280, dtype=np.float32)
        for _ in range(2):
            detector.process(silent)

        # Frame just below min_energy should not trigger
        below_floor = np.ones(1280, dtype=np.float32) * 0.08
        for _ in range(5):
            assert detector.process(below_floor) is False

    def test_calibration_sets_threshold_from_mean(self):
        detector = EnergyBargeInDetector(
            threshold_factor=2.0,
            calibration_frames=3,
            min_energy=0.01,
        )
        detector.start_calibration()

        # Feed 3 frames with different energies: 0.04, 0.06, 0.05
        # Mean = 0.05, threshold = 0.05 * 2.0 = 0.10
        detector.process(np.ones(1280, dtype=np.float32) * 0.04)
        detector.process(np.ones(1280, dtype=np.float32) * 0.06)
        detector.process(np.ones(1280, dtype=np.float32) * 0.05)

        assert not detector.is_calibrating
        assert abs(detector.threshold - 0.10) < 0.01

    def test_threshold_respects_min_energy(self):
        detector = EnergyBargeInDetector(
            threshold_factor=2.0,
            calibration_frames=2,
            min_energy=0.05,
        )
        detector.start_calibration()

        # Very quiet echo: mean=0.01, threshold would be 0.02 but floor is 0.05
        detector.process(np.ones(1280, dtype=np.float32) * 0.01)
        detector.process(np.ones(1280, dtype=np.float32) * 0.01)

        assert detector.threshold >= 0.05

    def test_no_trigger_during_calibration(self):
        detector = EnergyBargeInDetector(
            calibration_frames=5,
            consecutive_trigger=1,
            min_energy=0.001,
        )
        detector.start_calibration()

        # Even very loud frames during calibration should NOT trigger
        for _ in range(5):
            assert detector.process(np.ones(1280, dtype=np.float32) * 0.5) is False

    def test_reset_deactivates(self):
        detector = EnergyBargeInDetector()
        detector.start_calibration()
        assert detector.is_active
        detector.reset()
        assert not detector.is_active

    def test_concurrent_calibration_and_process(self):
        """start_calibration and process can be called from different threads."""
        detector = EnergyBargeInDetector(calibration_frames=5)
        errors = []

        def calibrate():
            try:
                for _ in range(100):
                    detector.start_calibration()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def process_frames():
            try:
                frame = np.ones(1280, dtype=np.float32) * 0.05
                for _ in range(1000):
                    detector.process(frame)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=calibrate)
        t2 = threading.Thread(target=process_frames)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0

    def test_concurrent_reset_and_process(self):
        """reset and process can be called from different threads."""
        detector = EnergyBargeInDetector(calibration_frames=3)
        errors = []

        def reset_loop():
            try:
                for _ in range(100):
                    detector.start_calibration()
                    time.sleep(0.002)
                    detector.reset()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def process_loop():
            try:
                frame = np.ones(1280, dtype=np.float32) * 0.05
                for _ in range(1000):
                    detector.process(frame)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=reset_loop)
        t2 = threading.Thread(target=process_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0


class TestStripEchoPrefix:
    def test_strips_tts_words_from_beginning(self):
        result = strip_echo_prefix("Yes how are you doing", "Yes of course")
        assert result == "how are you doing"

    def test_preserves_full_user_speech(self):
        result = strip_echo_prefix("open the browser", "sure thing")
        assert result == "open the browser"

    def test_empty_inputs(self):
        assert strip_echo_prefix("", "hello") == ""
        assert strip_echo_prefix("hello", "") == "hello"
        assert strip_echo_prefix(None, "hello") == ""

    def test_strips_leading_punctuation_after_removal(self):
        result = strip_echo_prefix("Hello, world, how are you", "Hello world")
        assert result == "how are you"

    def test_case_insensitive_matching(self):
        result = strip_echo_prefix("HELLO world goodbye", "hello world")
        assert result == "goodbye"


class TestEchoTextOverlap:
    def test_full_overlap(self):
        ratio = echo_text_overlap("yes of course", "yes of course I will")
        assert ratio == 1.0

    def test_no_overlap(self):
        ratio = echo_text_overlap("open browser", "yes of course")
        assert ratio == 0.0

    def test_partial_overlap(self):
        ratio = echo_text_overlap("yes please open it", "yes please")
        assert 0.4 < ratio < 0.6  # 2/4 words overlap

    def test_empty_inputs(self):
        assert echo_text_overlap("", "hello") == 0.0
        assert echo_text_overlap("hello", "") == 0.0
        assert echo_text_overlap("", "") == 0.0


class TestWhisperHallucinationFilter:
    """Tests for the WHISPER_HALLUCINATIONS set and prompt-leak detection."""

    def test_bilingual_in_hallucinations(self):
        from perception.audio import WHISPER_HALLUCINATIONS

        assert "bilingual" in WHISPER_HALLUCINATIONS
        assert "bilingual messages" in WHISPER_HALLUCINATIONS

    def test_common_hallucinations_present(self):
        from perception.audio import WHISPER_HALLUCINATIONS

        assert "thank you" in WHISPER_HALLUCINATIONS
        assert "thanks" in WHISPER_HALLUCINATIONS
        assert "you" in WHISPER_HALLUCINATIONS
        assert "the end" in WHISPER_HALLUCINATIONS

    def test_hallucination_check_strips_punctuation(self):
        """The hallucination check uses rstrip('.!?,') so 'Bilingual.' matches."""
        from perception.audio import WHISPER_HALLUCINATIONS

        text = "Bilingual."
        cleaned = text.strip().lower().rstrip(".!?,")
        assert cleaned in WHISPER_HALLUCINATIONS

    def test_prompt_leak_catches_bilingual(self):
        """WHISPER_PROMPT_WORDS subset check catches 'Bilingual' as prompt leak."""
        import re

        from perception.audio import WHISPER_PROMPT_WORDS

        text = "Bilingual."
        text_words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
        assert text_words.issubset(WHISPER_PROMPT_WORDS)

    def test_prompt_leak_catches_multi_word(self):
        """Multi-word prompt leaks like 'English and German' are caught."""
        import re

        from perception.audio import WHISPER_PROMPT_WORDS

        text = "English and German."
        text_words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
        assert text_words.issubset(WHISPER_PROMPT_WORDS)
