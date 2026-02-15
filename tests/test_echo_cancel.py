"""Tests for utils/echo_cancel.py — barge-in detection and echo text rejection."""

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

    def test_reset_deactivates(self):
        detector = EnergyBargeInDetector()
        detector.start_calibration()
        assert detector.is_active
        detector.reset()
        assert not detector.is_active


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
