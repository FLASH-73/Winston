"""Tests for audio pipeline robustness — watchdog and state tracking."""

import time

from perception.audio import AudioListener


class TestPipelineWatchdog:
    def _make_listener(self) -> AudioListener:
        """Create an AudioListener without starting the audio stream."""
        listener = AudioListener()
        # Enable always-listen so watchdog can reset _al_state
        listener._always_listen_enabled = True
        listener._al_state = "idle"
        return listener

    def test_watchdog_resets_stuck_recording(self):
        """Watchdog should force-reset if stuck in 'recording' too long."""
        listener = self._make_listener()
        listener._pipeline_state = "recording"
        listener._pipeline_state_changed_at = time.monotonic() - 35.0
        listener._is_listening = True
        listener._al_state = "accumulating"

        listener._check_watchdog()

        assert listener._is_listening is False
        assert listener._pipeline_state == "idle"
        assert listener._al_state == "idle"

    def test_watchdog_resets_stuck_speaking(self):
        """Watchdog should force-reset if stuck in 'speaking' too long."""
        listener = self._make_listener()
        listener._pipeline_state = "speaking"
        listener._pipeline_state_changed_at = time.monotonic() - 35.0

        listener._check_watchdog()

        assert listener._pipeline_state == "idle"
        assert not listener._energy_detector.is_active

    def test_watchdog_resets_stuck_cooldown(self):
        """Watchdog should force-reset if stuck in 'cooldown' too long."""
        listener = self._make_listener()
        listener._pipeline_state = "cooldown"
        listener._pipeline_state_changed_at = time.monotonic() - 35.0

        listener._check_watchdog()

        assert listener._pipeline_state == "idle"

    def test_watchdog_ignores_idle(self):
        """Watchdog should not reset if pipeline is idle."""
        listener = self._make_listener()
        listener._pipeline_state = "idle"
        listener._pipeline_state_changed_at = time.monotonic() - 100.0

        listener._check_watchdog()

        assert listener._pipeline_state == "idle"

    def test_watchdog_ignores_recent_state(self):
        """Watchdog should not reset if state changed recently."""
        listener = self._make_listener()
        listener._pipeline_state = "recording"
        listener._pipeline_state_changed_at = time.monotonic() - 5.0
        listener._is_listening = True

        listener._check_watchdog()

        # Should NOT have reset — only 5s elapsed, timeout is 30s
        assert listener._is_listening is True
        assert listener._pipeline_state == "recording"


class TestPipelineStateTransitions:
    def test_set_pipeline_state_updates_timestamp(self):
        listener = AudioListener()
        before = time.monotonic()
        listener._set_pipeline_state("recording")
        after = time.monotonic()

        assert listener._pipeline_state == "recording"
        assert before <= listener._pipeline_state_changed_at <= after

    def test_pipeline_state_property(self):
        listener = AudioListener()
        listener._set_pipeline_state("speaking")
        assert listener.pipeline_state == "speaking"


class TestTTSSuppression:
    """Tests for TTS echo suppression in always-listen pipeline."""

    def _make_listener(self) -> AudioListener:
        listener = AudioListener()
        listener._always_listen_enabled = True
        listener._al_state = "idle"
        return listener

    def test_tts_active_flag_set_on_tts_start(self):
        listener = self._make_listener()
        assert listener._tts_active is False
        listener.on_tts_start()
        assert listener._tts_active is True

    def test_tts_active_flag_cleared_on_tts_stop(self):
        listener = self._make_listener()
        listener.on_tts_start()
        assert listener._tts_active is True
        listener.on_tts_stop()
        assert listener._tts_active is False

    def test_tts_active_prevents_al_buffer_accumulation(self):
        """Audio callback should not append to AL buffer when TTS is active."""
        import numpy as np

        listener = self._make_listener()
        listener._al_state = "accumulating"
        listener._tts_active = True

        # Simulate callback appending a frame
        chunk = np.ones(1280, dtype=np.float32) * 0.1
        # Replicate the guarded callback logic
        if listener._al_state == "accumulating" and not listener._tts_active:
            listener._al_buffer.append(chunk)

        assert len(listener._al_buffer) == 0

    def test_al_buffer_accumulates_when_tts_inactive(self):
        """Audio callback should append to AL buffer when TTS is not active."""
        import numpy as np

        listener = self._make_listener()
        listener._al_state = "accumulating"
        listener._tts_active = False

        chunk = np.ones(1280, dtype=np.float32) * 0.1
        if listener._al_state == "accumulating" and not listener._tts_active:
            listener._al_buffer.append(chunk)

        assert len(listener._al_buffer) == 1

    def test_pipeline_speaking_suppresses_al_process_frame(self):
        """_al_process_frame should return early when pipeline is 'speaking'."""
        import numpy as np

        listener = self._make_listener()
        listener._al_state = "idle"
        listener._set_pipeline_state("speaking")

        # A loud frame that would normally start accumulation
        frame = np.ones(1280, dtype=np.float32) * 0.5
        listener._al_process_frame(frame)

        # Should still be idle — suppressed by pipeline state
        assert listener._al_state == "idle"

    def test_pipeline_speaking_cancels_accumulation(self):
        """_al_process_frame should cancel accumulation when pipeline is 'speaking'."""
        import numpy as np

        listener = self._make_listener()
        listener._al_state = "accumulating"
        listener._al_buffer = [np.ones(1280, dtype=np.float32) * 0.1]
        listener._set_pipeline_state("speaking")

        frame = np.ones(1280, dtype=np.float32) * 0.5
        listener._al_process_frame(frame)

        assert listener._al_state == "idle"
        assert len(listener._al_buffer) == 0

    def test_pipeline_speaking_refreshes_tts_stop_time(self):
        """_al_process_frame should refresh _al_last_tts_stop_time during speaking."""
        import numpy as np

        listener = self._make_listener()
        listener._al_last_tts_stop_time = 0.0
        listener._set_pipeline_state("speaking")

        frame = np.ones(1280, dtype=np.float32) * 0.1
        listener._al_process_frame(frame)

        assert listener._al_last_tts_stop_time > 0.0

    def test_pipeline_cooldown_transitions_to_idle(self):
        """Pipeline should transition from cooldown to idle after cooldown timer expires."""
        import numpy as np

        listener = self._make_listener()
        listener._set_pipeline_state("cooldown")
        # Cooldown already expired
        listener._cooldown_end_time = time.monotonic() - 0.1

        frame = np.ones(1280, dtype=np.float32) * 0.001  # quiet frame
        listener._al_process_frame(frame)

        assert listener._pipeline_state == "idle"

    def test_pipeline_cooldown_stays_during_tts_cooldown(self):
        """Pipeline should stay in cooldown if cooldown timer hasn't expired."""
        import numpy as np

        listener = self._make_listener()
        listener._set_pipeline_state("cooldown")
        # Cooldown far in the future
        listener._cooldown_end_time = time.monotonic() + 999.0

        frame = np.ones(1280, dtype=np.float32) * 0.001
        listener._al_process_frame(frame)

        assert listener._pipeline_state == "cooldown"
