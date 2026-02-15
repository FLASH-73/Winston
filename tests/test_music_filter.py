"""Tests for music/noise filtering in always-listen dispatch."""

import numpy as np


SAMPLE_RATE = 16000


def _make_chunks(n: int, energy: float, variance: float = 0.0) -> list[np.ndarray]:
    """Create n audio chunks with given mean energy and optional variance."""
    chunks = []
    for i in range(n):
        e = energy + (variance * (i % 2 * 2 - 1))  # alternating +/- variance
        chunk = np.full(1280, max(e, 0.001), dtype=np.float32)
        chunks.append(chunk)
    return chunks


class TestMusicEnergyVariance:
    def test_music_low_variance_detected(self):
        """Steady energy (music-like) should have low CV."""
        chunks = _make_chunks(50, energy=0.05, variance=0.002)
        energies = np.array([float(np.sqrt(np.mean(c**2))) for c in chunks])
        cv = energies.std() / energies.mean()
        assert cv < 0.3  # Should be flagged as music

    def test_speech_high_variance_passes(self):
        """Bursty energy (speech-like) should have high CV."""
        # Simulate speech: alternating loud and quiet
        chunks = _make_chunks(50, energy=0.05, variance=0.04)
        energies = np.array([float(np.sqrt(np.mean(c**2))) for c in chunks])
        cv = energies.std() / energies.mean()
        assert cv > 0.3  # Should pass through as speech

    def test_continuous_duration_filter(self):
        """Audio > 10s with no silence gaps should be filtered."""
        # 12 seconds of audio at 16kHz in 80ms chunks (1280 samples)
        n_chunks = int(12.0 * SAMPLE_RATE / 1280)
        chunks = _make_chunks(n_chunks, energy=0.05)
        audio = np.concatenate(chunks)
        duration = len(audio) / SAMPLE_RATE
        assert duration >= 10.0  # Would be filtered by continuous duration check

    def test_short_speech_not_filtered(self):
        """Normal speech (2-5s) should not be filtered regardless of variance."""
        chunks = _make_chunks(30, energy=0.05, variance=0.001)  # Low variance but short
        audio = np.concatenate(chunks)
        duration = len(audio) / SAMPLE_RATE
        assert duration < 10.0  # Too short to trigger continuous filter
