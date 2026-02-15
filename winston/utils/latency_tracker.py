"""Latency instrumentation for end-to-end voice interaction pipeline.

Tracks timing marks across threads using interaction IDs.
Singleton pattern — modules call LatencyTracker.get().mark(...).

Event timeline for one interaction:
    speech_end → stt_done → intent_done → context_ready → llm_done → tts_dequeued → audio_plays

Computed segments:
    stt:       speech_end → stt_done       (Groq ~200-600ms, local ~1-3s)
    intent:    stt_done → intent_done      (0-300ms)
    context:   intent_done → context_ready (50-200ms)
    llm:       context_ready → llm_done    (300-800ms)
    tts_queue: llm_done → tts_dequeued     (0-500ms)
    tts_synth: tts_dequeued → audio_plays  (200-400ms)
    e2e:       speech_end → audio_plays    (750-2350ms)
"""

import logging
import statistics
import threading
import time
from collections import deque
from typing import Optional

logger = logging.getLogger("winston.latency")

# Ordered segment definitions: (name, start_event, end_event)
SEGMENTS = [
    ("stt", "speech_end", "stt_done"),
    ("intent", "stt_done", "intent_done"),
    ("context", "intent_done", "context_ready"),
    ("llm", "context_ready", "llm_done"),
    ("tts_queue", "llm_done", "tts_dequeued"),
    ("tts_synth", "tts_dequeued", "audio_plays"),
    ("e2e", "speech_end", "audio_plays"),
]


class InteractionTrace:
    """One complete interaction: speech_end → audio_plays."""

    __slots__ = ("id", "marks", "created_at")

    def __init__(self, interaction_id: str):
        self.id = interaction_id
        self.marks: dict[str, float] = {}
        self.created_at = time.monotonic()

    def mark(self, event: str) -> None:
        self.marks[event] = time.monotonic()

    def elapsed(self, start: str, end: str) -> Optional[float]:
        """Return elapsed time in ms between two marks, or None if either is missing."""
        s = self.marks.get(start)
        e = self.marks.get(end)
        if s is not None and e is not None:
            return (e - s) * 1000
        return None

    def summary(self) -> dict:
        """Return all segment durations in ms."""
        segments = {}
        for name, start, end in SEGMENTS:
            val = self.elapsed(start, end)
            if val is not None:
                segments[name] = round(val, 1)
        return segments


class LatencyTracker:
    """Singleton latency tracker. Thread-safe."""

    _instance: Optional["LatencyTracker"] = None
    _init_lock = threading.Lock()

    def __init__(self):
        self._traces_lock = threading.Lock()
        self._active: dict[str, InteractionTrace] = {}
        self._completed: deque[dict] = deque(maxlen=100)

        self._stats_lock = threading.Lock()
        self._segment_buffers: dict[str, deque] = {name: deque(maxlen=100) for name, _, _ in SEGMENTS}
        self._last_e2e: float = 0.0
        self._interaction_count: int = 0

    @classmethod
    def get(cls) -> "LatencyTracker":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def begin(self, interaction_id: str) -> None:
        """Start tracking a new interaction."""
        with self._traces_lock:
            self._active[interaction_id] = InteractionTrace(interaction_id)

    def mark(self, interaction_id: str, event: str) -> None:
        """Record a timestamp for a named event."""
        with self._traces_lock:
            trace = self._active.get(interaction_id)
        if trace:
            trace.mark(event)

    def finish(self, interaction_id: str) -> Optional[dict]:
        """Compute deltas, store in history, log breakdown. Return the measurement."""
        with self._traces_lock:
            trace = self._active.pop(interaction_id, None)
        if not trace:
            return None

        summary = trace.summary()
        self._completed.append(summary)

        with self._stats_lock:
            self._interaction_count += 1
            for name, buf in self._segment_buffers.items():
                if name in summary:
                    buf.append(summary[name])
            if "e2e" in summary:
                self._last_e2e = summary["e2e"]

        parts = " | ".join(f"{k}={v:.0f}ms" for k, v in summary.items())
        logger.info("[latency] %s: %s", interaction_id[:8], parts)

        return summary

    def discard(self, interaction_id: str) -> None:
        """Discard a trace without finishing (e.g. speech not addressed to Winston)."""
        if not interaction_id:
            return
        with self._traces_lock:
            self._active.pop(interaction_id, None)

    def get_stats(self) -> dict:
        """Return p50, p95, avg, count for each tracked segment."""
        with self._stats_lock:
            result = {}
            for name, buf in self._segment_buffers.items():
                if buf:
                    sorted_vals = sorted(buf)
                    n = len(sorted_vals)
                    result[name] = {
                        "avg": round(statistics.mean(sorted_vals)),
                        "p50": round(sorted_vals[n // 2]),
                        "p95": round(sorted_vals[min(int(n * 0.95), n - 1)]),
                        "count": n,
                    }
            result["last_e2e"] = round(self._last_e2e)
            result["interaction_count"] = self._interaction_count
            return result

    def cleanup_stale(self, max_age_seconds: float = 60.0) -> None:
        """Remove traces that never finished (errors, timeouts, etc.)."""
        now = time.monotonic()
        with self._traces_lock:
            stale = [k for k, v in self._active.items() if (now - v.created_at) > max_age_seconds]
            for k in stale:
                del self._active[k]
                logger.debug("[latency] Cleaned stale trace: %s", k[:8])
