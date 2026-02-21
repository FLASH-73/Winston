"""
Temporal Narrative Memory — rolling activity log from visual cortex.

Thread-safe storage of timestamped activity entries. The visual cortex writes
entries every ~45s; the conversation brain reads them for context.

Older entries (> 30 min) are periodically compressed into paragraph summaries
by an external timer calling summarize_old_entries(), keeping the narrative
compact while preserving temporal continuity.
"""

import json
import logging
import os
import tempfile
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

from config import (
    TEMPORAL_NARRATIVE_FILE,
    TEMPORAL_NARRATIVE_MAX_FILE_KB,
    TEMPORAL_NARRATIVE_RECENT_THRESHOLD_MINUTES,
    TEMPORAL_NARRATIVE_WINDOW_HOURS,
)

logger = logging.getLogger("winston.temporal_memory")


class TemporalNarrative:
    """Rolling temporal narrative of workshop activity observed by the visual cortex."""

    def __init__(self, max_entries: int = 500):
        self._entries: deque[tuple[datetime, str]] = deque(maxlen=max_entries)
        self._summaries: list[tuple[datetime, datetime, str]] = []  # (start, end, text)
        self._frame_snapshots: deque[tuple[datetime, bytes]] = deque(maxlen=10)
        self._lock = threading.Lock()
        self.last_anomaly_time: Optional[datetime] = None
        self.last_anomaly_text: str = ""

    def add_entry(self, text: str, frame_bytes: Optional[bytes] = None) -> None:
        """Append a narrative entry with the current timestamp.

        If frame_bytes is provided, stores the frame for later retrieval
        by the conversation brain (multi-frame visual context).
        """
        now = datetime.now()
        with self._lock:
            self._entries.append((now, text))
            if frame_bytes is not None:
                self._frame_snapshots.append((now, frame_bytes))
            self._prune()

    def get_narrative(self, hours: float = 2.0) -> str:
        """Return formatted narrative combining summaries and recent raw entries.

        Summaries cover older periods (> 30 min ago) as compressed paragraphs.
        Recent entries (< 30 min) are individual timestamped lines for precision.

        Example: "[14:00-14:30] Roberto worked on motor assembly, tested servo
        connections. [14:45] Switched to soldering PCB. [14:48] Using fine-tip iron."
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_threshold = datetime.now() - timedelta(
            minutes=TEMPORAL_NARRATIVE_RECENT_THRESHOLD_MINUTES
        )

        with self._lock:
            # Summaries that overlap with the requested window
            summary_parts = []
            for start, end, text in self._summaries:
                if end >= cutoff:
                    time_label = f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
                    summary_parts.append(f"[{time_label}] {text}")

            # Older raw entries not yet covered by summaries
            summarized_end = max(
                (e for _, e, _ in self._summaries), default=None
            )
            older_raw = []
            for ts, txt in self._entries:
                if ts >= cutoff and ts < recent_threshold:
                    if summarized_end is None or ts > summarized_end:
                        older_raw.append(f"[{ts.strftime('%H:%M')}] {txt}")

            # Recent raw entries (< threshold) — individual timestamped lines
            recent_parts = []
            for ts, txt in self._entries:
                if ts >= cutoff and ts >= recent_threshold:
                    recent_parts.append(f"[{ts.strftime('%H:%M')}] {txt}")

        parts = summary_parts + older_raw + recent_parts
        if not parts:
            return ""

        return " ".join(parts)

    def get_latest_state(self) -> str:
        """Return the most recent narrative entry text, or empty string."""
        with self._lock:
            if self._entries:
                return self._entries[-1][1]
            return ""

    def get_recent_frames(self, n: int = 3) -> list[tuple[datetime, bytes]]:
        """Return the last n stored frame snapshots as (timestamp, jpeg_bytes).

        Frames come from visual cortex Gemini analysis cycles (~45s apart).
        Returns newest-last ordering (chronological).
        """
        with self._lock:
            frames = list(self._frame_snapshots)
        return frames[-n:] if len(frames) >= n else frames

    def record_anomaly(self, description: str) -> None:
        """Record an anomaly detection event."""
        self.last_anomaly_time = datetime.now()
        self.last_anomaly_text = description

    def summarize_old_entries(self, llm_client) -> bool:
        """Compress entries older than the recent threshold into paragraph summaries.

        Uses Claude Haiku via llm_client.text_only_chat(). Called externally on a
        timer (every 15 min from main.py).

        Returns True if summarization occurred, False if nothing to summarize.
        """
        threshold = datetime.now() - timedelta(
            minutes=TEMPORAL_NARRATIVE_RECENT_THRESHOLD_MINUTES
        )

        with self._lock:
            old_entries = [(ts, txt) for ts, txt in self._entries if ts < threshold]
            if len(old_entries) < 5:
                return False

            lines = [f"[{ts.strftime('%H:%M')}] {txt}" for ts, txt in old_entries]
            period_start = old_entries[0][0]
            period_end = old_entries[-1][0]

        # LLM call outside the lock (network call, may take seconds)
        prompt = (
            "Compress these timestamped workshop observations into a brief paragraph "
            "(2-3 sentences). Preserve key timestamps and activities. Be concise.\n\n"
            + "\n".join(lines)
        )

        try:
            summary = llm_client.text_only_chat(prompt=prompt, max_tokens=150)
            if not summary:
                return False
        except Exception as e:
            logger.error("Temporal narrative summarization failed: %s", e)
            return False

        with self._lock:
            self._summaries.append((period_start, period_end, summary.strip()))

            # Remove old raw entries that were summarized
            current_threshold = datetime.now() - timedelta(
                minutes=TEMPORAL_NARRATIVE_RECENT_THRESHOLD_MINUTES
            )
            while self._entries and self._entries[0][0] < current_threshold:
                self._entries.popleft()

            # Prune summaries beyond the narrative window
            window_cutoff = datetime.now() - timedelta(
                hours=TEMPORAL_NARRATIVE_WINDOW_HOURS
            )
            self._summaries = [
                (s, e, t) for s, e, t in self._summaries if e >= window_cutoff
            ]

        logger.info(
            "Summarized %d entries (%s-%s): %.80s",
            len(lines),
            period_start.strftime("%H:%M"),
            period_end.strftime("%H:%M"),
            summary.strip(),
        )
        return True

    def get_session_end_snapshot(self, n: int = 5) -> list[tuple[datetime, str]]:
        """Return the last N text entries for session summary purposes."""
        with self._lock:
            entries = list(self._entries)
        return entries[-n:] if len(entries) >= n else entries

    def save_to_disk(self, file_path: str | None = None) -> bool:
        """Save current narrative state to JSON file. Called at shutdown.

        Persists entries, summaries, and a session_end_snapshot (last 5 entries).
        Does NOT persist _frame_snapshots (raw bytes).

        Returns True if saved successfully.
        """
        path = file_path or TEMPORAL_NARRATIVE_FILE

        with self._lock:
            entries = [
                {"ts": ts.isoformat(), "text": txt}
                for ts, txt in self._entries
            ]
            summaries = [
                {"start": s.isoformat(), "end": e.isoformat(), "text": t}
                for s, e, t in self._summaries
            ]
            snapshot = entries[-5:] if entries else []

        data = {
            "saved_at": datetime.now().isoformat(),
            "entries": entries,
            "summaries": summaries,
            "session_end_snapshot": snapshot,
        }

        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".json")
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
            logger.info(
                "Temporal narrative saved: %d entries, %d summaries",
                len(entries), len(summaries),
            )
            return True
        except Exception as e:
            logger.error("Failed to save temporal narrative: %s", e)
            return False

    def load_from_disk(self, file_path: str | None = None) -> dict:
        """Load previous narrative from JSON file. Called at startup.

        Filters entries to those within the narrative time window.
        Auto-truncates oldest entries if file exceeds size limit.

        Returns dict with loaded status and the session_end_snapshot from
        the previous session (for greeting generation).
        """
        path = file_path or TEMPORAL_NARRATIVE_FILE
        result = {
            "loaded": False,
            "session_end_snapshot": [],
            "saved_at": "",
            "entries_loaded": 0,
        }

        if not os.path.exists(path):
            return result

        try:
            file_size_kb = os.path.getsize(path) / 1024
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to load temporal narrative: %s", e)
            return result

        result["saved_at"] = data.get("saved_at", "")
        result["session_end_snapshot"] = data.get("session_end_snapshot", [])

        raw_entries = data.get("entries", [])
        raw_summaries = data.get("summaries", [])

        # Auto-truncate oldest entries if file was too large
        if file_size_kb > TEMPORAL_NARRATIVE_MAX_FILE_KB:
            keep = int(len(raw_entries) * TEMPORAL_NARRATIVE_MAX_FILE_KB / file_size_kb)
            raw_entries = raw_entries[-max(keep, 1):]
            logger.warning(
                "Temporal narrative file too large (%.0fKB), truncated to %d entries",
                file_size_kb, len(raw_entries),
            )

        cutoff = datetime.now() - timedelta(hours=TEMPORAL_NARRATIVE_WINDOW_HOURS)

        # Parse and filter entries
        valid_entries = []
        for e in raw_entries:
            try:
                ts = datetime.fromisoformat(e["ts"])
                if ts >= cutoff:
                    valid_entries.append((ts, e["text"]))
            except (KeyError, ValueError):
                continue

        # Parse and filter summaries
        valid_summaries = []
        for s in raw_summaries:
            try:
                start = datetime.fromisoformat(s["start"])
                end = datetime.fromisoformat(s["end"])
                if end >= cutoff:
                    valid_summaries.append((start, end, s["text"]))
            except (KeyError, ValueError):
                continue

        with self._lock:
            for entry in valid_entries:
                self._entries.append(entry)
            self._summaries.extend(valid_summaries)

        result["loaded"] = True
        result["entries_loaded"] = len(valid_entries)
        logger.info(
            "Temporal narrative loaded: %d entries, %d summaries (saved at %s)",
            len(valid_entries), len(valid_summaries), result["saved_at"],
        )
        return result

    def _prune(self) -> None:
        """Remove entries, summaries, and frames older than the configured window. Called under lock."""
        cutoff = datetime.now() - timedelta(hours=TEMPORAL_NARRATIVE_WINDOW_HOURS)
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()
        while self._frame_snapshots and self._frame_snapshots[0][0] < cutoff:
            self._frame_snapshots.popleft()
        self._summaries = [
            (s, e, t) for s, e, t in self._summaries if e >= cutoff
        ]
