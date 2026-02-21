"""Circular frame buffer for workshop camera snapshots, clips, and timelapses.

Stores JPEG bytes only (not numpy arrays) to keep memory usage low (~60MB total).
Frames are decoded to numpy on-the-fly when rendering video.
"""

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from config import (
    CLIP_BUFFER_FPS,
    CLIP_BUFFER_SECONDS,
    CLIP_DEFAULT_DURATION,
    CLIP_MAX_DURATION,
    CLIP_OUTPUT_DIR,
    TIMELAPSE_MAX_HOURS,
    TIMELAPSE_OUTPUT_FPS,
    TIMELAPSE_WINDOW_HOURS,
)

logger = logging.getLogger(__name__)


@dataclass
class TimestampedFrame:
    timestamp: float
    jpeg_bytes: bytes


class ClipRecorder:
    """Captures frames from camera into circular buffers for clip/timelapse generation."""

    def __init__(self, camera, buffer_seconds: int = CLIP_BUFFER_SECONDS,
                 buffer_fps: float = CLIP_BUFFER_FPS):
        self._camera = camera
        self._buffer_seconds = buffer_seconds
        self._buffer_fps = buffer_fps
        self._min_interval = 1.0 / buffer_fps

        # Clip buffer — recent frames at full capture rate
        self._buffer: deque[TimestampedFrame] = deque()
        self._lock = threading.Lock()

        # Timelapse buffer — 1 frame per 30s, up to TIMELAPSE_MAX_HOURS
        self._timelapse_buffer: deque[TimestampedFrame] = deque(
            maxlen=int(TIMELAPSE_MAX_HOURS * 3600 / 30)
        )
        self._last_timelapse_time = 0.0

        # Thread control
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        os.makedirs(CLIP_OUTPUT_DIR, exist_ok=True)

    def start(self):
        """Start the capture thread."""
        self._cleanup_old_clips()
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="clip-recorder"
        )
        self._thread.start()
        logger.info("[clip] ClipRecorder started (%.1f fps, %ds buffer)", self._buffer_fps, self._buffer_seconds)

    def stop(self):
        """Stop the capture thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("[clip] ClipRecorder stopped")

    def _capture_loop(self):
        """Grab frames from camera at configured FPS."""
        cleanup_interval = 3600  # cleanup every hour
        last_cleanup = time.time()

        while not self._stop_event.is_set():
            try:
                jpeg = self._camera.get_frame_bytes()
                if jpeg:
                    now = time.time()
                    frame = TimestampedFrame(timestamp=now, jpeg_bytes=jpeg)

                    with self._lock:
                        self._buffer.append(frame)

                        # Evict old frames
                        cutoff = now - self._buffer_seconds
                        while self._buffer and self._buffer[0].timestamp < cutoff:
                            self._buffer.popleft()

                        # Timelapse: 1 frame per 30s
                        if now - self._last_timelapse_time >= 30.0:
                            self._timelapse_buffer.append(frame)
                            self._last_timelapse_time = now

                    # Periodic cleanup
                    if now - last_cleanup > cleanup_interval:
                        self._cleanup_old_clips()
                        last_cleanup = now

            except Exception as e:
                logger.error("[clip] Capture error: %s", e, exc_info=True)

            self._stop_event.wait(timeout=self._min_interval)

    def get_snapshot(self) -> bytes | None:
        """Return the most recent JPEG frame."""
        with self._lock:
            if self._buffer:
                return self._buffer[-1].jpeg_bytes
        return None

    def get_clip_frames(self, duration: float = CLIP_DEFAULT_DURATION) -> list[TimestampedFrame]:
        """Return frames from the last `duration` seconds."""
        duration = min(duration, CLIP_MAX_DURATION)
        cutoff = time.time() - duration
        with self._lock:
            return [f for f in self._buffer if f.timestamp >= cutoff]

    def get_timelapse_frames(self, hours: float = TIMELAPSE_WINDOW_HOURS) -> list[TimestampedFrame]:
        """Return sparse frames for timelapse over the given window."""
        hours = min(hours, TIMELAPSE_MAX_HOURS)
        cutoff = time.time() - (hours * 3600)
        with self._lock:
            return [f for f in self._timelapse_buffer if f.timestamp >= cutoff]

    def render_video(self, frames: list[TimestampedFrame], output_fps: int = 15,
                     filename: str | None = None) -> str | None:
        """Stitch frames into an MP4 file. Returns file path or None."""
        if not frames or len(frames) < 2:
            return None

        filename = filename or f"clip_{int(time.time())}.mp4"
        output_path = os.path.join(CLIP_OUTPUT_DIR, filename)

        # Decode first frame to get dimensions
        first = cv2.imdecode(np.frombuffer(frames[0].jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            logger.error("[clip] Failed to decode first frame")
            return None

        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))

        if not writer.isOpened():
            logger.error("[clip] Failed to open VideoWriter")
            return None

        try:
            writer.write(first)
            for f in frames[1:]:
                decoded = cv2.imdecode(np.frombuffer(f.jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    continue
                if decoded.shape[:2] != (h, w):
                    decoded = cv2.resize(decoded, (w, h))
                writer.write(decoded)
        finally:
            writer.release()

        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info("[clip] Rendered %d frames to %s (%.1f MB)", len(frames), output_path, size_mb)
            return output_path

        logger.warning("[clip] Video file too small or missing: %s", output_path)
        return None

    def render_clip(self, duration: float = CLIP_DEFAULT_DURATION) -> str | None:
        """Render a clip of the last N seconds."""
        frames = self.get_clip_frames(duration)
        if not frames:
            return None
        actual_fps = len(frames) / max(duration, 1.0)
        return self.render_video(frames, output_fps=max(int(actual_fps), 1))

    def render_timelapse(self, hours: float = TIMELAPSE_WINDOW_HOURS) -> str | None:
        """Render a timelapse of the last N hours."""
        frames = self.get_timelapse_frames(hours)
        if not frames:
            return None
        return self.render_video(
            frames,
            output_fps=TIMELAPSE_OUTPUT_FPS,
            filename=f"timelapse_{int(hours)}h_{int(time.time())}.mp4",
        )

    def _cleanup_old_clips(self):
        """Delete temp video files older than 1 hour."""
        try:
            cutoff = time.time() - 3600
            for name in os.listdir(CLIP_OUTPUT_DIR):
                path = os.path.join(CLIP_OUTPUT_DIR, name)
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.unlink(path)
                    logger.debug("[clip] Cleaned up old file: %s", name)
        except Exception as e:
            logger.error("[clip] Cleanup error: %s", e)
