"""
Visual Cortex — 24/7 Gemini-powered background scene observer.

Runs as a daemon thread, collecting frames at variable FPS (idle vs active),
batching them, and sending to Gemini 2.5 Flash Lite for temporal analysis.
Writes a rolling narrative to TemporalNarrative for the conversation brain.
"""

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

import cv2
import numpy as np
from config import (
    GEMINI_API_KEY,
    GEMINI_VISION_MODEL,
    VISUAL_CORTEX_ACTIVE_FPS,
    VISUAL_CORTEX_BATCH_INTERVAL,
    VISUAL_CORTEX_BATCH_SIZE,
    VISUAL_CORTEX_IDLE_FPS,
    VISUAL_CORTEX_MOTION_THRESHOLD,
    VISUAL_CORTEX_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from perception.audio import AudioListener
    from perception.camera import Camera

    from brain.temporal_memory import TemporalNarrative

logger = logging.getLogger("winston.visual_cortex")


class VisualCortex(threading.Thread):
    """24/7 Gemini-powered visual processing daemon."""

    def __init__(
        self,
        camera: "Camera",
        temporal_memory: "TemporalNarrative",
        on_anomaly: Optional[Callable[[str, Optional[bytes], int], None]] = None,
        audio: Optional["AudioListener"] = None,
    ):
        super().__init__(daemon=True, name="visual-cortex")
        self._camera = camera
        self._temporal_memory = temporal_memory
        self._on_anomaly = on_anomaly
        self._audio = audio
        self._stop_event = threading.Event()

        # Frame buffer for batching
        self._frame_buffer: deque[tuple[str, bytes]] = deque(maxlen=VISUAL_CORTEX_BATCH_SIZE)

        # Motion detection state
        self._prev_gray: Optional[np.ndarray] = None

        # Gemini client (initialized lazily in run())
        self._client = None

        # Backoff state
        self._backoff_until: float = 0.0

        # Batch interval (adjustable for presence-aware duty cycling)
        self._batch_interval: float = VISUAL_CORTEX_BATCH_INTERVAL

    def set_batch_interval_multiplier(self, multiplier: float):
        """Scale the Gemini batch interval (e.g. 3x slower when away)."""
        self._batch_interval = VISUAL_CORTEX_BATCH_INTERVAL * multiplier
        logger.info("Visual cortex batch interval: %.0fs (%.1fx)", self._batch_interval, multiplier)

    def stop(self) -> None:
        """Signal the thread to stop gracefully."""
        self._stop_event.set()
        logger.info("Visual cortex stop requested")

    def run(self) -> None:
        """Main loop: collect frames, detect motion, batch-analyze with Gemini."""
        logger.info("Visual cortex starting (model=%s)", GEMINI_VISION_MODEL)

        # Initialize Gemini client
        try:
            from google import genai

            self._client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("Gemini client initialized")
        except Exception as e:
            logger.error("Failed to initialize Gemini client: %s — visual cortex disabled", e)
            return

        last_batch_time = time.monotonic()

        while not self._stop_event.is_set():
            try:
                # Determine current FPS based on activity
                fps = self._get_current_fps()
                frame_interval = 1.0 / fps

                # Capture a frame
                frame_bytes = self._camera.get_frame_bytes()
                if frame_bytes is None:
                    # Camera unavailable — wait and retry
                    self._stop_event.wait(timeout=5.0)
                    continue

                # Local motion detection (updates FPS for next iteration)
                self._update_motion_state(frame_bytes)

                # Add frame to buffer
                timestamp = datetime.now().strftime("%H:%M:%S")
                self._frame_buffer.append((timestamp, frame_bytes))

                # Check if it's time to send a batch to Gemini
                elapsed = time.monotonic() - last_batch_time
                if elapsed >= self._batch_interval and len(self._frame_buffer) > 0:
                    self._analyze_batch()
                    last_batch_time = time.monotonic()

                # Sleep until next frame capture
                self._stop_event.wait(timeout=frame_interval)

            except Exception as e:
                logger.error("Visual cortex loop error: %s", e, exc_info=True)
                self._stop_event.wait(timeout=5.0)

        logger.info("Visual cortex stopped")

    def _get_current_fps(self) -> float:
        """Determine capture FPS based on motion and audio activity."""
        # Check audio activity
        if self._audio:
            try:
                state = self._audio.pipeline_state
                if state in ("recording", "speaking"):
                    return VISUAL_CORTEX_ACTIVE_FPS
            except Exception:
                pass

        # Check if we recently detected motion (stored in _prev_gray update)
        # Motion detection happens in _update_motion_state; for now use the
        # presence of recent frames as a proxy — if buffer is filling fast,
        # we're likely in active mode already.
        return VISUAL_CORTEX_IDLE_FPS

    def _update_motion_state(self, frame_bytes: bytes) -> bool:
        """Lightweight local motion detection using frame differencing.

        Returns True if motion detected.
        """
        try:
            # Decode JPEG to grayscale
            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                return False

            # Resize for fast comparison
            small = cv2.resize(frame, (320, 240))

            if self._prev_gray is None:
                self._prev_gray = small
                return False

            # Absolute difference
            diff = cv2.absdiff(self._prev_gray, small)
            changed_ratio = np.count_nonzero(diff > 25) / diff.size

            self._prev_gray = small

            motion_detected = changed_ratio > VISUAL_CORTEX_MOTION_THRESHOLD
            return motion_detected

        except Exception as e:
            logger.debug("Motion detection error: %s", e)
            return False

    def _analyze_batch(self) -> None:
        """Send buffered frames to Gemini for temporal analysis."""
        # Check backoff
        if time.monotonic() < self._backoff_until:
            return

        if not self._client:
            return

        # Snapshot the current buffer
        frames = list(self._frame_buffer)
        if not frames:
            return

        try:
            from google.genai import types

            # Build multipart content
            parts = []
            for i, (ts, frame_bytes) in enumerate(frames):
                parts.append(
                    types.Part(inline_data=types.Blob(data=frame_bytes, mime_type="image/jpeg"))
                )
                parts.append(types.Part(text=f"[Frame {i + 1}, {ts}]"))
            parts.append(
                types.Part(text="Analyze these sequential frames from a robotics workshop camera.")
            )

            response = self._client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=types.Content(parts=parts),
                config=types.GenerateContentConfig(
                    system_instruction=VISUAL_CORTEX_SYSTEM_PROMPT,
                    max_output_tokens=300,
                    temperature=0.1,
                ),
            )

            # Parse response
            result = self._parse_response(response.text)
            if result:
                narrative = result.get("narrative", "")
                if narrative:
                    latest_frame = frames[-1][1] if frames else None
                    self._temporal_memory.add_entry(narrative, frame_bytes=latest_frame)
                    logger.debug("Visual cortex: %s", narrative)

                # Check for anomalies
                anomaly = result.get("anomaly", {})
                if anomaly.get("detected") and anomaly.get("severity", 0) >= 7:
                    desc = anomaly.get("description", "Unknown anomaly")
                    logger.warning("Visual cortex anomaly (severity %d): %s", anomaly["severity"], desc)
                    self._temporal_memory.record_anomaly(desc)
                    if self._on_anomaly:
                        # Pass latest frame for Telegram notification
                        latest_frame = frames[-1][1] if frames else None
                        self._on_anomaly(desc, latest_frame, anomaly.get("severity", 7))

        except Exception as e:
            logger.warning("Gemini visual analysis failed: %s — backing off 60s", e)
            self._backoff_until = time.monotonic() + 60.0

    @staticmethod
    def _parse_response(text: str) -> Optional[dict]:
        """Parse Gemini JSON response, handling markdown fences."""
        if not text:
            return None

        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1 :]
            # Remove closing fence
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse visual cortex response: %s — text: %.200s", e, text)
            return None
