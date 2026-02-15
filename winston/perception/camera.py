import logging
import os
import platform
import subprocess
from collections import deque
from threading import Lock
from typing import Optional

import cv2
import numpy as np
from config import CAMERA_INDEX, FRAME_RESOLUTION, SCENE_CHANGE_THRESHOLD
from utils.frame_diff import compute_scene_change

logger = logging.getLogger("winston.camera")


def _get_camera_names() -> list[str]:
    """Get camera device names from macOS system_profiler.

    Returns a list of camera names in system order (matches AVFoundation indices).
    Returns empty list on non-macOS or on failure.
    """
    if platform.system() != "Darwin":
        return []
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        names = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            # Camera names are lines ending with ":" at 4-space indent
            # (not 8+ space = properties like "Model ID:")
            if stripped.endswith(":") and line.startswith("    ") and not line.startswith("        "):
                names.append(stripped.rstrip(":"))
        return names
    except Exception:
        return []


def list_cameras(max_index: int = 10) -> list[dict]:
    """Enumerate available cameras by testing indices 0 through max_index.

    Returns a list of dicts with keys: index, name, width, height, backend, channels, type.
    Detects color vs depth/IR cameras via test frame channel count.
    """
    names = _get_camera_names()
    cameras = []
    # Suppress OpenCV stderr spam ("camera failed to properly initialize") for missing indices
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    for i in range(max_index):
        os.dup2(devnull_fd, 2)  # Redirect stderr to /dev/null
        cap = cv2.VideoCapture(i)
        os.dup2(old_stderr, 2)  # Restore stderr
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            backend = cap.getBackendName()
            name = names[i] if i < len(names) else f"Camera {i}"

            # Detect color vs depth/IR via test frame
            ret, frame = cap.read()
            if ret and frame is not None:
                channels = frame.shape[2] if len(frame.shape) == 3 else 1
                cam_type = "Color" if channels == 3 else "Depth/IR"
            else:
                channels = 0
                cam_type = "Unknown"

            cameras.append(
                {
                    "index": i,
                    "name": name,
                    "width": w,
                    "height": h,
                    "backend": backend,
                    "channels": channels,
                    "type": cam_type,
                }
            )
            cap.release()
        else:
            cap.release()
            # Don't break on first failure — RealSense may have gaps between indices
            if not cameras:
                break  # Stop only if no cameras found yet (permission denied)
    os.close(devnull_fd)
    os.close(old_stderr)
    return cameras


def select_camera() -> int:
    """Interactively list cameras and let the user pick one. Returns the chosen index.

    Auto-selects the color camera when a mix of color and depth/IR cameras is found.
    """
    cameras = list_cameras()

    if not cameras:
        logger.warning("No cameras found. Falling back to index %d", CAMERA_INDEX)
        return CAMERA_INDEX

    # Auto-select the first color camera if available
    color_cameras = [c for c in cameras if c["type"] == "Color"]

    if len(cameras) == 1:
        cam = cameras[0]
        logger.info(
            "One camera found: %s (%s, index %d, %dx%d)",
            cam["name"],
            cam["type"],
            cam["index"],
            cam["width"],
            cam["height"],
        )
        return cam["index"]

    if color_cameras and len(color_cameras) == 1:
        cam = color_cameras[0]
        logger.info(
            "Auto-selected color camera: %s (index %d, %dx%d)", cam["name"], cam["index"], cam["width"], cam["height"]
        )
        return cam["index"]

    print("\n  Available cameras:")
    for idx, cam in enumerate(cameras):
        marker = " <-" if cam["type"] == "Color" else ""
        print(f"    [{idx}] {cam['name']}  —  {cam['width']}x{cam['height']}  ({cam['type']}){marker}")

    while True:
        try:
            choice = input(f"\n  Select camera [0-{len(cameras) - 1}]: ").strip()
            choice_int = int(choice)
            if 0 <= choice_int < len(cameras):
                return cameras[choice_int]["index"]
        except (ValueError, EOFError):
            pass
        print("  Invalid choice. Try again.")


class Camera:
    def __init__(self, resolution: tuple = FRAME_RESOLUTION):
        self._cap: Optional[cv2.VideoCapture] = None
        self._resolution = resolution
        self._lock = Lock()
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=5)
        self._previous_frame: Optional[np.ndarray] = None
        self._source = None  # Stored for reconnection (str URL or int index)

    def start(self, source=None) -> bool:
        """Open a camera source.

        Args:
            source: Camera source — one of:
                - None: interactive selection (default)
                - int: local camera index (e.g. 0)
                - str: RTSP or MJPEG URL (e.g. "rtsp://192.168.1.50:8554/garage")

        Returns True on success.
        """
        if source is None:
            source = select_camera()

        self._source = source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            logger.error("Failed to open camera source: %s", source)
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])

        # Verify with a test read
        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.error("Camera opened but failed to capture a test frame")
            self._cap.release()
            self._cap = None
            return False

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if isinstance(source, str):
            logger.info("Remote camera ready: %s, %dx%d", source, actual_w, actual_h)
        else:
            logger.info("Camera ready: index %d, %dx%d", source, actual_w, actual_h)
        return True

    @property
    def is_remote(self) -> bool:
        """True if using a network stream (RTSP/MJPEG)."""
        return isinstance(self._source, str)

    def _reconnect(self) -> bool:
        """Attempt to reconnect to a network camera stream. Returns True on success."""
        if not self.is_remote:
            return False
        logger.warning("Reconnecting to remote camera: %s", self._source)
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self._source)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
            ret, _ = self._cap.read()
            if ret:
                logger.info("Reconnected to remote camera: %s", self._source)
                return True
        logger.warning("Reconnection failed: %s", self._source)
        return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Capture and return the current frame (BGR numpy array). Returns None on failure."""
        with self._lock:
            if not self.is_open:
                return None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # For remote streams, try reconnecting once
                if self.is_remote and self._reconnect():
                    ret, frame = self._cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to read frame from camera")
                    return None

            # Resize if larger than target resolution
            h, w = frame.shape[:2]
            max_w, max_h = self._resolution
            if w > max_w or h > max_h:
                frame = cv2.resize(frame, (max_w, max_h))

            self._frame_buffer.append(frame)
            return frame

    def get_frame_bytes(self, quality: int = 70) -> Optional[bytes]:
        """Capture frame and return as JPEG bytes for API calls."""
        frame = self.get_frame()
        if frame is None:
            return None
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            logger.warning("Failed to encode frame as JPEG")
            return None
        return buffer.tobytes()

    def scene_changed(self) -> bool:
        """Check if current frame differs significantly from previous analysis frame.

        Updates the previous frame reference when a change IS detected,
        so the baseline is always the last frame that triggered analysis.
        """
        if not self._frame_buffer:
            return False

        current = self._frame_buffer[-1]

        if self._previous_frame is None:
            self._previous_frame = current.copy()
            return True  # First frame always counts as a change

        score = compute_scene_change(current, self._previous_frame)
        if score >= SCENE_CHANGE_THRESHOLD:
            self._previous_frame = current.copy()
            return True

        return False

    def get_scene_change_score(self) -> float:
        """Return the raw change score (0.0-1.0) without updating previous frame."""
        if not self._frame_buffer or self._previous_frame is None:
            return 0.0
        return compute_scene_change(self._frame_buffer[-1], self._previous_frame)

    def stop(self) -> None:
        """Release the camera."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                logger.info("Camera released")

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
