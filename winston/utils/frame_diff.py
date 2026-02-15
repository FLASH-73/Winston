import cv2
import numpy as np


def compute_scene_change(
    frame_current: np.ndarray,
    frame_previous: np.ndarray,
    pixel_threshold: int = 30,
) -> float:
    """Compare two frames and return the fraction of pixels that changed significantly.

    Args:
        frame_current: Current BGR frame from OpenCV.
        frame_previous: Previous BGR frame from OpenCV.
        pixel_threshold: Minimum per-pixel absolute difference to count as "changed".

    Returns:
        Float between 0.0 and 1.0 representing fraction of changed pixels.
    """
    gray_current = cv2.cvtColor(frame_current, cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(frame_previous, cv2.COLOR_BGR2GRAY)

    # Resize to match if dimensions differ
    if gray_current.shape != gray_previous.shape:
        gray_previous = cv2.resize(gray_previous, (gray_current.shape[1], gray_current.shape[0]))

    diff = cv2.absdiff(gray_current, gray_previous)
    changed_pixels = np.sum(diff > pixel_threshold)
    total_pixels = diff.size

    return float(changed_pixels / total_pixels)
