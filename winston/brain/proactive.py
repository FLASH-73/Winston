import logging
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from config import (
    PROACTIVE_INTERVAL,
    PROACTIVE_COOLDOWN,
    PROACTIVE_USEFULNESS_THRESHOLD,
    SYSTEM_PROMPT_PROACTIVE,
    SCENE_CHANGE_THRESHOLD,
)
from utils.frame_diff import compute_scene_change

logger = logging.getLogger("winston.proactive")


class ProactiveEngine:
    def __init__(self, camera, llm, memory, tts):
        self._camera = camera
        self._llm = llm
        self._memory = memory
        self._tts = tts

        self._proactive_interval = PROACTIVE_INTERVAL
        self._last_check_time = 0.0
        self._last_speak_time = 0.0
        self._recent_observations: deque[str] = deque(maxlen=10)
        self._last_proactive_frame: Optional[np.ndarray] = None

    def should_check(self) -> bool:
        """Return True if enough time has passed since last proactive check."""
        return (time.time() - self._last_check_time) >= self._proactive_interval

    def check(self, frame_bytes: bytes, recent_context: Optional[str] = None) -> Optional[str]:
        """Run a proactive observation check.

        Args:
            recent_context: Pre-assembled context string from Memory.assemble_context().

        Returns the spoken message if Winston decided to speak, None otherwise.
        """
        self._last_check_time = time.time()

        # Scene change gate: decode the JPEG to compare with last proactive frame
        frame_array = cv2.imdecode(
            np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame_array is None:
            return None

        if self._last_proactive_frame is not None:
            score = compute_scene_change(frame_array, self._last_proactive_frame)
            if score < SCENE_CHANGE_THRESHOLD:
                logger.debug("No significant scene change for proactive check (score=%.3f)", score)
                return None

        self._last_proactive_frame = frame_array.copy()

        # Build prompt with context window
        context_parts = []
        if self._recent_observations:
            context_parts.append("Recent observations:")
            for obs in self._recent_observations:
                context_parts.append(f"- {obs}")

        if recent_context:
            context_parts.append(f"\n{recent_context}")

        context_parts.append("\nBased on the context and what you see now, should you say something?")
        prompt = "\n".join(context_parts)

        # Call Claude with proactive prompt
        result = self._llm.analyze_frame(
            frame_bytes,
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_PROACTIVE,
            use_pro=False,
            max_output_tokens=200,
        )

        if result is None:
            return None

        # Store observation regardless of whether we speak
        scene_desc = result.get("reasoning", result.get("message", ""))
        if scene_desc:
            self._recent_observations.append(scene_desc)

        # Check if Winston should speak
        should_speak = result.get("should_speak", False)
        usefulness = result.get("usefulness_score", 0)
        message = result.get("message", "")

        if not should_speak or usefulness < PROACTIVE_USEFULNESS_THRESHOLD:
            logger.debug("Proactive check: not speaking (score=%d, should_speak=%s)",
                         usefulness, should_speak)
            return None

        if not self._cooldown_passed():
            logger.debug("Proactive check: cooldown active, suppressing speech")
            return None

        # Speak!
        logger.info("Proactive observation (score=%d): %s", usefulness, message)
        self._tts.speak_async(message)
        self._memory.store(
            f"[Proactive] {message}",
            entry_type="observation",
            activity=result.get("reasoning", ""),
        )
        self._last_speak_time = time.time()
        return message

    def _cooldown_passed(self) -> bool:
        """Check if enough time has passed since last unsolicited speech."""
        return (time.time() - self._last_speak_time) >= PROACTIVE_COOLDOWN

    def update_intervals(self, proactive_interval: float):
        """Allow main loop to adjust intervals (e.g., when budget is low)."""
        self._proactive_interval = proactive_interval
        logger.info("Proactive interval updated to %.0fs", proactive_interval)
