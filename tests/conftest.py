"""Global test configuration — mock hardware modules before any imports.

This file runs before pytest collects tests. We patch sys.modules so that
imports of hardware-dependent libraries (cv2, sounddevice, etc.) return
MagicMock objects instead of failing on headless CI machines.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock hardware modules BEFORE any test file imports project code.
# These must be set in sys.modules at module level (not inside a fixture)
# because Python caches imports — once a real import fails, it's too late.
# ---------------------------------------------------------------------------

_HARDWARE_MODULES = [
    # OpenCV
    "cv2",
    # Audio I/O
    "sounddevice",
    # Speech-to-text (local)
    "faster_whisper",
    # Wake word
    "openwakeword",
    "openwakeword.model",
    "openwakeword.vad",
    # TTS fallback
    "pyttsx3",
    # ElevenLabs
    "elevenlabs",
    "elevenlabs.client",
    # ChromaDB
    "chromadb",
    "chromadb.config",
    # Groq
    "groq",
]

for _mod_name in _HARDWARE_MODULES:
    if _mod_name not in sys.modules:
        mock = MagicMock(spec=ModuleType)
        mock.__name__ = _mod_name
        mock.__path__ = []  # needed for sub-package mocks
        sys.modules[_mod_name] = mock
