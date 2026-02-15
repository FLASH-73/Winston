"""macOS Computer Use controller for Anthropic's Computer Use API.

Executes screenshot/mouse/keyboard actions on macOS using native tools:
- screencapture: screenshots (native)
- cliclick: mouse control (brew install cliclick)
- osascript: keyboard input (native AppleScript)

Claude sends actions like {"action": "left_click", "coordinate": [500, 300]}
and this module executes them on the actual screen.
"""

import base64
import logging
import subprocess
import tempfile
import time
from typing import Optional

from PIL import Image

logger = logging.getLogger("winston.computer_use")

# Map of key names Claude uses → AppleScript key code names
APPLESCRIPT_KEY_MAP = {
    "return": "return",
    "enter": "return",
    "tab": "tab",
    "escape": "escape",
    "space": "space",
    "delete": "delete",
    "backspace": "delete",
    "up": "up arrow",
    "down": "down arrow",
    "left": "left arrow",
    "right": "right arrow",
    "home": "home",
    "end": "end",
    "pageup": "page up",
    "pagedown": "page down",
    "f1": "f1",
    "f2": "f2",
    "f3": "f3",
    "f4": "f4",
    "f5": "f5",
    "f6": "f6",
    "f7": "f7",
    "f8": "f8",
    "f9": "f9",
    "f10": "f10",
    "f11": "f11",
    "f12": "f12",
}

# Modifier key names → AppleScript modifier syntax
APPLESCRIPT_MODIFIER_MAP = {
    "command": "command down",
    "super": "command down",
    "cmd": "command down",
    "ctrl": "control down",
    "control": "control down",
    "alt": "option down",
    "option": "option down",
    "shift": "shift down",
}


class MacOSComputerController:
    """Execute Computer Use actions on macOS.

    Claude works at logical display dimensions (e.g. 1280×800).
    Native screen may be larger (Retina). Coordinates are scaled accordingly.
    """

    def __init__(self, display_width: int = 1280, display_height: int = 800):
        self.display_width = display_width
        self.display_height = display_height
        self._native_width: Optional[int] = None
        self._native_height: Optional[int] = None
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0
        self._init_screen_info()

    def _init_screen_info(self) -> None:
        """Detect native screen resolution and compute scale factors."""
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if "Resolution" in line and "x" in line:
                    # e.g. "Resolution: 2560 x 1600 Retina" or "3024 x 1964 @ 120.00Hz"
                    parts = line.split(":")[-1].strip().split()
                    w, h = int(parts[0]), int(parts[2])
                    self._native_width = w
                    self._native_height = h
                    self._scale_x = w / self.display_width
                    self._scale_y = h / self.display_height
                    logger.info(
                        "Screen: %dx%d native, %dx%d API, scale=%.2fx%.2f",
                        w,
                        h,
                        self.display_width,
                        self.display_height,
                        self._scale_x,
                        self._scale_y,
                    )
                    return
        except Exception as e:
            logger.warning("Could not detect screen resolution: %s", e)

        logger.info("Using 1:1 coordinate scaling (no Retina detected)")

    def get_tool_definition(self) -> dict:
        """Return the Anthropic Computer Use tool definition."""
        return {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": self.display_width,
            "display_height_px": self.display_height,
        }

    def execute(self, action: str, params: dict) -> list[dict]:
        """Execute a Computer Use action and return tool result content blocks.

        Always returns a screenshot after the action so Claude can see the result.
        Returns a list of content blocks (text + image) for the tool_result.
        """
        try:
            msg = self._dispatch(action, params)
        except Exception as e:
            msg = f"Error executing {action}: {e}"
            logger.error(msg, exc_info=True)

        # After every action, take a screenshot for Claude to see
        screenshot_b64 = self._capture_screenshot()

        result = [{"type": "text", "text": msg}]
        if screenshot_b64:
            result.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_b64,
                    },
                }
            )
        return result

    def _dispatch(self, action: str, params: dict) -> str:
        """Route action to the appropriate handler."""
        if action == "screenshot":
            return "Screenshot taken."

        elif action == "left_click":
            return self._click(params.get("coordinate"), button="c")

        elif action == "right_click":
            return self._click(params.get("coordinate"), button="rc")

        elif action == "double_click":
            return self._click(params.get("coordinate"), button="dc")

        elif action == "middle_click":
            return self._click(params.get("coordinate"), button="tc")

        elif action == "mouse_move":
            return self._mouse_move(params.get("coordinate"))

        elif action == "left_click_drag":
            return self._drag(
                params.get("start_coordinate"),
                params.get("coordinate"),
            )

        elif action == "type":
            return self._type_text(params.get("text", ""))

        elif action == "key":
            return self._key_press(params.get("text", ""))

        elif action == "scroll":
            return self._scroll(
                params.get("coordinate"),
                params.get("scroll_direction", "down"),
                params.get("scroll_amount", 3),
            )

        elif action == "wait":
            duration = params.get("duration", 1.0)
            time.sleep(min(duration, 5.0))
            return f"Waited {duration}s."

        elif action == "cursor_position":
            return self._get_cursor_position()

        else:
            return f"Unknown action: {action}"

    # ── Mouse ────────────────────────────────────────────────────────

    def _click(self, coordinate: Optional[list], button: str = "c") -> str:
        """Click at coordinate. button: c=left, rc=right, dc=double, tc=middle."""
        if not coordinate or len(coordinate) < 2:
            return "Error: coordinate required for click"
        x, y = self._scale(coordinate[0], coordinate[1])
        subprocess.run(["cliclick", f"{button}:{x},{y}"], timeout=5)
        return f"Clicked at ({coordinate[0]}, {coordinate[1]})"

    def _mouse_move(self, coordinate: Optional[list]) -> str:
        if not coordinate or len(coordinate) < 2:
            return "Error: coordinate required for mouse_move"
        x, y = self._scale(coordinate[0], coordinate[1])
        subprocess.run(["cliclick", f"m:{x},{y}"], timeout=5)
        return f"Moved mouse to ({coordinate[0]}, {coordinate[1]})"

    def _drag(self, start: Optional[list], end: Optional[list]) -> str:
        if not start or not end or len(start) < 2 or len(end) < 2:
            return "Error: start_coordinate and coordinate required for drag"
        sx, sy = self._scale(start[0], start[1])
        ex, ey = self._scale(end[0], end[1])
        subprocess.run(["cliclick", f"dd:{sx},{sy}", f"du:{ex},{ey}"], timeout=5)
        return f"Dragged from ({start[0]},{start[1]}) to ({end[0]},{end[1]})"

    def _get_cursor_position(self) -> str:
        result = subprocess.run(
            ["cliclick", "p:."],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return f"Cursor position: {result.stdout.strip()}"

    # ── Keyboard ─────────────────────────────────────────────────────

    def _type_text(self, text: str) -> str:
        """Type text string using AppleScript keystroke."""
        if not text:
            return "Error: text required"
        # Escape for AppleScript
        safe = text.replace("\\", "\\\\").replace('"', '\\"')
        script = f'tell application "System Events"\n  keystroke "{safe}"\nend tell'
        subprocess.run(["osascript", "-e", script], timeout=10)
        return f"Typed: {text[:50]}"

    def _key_press(self, combo: str) -> str:
        """Press a key or key combination. E.g. 'Return', 'command+f', 'ctrl+shift+a'."""
        if not combo:
            return "Error: key combo required"

        parts = [p.strip().lower() for p in combo.split("+")]

        modifiers = []
        key_part = None
        for p in parts:
            if p in APPLESCRIPT_MODIFIER_MAP:
                modifiers.append(APPLESCRIPT_MODIFIER_MAP[p])
            else:
                key_part = p

        if key_part is None:
            return f"Error: no key found in combo '{combo}'"

        # Build AppleScript
        if key_part in APPLESCRIPT_KEY_MAP:
            # Named key (Return, Tab, arrow keys, etc.)
            key_name = APPLESCRIPT_KEY_MAP[key_part]
            if modifiers:
                using = ", ".join(modifiers)
                script = (
                    'tell application "System Events"\n'
                    f'  key code (key code of "{key_name}") using {{{using}}}\n'
                    "end tell"
                )
                # Actually, for named keys with modifiers, use this pattern:
                script = (
                    f'tell application "System Events"\n  keystroke (ASCII character 0) using {{{using}}}\nend tell'
                )
                # Simpler: just use key code approach for special keys
                script = self._build_key_script(key_part, modifiers)
            else:
                script = f'tell application "System Events"\n  key code {self._get_key_code(key_part)}\nend tell'
        elif len(key_part) == 1:
            # Single character
            if modifiers:
                using = ", ".join(modifiers)
                script = f'tell application "System Events"\n  keystroke "{key_part}" using {{{using}}}\nend tell'
            else:
                script = f'tell application "System Events"\n  keystroke "{key_part}"\nend tell'
        else:
            return f"Error: unrecognized key '{key_part}'"

        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return f"Key press error: {result.stderr.strip()}"
        return f"Pressed: {combo}"

    @staticmethod
    def _get_key_code(key_name: str) -> int:
        """Get macOS virtual key code for named keys."""
        codes = {
            "return": 36,
            "enter": 36,
            "tab": 48,
            "escape": 53,
            "space": 49,
            "delete": 51,
            "backspace": 51,
            "up": 126,
            "down": 125,
            "left": 123,
            "right": 124,
            "home": 115,
            "end": 119,
            "pageup": 116,
            "pagedown": 121,
            "f1": 122,
            "f2": 120,
            "f3": 99,
            "f4": 118,
            "f5": 96,
            "f6": 97,
            "f7": 98,
            "f8": 100,
            "f9": 101,
            "f10": 109,
            "f11": 103,
            "f12": 111,
        }
        return codes.get(key_name.lower(), 36)

    @staticmethod
    def _build_key_script(key_name: str, modifiers: list[str]) -> str:
        """Build AppleScript for pressing a named key with modifiers."""
        code = MacOSComputerController._get_key_code(key_name)
        using = ", ".join(modifiers)
        return f'tell application "System Events"\n  key code {code} using {{{using}}}\nend tell'

    # ── Scroll ───────────────────────────────────────────────────────

    def _scroll(
        self,
        coordinate: Optional[list],
        direction: str,
        amount: int,
    ) -> str:
        """Scroll at a position. Uses cliclick for scroll if available."""
        # Move mouse to position first if coordinate given
        if coordinate and len(coordinate) >= 2:
            x, y = self._scale(coordinate[0], coordinate[1])
            subprocess.run(["cliclick", f"m:{x},{y}"], timeout=5)

        # AppleScript scroll
        if direction in ("up", "down"):
            delta = amount if direction == "up" else -amount
            script = (
                'tell application "System Events"\n'
                f"  repeat {abs(amount)} times\n"
                f"    scroll {'up' if delta > 0 else 'down'}\n"
                "  end repeat\n"
                "end tell"
            )
            # Actually, use key codes for more reliable scrolling
            key_code = 126 if direction == "up" else 125  # arrow up/down
            script = (
                'tell application "System Events"\n'
                f"  repeat {amount} times\n"
                f"    key code {key_code}\n"
                "    delay 0.05\n"
                "  end repeat\n"
                "end tell"
            )
        elif direction in ("left", "right"):
            key_code = 123 if direction == "left" else 124
            script = (
                'tell application "System Events"\n'
                f"  repeat {amount} times\n"
                f"    key code {key_code}\n"
                "    delay 0.05\n"
                "  end repeat\n"
                "end tell"
            )
        else:
            return f"Unknown scroll direction: {direction}"

        subprocess.run(["osascript", "-e", script], timeout=15)
        return f"Scrolled {direction} by {amount}"

    # ── Screenshot ───────────────────────────────────────────────────

    def _capture_screenshot(self) -> Optional[str]:
        """Capture screenshot, resize to API dimensions, return base64 PNG."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_path = f.name

            subprocess.run(
                ["screencapture", "-x", "-C", tmp_path],
                timeout=10,
            )

            img = Image.open(tmp_path)
            img = img.resize(
                (self.display_width, self.display_height),
                Image.LANCZOS,
            )

            # Save as PNG to buffer
            import io

            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        except Exception as e:
            logger.error("Screenshot failed: %s", e)
            return None

    # ── Coordinate scaling ───────────────────────────────────────────

    def _scale(self, x: int, y: int) -> tuple[int, int]:
        """Scale API coordinates to native screen coordinates."""
        return int(x * self._scale_x), int(y * self._scale_y)
