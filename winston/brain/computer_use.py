"""Computer Use controller for Anthropic's Computer Use API.

Executes screenshot/mouse/keyboard actions using platform-appropriate tools:
- Linux: scrot (screenshots), xdotool (mouse + keyboard)
- macOS: screencapture, cliclick, osascript (kept as fallback)

Claude sends actions like {"action": "left_click", "coordinate": [500, 300]}
and this module executes them on the actual screen.
"""

import base64
import io
import logging
import platform
import re
import subprocess
import tempfile
import time
from typing import Optional

from PIL import Image

logger = logging.getLogger("winston.computer_use")

# ── Key maps for xdotool (Linux) ─────────────────────────────────────

XDOTOOL_KEY_MAP = {
    "return": "Return",
    "enter": "Return",
    "tab": "Tab",
    "escape": "Escape",
    "space": "space",
    "delete": "Delete",
    "backspace": "BackSpace",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "home": "Home",
    "end": "End",
    "pageup": "Prior",
    "pagedown": "Next",
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
}

XDOTOOL_MODIFIER_MAP = {
    "command": "super",
    "super": "super",
    "cmd": "super",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "option": "alt",
    "shift": "shift",
}

# ── Key maps for AppleScript (macOS) ─────────────────────────────────

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

MACOS_KEY_CODES = {
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


class ComputerController:
    """Execute Computer Use actions. Auto-detects platform (Linux or macOS).

    Claude works at logical display dimensions (e.g. 1280x800).
    Native screen may be larger. Coordinates are scaled accordingly.
    """

    def __init__(self, display_width: int = 1280, display_height: int = 800):
        self.display_width = display_width
        self.display_height = display_height
        self._native_width: Optional[int] = None
        self._native_height: Optional[int] = None
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0
        self._platform = platform.system()
        self._init_screen_info()

    def _init_screen_info(self) -> None:
        """Detect native screen resolution and compute scale factors."""
        try:
            if self._platform == "Linux":
                self._init_screen_info_linux()
            else:
                self._init_screen_info_macos()
        except Exception as e:
            logger.warning("Could not detect screen resolution: %s", e)
            logger.info("Using 1:1 coordinate scaling")

    def _init_screen_info_linux(self) -> None:
        """Detect screen resolution on Linux using xdotool."""
        result = subprocess.run(
            ["xdotool", "getdisplaygeometry"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()
            w, h = int(parts[0]), int(parts[1])
            self._native_width = w
            self._native_height = h
            self._scale_x = w / self.display_width
            self._scale_y = h / self.display_height
            logger.info(
                "Screen: %dx%d native, %dx%d API, scale=%.2fx%.2f",
                w, h, self.display_width, self.display_height,
                self._scale_x, self._scale_y,
            )

    def _init_screen_info_macos(self) -> None:
        """Detect screen resolution on macOS using system_profiler."""
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if "Resolution" in line and "x" in line:
                parts = line.split(":")[-1].strip().split()
                w, h = int(parts[0]), int(parts[2])
                self._native_width = w
                self._native_height = h
                self._scale_x = w / self.display_width
                self._scale_y = h / self.display_height
                logger.info(
                    "Screen: %dx%d native, %dx%d API, scale=%.2fx%.2f",
                    w, h, self.display_width, self.display_height,
                    self._scale_x, self._scale_y,
                )
                return
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
            return self._click(params.get("coordinate"), button="left")

        elif action == "right_click":
            return self._click(params.get("coordinate"), button="right")

        elif action == "double_click":
            return self._click(params.get("coordinate"), button="double")

        elif action == "middle_click":
            return self._click(params.get("coordinate"), button="middle")

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

    def _click(self, coordinate: Optional[list], button: str = "left") -> str:
        """Click at coordinate."""
        if not coordinate or len(coordinate) < 2:
            return "Error: coordinate required for click"
        x, y = self._scale(coordinate[0], coordinate[1])

        if self._platform == "Linux":
            subprocess.run(["xdotool", "mousemove", "--sync", str(x), str(y)], timeout=5)
            if button == "left":
                subprocess.run(["xdotool", "click", "1"], timeout=5)
            elif button == "right":
                subprocess.run(["xdotool", "click", "3"], timeout=5)
            elif button == "double":
                subprocess.run(["xdotool", "click", "--repeat", "2", "--delay", "50", "1"], timeout=5)
            elif button == "middle":
                subprocess.run(["xdotool", "click", "2"], timeout=5)
        else:
            cliclick_map = {"left": "c", "right": "rc", "double": "dc", "middle": "tc"}
            btn = cliclick_map.get(button, "c")
            subprocess.run(["cliclick", f"{btn}:{x},{y}"], timeout=5)

        return f"Clicked at ({coordinate[0]}, {coordinate[1]})"

    def _mouse_move(self, coordinate: Optional[list]) -> str:
        if not coordinate or len(coordinate) < 2:
            return "Error: coordinate required for mouse_move"
        x, y = self._scale(coordinate[0], coordinate[1])

        if self._platform == "Linux":
            subprocess.run(["xdotool", "mousemove", "--sync", str(x), str(y)], timeout=5)
        else:
            subprocess.run(["cliclick", f"m:{x},{y}"], timeout=5)

        return f"Moved mouse to ({coordinate[0]}, {coordinate[1]})"

    def _drag(self, start: Optional[list], end: Optional[list]) -> str:
        if not start or not end or len(start) < 2 or len(end) < 2:
            return "Error: start_coordinate and coordinate required for drag"
        sx, sy = self._scale(start[0], start[1])
        ex, ey = self._scale(end[0], end[1])

        if self._platform == "Linux":
            subprocess.run(["xdotool", "mousemove", "--sync", str(sx), str(sy)], timeout=5)
            subprocess.run(["xdotool", "mousedown", "1"], timeout=5)
            subprocess.run(["xdotool", "mousemove", "--sync", str(ex), str(ey)], timeout=5)
            subprocess.run(["xdotool", "mouseup", "1"], timeout=5)
        else:
            subprocess.run(["cliclick", f"dd:{sx},{sy}", f"du:{ex},{ey}"], timeout=5)

        return f"Dragged from ({start[0]},{start[1]}) to ({end[0]},{end[1]})"

    def _get_cursor_position(self) -> str:
        if self._platform == "Linux":
            result = subprocess.run(
                ["xdotool", "getmouselocation"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            match = re.search(r"x:(\d+)\s+y:(\d+)", result.stdout)
            if match:
                return f"Cursor position: {match.group(1)},{match.group(2)}"
            return f"Cursor position: {result.stdout.strip()}"
        else:
            result = subprocess.run(
                ["cliclick", "p:."],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return f"Cursor position: {result.stdout.strip()}"

    # ── Keyboard ─────────────────────────────────────────────────────

    def _type_text(self, text: str) -> str:
        """Type text string."""
        if not text:
            return "Error: text required"

        if self._platform == "Linux":
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "12", text],
                timeout=30,
            )
        else:
            safe = text.replace("\\", "\\\\").replace('"', '\\"')
            script = f'tell application "System Events"\n  keystroke "{safe}"\nend tell'
            subprocess.run(["osascript", "-e", script], timeout=10)

        return f"Typed: {text[:50]}"

    def _key_press(self, combo: str) -> str:
        """Press a key or key combination. E.g. 'Return', 'command+f', 'ctrl+shift+a'."""
        if not combo:
            return "Error: key combo required"

        parts = [p.strip().lower() for p in combo.split("+")]

        if self._platform == "Linux":
            return self._key_press_linux(parts, combo)
        else:
            return self._key_press_macos(parts, combo)

    def _key_press_linux(self, parts: list[str], combo: str) -> str:
        """Press key combo using xdotool."""
        modifiers = []
        key_part = None
        for p in parts:
            if p in XDOTOOL_MODIFIER_MAP:
                modifiers.append(XDOTOOL_MODIFIER_MAP[p])
            else:
                key_part = p

        if key_part is None:
            return f"Error: no key found in combo '{combo}'"

        xdo_key = XDOTOOL_KEY_MAP.get(key_part, key_part)

        if modifiers:
            key_combo = "+".join(modifiers) + "+" + xdo_key
        else:
            key_combo = xdo_key

        result = subprocess.run(
            ["xdotool", "key", key_combo],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return f"Key press error: {result.stderr.strip()}"
        return f"Pressed: {combo}"

    def _key_press_macos(self, parts: list[str], combo: str) -> str:
        """Press key combo using osascript."""
        modifiers = []
        key_part = None
        for p in parts:
            if p in APPLESCRIPT_MODIFIER_MAP:
                modifiers.append(APPLESCRIPT_MODIFIER_MAP[p])
            else:
                key_part = p

        if key_part is None:
            return f"Error: no key found in combo '{combo}'"

        if key_part in APPLESCRIPT_KEY_MAP:
            code = MACOS_KEY_CODES.get(key_part, 36)
            if modifiers:
                using = ", ".join(modifiers)
                script = f'tell application "System Events"\n  key code {code} using {{{using}}}\nend tell'
            else:
                script = f'tell application "System Events"\n  key code {code}\nend tell'
        elif len(key_part) == 1:
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

    # ── Scroll ───────────────────────────────────────────────────────

    def _scroll(
        self,
        coordinate: Optional[list],
        direction: str,
        amount: int,
    ) -> str:
        """Scroll at a position."""
        if coordinate and len(coordinate) >= 2:
            x, y = self._scale(coordinate[0], coordinate[1])
            if self._platform == "Linux":
                subprocess.run(["xdotool", "mousemove", "--sync", str(x), str(y)], timeout=5)
            else:
                subprocess.run(["cliclick", f"m:{x},{y}"], timeout=5)

        if self._platform == "Linux":
            # xdotool: button 4=scroll up, 5=scroll down, 6=left, 7=right
            direction_map = {"up": "4", "down": "5", "left": "6", "right": "7"}
            button = direction_map.get(direction, "5")
            clicks = max(1, amount)
            subprocess.run(
                ["xdotool", "click", "--repeat", str(clicks), "--delay", "30", button],
                timeout=15,
            )
        else:
            if direction in ("up", "down"):
                key_code = 126 if direction == "up" else 125
            elif direction in ("left", "right"):
                key_code = 123 if direction == "left" else 124
            else:
                return f"Unknown scroll direction: {direction}"
            script = (
                'tell application "System Events"\n'
                f"  repeat {amount} times\n"
                f"    key code {key_code}\n"
                "    delay 0.05\n"
                "  end repeat\n"
                "end tell"
            )
            subprocess.run(["osascript", "-e", script], timeout=15)

        return f"Scrolled {direction} by {amount}"

    # ── Screenshot ───────────────────────────────────────────────────

    def _capture_screenshot(self) -> Optional[str]:
        """Capture screenshot, resize to API dimensions, return base64 PNG."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_path = f.name

            if self._platform == "Linux":
                subprocess.run(["scrot", "-o", tmp_path], timeout=10)
            else:
                subprocess.run(["screencapture", "-x", "-C", tmp_path], timeout=10)

            img = Image.open(tmp_path)
            img = img.resize(
                (self.display_width, self.display_height),
                Image.LANCZOS,
            )

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


# Backward-compatible alias
MacOSComputerController = ComputerController
