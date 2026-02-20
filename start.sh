#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find Python in the virtual environment
PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Virtual environment not found at .venv/"
    echo "  Run: conda create -y -p .venv python=3.12 && .venv/bin/pip install -e '.[dev]'"
    exit 1
fi

# Load .env if API keys not already exported
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f winston/.env ]; then
    set -a
    source winston/.env
    set +a
fi

# Verify critical API key
if [ -z "${ANTHROPIC_API_KEY:-}" ] || [ "${ANTHROPIC_API_KEY}" = "your_anthropic_key_here" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set. Edit winston/.env with your real API keys."
    exit 1
fi

# Warn if no display (Computer Use needs X11)
if [ -z "${DISPLAY:-}" ]; then
    echo "WARNING: \$DISPLAY not set. Computer Use (mouse/keyboard/screenshot) will not work."
    echo "If you're running via SSH, try: export DISPLAY=:0"
fi

echo "Starting Winston..."
cd winston && exec "$PYTHON" main.py "$@"
