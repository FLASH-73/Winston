# Contributing to Winston

## Development Setup

### Prerequisites

- Python 3.11+
- macOS (required for Computer Use features — `cliclick`, `screencapture`, `osascript`)
- Homebrew

### Clone and Install

```bash
git clone <repo-url>
cd "Intelligent Personal Assistant"

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# macOS system dependencies
brew install portaudio    # Required by sounddevice (audio I/O)
brew install cliclick     # Required for Computer Use (mouse control)
```

This installs the runtime dependencies plus `pytest`, `ruff`, and `mypy` from the `[project.optional-dependencies] dev` section in `pyproject.toml`.

### API Keys

Unit tests do **not** require API keys. For manual testing:

```bash
# Create .env in winston/
cat > winston/.env << 'EOF'
ANTHROPIC_API_KEY=your_key
ELEVENLABS_API_KEY=your_key    # Optional — falls back to pyttsx3
GROQ_API_KEY=your_key          # Optional — falls back to local Whisper
EOF
```

### Running Winston

```bash
make run
# or: cd winston && python main.py
```

---

## Running Tests

```bash
# Run all tests
make test

# Run a specific test file
pytest winston/tests/test_routing.py -v

# Run with output visible
pytest winston/tests/ -v -s
```

Test configuration is in `pyproject.toml`:
- `pythonpath = ["jarvis"]` resolves imports (legacy path alias)
- `testpaths = ["tests"]` points to `winston/tests/`

All tests run offline — no API keys, no network, no hardware.

---

## Testing Philosophy

### Mock External Services

Every test must work without network access. Use `unittest.mock.patch` for:
- Claude API calls (`anthropic.Anthropic`)
- Groq API calls (`groq.Groq`)
- Subprocess calls (`subprocess.run`)
- File system operations when testing persistence

### Use Dataclass Mocks for Anthropic Responses

**Do not use plain `MagicMock` for Claude API responses.** The source code does `block.type == "tool_use"` (comparison) and `hasattr(block, "text")` (attribute check). MagicMock auto-creates attributes, which breaks these checks.

Use the helpers in `winston/tests/conftest.py`:

```python
from conftest import MockTextBlock, MockToolUseBlock, make_text_response, make_tool_response

# Simple text response
response = make_text_response("Hello, how can I help?")

# Tool-use response (e.g., routing to agent)
response = make_tool_response(
    tool_name="delegate_to_agent",
    tool_input={"task": "search for Python tutorials"},
    text="I'll search for that.",  # optional preceding text
)
```

### Memory Tests

Use `chromadb.EphemeralClient()` for in-memory ChromaDB — no disk I/O, no cleanup needed. See the `memory_instance` fixture in `conftest.py`.

### File I/O Tests

Use pytest's `tmp_path` fixture for any test that writes files:

```python
def test_persistent_store(tmp_path):
    store = PersistentStore(str(tmp_path / "test.json"), {"items": []})
    store.append_to_list("items", {"id": "1"})
    assert len(store.get("items")) == 1
```

---

## Code Style

Configuration lives in `pyproject.toml`:

- **Linter**: ruff (rules: E, F, I — pycodestyle errors, pyflakes, isort)
- **Line length**: 120 characters
- **Target**: Python 3.11
- **Type hints**: Encouraged on all public method signatures. Use `Optional[T]` for nullable returns.
- **Imports**: stdlib, then third-party, then local. Blank lines between groups. Enforced by ruff's isort (I) rules.

### Conventions

- **Logging**: `logger = logging.getLogger("winston.<module>")`. Use INFO for operations, DEBUG for data flow, WARNING for recoverable issues, ERROR for failures.
- **Config**: All tunable values go in `winston/config.py`. No magic numbers in module code.
- **Docstrings**: Required on all public classes and methods. One line for trivial setters, 2-3 lines for non-obvious behavior.

### Running Linters

```bash
make lint      # ruff check + mypy
make format    # ruff format (auto-fix)
```

---

## PR Process

1. Create a feature branch from `main`
2. Make your changes
3. Run `make lint && make test` — both must pass
4. Describe what changed and why in the PR description
5. For new features: add tests that work offline (mock external services)

---

## Architecture Overview

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system architecture, including data flow diagrams, threading model, and error handling philosophy.

Key entry points for contributors:

| What | Where |
|------|-------|
| Orchestrator (event loop, callback wiring) | `winston/main.py` |
| Claude API + intelligent routing | `winston/brain/claude_client.py` — `process_user_input()` |
| Agent tools (pluggable registry) | `winston/brain/agent_tools.py` — see [ADDING_TOOLS.md](ADDING_TOOLS.md) |
| STT providers (pluggable backends) | `winston/perception/stt.py` — see [ADDING_STT_PROVIDERS.md](ADDING_STT_PROVIDERS.md) |
| Shared state (thread-safe) | `winston/dashboard/state.py` |
| Configuration (all tunable values) | `winston/config.py` |
| Test fixtures and mock helpers | `winston/tests/conftest.py` |
