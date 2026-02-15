# Adding Agent Tools

This guide walks through adding a new tool to Winston's autonomous agent (Opus). Tools are defined in `winston/brain/agent_tools.py`.

## How Tools Work

Each tool is an `AgentTool` dataclass:

```python
@dataclass
class AgentTool:
    name: str                          # Unique identifier (e.g. "web_search")
    description: str                   # What it does (shown to Claude)
    input_schema: dict                 # JSON Schema for parameters
    execute_fn: Callable[[dict], str]  # Function that runs the tool
```

The `ToolRegistry` manages tools:

- `register(tool)` — add a tool
- `get_api_definitions()` — format all tools for the Anthropic `tools=` parameter
- `execute(name, inputs)` — run a tool by name, returns result string
- `list_tools()` — list registered tool names

**Output truncation**: Results longer than `MAX_RESULT_CHARS` (4000 characters) are automatically truncated by `ToolRegistry.execute()`. If your tool might produce large output, consider summarizing rather than dumping raw data.

---

## Step-by-Step: Adding a Tool

### 1. Write the Execute Function

Add a module-level function in `agent_tools.py`, prefixed with `_`:

```python
def _take_screenshot_and_describe(inputs: dict) -> str:
    """Take a screenshot and return a text description of screen contents."""
    from brain.computer_use import MacOSComputerController
    from config import COMPUTER_USE_DISPLAY_WIDTH, COMPUTER_USE_DISPLAY_HEIGHT

    controller = MacOSComputerController(
        display_width=COMPUTER_USE_DISPLAY_WIDTH,
        display_height=COMPUTER_USE_DISPLAY_HEIGHT,
    )
    screenshot_b64 = controller._capture_screenshot()
    if not screenshot_b64:
        return "Error: Failed to capture screenshot"

    import anthropic
    from config import ANTHROPIC_API_KEY, FAST_MODEL

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=0)
    response = client.messages.create(
        model=FAST_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_b64,
                    },
                },
                {
                    "type": "text",
                    "text": inputs.get("prompt", "Describe what is visible on this screen."),
                },
            ],
        }],
    )
    return response.content[0].text if response.content else "No description available"
```

Key conventions:
- Always return a **string** (never dict, never None)
- Catch errors and return `"Error: ..."` strings
- Import heavy dependencies inside the function (lazy imports)

> **Note**: This example creates a standalone Anthropic client, which bypasses Winston's cost tracker. For production tools, consider passing a `ClaudeClient` reference through the registry factory instead.

### 2. Register in `create_default_registry()`

Inside the `create_default_registry()` function in `agent_tools.py`:

```python
registry.register(AgentTool(
    name="take_screenshot_and_describe",
    description=(
        "Take a screenshot of the current screen and return a text description "
        "of what's visible. Useful for understanding screen state without using "
        "the full Computer Use tool. Optionally provide a prompt to focus the "
        "description on specific elements."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "What to focus on in the description (optional)",
            },
        },
    },
    execute_fn=_take_screenshot_and_describe,
))
```

### 3. Write a Test

Add tests in `winston/tests/test_tools.py`:

```python
from unittest.mock import patch, MagicMock
from brain.agent_tools import create_default_registry, _take_screenshot_and_describe


def test_screenshot_tool_registered():
    """Tool should appear in the default registry."""
    registry = create_default_registry()
    assert "take_screenshot_and_describe" in registry.list_tools()


def test_screenshot_handles_capture_failure():
    """Return error string when screenshot fails (don't crash)."""
    with patch("brain.agent_tools.MacOSComputerController") as mock_ctrl:
        mock_instance = MagicMock()
        mock_instance._capture_screenshot.return_value = None
        mock_ctrl.return_value = mock_instance

        result = _take_screenshot_and_describe({"prompt": "test"})
        assert "Error" in result
```

---

## Input Schema Format

The `input_schema` uses standard [JSON Schema](https://json-schema.org/):

```python
{
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "What to search for",
        },
        "max_results": {
            "type": "integer",
            "description": "Number of results (default 5, max 10)",
        },
    },
    "required": ["query"],  # Omit for all-optional parameters
}
```

Supported types: `"string"`, `"integer"`, `"number"`, `"boolean"`, `"array"`, `"object"`.

The `description` field is important — Claude reads it to understand how to use the parameter.

---

## Tool Safety

Tools that access the filesystem or run commands follow security patterns:

- **Path validation**: `_validate_path(path)` resolves symlinks and checks against `ALLOWED_READ_PATHS` (home directory, `/tmp`). Used by `read_local_file`, `search_local_files`, `list_local_directory`.
- **Command validation**: `_validate_command_segment(segment)` checks each segment of piped/chained commands against `ALLOWED_SHELL_COMMANDS` in `config.py`. Rejects command substitution patterns (`$(...)`, backticks).
- **Timeouts**: All subprocess calls use `timeout` parameters (`TOOL_TIMEOUT = 15s` default, `SHELL_COMMAND_TIMEOUT` for shell).

When adding tools that access the filesystem or run commands, use these existing validators.

---

## Checklist

- [ ] Execute function defined (module-level, prefixed with `_`)
- [ ] Returns a string (not dict, not None)
- [ ] Errors caught and returned as `"Error: ..."` strings
- [ ] Registered in `create_default_registry()`
- [ ] `input_schema` follows JSON Schema with `description` on each property
- [ ] Output fits within 4000 chars or truncates gracefully
- [ ] Test added to `winston/tests/test_tools.py`
- [ ] Test works offline (mock external services)
