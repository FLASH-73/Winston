"""Shared fixtures and mock helpers for Winston test suite."""

import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock python-telegram-bot if not installed (CI / headless environments)
# ---------------------------------------------------------------------------
_TELEGRAM_MODULES = [
    "telegram",
    "telegram.constants",
    "telegram.ext",
    "telegram._update",
]

for _mod_name in _TELEGRAM_MODULES:
    if _mod_name not in sys.modules:
        mock = MagicMock()
        mock.__name__ = _mod_name
        mock.__path__ = []
        sys.modules[_mod_name] = mock

# ---------------------------------------------------------------------------
# Mock Anthropic response objects
# ---------------------------------------------------------------------------
# Dataclass-based (not MagicMock) because source code does:
#   block.type == "tool_use"   (comparison)
#   hasattr(block, "text")     (attribute check)
# MagicMock auto-creates attributes, breaking these checks.


@dataclass
class MockUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class MockToolUseBlock:
    type: str = "tool_use"
    name: str = ""
    input: dict = field(default_factory=dict)
    id: str = "tool_call_123"


def make_text_response(text: str, input_tokens=100, output_tokens=50):
    """Create a mock Claude response with a single text block."""
    response = MagicMock()
    response.content = [MockTextBlock(type="text", text=text)]
    response.usage = MockUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    response.stop_reason = "end_turn"
    return response


def make_tool_response(tool_name: str, tool_input: dict, text: str = "", input_tokens=100, output_tokens=50):
    """Create a mock Claude response with a tool_use block (and optional preceding text)."""
    response = MagicMock()
    blocks = []
    if text:
        blocks.append(MockTextBlock(type="text", text=text))
    blocks.append(MockToolUseBlock(type="tool_use", name=tool_name, input=tool_input))
    response.content = blocks
    response.usage = MockUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    response.stop_reason = "tool_use"
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def noop_cost_tracker():
    """A CostTracker mock that does nothing (for ClaudeClient construction)."""
    from utils.cost_tracker import CostTracker

    tracker = MagicMock(spec=CostTracker)
    tracker.check_budget.return_value = True
    tracker.get_daily_cost.return_value = 0.0
    return tracker


@pytest.fixture
def claude_client(noop_cost_tracker):
    """ClaudeClient with mocked internals, ready for testing."""
    from brain.claude_client import ClaudeClient

    client = ClaudeClient(cost_tracker=noop_cost_tracker)
    client._client = MagicMock()
    return client


@pytest.fixture
def cost_tracker(tmp_path, monkeypatch):
    """Real CostTracker that writes to a temp directory."""
    import utils.cost_tracker as ct_module

    monkeypatch.setattr(ct_module, "COST_FILE", str(tmp_path / "test_costs.json"))
    from utils.cost_tracker import CostTracker

    return CostTracker()


@pytest.fixture
def memory_instance(tmp_path):
    """Memory instance with in-memory ChromaDB (no disk I/O for episodes)."""
    import chromadb
    from brain.memory import Memory

    mem = Memory()
    client = chromadb.EphemeralClient()
    mem._client = client
    mem.episodic.initialize(client)
    mem.semantic.initialize(client, str(tmp_path))
    return mem
