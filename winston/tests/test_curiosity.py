"""Tests for the Curiosity Engine — autonomous background thinking loop."""

import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Patch config before importing curiosity module
_tmp_state = tempfile.mktemp(suffix=".json")


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch):
    """Patch curiosity config for all tests."""
    monkeypatch.setattr("brain.curiosity.CURIOSITY_MIN_INTERVAL", 60)
    monkeypatch.setattr("brain.curiosity.CURIOSITY_MAX_INTERVAL", 120)
    monkeypatch.setattr("brain.curiosity.CURIOSITY_QUIET_START", 1)
    monkeypatch.setattr("brain.curiosity.CURIOSITY_QUIET_END", 7)
    monkeypatch.setattr("brain.curiosity.CURIOSITY_DAILY_CAP", 5)
    monkeypatch.setattr("brain.curiosity.CURIOSITY_ABSENCE_HOURS", 6)
    monkeypatch.setattr("brain.curiosity.CURIOSITY_STATE_FILE", _tmp_state)
    monkeypatch.setattr("brain.curiosity._TOPIC_DEDUP_HOURS", 48)
    # Clean up state file
    if os.path.exists(_tmp_state):
        os.remove(_tmp_state)


def _make_engine():
    """Create a CuriosityEngine with mocked dependencies."""
    from brain.curiosity import CuriosityEngine

    llm = MagicMock()
    llm.text_only_chat = MagicMock(return_value=None)

    memory = MagicMock()
    memory.semantic.get_all_facts_as_text = MagicMock(return_value="Roberto builds robots.")
    memory.working.get_session_narrative = MagicMock(return_value="")
    memory.store_proactive = MagicMock()

    temporal = MagicMock()
    temporal.get_narrative = MagicMock(return_value="Workshop active.")
    temporal.get_latest_state = MagicMock(return_value="Working on motors.")

    notifier = MagicMock()
    notifier.send_curiosity_message = MagicMock()

    cost_tracker = MagicMock()
    cost_tracker.check_budget = MagicMock(return_value=True)

    engine = CuriosityEngine(
        llm=llm,
        memory=memory,
        temporal_memory=temporal,
        telegram_notifier=notifier,
        cost_tracker=cost_tracker,
    )
    return engine


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------


def test_quiet_hours_skips_cycle():
    """Cycle should skip during quiet hours (1am-7am)."""
    engine = _make_engine()
    # Simulate 3am
    with patch("brain.curiosity.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 15, 3, 0)
        mock_dt.fromisoformat = datetime.fromisoformat
        engine._cycle()
    # LLM should not have been called
    engine._llm.text_only_chat.assert_not_called()


def test_daily_cap_skips_cycle():
    """Cycle should skip when daily cap is reached."""
    engine = _make_engine()
    engine._sends_today = [{"topic": f"t{i}", "timestamp": "2025-01-15T12:00:00", "message": "m"} for i in range(5)]

    with patch("brain.curiosity.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 15, 14, 0)
        mock_dt.fromisoformat = datetime.fromisoformat
        engine._cycle()

    engine._llm.text_only_chat.assert_not_called()


def test_budget_exceeded_skips_cycle():
    """Cycle should skip when daily budget is exceeded."""
    engine = _make_engine()
    engine._cost_tracker.check_budget.return_value = False

    with patch("brain.curiosity.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 15, 14, 0)
        mock_dt.fromisoformat = datetime.fromisoformat
        engine._cycle()

    engine._llm.text_only_chat.assert_not_called()


# ---------------------------------------------------------------------------
# Topic deduplication
# ---------------------------------------------------------------------------


def test_topic_dedup_recent():
    """A topic explored 1 hour ago should be considered recent (within 48h window)."""
    engine = _make_engine()
    one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
    engine._recent_topics = [{"topic": "motor specs", "timestamp": one_hour_ago}]

    assert engine._is_topic_recent("motor specs") is True
    assert engine._is_topic_recent("Motor Specs") is True  # Case insensitive
    assert engine._is_topic_recent("something else") is False


def test_topic_dedup_expired():
    """A topic explored 50 hours ago should not be considered recent (outside 48h)."""
    engine = _make_engine()
    old = (datetime.now() - timedelta(hours=50)).isoformat()
    engine._recent_topics = [{"topic": "motor specs", "timestamp": old}]

    assert engine._is_topic_recent("motor specs") is False


# ---------------------------------------------------------------------------
# Day rotation
# ---------------------------------------------------------------------------


def test_day_rotation_resets_sends():
    """Day rotation should clear sends_today when date changes."""
    engine = _make_engine()
    engine._today = "2025-01-14"
    engine._sends_today = [{"topic": "t1", "timestamp": "2025-01-14T10:00:00", "message": "m"}]
    engine._recent_topics = [
        {"topic": "old", "timestamp": (datetime.now() - timedelta(hours=50)).isoformat()},
        {"topic": "recent", "timestamp": (datetime.now() - timedelta(hours=1)).isoformat()},
    ]

    engine._rotate_day()

    assert engine._sends_today == []
    assert engine._today == datetime.now().strftime("%Y-%m-%d")
    # Old topic should be pruned, recent one kept
    topic_names = [t["topic"] for t in engine._recent_topics]
    assert "recent" in topic_names
    assert "old" not in topic_names


# ---------------------------------------------------------------------------
# Phase 1: REFLECT
# ---------------------------------------------------------------------------


def test_reflect_parses_valid_json():
    """Phase 1 should parse valid JSON topic suggestions."""
    engine = _make_engine()
    engine._llm.text_only_chat.return_value = json.dumps({
        "topics": [
            {"topic": "Damiao motor updates", "why_interesting": "New firmware", "search_query": "Damiao motor firmware 2025"},
        ]
    })

    with patch("personality.get_personality") as mock_p:
        mock_p.return_value = MagicMock()
        mock_p.return_value.companion.reflect_prompt = "Reflect on Roberto's world."
        topics = engine._phase_reflect()

    assert len(topics) == 1
    assert topics[0]["topic"] == "Damiao motor updates"


def test_reflect_handles_code_fences():
    """Phase 1 should strip markdown code fences from LLM response."""
    engine = _make_engine()
    engine._llm.text_only_chat.return_value = '```json\n{"topics": [{"topic": "test", "why_interesting": "x", "search_query": "q"}]}\n```'

    with patch("personality.get_personality") as mock_p:
        mock_p.return_value = MagicMock()
        mock_p.return_value.companion.reflect_prompt = "Reflect."
        topics = engine._phase_reflect()

    assert len(topics) == 1
    assert topics[0]["topic"] == "test"


def test_reflect_handles_garbage():
    """Phase 1 should return empty list on non-JSON response."""
    engine = _make_engine()
    engine._llm.text_only_chat.return_value = "I don't know what to think about."

    with patch("personality.get_personality") as mock_p:
        mock_p.return_value = MagicMock()
        mock_p.return_value.companion.reflect_prompt = "Reflect."
        topics = engine._phase_reflect()

    assert topics == []


def test_reflect_empty_prompt_returns_empty():
    """Phase 1 should return empty if companion.reflect_prompt is empty."""
    engine = _make_engine()

    with patch("personality.get_personality") as mock_p:
        mock_p.return_value = MagicMock()
        mock_p.return_value.companion.reflect_prompt = ""
        topics = engine._phase_reflect()

    assert topics == []


# ---------------------------------------------------------------------------
# Phase 2: EXPLORE
# ---------------------------------------------------------------------------


def test_explore_extracts_finding():
    """Phase 2 should call web search and extract finding."""
    engine = _make_engine()
    engine._llm.text_only_chat.return_value = "PI just released a new dexterous hand model."

    with patch("brain.agent_tools._web_search", return_value="1. Article about PI\n   http://example.com\n   Details..."), \
         patch("personality.get_personality") as mock_p:
        mock_p.return_value = MagicMock()
        mock_p.return_value.companion.explore_prompt = "Search: {query}\nResults: {results}\nInterest: {why_interesting}"

        finding = engine._phase_explore("Physical Intelligence new hand", "competitor update")

    assert finding == "PI just released a new dexterous hand model."


def test_explore_returns_none_on_no_results():
    """Phase 2 should return None when search returns no results."""
    engine = _make_engine()

    with patch("brain.agent_tools._web_search", return_value="No results found for: xyz"):
        finding = engine._phase_explore("xyz", "test")

    assert finding is None


# ---------------------------------------------------------------------------
# Phase 3: CRAFT
# ---------------------------------------------------------------------------


def test_craft_uses_smart_model():
    """Phase 3 should use SMART_MODEL for crafting messages."""
    engine = _make_engine()
    engine._llm.text_only_chat.return_value = "So apparently PI just shipped a new hand."

    with patch("personality.get_personality") as mock_p:
        mock_p.return_value = MagicMock()
        mock_p.return_value.companion.craft_prompt = "Context: {context}\nTopic: {topic}\nFinding: {finding}"
        message = engine._phase_craft("PI update", "New dexterous hand")

    assert message == "So apparently PI just shipped a new hand."
    # Verify SMART_MODEL was passed
    call_kwargs = engine._llm.text_only_chat.call_args
    assert call_kwargs.kwargs.get("model") is not None  # Should be SMART_MODEL


def test_craft_not_worth_sending():
    """Phase 3 should return NOT_WORTH_SENDING string when LLM decides not to send."""
    engine = _make_engine()
    engine._llm.text_only_chat.return_value = "NOT_WORTH_SENDING"

    with patch("personality.get_personality") as mock_p:
        mock_p.return_value = MagicMock()
        mock_p.return_value.companion.craft_prompt = "Context: {context}\nTopic: {topic}\nFinding: {finding}"
        message = engine._phase_craft("boring topic", "nothing interesting")

    assert "NOT_WORTH_SENDING" in message


# ---------------------------------------------------------------------------
# Absence check-in
# ---------------------------------------------------------------------------


def test_absence_checkin_triggers_after_threshold():
    """Absence check-in should trigger after CURIOSITY_ABSENCE_HOURS."""
    engine = _make_engine()
    engine._last_activity_time = time.time() - (7 * 3600)  # 7 hours ago
    engine._llm.text_only_chat.return_value = "Workshop's been quiet. How's the motor assembly going?"

    with patch("personality.get_personality") as mock_p, \
         patch("brain.curiosity.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 15, 14, 0)
        mock_dt.fromisoformat = datetime.fromisoformat
        mock_p.return_value = MagicMock()
        mock_p.return_value.companion.reflect_prompt = "Reflect."
        mock_p.return_value.companion.absence_prompt = "Check in. Hours: {hours}. Context: {context}"

        engine._cycle()

    # Should have sent a message
    engine._notifier.send_curiosity_message.assert_called_once()
    msg = engine._notifier.send_curiosity_message.call_args[0][0]
    assert "motor" in msg.lower() or "quiet" in msg.lower() or "workshop" in msg.lower() or len(msg) > 0


def test_absence_checkin_deduplicated():
    """Absence check-in should only happen once per absence stretch."""
    engine = _make_engine()
    engine._last_activity_time = time.time() - (7 * 3600)
    # Already checked in
    engine._recent_topics.append({
        "topic": "__absence_checkin__",
        "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
    })
    engine._llm.text_only_chat.return_value = json.dumps({"topics": []})

    with patch("personality.get_personality") as mock_p, \
         patch("brain.curiosity.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 15, 14, 0)
        mock_dt.fromisoformat = datetime.fromisoformat
        mock_p.return_value = MagicMock()
        mock_p.return_value.companion.reflect_prompt = "Reflect."

        engine._cycle()

    # Should NOT send — already checked in
    engine._notifier.send_curiosity_message.assert_not_called()


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def test_state_save_and_load():
    """State should survive save/load round-trip."""
    engine = _make_engine()
    engine._sends_today = [{"topic": "t1", "timestamp": "2025-01-15T10:00:00", "message": "hello"}]
    engine._recent_topics = [{"topic": "r1", "timestamp": "2025-01-15T09:00:00"}]
    engine._last_activity_time = 1700000000.0

    engine._save_state()

    # Create a new engine and verify state loaded
    engine2 = _make_engine()
    assert len(engine2._sends_today) == 1
    assert engine2._sends_today[0]["topic"] == "t1"
    assert len(engine2._recent_topics) == 1
    assert engine2._recent_topics[0]["topic"] == "r1"
    assert engine2._last_activity_time == 1700000000.0


def test_state_sends_reset_on_new_day():
    """Sends from a previous day should not be loaded."""
    engine = _make_engine()
    engine._today = "2024-01-01"
    engine._sends_today = [{"topic": "old", "timestamp": "2024-01-01T10:00:00", "message": "m"}]
    engine._save_state()

    engine2 = _make_engine()
    # Sends from a different day should be cleared
    assert engine2._sends_today == []


# ---------------------------------------------------------------------------
# Full cycle integration
# ---------------------------------------------------------------------------


def test_full_cycle_sends_message():
    """A complete reflect->explore->craft cycle should send a message."""
    engine = _make_engine()

    reflect_json = json.dumps({
        "topics": [
            {"topic": "YC batch updates", "why_interesting": "New robotics startups", "search_query": "YC W25 robotics startups"},
        ]
    })

    # Configure LLM to return different things for each call
    call_count = {"n": 0}
    responses = [
        reflect_json,                          # Phase 1: reflect
        "3 new robotics startups in YC W25.",  # Phase 2: explore finding
        "Three new robotics companies in the current YC batch. Two are doing manipulation, one force feedback. https://example.com",  # Phase 3: craft
    ]

    def mock_text_only_chat(prompt, **kwargs):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return responses[idx]

    engine._llm.text_only_chat = mock_text_only_chat

    with patch("personality.get_personality") as mock_p, \
         patch("brain.agent_tools._web_search", return_value="1. YC startups\n   http://example.com\n   Details"), \
         patch("brain.curiosity.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 15, 14, 0)
        mock_dt.fromisoformat = datetime.fromisoformat
        companion = MagicMock()
        companion.reflect_prompt = "Reflect."
        companion.explore_prompt = "Search: {query}\nResults: {results}\nInterest: {why_interesting}"
        companion.craft_prompt = "Context: {context}\nTopic: {topic}\nFinding: {finding}"
        mock_p.return_value = MagicMock()
        mock_p.return_value.companion = companion

        engine._cycle()

    # Should have sent the crafted message
    engine._notifier.send_curiosity_message.assert_called_once()
    engine._memory.store_proactive.assert_called_once()
    assert len(engine._sends_today) == 1
    assert engine._sends_today[0]["topic"] == "YC batch updates"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_strip_code_fences():
    """Code fence stripping should handle various formats."""
    from brain.curiosity import _strip_code_fences

    assert _strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_code_fences('{"a": 1}') == '{"a": 1}'
    assert _strip_code_fences('  {"a": 1}  ') == '{"a": 1}'


def test_record_activity_resets_timer():
    """record_activity should reset the absence timer."""
    engine = _make_engine()
    engine._last_activity_time = time.time() - 10000
    engine.record_activity()
    assert time.time() - engine._last_activity_time < 1.0


# ---------------------------------------------------------------------------
# Personality loading
# ---------------------------------------------------------------------------


def test_companion_config_loads_from_yaml():
    """CompanionConfig should load from personality YAML."""
    from personality import load_personality

    p = load_personality("default")
    assert p.companion.reflect_prompt != ""
    assert p.companion.explore_prompt != ""
    assert p.companion.craft_prompt != ""
    assert p.companion.absence_prompt != ""
    assert "{query}" in p.companion.explore_prompt
    assert "{topic}" in p.companion.craft_prompt
    assert "{hours}" in p.companion.absence_prompt
