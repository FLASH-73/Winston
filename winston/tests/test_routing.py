"""Tests for process_user_input() — the intelligent routing system."""

from unittest.mock import patch

from tests.conftest import make_text_response, make_tool_response


def test_process_user_input_conversation(claude_client):
    """No tool_use blocks → returns ('conversation', {'text': ...})."""
    mock_resp = make_text_response("The workshop opens at 8 AM.")

    with (
        patch.object(claude_client, "_call_with_retry", return_value=mock_resp),
        patch.object(claude_client, "_track_usage"),
    ):
        action, data = claude_client.process_user_input("What time does the workshop open?")

    assert action == "conversation"
    assert data["text"] == "The workshop opens at 8 AM."


def test_process_user_input_agent_delegation(claude_client):
    """delegate_to_agent tool_use → returns ('agent', {'task': ...})."""
    mock_resp = make_tool_response("delegate_to_agent", {"task": "Open browser and search for Damiao specs"})

    with (
        patch.object(claude_client, "_call_with_retry", return_value=mock_resp),
        patch.object(claude_client, "_track_usage"),
    ):
        action, data = claude_client.process_user_input("Search for Damiao motor specs")

    assert action == "agent"
    assert data["task"] == "Open browser and search for Damiao specs"


def test_process_user_input_save_note(claude_client):
    """save_note tool_use → returns ('note', {'content': ...})."""
    mock_resp = make_tool_response("save_note", {"content": "buy M3 bolts tomorrow"})

    with (
        patch.object(claude_client, "_call_with_retry", return_value=mock_resp),
        patch.object(claude_client, "_track_usage"),
    ):
        action, data = claude_client.process_user_input("Remember to buy M3 bolts tomorrow")

    assert action == "note"
    assert data["content"] == "buy M3 bolts tomorrow"


def test_process_user_input_shutdown(claude_client):
    """shutdown_system tool_use → returns ('shutdown', {})."""
    mock_resp = make_tool_response("shutdown_system", {})

    with (
        patch.object(claude_client, "_call_with_retry", return_value=mock_resp),
        patch.object(claude_client, "_track_usage"),
    ):
        action, data = claude_client.process_user_input("Good night Winston, go to sleep")

    assert action == "shutdown"
    assert data == {}


def test_process_user_input_get_time(claude_client):
    """get_current_time tool_use → returns ('conversation', {'text': 'It's ...'})."""
    mock_resp = make_tool_response("get_current_time", {})

    with (
        patch.object(claude_client, "_call_with_retry", return_value=mock_resp),
        patch.object(claude_client, "_track_usage"),
    ):
        action, data = claude_client.process_user_input("What time is it?")

    assert action == "conversation"
    assert data["text"].startswith("It's ")


def test_process_user_input_api_failure(claude_client):
    """API returns None → graceful fallback message."""
    with patch.object(claude_client, "_call_with_retry", return_value=None):
        action, data = claude_client.process_user_input("Hello Winston")

    assert action == "conversation"
    assert data["text"] == "Sorry, I couldn't process that. Try again?"


def test_process_user_input_with_conversation_history(claude_client):
    """Conversation history is properly included in the messages array."""
    history = [
        {"role": "user", "content": "Do you know about CAN bus?"},
        {"role": "assistant", "content": "Yes, CAN bus is a serial communication protocol."},
    ]
    mock_resp = make_text_response("What would you like to know?")

    with (
        patch.object(claude_client, "_call_with_retry", return_value=mock_resp) as mock_call,
        patch.object(claude_client, "_track_usage"),
    ):
        claude_client.process_user_input("Tell me more", conversation_history=history)

    # Verify the messages kwarg contains history + current message
    call_kwargs = mock_call.call_args[1]  # keyword arguments
    messages = call_kwargs["messages"]

    assert len(messages) >= 3  # 2 history + 1 current
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Do you know about CAN bus?"
    assert messages[1]["role"] == "assistant"


def test_process_user_input_german_language(claude_client):
    """language='de' prepends German instruction to system prompt."""
    mock_resp = make_text_response("Es ist 15 Uhr.")

    with (
        patch.object(claude_client, "_call_with_retry", return_value=mock_resp) as mock_call,
        patch.object(claude_client, "_track_usage"),
    ):
        claude_client.process_user_input("Wie spät ist es?", language="de")

    call_kwargs = mock_call.call_args[1]
    system = call_kwargs["system"]
    assert system.startswith("IMPORTANT: Respond entirely in German")
