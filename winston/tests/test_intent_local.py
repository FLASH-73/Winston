"""Tests for classify_intent_local() — pure heuristic intent classification."""

from brain.claude_client import ClaudeClient


def test_winston_keyword_always_addressed():
    """'Winston' in text → always True."""
    assert ClaudeClient.classify_intent_local("Winston, what time is it?") is True


def test_question_word_addressed():
    """First word is a question word → True."""
    assert ClaudeClient.classify_intent_local("What is the torque spec?") is True


def test_short_utterance_not_addressed():
    """Single non-conversational word → False."""
    assert ClaudeClient.classify_intent_local("hmm") is False


def test_conversational_response_in_active_conversation():
    """'yeah' during active conversation → True (it's in CONVERSATIONAL_RESPONSES)."""
    assert ClaudeClient.classify_intent_local("yeah", conversation_active=True) is True


def test_ambiguous_question_mark():
    """Question mark without keyword match → None (needs API)."""
    # "that's" is not in QUESTION_WORDS, no ADDRESSED_KEYWORDS match
    assert ClaudeClient.classify_intent_local("That's interesting?") is None


def test_german_keywords():
    """German addressed keyword 'kannst du' → True."""
    assert ClaudeClient.classify_intent_local("Kannst du mir helfen?") is True


def test_empty_text():
    """Empty string → False."""
    assert ClaudeClient.classify_intent_local("") is False


def test_multi_word_in_conversation():
    """Multi-word utterance during active conversation → True."""
    assert ClaudeClient.classify_intent_local("I think that's correct", conversation_active=True) is True
