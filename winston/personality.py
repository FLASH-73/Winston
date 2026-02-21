"""
Personality configuration for Winston.

Loads personality from YAML presets (winston/personalities/*.yaml) or uses defaults.
Singleton pattern: call set_personality() once at startup, then get_personality() anywhere.
Dynamic mood tracking: update_mood() / set_mood() / get_mood_context().
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("winston.personality")

PERSONALITIES_DIR = Path(__file__).parent / "personalities"

# Default catchphrases — exact mirror of previously hardcoded strings in main.py
_DEFAULT_CATCHPHRASES = {
    "en": {
        "greeting": "I'm online. I can see your workshop.",
        "farewell": "Going offline. Goodbye.",
        "error_generic": "Sorry, something went wrong. Try again?",
        "error_process": "Sorry, I couldn't process that. Try again?",
        "error_investigation": "Sorry, the investigation ran into an error.",
        "agent_busy": "I'm still working on the previous task.",
        "agent_starting": "Let me investigate that. I'll let you know what I find.",
        "agent_no_findings": "I couldn't find anything conclusive. Can you give me more details?",
        "agent_previous": "From a previous investigation: {result}",
        "agent_interrupted": "A previous investigation was interrupted: {query}",
        "note_prompt": "What should I write down?",
        "note_confirm": "Got it. Noted: {text}",
        "note_label": "Noted",
        "budget_warning": "Budget limit approaching. I'll reduce my observation frequency.",
    },
    "de": {
        "greeting": "Ich bin online. Ich kann deine Werkstatt sehen.",
        "farewell": "Gehe offline. Tschüss.",
        "error_generic": "Entschuldigung, etwas ist schiefgelaufen. Nochmal versuchen?",
        "error_process": "Entschuldigung, das konnte ich nicht verarbeiten. Nochmal versuchen?",
        "error_investigation": "Entschuldigung, bei der Untersuchung ist ein Fehler aufgetreten.",
        "agent_busy": "Ich arbeite noch an der vorherigen Aufgabe.",
        "agent_starting": "Ich schaue mir das an. Ich melde mich.",
        "agent_no_findings": "Ich konnte nichts Eindeutiges finden. Kannst du mir mehr Details geben?",
        "agent_previous": "Aus einer früheren Untersuchung: {result}",
        "agent_interrupted": "Eine vorherige Untersuchung wurde unterbrochen: {query}",
        "note_prompt": "Was soll ich notieren?",
        "note_confirm": "Notiert: {text}",
        "note_label": "Notiert",
        "budget_warning": "Budgetgrenze erreicht. Ich reduziere meine Beobachtungsfrequenz.",
    },
}


@dataclass
class VoiceConfig:
    voice_id: str = "IRHApOXLvnW57QJPQH2P"
    stability: float = 0.70
    similarity: float = 0.80
    style: float = 0.0


@dataclass
class CharacterConfig:
    core_traits: list[str] = field(default_factory=list)
    forbidden_phrases: list[str] = field(default_factory=list)
    tonal_influences: str = ""
    negative_prompting: str = ""


@dataclass
class CompanionConfig:
    """Prompts for the Curiosity Engine's Telegram companion behavior."""
    reflect_prompt: str = ""
    explore_prompt: str = ""
    craft_prompt: str = ""
    absence_prompt: str = ""


@dataclass
class PersonalityConfig:
    name: str = "Winston"
    style: str = "concise"  # "concise" | "conversational" | "technical"
    formality: float = 0.7  # 0=casual, 1=formal
    humor: bool = False
    proactive_personality: str = "quiet"  # "quiet" | "engaged" | "mentor"
    proactive_threshold: int = 8  # 1-10, only speak if usefulness >= this
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    catchphrases: dict = field(default_factory=lambda: _DEFAULT_CATCHPHRASES)
    character: CharacterConfig = field(default_factory=CharacterConfig)
    companion: CompanionConfig = field(default_factory=CompanionConfig)
    moods: dict[str, str] = field(default_factory=dict)


# Module-level singleton
_personality: Optional[PersonalityConfig] = None

# Dynamic mood state
_current_mood: str = "default"
_mood_context: str = ""


def load_personality(name_or_path: str) -> PersonalityConfig:
    """Load a personality from a YAML preset name or file path.

    Looks for winston/personalities/{name}.yaml first, then treats the arg
    as a direct file path. Missing fields fall back to dataclass defaults.
    """
    # Resolve YAML file
    yaml_path = PERSONALITIES_DIR / f"{name_or_path}.yaml"
    if not yaml_path.is_file():
        yaml_path = Path(name_or_path)
    if not yaml_path.is_file():
        available = [f.stem for f in PERSONALITIES_DIR.glob("*.yaml")]
        raise FileNotFoundError(f"Personality '{name_or_path}' not found. Available: {', '.join(available) or 'none'}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    # Build VoiceConfig from nested dict
    voice_data = data.pop("voice", {})
    voice = VoiceConfig(**{k: v for k, v in voice_data.items() if k in VoiceConfig.__dataclass_fields__})

    # Build CharacterConfig from nested dict (empty dict = backward compatible defaults)
    char_data = data.pop("character", {})
    character = CharacterConfig(**{k: v for k, v in char_data.items() if k in CharacterConfig.__dataclass_fields__})

    # Build CompanionConfig from nested dict
    companion_data = data.pop("companion", {})
    companion = CompanionConfig(**{k: v for k, v in companion_data.items() if k in CompanionConfig.__dataclass_fields__})

    # Extract moods (simple dict of mood_name -> description string)
    moods = data.pop("moods", {})

    # Build catchphrases — merge with defaults so missing keys don't crash
    raw_catchphrases = data.pop("catchphrases", {})
    catchphrases = {}
    for lang in set(list(_DEFAULT_CATCHPHRASES.keys()) + list(raw_catchphrases.keys())):
        defaults = _DEFAULT_CATCHPHRASES.get(lang, {})
        overrides = raw_catchphrases.get(lang, {})
        catchphrases[lang] = {**defaults, **overrides}

    # Build PersonalityConfig — filter to known fields
    known_fields = PersonalityConfig.__dataclass_fields__
    config_kwargs = {k: v for k, v in data.items() if k in known_fields}

    return PersonalityConfig(
        voice=voice, character=character, companion=companion,
        moods=moods, catchphrases=catchphrases, **config_kwargs,
    )


def set_personality(config: PersonalityConfig) -> None:
    """Set the active personality singleton."""
    global _personality
    _personality = config
    logger.info(
        "Personality set: %s (style=%s, proactive=%s, threshold=%d)",
        config.name,
        config.style,
        config.proactive_personality,
        config.proactive_threshold,
    )


def get_personality() -> PersonalityConfig:
    """Get the active personality. Returns defaults if none set."""
    global _personality
    if _personality is None:
        _personality = PersonalityConfig()
    return _personality


def get_catchphrase(key: str, lang: str = "en") -> str:
    """Look up a catchphrase by key and language.

    Falls back to English if the language isn't found, then returns the key
    itself if the catchphrase doesn't exist (safe default — never crashes).
    """
    p = get_personality()
    phrases = p.catchphrases.get(lang, p.catchphrases.get("en", {}))
    return phrases.get(key, p.catchphrases.get("en", {}).get(key, key))


# ── Dynamic mood tracking ─────────────────────────────────────────────

# Single words matched with \b word boundaries to avoid substring false positives
_EMERGENCY_WORDS = [
    "fire", "smoke", "burn", "burning", "injury", "bleeding", "sparks",
    "emergency", "broken", "leak", "leaking", "help",
    # German equivalents
    "feuer", "rauch", "brand", "verletzung", "blutung", "funken",
    "notfall", "kaputt", "leck", "hilfe",
]
# Multi-word phrases matched as literal substrings (already specific enough)
_EMERGENCY_PHRASES = [
    "short circuit", "something is wrong", "help me",
    "kurzschluss", "etwas stimmt nicht", "hilf mir",
]
_EMERGENCY_WORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _EMERGENCY_WORDS) + r")\b",
    re.IGNORECASE,
)

_emergency_set_at: float | None = None  # monotonic timestamp when emergency was set
_EMERGENCY_AUTO_REVERT_SECONDS = 120  # auto-revert after 2 minutes


def _narrative_has_emergency(narrative: str) -> bool:
    """Check if narrative contains actual danger indicators (not profanity)."""
    narrative_lower = narrative.lower()
    if _EMERGENCY_WORD_RE.search(narrative_lower):
        return True
    return any(phrase in narrative_lower for phrase in _EMERGENCY_PHRASES)


def update_mood(session_duration_min: float, narrative: str = "") -> None:
    """Update Winston's mood based on session state. Rule-based, no API calls.

    Priority: emergency > long_session > idle_night > default.
    'success' and 'mistake' are set externally via set_mood().
    Emergency auto-reverts after 2 minutes without continued danger signals.
    """
    global _current_mood, _mood_context, _emergency_set_at

    p = get_personality()
    if not p.moods:
        return

    has_emergency = _narrative_has_emergency(narrative)

    if has_emergency:
        new_mood = "emergency"
    elif _current_mood == "emergency":
        # Auto-revert: emergency clears after 2 min without new danger signals
        if _emergency_set_at and (time.monotonic() - _emergency_set_at) >= _EMERGENCY_AUTO_REVERT_SECONDS:
            new_mood = "default"
            _emergency_set_at = None
            logger.info("Emergency mood auto-reverted after %ds", _EMERGENCY_AUTO_REVERT_SECONDS)
        else:
            new_mood = "emergency"  # hold emergency until timer expires
    elif session_duration_min >= 360:  # 6+ hours
        new_mood = "long_session"
    else:
        hour = datetime.now().hour
        is_night = hour >= 23 or hour < 5
        is_idle = len(narrative.strip()) < 20
        if is_night and is_idle and "idle_night" in p.moods:
            new_mood = "idle_night"
        else:
            new_mood = "default"

    if new_mood != _current_mood and new_mood in p.moods:
        logger.info("Mood shift: %s -> %s", _current_mood, new_mood)
        _current_mood = new_mood
        _mood_context = p.moods.get(new_mood, "")
        if new_mood == "emergency":
            _emergency_set_at = time.monotonic()


def set_mood(mood: str) -> None:
    """Explicitly set mood (e.g., 'success' after a task, 'mistake' on error)."""
    global _current_mood, _mood_context
    p = get_personality()
    if mood in p.moods and mood != _current_mood:
        logger.info("Mood set: %s -> %s", _current_mood, mood)
        _current_mood = mood
        _mood_context = p.moods[mood]


def get_mood_context() -> tuple[str, str]:
    """Return (mood_name, mood_description) for the current mood."""
    return _current_mood, _mood_context
