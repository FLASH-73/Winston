"""
Personality configuration for Winston.

Loads personality from YAML presets (winston/personalities/*.yaml) or uses defaults.
Singleton pattern: call set_personality() once at startup, then get_personality() anywhere.
"""

import logging
from dataclasses import dataclass, field
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
class PersonalityConfig:
    name: str = "Winston"
    style: str = "concise"  # "concise" | "conversational" | "technical"
    formality: float = 0.7  # 0=casual, 1=formal
    humor: bool = False
    proactive_personality: str = "quiet"  # "quiet" | "engaged" | "mentor"
    proactive_threshold: int = 8  # 1-10, only speak if usefulness >= this
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    catchphrases: dict = field(default_factory=lambda: _DEFAULT_CATCHPHRASES)


# Module-level singleton
_personality: Optional[PersonalityConfig] = None


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

    return PersonalityConfig(voice=voice, catchphrases=catchphrases, **config_kwargs)


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
