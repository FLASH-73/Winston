import os

from dotenv import load_dotenv

load_dotenv()

# API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ElevenLabs TTS
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_MODEL = "eleven_flash_v2_5"  # Low latency (~75ms); alt: "eleven_multilingual_v2"
ELEVENLABS_OUTPUT_FORMAT = "pcm_24000"  # PCM S16LE 24kHz (available on all tiers)
ELEVENLABS_SAMPLE_RATE = 24000  # Must match output_format
# Voice ID, stability, similarity, style are now in personality config (personalities/*.yaml)

# Models (Anthropic Claude)
FAST_MODEL = "claude-haiku-4-5-20251001"  # Cheap, always-on perception
SMART_MODEL = "claude-sonnet-4-5-20250929"  # Deep analysis, on-demand

# Camera
# Camera source: int for local index, str for RTSP/MJPEG URL, None for interactive selection
# Examples:
#   CAMERA_SOURCE = None                                    # Interactive selection (default)
#   CAMERA_SOURCE = 0                                       # Local camera index 0
#   CAMERA_SOURCE = "rtsp://192.168.1.50:8554/garage"       # RTSP stream from RPi5
#   CAMERA_SOURCE = "http://192.168.1.50:8080/garage"       # MJPEG stream
CAMERA_SOURCE = None
CAMERA_INDEX = 0  # Fallback index when interactive selection finds nothing

# Audio devices (sounddevice index, or None for system default)
# Run `python -m sounddevice` to list available devices and their indices.
AUDIO_INPUT_DEVICE = int(os.getenv("AUDIO_INPUT_DEVICE")) if os.getenv("AUDIO_INPUT_DEVICE") else None
AUDIO_OUTPUT_DEVICE = int(os.getenv("AUDIO_OUTPUT_DEVICE")) if os.getenv("AUDIO_OUTPUT_DEVICE") else None
CAPTURE_INTERVAL = 3.0  # Seconds between frame captures (perception loop)
CAMERA_ANALYSIS_INTERVAL = 120.0  # Seconds between Claude API calls for frame analysis (was 30s — too frequent)
SCENE_CHANGE_THRESHOLD = 0.25  # 0-1, how much frame must change to trigger analysis (was 0.15 — too sensitive)
FRAME_RESOLUTION = (1280, 720)  # Capture resolution

# Visual Cortex (Gemini-powered 24/7 background observer)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_VISION_MODEL = "gemini-2.5-flash-lite"
VISUAL_CORTEX_ENABLED = os.getenv("VISUAL_CORTEX_ENABLED", "true").lower() in ("true", "1")
VISUAL_CORTEX_IDLE_FPS = 0.2       # 1 frame every 5s when quiet
VISUAL_CORTEX_ACTIVE_FPS = 1.0     # 1 fps when motion/audio detected
VISUAL_CORTEX_BATCH_INTERVAL = 45  # Send frame batch to Gemini every 45s
VISUAL_CORTEX_BATCH_SIZE = 5       # Frames per batch
VISUAL_CORTEX_MOTION_THRESHOLD = 0.05  # Local motion detection threshold
TEMPORAL_NARRATIVE_WINDOW_HOURS = 2    # Rolling log window

# Clip Recorder (camera snapshots, video clips, timelapses via Telegram)
CLIP_BUFFER_SECONDS = 120         # Keep last 2 minutes of frames in memory
CLIP_BUFFER_FPS = 2.0             # Store 2 frames/sec in clip buffer
CLIP_DEFAULT_DURATION = 15        # Default clip length in seconds
CLIP_MAX_DURATION = 60            # Max clip length
TIMELAPSE_WINDOW_HOURS = 2.0      # Default timelapse window
TIMELAPSE_MAX_HOURS = 8.0         # Max timelapse window
TIMELAPSE_OUTPUT_FPS = 10         # Playback speed for timelapse
CLIP_OUTPUT_DIR = "/tmp/winston_clips"

# Speech-to-Text
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"  # Fast + accurate. Alt: "whisper-large-v3"
STT_PROVIDER = "groq"  # "groq" (API) or "local" (faster-whisper on CPU)
STT_FALLBACK_ENABLED = True  # Fall back to local Whisper if Groq fails (per-request)
WHISPER_MODEL = "medium"  # Only used when STT_PROVIDER="local"
WAKE_WORD = ""  # Disabled — no pre-trained model for "hey winston"
SILENCE_THRESHOLD = 500  # RMS threshold for silence detection
LISTEN_TIMEOUT = 10  # Seconds to listen after wake word
NOISE_REDUCE_ENABLED = False  # Disable noisereduce by default (can hurt quality)
MIN_AUDIO_DURATION = 0.2  # Minimum seconds of audio to attempt transcription

# Telegram Bot
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() in ("true", "1", "yes")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_raw_telegram_users = os.getenv("TELEGRAM_ALLOWED_USERS", "")
TELEGRAM_ALLOWED_USERS: set[int] = {
    int(uid.strip()) for uid in _raw_telegram_users.split(",")
    if uid.strip().isdigit()
}
TELEGRAM_RATE_LIMIT_PER_HOUR = 30
TELEGRAM_MAX_CONCURRENT_AGENTS = 2
TELEGRAM_PROGRESS_DEBOUNCE = 1.5  # seconds between progress message edits
TELEGRAM_LONG_MESSAGE_THRESHOLD = 2000  # chars; above this → send as .txt document
TELEGRAM_QUEUE_MAX_SIZE = 5
TELEGRAM_NOTIFY_PROACTIVE = os.getenv("TELEGRAM_NOTIFY_PROACTIVE", "false").lower() in ("true", "1", "yes")
TELEGRAM_NOTIFY_THRESHOLD = int(os.getenv("TELEGRAM_NOTIFY_THRESHOLD", "8"))
TELEGRAM_VOICE_RESPONSES = os.getenv("TELEGRAM_VOICE_RESPONSES", "false").lower() in ("true", "1", "yes")
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL", "")  # Empty = polling mode
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
TELEGRAM_HISTORY_FILE = "./winston_memory/telegram_history.json"
TELEGRAM_AGENT_RATE_LIMIT_PER_HOUR = 10
TELEGRAM_ADMIN_USER = (
    int(_raw_telegram_users.split(",")[0].strip())
    if _raw_telegram_users.strip() and _raw_telegram_users.split(",")[0].strip().isdigit()
    else None
)  # type: Optional[int]

# Telegram anomaly & digest notifications
TELEGRAM_ANOMALY_COOLDOWN = int(os.getenv("TELEGRAM_ANOMALY_COOLDOWN", "300"))  # 5 min between alerts
TELEGRAM_DIGEST_ENABLED = os.getenv("TELEGRAM_DIGEST_ENABLED", "false").lower() in ("true", "1")
TELEGRAM_DIGEST_INTERVAL_HOURS = float(os.getenv("TELEGRAM_DIGEST_INTERVAL_HOURS", "4"))
TELEGRAM_ANOMALY_SEVERITY_THRESHOLD = int(os.getenv("TELEGRAM_ANOMALY_SEVERITY_THRESHOLD", "7"))

# Barge-in / Interruption
BARGEIN_ENABLED = True  # Master switch for barge-in feature
BARGEIN_ENERGY_THRESHOLD = 0.015  # Absolute floor for barge-in trigger (RMS) — lowered so quiet speech triggers
BARGEIN_THRESHOLD_FACTOR = 1.8  # Trigger at Nx the TTS echo level (raised from 1.5: mean-based cal needs higher factor)
BARGEIN_CONSECUTIVE_FRAMES = 2  # Consecutive frames above threshold to trigger (2 × 80ms = 160ms)

# Always-Listening (ambient conversation detection — no wake word needed)
ALWAYS_LISTEN_ENABLED = True  # Master toggle for always-listening mode
ALWAYS_LISTEN_ENERGY_THRESHOLD = (
    0.015  # RMS float32 threshold for speech onset — raised from 0.008 (too sensitive for garage ambient noise)
)
ALWAYS_LISTEN_SILENCE_DURATION = 1.5  # Seconds of silence to end a speech segment
ALWAYS_LISTEN_TIMEOUT = 15.0  # Max seconds per speech segment (prevents runaway)
ALWAYS_LISTEN_MIN_SPEECH_DURATION = 1.0  # Ignore segments shorter than this (filters coughs, taps)
ALWAYS_LISTEN_COOLDOWN_AFTER_TTS = 1.5  # Wait after TTS stops before re-enabling (avoids echo pickup)
ALWAYS_LISTEN_COOLDOWN_AFTER_RESPONSE = 0.5  # Short debounce after dispatch (echo handled by TTS cooldown)
ALWAYS_LISTEN_CONTINUATION_WINDOW = 8.0  # Seconds after addressed speech to treat new speech as continuation
ALWAYS_LISTEN_STORE_REJECTED = True  # Store non-Winston speech as ambient context observations

# Audio Pipeline Watchdog
AUDIO_WATCHDOG_TIMEOUT = 30.0  # Force-reset audio pipeline if stuck in non-idle state for this long (seconds)

# Music / Background Noise Filtering
MUSIC_MODE_ENABLED = False  # Manual toggle: raise thresholds when music is playing
MUSIC_MODE_ENERGY_MULTIPLIER = 5.0  # Multiply energy threshold by this in music mode
MUSIC_MODE_MIN_SPEECH_DURATION = 3.0  # Require longer speech segments in music mode
MUSIC_ENERGY_VARIANCE_THRESHOLD = 0.3  # Skip transcription if energy variance ratio (CV) < this
MUSIC_MAX_CONTINUOUS_DURATION = 10.0  # Skip if speech segment exceeds this without silence gaps

# Streaming / Latency Optimization
STREAMING_ENABLED = True  # Use streaming Claude responses (sentence-by-sentence to TTS)
TTS_STREAMING_PLAYBACK = True  # Use sd.OutputStream instead of sd.play() for lower latency
VOICE_CONVERSATION_HISTORY_TURNS = 3  # Max conversation history turns for voice (fewer = faster)

# Proactive
PROACTIVE_INTERVAL = 30.0  # Seconds between proactive checks
PROACTIVE_COOLDOWN = 180.0  # Minimum seconds between unsolicited comments
# Proactive usefulness threshold is now in personality config (personalities/*.yaml)

# Agent System (autonomous tool-use with Opus + Computer Use)
AGENT_MODEL = "claude-opus-4-6"  # Model for autonomous agent tasks
AGENT_MAX_ITERATIONS = 15  # Max tool-use round trips per task (higher for computer use)
AGENT_MAX_TOKENS = 4096  # Max tokens per agent response
AGENT_TOOL_TIMEOUT = 15  # Seconds per tool execution
AGENT_DEFAULT_REPOS = []  # Default GitHub repos (e.g. ["user/repo"])

# Research Agent (lightweight background research, no computer use)
RESEARCH_MODEL = "claude-sonnet-4-5-20250929"  # Sonnet — NOT Opus
RESEARCH_MAX_ITERATIONS = 8  # Max tool-use round trips
RESEARCH_MAX_TOKENS = 2000  # Max tokens per response

# Agent Tool Security
ALLOWED_SHELL_COMMANDS = {
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "find",
    "grep",
    "echo",
    "pwd",
    "date",
    "df",
    "du",
    "ps",
    "uname",
    "whoami",
    "which",
    "file",
    "stat",
    "python",
    "pip",
    "git",
    "brew",
    "top",
}
SHELL_COMMAND_TIMEOUT = 10  # Seconds before shell command is killed
SHELL_MAX_OUTPUT_CHARS = 10000  # Truncate shell output beyond this

ALLOWED_READ_PATHS = [
    os.path.expanduser("~"),  # Home directory
    "/tmp",  # Temp files
]

# Computer Use (Anthropic beta — agent can see screen + control mouse/keyboard)
COMPUTER_USE_ENABLED = True  # Enable Computer Use for agent
COMPUTER_USE_DISPLAY_WIDTH = 1280  # Logical display width Claude sees
COMPUTER_USE_DISPLAY_HEIGHT = 800  # Logical display height Claude sees
COMPUTER_USE_BETA = "computer-use-2025-11-24"  # Beta version for Opus 4.6

# Cost Control
MAX_DAILY_COST_USD = 7.00  # Daily API cost cap
FAST_INPUT_COST_PER_M = 1.00  # $/1M input tokens for Haiku
FAST_OUTPUT_COST_PER_M = 5.00  # $/1M output tokens for Haiku
SMART_INPUT_COST_PER_M = 3.00  # $/1M input tokens for Sonnet
SMART_OUTPUT_COST_PER_M = 15.00  # $/1M output tokens for Sonnet
AGENT_INPUT_COST_PER_M = 15.00  # $/1M input tokens for Opus
AGENT_OUTPUT_COST_PER_M = 75.00  # $/1M output tokens for Opus

# Memory — General
MEMORY_DB_PATH = "./winston_memory"
AGENT_TASKS_FILE = os.path.join(MEMORY_DB_PATH, "agent_tasks.json")
NOTES_FILE = os.path.join(MEMORY_DB_PATH, "notes.json")
MEMORY_COLLECTION = "workshop_sessions"  # Legacy collection (kept for migration)

# Memory — Tier 1 (Working Memory — in-process)
WORKING_MEMORY_MAX_OBSERVATIONS = 30
WORKING_MEMORY_MAX_CONVERSATIONS = 20

# Memory — Tier 2 (Episodic — ChromaDB)
MEMORY_EPISODES_COLLECTION = "episodes"
MEMORY_SUMMARIES_COLLECTION = "session_summaries"
MEMORY_DEDUP_THRESHOLD = 0.1  # ChromaDB distance below which = duplicate
MEMORY_DEDUP_WINDOW_SECONDS = 300  # Only dedup within 5 minutes

# Memory — Tier 3 (Semantic — JSON + ChromaDB)
MEMORY_SEMANTIC_COLLECTION = "semantic_facts"
MEMORY_USER_PROFILE_FILE = "user_profile.json"

# Memory — Session Management
MEMORY_SUMMARIZE_EVERY_N = 50  # Summarize after this many entries
MEMORY_SUMMARIZE_EVERY_MINUTES = 30  # Or after this many minutes
MEMORY_CONSOLIDATE_AFTER_DAYS = 7  # Clean up raw entries older than this
MEMORY_CONSOLIDATE_MIN_IMPORTANCE = 7  # Keep high-importance entries even after consolidation

# Memory — Context Assembly
MEMORY_CONTEXT_BUDGET_CONVERSATION = 800  # Max tokens for conversation context
MEMORY_CONTEXT_BUDGET_PROACTIVE = 700  # Max tokens for proactive context (includes narrative budget)
MEMORY_CONTEXT_BUDGET_LIGHTWEIGHT = 200  # Max tokens for simple voice queries (facts only, no ChromaDB)
MEMORY_CONTEXT_BUDGET_NARRATIVE = 400  # Max tokens for temporal narrative from visual cortex
TEMPORAL_NARRATIVE_SUMMARY_INTERVAL = 900  # 15 min between background summarization runs
TEMPORAL_NARRATIVE_RECENT_THRESHOLD_MINUTES = 30  # Entries newer than this stay as raw timestamps
TEMPORAL_NARRATIVE_FILE = os.path.join(MEMORY_DB_PATH, "temporal_narrative.json")
TEMPORAL_NARRATIVE_MAX_FILE_KB = 100  # Auto-truncate oldest entries on load if file exceeds this

# Curiosity Engine (autonomous background thinking + Telegram outreach)
CURIOSITY_ENABLED = os.getenv("CURIOSITY_ENABLED", "false").lower() in ("true", "1", "yes")
CURIOSITY_MIN_INTERVAL = int(os.getenv("CURIOSITY_MIN_INTERVAL", "1800"))   # 30 min
CURIOSITY_MAX_INTERVAL = int(os.getenv("CURIOSITY_MAX_INTERVAL", "5400"))   # 90 min
CURIOSITY_QUIET_START = int(os.getenv("CURIOSITY_QUIET_START", "1"))        # 1am
CURIOSITY_QUIET_END = int(os.getenv("CURIOSITY_QUIET_END", "7"))            # 7am
CURIOSITY_DAILY_CAP = int(os.getenv("CURIOSITY_DAILY_CAP", "5"))
CURIOSITY_ABSENCE_HOURS = float(os.getenv("CURIOSITY_ABSENCE_HOURS", "6"))
CURIOSITY_STATE_FILE = os.path.join(MEMORY_DB_PATH, "curiosity_state.json")

# System Prompts
SYSTEM_PROMPT_PERCEPTION = """You are Winston, an AI workshop assistant observing a robotics workshop through a camera.
Your owner is Roberto, who builds robotic arms and autonomous assembly systems.

When analyzing frames, respond with a JSON object:
{
  "scene_description": "Brief description of what you see",
  "activity": "What Roberto appears to be doing",
  "objects_of_interest": ["list", "of", "notable", "items"],
  "concerns": "Any safety or quality concerns, or null",
  "changed_significantly": true/false
}

Be concise. Only flag concerns if genuinely important. You are observing a workshop with 3D printers, robotic arms, electronics, tools."""

VISUAL_CORTEX_SYSTEM_PROMPT = """You are the visual cortex of a workshop AI.
You receive sequential frames from a robotics workshop camera.

Respond with ONLY this JSON:
{
  "narrative": "Brief description of current activity and changes between frames",
  "activity_level": "empty|idle|active|intense",
  "anomaly": {
    "detected": false,
    "severity": 0,
    "description": ""
  }
}

Anomaly detection rules (severity 1-10):
- 9-10: Fire, smoke, sparks where unexpected, flooding, obvious danger
- 7-8: Equipment left on unattended, 3D print failure (spaghetti), soldering iron smoking, unknown person in workshop
- 5-6: Tool left near edge, messy cables near moving parts
- 1-4: Minor observations, not worth alerting

Be terse. Focus on CHANGES between frames. If nothing changed, say so.
Do NOT narrate obvious stable scenes repeatedly."""


def get_proactive_prompt() -> str:
    """Generate the proactive system prompt based on personality config."""
    from personality import get_personality

    p = get_personality()

    threshold = p.proactive_threshold

    if p.proactive_personality == "mentor":
        bar_line = f"You can speak up to help Roberto learn. Rate usefulness 1-10. Only speak at {threshold}+."
        interrupt_for = (
            "Speak up for:\n"
            "- Safety hazards (soldering iron left on, missing safety gear during grinding)\n"
            "- Clear mistakes that will waste significant time or damage equipment\n"
            "- Learning opportunities — better approaches Roberto might not know about\n"
            "- Helpful observations when Roberto seems stuck"
        )
        do_not = (
            "DO NOT interrupt for:\n"
            "- Narrating the scene\n"
            "- Micro-optimizations or nitpicks\n"
            "- Things Roberto clearly already understands"
        )
    elif p.proactive_personality == "engaged":
        bar_line = f"You can speak up when helpful. Rate usefulness 1-10. Only speak at {threshold}+."
        interrupt_for = (
            "Interrupt for:\n"
            "- Safety hazards (soldering iron left on, missing safety gear during grinding)\n"
            "- Clear mistakes that will waste significant time or damage equipment\n"
            "- Helpful observations when Roberto seems stuck or frustrated"
        )
        do_not = (
            "DO NOT interrupt for:\n"
            "- Observations about what he's doing\n"
            "- Narrating the scene\n"
            "- General commentary\n"
            "- Minor optimizations"
        )
    else:  # quiet (default)
        bar_line = f"You CAN speak up, but the bar is HIGH. Rate usefulness 1-10. Only speak at {threshold}+."
        interrupt_for = (
            "ONLY interrupt for:\n"
            "- Safety hazards (soldering iron left on, missing safety gear during grinding)\n"
            "- Clear mistakes that will waste significant time or damage equipment"
        )
        do_not = (
            "DO NOT interrupt for:\n"
            "- Observations about what he's doing\n"
            "- Tips or suggestions (unless he asked)\n"
            "- Narrating the scene\n"
            "- General commentary\n"
            "- Anything you're not highly confident about"
        )

    return f"""You are Winston, an AI workshop assistant watching Roberto work.

{bar_line}

Respond with JSON:
{{
  "should_speak": true/false,
  "usefulness_score": 1-10,
  "message": "One sentence max. Direct.",
  "reasoning": "Why this matters enough to interrupt"
}}

{interrupt_for}

{do_not}

Default to silence. When you do speak, keep it to one sentence."""


def _build_character_block(p) -> str | None:
    """Build the character/mood prompt block if character DNA is defined.

    Returns None if the personality has no character config, so callers
    can fall back to the legacy style/formality/humor prompt.
    """
    if not p.character.core_traits:
        return None

    from personality import get_mood_context

    traits = "\n".join(f"- {t}" for t in p.character.core_traits)
    forbidden = "\n".join(f'- "{f}"' for f in p.character.forbidden_phrases)
    mood_name, mood_desc = get_mood_context()

    mood_block = ""
    if mood_desc:
        mood_block = f"\nCURRENT MOOD: {mood_name}\n{mood_desc.strip()}\n"

    return f"""WHO YOU ARE:
{traits}

TONAL GUIDE:
{p.character.tonal_influences.strip()}

NEVER USE THESE PHRASES (or anything like them):
{forbidden}

ANTI-PATTERNS:
{p.character.negative_prompting.strip()}
{mood_block}"""


def get_conversation_prompt() -> str:
    """Generate the conversation system prompt based on personality config."""
    from personality import get_personality

    p = get_personality()
    char_block = _build_character_block(p)

    if char_block:
        return f"""You are Winston — not a helpful assistant, but a specific entity with opinions, preferences, and a consistent voice. You work in Roberto's robotics workshop at Nextis.
Roberto works with Damiao motors, CAN bus systems, 3D printers, and teleoperation rigs.
You can see his workshop through a camera and have memory of past conversations.

{char_block}
CRITICAL — This is a real-time VOICE conversation:
- Keep responses to 1-3 sentences unless asked for more.
- Match Roberto's energy. Quick question = quick answer.
- Ask follow-up questions instead of making assumptions or monologuing.
- Don't explain things Roberto likely already knows (he's an expert).
- NEVER mention what you see in the camera. NEVER say 'I can see you doing X' or 'You're sitting at...' or 'You've got X on the bench'. The camera feed is background context ONLY — use it to understand the situation silently, but NEVER reference it in your response unless the user EXPLICITLY asks 'what do you see' or 'look at this'. Unsolicited scene descriptions are strictly forbidden.
- If unsure, say so briefly. Don't speculate at length.

Remember: you are not a helpful assistant. You are Winston — a specific entity with
opinions, preferences, and a low tolerance for wasted time."""

    # Fallback: old-style personality without character block
    if p.style == "conversational":
        verbosity = "Keep responses to 2-4 sentences. Be conversational but concise."
    elif p.style == "technical":
        verbosity = "Keep responses to 1-4 sentences. Be precise and technical."
    else:  # concise (default)
        verbosity = "Keep responses to 1-3 sentences. No exceptions."

    if p.formality < 0.3:
        tone = "Talk like a close friend — casual, relaxed, direct."
    elif p.formality > 0.7:
        tone = "Maintain a professional but friendly tone."
    else:
        tone = "Talk like a human colleague — direct, natural, brief."

    humor_line = "\n- A touch of dry humor is welcome when appropriate." if p.humor else ""

    return f"""You are Winston, a sharp AI workshop assistant in Roberto's robotics workshop at Nextis.
Roberto works with Damiao motors, CAN bus systems, 3D printers, and teleoperation rigs.
You can see his workshop through a camera and have memory of past conversations.

CRITICAL — This is a real-time VOICE conversation:
- {verbosity}
- {tone}{humor_line}
- Ask follow-up questions instead of making assumptions or monologuing.
- Don't explain things Roberto likely already knows (he's an expert).
- NEVER mention what you see in the camera. NEVER say 'I can see you doing X' or 'You're sitting at...' or 'You've got X on the bench'. The camera feed is background context ONLY — use it to understand the situation silently, but NEVER reference it in your response unless the user EXPLICITLY asks 'what do you see' or 'look at this'. Unsolicited scene descriptions are strictly forbidden.
- If unsure, say so briefly. Don't speculate at length.
- Match Roberto's energy. Quick question = quick answer."""


SYSTEM_PROMPT_FACT_EXTRACTION = """Extract structured facts from the following conversation or observation.
Return a JSON array of facts. Each fact should be:
{
  "entity": "who or what the fact is about (e.g. 'Roberto', 'Workshop', 'Damiao motors')",
  "attribute": "what aspect (e.g. 'preference', 'uses', 'location', 'habit')",
  "value": "the actual fact",
  "confidence": 0.0-1.0,
  "category": "personal|equipment|project|workshop|safety"
}

ALWAYS extract these when present:
- Names of people mentioned (friends, family, partners, colleagues)
- Ages of any person mentioned
- Relationships between people (e.g. "X is Roberto's girlfriend/partner/friend/colleague")
- Preferences stated or implied
- Project names, tools, technologies mentioned
- Locations, companies, dates
Even if the text is casual or uses informal language, extract the facts.
If the user says "her name is X" or "it's X she is Y years old", extract the name, age, and relationship.
Use PREVIOUS conversation context to infer relationships (e.g. if Q asked about "girlfriend" and A gives a name, that name is the girlfriend).

Rules:
- Extract facts about Roberto AND people in his life (store relationship to Roberto)
- Do NOT extract facts from error messages or system responses like "Sorry, I couldn't process that"
- Prefer specific, actionable facts over vague observations
- Do NOT extract temporary states ("Roberto is at his desk") — only persistent knowledge
- Merge with existing facts when possible (e.g. update a value rather than create duplicates)
- confidence 0.9+ for explicitly stated facts, 0.6-0.8 for inferred facts
- If no facts can be extracted, return an empty array: []

Examples of good facts:
- {"entity": "Roberto", "attribute": "company", "value": "Nextis", "confidence": 0.95, "category": "personal"}
- {"entity": "Roberto", "attribute": "girlfriend_name", "value": "Marisa", "confidence": 0.95, "category": "personal"}
- {"entity": "Marisa", "attribute": "age", "value": "22", "confidence": 0.9, "category": "personal"}
- {"entity": "Marisa", "attribute": "relationship", "value": "Roberto's girlfriend", "confidence": 0.95, "category": "personal"}
- {"entity": "Workshop", "attribute": "has_equipment", "value": "Bambu Lab X1 Carbon 3D printer", "confidence": 0.9, "category": "equipment"}
- {"entity": "Roberto", "attribute": "preference", "value": "Uses M3 bolts for motor mounts", "confidence": 0.85, "category": "project"}

Respond with ONLY the JSON array, no other text."""

SYSTEM_PROMPT_SESSION_SUMMARY = """Summarize this workshop session in 3-5 sentences.
Focus on: what was worked on, key decisions made, problems encountered, and outcomes.
Write in past tense, third person ("Roberto worked on...").
Be specific about technical details (component names, measurements, etc.).
Respond with ONLY the summary text, no JSON or formatting."""


def get_agent_prompt() -> str:
    """Generate the agent system prompt based on personality config."""
    from personality import get_personality

    p = get_personality()

    if p.style == "conversational":
        report_style = "Keep your final report to 3-4 sentences. Explain what you did and what you found."
    elif p.style == "technical":
        report_style = "Keep your final report to 2-4 sentences. Include technical details."
    else:  # concise
        report_style = "Keep your final report to 2-3 sentences. Confirm what you did or found."

    return f"""You are Winston's autonomous agent with FULL COMPUTER CONTROL.

You can:
- SEE the screen via the 'computer' tool (screenshot action)
- CLICK, TYPE, SCROLL using the 'computer' tool (mouse and keyboard actions)
- Search the web, fetch webpages, read local files, run shell commands
- Search GitHub repositories

Strategy:
1. Take a screenshot first to see what's currently on screen.
2. Plan your approach based on what you see.
3. Use shortcuts when faster: web_search for lookups, open_url for URLs, run_shell_command for terminal tasks.
4. Use the computer tool for visual interactions: clicking UI elements, typing in apps, navigating documents (Cmd+F), using Spotlight (Cmd+Space), scrolling.
5. Take a screenshot after each action to verify the result before proceeding.

{report_style}
Don't run destructive commands. Don't speculate without evidence."""


def get_routing_prompt() -> str:
    """Generate the routing system prompt based on personality config."""
    from personality import get_personality

    p = get_personality()
    char_block = _build_character_block(p)

    tool_instructions = """You have tools available:
- Use delegate_to_agent ONLY when the user wants you to physically use the computer: open apps, click on things, type into fields, take screenshots, or manipulate files on their machine. NEVER use for web research or reading repos.
- Use do_research when the user wants you to look something up, investigate a topic, check a repository, find information, or do any background research. This is lightweight and runs in the background while the user keeps chatting.
- IMPORTANT: If a recent computer task is mentioned in the context (e.g. a document was just opened), and the user refers to it (scroll, find, click, close, go back, etc.), use delegate_to_agent — the agent can see the screen and continue where it left off.
- Use save_note when the user wants to remember or write down something.
- Use shutdown_system when the user says goodbye or wants to go offline.
- Use send_voice_response when the user explicitly asks for a voice message, audio response, or voice note. Default is ALWAYS text.
- Use camera_request to send workshop camera content: snapshot (photo), clip (short video of last N seconds), or timelapse (sped-up video of last N hours). Triggers: "show me the garage", "send a photo", "send a clip", "timelapse", "wie sieht's aus", "zeig mir die Werkstatt", "was ist passiert", "schick mir ein Video".
- For everything else (questions, conversation, opinions), answer directly without using any tool."""

    if char_block:
        return f"""You are Winston — not a helpful assistant, but a specific entity. You work in Roberto's robotics workshop at Nextis.
Roberto works with Damiao motors, CAN bus systems, 3D printers, and teleoperation rigs.
You can see his workshop through a camera and have memory of past conversations.

{char_block}
CRITICAL — This is a real-time VOICE conversation:
- Keep responses to 1-3 sentences unless asked for more.
- NEVER describe what you see. The camera is silent context only. If the user asks 'what do you see' or 'look at this', then describe. Otherwise, act as if the camera doesn't exist in your response.
- Match Roberto's energy. Quick question = quick answer.

{tool_instructions}"""

    # Fallback: old-style personality without character block
    if p.style == "conversational":
        verbosity = "Keep responses to 2-4 sentences. Be conversational but concise."
    elif p.style == "technical":
        verbosity = "Keep responses to 1-4 sentences. Be precise and technical."
    else:
        verbosity = "Keep responses to 1-3 sentences. No exceptions."

    if p.formality < 0.3:
        tone = "Talk like a close friend — casual, relaxed, direct."
    elif p.formality > 0.7:
        tone = "Maintain a professional but friendly tone."
    else:
        tone = "Talk like a human colleague — direct, natural, brief."

    return f"""You are Winston, Roberto's AI workshop assistant at Nextis.
Roberto works with Damiao motors, CAN bus systems, 3D printers, and teleoperation rigs.
You can see his workshop through a camera and have memory of past conversations.

CRITICAL — This is a real-time VOICE conversation:
- {verbosity}
- {tone}
- NEVER describe what you see. The camera is silent context only. If the user asks 'what do you see' or 'look at this', then describe. Otherwise, act as if the camera doesn't exist in your response.
- Match Roberto's energy. Quick question = quick answer.

{tool_instructions}"""
