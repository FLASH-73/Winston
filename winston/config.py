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
CAPTURE_INTERVAL = 3.0  # Seconds between frame captures (perception loop)
SCENE_CHANGE_THRESHOLD = 0.15  # 0-1, how much the frame must change to trigger analysis
FRAME_RESOLUTION = (1280, 720)  # Capture resolution

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

# Barge-in / Interruption
BARGEIN_ENABLED = True  # Master switch for barge-in feature
BARGEIN_ENERGY_THRESHOLD = 0.015  # Absolute floor for barge-in trigger (RMS) — lowered so quiet speech triggers
BARGEIN_THRESHOLD_FACTOR = 1.8  # Trigger at Nx the TTS echo level (raised from 1.5: mean-based cal needs higher factor)
BARGEIN_CONSECUTIVE_FRAMES = 2  # Consecutive frames above threshold to trigger (2 × 80ms = 160ms)

# Always-Listening (ambient conversation detection — no wake word needed)
ALWAYS_LISTEN_ENABLED = True  # Master toggle for always-listening mode
ALWAYS_LISTEN_ENERGY_THRESHOLD = (
    0.008  # RMS float32 threshold for speech onset (~0.01-0.05 for speech, ~0.001-0.005 for silence)
)
ALWAYS_LISTEN_SILENCE_DURATION = 1.5  # Seconds of silence to end a speech segment
ALWAYS_LISTEN_TIMEOUT = 15.0  # Max seconds per speech segment (prevents runaway)
ALWAYS_LISTEN_MIN_SPEECH_DURATION = 0.8  # Ignore segments shorter than this (filters coughs, taps)
ALWAYS_LISTEN_COOLDOWN_AFTER_TTS = 1.5  # Wait after TTS stops before re-enabling (avoids echo pickup)
ALWAYS_LISTEN_COOLDOWN_AFTER_RESPONSE = 0.5  # Short debounce after dispatch (echo handled by TTS cooldown)
ALWAYS_LISTEN_CONTINUATION_WINDOW = 8.0  # Seconds after addressed speech to treat new speech as continuation
ALWAYS_LISTEN_STORE_REJECTED = True  # Store non-Winston speech as ambient context observations

# Music / Background Noise Filtering
MUSIC_MODE_ENABLED = False  # Manual toggle: raise thresholds when music is playing
MUSIC_MODE_ENERGY_MULTIPLIER = 5.0  # Multiply energy threshold by this in music mode
MUSIC_MODE_MIN_SPEECH_DURATION = 3.0  # Require longer speech segments in music mode
MUSIC_ENERGY_VARIANCE_THRESHOLD = 0.3  # Skip transcription if energy variance ratio (CV) < this
MUSIC_MAX_CONTINUOUS_DURATION = 10.0  # Skip if speech segment exceeds this without silence gaps

# Streaming / Latency Optimization
STREAMING_ENABLED = True  # Use streaming Claude responses (sentence-by-sentence to TTS)
TTS_STREAMING_PLAYBACK = True  # Use sd.OutputStream instead of sd.play() for lower latency

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
MAX_DAILY_COST_USD = 2.00  # Daily API cost cap
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
MEMORY_CONTEXT_BUDGET_PROACTIVE = 500  # Max tokens for proactive context

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


def get_conversation_prompt() -> str:
    """Generate the conversation system prompt based on personality config."""
    from personality import get_personality

    p = get_personality()

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
- Don't describe what you see in the camera unless asked or there's a safety issue.
- Use camera context silently to inform your answers.
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

Rules:
- Only extract facts about Roberto, his workshop, his equipment, or his projects at Nextis
- Do NOT extract facts about other people mentioned by name (e.g. if someone says "Hey Chávez", don't create facts about Chávez)
- Do NOT extract facts from error messages or system responses like "Sorry, I couldn't process that"
- Only extract facts that are clearly stated or strongly implied
- Prefer specific, actionable facts over vague observations
- Do NOT extract temporary states ("Roberto is at his desk") — only persistent knowledge
- Merge with existing facts when possible (e.g. update a value rather than create duplicates)
- confidence 0.9+ for explicitly stated facts, 0.6-0.8 for inferred facts
- If no facts can be extracted, return an empty array: []

Examples of good facts:
- {"entity": "Roberto", "attribute": "company", "value": "Nextis", "confidence": 0.95, "category": "personal"}
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
- Don't describe what you see unless asked.
- Match Roberto's energy. Quick question = quick answer.

You have tools available:
- Use delegate_to_agent when the user wants you to DO something on the computer: open files, show documents, search the web, navigate apps, investigate code, or any computer action.
- IMPORTANT: If a recent computer task is mentioned in the context (e.g. a document was just opened), and the user refers to it (scroll, find, click, close, go back, etc.), use delegate_to_agent — the agent can see the screen and continue where it left off.
- Use save_note when the user wants to remember or write down something.
- Use shutdown_system when the user says goodbye or wants to go offline.
- For everything else (questions, conversation, opinions), answer directly without using any tool."""
