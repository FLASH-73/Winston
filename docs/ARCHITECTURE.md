# WINSTON Workshop Assistant

## Overview

Winston is an AI workshop assistant that runs on a MacBook in Roberto's robotics workshop at Nextis. It watches what's happening through a camera, listens continuously for speech (no wake word needed), and can proactively comment when something genuinely important comes up — like a safety concern or a clear mistake.

Winston behaves like a sharp, quiet colleague. He's always paying attention, but only speaks up when he has something worth saying. He answers questions directly in 1-3 sentences, can take autonomous action on the computer (open files, search the web, navigate apps), remembers context across sessions, and supports both English and German.

### Architecture: Two-Tier Claude + Autonomous Agent

```
                                    ┌──────────────────┐
                         ┌─────────►│  Claude Opus 4.6  │ (agent: Computer Use + 14 tools)
                         │          └──────────────────┘
┌──────────┐    ┌────────┴───────┐  ┌──────────────────┐
│ Camera   │───►│  Orchestrator  │─►│  Claude Haiku     │ (routing + conversation + perception)
│ Mic/STT  │───►│  (main.py)     │  └──────────────────┘
│ Memory   │◄──►│                │  ┌──────────────────┐
└──────────┘    └────────┬───────┘  │  ElevenLabs TTS   │ (streaming, 75ms latency)
                         │          └──────────────────┘
                         ├─────────►│  ChromaDB         │ (3-tier memory)
                         │          └──────────────────┘
                         └─────────►│  Dashboard        │ (FastAPI + WebSocket)
                                    └──────────────────┘
```

- **Haiku** (cheap, fast): Handles all routing decisions, conversation, frame analysis, proactive checks, and fact extraction. Uses tool-use to decide intent — no keyword matching.
- **Opus 4.6** (powerful, on-demand): Runs autonomous multi-step tasks with full Computer Use (screenshot, mouse, keyboard) plus 14 shortcut tools. Only invoked when the user asks Winston to *do* something on the computer.

### Intelligent Routing (No Keywords)

User speech goes through `process_user_input()` in `claude_client.py`, which gives Haiku four tools:

| Tool | When Haiku calls it |
|------|-------------------|
| `delegate_to_agent` | User wants something done on the computer (open, search, navigate, etc.) |
| `save_note` | User wants to remember something |
| `shutdown_system` | User says goodbye / wants to go offline |
| `get_current_time` | User asks what time or date it is |

If Haiku doesn't call any tool, it answers directly in 1-3 sentences. This eliminates keyword-based routing entirely — Haiku understands intent natively across languages.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11+ |
| **LLM** | Anthropic Claude (Haiku 4.5 + Opus 4.6) |
| **TTS** | ElevenLabs (`eleven_flash_v2_5`, 75ms latency) + pyttsx3 fallback |
| **STT** | `faster-whisper` with `medium` model (multilingual, CPU-friendly) |
| **Camera** | OpenCV (`opencv-python-headless`) |
| **Memory** | ChromaDB (vector DB) + JSON (semantic facts, notes, agent tasks) |
| **Wake Word** | `openwakeword` (available but disabled — always-listening used instead) |
| **Web Search** | `ddgs` (DuckDuckGo) |
| **Dashboard** | FastAPI + vanilla JS, WebSocket real-time updates |
| **Computer Control** | `cliclick` (mouse) + `osascript` (keyboard) + `screencapture` |

### Dependencies (requirements.txt)

```
opencv-python-headless>=4.8
numpy
anthropic>=0.40.0
chromadb>=0.4
faster-whisper>=1.0.0
openwakeword>=0.6
pyttsx3
elevenlabs>=2.0
python-dotenv
Pillow
sounddevice
scipy
fastapi>=0.100
uvicorn[standard]>=0.20
websockets>=11.0
ddgs>=9.0
```

---

## Project Structure

```
winston/
├── main.py                      # Winston class — orchestrator, event loop, routing
├── config.py                    # All config: models, thresholds, system prompts
├── requirements.txt
├── test_claude.py               # API diagnostic tool
├── .env                         # ANTHROPIC_API_KEY, ELEVENLABS_API_KEY
│
├── brain/
│   ├── claude_client.py         # ClaudeClient: process_user_input() routing, chat, analyze_frame
│   ├── agent_executor.py        # AgentExecutor: Opus agentic loop with tools + Computer Use
│   ├── agent_tools.py           # ToolRegistry + 14 tools (GitHub, web, shell, file)
│   ├── computer_use.py          # MacOSComputerController (screenshot, mouse, keyboard)
│   ├── memory.py                # 3-tier memory (Working + Episodic + Semantic)
│   └── proactive.py             # ProactiveEngine: observes, speaks up for safety/mistakes
│
├── perception/
│   ├── camera.py                # Camera capture, scene change detection (SSIM)
│   ├── audio.py                 # AudioListener: mic, STT, always-listening, barge-in
│   └── tts.py                   # TTSEngine: ElevenLabs streaming + pyttsx3 fallback
│
├── utils/
│   ├── persistent_store.py      # PersistentStore: thread-safe atomic JSON
│   ├── cost_tracker.py          # CostTracker: daily API budget enforcement
│   ├── echo_cancel.py           # EnergyBargeInDetector + echo text rejection
│   └── frame_diff.py            # compute_scene_change() for frame differencing
│
├── dashboard/
│   ├── server.py                # FastAPI + WebSocket real-time updates (port 8420)
│   └── state.py                 # WinstonState: thread-safe shared state
│
└── winston_memory/              # Persistent storage
    ├── agent_tasks.json         # Agent task history
    └── notes.json               # User notes/reminders
```

---

## Module Specifications

### config.py

Central configuration. All tunable values live here. Key sections:

```python
# Models (Anthropic Claude)
FAST_MODEL = "claude-haiku-4-5-20251001"      # Routing, conversation, perception
SMART_MODEL = "claude-sonnet-4-5-20250929"    # Deep analysis (reserved)
AGENT_MODEL = "claude-opus-4-6"               # Autonomous agent with Computer Use

# ElevenLabs TTS
ELEVENLABS_MODEL = "eleven_flash_v2_5"        # 75ms latency, 32 languages
ELEVENLABS_VOICE_ID = "IRHApOXLvnW57QJPQH2P" # Adam
ELEVENLABS_OUTPUT_FORMAT = "pcm_24000"        # PCM S16LE 24kHz

# Camera
CAPTURE_INTERVAL = 3.0                        # Seconds between frame captures
SCENE_CHANGE_THRESHOLD = 0.15                 # 0-1, triggers analysis
FRAME_RESOLUTION = (1280, 720)

# Audio
WHISPER_MODEL = "medium"                      # Multilingual, CPU-friendly
ALWAYS_LISTEN_ENABLED = True                  # No wake word needed
ALWAYS_LISTEN_ENERGY_THRESHOLD = 0.008        # RMS for speech onset
BARGEIN_ENABLED = True                        # Interrupt TTS by speaking
BARGEIN_THRESHOLD_FACTOR = 2.0                # 2x TTS echo level to trigger
BARGEIN_CONSECUTIVE_FRAMES = 3                # 240ms sustained speech

# Agent System
AGENT_MAX_ITERATIONS = 15                     # Max tool-use rounds per task
COMPUTER_USE_ENABLED = True
COMPUTER_USE_DISPLAY_WIDTH = 1280
COMPUTER_USE_DISPLAY_HEIGHT = 800

# Proactive
PROACTIVE_INTERVAL = 30.0                     # Seconds between checks
PROACTIVE_COOLDOWN = 180.0                    # Min seconds between comments
PROACTIVE_USEFULNESS_THRESHOLD = 8            # High bar (1-10)

# Cost Control
MAX_DAILY_COST_USD = 2.00
```

**System Prompts** (6 total, all in config.py):
- `SYSTEM_PROMPT_PERCEPTION` — Frame analysis (JSON output)
- `SYSTEM_PROMPT_PROACTIVE` — Decide if worth speaking (JSON with usefulness score)
- `SYSTEM_PROMPT_CONVERSATION` — Voice response style (1-3 sentences, direct)
- `SYSTEM_PROMPT_ROUTING` — Haiku intent routing with tool descriptions
- `SYSTEM_PROMPT_AGENT` — Opus agent with Computer Use instructions
- `SYSTEM_PROMPT_FACT_EXTRACTION` — Extract structured facts from conversations

---

### perception/camera.py

Camera capture and scene change detection.

- Opens camera via OpenCV on startup with interactive camera selection
- Captures frames at `CAPTURE_INTERVAL` (3s)
- Scene change detection using SSIM comparison with previous frame
- Encodes frames as JPEG (quality 70) for API calls — 1280x720 ~1100 tokens
- Ring buffer of last 5 frames for context
- `get_frame()` returns raw BGR numpy array
- `get_frame_bytes()` returns JPEG bytes

---

### perception/audio.py — AudioListener

The most complex perception module. Handles microphone input, speech-to-text, always-listening mode, and barge-in (interrupting TTS).

**Always-Listening Mode** (no wake word):
- Energy-threshold state machine: `idle` → `accumulating` → `dispatching`
- Speech onset at 0.008 RMS, ends after 1.0s silence
- Minimum 0.8s speech duration (filters coughs, taps)
- Cooldown after TTS (1.5s) and after response (2.0s) to avoid echo
- Ambient speech (not addressed to Winston) stored as context observations
- Intent classification: local heuristics first, then Haiku API fallback

**Barge-in** (speech interruption during TTS):
- `EnergyBargeInDetector` calibrates on TTS echo for 5 frames (400ms)
- Fixed threshold = echo_peak × 2.0 (min floor 0.03)
- Triggers after 3 consecutive frames above threshold (240ms)
- On trigger: aborts TTS playback, records user speech, transcribes

**Speech-to-Text**:
- `faster-whisper` with `medium` model (int8 quantization)
- Auto-detects language (`language=None`) — returns `(text, lang_code)` tuple
- Supports English and German natively

**Audio Pipeline**:
- `sounddevice.InputStream` at 16kHz mono, 80ms chunks (1280 samples)
- Processing loop runs on dedicated thread
- Callbacks: `on_transcription(text, lang)`, `on_bargein()`, `on_ambient_transcription(text)`

---

### perception/tts.py — TTSEngine

Text-to-speech with streaming playback and interrupt support.

**Backends** (auto-selected on startup):
1. **ElevenLabsBackend** (primary) — Streams PCM chunks via `sd.OutputStream` at 24kHz. First audio plays within ~200ms of API response start.
2. **Pyttsx3Backend** (fallback) — Offline TTS using macOS NSSpeechSynthesizer. Zero cost, used when ElevenLabs API key not set.

**Key Features**:
- `speak(text)` — blocking
- `speak_async(text)` — non-blocking, queued
- `interrupt()` — stops playback immediately (barge-in support)
- `begin_streaming_response()` / `end_streaming_response()` — keeps `is_speaking=True` between sentences in streaming mode
- State callbacks: `on_speaking_start` (fires when audio actually plays, not before HTTP download) and `on_speaking_stop` (fires when audio ends)
- Tracks `last_spoken_text` for echo rejection

---

### brain/claude_client.py — ClaudeClient

The central brain. Handles all Claude API calls, intelligent routing, and retry logic.

**Core Methods**:
- `process_user_input(text, frame_bytes, context, language)` → `(action, data)` — Haiku routing with tools. Returns one of: `"conversation"`, `"agent"`, `"note"`, `"shutdown"`
- `analyze_frame(frame_bytes, prompt, system_prompt)` → dict — Scene analysis (JSON)
- `chat(message, frame_bytes, context, language)` → str — Conversational response with optional streaming

**Routing Tools** given to Haiku:
```python
ROUTING_TOOLS = [
    {"name": "delegate_to_agent", ...},   # Computer tasks → Opus
    {"name": "save_note", ...},            # Remember something
    {"name": "shutdown_system", ...},      # Go offline
    {"name": "get_current_time", ...},     # Date/time queries
]
```

**Retry Logic**:
- `_call_with_retry(**kwargs)` — 2 attempts with exponential backoff (1s, 2s)
- SDK retries disabled (`max_retries=0`) to avoid double-retry
- Returns `None` on exhaustion (callers check for `None`)

**Bilingual**:
- When `language="de"`, prepends German system prompt
- Haiku handles any language natively — no separate routing

---

### brain/agent_executor.py — AgentExecutor

Opus 4.6 agentic loop with Computer Use and shortcut tools.

- Uses `client.beta.messages.create()` with `betas=["computer-use-2025-11-24"]`
- Max 15 iterations per task (configurable via `AGENT_MAX_ITERATIONS`)
- Each iteration: Opus decides which tool to call → tool executes → result fed back
- Early termination after 3 consecutive empty results (sends wrap-up hint)
- Follow-up context: `_get_recent_agent_result()` passes last 5-min completed task

---

### brain/agent_tools.py — ToolRegistry

14 tools available to the agent:

| Category | Tools |
|----------|-------|
| **GitHub** (6) | `github_search_code`, `github_get_issues`, `github_get_file_content`, `github_search_issues`, `github_list_commits`, `github_create_issue` |
| **Web** (2) | `web_search` (DuckDuckGo), `fetch_webpage` |
| **System** (3) | `run_shell_command`, `open_url`, `get_current_time` |
| **Files** (3) | `read_local_file`, `write_file`, `list_files` |

---

### brain/computer_use.py — MacOSComputerController

Gives the agent eyes and hands on macOS.

- **Screenshot**: `screencapture -x -C` → PIL resize to 1280x800 → base64
- **Mouse**: `cliclick` — click, right-click, double-click, move, drag
- **Keyboard**: `osascript` AppleScript — keystroke, key codes (return=36, tab=48, escape=53)
- **Coordinate scaling**: Claude sees 1280x800, native display may be 3440x1440. Scale = native/display.
- **Requirements**: `brew install cliclick`, macOS Accessibility + Screen Recording permissions

---

### brain/memory.py — Three-Tier Memory

**Tier 1 — Working Memory** (in-process, ephemeral):
- Last 30 observations, 20 conversations
- Zero-cost reads, dies with session

**Tier 2 — Episodic Memory** (ChromaDB):
- All observations, conversations, proactive comments with importance scoring (1-10)
- Semantic search with time-decay + importance boosting
- Deduplication (distance < 0.1 within 5-min window)
- Session summaries generated at shutdown
- Consolidation: entries >7 days old archived if importance <7

**Tier 3 — Semantic Memory** (JSON + ChromaDB):
- Structured facts: `{entity, attribute, value, confidence, category}`
- Categories: personal, equipment, project, workshop, safety
- Extracted via Haiku after conversations (background thread)
- Persists across sessions

**Context Assembly**:
- `assemble_context(query, purpose)` builds context with token budgets
- Conversation: 800 tokens budget
- Proactive: 500 tokens budget
- Includes recent observations + relevant semantic facts

---

### brain/proactive.py — ProactiveEngine

Watches the workshop and decides when to speak up.

- Runs every 30 seconds (configurable)
- **Scene change gate**: If frame hasn't changed, skip API call entirely (biggest cost saver)
- Sends current frame + recent context to Haiku with `SYSTEM_PROMPT_PROACTIVE`
- Parses JSON: `{should_speak, usefulness_score, message, reasoning}`
- Only speaks if `usefulness_score >= 8` AND cooldown has elapsed (180s)
- **High bar**: Only interrupts for safety hazards or clear mistakes. Not tips, not suggestions, not narration.

---

### utils/persistent_store.py — PersistentStore

Thread-safe atomic JSON storage.

- `get(key)`, `update(key, value)`, `append_to_list()`, `update_in_list()`, `remove_from_list()`
- Atomic writes via tempfile + `os.replace`
- Used for: `winston_memory/agent_tasks.json`, `winston_memory/notes.json`

---

### utils/cost_tracker.py — CostTracker

Tracks API costs and enforces daily budget.

- Records token usage per model per call
- Warns at 50%, 75%, 90% of daily budget ($2.00 default)
- `check_budget()` → False when over budget → orchestrator reduces frequency
- Haiku: $1.00/$5.00 per 1M tokens (input/output)
- Sonnet: $3.00/$15.00 per 1M tokens

---

### utils/echo_cancel.py

Barge-in energy detection + echo text rejection.

- `EnergyBargeInDetector`: Calibrates on TTS echo baseline (5 frames), triggers on 2x energy spike (3 consecutive frames)
- `strip_echo_prefix()`: Removes leading TTS words from barge-in transcriptions
- `echo_text_overlap()`: Computes word overlap ratio to detect pure echo transcriptions

---

### dashboard/ — Web Dashboard

FastAPI server on `http://localhost:8420` with WebSocket real-time updates.

- **WinstonState** (`state.py`): Thread-safe shared state — status, audio level, cost, conversation, observations, agent tasks, notes
- **REST API**: GET/POST/DELETE notes, GET agent tasks
- **WebSocket**: Pushes state diffs on every change (version-gated)
- **UI**: Vanilla HTML/CSS/JS, no framework

---

## Setup Instructions

### 1. Environment

```bash
cd winston
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# macOS dependencies
brew install portaudio    # Required by sounddevice
brew install cliclick     # Required for Computer Use (mouse control)
```

### 2. API Keys

```bash
# Create .env file in winston/
cat > .env << 'EOF'
ANTHROPIC_API_KEY=your_anthropic_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
EOF
```

- Anthropic API key: https://console.anthropic.com/
- ElevenLabs API key: https://elevenlabs.io/ (optional — falls back to pyttsx3)

### 3. macOS Permissions

System Settings → Privacy & Security:
- **Camera** → Enable for Terminal / your IDE
- **Microphone** → Enable for Terminal / your IDE
- **Accessibility** → Enable for Terminal (required for Computer Use keyboard control)
- **Screen Recording** → Enable for Terminal (required for Computer Use screenshots)

### 4. Run

```bash
cd winston
python main.py
```

Startup output:
```
[Winston] Starting up...
[Winston] Dashboard live: http://localhost:8420
[Winston] Camera ready: index 0, 1280x720
[Winston] Whisper model loaded (faster-whisper, int8)
[Winston] Audio stream opened (16kHz mono)
[Winston] Always-listening mode enabled
[Winston] ElevenLabs backend initialized
[Winston] Memory system online — N episodes, M facts
[Winston] Claude API connected (ready)
[Winston] System online.
```

---

## Cost Estimation

With default settings (3s frame capture, 30s proactive interval, scene gating):

| Activity | Model | Est. calls/day | Est. cost/day |
|----------|-------|----------------|---------------|
| Scene analysis | Haiku | ~200 (after gating) | ~$0.30 |
| Proactive checks | Haiku | ~100 (after gating) | ~$0.15 |
| Conversations | Haiku | ~30 | ~$0.15 |
| Fact extraction | Haiku | ~30 | ~$0.05 |
| Agent tasks | Opus | ~3-5 | ~$0.50 |
| **Total estimated** | | | **~$1.15/day** |

Budget cap: $2.00/day. Scene gating reduces API calls by ~70-80%.

---

## Key Design Principles

1. **Scene gating is the #1 cost saver.** If nothing changed in the camera, don't call the API.

2. **Haiku does everything except agentic tasks.** Routing, conversation, perception, fact extraction — all Haiku. Opus only fires for autonomous multi-step computer tasks.

3. **Intelligent routing via tool-use, not keywords.** Haiku decides intent natively. No keyword lists to maintain. Works across languages.

4. **Short voice responses.** 1-3 sentences maximum. This is a voice conversation, not a chatbot. Ask follow-up questions instead of monologuing.

5. **Proactive but quiet.** Usefulness threshold of 8/10 + 180s cooldown. Only interrupt for safety hazards or clear mistakes. Default to silence.

6. **Echo rejection over echo prevention.** The barge-in detector errs toward triggering (low threshold), and downstream `strip_echo_prefix()` handles false positives. Better to catch a real interruption than miss one.

7. **Graceful degradation.** If the API is overloaded, log a warning and continue. If the daily budget is hit, reduce frequency. Never just stop.

8. **Memory reduces repeated context.** Text summaries from ChromaDB are orders of magnitude cheaper than sending image history. Three-tier memory (working → episodic → semantic) keeps context relevant and affordable.

---

## Latency

The critical path for voice response is:

```
User stops speaking → Whisper transcription → Intent classification → Claude API call → First TTS audio chunk plays
```

### Pipeline Stages

| Stage | What happens | Typical latency |
|-------|-------------|----------------|
| **STT** | Silence detected → Groq Whisper API transcribes audio | 200–600ms (Groq), 1–3s (local Whisper) |
| **Intent** | Local heuristic classifies if speech is addressed to Winston | 0–5ms (local), 200–300ms (API fallback) |
| **Context** | Frame capture + ChromaDB query in parallel (ThreadPoolExecutor) | 50–200ms |
| **LLM** | Haiku routing with tool-use (non-streaming, includes response) | 300–800ms |
| **TTS queue** | Response text queued → TTS worker dequeues | 0–500ms (worker polls at 500ms) |
| **TTS synth** | ElevenLabs `eleven_flash_v2_5` streams first audio chunk | 75–150ms (API spec), 200–400ms (with network) |
| **End-to-end** | Silence detected → first audio plays through speaker | **750–2350ms** |

### Instrumentation

Latency is tracked via `utils/latency_tracker.py` — a singleton `LatencyTracker` that correlates timing marks across threads using interaction IDs.

**Events tracked:** `speech_end` → `stt_done` → `intent_done` → `context_ready` → `llm_done` → `tts_dequeued` → `audio_plays`

**Log output** (INFO level, every interaction):
```
[latency] STT: 340ms
[latency] LLM first-token: 450ms
[latency] LLM total: 450ms
[latency] TTS first-audio: 280ms
[latency] a1b2c3d4: stt=340ms | intent=3ms | context=120ms | llm=450ms | tts_queue=12ms | tts_synth=280ms | e2e=1205ms
```

**Dashboard:** System panel shows Last E2E, STT/LLM/TTS p50, and interaction count. REST endpoint: `GET /api/latency` returns full stats (p50, p95, avg, count per segment).

**Stats:** Rolling window of last 100 interactions. `get_stats()` returns p50, p95, avg for each segment.

### Why non-streaming for routing

`process_user_input()` uses Haiku with tool-use, which does not support streaming in the Anthropic API. However, the full response (including Haiku's text reply) arrives in a single non-streaming call, and goes to TTS immediately — no post-processing delay. This is ~300–500ms, comparable to the time it would take to stream the first sentence.

---

## Data Flow: Voice Interaction

A complete voice interaction from microphone to speaker:

```
Microphone (16kHz mono, sounddevice)
  │
  ▼
AudioListener._audio_callback()              [PortAudio thread]
  │  Appends float32 chunk to _audio_buffer (deque, lock-free)
  ▼
AudioListener._processing_loop()             [audio-processor thread]
  │  Drains buffer, processes 80ms frames
  │
  ├──▶ _al_process_frame()                   [Energy state machine]
  │    States: idle → accumulating → dispatching → cooldown
  │    Onset: RMS energy ≥ 0.008
  │    End: 1.5s silence or 15s max duration
  │
  ├──▶ _al_dispatch()                        [Spawns daemon thread]
  │      ▼
  │    _al_transcribe_and_classify()          [al-transcribe thread]
  │      │  Pre-check: skip if avg energy < threshold × 0.5
  │      │  STT: stt_provider.transcribe(audio)
  │      │    → GroqWhisperProvider (primary, API)
  │      │    → LocalWhisperProvider (fallback, on-CPU)
  │      │  Filters: hallucination set, prompt leak detection, echo rejection
  │      ▼
  │    _on_ambient_transcription(text, lang)  [callback → main.py]
  │      │  Continuation check: skip classification if within 8s window
  │      │  classify_intent_local(text)
  │      │    → True:  addressed to Winston → route to _on_transcription
  │      │    → False: ambient speech → store as observation
  │      │    → None:  uncertain → API fallback via classify_intent()
  │      ▼
  │    _on_transcription(text, lang)          [daemon thread]
  │      │  camera.get_frame_bytes() + memory.assemble_context()  [parallel]
  │      │  Build conversation_history (last 10 turns, merged by role)
  │      ▼
  │    llm.process_user_input(text, frame, context, lang, history)
  │      │  Haiku with ROUTING_TOOLS (delegate_to_agent, save_note,
  │      │  shutdown_system, get_current_time)
  │      │  Returns (action_type, data)
  │      │
  │      ├──▶ "conversation": tts.speak_async(response)
  │      │      + memory.store() + Thread(extract_facts_from_text)
  │      ├──▶ "agent": _spawn_agent_task(task)
  │      │      → Thread "agent-task": AgentExecutor.run()
  │      │        Opus + tools + Computer Use, max 15 iterations
  │      │      → On completion: tts.speak_async(summary)
  │      ├──▶ "note": _notes_store.append_to_list()
  │      │      + tts.speak_async("Noted: ...")
  │      └──▶ "shutdown": tts.speak("Going offline") → _running = False
  │
  └──▶ _check_bargein(frame)                 [During TTS playback only]
       EnergyBargeInDetector: calibrates on TTS echo (5 frames),
       triggers after 3 consecutive frames at 1.5× echo level
         ▼
       _handle_bargein()
         │  Calls on_bargein → streaming_abort.set() + tts.interrupt()
         │  Records + transcribes interrupted speech
         └  Echo text rejection: strip_echo_prefix() + echo_text_overlap()
```

---

## Threading Model

Winston uses a main thread for the event loop plus daemon threads for concurrent I/O. All threads are named for debuggability.

| Thread | Target | Started By | Purpose |
|--------|--------|------------|---------|
| `audio-processor` | `AudioListener._processing_loop()` | `audio.start()` | Drains mic buffer, runs always-listen state machine + barge-in |
| `al-transcribe` | `_al_transcribe_and_classify()` | `_al_dispatch()` | Per-utterance: STT + intent classification |
| `transcription-handler` | `Winston._on_transcription()` | `_record_and_transcribe()` | Process wake-word/manual-trigger transcriptions |
| `tts-worker` | `TTSEngine._worker()` | `tts.start()` | Dequeues and plays TTS sentences via sounddevice |
| `agent-task` | `Winston._run_agent_task()` | `_spawn_agent_task()` | Opus agentic loop (one at a time, guarded by Lock) |
| `fact-extraction` | `Memory.extract_facts_from_text()` | `_on_transcription()` | Background Haiku call to extract semantic facts |
| `memory-consolidate` | `Winston._consolidate_memory()` | `Winston.start()` | One-shot startup cleanup of old episodic entries |
| `dashboard-server` | `uvicorn.run()` | `start_server()` | FastAPI HTTP + WebSocket on port 8420 |
| `manual-listen` | `AudioListener._handle_wake_word()` | `trigger_listen()` | Dashboard button listen trigger |
| `shutdown-memory` | `Winston._safe_shutdown_memory()` | `Winston.shutdown()` | Session summary + fact extraction with 3s timeout |

### Thread Communication

| Mechanism | Where Used |
|-----------|-----------|
| **Callbacks** | AudioListener fires `on_transcription`, `on_bargein`, `on_ambient_transcription` into daemon threads. Set via `audio.set_callbacks()`. TTS fires `on_speaking_start`/`on_speaking_stop` for barge-in calibration. |
| **Deque buffers** | `AudioListener._audio_buffer` — lock-free deque with maxlen. Single producer (PortAudio), single consumer (processor thread). |
| **threading.Lock** | `WinstonState._lock` (all state reads/writes), `Winston._agent_lock` (one agent at a time), `AudioListener._al_active_threads_lock` (concurrent transcription guard). |
| **threading.Event** | `Winston._streaming_abort` — barge-in cancels in-progress Claude streaming. |
| **PersistentStore** | Atomic JSON via `tempfile` + `os.replace()` for `agent_tasks.json` and `notes.json`. |
| **WinstonState.version** | Monotonic counter; WebSocket only pushes when version changes. |

### Concurrency Constraints

1. **One agent at a time**: `Winston._agent_lock` uses non-blocking `acquire()`. A second task gets "I'm still working on the previous task."
2. **One always-listen transcription at a time**: `_al_active_threads` counter checked before `_al_dispatch()`.
3. **Barge-in cooldown**: 3s cooldown after false positive (empty transcription from barge-in) to prevent loops.
4. **Post-TTS cooldown**: 1.5s after TTS stops before always-listen re-enables (avoids echo pickup).

---

## Error Handling Philosophy

Winston runs unattended in a workshop for hours. The core principle: **never crash, always degrade gracefully**.

### Patterns

1. **Retry with backoff**: `ClaudeClient._call_with_retry()` — 2 attempts with exponential backoff (1s, 2s). SDK retries disabled (`max_retries=0`) to avoid double-retry.

2. **Return None on failure**: API methods return `None` on failure. Callers always check: `if response is None: return`. No exceptions propagate to the event loop.

3. **Budget-based frequency reduction**: When daily budget is exceeded, `_capture_interval` increases from 3s to 30s, proactive interval from 30s to 60s. System keeps running at reduced observation frequency.

4. **Rate limit retry in agent**: `AgentExecutor` catches 429 errors, waits 30s, retries once. Sufficient for per-minute token limits on Opus.

5. **Daemon thread exception hook**: `threading.excepthook` logs unhandled exceptions in daemon threads with full tracebacks instead of silently dying.

6. **Graceful shutdown with timeout**: `Winston.shutdown()` runs memory session summary in a thread with `join(timeout=3.0)`. If it hangs, shutdown proceeds.

7. **Module-level isolation**: Each `start()` method (camera, audio, memory, TTS, LLM) can fail independently. Camera failure disables scene analysis but conversation still works.

8. **Agent crash recovery**: On startup, `_check_pending_agent_results()` marks interrupted tasks (status="running" from previous session) as "failed" and reports them.

---

## Future Roadmap

### Hardware Upgrades
- miniDSP UMA-8 mic array for far-field pickup (workshop is noisy)
- Obsbot Tiny 2 PTZ camera with SDK control (track Roberto, zoom on work)
- Second camera for 3D printer monitoring
- Projector output for AR overlay on workbench (highlight parts, show measurements)

### Software Upgrades
- Drone inspection module (Crazyflie 2.1+) for hard-to-reach inspection
- DeepFilterNet for real-time noise suppression before Whisper
- Fine-tune a small local vision model for workshop-specific scene classification
- Streaming video understanding (Gemini Live API or equivalent when available)
- Multi-agent coordination (one agent runs while another monitors)
- Tool learning — let the agent remember and reuse successful tool sequences
- Voice cloning for more natural personality
- Local LLM fallback for offline operation (when API is down)
