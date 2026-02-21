# Winston

**An AI companion that lives in your workshop — watches, listens, thinks, and reaches out.**

![CI](https://github.com/FLASH-73/Winston/actions/workflows/ci.yml/badge.svg)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![Claude API](https://img.shields.io/badge/Brain-Claude%20API-orange)
![Gemini](https://img.shields.io/badge/Vision-Gemini-4285F4)
![ElevenLabs](https://img.shields.io/badge/TTS-ElevenLabs-purple)
![ChromaDB](https://img.shields.io/badge/Memory-ChromaDB-green)
![Telegram](https://img.shields.io/badge/Chat-Telegram-26A5E4)
![Groq](https://img.shields.io/badge/STT-Groq%20Whisper-red)

Winston runs 24/7 in a robotics workshop. He sees through a camera (Gemini visual cortex), listens without a wake word (Groq Whisper), speaks with a dry personality (ElevenLabs), and thinks with a three-tier Claude brain. When he's not answering questions, he's autonomously researching topics relevant to your work and sending findings via Telegram. He remembers everything across sessions, speaks English and German interchangeably, and evolves his understanding as your projects progress.

<!-- TODO: Add demo GIF or screenshot of Telegram conversation -->
<!-- TODO: Dashboard screenshot -->

## Capabilities

### Vision

- **Gemini 2.5 Flash Lite visual cortex** — continuous 24/7 scene understanding, not discrete frame analysis
- Dynamic frame rate: 0.2 FPS idle → 1.0 FPS when motion or audio detected
- Temporal narrative: rolling 2-hour activity log with automatic compression of older entries
- Local motion detection gates API calls — only sends batches when something is happening
- On-demand camera snapshots and `/watch` mode (30s interval monitoring) via Telegram
- Clip recorder: frame buffer for video clips (up to 60s) and timelapses (up to 8hrs)

### Audio

- Always-on listening with no wake word (Groq Whisper large-v3-turbo)
- Barge-in support — interrupt Winston mid-sentence
- Echo cancellation for speaker/mic feedback loops
- Ambient noise calibration on startup
- Falls back to local Whisper if Groq is unavailable

### Intelligence

- **Three-tier Claude**: Haiku 4.5 (routing, conversation, perception) → Sonnet 4.5 (research, deep analysis) → Opus 4.6 (autonomous agent with Computer Use)
- 14 agent tools: GitHub (search, read files, list repos, view issues/PRs), web search, web fetch, file operations, shell commands
- Autonomous computer control: screenshot, mouse, keyboard (Linux: xdotool + scrot)
- Background research agent (Sonnet + web tools) — runs in a thread, you keep chatting while it investigates

### Communication

- ElevenLabs streaming TTS (~75ms latency), pyttsx3 fallback
- Full Telegram bot: text, voice notes, camera snapshots, agent task delegation
- Voice message transcription and voice responses via Telegram
- Natural bilingual support (English + German, code-switches mid-conversation)

### Proactive Companion

- **Curiosity engine**: autonomous 30–90 min thinking loop
  1. Reflect — Haiku reviews memory and context, picks interesting topics
  2. Explore — DuckDuckGo web search, extracts findings
  3. Craft — Sonnet writes a natural message if the finding is genuinely worth sharing
- Check-ins when you haven't been in the workshop for hours
- Max 5 unsolicited messages/day, quiet hours (1am–7am) respected
- Daily budget cap prevents runaway costs

### Memory

- **Working memory**: conversation turns + recent observations (in-process, zero-cost)
- **Episodic memory**: ChromaDB with importance scoring and time-decay retrieval
- **Semantic memory**: auto-extracted facts as entity-attribute-value triples
- **Temporal narrative**: timestamped activity log from visual cortex, compressed into rolling summaries
- Cross-session persistence — remembers what you were working on last time

### Personality

- Dynamic mood engine: default, long_session, emergency, success, mistake, idle_night
- Character DNA: dry humor, genuinely curious, loyal via honest criticism, impatient with inefficiency
- Evolves tone based on time of day, session length, and workshop state
- Over Telegram: slightly more relaxed — texts like a friend, not a notification
- Three personality presets: default (dry/opinionated), mentor (proactive), minimal (terse)

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │             Telegram Bot                 │
                    │   text · voice · photos · agent tasks    │
                    └──────────┬──────────────────────────────┘
                               │
┌──────────┐   ┌───────────────┴───────────┐   ┌──────────────────┐
│ Camera   │──►│                           │──►│  Claude Haiku     │ routing + conversation
│ Mic/STT  │──►│       Orchestrator        │   └──────────────────┘
│ Speaker  │◄──│        (main.py)          │   ┌──────────────────┐
└──────────┘   │                           │──►│  Claude Sonnet    │ research + curiosity
               └──┬──────┬──────┬──────────┘   └──────────────────┘
                  │      │      │              ┌──────────────────┐
                  │      │      └─────────────►│  Claude Opus      │ agent + Computer Use
                  │      │                     └──────────────────┘
           ┌──────┴┐  ┌──┴─────┐  ┌──────────────────┐
           │Gemini  │  │Memory  │  │  ElevenLabs TTS   │
           │Visual  │  │3-tier  │  └──────────────────┘
           │Cortex  │  │+temp.  │  ┌──────────────────┐
           │24/7    │  │narr.   │  │  Clip Recorder    │
           └────────┘  └────────┘  └──────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full technical documentation.

## Quick Start

### Prerequisites

- Python 3.11+
- Linux (tested on Ubuntu 24) — also works on macOS with adjustments
- ffmpeg (voice messages and video clips)
- [Anthropic API key](https://console.anthropic.com/) (required)
- [Gemini API key](https://aistudio.google.com/) (recommended — visual cortex)
- [ElevenLabs API key](https://elevenlabs.io/) (recommended — falls back to pyttsx3)
- [Groq API key](https://console.groq.com/) (recommended — falls back to local Whisper)
- [Telegram Bot Token](https://t.me/BotFather) (optional — for remote access)

### Install

```bash
git clone https://github.com/FLASH-73/Winston.git && cd Winston
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Ubuntu/Debian system dependencies
sudo apt install portaudio19-dev ffmpeg
```

### Configure

```bash
cp winston/.env.example winston/.env
# Edit winston/.env with your API keys
```

### Run

```bash
./start.sh
# or: make run
```

Dashboard: [http://localhost:8420](http://localhost:8420)

## Telegram Setup

1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_USERS` to `.env`, set `TELEGRAM_ENABLED=true`
3. Enable `CURIOSITY_ENABLED=true` for autonomous research and proactive messages

## Cost

Estimated monthly cost with default settings:

| Component | Model | ~Monthly Cost |
|-----------|-------|--------------|
| Visual Cortex (24/7) | Gemini 2.5 Flash Lite | ~$19 |
| Conversation | Claude Haiku 4.5 | ~$5–15 |
| Agent tasks | Claude Opus 4.6 | ~$3–8 |
| Curiosity Engine | Haiku + Sonnet | ~$15–20 |
| Fact Extraction | Claude Haiku 4.5 | ~$2 |
| TTS | ElevenLabs | ~$5 |
| **Total** | | **~$50–70/month** |

Daily budget cap: $2.00 (configurable in `config.py`).

## Project Structure

```
winston/
├── main.py                # Orchestrator, event loop, routing
├── config.py              # All config, system prompts, constants
├── personality.py         # Dynamic mood engine
├── telegram_bot.py        # Full Telegram bot (text, voice, media)
├── brain/
│   ├── claude_client.py   # Claude API wrapper + process_user_input()
│   ├── memory.py          # 3-tier memory system
│   ├── temporal_memory.py # Rolling temporal narrative
│   ├── visual_cortex.py   # Gemini 24/7 vision
│   ├── curiosity.py       # Autonomous research loop
│   ├── research_agent.py  # Background research (Sonnet + web tools)
│   ├── proactive.py       # Proactive observation engine
│   ├── agent_executor.py  # Opus agentic loop with Computer Use
│   ├── agent_tools.py     # Tool registry (14 tools)
│   └── computer_use.py    # Screenshot, mouse, keyboard (Linux/macOS)
├── perception/
│   ├── camera.py          # Camera capture + scene change detection
│   ├── audio.py           # Always-on audio pipeline
│   ├── clip_recorder.py   # Frame buffer for clips + timelapses
│   ├── stt.py             # Speech-to-text (Groq / local Whisper)
│   └── tts.py             # Text-to-speech (ElevenLabs / pyttsx3)
├── personalities/
│   ├── default.yaml       # Character DNA, moods, traits
│   ├── mentor.yaml        # Proactive, conversational
│   └── minimal.yaml       # Ultra-terse
├── dashboard/             # FastAPI web UI + WebSocket
├── utils/                 # Cost tracking, echo cancel, frame diff
└── winston_memory/        # Runtime data (ChromaDB, JSON, narratives)
```

## License

MIT — see [LICENSE](LICENSE).
