# Winston

**AI workshop assistant that watches, listens, and acts.**

![CI](https://github.com/FLASH-73/Winston/actions/workflows/ci.yml/badge.svg)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![Claude API](https://img.shields.io/badge/LLM-Claude%20API-orange)
![ElevenLabs](https://img.shields.io/badge/TTS-ElevenLabs-purple)
![ChromaDB](https://img.shields.io/badge/Memory-ChromaDB-green)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red)

Winston runs on a MacBook in a robotics workshop. It watches through a camera, listens continuously for speech (no wake word needed), and can autonomously control the computer to complete tasks. It answers questions in 1-3 sentences, supports English and German, and proactively speaks up when it spots a safety hazard or a clear mistake.

## Architecture

```
                                ┌──────────────────┐
                     ┌─────────►│  Claude Opus 4.6  │  agent: Computer Use + 14 tools
                     │          └──────────────────┘
┌──────────┐   ┌─────┴──────┐  ┌──────────────────┐
│ Camera   │──►│            │─►│  Claude Haiku     │  routing + conversation + perception
│ Mic/STT  │──►│ Orchestrat.│  └──────────────────┘
│ Memory   │◄─►│ (main.py)  │  ┌──────────────────┐
└──────────┘   └─────┬──────┘  │  ElevenLabs TTS   │  streaming, ~75ms latency
                     │         └──────────────────┘
                     ├────────►│  ChromaDB         │  3-tier memory
                     │         └──────────────────┘
                     └────────►│  Dashboard        │  FastAPI + WebSocket
                               └──────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full technical documentation.

## Quick Start

### Prerequisites

- Python 3.11+
- macOS (Computer Use requires `cliclick`, `osascript`, `screencapture`)
- [Anthropic API key](https://console.anthropic.com/)
- [ElevenLabs API key](https://elevenlabs.io/) (optional — falls back to pyttsx3)
- [Groq API key](https://console.groq.com/) (optional — falls back to local Whisper)

### Install

```bash
git clone <repo-url> && cd <repo-dir>

# Option A: pip install (editable)
pip install -e ".[dev]"

# Option B: requirements.txt
cd winston
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# macOS dependencies
brew install portaudio cliclick
```

### Configure

```bash
cp winston/.env.example winston/.env
# Edit winston/.env with your API keys
```

### macOS Permissions

System Settings > Privacy & Security — enable for Terminal:
- Camera
- Microphone
- Accessibility (required for keyboard control)
- Screen Recording (required for screenshots)

### Run

```bash
make run
# or: cd winston && python main.py
```

Dashboard: [http://localhost:8420](http://localhost:8420)

## How It Works

Winston uses a **two-tier Claude architecture** to balance cost and capability. All incoming speech is routed through Claude Haiku, which is cheap and fast (~300ms). Haiku decides intent using tool-use — no keyword matching, no regex, just native language understanding across both English and German. If the user wants something done on the computer (open a file, search the web, navigate an app), Haiku delegates to Claude Opus 4.6, which runs an autonomous agent loop with full Computer Use (screenshot, mouse, keyboard) plus 14 shortcut tools for common operations like web search, GitHub queries, and shell commands. For everything else — questions, conversation, notes — Haiku answers directly in 1-3 sentences.

The **scene gating** system is the primary cost control mechanism. The camera captures frames every 3 seconds, but frames are only sent to the API when SSIM-based change detection indicates something meaningful has changed. This eliminates 70-80% of API calls that would otherwise be wasted on static scenes. A proactive engine runs every 30 seconds to check if Winston should speak up, but only interrupts for genuine safety concerns or clear mistakes (usefulness threshold: 8/10 with a 3-minute cooldown).

Winston maintains a **three-tier memory** system. Working memory (in-process, ephemeral) holds the last 30 observations and 20 conversation turns for zero-cost reads. Episodic memory (ChromaDB) stores all interactions with importance scoring and time-decay, enabling semantic search over the workshop's history. Semantic memory (structured JSON + ChromaDB) captures facts about the user, equipment, and projects as entity-attribute-value triples, extracted automatically after each conversation.

## Project Structure

```
winston/
├── main.py              # Orchestrator, event loop, routing
├── config.py            # All configuration and system prompts
├── brain/               # Claude API, agent, memory, proactive engine
├── perception/          # Camera, audio/STT, TTS
├── utils/               # Cost tracking, echo cancellation, frame diff
├── dashboard/           # FastAPI web UI + WebSocket
└── winston_memory/      # Runtime data (ChromaDB, JSON stores)
```

## Cost

Estimated daily cost with default settings (scene gating reduces API calls by 70-80%):

| Activity | Model | Est. calls/day | Est. cost/day |
|----------|-------|----------------|---------------|
| Scene analysis | Haiku | ~200 | ~$0.30 |
| Proactive checks | Haiku | ~100 | ~$0.15 |
| Conversations | Haiku | ~30 | ~$0.15 |
| Fact extraction | Haiku | ~30 | ~$0.05 |
| Agent tasks | Opus | ~3-5 | ~$0.50 |
| **Total** | | | **~$1.15/day** |

Daily budget cap: $2.00 (configurable in `config.py`).

## License

MIT — see [LICENSE](LICENSE).
