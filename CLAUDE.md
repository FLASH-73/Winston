# Winston Workshop Assistant

AI workshop assistant for a robotics workshop. Camera + voice + autonomous computer control.

## Quick Reference

- **Run**: `make run` or `cd winston && python main.py`
- **Install**: `make install` (editable with dev deps) or `cd winston && pip install -r requirements.txt`
- **Test**: `make test`
- **Lint**: `make lint`

## Architecture

Two-tier Claude: Haiku (routing, conversation, perception) + Opus 4.6 (agent with Computer Use).
All routing via tool-use in `process_user_input()` — no keyword matching.

## Full Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for complete module specs, system prompts, setup instructions, and design principles.

## Key Files

| File | Purpose |
|------|---------|
| `winston/main.py` | Orchestrator, event loop, routing |
| `winston/config.py` | All config: models, thresholds, system prompts |
| `winston/brain/claude_client.py` | Claude API + `process_user_input()` routing |
| `winston/brain/agent_executor.py` | Opus agentic loop with Computer Use |
| `winston/brain/agent_tools.py` | ToolRegistry + 14 tools |
| `winston/brain/computer_use.py` | macOS screenshot/mouse/keyboard control |
| `winston/brain/memory.py` | 3-tier memory (working + episodic + semantic) |
| `winston/perception/audio.py` | Mic, STT, always-listen, barge-in |
| `winston/perception/tts.py` | ElevenLabs streaming + pyttsx3 fallback |
| `winston/perception/camera.py` | Camera capture, scene change detection |
| `winston/dashboard/server.py` | FastAPI + WebSocket dashboard |
