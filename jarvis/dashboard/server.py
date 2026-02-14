"""FastAPI server for the WINSTON web dashboard.

Endpoints:
  GET  /                  — Dashboard HTML
  GET  /camera/stream     — MJPEG live camera feed
  WS   /ws                — Real-time state via WebSocket
  GET  /api/facts         — Memory facts (on-demand)
  GET  /api/cost          — Cost report details
"""

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

logger = logging.getLogger("winston.dashboard")

# References set by create_app() / set_audio()
_state = None
_camera = None
_memory = None
_cost_tracker = None
_audio = None
_notes_store = None


def create_app(state, camera, memory=None, cost_tracker=None) -> FastAPI:
    """Create the FastAPI app with references to WINSTON components."""
    global _state, _camera, _memory, _cost_tracker
    _state = state
    _camera = camera
    _memory = memory
    _cost_tracker = cost_tracker
    return app


def set_audio(audio):
    """Set audio reference after audio init (dashboard starts before audio)."""
    global _audio
    _audio = audio


def set_camera(camera):
    """Set camera reference after camera init."""
    global _camera
    _camera = camera


def set_memory(memory):
    """Set memory reference after memory init."""
    global _memory
    _memory = memory


def set_cost_tracker(cost_tracker):
    """Set cost tracker reference after init."""
    global _cost_tracker
    _cost_tracker = cost_tracker


def set_notes_store(notes_store):
    """Set notes store reference for dashboard API."""
    global _notes_store
    _notes_store = notes_store


app = FastAPI(title="WINSTON Dashboard", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.get("/camera/stream")
async def camera_stream():
    """MJPEG stream from the camera at ~10 FPS."""
    async def generate():
        while True:
            if _camera is not None and _camera.is_open:
                frame_bytes = _camera.get_frame_bytes(quality=50)
                if frame_bytes:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
            await asyncio.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Stream real-time state updates to the dashboard."""
    await ws.accept()
    last_version = -1
    try:
        while True:
            if _state is None:
                await asyncio.sleep(0.5)
                continue

            current_version = _state.version
            # Send full state if version changed, or every 500ms for audio level
            if current_version != last_version:
                data = _state.to_dict()
                await ws.send_text(json.dumps({"type": "state", "data": data}))
                last_version = current_version
            else:
                # Still send audio level updates (not version-tracked)
                await ws.send_text(json.dumps({
                    "type": "audio",
                    "data": {"audioLevel": _state.audio_level},
                }))

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WebSocket closed: %s", e)


@app.get("/api/facts")
async def get_facts():
    """Return semantic memory facts."""
    if _memory is not None:
        try:
            text = _memory.semantic.get_all_facts_as_text()
            return JSONResponse({"facts": text})
        except Exception:
            pass
    return JSONResponse({"facts": "No facts stored yet."})


@app.get("/api/cost")
async def get_cost():
    """Return cost tracker report."""
    if _cost_tracker is not None:
        try:
            return JSONResponse({
                "report": _cost_tracker.get_daily_report(),
                "cost": _cost_tracker.get_daily_cost(),
            })
        except Exception:
            pass
    return JSONResponse({"report": "No cost data.", "cost": 0.0})


@app.post("/api/notes/{note_id}/toggle")
async def toggle_note(note_id: str):
    """Toggle a note's done status."""
    if _state is None:
        return JSONResponse({"status": "error"}, status_code=503)
    found = _state.toggle_note(note_id)
    if found and _notes_store is not None:
        _notes_store.update_in_list("notes", note_id, {"done": next(
            (n.get("done", False) for n in _state.notes if n.get("id") == note_id), False
        )})
    return JSONResponse({"status": "ok" if found else "not_found"})


@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: str):
    """Delete a note."""
    if _state is None:
        return JSONResponse({"status": "error"}, status_code=503)
    found = _state.remove_note(note_id)
    if found and _notes_store is not None:
        _notes_store.remove_from_list("notes", note_id)
    return JSONResponse({"status": "ok" if found else "not_found"})


@app.post("/api/listen")
async def trigger_listen():
    """Manually trigger listen mode (dashboard Talk button)."""
    if _audio is not None:
        try:
            _audio.trigger_listen()
            return JSONResponse({"status": "listening"})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    return JSONResponse({"status": "error", "message": "Audio not available"}, status_code=503)


def start_server(application: FastAPI, port: int = 8420):
    """Run uvicorn in a daemon thread so it doesn't block WINSTON."""
    import uvicorn

    def _run():
        uvicorn.run(
            application,
            host="0.0.0.0",
            port=port,
            log_level="warning",
        )

    thread = threading.Thread(target=_run, daemon=True, name="dashboard-server")
    thread.start()
    logger.info("Dashboard server started at http://localhost:%d", port)
