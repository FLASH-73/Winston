"""Thread-safe shared state between WINSTON core and the web dashboard."""

import threading
import time
from typing import Optional


class WinstonState:
    """Shared state that WINSTON modules write to and the WebSocket reads from.

    All writes are thread-safe (protected by a lock).
    A version counter increments on every change so the WebSocket
    only sends updates when something actually changed.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._version = 0
        self._start_time = time.time()

        # System state
        self.status = "initializing"
        self.modules = {}  # {name: {"status": str, "load_time_ms": float|None}}

        # Real-time metrics
        self.audio_level = 0.0       # RMS 0.0-1.0 from mic
        self.scene_change_score = 0.0
        self.cost_today = 0.0
        self.budget_max = 2.0
        self.always_listen_state = "disabled"

        # Content
        self.conversation = []       # [{"role": str, "text": str, "timestamp": str}]
        self.observations = []       # [{"description": str, "timestamp": str}]
        self.memory_episodes = 0
        self.memory_facts = 0

        # Agent tasks & Notes
        self.agent_tasks = []        # [{"id", "query", "status", "created_at", "result"}]
        self.notes = []              # [{"id", "text", "created_at", "done"}]

    # -- Thread-safe setters --

    def set_status(self, status: str):
        with self._lock:
            self.status = status
            self._version += 1

    def set_module_status(self, name: str, status: str, load_time_ms: Optional[float] = None):
        with self._lock:
            self.modules[name] = {"status": status, "load_time_ms": load_time_ms}
            self._version += 1

    def set_audio_level(self, level: float):
        with self._lock:
            self.audio_level = level
            # Don't bump version for audio — too frequent. WebSocket polls it.

    def set_scene_score(self, score: float):
        with self._lock:
            self.scene_change_score = score
            self._version += 1

    def set_cost(self, cost: float):
        with self._lock:
            self.cost_today = cost
            self._version += 1

    def set_memory_stats(self, episodes: int, facts: int):
        with self._lock:
            self.memory_episodes = episodes
            self.memory_facts = facts
            self._version += 1

    def set_always_listen_state(self, state: str):
        with self._lock:
            if self.always_listen_state != state:
                self.always_listen_state = state
                self._version += 1

    def add_conversation(self, role: str, text: str):
        with self._lock:
            self.conversation.append({
                "role": role,
                "text": text,
                "timestamp": time.strftime("%H:%M:%S"),
            })
            # Keep last 50 messages
            if len(self.conversation) > 50:
                self.conversation = self.conversation[-50:]
            self._version += 1

    def add_observation(self, description: str):
        with self._lock:
            self.observations.append({
                "description": description,
                "timestamp": time.strftime("%H:%M:%S"),
            })
            # Keep last 20
            if len(self.observations) > 20:
                self.observations = self.observations[-20:]
            self._version += 1

    # -- Agent tasks --

    def set_agent_tasks(self, tasks: list):
        with self._lock:
            self.agent_tasks = tasks[-20:]
            self._version += 1

    def add_agent_task(self, task: dict):
        with self._lock:
            self.agent_tasks.append(task)
            if len(self.agent_tasks) > 20:
                self.agent_tasks = self.agent_tasks[-20:]
            self._version += 1

    def update_agent_task(self, task_id: str, status: str, result: str = None):
        with self._lock:
            for t in self.agent_tasks:
                if t.get("id") == task_id:
                    t["status"] = status
                    if result is not None:
                        t["result"] = result
                    self._version += 1
                    break

    # -- Notes --

    def add_note(self, note: dict):
        with self._lock:
            self.notes.append(note)
            if len(self.notes) > 50:
                self.notes = self.notes[-50:]
            self._version += 1

    def set_notes(self, notes: list):
        with self._lock:
            self.notes = notes[-50:]
            self._version += 1

    def toggle_note(self, note_id: str) -> bool:
        with self._lock:
            for n in self.notes:
                if n.get("id") == note_id:
                    n["done"] = not n.get("done", False)
                    self._version += 1
                    return True
            return False

    def remove_note(self, note_id: str) -> bool:
        with self._lock:
            original_len = len(self.notes)
            self.notes = [n for n in self.notes if n.get("id") != note_id]
            if len(self.notes) < original_len:
                self._version += 1
                return True
            return False

    # -- Snapshot for WebSocket --

    @property
    def version(self) -> int:
        return self._version

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "modules": dict(self.modules),
                "audioLevel": self.audio_level,
                "sceneChangeScore": self.scene_change_score,
                "costToday": round(self.cost_today, 4),
                "budgetMax": self.budget_max,
                "conversation": list(self.conversation),
                "observations": list(self.observations),
                "memoryEpisodes": self.memory_episodes,
                "memoryFacts": self.memory_facts,
                "uptime": round(time.time() - self._start_time, 1),
                "alwaysListenState": self.always_listen_state,
                "agentTasks": list(self.agent_tasks),
                "notes": list(self.notes),
                "version": self._version,
            }
