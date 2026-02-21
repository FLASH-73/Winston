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
        self.audio_level = 0.0  # RMS 0.0-1.0 from mic
        self.scene_change_score = 0.0
        self.cost_today = 0.0
        self.budget_max = 2.0
        self.always_listen_state = "disabled"
        self.music_mode = False
        self.muted = False

        # Content
        self.conversation = []  # [{"role": str, "text": str, "timestamp": str}]
        self.observations = []  # [{"description": str, "timestamp": str}]
        self.memory_episodes = 0
        self.memory_facts = 0

        # Agent tasks & Notes
        self.agent_tasks = []  # [{"id", "query", "status", "created_at", "result"}]
        self.notes = []  # [{"id", "text", "created_at", "done"}]

        # Latency stats
        self.latency_stats = {}  # {"e2e": {"avg", "p50", "p95", "count"}, ...}
        self.last_e2e_ms = 0.0
        self.interaction_count = 0

    # -- Thread-safe setters --

    def set_status(self, status: str):
        """Set the system-wide status label (e.g. 'idle', 'thinking', 'speaking')."""
        with self._lock:
            self.status = status
            self._version += 1

    def set_module_status(self, name: str, status: str, load_time_ms: Optional[float] = None):
        """Set a module's status and optional load time for the dashboard."""
        with self._lock:
            self.modules[name] = {"status": status, "load_time_ms": load_time_ms}
            self._version += 1

    def set_audio_level(self, level: float):
        """Update current mic RMS level (0.0-1.0). Does not bump version (too frequent)."""
        with self._lock:
            self.audio_level = level
            # Don't bump version for audio — too frequent. WebSocket polls it.

    def set_scene_score(self, score: float):
        """Update the latest scene change score (0.0-1.0)."""
        with self._lock:
            self.scene_change_score = score
            self._version += 1

    def set_cost(self, cost: float):
        """Update today's cumulative API cost in USD."""
        with self._lock:
            self.cost_today = cost
            self._version += 1

    def set_memory_stats(self, episodes: int, facts: int):
        """Update the episode and fact counts shown on the dashboard."""
        with self._lock:
            self.memory_episodes = episodes
            self.memory_facts = facts
            self._version += 1

    def set_always_listen_state(self, state: str):
        """Update always-listen state machine label (idle/accumulating/cooldown/disabled)."""
        with self._lock:
            if self.always_listen_state != state:
                self.always_listen_state = state
                self._version += 1

    def set_music_mode(self, enabled: bool):
        """Update music mode status."""
        with self._lock:
            if self.music_mode != enabled:
                self.music_mode = enabled
                self._version += 1

    def set_muted(self, muted: bool):
        """Update microphone mute status."""
        with self._lock:
            if self.muted != muted:
                self.muted = muted
                self._version += 1

    def set_latency_stats(self, stats: dict):
        """Update latency statistics from the latency tracker."""
        with self._lock:
            self.latency_stats = stats
            self.last_e2e_ms = stats.get("last_e2e", 0.0)
            self.interaction_count = stats.get("interaction_count", 0)
            self._version += 1

    def add_conversation(self, role: str, text: str):
        """Append a conversation turn. Keeps the last 50 messages."""
        with self._lock:
            self.conversation.append(
                {
                    "role": role,
                    "text": text,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            )
            # Keep last 50 messages
            if len(self.conversation) > 50:
                self.conversation = self.conversation[-50:]
            self._version += 1

    def add_observation(self, description: str):
        """Append a scene observation. Keeps the last 20."""
        with self._lock:
            self.observations.append(
                {
                    "description": description,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            )
            # Keep last 20
            if len(self.observations) > 20:
                self.observations = self.observations[-20:]
            self._version += 1

    # -- Agent tasks --

    def set_agent_tasks(self, tasks: list):
        """Replace the agent tasks list. Keeps the last 20."""
        with self._lock:
            self.agent_tasks = tasks[-20:]
            self._version += 1

    def add_agent_task(self, task: dict):
        """Append an agent task record. Keeps the last 20."""
        with self._lock:
            self.agent_tasks.append(task)
            if len(self.agent_tasks) > 20:
                self.agent_tasks = self.agent_tasks[-20:]
            self._version += 1

    def update_agent_task(self, task_id: str, status: str, result: str = None):
        """Update status and optional result for an agent task by ID."""
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
        """Append a note. Keeps the last 50."""
        with self._lock:
            self.notes.append(note)
            if len(self.notes) > 50:
                self.notes = self.notes[-50:]
            self._version += 1

    def set_notes(self, notes: list):
        """Replace the notes list. Keeps the last 50."""
        with self._lock:
            self.notes = notes[-50:]
            self._version += 1

    def toggle_note(self, note_id: str) -> bool:
        """Toggle a note's done state. Returns True if found."""
        with self._lock:
            for n in self.notes:
                if n.get("id") == note_id:
                    n["done"] = not n.get("done", False)
                    self._version += 1
                    return True
            return False

    def remove_note(self, note_id: str) -> bool:
        """Remove a note by ID. Returns True if found and removed."""
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
        """Monotonic counter incremented on state changes. Used by WebSocket."""
        return self._version

    def to_dict(self) -> dict:
        """Snapshot full state as a JSON-serializable dict for WebSocket push."""
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
                "musicMode": self.music_mode,
                "muted": self.muted,
                "agentTasks": list(self.agent_tasks),
                "notes": list(self.notes),
                "latencyStats": dict(self.latency_stats),
                "lastE2eMs": self.last_e2e_ms,
                "interactionCount": self.interaction_count,
                "version": self._version,
            }
