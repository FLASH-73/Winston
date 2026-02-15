"""Tests for dashboard/state.py — thread-safe shared state."""

import threading

from dashboard.state import WinstonState


class TestWinstonStateBasics:
    def test_initial_state(self):
        state = WinstonState()
        assert state.status == "initializing"
        assert state.conversation == []
        assert state.observations == []
        assert state.agent_tasks == []
        assert state.notes == []
        assert state.version == 0

    def test_set_status_bumps_version(self):
        state = WinstonState()
        state.set_status("online")
        assert state.status == "online"
        assert state.version == 1

    def test_set_module_status(self):
        state = WinstonState()
        state.set_module_status("camera", "ready", load_time_ms=150.0)
        assert state.modules["camera"]["status"] == "ready"
        assert state.modules["camera"]["load_time_ms"] == 150.0

    def test_audio_level_no_version_bump(self):
        state = WinstonState()
        state.set_audio_level(0.5)
        assert state.audio_level == 0.5
        assert state.version == 0  # audio level doesn't bump version

    def test_set_cost(self):
        state = WinstonState()
        state.set_cost(0.42)
        assert state.cost_today == 0.42
        assert state.version == 1

    def test_set_memory_stats(self):
        state = WinstonState()
        state.set_memory_stats(episodes=100, facts=25)
        assert state.memory_episodes == 100
        assert state.memory_facts == 25

    def test_always_listen_state_only_bumps_on_change(self):
        state = WinstonState()
        state.set_always_listen_state("idle")
        v1 = state.version
        state.set_always_listen_state("idle")  # same value
        assert state.version == v1  # no bump


class TestWinstonStateConversation:
    def test_add_conversation(self):
        state = WinstonState()
        state.add_conversation("user", "Hello Winston")
        assert len(state.conversation) == 1
        assert state.conversation[0]["role"] == "user"
        assert state.conversation[0]["text"] == "Hello Winston"
        assert "timestamp" in state.conversation[0]

    def test_conversation_truncates_at_50(self):
        state = WinstonState()
        for i in range(55):
            state.add_conversation("user", f"message {i}")
        assert len(state.conversation) == 50
        assert state.conversation[0]["text"] == "message 5"
        assert state.conversation[-1]["text"] == "message 54"


class TestWinstonStateObservations:
    def test_add_observation(self):
        state = WinstonState()
        state.add_observation("Person at workbench")
        assert len(state.observations) == 1
        assert state.observations[0]["description"] == "Person at workbench"

    def test_observations_truncate_at_20(self):
        state = WinstonState()
        for i in range(25):
            state.add_observation(f"obs {i}")
        assert len(state.observations) == 20
        assert state.observations[0]["description"] == "obs 5"


class TestWinstonStateAgentTasks:
    def test_add_agent_task(self):
        state = WinstonState()
        state.add_agent_task({"id": "t1", "query": "search web", "status": "running"})
        assert len(state.agent_tasks) == 1

    def test_update_agent_task(self):
        state = WinstonState()
        state.add_agent_task({"id": "t1", "query": "search", "status": "running"})
        state.update_agent_task("t1", "completed", result="Found 3 results")
        assert state.agent_tasks[0]["status"] == "completed"
        assert state.agent_tasks[0]["result"] == "Found 3 results"

    def test_set_agent_tasks_truncates(self):
        state = WinstonState()
        tasks = [{"id": str(i)} for i in range(25)]
        state.set_agent_tasks(tasks)
        assert len(state.agent_tasks) == 20


class TestWinstonStateNotes:
    def test_add_note(self):
        state = WinstonState()
        state.add_note({"id": "n1", "text": "Buy soldering wire", "done": False})
        assert len(state.notes) == 1

    def test_toggle_note(self):
        state = WinstonState()
        state.add_note({"id": "n1", "text": "Buy wire", "done": False})
        result = state.toggle_note("n1")
        assert result is True
        assert state.notes[0]["done"] is True

        # Toggle back
        state.toggle_note("n1")
        assert state.notes[0]["done"] is False

    def test_toggle_nonexistent_note(self):
        state = WinstonState()
        assert state.toggle_note("nonexistent") is False

    def test_remove_note(self):
        state = WinstonState()
        state.add_note({"id": "n1", "text": "keep"})
        state.add_note({"id": "n2", "text": "remove"})
        result = state.remove_note("n2")
        assert result is True
        assert len(state.notes) == 1
        assert state.notes[0]["id"] == "n1"

    def test_remove_nonexistent_note(self):
        state = WinstonState()
        assert state.remove_note("nonexistent") is False


class TestWinstonStateSnapshot:
    def test_to_dict_structure(self):
        state = WinstonState()
        state.set_status("online")
        d = state.to_dict()
        assert d["status"] == "online"
        assert "audioLevel" in d
        assert "costToday" in d
        assert "conversation" in d
        assert "observations" in d
        assert "agentTasks" in d
        assert "notes" in d
        assert "uptime" in d
        assert "version" in d
        assert isinstance(d["uptime"], float)


class TestWinstonStateThreadSafety:
    def test_concurrent_conversation_adds(self):
        state = WinstonState()
        errors = []

        def add_messages(prefix):
            try:
                for i in range(20):
                    state.add_conversation("user", f"{prefix}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_messages, args=(f"t{t}",)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # 100 messages added, capped at 50
        assert len(state.conversation) == 50
