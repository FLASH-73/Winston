"""Tests for utils/persistent_store.py — thread-safe atomic JSON storage."""

import json
import os
import threading

from utils.persistent_store import PersistentStore


class TestPersistentStoreBasics:
    def test_create_empty(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PersistentStore(path)
        assert store.data == {}

    def test_create_with_defaults(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PersistentStore(path, default_data={"items": [], "count": 0})
        assert store.get("items") == []
        assert store.get("count") == 0

    def test_get_missing_key_returns_default(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PersistentStore(path)
        assert store.get("missing") is None
        assert store.get("missing", 42) == 42

    def test_update_and_get(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PersistentStore(path)
        store.update("name", "Winston")
        assert store.get("name") == "Winston"

    def test_persists_to_disk(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PersistentStore(path)
        store.update("key", "value")

        # Read the file directly
        with open(path) as f:
            data = json.load(f)
        assert data["key"] == "value"

    def test_loads_from_existing_file(self, tmp_path):
        path = str(tmp_path / "store.json")
        with open(path, "w") as f:
            json.dump({"existing": True}, f)

        store = PersistentStore(path)
        assert store.get("existing") is True

    def test_creates_parent_directories(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "store.json")
        store = PersistentStore(path)
        store.update("works", True)
        assert os.path.exists(path)


class TestPersistentStoreListOperations:
    def test_append_to_list(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PersistentStore(path, default_data={"items": []})
        store.append_to_list("items", {"id": "1", "text": "first"})
        store.append_to_list("items", {"id": "2", "text": "second"})
        assert len(store.get("items")) == 2

    def test_append_to_list_truncates_at_max(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PersistentStore(path, default_data={"items": []})
        for i in range(15):
            store.append_to_list("items", {"id": str(i)}, max_items=10)
        items = store.get("items")
        assert len(items) == 10
        # Should keep the most recent items
        assert items[0]["id"] == "5"
        assert items[-1]["id"] == "14"

    def test_update_in_list(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PersistentStore(path, default_data={"items": []})
        store.append_to_list("items", {"id": "1", "status": "pending"})
        result = store.update_in_list("items", "1", {"status": "done"})
        assert result is True
        assert store.get("items")[0]["status"] == "done"

    def test_update_in_list_missing_id(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PersistentStore(path, default_data={"items": []})
        result = store.update_in_list("items", "nonexistent", {"status": "done"})
        assert result is False

    def test_remove_from_list(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PersistentStore(path, default_data={"items": []})
        store.append_to_list("items", {"id": "1", "text": "keep"})
        store.append_to_list("items", {"id": "2", "text": "remove"})
        result = store.remove_from_list("items", "2")
        assert result is True
        assert len(store.get("items")) == 1
        assert store.get("items")[0]["id"] == "1"

    def test_remove_from_list_missing_id(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PersistentStore(path, default_data={"items": []})
        result = store.remove_from_list("items", "nonexistent")
        assert result is False


class TestPersistentStoreThreadSafety:
    def test_concurrent_writes(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PersistentStore(path, default_data={"items": []})
        errors = []

        def append_items(start):
            try:
                for i in range(20):
                    store.append_to_list("items", {"id": f"{start}-{i}"}, max_items=500)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=append_items, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store.get("items")) == 100  # 5 threads * 20 items

    def test_data_property_returns_snapshot(self, tmp_path):
        path = str(tmp_path / "store.json")
        store = PersistentStore(path, default_data={"key": "original"})
        snapshot = store.data
        store.update("key", "modified")
        # Snapshot should NOT reflect the update
        assert snapshot["key"] == "original"
        assert store.get("key") == "modified"
