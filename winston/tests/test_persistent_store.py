"""Tests for PersistentStore — thread-safe atomic JSON storage."""

import json

from utils.persistent_store import PersistentStore


def test_get_set_basic(tmp_path):
    """Basic key-value get/update operations."""
    store = PersistentStore(str(tmp_path / "test.json"))

    assert store.get("key1") is None
    assert store.get("missing", "default") == "default"

    store.update("key1", "value1")
    assert store.get("key1") == "value1"

    store.update("key1", "value2")
    assert store.get("key1") == "value2"


def test_append_to_list(tmp_path):
    """Append items to a list field."""
    store = PersistentStore(str(tmp_path / "test.json"))

    for i in range(5):
        store.append_to_list("items", {"id": str(i), "val": i})

    items = store.get("items")
    assert len(items) == 5
    assert items[0]["id"] == "0"
    assert items[4]["id"] == "4"


def test_append_max_items(tmp_path):
    """Append with max_items=3: only the last 3 remain after adding 5."""
    store = PersistentStore(str(tmp_path / "test.json"))

    for i in range(5):
        store.append_to_list("items", {"id": str(i)}, max_items=3)

    items = store.get("items")
    assert len(items) == 3
    assert items[0]["id"] == "2"  # oldest surviving
    assert items[2]["id"] == "4"  # newest


def test_atomic_write(tmp_path):
    """After update, the file on disk is valid JSON."""
    path = tmp_path / "test.json"
    store = PersistentStore(str(path))

    store.update("hello", "world")
    store.append_to_list("nums", 42)

    # Read file directly from disk
    with open(path, "r") as f:
        data = json.load(f)

    assert data["hello"] == "world"
    assert data["nums"] == [42]


def test_update_in_list(tmp_path):
    """Update a specific item in a list by its ID field."""
    store = PersistentStore(str(tmp_path / "test.json"))

    store.update(
        "tasks",
        [
            {"id": "t1", "status": "pending"},
            {"id": "t2", "status": "pending"},
            {"id": "t3", "status": "pending"},
        ],
    )

    # Update existing item
    result = store.update_in_list("tasks", "t2", {"status": "done"})
    assert result is True

    tasks = store.get("tasks")
    t2 = [t for t in tasks if t["id"] == "t2"][0]
    assert t2["status"] == "done"

    # Other items unchanged
    t1 = [t for t in tasks if t["id"] == "t1"][0]
    assert t1["status"] == "pending"

    # Non-existent item returns False
    result = store.update_in_list("tasks", "t99", {"status": "done"})
    assert result is False
