"""Thread-safe, crash-safe JSON file store.

Uses atomic writes (tempfile + os.replace) so data is never corrupted
even if the process is killed mid-write. Used for agent tasks and notes.
"""

import json
import logging
import os
import tempfile
import threading
from typing import Any

logger = logging.getLogger("winston.store")


class PersistentStore:
    """Thread-safe JSON file with atomic writes."""

    def __init__(self, file_path: str, default_data: dict = None):
        self._path = os.path.abspath(file_path)
        self._default = default_data or {}
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        """Load from disk or return default."""
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load %s: %s — using defaults", self._path, e)
        return dict(self._default)

    def _save(self) -> None:
        """Atomic write: write to tempfile, then os.replace."""
        dir_path = os.path.dirname(self._path)
        os.makedirs(dir_path, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, default=str)
            os.replace(tmp_path, self._path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe read."""
        with self._lock:
            return self._data.get(key, default)

    def update(self, key: str, value: Any) -> None:
        """Thread-safe write + persist."""
        with self._lock:
            self._data[key] = value
            self._save()

    def append_to_list(self, key: str, item: Any, max_items: int = 100) -> None:
        """Append an item to a list field, keeping at most max_items."""
        with self._lock:
            lst = self._data.get(key, [])
            lst.append(item)
            if len(lst) > max_items:
                lst = lst[-max_items:]
            self._data[key] = lst
            self._save()

    def update_in_list(self, key: str, item_id: str, updates: dict, id_field: str = "id") -> bool:
        """Update a specific item in a list by its ID field."""
        with self._lock:
            lst = self._data.get(key, [])
            for item in lst:
                if item.get(id_field) == item_id:
                    item.update(updates)
                    self._save()
                    return True
            return False

    def remove_from_list(self, key: str, item_id: str, id_field: str = "id") -> bool:
        """Remove an item from a list by its ID field."""
        with self._lock:
            lst = self._data.get(key, [])
            original_len = len(lst)
            lst = [item for item in lst if item.get(id_field) != item_id]
            if len(lst) < original_len:
                self._data[key] = lst
                self._save()
                return True
            return False

    @property
    def data(self) -> dict:
        """Read-only snapshot of all data."""
        with self._lock:
            return dict(self._data)
