"""Atomic, operator-editable controls for a live training process."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RuntimeControl:
    """Poll a JSON control file without interrupting the training loop.

    Operators should write a temporary JSON file and atomically rename it over
    ``runtime_control.json``. The trainer also tolerates a partially-written
    file by retaining the last valid state.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.last_mtime_ns = -1
        self.values: dict[str, Any] = {}

    def poll(self) -> tuple[bool, dict[str, Any]]:
        if not self.path.is_file():
            return False, self.values
        try:
            stat = self.path.stat()
            if stat.st_mtime_ns == self.last_mtime_ns:
                return False, self.values
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("runtime control must contain a JSON object")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"[RuntimeControl] Ignoring invalid control file: {exc}", flush=True)
            return False, self.values
        self.last_mtime_ns = stat.st_mtime_ns
        self.values = parsed
        return True, self.values

