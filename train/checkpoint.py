"""Atomic, resumable checkpoint format for the V0 run."""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch


def _rng_state() -> dict[str, Any]:
    value: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        value["cuda"] = torch.cuda.get_rng_state_all()
    return value


def _restore_rng(value: dict[str, Any]) -> None:
    if not value:
        return
    random.setstate(value["python"])
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and "cuda" in value:
        torch.cuda.set_rng_state_all(value["cuda"])


class CheckpointManager:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, *, model: torch.nn.Module, optimizer: Any,
             scheduler: Any = None, token_count: int = 0, step: int = 0,
             data_cursor: Any = None, config: dict[str, Any] | None = None,
             manifest_hashes: dict[str, str] | None = None,
             telemetry: dict[str, Any] | None = None) -> Path:
        destination = self.directory / name
        payload = {
            "format_version": 1,
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "token_count": int(token_count),
            "step": int(step),
            "data_cursor": data_cursor,
            "rng": _rng_state(),
            "config": config,
            "manifest_hashes": manifest_hashes or {},
            "parameter_report": getattr(model, "parameter_report", lambda: {})(),
            "telemetry": telemetry or {},
        }
        with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def load(self, path: str | Path, *, model: torch.nn.Module, optimizer: Any = None,
             scheduler: Any = None, restore_rng: bool = True, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported V0 checkpoint format")
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if restore_rng:
            _restore_rng(payload.get("rng", {}))
        return payload
