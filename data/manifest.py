"""Reproducible source/license and shard manifest records."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    path: str
    license: str
    url: str | None = None
    split: str = "train"
    sha256: str | None = None
    document_count: int | None = None


@dataclass
class CorpusManifest:
    tokenizer_name: str
    tokenizer_hash: str
    context_length: int
    source_records: list[SourceRecord] = field(default_factory=list)
    filter_config: dict[str, Any] = field(default_factory=dict)
    train_shards: list[str] = field(default_factory=list)
    validation_shards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "source_records": [asdict(record) for record in self.source_records],
        }

    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return sha256_bytes(encoded)

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.to_dict(), "manifest_sha256": self.digest()}
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "CorpusManifest":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        value.pop("manifest_sha256", None)
        value["source_records"] = [SourceRecord(**record) for record in value.get("source_records", [])]
        return cls(**value)
