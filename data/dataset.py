from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import Dataset

from .pack import PackedBatch


class PackedWindowDataset(Dataset):
    """Map-style dataset that keeps segment and state-reset metadata with tokens."""

    def __init__(self, packed: PackedBatch, window_size: int | None = None) -> None:
        self.packed = packed
        total_tokens = int(self.packed.input_ids.numel())
        self.window_size = int(window_size) if window_size is not None else int(self.packed.input_ids.shape[1])
        self.num_windows = max(0, total_tokens // self.window_size)
        self.input_ids_flat = self.packed.input_ids.reshape(-1)
        self.labels_flat = self.packed.labels.reshape(-1)
        self.segment_ids_flat = self.packed.segment_ids.reshape(-1)
        self.reset_mask_flat = self.packed.reset_mask.reshape(-1)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = index * self.window_size
        end = start + self.window_size
        reset = self.reset_mask_flat[start:end].clone()
        if reset.numel() > 0:
            reset[0] = True
        return {
            "input_ids": self.input_ids_flat[start:end],
            "labels": self.labels_flat[start:end],
            "segment_ids": self.segment_ids_flat[start:end],
            "reset_mask": reset,
        }


def collate_packed_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collates a list of window dicts into stacked batch tensors."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "segment_ids": torch.stack([item["segment_ids"] for item in batch], dim=0),
        "reset_mask": torch.stack([item["reset_mask"] for item in batch], dim=0),
    }


class ShardedPackedWindowLoader:
    """Streams micro-batches continuously across multiple PackedBatch .pt shards."""

    def __init__(self, shard_paths: list[str | Path], *, window_size: int = 256,
                 batch_size: int = 2, shuffle_shards: bool = True,
                 shuffle_windows: bool = True, seed: int = 1337,
                 start_tokens: int = 0) -> None:
        from pathlib import Path
        self.shard_paths = [Path(p) for p in sorted(shard_paths)]
        if not self.shard_paths:
            raise ValueError("No shard paths provided to ShardedPackedWindowLoader")
        self.window_size = int(window_size)
        self.batch_size = int(batch_size)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_windows = bool(shuffle_windows)
        self.seed = int(seed)
        self.start_tokens = int(start_tokens)

    def __iter__(self):
        import random
        from torch.utils.data import DataLoader

        epoch = 0
        tokens_skipped = 0
        while True:
            rng = random.Random(self.seed + epoch)
            indices = list(range(len(self.shard_paths)))
            if self.shuffle_shards:
                rng.shuffle(indices)
            for idx in indices:
                shard_path = self.shard_paths[idx]
                try:
                    packed = torch.load(
                        shard_path, map_location="cpu", weights_only=False, mmap=True
                    )
                except (RuntimeError, TypeError):
                    # Older non-zip shards cannot be memory-mapped; keep them
                    # readable while new shards use the zero-copy path.
                    packed = torch.load(shard_path, map_location="cpu", weights_only=False)
                if not isinstance(packed, PackedBatch):
                    continue
                dataset = PackedWindowDataset(packed, window_size=self.window_size)
                worker_count = 0 if os.name == "nt" else 2
                loader_kwargs = {
                    "batch_size": self.batch_size,
                    "shuffle": self.shuffle_windows,
                    "collate_fn": collate_packed_windows,
                    "num_workers": worker_count,
                    "persistent_workers": False,
                    "pin_memory": False,
                }
                if worker_count:
                    loader_kwargs["prefetch_factor"] = 4
                loader = DataLoader(dataset, **loader_kwargs)
                for batch in loader:
                    batch_tok = int(batch["input_ids"].numel())
                    if tokens_skipped + batch_tok <= self.start_tokens:
                        tokens_skipped += batch_tok
                        continue
                    batch["_has_reset"] = bool(batch["reset_mask"][:, 0].any())
                    yield batch
            epoch += 1
