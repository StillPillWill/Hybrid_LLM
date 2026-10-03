"""Concurrent Background Streaming Dataset for FineWeb-Edu.

Features:
- Spawns a background thread that streams from HuggingFace, tokenizes, and packs shards.
- Starts training immediately after the first shard is ready (zero idle waiting).
- Automatically tracks and skips already-trained tokens when resuming.
"""
from __future__ import annotations

import glob
import os
import queue
import random
import threading
import time
from pathlib import Path
from typing import Iterator

import torch

from data.dataset import PackedWindowDataset, collate_packed_windows
from data.pack import PackedBatch


def _reset_mask_from_segments(segment_ids: torch.Tensor) -> torch.Tensor:
    """Reset recurrent state at row starts and every document transition."""
    if segment_ids.ndim != 2:
        raise ValueError(f"segment_ids must be [rows, tokens], got {tuple(segment_ids.shape)}")
    reset = torch.zeros_like(segment_ids, dtype=torch.bool)
    if segment_ids.shape[1] == 0:
        return reset
    reset[:, 0] = True
    if segment_ids.shape[1] > 1:
        reset[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
    return reset


def _save_packed_shard(
    shard_path: Path,
    rows: list[list[int]],
    segment_rows: list[list[int]],
    document_rows: list[list[str]],
    *,
    provenance: dict | None = None,
) -> None:
    """Atomically save a shard with consistent boundary metadata."""
    if not rows:
        raise ValueError("cannot save an empty shard")
    input_ids = torch.tensor(rows, dtype=torch.int64)
    labels = input_ids.clone()
    segment_ids = torch.tensor(segment_rows, dtype=torch.int32)
    batch_obj = PackedBatch(
        input_ids=input_ids,
        labels=labels,
        segment_ids=segment_ids,
        reset_mask=_reset_mask_from_segments(segment_ids),
        document_ids=[list(row) for row in document_rows],
        provenance=provenance,
    )
    temporary = shard_path.with_name(f".{shard_path.name}.tmp")
    try:
        torch.save(batch_obj, temporary)
        os.replace(temporary, shard_path)
    finally:
        temporary.unlink(missing_ok=True)


class ConcurrentStreamingLoader:
    """Streams data shards concurrently while GPU trains."""

    def __init__(
        self,
        shard_dir: str | Path,
        *,
        window_size: int = 256,
        batch_size: int = 16,
        start_tokens: int = 0,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_config: str = "CC-MAIN-2024-10",
        min_shards_to_start: int = 1,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = int(window_size)
        self.batch_size = int(batch_size)
        self.start_tokens = int(start_tokens)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.min_shards_to_start = int(min_shards_to_start)
        # Keep the downloader from competing with the trainer once there is a
        # healthy queue of ready shards.  A large unbounded cache creates
        # sustained tokenizer/disk contention and leaves the GPU underfed.
        self.max_prefetch_shards = max(
            8, int(os.getenv("LLMY_MAX_PREFETCH_SHARDS", "64"))
        )

        self.shards_ready = queue.Queue()
        self.processed_shards = set()
        self.total_downloaded_shards = 0
        self.is_downloading = True
        self._stop_event = threading.Event()

        # Check existing shards on disk first
        existing_shards = sorted(self.shard_dir.glob("train-*.pt"))
        for s in existing_shards:
            self.shards_ready.put(s)
            self.total_downloaded_shards += 1

        # Start background streamer
        self.worker_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.worker_thread.start()

    def _stream_worker(self) -> None:
        """Background thread that streams, tokenizes, and writes shards to disk."""
        try:
            from datasets import load_dataset
            from data.download_v6_pretraining import StreamingPacker
            from data.filter import is_pathological, normalize_text
            from data.tokenize import load_qwen35_tokenizer

            tokenizer = load_qwen35_tokenizer()
            packer = StreamingPacker(context_length=self.window_size, rows_per_shard=1024)

            dataset = load_dataset(
                self.dataset_name,
                name=self.dataset_config,
                split="train",
                streaming=True,
            )

            existing_indices = []
            for shard in self.shard_dir.glob("train-*.pt"):
                try:
                    existing_indices.append(int(shard.stem.rsplit("-", 1)[1]))
                except (IndexError, ValueError):
                    continue
            current_shard_idx = max(existing_indices, default=-1) + 1

            for item in dataset:
                if self._stop_event.is_set():
                    break
                while (
                    not self._stop_event.is_set()
                    and self.shards_ready.qsize() >= self.max_prefetch_shards
                ):
                    time.sleep(0.5)
                if self._stop_event.is_set():
                    break

                text = normalize_text(item.get("text", ""))
                if not text or is_pathological(text):
                    continue

                token_ids = tokenizer.encode(text, add_special_tokens=False)
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if eos_token_id is not None:
                    token_ids.append(int(eos_token_id))
                packer.add(item.get("id", f"doc_{packer.segment_id}"), token_ids)

                # A long document can make StreamingPacker cross the shard
                # boundary by many rows.  Drain fixed-size chunks and retain
                # the remainder for the next shard instead of writing 1,024+
                # rows into one file.
                while packer.row_count >= packer.rows_per_shard:
                    shard_name = f"train-{current_shard_idx:05d}.pt"
                    shard_path = self.shard_dir / shard_name
                    rows = packer.rows[:packer.rows_per_shard]
                    segment_rows = packer.segment_rows[:packer.rows_per_shard]
                    document_rows = packer.document_rows[:packer.rows_per_shard]
                    _save_packed_shard(
                        shard_path,
                        rows,
                        segment_rows,
                        document_rows,
                        provenance={"shard": shard_name, "rows": packer.rows_per_shard},
                    )
                    self.shards_ready.put(shard_path)
                    self.total_downloaded_shards += 1
                    current_shard_idx += 1
                    del packer.rows[:packer.rows_per_shard]
                    del packer.segment_rows[:packer.rows_per_shard]
                    del packer.document_rows[:packer.rows_per_shard]

        except Exception as e:
            print(f"[StreamingWorker] Warning: {e}")
        finally:
            self.is_downloading = False

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # Skip by the actual tensor length of each shard.  Shard indices and a
        # nominal 1,024-row capacity are not valid offsets because older live
        # shards may contain 1,024--1,216 rows.
        tokens_to_skip = self.start_tokens
        pending_windows: list[dict[str, object]] = []

        while self.shards_ready.empty() and self.is_downloading:
            time.sleep(0.5)

        while True:
            try:
                shard_path = self.shards_ready.get(timeout=2.0)
            except queue.Empty:
                if not self.is_downloading:
                    break
                continue

            try:
                try:
                    packed = torch.load(
                        shard_path, map_location="cpu", weights_only=False, mmap=True
                    )
                except (RuntimeError, TypeError):
                    packed = torch.load(shard_path, map_location="cpu", weights_only=False)
            except Exception:
                continue

            shard_tokens = int(packed.input_ids.numel())
            if tokens_to_skip >= shard_tokens:
                tokens_to_skip -= shard_tokens
                continue

            dataset = PackedWindowDataset(packed, window_size=self.window_size)
            if len(dataset) == 0:
                continue
            item_tokens = int(dataset[0]["input_ids"].numel())
            if tokens_to_skip > 0:
                first_item = min(tokens_to_skip // item_tokens, len(dataset))
                tokens_to_skip = max(0, tokens_to_skip - first_item * item_tokens)
            else:
                first_item = 0

            # Keep windows in source order and carry the tail of one shard
            # into the next.  This guarantees every yielded batch is full and
            # makes a token offset deterministic across variable shard sizes.
            for index in range(first_item, len(dataset)):
                item = dataset[index]
                if torch.cuda.is_available():
                    for key in ("input_ids", "labels", "segment_ids", "reset_mask"):
                        item[key] = item[key].pin_memory()
                pending_windows.append(item)
                if len(pending_windows) < self.batch_size:
                    continue
                batch = collate_packed_windows(pending_windows)  # type: ignore[arg-type]
                # Compute this on the CPU loader thread once instead of
                # forcing a host synchronization in every model forward.
                batch["_has_reset"] = bool(batch["reset_mask"][:, 0].any())
                yield batch
                pending_windows.clear()

            if not self.is_downloading and self.shards_ready.empty():
                # Do not emit a short final batch: the trainer's global token
                # accounting assumes full batches, and the stream should have
                # enough data for the requested budget.
                pending_windows.clear()

    def stop(self) -> None:
        self._stop_event.set()


class WeightedInterleavedLoader:
    """Randomly interleave full batches from multiple streaming sources.

    Each source keeps its own bounded downloader and iterator.  The trainer
    chooses the source independently for every batch using the configured
    weights, so source order is randomized without concatenating source runs
    into recognizable blocks.
    """

    def __init__(
        self,
        sources: list[dict[str, object]],
        *,
        window_size: int,
        batch_size: int,
        seed: int = 1337,
        start_batches: list[int] | dict[str, int] | None = None,
        initial_rng_state: object | None = None,
    ) -> None:
        if not sources:
            raise ValueError("weighted interleaving requires at least one source")
        self.sources = [dict(source) for source in sources]
        self.shard_dir = Path("<weighted-interleave>")
        self.dataset_name = "weighted-interleave"
        self.dataset_config = ",".join(
            str(source.get("name", index)) for index, source in enumerate(self.sources)
        )
        self.rng = random.Random(int(seed))
        if initial_rng_state is not None:
            self.rng.setstate(initial_rng_state)
        self.loaders: list[ConcurrentStreamingLoader] = []
        self.iterators: list[Iterator[dict[str, torch.Tensor]]] = []
        self.weights: list[float] = []
        self.active: list[bool] = []
        self.batches_by_source = [0 for _ in self.sources]
        for index, source in enumerate(self.sources):
            weight = float(source.get("weight", 1.0))
            if weight < 0:
                raise ValueError(f"source weight must be non-negative: {source}")
            self.weights.append(weight)
            source_start_batches = 0
            if isinstance(start_batches, list) and index < len(start_batches):
                source_start_batches = int(start_batches[index])
            elif isinstance(start_batches, dict):
                source_start_batches = int(start_batches.get(str(source.get("name", index)), 0))
            self.batches_by_source[index] = source_start_batches
            source_start_tokens = source_start_batches * batch_size * window_size
            loader = ConcurrentStreamingLoader(
                str(source["dir"]),
                window_size=window_size,
                batch_size=batch_size,
                start_tokens=source_start_tokens,
                dataset_name=str(source.get("dataset_name", "HuggingFaceFW/fineweb-edu")),
                dataset_config=str(source.get("dataset_config", "CC-MAIN-2024-10")),
            )
            self.loaders.append(loader)
            self.iterators.append(iter(loader))
            self.active.append(True)
        if not any(weight > 0 for weight in self.weights):
            raise ValueError("at least one interleaved source must have positive weight")

    def __iter__(self) -> "WeightedInterleavedLoader":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        while any(self.active):
            active_indices = [index for index, active in enumerate(self.active) if active]
            active_weights = [self.weights[index] for index in active_indices]
            if not any(weight > 0 for weight in active_weights):
                active_weights = [1.0 for _ in active_indices]
            source_index = self.rng.choices(active_indices, weights=active_weights, k=1)[0]
            try:
                batch = next(self.iterators[source_index])
            except StopIteration:
                self.active[source_index] = False
                continue
            self.batches_by_source[source_index] += 1
            # Preserve provenance through the async CUDA prefetcher so the
            # trainer can log a loss series for each interleaved source.
            batch = dict(batch)
            batch["_source_name"] = str(
                self.sources[source_index].get("name", source_index)
            )
            return batch
        raise StopIteration

    @property
    def total_downloaded_shards(self) -> int:
        return sum(loader.total_downloaded_shards for loader in self.loaders)

    def state_dict(self) -> dict[str, object]:
        return {
            "rng_state": self.rng.getstate(),
            "batches_by_source": list(self.batches_by_source),
            "sources": [str(source.get("name", index)) for index, source in enumerate(self.sources)],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if "rng_state" in state and state["rng_state"] is not None:
            self.rng.setstate(state["rng_state"])
        if "batches_by_source" in state and isinstance(state["batches_by_source"], list):
            self.batches_by_source = list(state["batches_by_source"])

    def stop(self) -> None:
        for loader in self.loaders:
            loader.stop()


class JointMultiSourceLoader:
    """Streams batches where every batch contains samples from ALL configured sources simultaneously.

    Rather than alternating sources on different steps, every step draws samples
    from every active source, concatenating them into a single joint batch.
    """

    def __init__(
        self,
        sources: list[dict[str, object]],
        *,
        window_size: int = 2048,
        batch_size: int = 5,
        seed: int = 1337,
        start_batches: list[int] | dict[str, int] | None = None,
        initial_rng_state: object | None = None,
    ) -> None:
        if not sources:
            raise ValueError("joint multi-source loading requires at least one source")
        self.sources = [dict(source) for source in sources]
        self.shard_dir = Path("<joint-multi-source>")
        self.dataset_name = "joint-multi-source"
        self.dataset_config = ",".join(
            str(source.get("name", index)) for index, source in enumerate(self.sources)
        )
        self.window_size = int(window_size)
        self.batch_size = int(batch_size)
        self.num_sources = len(self.sources)
        self.rows_per_source = max(1, self.batch_size // self.num_sources)
        self.loaders: list[ConcurrentStreamingLoader] = []
        self.iterators: list[Iterator[dict[str, torch.Tensor]]] = []
        self.batches_by_source = [0 for _ in self.sources]

        for index, source in enumerate(self.sources):
            source_start_batches = 0
            if isinstance(start_batches, list) and index < len(start_batches):
                source_start_batches = int(start_batches[index])
            elif isinstance(start_batches, dict):
                source_start_batches = int(start_batches.get(str(source.get("name", index)), 0))
            self.batches_by_source[index] = source_start_batches
            source_start_tokens = source_start_batches * self.rows_per_source * self.window_size
            loader = ConcurrentStreamingLoader(
                str(source["dir"]),
                window_size=self.window_size,
                batch_size=self.rows_per_source,
                start_tokens=source_start_tokens,
                dataset_name=str(source.get("dataset_name", "HuggingFaceFW/fineweb-edu")),
                dataset_config=str(source.get("dataset_config", "CC-MAIN-2024-10")),
            )
            self.loaders.append(loader)
            self.iterators.append(iter(loader))

    def __iter__(self) -> "JointMultiSourceLoader":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        sub_batches: list[dict[str, torch.Tensor]] = []
        source_names: list[str] = []

        for index, source in enumerate(self.sources):
            try:
                sub_batch = next(self.iterators[index])
            except StopIteration:
                self.iterators[index] = iter(self.loaders[index])
                try:
                    sub_batch = next(self.iterators[index])
                except StopIteration:
                    continue
            self.batches_by_source[index] += 1
            sub_batches.append(sub_batch)
            source_name = str(source.get("name", index))
            for _ in range(sub_batch["input_ids"].shape[0]):
                source_names.append(source_name)

        if not sub_batches:
            raise StopIteration

        combined = {
            "input_ids": torch.cat([b["input_ids"] for b in sub_batches], dim=0),
            "labels": torch.cat([b["labels"] for b in sub_batches], dim=0),
            "segment_ids": torch.cat([b["segment_ids"] for b in sub_batches], dim=0),
            "reset_mask": torch.cat([b["reset_mask"] for b in sub_batches], dim=0),
            "_source_names": source_names,
            "_source_name": ",".join(dict.fromkeys(source_names)),
            "_has_reset": any(b.get("_has_reset", False) for b in sub_batches),
        }
        return combined

    @property
    def total_downloaded_shards(self) -> int:
        return sum(loader.total_downloaded_shards for loader in self.loaders)

    def state_dict(self) -> dict[str, object]:
        return {
            "batches_by_source": list(self.batches_by_source),
            "sources": [str(source.get("name", index)) for index, source in enumerate(self.sources)],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if "batches_by_source" in state and isinstance(state["batches_by_source"], list):
            self.batches_by_source = list(state["batches_by_source"])

    def stop(self) -> None:
        for loader in self.loaders:
            loader.stop()


def repair_shard_metadata(shard_dir: str | Path) -> dict[str, int]:
    """Repair reset masks in existing shards without changing token order."""
    directory = Path(shard_dir)
    repaired = 0
    unchanged = 0
    rows = 0
    tokens = 0
    for shard_path in sorted(directory.glob("train-*.pt")):
        packed = torch.load(shard_path, map_location="cpu", weights_only=False)
        expected = _reset_mask_from_segments(packed.segment_ids)
        rows += int(packed.input_ids.shape[0])
        tokens += int(packed.input_ids.numel())
        if torch.equal(packed.reset_mask, expected):
            unchanged += 1
            continue
        _save_packed_shard(
            shard_path,
            packed.input_ids.tolist(),
            packed.segment_ids.tolist(),
            packed.document_ids or [[] for _ in range(int(packed.input_ids.shape[0]))],
            provenance=packed.provenance,
        )
        repaired += 1
    return {"shards": repaired + unchanged, "repaired": repaired, "unchanged": unchanged,
            "rows": rows, "tokens": tokens}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Repair streamed shard reset metadata")
    parser.add_argument("shard_dir")
    args = parser.parse_args()
    print(repair_shard_metadata(args.shard_dir), flush=True)
