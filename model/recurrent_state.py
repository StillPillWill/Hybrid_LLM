"""Explicit recurrent state carried through blocks and loop boundaries."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class MemoryState:
    """Causal compressed-attention memory with document identity metadata."""

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    segment_ids: torch.Tensor | None = None
    latents: torch.Tensor | None = None
    # Incremental temporal-compression bucket.  A compressed entry is emitted
    # only when a complete ratio-sized block has arrived, matching the
    # sequence training path instead of treating every decode token as a
    # complete block.
    pending_sum: torch.Tensor | None = None
    pending_count: torch.Tensor | None = None
    pending_segment_ids: torch.Tensor | None = None
    # ``length`` is the active prefix in a decode cache. ``capacity`` is zero
    # for ordinary sequence states and positive for preallocated decode state.
    length: int | None = None
    capacity: int = 0
    static_cache: bool = False

    def detach(self) -> "MemoryState":
        return MemoryState(
            keys=None if self.keys is None else self.keys.detach(),
            values=None if self.values is None else self.values.detach(),
            positions=None if self.positions is None else self.positions.detach(),
            segment_ids=None if self.segment_ids is None else self.segment_ids.detach(),
            latents=None if self.latents is None else self.latents.detach(),
            pending_sum=None if self.pending_sum is None else self.pending_sum.detach(),
            pending_count=None if self.pending_count is None else self.pending_count.detach(),
            pending_segment_ids=(
                None if self.pending_segment_ids is None else self.pending_segment_ids.detach()
            ),
            length=self.length,
            capacity=self.capacity,
            static_cache=self.static_cache,
        )

    def reserve(self, capacity: int) -> "MemoryState":
        """Return a detached, writable decode cache with fixed capacity."""
        if self.keys is None or self.values is None or self.positions is None or self.segment_ids is None:
            return self
        active = self.keys.shape[1] if self.length is None else int(self.length)
        capacity = max(int(capacity), active)
        if self.capacity >= capacity:
            self.length = active
            return self
        keys = self.keys.new_zeros((self.keys.shape[0], capacity, self.keys.shape[2]))
        values = self.values.new_zeros((self.values.shape[0], capacity, self.values.shape[2]))
        positions = self.positions.new_full((self.positions.shape[0], capacity), -1)
        segment_ids = self.segment_ids.new_full((self.segment_ids.shape[0], capacity), -1)
        keys[:, :active] = self.keys[:, :active]
        values[:, :active] = self.values[:, :active]
        positions[:, :active] = self.positions[:, :active]
        segment_ids[:, :active] = self.segment_ids[:, :active]
        latents = None
        if self.latents is not None:
            latents = self.latents.new_zeros((self.latents.shape[0], capacity, self.latents.shape[2]))
            latents[:, :active] = self.latents[:, :active]
        self.keys, self.values = keys, values
        self.positions, self.segment_ids, self.latents = positions, segment_ids, latents
        self.length, self.capacity = active, capacity
        return self


@dataclass
class RecurrentState:
    """All state that may survive one physical loop execution.

    ``streams`` is never collapsed between passes.  Per-block dictionaries make
    state reuse explicit and make checkpoint serialization deterministic.
    """

    streams: torch.Tensor | None = None
    gdn_states: dict[str, torch.Tensor] = field(default_factory=dict)
    csa_memories: dict[str, MemoryState] = field(default_factory=dict)
    hca_memories: dict[str, MemoryState] = field(default_factory=dict)
    # Short causal histories used by the one-token decode path. The regular
    # sequence path does not need these, but retaining them avoids recomputing
    # the local convolution/attention window during generation.
    gdn_conv_histories: dict[str, torch.Tensor] = field(default_factory=dict)
    local_histories: dict[str, torch.Tensor] = field(default_factory=dict)
    local_history_valid: dict[str, torch.Tensor] = field(default_factory=dict)
    router_latent_histories: dict[str, torch.Tensor] = field(default_factory=dict)
    router_history_valid: dict[str, torch.Tensor] = field(default_factory=dict)
    loop_endpoints: list[torch.Tensor] = field(default_factory=list)
    token_positions: torch.Tensor | None = None
    graph_mode: bool = False

    def detach(self) -> "RecurrentState":
        return RecurrentState(
            streams=None if self.streams is None else self.streams.detach(),
            gdn_states={k: v.detach() for k, v in self.gdn_states.items()},
            csa_memories={k: v.detach() for k, v in self.csa_memories.items()},
            hca_memories={k: v.detach() for k, v in self.hca_memories.items()},
            gdn_conv_histories={k: v.detach() for k, v in self.gdn_conv_histories.items()},
            local_histories={k: v.detach() for k, v in self.local_histories.items()},
            local_history_valid={k: v.detach() for k, v in self.local_history_valid.items()},
            router_latent_histories={k: v.detach() for k, v in self.router_latent_histories.items()},
            router_history_valid={k: v.detach() for k, v in self.router_history_valid.items()},
            loop_endpoints=[v.detach() for v in self.loop_endpoints],
            token_positions=None if self.token_positions is None else self.token_positions.detach(),
            graph_mode=self.graph_mode,
        )

    @staticmethod
    def _select_tensor(value: torch.Tensor | None, rows: torch.Tensor) -> torch.Tensor | None:
        return None if value is None else value.index_select(0, rows)

    @classmethod
    def _select_memory(cls, memory: MemoryState, rows: torch.Tensor) -> MemoryState:
        return MemoryState(
            keys=cls._select_tensor(memory.keys, rows),
            values=cls._select_tensor(memory.values, rows),
            positions=cls._select_tensor(memory.positions, rows),
            segment_ids=cls._select_tensor(memory.segment_ids, rows),
            latents=cls._select_tensor(memory.latents, rows),
            pending_sum=cls._select_tensor(memory.pending_sum, rows),
            pending_count=cls._select_tensor(memory.pending_count, rows),
            pending_segment_ids=cls._select_tensor(memory.pending_segment_ids, rows),
            length=memory.length,
            capacity=memory.capacity,
            static_cache=memory.static_cache,
        )

    def select_rows(self, rows: torch.Tensor) -> "RecurrentState":
        """Select independent batch rows for token-wise decode early exit."""
        rows = rows.to(device=(self.streams.device if self.streams is not None else rows.device), dtype=torch.long)
        return RecurrentState(
            streams=self._select_tensor(self.streams, rows),
            gdn_states={k: self._select_tensor(v, rows) for k, v in self.gdn_states.items()},
            csa_memories={k: self._select_memory(v, rows) for k, v in self.csa_memories.items()},
            hca_memories={k: self._select_memory(v, rows) for k, v in self.hca_memories.items()},
            gdn_conv_histories={k: self._select_tensor(v, rows) for k, v in self.gdn_conv_histories.items()},
            local_histories={k: self._select_tensor(v, rows) for k, v in self.local_histories.items()},
            local_history_valid={k: self._select_tensor(v, rows) for k, v in self.local_history_valid.items()},
            router_latent_histories={k: self._select_tensor(v, rows) for k, v in self.router_latent_histories.items()},
            router_history_valid={k: self._select_tensor(v, rows) for k, v in self.router_history_valid.items()},
            loop_endpoints=[self._select_tensor(v, rows) for v in self.loop_endpoints],
            token_positions=self._select_tensor(self.token_positions, rows),
            graph_mode=self.graph_mode,
        )

    @staticmethod
    def _scatter_tensor(
        target: torch.Tensor | None, source: torch.Tensor | None, rows: torch.Tensor,
        batch_size: int, *, fill_value: float = 0.0,
    ) -> torch.Tensor | None:
        if source is None:
            return target
        if target is None or tuple(target.shape[1:]) != tuple(source.shape[1:]):
            shape = (batch_size, *source.shape[1:])
            resized = source.new_full(shape, fill_value)
            if target is not None and target.ndim == source.ndim:
                common = tuple(min(old, new) for old, new in zip(target.shape[1:], source.shape[1:]))
                if common:
                    target_slices = (slice(None),) + tuple(slice(0, size) for size in common)
                    resized[target_slices] = target[target_slices]
            target = resized
        # Some state tensors (notably token_positions after prefill) can be
        # expanded views with overlapping storage. Clone before row updates.
        target = target.clone()
        target.index_copy_(0, rows, source)
        return target

    @classmethod
    def _scatter_memory(
        cls, target: MemoryState | None, source: MemoryState, rows: torch.Tensor, batch_size: int,
    ) -> MemoryState:
        if target is None:
            target = MemoryState(length=source.length, capacity=source.capacity, static_cache=source.static_cache)
        for name in (
            "keys", "values", "positions", "segment_ids", "latents",
            "pending_sum", "pending_count", "pending_segment_ids",
        ):
            source_value = getattr(source, name)
            fill_value = -1 if name in {"positions", "segment_ids", "pending_segment_ids"} else 0.0
            setattr(target, name, cls._scatter_tensor(
                getattr(target, name), source_value, rows, batch_size, fill_value=fill_value,
            ))
        target.length = source.length
        target.capacity = source.capacity
        target.static_cache = source.static_cache
        return target

    def scatter_rows(self, rows: torch.Tensor, source: "RecurrentState") -> None:
        """Merge selected rows from a compact decode state back into this state."""
        rows = rows.to(device=(self.streams.device if self.streams is not None else rows.device), dtype=torch.long)
        batch_size = int(self.streams.shape[0]) if self.streams is not None else int(self.token_positions.shape[0])
        self.streams = self._scatter_tensor(self.streams, source.streams, rows, batch_size)
        for name in ("gdn_states", "gdn_conv_histories", "local_histories", "local_history_valid",
                     "router_latent_histories", "router_history_valid"):
            target = getattr(self, name)
            for key, value in getattr(source, name).items():
                target[key] = self._scatter_tensor(target.get(key), value, rows, batch_size)
        for name in ("csa_memories", "hca_memories"):
            target = getattr(self, name)
            for key, value in getattr(source, name).items():
                target[key] = self._scatter_memory(target.get(key), value, rows, batch_size)
        if source.token_positions is not None:
            self.token_positions = self._scatter_tensor(self.token_positions, source.token_positions, rows, batch_size)

    def reset_rows(self, reset_rows: torch.Tensor) -> None:
        """Reset packed-document rows before processing a new packed sample."""
        reset_rows = reset_rows.to(dtype=torch.bool)
        if self.streams is not None:
            self.streams[reset_rows] = 0
        for state in self.gdn_states.values():
            state[reset_rows] = 0
        for histories in (self.gdn_conv_histories, self.local_histories, self.router_latent_histories):
            for value in histories.values():
                value[reset_rows] = 0
        for valid_histories in (self.local_history_valid, self.router_history_valid):
            for value in valid_histories.values():
                value[reset_rows] = False
        for memory in (*self.csa_memories.values(), *self.hca_memories.values()):
            if memory.pending_sum is not None:
                memory.pending_sum[reset_rows] = 0
            if memory.pending_count is not None:
                memory.pending_count[reset_rows] = 0
            if memory.pending_segment_ids is not None:
                memory.pending_segment_ids[reset_rows] = -1
        for memory in (*self.csa_memories.values(), *self.hca_memories.values()):
            if memory.segment_ids is None:
                continue
            if bool(reset_rows.all()):
                memory.keys = memory.values = memory.positions = memory.segment_ids = memory.latents = None
                memory.pending_sum = memory.pending_count = memory.pending_segment_ids = None
                memory.length = memory.capacity = 0
                continue
            for name in ("keys", "values", "positions", "segment_ids", "latents"):
                value = getattr(memory, name)
                if value is not None and value.shape[0] == reset_rows.shape[0]:
                    value[reset_rows] = -1 if name in {"positions", "segment_ids"} else 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "streams": self.streams,
            "gdn_states": self.gdn_states,
            "csa_memories": {k: vars(v) for k, v in self.csa_memories.items()},
            "hca_memories": {k: vars(v) for k, v in self.hca_memories.items()},
            "gdn_conv_histories": self.gdn_conv_histories,
            "local_histories": self.local_histories,
            "local_history_valid": self.local_history_valid,
            "router_latent_histories": self.router_latent_histories,
            "router_history_valid": self.router_history_valid,
            "loop_endpoints": self.loop_endpoints,
            "token_positions": self.token_positions,
            "graph_mode": self.graph_mode,
        }

    @classmethod
    def from_state_dict(cls, value: dict[str, Any]) -> "RecurrentState":
        return cls(
            streams=value.get("streams"),
            gdn_states=dict(value.get("gdn_states", {})),
            csa_memories={k: MemoryState(**v) for k, v in value.get("csa_memories", {}).items()},
            hca_memories={k: MemoryState(**v) for k, v in value.get("hca_memories", {}).items()},
            gdn_conv_histories=dict(value.get("gdn_conv_histories", {})),
            local_histories=dict(value.get("local_histories", {})),
            local_history_valid=dict(value.get("local_history_valid", {})),
            router_latent_histories=dict(value.get("router_latent_histories", {})),
            router_history_valid=dict(value.get("router_history_valid", {})),
            loop_endpoints=list(value.get("loop_endpoints", [])),
            token_positions=value.get("token_positions"),
            graph_mode=bool(value.get("graph_mode", False)),
        )
