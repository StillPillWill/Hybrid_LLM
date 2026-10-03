"""The V0 2B looped hybrid-MoE language model."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .config import V6Config
from .layers.csa import CompressedSelectiveAttention
from .layers.dense_ffn import DenseSwiGLU
from .layers.gated_deltanet import GatedDeltaNet
from .layers.hca import HierarchicalCompressedAttention
from .layers.mhc import MultiStreamHyperConnection
from .layers.mla_memory import MLALatentMemory
from .layers.moe import SparseMoE
from .layers.norms import RMSNorm
from .loop_memory import LoopMemoryHook
from .recurrent_state import MemoryState, RecurrentState


_ROUTER_HEALTH_KEYS = (
    "route_entropy", "router_balance_loss", "moe_aux_loss",
    "active_experts", "dead_experts", "expert_load_entropy",
    "max_expert_fraction", "mean_selected_probability",
)

def _checkpointed_vocab_ce(
    hidden_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    embed_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute one vocabulary CE chunk without retaining its logits graph."""
    projected = F.linear(hidden_chunk, embed_weight).float()
    return F.cross_entropy(projected, target_chunk, reduction="sum")


def _chunk_ce_eager(
    hidden_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    embed_weight: torch.Tensor,
) -> torch.Tensor:
    logits = F.linear(hidden_chunk, embed_weight).float()
    return F.cross_entropy(logits, target_chunk, reduction="sum")


_chunk_ce_compiled = None


def _chunk_ce(hidden_chunk: torch.Tensor, target_chunk: torch.Tensor,
              embed_weight: torch.Tensor) -> torch.Tensor:
    """Use Inductor only for CUDA tensors; keep CPU inspection tools eager."""
    global _chunk_ce_compiled
    if hidden_chunk.is_cuda and os.getenv("LLMY_ENABLE_COMPILED_CE", "0") == "1":
        if _chunk_ce_compiled is None and hasattr(torch, "compile"):
            _chunk_ce_compiled = torch.compile(_chunk_ce_eager, dynamic=True)
        if _chunk_ce_compiled is not None:
            return _chunk_ce_compiled(hidden_chunk, target_chunk, embed_weight)
    return _chunk_ce_eager(hidden_chunk, target_chunk, embed_weight)


@dataclass
class LoopedHybridOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    state: RecurrentState
    auxiliary_loss: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


class PhysicalBlock(nn.Module):
    """One physical block; the same instance is called on every loop pass."""

    def __init__(self, cfg: V6Config, kind: str, *, block_name: str, with_moe: bool) -> None:
        super().__init__()
        self.kind = kind
        self.block_name = block_name
        self.with_moe = bool(with_moe)
        self.mhc = MultiStreamHyperConnection(
            cfg.d_model, cfg.mhc.streams, cfg.mhc.sinkhorn_iterations, cfg.mhc.init_gating_factor
        ) if cfg.mhc.enabled else None
        self.mixer_norm = RMSNorm(cfg.d_model)
        self.ffn_norm = RMSNorm(cfg.d_model)
        if kind == "gdn":
            self.mixer = GatedDeltaNet(
                cfg.d_model, cfg.gdn.qk_heads, cfg.gdn.value_heads,
                cfg.gdn.qk_head_dim, cfg.gdn.value_head_dim, cfg.gdn.conv_kernel,
                torch.float32 if cfg.gdn.sensitive_state_dtype.lower() in {"float32", "fp32"} else torch.float16,
            )
        elif kind == "csa":
            self.mixer = CompressedSelectiveAttention(
                cfg.d_model, compression=cfg.csa.temporal_compression,
                local_window=cfg.csa.local_window, index_heads=cfg.csa.index_heads,
                index_head_dim=cfg.csa.index_head_dim, top_k=cfg.csa.index_top_k,
            )
        elif kind == "hca":
            self.mixer = HierarchicalCompressedAttention(
                cfg.d_model, compression=cfg.hca.temporal_compression,
                local_window=cfg.hca.local_window, dense_global_attention=cfg.hca.dense_global_attention,
            )
        else:
            raise ValueError(f"unknown V0 mixer kind: {kind}")
        self.moe = SparseMoE(
            cfg.d_model, routed_experts=cfg.moe.routed_experts,
            shared_experts=cfg.moe.shared_experts, top_k=cfg.moe.routed_top_k,
            expert_hidden=cfg.moe.expert_hidden, latent_dim=cfg.mla.latent_dim,
            context_aware=cfg.router.context_aware, query_heads=cfg.router.query_heads,
            query_head_dim=cfg.router.query_head_dim, local_context=cfg.router.local_context,
            aux_balance_coef=cfg.moe.aux_balance_coef,
        ) if with_moe else None
        self.ffn = DenseSwiGLU(cfg.d_model, cfg.dense_ffn.hidden) if not with_moe else None
        self.hca_latent = MLALatentMemory(cfg.d_model, cfg.mla.latent_dim) if kind == "hca" else None

    def _read(self, streams: torch.Tensor):
        if self.mhc is None:
            return streams.mean(dim=-2), None
        mappings = self.mhc.compute_mappings(streams)
        return self.mhc.read(streams, mappings), mappings

    def _write(self, streams: torch.Tensor, update: torch.Tensor, mappings=None) -> torch.Tensor:
        return self.mhc.write(streams, update, mappings) if self.mhc is not None else streams + update.unsqueeze(-2)

    @staticmethod
    def _latest_hca_latents(
        state: RecurrentState,
        state_namespace: str | None,
    ) -> torch.Tensor | None:
        prefix = None if state_namespace is None else f"{state_namespace}."
        latest = None
        for key, memory in state.hca_memories.items():
            if prefix is None:
                if key.startswith("loop"):
                    continue
            elif not key.startswith(prefix):
                continue
            if memory.latents is not None:
                latest = memory.latents
        return latest

    def _moe_checkpointed(self, hidden: torch.Tensor, hca_latents: torch.Tensor | None,
                          collect_router_diagnostics: bool = False):
        """Recompute the pure MoE path in backward instead of retaining dispatch buffers.

        Training-time MoE dispatch stacks every expert's weights and creates large
        grouped-routing intermediates.  Those tensors are safe to recompute because
        the MoE path has no recurrent-state side effects; retaining them across both
        recurrent loops was the real activation peak, not the small mHC parameter set.
        """
        output, aux, metrics = self.moe(
            hidden, hca_global_latents=hca_latents, return_stats=False,
            return_health=collect_router_diagnostics,
        )
        if collect_router_diagnostics:
            self.moe._last_router_health = metrics
        return output, aux

    def forward(self, streams: torch.Tensor, state: RecurrentState, *, segment_ids: torch.Tensor | None,
                reset_mask: torch.Tensor | None, diagnostics: dict[str, Any],
                collect_diagnostics: bool = True,
                collect_router_diagnostics: bool = False,
                position: torch.Tensor | int | None = None,
                state_namespace: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        state_key = (
            f"{state_namespace}.{self.block_name}"
            if state_namespace else self.block_name
        )
        hidden, mixer_mappings = self._read(streams)
        normed = self.mixer_norm(hidden)
        if self.kind == "gdn":
            prior = state.gdn_states.get(state_key)
            # The fused GDN scan performs its own activation checkpointing.
            # Wrapping the entire mixer again makes stateful recomputation
            # invalid when truncated BPTT changes the grad boundary.
            mixed, next_state = self.mixer(normed, prior, reset_mask)
            state.gdn_states[state_key] = next_state.detach()
            history_size = self.mixer.conv.kernel_size[0] - 1
            state.gdn_conv_histories[state_key] = normed[:, -history_size:].detach()
        else:
            prior = state.csa_memories.get(state_key) if self.kind == "csa" else state.hca_memories.get(state_key)
            # CSA's remote-value tiles are checkpointed internally. Avoid an
            # outer stateful checkpoint here; it can replay against mutated
            # compressed-memory metadata after a loop is detached.
            mixed, compressed, metrics = self.mixer(
                normed, segment_ids=segment_ids, memory=prior,
                return_stats=collect_diagnostics,
            )
            state_value = MemoryState(
                compressed.keys.detach(), compressed.values.detach(), compressed.positions,
                compressed.segment_ids, pending_sum=compressed.pending_sum,
                pending_count=compressed.pending_count,
                pending_segment_ids=compressed.pending_segment_ids,
            )
            if self.kind == "csa":
                state.csa_memories[state_key] = state_value
            else:
                state.hca_memories[state_key] = state_value
                state.hca_memories[state_key].latents = self.hca_latent.encode(
                    compressed.values
                ).detach()
            for key, value in metrics.items():
                diagnostics[key] = value
            history_size = self.mixer.local_window - 1
            state.local_histories[state_key] = normed[:, -history_size:].detach()
            state.local_history_valid[state_key] = torch.ones(
                state.local_histories[state_key].shape[:2], device=normed.device, dtype=torch.bool
            )
        streams = self._write(streams, mixed, mixer_mappings)

        hidden, ffn_mappings = self._read(streams)
        if self.moe is not None:
            hca_latents = None
            # The HCA block's current global summaries condition its own MoE;
            # later blocks can reuse the explicit memory from the state object.
            hca_latents = self._latest_hca_latents(state, state_namespace)
            moe_input = self.ffn_norm(hidden)
            if self.training and torch.is_grad_enabled() and not collect_diagnostics:
                # Avoid retaining the per-expert stacked weights and grouped dispatch
                # activations for every recurrent pass.  The backward recomputation is
                # substantially cheaper than reducing the whole training microbatch.
                ffn, aux = activation_checkpoint(
                    self._moe_checkpointed, moe_input, hca_latents,
                    # Routed dispatch is stateful (it records router stats and
                    # uses shape-dependent grouped buffers).  The non-reentrant
                    # checkpoint engine replays that path with different
                    # metadata on this model and fails during backward.  Keep
                    # the memory-saving boundary, but use the stable reentrant
                    # engine for this particular function.
                    collect_router_diagnostics, use_reentrant=True,
                )
                metrics = self.moe._last_router_health if collect_router_diagnostics else {}
            else:
                ffn, aux, metrics = self.moe(
                    moe_input, hca_global_latents=hca_latents,
                    return_stats=collect_diagnostics,
                    return_health=collect_router_diagnostics,
                )
            if not self.training and not collect_diagnostics:
                router_input = self.ffn_norm(hidden)
                router = self.moe.router
                router_local = router.norm(router_input)
                router_latents = router.latent_memory.encode(router_local)
                state.router_latent_histories[state_key] = router_latents[
                    :, -router.local_context:
                ].detach()
                state.router_history_valid[state_key] = torch.ones(
                    state.router_latent_histories[state_key].shape[:2],
                    device=hidden.device, dtype=torch.bool,
                )
            if collect_router_diagnostics:
                health = {key: metrics[key] for key in _ROUTER_HEALTH_KEYS if key in metrics}
                if health:
                    diagnostics.setdefault("_router_health_samples", []).append(health)
            if collect_diagnostics:
                diagnostics.update(metrics)
        else:
            ffn = self.ffn(self.ffn_norm(hidden))
            aux = hidden.new_zeros(())
        streams = self._write(streams, ffn, ffn_mappings)
        return streams, aux

    @torch.inference_mode()
    def forward_step(self, streams: torch.Tensor, state: RecurrentState, *, position: torch.Tensor | int,
                     segment_ids: torch.Tensor | None, reset_mask: torch.Tensor | None,
                     state_namespace: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply this physical block to one token using retained state."""
        state_key = (
            f"{state_namespace}.{self.block_name}"
            if state_namespace else self.block_name
        )
        hidden, mixer_mappings = self._read(streams)
        normed = self.mixer_norm(hidden)
        if self.kind == "gdn":
            prior = state.gdn_states.get(state_key)
            history = state.gdn_conv_histories.get(state_key)
            mixed, next_state = self.mixer(normed, prior, reset_mask, conv_history=history)
            state.gdn_states[state_key] = next_state.detach()
            history_size = self.mixer.conv.kernel_size[0] - 1
            if history is not None and history.shape[1] == history_size:
                history[:, :-1].copy_(history[:, 1:].clone())
                history[:, -1:].copy_(normed)
                state.gdn_conv_histories[state_key] = history
            else:
                combined = normed if history is None else torch.cat((history, normed), dim=1)
                state.gdn_conv_histories[state_key] = combined[:, -history_size:].detach()
        else:
            prior = state.csa_memories.get(state_key) if self.kind == "csa" else state.hca_memories.get(state_key)
            if prior is not None and prior.keys is not None:
                # Generation is bounded by the training context window. Keep
                # the decode cache writable so appending a token does not
                # copy the complete history on every step.
                active_length = prior.keys.shape[1] if prior.length is None else int(prior.length)
                prior.reserve(max(256, active_length + 1))
            history = state.local_histories.get(state_key)
            history_valid = state.local_history_valid.get(state_key)
            mixed, compressed, _ = self.mixer.forward_step(
                normed, segment_ids=segment_ids, memory=prior,
                local_history=history, local_history_valid=history_valid, position=position,
            )
            if prior is not None and prior.capacity > 0:
                state_value = prior
            else:
                state_value = MemoryState(
                    compressed.keys.detach(), compressed.values.detach(), compressed.positions,
                    compressed.segment_ids, pending_sum=compressed.pending_sum,
                    pending_count=compressed.pending_count,
                    pending_segment_ids=compressed.pending_segment_ids,
                    length=compressed.keys.shape[1],
                ).reserve(256)
            if self.kind == "csa":
                state.csa_memories[state_key] = state_value
            else:
                state.hca_memories[state_key] = state_value
                if state_value.capacity > 0:
                    index = torch.as_tensor(position, device=normed.device, dtype=torch.long).view(normed.shape[0], 1)
                    index = index.remainder(state_value.capacity)
                    current_value = state_value.values.gather(
                        1, index.unsqueeze(-1).expand(-1, -1, state_value.values.shape[-1])
                    )
                    current_latent = self.hca_latent.encode(current_value).detach()
                    if state_value.latents is None:
                        state_value.latents = current_latent.new_zeros(
                            state_value.keys.shape[0], state_value.capacity, current_latent.shape[-1]
                        )
                    state_value.latents.scatter_(
                        1, index.unsqueeze(-1).expand(-1, -1, current_latent.shape[-1]), current_latent
                    )
                else:
                    current_latent = self.hca_latent.encode(compressed.values[:, -1:]).detach()
                    state_value.latents = current_latent
            history_size = self.mixer.local_window - 1
            if state.graph_mode and history is not None and history.shape[1] == history_size:
                history[:, :-1].copy_(history[:, 1:].clone())
                history[:, -1:].copy_(normed)
                valid = state.local_history_valid[state_key]
                valid[:, :-1].copy_(valid[:, 1:].clone())
                valid[:, -1] = True
            else:
                combined = normed if history is None else torch.cat((history, normed), dim=1)
                state.local_histories[state_key] = combined[:, -history_size:].detach()
                state.local_history_valid[state_key] = torch.ones(
                    state.local_histories[state_key].shape[:2], device=normed.device, dtype=torch.bool
                )

        streams = self._write(streams, mixed, mixer_mappings)
        hidden, ffn_mappings = self._read(streams)
        if self.moe is not None:
            hca_latents = None
            hca_latents = self._latest_hca_latents(state, state_namespace)
            ffn, aux, _, router_latents = self.moe.forward_step(
                self.ffn_norm(hidden),
                local_history=state.router_latent_histories.get(state_key),
                local_history_valid=state.router_history_valid.get(state_key),
                hca_global_latents=hca_latents,
            )
            router_history = state.router_latent_histories.get(state_key)
            if state.graph_mode and router_history is not None and router_history.shape[1] == self.moe.router.local_context:
                router_history[:, :-1].copy_(router_history[:, 1:].clone())
                router_history[:, -1:].copy_(router_latents[:, -1:])
                valid = state.router_history_valid[state_key]
                valid[:, :-1].copy_(valid[:, 1:].clone())
                valid[:, -1] = True
            else:
                state.router_latent_histories[state_key] = router_latents[:, -self.moe.router.local_context:].detach()
                state.router_history_valid[state_key] = torch.ones(
                    state.router_latent_histories[state_key].shape[:2], device=hidden.device, dtype=torch.bool
                )
        else:
            ffn = self.ffn(self.ffn_norm(hidden))
            aux = hidden.new_zeros(())
        streams = self._write(streams, ffn, ffn_mappings)
        return streams, aux


class LoopedHybridMoEForCausalLM(nn.Module):
    """Ten physical blocks with a shared six-block recurrent middle stack."""

    def __init__(self, config: V6Config | None = None) -> None:
        super().__init__()
        self.config = config or V6Config()
        cfg = self.config
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        # GPT-style init: std = 1/sqrt(d_model) prevents NaN logits at large vocab
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=cfg.d_model ** -0.5)
        self.stream_seed = nn.Parameter(torch.zeros(cfg.mhc.streams, cfg.d_model))
        nn.init.normal_(self.stream_seed, mean=0.0, std=1e-3)
        self.stem = nn.ModuleList([
            PhysicalBlock(cfg, "gdn", block_name=f"stem.{i}", with_moe=False)
            for i in range(cfg.stem_blocks)
        ])
        self.recurrent_core = nn.ModuleList([
            PhysicalBlock(cfg, layout.split("_")[0], block_name=f"core.{i}", with_moe=True)
            for i, layout in enumerate(cfg.recurrent_layout)
        ])
        self.recurrent_core_extra = nn.ModuleList()
        if not cfg.mhc.shared_across_loops:
            for loop_index in range(1, max(1, cfg.num_loops)):
                loop_blocks = nn.ModuleList()
                for block_index, layout in enumerate(cfg.recurrent_layout):
                    block = copy.deepcopy(self.recurrent_core[block_index])
                    block.block_name = f"core.loop{loop_index}.{block_index}"
                    if cfg.moe.expert_weights_shared_across_loops and block.moe is not None:
                        block.moe.experts = self.recurrent_core[block_index].moe.experts
                        block.moe.shared = self.recurrent_core[block_index].moe.shared
                    loop_blocks.append(block)
                self.recurrent_core_extra.append(loop_blocks)
        self.tail = nn.ModuleList([
            PhysicalBlock(cfg, "gdn", block_name=f"tail.{i}", with_moe=False)
            for i in range(cfg.tail_blocks)
        ])
        self.mhc_collapse = nn.Linear(cfg.mhc.streams * cfg.d_model, cfg.d_model, bias=False)
        with torch.no_grad():
            self.mhc_collapse.weight.zero_()
            for stream in range(cfg.mhc.streams):
                start = stream * cfg.d_model
                self.mhc_collapse.weight[:, start:start + cfg.d_model].copy_(
                    torch.eye(cfg.d_model) / cfg.mhc.streams
                )
        self.final_norm = RMSNorm(cfg.d_model)
        self.loop_memory = LoopMemoryHook(
            cfg.d_model,
            cfg.loop_attnres.latent_dim,
            enabled=cfg.loop_attnres.enabled,
            heads=cfg.loop_attnres.heads,
            head_dim=cfg.loop_attnres.head_dim,
            residual_gate_init=cfg.loop_attnres.residual_gate_init,
        )
        # Reads the representation emitted by the final recurrent block and
        # predicts whether each token should enter another recursive pass.
        self.recurrent_halt_head = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.recurrent_halt_head.weight)
        initial_prob = cfg.adaptive_loops.initial_continue_prob
        with torch.no_grad():
            self.recurrent_halt_head.bias.fill_(
                torch.logit(torch.tensor(initial_prob)).item()
            )

    def _streams_from_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        return hidden.unsqueeze(-2) + self.stream_seed.to(hidden.dtype).view(1, 1, -1, self.config.d_model)

    def _run_stack(self, blocks: nn.ModuleList, streams: torch.Tensor, state: RecurrentState,
                   *, segment_ids: torch.Tensor | None, reset_mask: torch.Tensor | None,
                   diagnostics: dict[str, Any], collect_diagnostics: bool = True,
                   collect_router_diagnostics: bool = False,
                   position: torch.Tensor | int | None = None,
                   state_namespace: str | None = None,
                   active_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        auxiliary = streams.new_zeros((), dtype=torch.float32)
        for block in blocks:
            previous = streams
            streams, block_aux = block(
                streams, state, segment_ids=segment_ids, reset_mask=reset_mask,
                diagnostics=diagnostics, collect_diagnostics=collect_diagnostics,
                collect_router_diagnostics=collect_router_diagnostics, position=position,
                state_namespace=state_namespace,
            )
            # Halted tokens are frozen at every physical block boundary.  This
            # is the sequence-path equivalent of decode_step's token-wise
            # early-exit behavior; masking only after the complete recurrent
            # stack would allow later blocks to update halted tokens.
            if active_mask is not None:
                active_view = active_mask.to(dtype=streams.dtype).unsqueeze(-1).unsqueeze(-1)
                streams = streams * active_view + previous * (1.0 - active_view)
            auxiliary = auxiliary + block_aux.float()
        return streams, auxiliary

    def _recurrent_stack(self, loop_index: int) -> nn.ModuleList:
        if loop_index == 0 or not self.recurrent_core_extra:
            return self.recurrent_core
        return self.recurrent_core_extra[min(loop_index - 1, len(self.recurrent_core_extra) - 1)]

    def _sample_loop_budgets(self, batch: int, length: int, loops: int,
                             device: torch.device) -> torch.Tensor:
        """Sample a loop budget per token for training-time adaptive compute.

        The recurrent stack is still evaluated as a dense tensor for kernel and
        state compatibility; inactive tokens are restored after each pass. This
        preserves token-wise behavior and the loop attention residual path while
        avoiding unsafe per-token state mutation.
        """
        cfg = self.config.adaptive_loops
        if not self.training or not cfg.enabled:
            return torch.full((batch, length), loops, device=device, dtype=torch.long)
        minimum = max(1, min(int(cfg.min_loops), loops))
        maximum = max(minimum, min(int(cfg.max_loops), loops))
        if minimum == maximum:
            return torch.full((batch, length), minimum, device=device, dtype=torch.long)
        return torch.randint(minimum, maximum + 1, (batch, length), device=device)

    def _halt_decision(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a hard forward decision with a soft backward gradient."""
        cfg = self.config.adaptive_loops
        hidden = self.mhc_collapse(streams.reshape(streams.shape[0], streams.shape[1], -1))
        logits = self.recurrent_halt_head(hidden).squeeze(-1)
        probabilities = torch.clamp(
            torch.sigmoid(logits.float() / float(cfg.halt_temperature)),
            0.0, 1.0
        ).nan_to_num(0.5)
        if self.training:
            hard = torch.bernoulli(probabilities).to(streams.dtype)
        else:
            hard = (probabilities >= float(cfg.halt_threshold)).to(streams.dtype)
        straight_through = hard.detach() - probabilities.detach() + probabilities
        return straight_through, probabilities

    def _run_step_stack(self, blocks: nn.ModuleList, streams: torch.Tensor, state: RecurrentState,
                        *, position: torch.Tensor | int, segment_ids: torch.Tensor | None,
                        reset_mask: torch.Tensor | None,
                        state_namespace: str | None = None,
                        active_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        auxiliary = streams.new_zeros((), dtype=torch.float32)
        for block in blocks:
            previous = streams
            streams, block_aux = block.forward_step(
                streams, state, position=position, segment_ids=segment_ids,
                reset_mask=reset_mask,
                state_namespace=state_namespace,
            )
            if active_mask is not None:
                active_view = active_mask.to(dtype=streams.dtype).view(-1, 1, 1, 1)
                streams = streams * active_view + previous * (1.0 - active_view)
            auxiliary = auxiliary + block_aux.float()
        return streams, auxiliary

    def forward(self, input_ids: torch.Tensor, *, labels: torch.Tensor | None = None,
                state: RecurrentState | None = None, num_loops: int | None = None,
                segment_ids: torch.Tensor | None = None, reset_mask: torch.Tensor | None = None,
                record_loop_memory: bool | None = None,
                collect_diagnostics: bool = True,
                collect_router_diagnostics: bool = False,
                last_logits_only: bool = False,
                batch_has_reset: bool | None = None) -> LoopedHybridOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        loops = int(self.config.num_loops if num_loops is None else num_loops)
        if loops < 1:
            raise ValueError("num_loops must be positive")
        batch, length = input_ids.shape
        state = state or RecurrentState()
        reset_rows = None
        has_reset = False
        if reset_mask is not None:
            if reset_mask.shape != input_ids.shape:
                raise ValueError("reset_mask must have the same shape as input_ids")
            reset_rows = reset_mask[:, 0]
            has_reset = bool(reset_rows.any()) if batch_has_reset is None else bool(batch_has_reset)
            if state.streams is not None and has_reset:
                state.reset_rows(reset_rows)
        if segment_ids is not None and segment_ids.shape != input_ids.shape:
            raise ValueError("segment_ids must have the same shape as input_ids")
        if state.streams is None:
            streams = self._streams_from_tokens(input_ids)
        else:
            streams = state.streams.clone()
            if tuple(streams.shape) != (batch, length, self.config.mhc.streams, self.config.d_model):
                raise ValueError("recurrent state streams do not match input shape")
            if reset_rows is not None and has_reset:
                streams[reset_rows] = self._streams_from_tokens(input_ids[reset_rows])
        adaptive_cfg = self.config.adaptive_loops
        adaptive_on = bool(adaptive_cfg.enabled)
        learned_halting = bool(adaptive_on and adaptive_cfg.learned_halting)
        if learned_halting:
            # max_loops is an actual inference/training budget, not merely a
            # reporting hint.  This also makes sampler --max-loops overrides
            # match the recurrent depth used by the prefill path.
            loops = min(loops, max(1, int(adaptive_cfg.max_loops)))
        if not adaptive_on:
            loop_budgets = None
            active = None
            effective_loop_count = None
            halt_penalty_loss = streams.new_zeros(())
            halt_probability_sum = streams.new_zeros(())
        else:
            loop_budgets = (
                None if learned_halting
                else self._sample_loop_budgets(batch, length, loops, input_ids.device)
            )
            active = streams.new_ones((batch, length), dtype=torch.float32)
            effective_loop_count = streams.new_zeros((batch, length), dtype=torch.float32)
            halt_penalty_loss = streams.new_zeros(())
            halt_probability_sum = streams.new_zeros(())
        diagnostics: dict[str, Any] = {
            "num_loops": loops,
            "loop_auxiliary_loss": [],
        }
        if loop_budgets is not None:
            diagnostics["adaptive_loop_budget_mean"] = float(loop_budgets.float().mean().detach())
        if learned_halting:
            diagnostics["learned_halt_penalty"] = float(adaptive_cfg.halt_penalty)
        if collect_diagnostics and self.training and adaptive_cfg.enabled and not learned_halting:
            for budget in range(1, loops + 1):
                diagnostics[f"adaptive_loop_budget_{budget}_fraction"] = float(
                    (loop_budgets == budget).float().mean().detach()
                )
        streams, auxiliary = self._run_stack(
            self.stem, streams, state, segment_ids=segment_ids,
            reset_mask=reset_mask, diagnostics=diagnostics,
            collect_diagnostics=collect_diagnostics,
            collect_router_diagnostics=collect_router_diagnostics,
            position=torch.arange(length, device=input_ids.device, dtype=torch.long),
        )
        position = torch.arange(length, device=input_ids.device, dtype=torch.long)
        loop_endpoints: list[torch.Tensor] = []
        loop_boundary_streams = streams
        halt_decision_count = 0
        configured_backprop_loops = int(adaptive_cfg.backprop_loops)
        backprop_loops = loops if configured_backprop_loops <= 0 else min(
            configured_backprop_loops, loops
        )
        backprop_start = loops - backprop_loops
        diagnostics["backprop_loops"] = backprop_loops
        diagnostics["backprop_start_loop"] = backprop_start
        for loop_index in range(loops):
            # Once every token has halted, no later recurrent pass contributes
            # to the output.  This is a real early exit, not post-hoc masking.
            if active is not None and not bool(active.any()):
                diagnostics["executed_loop_count"] = loop_index
                break
            loop_requires_grad = loop_index >= backprop_start
            if loop_index == backprop_start and backprop_start > 0:
                # Truncated BPTT: preserve the forward value, but sever the
                # graph accumulated through earlier recurrent passes.
                streams = streams.detach()
                loop_endpoints = [endpoint.detach() for endpoint in loop_endpoints]
            # Re-run every MoE router on every recurrent pass.  Routing indices
            # are intentionally not carried in RecurrentState, so changed
            # hidden states may select different routed experts per loop.
            with torch.set_grad_enabled(loop_requires_grad):
                if loop_index > 0 and not self.config.mhc.preserve_streams_across_loop_boundary:
                    streams = loop_boundary_streams.clone()
                streams = self.loop_memory.inject(streams, loop_endpoints)
                if loop_budgets is not None:
                    active = (loop_budgets >= (loop_index + 1)).to(torch.float32)
                candidate, loop_aux = self._run_stack(
                    self._recurrent_stack(loop_index), streams, state, segment_ids=segment_ids,
                    reset_mask=reset_mask, diagnostics=diagnostics,
                    collect_diagnostics=collect_diagnostics,
                    collect_router_diagnostics=collect_router_diagnostics,
                    position=position,
                    state_namespace=f"loop{loop_index}",
                    active_mask=active,
                )
                # ``_run_stack`` freezes inactive tokens at every physical
                # block.  Keep this boundary mask as a final guard for the
                # loop-attention residual path.
                if active is not None:
                    active_view = active.to(dtype=candidate.dtype).unsqueeze(-1).unsqueeze(-1)
                    streams = candidate * active_view + streams * (1.0 - active_view)
                else:
                    streams = candidate
                auxiliary = auxiliary + loop_aux
                if effective_loop_count is not None:
                    effective_loop_count = effective_loop_count + active
                if collect_diagnostics:
                    diagnostics["loop_auxiliary_loss"].append(float(loop_aux.detach()))
                should_record = record_loop_memory if record_loop_memory is not None else self.config.loop_attnres.enabled
                if should_record:
                    endpoint = self.loop_memory.record(
                        streams, state, force=record_loop_memory is True
                    )
                    if endpoint is not None:
                        loop_endpoints.append(endpoint)
                if loop_index < loops - 1:
                    if not adaptive_on:
                        continue
                    if loop_index + 1 < adaptive_cfg.min_loops:
                        continue_decision = active
                        continue_probability = active
                    elif learned_halting:
                        continue_decision, continue_probability = self._halt_decision(streams)
                    else:
                        continue_decision = (loop_budgets >= (loop_index + 2)).to(torch.float32)
                        continue_probability = continue_decision
                    if learned_halting or loop_budgets is not None:
                        halt_probability_sum = halt_probability_sum + (
                            active * continue_probability
                        ).mean()
                        halt_penalty_loss = halt_penalty_loss + (
                            float(adaptive_cfg.halt_penalty) * (active * continue_probability).mean()
                        )
                        halt_decision_count += 1
                    active = active * continue_decision
        diagnostics.setdefault("executed_loop_count", loops)
        streams, tail_aux = self._run_stack(
            self.tail, streams, state, segment_ids=segment_ids,
            reset_mask=reset_mask, diagnostics=diagnostics,
            collect_diagnostics=collect_diagnostics,
            collect_router_diagnostics=collect_router_diagnostics,
            position=position,
        )
        auxiliary = auxiliary + tail_aux
        state.streams = streams.detach()
        state.token_positions = torch.arange(length, device=input_ids.device).view(1, -1).expand(batch, -1)
        collapsed = self.mhc_collapse(streams.reshape(batch, length, -1))
        hidden = self.final_norm(collapsed)
        # Only materialise full [B, T, V] logits for inference (labels=None).
        # During training we use chunked CE directly on hidden to save ~4 GB peak VRAM.
        if labels is None:
            projection_hidden = hidden[:, -1:, :] if last_logits_only else hidden
            logits = F.linear(projection_hidden, self.embed_tokens.weight)
        else:
            logits = hidden.new_zeros(*hidden.shape[:-1], 1)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            # Chunked fused cross-entropy.  MoE activation checkpointing keeps enough
            # headroom for the original large GEMMs without thousands of tiny matmuls.
            hidden_flat = hidden[:, :-1].reshape(-1, hidden.shape[-1])           # [(B*(T-1)), D]
            target_flat = labels[:, 1:].reshape(-1)                              # [(B*(T-1))]
            # Keep the fused full-vocabulary GEMM while leaving peak headroom for
            # the checkpointed MoE backward graph on the 24GB card.
            # Keep the full-vocabulary projection below the RTX 3090's peak
            # allocation when a recurrent/MoE activation spike coincides with
            # the loss projection.  At vocab_size=248,320 and fp16/bf16,
            # 512 rows creates a ~242.5 MiB temporary; 16 rows is ~7.6 MiB
            # for the fp16/bf16 projection plus ~15.1 MiB after FP32 casting.
            # Make the safety margin runtime-tunable without changing weights.
            try:
                chunk_size = max(8, int(os.getenv("LLMY_VOCAB_CHUNK", "16")))
            except ValueError:
                chunk_size = 16
            embed_w = self.embed_tokens.weight
            row_losses = []
            loss_total = hidden.new_zeros((), dtype=torch.float32)
            for b_idx in range(batch):
                row_h = hidden[b_idx, :-1]
                row_t = labels[b_idx, 1:]
                row_total = row_h.shape[0]
                row_loss_accum = hidden.new_zeros((), dtype=torch.float32)
                row_count_t = torch.zeros((), device=hidden.device, dtype=torch.long)
                for chunk_start in range(0, row_total, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, row_total)
                    h_chunk = row_h[chunk_start:chunk_end]
                    t_chunk = row_t[chunk_start:chunk_end]
                    h_proj = h_chunk if h_chunk.dtype == embed_w.dtype else h_chunk.to(embed_w.dtype)
                    loss_chunk = activation_checkpoint(
                        _checkpointed_vocab_ce,
                        h_proj,
                        t_chunk,
                        embed_w,
                        use_reentrant=False,
                    )
                    row_loss_accum = row_loss_accum + loss_chunk
                    row_count_t = row_count_t + (t_chunk >= 0).sum()
                r_loss = row_loss_accum / row_count_t.clamp_min(1).float()
                loss_total = loss_total + r_loss
                row_losses.append(float(r_loss.detach()))
            loss = (loss_total / float(max(1, batch))) + auxiliary
            if adaptive_on:
                loss = loss + halt_penalty_loss
            diagnostics["row_losses"] = row_losses
        if collect_diagnostics:
            mhc_metrics: dict[str, float] = {}
            for module_name, module in self.named_modules():
                if isinstance(module, MultiStreamHyperConnection):
                    for metric_name, value in module.constraint_errors().items():
                        mhc_metrics[f"{module_name}/{metric_name}"] = value
            diagnostics["mhc"] = mhc_metrics
            diagnostics["auxiliary_loss"] = float(auxiliary.detach())
        # These are intentionally emitted even when the trainer disables the
        # expensive per-block diagnostics.  Otherwise the learned halting
        # controller would be invisible in the normal low-overhead telemetry.
        if learned_halting:
            diagnostics["effective_loop_count"] = float(effective_loop_count.mean().detach())
            diagnostics["halt_penalty_loss"] = float(halt_penalty_loss.detach())
            if halt_decision_count:
                diagnostics["halt_continue_probability"] = float(
                    (halt_probability_sum / halt_decision_count).detach()
                )
        if collect_router_diagnostics:
            samples = diagnostics.pop("_router_health_samples", [])
            if samples:
                diagnostics["router_health"] = {
                    key: sum(float(sample[key]) for sample in samples if key in sample)
                    / max(1, sum(1 for sample in samples if key in sample))
                    for key in _ROUTER_HEALTH_KEYS
                    if any(key in sample for sample in samples)
                }
        return LoopedHybridOutput(logits, loss, state, auxiliary, diagnostics)

    @torch.inference_mode()
    def decode_step(self, input_ids: torch.Tensor, *, state: RecurrentState,
                    segment_ids: torch.Tensor | None = None,
                    reset_mask: torch.Tensor | None = None,
                    num_loops: int | None = None,
                    position: torch.Tensor | None = None) -> LoopedHybridOutput:
        """Generate logits for one token using recurrent and local state.

        The ordinary ``forward`` method intentionally remains a sequence
        implementation for training and evaluation. This path carries GDN
        state, compressed attention memory, and bounded local histories so a
        new token is not reprocessed with the entire prompt.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids[:, None]
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_step expects input_ids with shape [batch, 1]")
        batch = input_ids.shape[0]
        if state.token_positions is None:
            raise ValueError("decode_step requires state initialized by a prefill forward pass")
        if state.token_positions.ndim != 2 or state.token_positions.shape[0] != batch:
            raise ValueError("state token positions do not match input batch")
        if segment_ids is not None and segment_ids.shape != input_ids.shape:
            raise ValueError("segment_ids must have shape [batch, 1]")
        if reset_mask is not None and reset_mask.shape != input_ids.shape:
            raise ValueError("reset_mask must have shape [batch, 1]")

        reset_rows = reset_mask[:, 0].to(torch.bool) if reset_mask is not None else None
        if reset_rows is not None and bool(reset_rows.any()):
            state.reset_rows(reset_rows)
        if position is None:
            previous = state.token_positions[:, -1]
            position = previous + 1
            if reset_rows is not None:
                position = torch.where(reset_rows, torch.zeros_like(position), position)
        else:
            position = torch.as_tensor(position, device=input_ids.device, dtype=torch.long).view(batch)

        streams = self._streams_from_tokens(input_ids)
        loops = int(self.config.num_loops if num_loops is None else num_loops)
        if loops < 1:
            raise ValueError("num_loops must be positive")
        adaptive_cfg = self.config.adaptive_loops
        learned_halting = bool(adaptive_cfg.enabled and adaptive_cfg.learned_halting)
        if learned_halting:
            loops = min(loops, max(1, int(adaptive_cfg.max_loops)))
        minimum_loops = min(loops, max(1, int(adaptive_cfg.min_loops))) if learned_halting else loops
        auxiliary = streams.new_zeros((), dtype=torch.float32)
        streams, block_aux = self._run_step_stack(
            self.stem, streams, state, position=position,
            segment_ids=segment_ids, reset_mask=reset_mask,
        )
        auxiliary = auxiliary + block_aux
        loop_endpoints: list[torch.Tensor] = []
        loop_boundary_streams = streams
        active = streams.new_ones((batch, 1), dtype=torch.float32)
        executed_loop_counts = torch.zeros((batch,), device=streams.device, dtype=torch.float32)
        continue_probability_sum = streams.new_zeros(())
        continue_probability_count = 0
        for loop_index in range(loops):
            active_rows = torch.nonzero(active[:, 0] > 0, as_tuple=False).flatten()
            if active_rows.numel() == 0:
                break
            executed_loop_counts.index_add_(
                0,
                active_rows,
                torch.ones((active_rows.numel(),), device=streams.device, dtype=executed_loop_counts.dtype),
            )
            # Decode rows are independent at this point.  The compact path is
            # needed when only some batch rows continue, but the common batch-1
            # case has every row active.  Avoiding select/scatter copies of all
            # recurrent memories on every loop removes a large decode-only
            # overhead while preserving the exact same state transitions.
            all_rows_active = active_rows.numel() == batch
            if all_rows_active:
                sub_state = state
                sub_streams = streams
                sub_position = position
                sub_segments = segment_ids
                sub_reset = reset_mask
                sub_endpoints = loop_endpoints
            else:
                sub_state = state.select_rows(active_rows)
                sub_streams = streams.index_select(0, active_rows)
                sub_position = position.index_select(0, active_rows)
                sub_segments = None if segment_ids is None else segment_ids.index_select(0, active_rows)
                sub_reset = None if reset_mask is None else reset_mask.index_select(0, active_rows)
                sub_endpoints = [endpoint.index_select(0, active_rows) for endpoint in loop_endpoints]
            if loop_index > 0 and not self.config.mhc.preserve_streams_across_loop_boundary:
                sub_streams = loop_boundary_streams.clone() if all_rows_active else loop_boundary_streams.index_select(0, active_rows).clone()
            sub_streams = self.loop_memory.inject(sub_streams, sub_endpoints)
            sub_streams, loop_aux = self._run_step_stack(
                self._recurrent_stack(loop_index), sub_streams, sub_state, position=sub_position,
                segment_ids=sub_segments, reset_mask=sub_reset,
                state_namespace=f"loop{loop_index}",
            )
            auxiliary = auxiliary + loop_aux * (active_rows.numel() / max(batch, 1))
            endpoint = self.loop_memory.record(sub_streams, sub_state)
            if endpoint is not None:
                if all_rows_active:
                    loop_endpoints.append(endpoint)
                else:
                    full_endpoint = endpoint.new_zeros((batch, *endpoint.shape[1:]))
                    full_endpoint.index_copy_(0, active_rows, endpoint)
                    loop_endpoints.append(full_endpoint)
            if not all_rows_active:
                streams.index_copy_(0, active_rows, sub_streams)
                state.scatter_rows(active_rows, sub_state)
            if loop_index < loops - 1 and learned_halting:
                if loop_index + 1 < minimum_loops:
                    continue_decision = torch.ones_like(active_rows, dtype=torch.float32).view(-1, 1)
                    continue_probability = continue_decision
                else:
                    continue_decision, continue_probability = self._halt_decision(sub_streams)
                continue_probability_sum = continue_probability_sum + continue_probability.float().sum()
                continue_probability_count += int(continue_probability.numel())
                next_active = torch.zeros_like(active)
                next_active.index_copy_(0, active_rows, continue_decision.to(next_active.dtype))
                active = next_active
        streams, tail_aux = self._run_step_stack(
            self.tail, streams, state, position=position,
            segment_ids=segment_ids, reset_mask=reset_mask,
        )
        auxiliary = auxiliary + tail_aux
        state.streams = streams.detach()
        state.token_positions = position[:, None].detach()
        collapsed = self.mhc_collapse(streams.reshape(batch, 1, -1))
        hidden = self.final_norm(collapsed)
        logits = F.linear(hidden, self.embed_tokens.weight)
        diagnostics: dict[str, Any] = {
            "num_loops": loops,
            "min_loops": minimum_loops,
            "max_loops": loops,
            "executed_loop_count": int(executed_loop_counts.max().item()),
            "effective_loop_count": float(executed_loop_counts.mean().item()),
            "executed_loop_counts": executed_loop_counts.detach(),
            "loop_attention_residual_count": len(loop_endpoints),
        }
        if continue_probability_count:
            diagnostics["halt_continue_probability"] = float(
                (continue_probability_sum / continue_probability_count).item()
            )
        return LoopedHybridOutput(logits, None, state, auxiliary, diagnostics)

    def parameter_report(self) -> dict[str, Any]:
        from .config import count_unique_parameters

        return count_unique_parameters(self)
