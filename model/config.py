"""Typed configuration and parameter accounting for the V0 architecture."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass
class MHCConfig:
    enabled: bool = True
    streams: int = 4
    sinkhorn_iterations: int = 10
    init_gating_factor: float = 0.01
    shared_across_loops: bool = True
    preserve_streams_across_loop_boundary: bool = True


@dataclass
class GDNConfig:
    qk_heads: int = 16
    value_heads: int = 16
    qk_head_dim: int = 128
    value_head_dim: int = 128
    conv_kernel: int = 4
    sensitive_state_dtype: str = "float32"


@dataclass
class MoEConfig:
    routed_experts: int = 128
    shared_experts: int = 4
    routed_top_k: int = 20
    expert_hidden: int = 640
    activation: str = "swiglu"
    normalize_topk_prob: bool = True
    aux_balance_coef: float = 0.001
    token_dropping: bool = False
    expert_weights_shared_across_loops: bool = True


@dataclass
class MLAConfig:
    latent_dim: int = 256


@dataclass
class RouterConfig:
    context_aware: bool = True
    query_heads: int = 4
    query_head_dim: int = 64
    local_context: int = 128
    use_hca_global_latents: bool = True
    scoring_dtype: str = "float32"


@dataclass
class CSAConfig:
    temporal_compression: int = 4
    local_window: int = 128
    index_heads: int = 16
    index_head_dim: int = 128
    index_top_k: int = 32


@dataclass
class HCAConfig:
    temporal_compression: int = 32
    local_window: int = 128
    dense_global_attention: bool = True
    global_top_k: int = 32


@dataclass
class DenseFFNConfig:
    hidden: int = 3584
    activation: str = "swiglu"


@dataclass
class LoopAttnResConfig:
    implemented_hook: bool = True
    enabled: bool = False
    latent_dim: int = 128
    heads: int = 4
    head_dim: int = 32
    residual_gate_init: float = 0.05


@dataclass
class AdaptiveLoopConfig:
    """Token-wise recurrent depth and truncated-backpropagation controls."""

    enabled: bool = False
    learned_halting: bool = True
    min_loops: int = 1
    max_loops: int = 1
    halt_penalty: float = 0.01
    halt_threshold: float = 0.5
    halt_temperature: float = 1.0
    initial_continue_prob: float = 0.6
    # ``0`` means backpropagate through every executed loop.  A positive value
    # keeps the forward depth unchanged but truncates gradients to the final N
    # recurrent passes.
    backprop_loops: int = 0


@dataclass
class V6Config:
    vocab_size: int = 248_320
    d_model: int = 1024
    tie_word_embeddings: bool = True
    physical_blocks: int = 10
    stem_blocks: int = 2
    recurrent_blocks: int = 6
    tail_blocks: int = 2
    num_loops: int = 2
    recurrent_layout: tuple[str, ...] = (
        "gdn_moe", "gdn_moe", "csa_moe", "gdn_moe", "gdn_moe", "gdn_moe"
    )
    max_position_embeddings: int = 4096
    pad_token_id: int | None = None
    mhc: MHCConfig = field(default_factory=MHCConfig)
    gdn: GDNConfig = field(default_factory=GDNConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    mla: MLAConfig = field(default_factory=MLAConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    csa: CSAConfig = field(default_factory=CSAConfig)
    hca: HCAConfig = field(default_factory=HCAConfig)
    dense_ffn: DenseFFNConfig = field(default_factory=DenseFFNConfig)
    loop_attnres: LoopAttnResConfig = field(default_factory=LoopAttnResConfig)
    adaptive_loops: AdaptiveLoopConfig = field(default_factory=AdaptiveLoopConfig)

    def __post_init__(self) -> None:
        if self.physical_blocks != self.stem_blocks + self.recurrent_blocks + self.tail_blocks:
            raise ValueError("physical_blocks must equal stem + recurrent + tail blocks")
        if len(self.recurrent_layout) != self.recurrent_blocks:
            raise ValueError("recurrent_layout must contain one entry per recurrent block")
        if self.num_loops < 1:
            raise ValueError("num_loops must be positive")
        if self.adaptive_loops.min_loops < 1:
            raise ValueError("adaptive loop minimum must be positive")
        if self.adaptive_loops.max_loops < self.adaptive_loops.min_loops:
            raise ValueError("adaptive loop maximum must be >= minimum")
        if self.adaptive_loops.max_loops > self.num_loops:
            raise ValueError("adaptive loop maximum cannot exceed num_loops")
        if not 0.0 < self.adaptive_loops.halt_threshold < 1.0:
            raise ValueError("halt threshold must be in (0, 1)")
        if self.adaptive_loops.halt_temperature <= 0.0:
            raise ValueError("halt temperature must be positive")
        if not 0.0 < self.adaptive_loops.initial_continue_prob < 1.0:
            raise ValueError("initial continue probability must be in (0, 1)")
        if self.adaptive_loops.halt_penalty < 0.0:
            raise ValueError("halt penalty cannot be negative")
        if self.adaptive_loops.backprop_loops < 0:
            raise ValueError("backprop loop count cannot be negative")
        if self.adaptive_loops.backprop_loops > self.num_loops:
            raise ValueError("backprop loop count cannot exceed num_loops")
        if self.mhc.streams < 1:
            raise ValueError("mHC streams must be positive")
        if self.moe.routed_top_k > self.moe.routed_experts:
            raise ValueError("routed_top_k cannot exceed routed_experts")
        if self.hca.global_top_k < 1:
            raise ValueError("hca.global_top_k must be positive")
        if self.gdn.qk_heads != self.gdn.value_heads:
            raise ValueError("V0 requires matching GDN query/key and value head counts")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "V6Config":
        data = dict(value.get("model", value))
        nested = {
            "mhc": MHCConfig,
            "gdn": GDNConfig,
            "moe": MoEConfig,
            "mla": MLAConfig,
            "router": RouterConfig,
            "csa": CSAConfig,
            "hca": HCAConfig,
            "dense_ffn": DenseFFNConfig,
            "loop_attnres": LoopAttnResConfig,
            "adaptive_loops": AdaptiveLoopConfig,
        }
        for name, typ in nested.items():
            if isinstance(data.get(name), dict):
                data[name] = typ(**data[name])
        if isinstance(data.get("recurrent_layout"), list):
            data["recurrent_layout"] = tuple(data["recurrent_layout"])
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_config() -> V6Config:
    """Return the exact architecture defaults from section 24 of the spec."""
    return V6Config()


def count_unique_parameters(module: Any) -> dict[str, Any]:
    """Count physical tensors, avoiding double-counts from loop reuse/tied weights."""
    named_parameters = list(module.named_parameters())
    unique: dict[int, tuple[str, Any]] = {}
    for name, parameter in named_parameters:
        unique.setdefault(id(parameter), (name, parameter))
    categories: dict[str, int] = {}
    for parameter_name, parameter in unique.values():
        parameter_category = "other"
        for prefix, category in (
            ("embed", "embedding_head"), ("lm_head", "embedding_head"),
            ("experts", "routed_experts"), ("shared", "shared_experts"),
            ("gdn", "gdn"), ("csa", "csa"), ("hca", "hca"),
            ("router", "mla_router_indexer"), ("mla", "mla_router_indexer"),
            ("mhc", "mhc"), ("norm", "norms_other"),
        ):
            if prefix in parameter_name:
                parameter_category = category
                break
        categories[parameter_category] = categories.get(parameter_category, 0) + int(parameter.numel())
    unique_total = sum(int(parameter.numel()) for _, parameter in unique.values() if parameter.requires_grad)
    frozen_total = sum(int(parameter.numel()) for _, parameter in unique.values() if not parameter.requires_grad)
    active_applications = sum(int(parameter.numel()) for _, parameter in named_parameters if parameter.requires_grad)
    report = {
        "unique_parameter_ids": len(unique),
        "unique_trainable_parameters": unique_total,
        "unique_frozen_parameters": frozen_total,
        "active_parameter_applications": active_applications,
        "by_category": categories,
    }
    config = getattr(module, "config", None)
    if config is not None and hasattr(config, "num_loops"):
        stem = sum(int(p.numel()) for name, p in named_parameters if name.startswith("stem."))
        core = sum(int(p.numel()) for name, p in named_parameters if name.startswith("recurrent_core."))
        tail = sum(int(p.numel()) for name, p in named_parameters if name.startswith("tail."))
        routed = sum(int(p.numel()) for name, p in named_parameters if ".moe.experts." in name)
        shared = sum(int(p.numel()) for name, p in named_parameters if ".moe.shared." in name)
        core_non_routed = max(0, core - routed - shared)
        routed_active = routed * float(config.moe.routed_top_k) / max(1, int(config.moe.routed_experts))
        report["effective_block_applications"] = int(config.stem_blocks + config.recurrent_blocks * config.num_loops + config.tail_blocks)
        report["active_parameter_applications_per_token_estimate"] = int(
            self_numel(module, "embed_tokens") + stem + tail
            + int(config.num_loops) * (core_non_routed + shared + routed_active)
        )
    return report


def self_numel(module: Any, prefix: str) -> int:
    return sum(int(parameter.numel()) for name, parameter in module.named_parameters() if name.startswith(prefix))
