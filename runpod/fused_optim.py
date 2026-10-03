"""Optimized hybrid Muon/AdamW (with legacy MUD support)."""
from __future__ import annotations
import torch
from optim.param_groups import partition_parameters, HybridOptimizer
from optim.mud import MUD
from optim.muon import Muon

def build_a5000_fused_optimizer(module: torch.nn.Module, *, mud_lr: float = 1e-3,
                                mud_beta: float = 0.95, mud_passes: int = 1,
                                adamw_lr: float | None = None,
                                muon_lr: float | None = None,
                                muon_beta: float = 0.95,
                                muon_ns_steps: int = 5,
                                betas: tuple[float, float] = (0.9, 0.95),
                                weight_decay: float = 0.01,
                                matrix_optimizer: str = "mud") -> tuple[HybridOptimizer, dict]:
    groups = partition_parameters(module)
    matrix_params = [p for _, p in groups["mud"]]
    adamw_params = [p for _, p in groups["adamw"]]
    matrix_kind = str(matrix_optimizer).lower()

    # Keep the matrix group explicit.  Muon and MUD have incompatible state,
    # so the trainer must reset the optimizer state when switching between them.
    if matrix_kind == "muon":
        matrix = (
            Muon(
                matrix_params,
                lr=muon_lr if muon_lr is not None else 0.02,
                beta=muon_beta,
                weight_decay=weight_decay,
                ns_steps=muon_ns_steps,
            )
            if matrix_params
            else None
        )
    elif matrix_kind == "mud":
        matrix = (
            MUD(
                matrix_params,
                lr=mud_lr,
                beta=mud_beta,
                weight_decay=weight_decay,
                passes=mud_passes,
            )
            if matrix_params
            else None
        )
    elif matrix_kind == "adamw":
        adamw_params = matrix_params + adamw_params
        matrix = None
    else:
        raise ValueError(f"unsupported matrix optimizer: {matrix_optimizer!r}")

    fused_available = torch.cuda.is_available()
    adamw = torch.optim.AdamW(
        adamw_params,
        lr=adamw_lr if adamw_lr is not None else mud_lr,
        betas=betas,
        weight_decay=weight_decay,
        fused=fused_available,
    ) if adamw_params else None

    if matrix is not None:
        matrix.param_groups[0]["group_name"] = matrix_kind
        matrix.param_groups[0]["base_lr"] = float(
            muon_lr if matrix_kind == "muon" and muon_lr is not None else mud_lr
        )
    if adamw is not None:
        adamw.param_groups[0]["group_name"] = "adamw"
        adamw.param_groups[0]["base_lr"] = float(
            adamw_lr if adamw_lr is not None else mud_lr
        )

    return HybridOptimizer(matrix, adamw, matrix_name=matrix_kind), {
        "matrix_parameters": [name for name, _ in groups["mud"]],
        "matrix_optimizer": matrix_kind,
        "muon_parameters": [name for name, _ in groups["mud"]] if matrix_kind == "muon" else [],
        "adamw_parameters": [name for name, _ in groups["adamw"]],
        "mud_lr": mud_lr,
        "muon_lr": muon_lr,
        "adamw_lr": adamw_lr if adamw_lr is not None else mud_lr,
        "weight_decay": weight_decay,
        "fused_adamw": fused_available,
    }
