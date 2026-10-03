"""Deterministic matrix-optimizer/AdamW parameter partitioning."""
from __future__ import annotations

from typing import Any

import torch

from .mud import MUD
from .muon import Muon


def partition_parameters(module: torch.nn.Module) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    mud: list[tuple[str, torch.nn.Parameter]] = []
    adamw: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        lower = name.lower()
        use_adamw = (
            parameter.ndim < 2 or "embed" in lower or "norm" in lower or "bias" in lower
            or any(token in lower for token in ("router", "state", "gating_logit"))
        )
        (adamw if use_adamw else mud).append((name, parameter))
    return {"mud": mud, "adamw": adamw}


class HybridOptimizer:
    """A matrix optimizer plus AdamW for irregular/small tensors."""

    def __init__(
        self,
        matrix_optimizer: MUD | Muon | None,
        adamw_optimizer: torch.optim.Optimizer | None,
        *,
        matrix_name: str = "mud",
    ) -> None:
        # Keep ``mud`` as a compatibility alias for older callers/checkpoints.
        self.matrix = matrix_optimizer
        self.mud = matrix_optimizer
        self.matrix_name = str(matrix_name)
        self.adamw = adamw_optimizer

    @property
    def param_groups(self):
        groups = []
        if self.matrix is not None:
            groups.extend(self.matrix.param_groups)
        if self.adamw is not None:
            groups.extend(self.adamw.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.matrix is not None:
            self.matrix.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if self.matrix is not None:
            loss = self.matrix.step(closure)
            closure = None
        if self.adamw is not None:
            adam_loss = self.adamw.step(closure)
            loss = loss if loss is not None else adam_loss
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "matrix_optimizer": self.matrix_name,
            "matrix": None if self.matrix is None else self.matrix.state_dict(),
            "adamw": None if self.adamw is None else self.adamw.state_dict(),
        }

    def load_state_dict(self, value: dict[str, Any]) -> None:
        saved_name = value.get("matrix_optimizer")
        if saved_name is not None and str(saved_name) != self.matrix_name:
            raise ValueError(
                f"incompatible matrix optimizer state: saved={saved_name}, "
                f"current={self.matrix_name}"
            )
        matrix_state = value.get("matrix")
        if matrix_state is None:
            # Legacy checkpoints stored the matrix optimizer under ``mud``.
            matrix_state = value.get(self.matrix_name) or value.get("mud")
        def load_compatible(optimizer: torch.optim.Optimizer, saved: dict[str, Any]) -> None:
            """Load compatible prefix state when a checkpoint gained parameters.

            The fused mHC migration replaces three projections with one fused
            projection, and older checkpoints can therefore have a one- or
            two-parameter group-count difference.  PyTorch rejects that
            harmless topology drift before loading any state.  Parameter
            order remains deterministic, so preserve the matching prefix and
            leave newly introduced parameters state-less for their first step.
            """
            import copy
            adapted = copy.deepcopy(saved)
            current_groups = optimizer.param_groups
            saved_groups = adapted.get("param_groups", [])
            if len(saved_groups) != len(current_groups):
                raise ValueError("optimizer group count changed")
            next_id = max(adapted.get("state", {}).keys(), default=-1) + 1
            for saved_group, current_group in zip(saved_groups, current_groups):
                current_count = len(current_group.get("params", []))
                params = list(saved_group.get("params", []))
                if len(params) < current_count:
                    for _ in range(current_count - len(params)):
                        params.append(next_id)
                        next_id += 1
                elif len(params) > current_count:
                    dropped = params[current_count:]
                    params = params[:current_count]
                    for parameter_id in dropped:
                        adapted.get("state", {}).pop(parameter_id, None)
                saved_group["params"] = params
            optimizer.load_state_dict(adapted)

        if self.matrix is not None and matrix_state is not None:
            load_compatible(self.matrix, matrix_state)
        if self.adamw is not None and value.get("adamw") is not None:
            load_compatible(self.adamw, value["adamw"])


def build_hybrid_optimizer(module: torch.nn.Module, *, mud_lr: float = 1e-3,
                           mud_beta: float = 0.95, mud_passes: int = 1,
                           adamw_lr: float | None = None,
                           betas: tuple[float, float] = (0.9, 0.95),
                           weight_decay: float = 0.01,
                           matrix_optimizer: str = "mud",
                           muon_lr: float | None = None,
                           muon_beta: float = 0.95,
                           muon_ns_steps: int = 5) -> tuple[HybridOptimizer, dict[str, Any]]:
    groups = partition_parameters(module)
    mud_params = [parameter for _, parameter in groups["mud"]]
    adamw_params = [parameter for _, parameter in groups["adamw"]]
    matrix_kind = str(matrix_optimizer).lower()
    if matrix_kind == "muon":
        matrix = Muon(
            mud_params,
            lr=muon_lr if muon_lr is not None else 0.02,
            beta=muon_beta,
            weight_decay=weight_decay,
            ns_steps=muon_ns_steps,
        ) if mud_params else None
    elif matrix_kind == "mud":
        matrix = MUD(
            mud_params,
            lr=mud_lr,
            beta=mud_beta,
            weight_decay=weight_decay,
            passes=mud_passes,
        ) if mud_params else None
    else:
        matrix = None
        adamw_lr = mud_lr
        adamw_params = mud_params + adamw_params
    adamw = torch.optim.AdamW(
        adamw_params, lr=adamw_lr or mud_lr, betas=betas, weight_decay=weight_decay
    ) if adamw_params else None
    if matrix is not None:
        matrix.param_groups[0]["group_name"] = matrix_kind
    if adamw is not None:
        adamw.param_groups[0]["group_name"] = "adamw"
    return HybridOptimizer(matrix, adamw, matrix_name=matrix_kind), {
        "mud_parameters": [name for name, _ in groups["mud"]],
        "adamw_parameters": [name for name, _ in groups["adamw"]],
        "mud_lr": mud_lr,
        "adamw_lr": adamw_lr or mud_lr,
        "weight_decay": weight_decay,
    }
