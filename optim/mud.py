"""MomentUm Decorrelation optimizer used for substantial matrix parameters."""
from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


def _decorrelate(momentum: torch.Tensor, passes: int) -> torch.Tensor:
    if momentum.ndim < 2:
        return momentum
    matrix = momentum.reshape(momentum.shape[0], -1)
    transpose = matrix.shape[0] < matrix.shape[1]
    if transpose:
        matrix = matrix.transpose(0, 1)
    scale = matrix.norm().clamp_min(1e-6)
    matrix = matrix / scale
    # Newton-Schulz iterations approximate the polar factor without an SVD.
    for _ in range(max(1, int(passes))):
        matrix = 1.5 * matrix - 0.5 * matrix @ (matrix.transpose(0, 1) @ matrix)
    if transpose:
        matrix = matrix.transpose(0, 1)
    return matrix.reshape_as(momentum) * scale


class MUD(Optimizer):
    """A compact MUD implementation with explicit, serializable state.

    The optimizer keeps decorrelated first moments for matrix-shaped tensors.
    It intentionally exposes only the narrow V0 surface needed by the hybrid
    trainer; comparisons against AdamW remain a required preflight.
    """

    def __init__(self, params: Iterable, lr: float = 1e-3, beta: float = 0.95,
                 weight_decay: float = 0.01, passes: int = 1, eps: float = 1e-8):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0, 1)")
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay,
                        passes=int(passes), eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            decay = group["weight_decay"]
            eps = group["eps"]
            passes = group["passes"]
            parameters = []
            updates = []
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("MUD does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(parameter)
                state["step"] += 1
                momentum = state["momentum"]
                momentum.mul_(beta).add_(grad, alpha=1.0 - beta)
                update = _decorrelate(momentum, passes)
                parameters.append(parameter)
                updates.append(update)
            if not updates:
                continue
            norms = torch._foreach_norm(updates)
            normalized = [
                update / (norm.float().div(update.numel() ** 0.5).to(update.dtype) + eps)
                for update, norm in zip(updates, norms)
            ]
            if decay:
                torch._foreach_mul_(parameters, 1.0 - lr * decay)
            torch._foreach_add_(parameters, normalized, alpha=-lr)
        return loss
