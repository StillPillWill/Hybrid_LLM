"""Muon optimizer for matrix parameters.

Muon keeps a momentum buffer and orthogonalizes the momentum with a short
Newton--Schulz iteration.  It is intended for matrix-shaped weights; the
hybrid optimizer keeps embeddings, norms, biases, routers, and state tensors
on AdamW instead.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(
    gradient: torch.Tensor, *, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate the polar factor of a matrix with Newton--Schulz steps."""
    if gradient.ndim < 2:
        raise ValueError("Muon requires tensors with at least two dimensions")

    original_shape = gradient.shape
    rows = original_shape[0]
    cols = 1
    for size in original_shape[1:]:
        cols *= size
    matrix = gradient.reshape(rows, cols)
    transposed = rows > cols
    if transposed:
        matrix = matrix.transpose(0, 1)

    # BF16 is substantially faster on the target Ampere GPU and is sufficient
    # for the polynomial.  Keep CPU/unit-test execution in float32.
    work_dtype = (
        torch.bfloat16
        if matrix.is_cuda and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    x = matrix.to(work_dtype)
    x = x / (x.float().norm() + eps).to(x.dtype)

    # Quintic coefficients used by the standard Muon Newton--Schulz variant.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(max(1, int(steps))):
        gram = x @ x.transpose(0, 1)
        x = a * x + (b * gram + c * (gram @ gram)) @ x

    if transposed:
        x = x.transpose(0, 1)
    update = x.to(dtype=gradient.dtype).reshape(original_shape)

    # Shape-aware scaling is part of Muon's matrix update rule.  Flattening
    # higher-rank weights above makes this well-defined for expert projections
    # and convolutional kernels as well as ordinary Linear weights.
    update = update * max(1.0, rows / max(1, cols)) ** 0.5
    return update


def zeropower_via_newtonschulz5_batched(
    stack: torch.Tensor, *, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate polar factors for a batch of identically-shaped matrices via batched GEMM."""
    orig_shape = stack.shape
    b = orig_shape[0]
    rows = orig_shape[1]
    cols = 1
    for s in orig_shape[2:]:
        cols *= s
    matrix = stack.reshape(b, rows, cols)
    transposed = rows > cols
    if transposed:
        matrix = matrix.transpose(1, 2)

    work_dtype = (
        torch.bfloat16
        if matrix.is_cuda and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    x = matrix.to(work_dtype)
    norm = torch.linalg.vector_norm(x.float(), dim=(-2, -1), keepdim=True).to(x.dtype)
    x = x / (norm + eps)

    a, b_coef, c = 3.4445, -4.7750, 2.0315
    for _ in range(max(1, int(steps))):
        gram = torch.bmm(x, x.transpose(1, 2))
        x = a * x + torch.bmm(b_coef * gram + c * torch.bmm(gram, gram), x)

    if transposed:
        x = x.transpose(1, 2)
    update = x.to(dtype=stack.dtype).reshape(orig_shape)
    scale = max(1.0, rows / max(1, cols)) ** 0.5
    if scale != 1.0:
        update = update * scale
    return update


class Muon(Optimizer):
    """Momentum-orthogonalized optimizer for matrix-shaped parameters."""

    def __init__(
        self,
        params: Iterable,
        *,
        lr: float = 0.02,
        beta: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        eps: float = 1e-7,
        nesterov: bool = True,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0, 1)")
        defaults = dict(
            lr=float(lr),
            beta=float(beta),
            weight_decay=float(weight_decay),
            ns_steps=int(ns_steps),
            eps=float(eps),
            nesterov=bool(nesterov),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta = float(group["beta"])
            decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            eps = float(group["eps"])
            nesterov = bool(group["nesterov"])

            parameters = []
            gradients = []
            momentums = []
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if parameter.ndim < 2:
                    raise RuntimeError("Muon received a non-matrix parameter")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(parameter)
                state["step"] += 1
                momentum = state["momentum"]
                parameters.append(parameter)
                gradients.append(gradient)
                momentums.append(momentum)
            if not parameters:
                continue

            torch._foreach_lerp_(momentums, gradients, 1.0 - beta)
            if nesterov:
                directions = torch._foreach_lerp(gradients, momentums, beta)
            else:
                directions = momentums

            # Group directions by shape for fast batched Tensor Core execution
            shape_groups: dict[tuple, list[int]] = {}
            for i, d in enumerate(directions):
                key = (d.shape, d.dtype, d.device)
                shape_groups.setdefault(key, []).append(i)

            updates: list[torch.Tensor] = [None] * len(directions)  # type: ignore[list-item]
            for key, indices in shape_groups.items():
                if len(indices) >= 2:
                    stack = torch.stack([directions[idx] for idx in indices])
                    batched_updates = zeropower_via_newtonschulz5_batched(
                        stack, steps=ns_steps, eps=eps
                    )
                    for idx, single_update in zip(indices, torch.unbind(batched_updates, dim=0)):
                        updates[idx] = single_update
                else:
                    idx = indices[0]
                    updates[idx] = zeropower_via_newtonschulz5(
                        directions[idx], steps=ns_steps, eps=eps
                    )

            if decay:
                torch._foreach_mul_(parameters, 1.0 - lr * decay)
            torch._foreach_add_(parameters, updates, alpha=-lr)
        return loss
