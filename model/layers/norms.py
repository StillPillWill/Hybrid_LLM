from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        variance = value.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + self.eps).to(dtype=value.dtype)
        return normalized * self.weight.to(dtype=value.dtype)
