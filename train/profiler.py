from __future__ import annotations

from contextlib import nullcontext
from typing import Any


def profile_context(enabled: bool = False, *, wait: int = 1, warmup: int = 1,
                   active: int = 2, output_dir: str | None = None) -> Any:
    if not enabled:
        return nullcontext()
    import torch

    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
        + ([torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
        on_trace_ready=(torch.profiler.tensorboard_trace_handler(output_dir) if output_dir else None),
        record_shapes=True, profile_memory=True, with_stack=False,
    )
