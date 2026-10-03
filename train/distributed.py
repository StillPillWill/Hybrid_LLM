"""Small distributed setup boundary; expert parallelism can replace this later."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0


def initialize_distributed() -> DistributedContext:
    if "RANK" not in os.environ:
        return DistributedContext()
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend=os.environ.get("V6_DIST_BACKEND", "nccl"))
    return DistributedContext(dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0)))
