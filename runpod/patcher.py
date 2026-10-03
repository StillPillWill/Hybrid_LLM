"""Dynamic Model Patcher to inject all optimized Triton kernels into LoopedHybridMoE."""
from __future__ import annotations
import os
import torch
import torch.nn as nn

def apply_all_triton_optimizations(model: nn.Module) -> nn.Module:
    """Patches all model layers with high-performance Triton kernels."""
    from runpod.kernels.triton_rmsnorm import TritonRMSNorm
    from runpod.kernels.triton_banded_attention import _TRITON_AVAILABLE
    from model.layers.csa import CompressedSelectiveAttention
    from model.layers.hca import HierarchicalCompressedAttention
    from model.layers.mhc import MultiStreamHyperConnection
    experimental_triton = os.getenv("LLMY_ENABLE_EXPERIMENTAL_TRITON", "0") == "1"

    # 1. Patch RMSNorm modules
    patched_norms = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child.__class__.__name__ in ("RMSNorm", "LlamaRMSNorm"):
                eps = getattr(child, "eps", None) or 1e-6
                triton_norm = TritonRMSNorm(child.weight.shape[0], eps=eps).to(
                    device=child.weight.device, dtype=child.weight.dtype
                )
                triton_norm.weight.data.copy_(child.weight.data)
                setattr(module, child_name, triton_norm)
                patched_norms += 1

    patched_mhc = 0
    patched_hca = 0
    patched_csa = 0
    for module in model.modules():
        if isinstance(module, MultiStreamHyperConnection):
            # mHC was the stable fast path in the historical 40–42k run.
            # Keep the attention kernels opt-in because their backward paths
            # are not validated for this model, but do not gate mHC with them.
            module.use_triton_kernels = bool(_TRITON_AVAILABLE)
            patched_mhc += int(module.use_triton_kernels)
        elif isinstance(module, HierarchicalCompressedAttention):
            module.use_triton_banded = bool(_TRITON_AVAILABLE and experimental_triton)
            patched_hca += int(module.use_triton_banded)
        elif isinstance(module, CompressedSelectiveAttention):
            compatible = module.index_head_dim in (32, 64, 128)
            module.use_triton_banded = bool(
                _TRITON_AVAILABLE and experimental_triton and compatible
            )
            patched_csa += int(module.use_triton_banded)

    print(
        f"[Triton Optimization Patcher] RMSNorm={patched_norms}, "
        f"mHC={patched_mhc}, HCA-banded={patched_hca}, CSA-banded={patched_csa}; "
        "GDN/MoE use their native guarded paths"
    )
    return model
