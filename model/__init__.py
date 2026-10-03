"""V0 2B looped hybrid-MoE model package.

The package is deliberately independent from the historical V5/V6 compression
runner in the repository.  It exposes the architecture described by
``v6.docx`` as ordinary PyTorch modules so each subsystem can be tested in
isolation before a long training run is started.
"""

from .config import AdaptiveLoopConfig, V6Config, canonical_config, count_unique_parameters
from .model import LoopedHybridMoEForCausalLM
from .recurrent_state import RecurrentState

__all__ = [
    "LoopedHybridMoEForCausalLM",
    "RecurrentState",
    "V6Config",
    "AdaptiveLoopConfig",
    "canonical_config",
    "count_unique_parameters",
]
