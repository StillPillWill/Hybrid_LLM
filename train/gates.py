"""Fail-closed V0 promotion gates evaluated from recorded telemetry."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class V0GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    measured: dict[str, float] = field(default_factory=dict)


def evaluate_v0_gate(metrics: dict[str, Any], *, min_loop_delta: float = 0.0,
                     max_mhc_error: float = 0.05, min_csa_remote_fraction: float = 0.01,
                     min_hca_global_mass: float = 0.01, max_dead_experts: int = 8) -> V0GateResult:
    failures: list[str] = []
    required = ("ce_pass1", "ce_pass2", "dead_experts", "mhc_max_constraint_error",
                "csa_remote_fraction", "hca_global_mass", "validation_loss")
    failures.extend(f"missing:{name}" for name in required if name not in metrics)
    if failures:
        return V0GateResult(False, failures)
    measured = {name: float(metrics[name]) for name in required}
    measured["delta_loop"] = measured["ce_pass1"] - measured["ce_pass2"]
    if measured["delta_loop"] < min_loop_delta:
        failures.append("pass_2_does_not_improve_held_out_ce")
    if measured["mhc_max_constraint_error"] > max_mhc_error:
        failures.append("mhc_constraint_error_exceeds_limit")
    if measured["csa_remote_fraction"] < min_csa_remote_fraction:
        failures.append("csa_has_no_nonlocal_retrieval")
    if measured["hca_global_mass"] < min_hca_global_mass:
        failures.append("hca_has_no_global_attention_mass")
    if int(metrics.get("dead_experts", 0)) > max_dead_experts:
        failures.append("dead_routed_experts_present")
    return V0GateResult(not failures, failures, measured)
