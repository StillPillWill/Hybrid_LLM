"""Pass-1/pass-2/pass-3 recurrence diagnostics."""
from __future__ import annotations

from typing import Any, Iterable

from .lm_eval import evaluate_causal_lm


def evaluate_recurrence(model, batches: Iterable[dict[str, Any]], *, device=None,
                        loop_counts: tuple[int, ...] = (1, 2, 3), telemetry: Any = None,
                        split: str = "validation", step: int | None = None) -> dict[str, float]:
    materialized = list(batches)
    results: dict[str, float] = {}
    for loops in loop_counts:
        metrics = evaluate_causal_lm(
            model, materialized, device=device, num_loops=loops,
            telemetry=telemetry, split=f"{split}_pass{loops}", step=step,
        )
        results[f"ce_pass{loops}"] = metrics["cross_entropy"]
    if 1 in loop_counts and 2 in loop_counts:
        results["delta_loop"] = results["ce_pass1"] - results["ce_pass2"]
    if telemetry is not None:
        if hasattr(telemetry, "evaluation"):
            telemetry.evaluation(split=f"{split}_recurrence", metrics=results, step=step)
        else:
            telemetry.emit("evaluation", split=f"{split}_recurrence", metrics=results, step=step)
    return results
