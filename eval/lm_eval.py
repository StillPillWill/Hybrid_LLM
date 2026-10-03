"""Causal-LM evaluation after the final loop and tail."""
from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_causal_lm(model: torch.nn.Module, batches: Iterable[dict[str, torch.Tensor]], *, device=None,
                       num_loops: int | None = None, telemetry: Any = None,
                       split: str = "validation", step: int | None = None) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        input_ids = batch["input_ids"].to(device) if device is not None else batch["input_ids"]
        labels = batch.get("labels", input_ids).to(device) if device is not None else batch.get("labels", input_ids)
        kwargs = {key: batch[key].to(device) for key in ("segment_ids", "reset_mask") if key in batch and device is not None}
        if device is None:
            kwargs = {key: batch[key] for key in ("segment_ids", "reset_mask") if key in batch}
        output = model(input_ids, num_loops=num_loops, **kwargs)
        logits = output.logits[:, :-1].float()
        target = labels[:, 1:]
        per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
        valid = target.reshape(-1) != -100
        total_loss += float(per_token[valid].sum())
        total_tokens += int(valid.sum())
    if not total_tokens:
        raise ValueError("evaluation received no target tokens")
    ce = total_loss / total_tokens
    metrics = {"cross_entropy": ce, "perplexity": float(torch.exp(torch.tensor(ce))),
               "tokens": total_tokens}
    if telemetry is not None:
        if hasattr(telemetry, "evaluation"):
            telemetry.evaluation(split=split, metrics=metrics, step=step)
        else:
            telemetry.emit("evaluation", split=split, metrics=metrics, step=step)
    return metrics
