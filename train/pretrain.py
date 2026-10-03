"""Entry point for the controlled 1B-token V0 pretraining run.

This module defines the launch contract only; importing it does not allocate a
model or start work.  Invoke ``main`` explicitly from an operator-controlled
launch after the proxy and preflight gates have passed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Ensure repo root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from data.dataset import PackedWindowDataset, ShardedPackedWindowLoader, collate_packed_windows
from data.pack import PackedBatch
from eval.recurrence import evaluate_recurrence
from model import LoopedHybridMoEForCausalLM, V6Config
from optim import build_hybrid_optimizer

try:
    from .checkpoint import CheckpointManager
    from .telemetry import JsonlTelemetry
    from .telemetry import WandbTelemetry, state_scalars
except ImportError:  # Allows the documented ``python train/pretrain.py`` form.
    from train.checkpoint import CheckpointManager
    from train.telemetry import JsonlTelemetry, WandbTelemetry, state_scalars


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("training config must be a mapping")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def objective(output, labels: torch.Tensor) -> torch.Tensor:
    if output.loss is None:
        raise ValueError("the V0 objective requires final-loop labels")
    return output.loss


def train_steps(model: LoopedHybridMoEForCausalLM, loader: Iterable[dict[str, Any]],
                optimizer: Any, *, device: torch.device, token_budget: int,
                grad_clip: float = 1.0, grad_accum_steps: int = 4,
                telemetry: JsonlTelemetry | None = None,
                checkpoints: CheckpointManager | None = None, checkpoint_every: int = 100,
                start_step: int = 0, start_tokens: int = 0,
                warmup_fraction: float = 0.02, min_lr_ratio: float = 0.1) -> dict[str, int]:
    model.train()
    step, token_count = int(start_step), int(start_tokens)
    base_lrs = []
    for group in optimizer.param_groups:
        group.setdefault("base_lr", float(group["lr"]))
        base_lrs.append(float(group["base_lr"]))
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    while token_count < token_budget:
        accum_loss = 0.0
        accum_tokens = 0
        step_diagnostics = {}
        t0_step = time.time()
        
        for _ in range(grad_accum_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            input_ids = batch["input_ids"].to(device).unsqueeze(0) if batch["input_ids"].ndim == 1 else batch["input_ids"].to(device)
            labels = batch["labels"].to(device).unsqueeze(0) if batch["labels"].ndim == 1 else batch["labels"].to(device)
            segments = batch["segment_ids"].to(device).unsqueeze(0) if batch["segment_ids"].ndim == 1 else batch["segment_ids"].to(device)
            reset = batch["reset_mask"].to(device).unsqueeze(0) if batch["reset_mask"].ndim == 1 else batch["reset_mask"].to(device)
            
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                output = model(input_ids, labels=labels, segment_ids=segments, reset_mask=reset)
                loss = objective(output, labels)
                scaled_loss = loss / grad_accum_steps
                
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite V0 loss at step {step}: {loss.item()}")
                
            scaler.scale(scaled_loss).backward()
            accum_loss += float(loss.detach().item())
            accum_tokens += int(input_ids.numel())
            step_diagnostics = {**output.diagnostics, **state_scalars(output.state)}

        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        avg_loss = accum_loss / grad_accum_steps
        step += 1
        token_count += accum_tokens
        
        warmup_tokens = max(1, int(token_budget * warmup_fraction))
        if token_count < warmup_tokens:
            scale = token_count / warmup_tokens
        else:
            progress = min(1.0, (token_count - warmup_tokens) / max(1, token_budget - warmup_tokens))
            scale = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * scale
            
        if telemetry is not None:
            if isinstance(telemetry, WandbTelemetry):
                telemetry.log_step(
                    step=step, token_count=token_count, loss=avg_loss,
                    diagnostics=step_diagnostics, optimizer=optimizer,
                    device=device, batch_tokens=accum_tokens, model=model,
                )
            else:
                telemetry.emit("optimizer_step", step=step, token_count=token_count,
                               loss=avg_loss, diagnostics=step_diagnostics)
                               
        if step % 1 == 0 or step == 1:
            curr_lr = optimizer.param_groups[0]["lr"]
            vram_mb = torch.cuda.memory_allocated(device) / (1024**2) if device.type == "cuda" else 0
            step_dt = max(time.time() - t0_step, 1e-6)
            tps = accum_tokens / step_dt
            print(f"[Step {step:5d}] tokens={token_count:8,d} | loss={avg_loss:.4f} | lr={curr_lr:.2e} | {tps:5.1f} tok/s | vram={vram_mb:.0f} MB", flush=True)
            
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda" and step % 50 == 0:
            torch.cuda.empty_cache()
        if checkpoints is not None and step % max(1, checkpoint_every) == 0:
            checkpoint_path = checkpoints.save(
                "latest.pt", model=model, optimizer=optimizer, step=step,
                token_count=token_count, telemetry=step_diagnostics,
            )
            if isinstance(telemetry, WandbTelemetry):
                telemetry.checkpoint(step=step, token_count=token_count, path=checkpoint_path)
    return {"step": step, "token_count": token_count}


def build_model_and_optimizer(raw_config: dict[str, Any]):
    model_cfg = V6Config.from_mapping(raw_config)
    model = LoopedHybridMoEForCausalLM(model_cfg)
    optimizer_cfg = raw_config.get("optimizer", {})
    optimizer, partition = build_hybrid_optimizer(
        model, mud_lr=float(optimizer_cfg.get("mud_lr", 1e-3)),
        mud_beta=float(optimizer_cfg.get("mud_beta", 0.95)),
        mud_passes=int(optimizer_cfg.get("mud_passes", 1)),
        betas=tuple(optimizer_cfg.get("adamw_betas", (0.9, 0.95))),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
    )
    return model, optimizer, partition


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the V0 2B looped hybrid-MoE pretrain")
    parser.add_argument("--config", required=True)
    parser.add_argument("--packed-batch", help="torch file containing a PackedBatch")
    parser.add_argument("--data-dir", help="directory containing sharded train-*.pt PackedBatch files")
    parser.add_argument("--validation-packed-batch", help="optional validation PackedBatch for pass-1/2/3 W&B metrics")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 1337)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the V0 pretraining entry point")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(24)
    model, optimizer, partition = build_model_and_optimizer(config)
    model.to(device)
    context_length = int(config.get("training", {}).get("context_length", 256))
    batch_size = int(config.get("training", {}).get("batch_size", 2))
    
    if args.data_dir:
        import glob
        shard_paths = sorted(glob.glob(str(Path(args.data_dir) / "train-*.pt")))
        if not shard_paths:
            raise FileNotFoundError(f"No train-*.pt shards found in {args.data_dir}")
        print(f"[DataLoader] Initialized sharded loader with {len(shard_paths)} shards from {args.data_dir}", flush=True)
        loader = ShardedPackedWindowLoader(
            shard_paths, window_size=context_length, batch_size=batch_size,
            shuffle_shards=True, shuffle_windows=True, seed=int(config.get("seed", 1337)),
        )
    elif args.packed_batch:
        packed = torch.load(args.packed_batch, map_location="cpu", weights_only=False)
        if not isinstance(packed, PackedBatch):
            raise TypeError("--packed-batch must contain a data.pack.PackedBatch")
        loader = DataLoader(PackedWindowDataset(packed, window_size=context_length), batch_size=batch_size, shuffle=True, collate_fn=collate_packed_windows)
    else:
        raise ValueError("Either --data-dir or --packed-batch must be specified")
        
    validation_loader = None
    if args.validation_packed_batch:
        validation_packed = torch.load(args.validation_packed_batch, map_location="cpu", weights_only=False)
        if not isinstance(validation_packed, PackedBatch):
            raise TypeError("--validation-packed-batch must contain a data.pack.PackedBatch")
        validation_loader = DataLoader(PackedWindowDataset(validation_packed, window_size=context_length), batch_size=batch_size, shuffle=False, collate_fn=collate_packed_windows)
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    checkpoints = CheckpointManager(run)
    wandb_cfg = config.get("wandb", {})
    telemetry = WandbTelemetry(
        run / "events.jsonl", run_dir=run,
        config={**config, "parameter_report": model.parameter_report(),
                "optimizer_partition": partition},
        model=model,
        project=str(wandb_cfg.get("project", "hybrid-2b-looped-v0")),
        entity=wandb_cfg.get("entity"),
        run_name=wandb_cfg.get("run_name"),
        group=wandb_cfg.get("group", config.get("campaign", "v0")),
        tags=list(wandb_cfg.get("tags", [])) or ["v0", "looped-hybrid-moe", "qwen", "resumable"],
        online_required=bool(wandb_cfg.get("online_required", True)),
        watch=bool(wandb_cfg.get("watch", False)),
    )
    metadata = {"config": config, "optimizer_partition": partition,
                "parameter_report": model.parameter_report()}
    (run / "manifest.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    try:
        state = {"step": 0, "token_count": 0}
        if args.resume:
            checkpoint_path = run / "latest.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Resume requested but no checkpoint found at {checkpoint_path}")
            payload = checkpoints.load(checkpoint_path, model=model, optimizer=optimizer, map_location=device)
            state = {"step": int(payload["step"]), "token_count": int(payload["token_count"])}
            telemetry.emit("resumed", checkpoint=str(checkpoint_path), **state)
        result = train_steps(
            model, loader, optimizer, device=device,
            token_budget=int(config.get("training", {}).get("tokens", 1_000_000_000)),
            grad_clip=float(config.get("optimizer", {}).get("grad_clip", 1.0)),
            grad_accum_steps=int(config.get("training", {}).get("gradient_accumulation_steps", 4)),
            telemetry=telemetry, checkpoints=checkpoints,
            start_step=state["step"], start_tokens=state["token_count"],
            warmup_fraction=float(config.get("schedule", {}).get("warmup_token_fraction", 0.02)),
            min_lr_ratio=float(config.get("schedule", {}).get("min_lr_ratio", 0.1)),
            checkpoint_every=int(config.get("training", {}).get("checkpoint_every", 20)),
        )
        final_path = checkpoints.save(
            "final.pt", model=model, optimizer=optimizer, step=result["step"],
            token_count=result["token_count"], config=config,
        )
        if isinstance(telemetry, WandbTelemetry):
            telemetry.checkpoint(step=result["step"], token_count=result["token_count"], path=final_path)
        if validation_loader is not None:
            evaluate_recurrence(
                model, validation_loader, device=device, telemetry=telemetry,
                split="validation", step=result["step"],
            )
        if isinstance(telemetry, WandbTelemetry):
            telemetry.finish(status="completed")
        else:
            telemetry.emit("run_finished", status="completed", **result)
    except BaseException as exc:
        if isinstance(telemetry, WandbTelemetry):
            try:
                telemetry.finish(status="failed", error=exc)
            except Exception:
                pass
        else:
            telemetry.emit("run_finished", status="failed", error=repr(exc))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
