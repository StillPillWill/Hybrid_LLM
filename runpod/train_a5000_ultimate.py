"""A5000 pretraining with streamed data and 15-minute checkpoints."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_a5000")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_a5000")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("LLMY_ENABLE_COMPILED_CE", "1")

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from torch.nn.utils import clip_grad_norm_

from train.pretrain import load_config, set_seed
from model import LoopedHybridMoEForCausalLM, V6Config
from model.layers.mla_memory import MLALatentMemory, LatentAttention
from runpod.streaming_dataset import ConcurrentStreamingLoader, WeightedInterleavedLoader, JointMultiSourceLoader
from runpod.distill_dataset import DistillationStreamingLoader
from runpod.async_loader import AsyncCUDAPrefetcher
from runpod.remote_sync import PeriodicRemoteCheckpointSync
from runpod.runtime_control import RuntimeControl
from runpod.terminal_dashboard import TerminalDashboard
from runpod.fused_optim import build_a5000_fused_optimizer
from runpod.patcher import apply_all_triton_optimizations
from train.telemetry import WandbTelemetry


def _configure_training_backends() -> None:
    """Enable Ampere-friendly kernels without assuming private APIs exist."""
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")
    try:
        torch._C._set_cached_tensors_enabled(True)
    except (AttributeError, RuntimeError):
        pass
    try:
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.suppress_errors = True
    except (AttributeError, RuntimeError):
        pass
    try:
        torch._inductor.config.triton.cudagraphs = False
        torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.fx_graph_cache = True
        torch._inductor.config.autotune_local_cache = True
    except (AttributeError, RuntimeError):
        pass


def _capture_gradient_norms(optimizer):
    """Capture gradient norms before zero_grad clears them."""
    group_norms = {}
    total_squared = 0.0
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("group_name", index))
        grads = [parameter.grad.detach() for parameter in group.get("params", [])
                 if parameter.grad is not None]
        if not grads:
            group_norms[name] = 0.0
            continue
        norms = torch._foreach_norm(grads)
        squared = float(torch.stack([norm.float().square() for norm in norms]).sum())
        group_norms[name] = squared ** 0.5
        total_squared += squared
    return total_squared ** 0.5, group_norms


def main():
    parser = argparse.ArgumentParser(description="A5000 pretraining engine")
    parser.add_argument("--config", default="runpod/config_a5000_moe24.yaml")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the primary data directory from the config",
    )
    parser.add_argument("--run-dir", default="runs/a5000_ultimate")
    parser.add_argument("--remote-sync-dest", help="Optional remote rsync destination, e.g. user@my-pc:/backup")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Resume model weights/counters but intentionally start a fresh optimizer state",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 1337)))
    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    _configure_training_backends()

    print(f"=== [A5000 Training Engine] Initializing on {torch.cuda.get_device_name(device)} ===")
    resume_activation_offload = bool(
        args.resume and os.getenv("LLMY_RESUME_ACTIVATION_OFFLOAD", "0") == "1"
    )
    try:
        activation_offload_min_tokens = max(
            1, int(os.getenv("LLMY_ACTIVATION_OFFLOAD_MIN_TOKENS", "8192"))
        )
    except ValueError:
        activation_offload_min_tokens = 8192
    try:
        long_sequence_loop_threshold = max(
            1, int(os.getenv("LLMY_LONG_SEQUENCE_LOOP_THRESHOLD", "8192"))
        )
    except ValueError:
        long_sequence_loop_threshold = 8192
    try:
        long_sequence_max_loops = max(
            1, int(os.getenv("LLMY_LONG_SEQUENCE_MAX_LOOPS", "2"))
        )
    except ValueError:
        long_sequence_max_loops = 2
    if resume_activation_offload:
        print(
            "[Memory] Resumed-run activation saves are offloaded to CPU; "
            "training configuration is unchanged."
        )

    model_cfg = V6Config.from_mapping(config)
    model = LoopedHybridMoEForCausalLM(model_cfg).to(device)
    model = apply_all_triton_optimizations(model)
    # The MLA compiler path is opt-in.  On the resumed A5000 run, dynamic
    # compilation produced NaN logits while the eager MLA path remained
    # finite on the exact same checkpoint and batch.  Keep the switch
    # available for isolated benchmarking, but never enable it implicitly in
    # a production training run.
    if (
        device.type == "cuda"
        and hasattr(torch, "compile")
        and os.getenv("LLMY_ENABLE_COMPILED_MLA", "0") == "1"
    ):
        try:
            MLALatentMemory.forward = torch.compile(MLALatentMemory.forward, dynamic=True)
            LatentAttention.forward = torch.compile(LatentAttention.forward, dynamic=True)
            print("[Compile] MLA latent memory and attention enabled")
        except Exception as exc:
            print(f"[Compile] MLA submodule compile skipped: {exc}")
        if os.getenv("LLMY_FULL_COMPILE", "0") == "1":
            try:
                model = torch.compile(
                    model,
                    mode="max-autotune-no-cudagraphs",
                    fullgraph=False,
                    dynamic=True,
                )
                print("[Compile] full model compile enabled")
            except Exception as exc:
                print(f"[Compile] full model compile skipped: {exc}")

    optim_cfg = config.get("optimizer", {})
    optimizer, partition = build_a5000_fused_optimizer(
        model,
        mud_lr=float(optim_cfg.get("mud_lr", 1e-3)),
        mud_beta=float(optim_cfg.get("mud_beta", 0.95)),
        mud_passes=int(optim_cfg.get("mud_passes", 1)),
        muon_lr=(
            float(optim_cfg["muon_lr"])
            if optim_cfg.get("muon_lr") is not None
            else None
        ),
        muon_beta=float(optim_cfg.get("muon_beta", 0.95)),
        muon_ns_steps=int(optim_cfg.get("muon_ns_steps", 5)),
        adamw_lr=(
            float(optim_cfg["adamw_lr"])
            if optim_cfg.get("adamw_lr") is not None
            else None
        ),
        betas=tuple(optim_cfg.get("adamw_betas", (0.9, 0.95))),
        weight_decay=float(optim_cfg.get("weight_decay", 0.01)),
        matrix_optimizer=str(optim_cfg.get("matrix_optimizer", "mud")),
    )
    print(
        f"[Optimizer] matrix={partition.get('matrix_optimizer')} "
        f"matrix_params={len(partition.get('matrix_parameters', [])):,} "
        f"adamw_params={len(partition.get('adamw_parameters', [])):,}"
    )

    context_length = int(config.get("training", {}).get("context_length", 256))
    batch_size = int(config.get("training", {}).get("batch_size", 16))
    grad_accum_steps = int(config.get("training", {}).get("gradient_accumulation_steps", 1))
    token_budget = int(config.get("training", {}).get("tokens", 1_000_000_000))
    router_health_every = max(1, int(config.get("training", {}).get("report_every", 5)))
    tokens_per_step = batch_size * context_length * grad_accum_steps

    run_path = Path(args.run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    sync_manager = PeriodicRemoteCheckpointSync(
        run_path,
        sync_interval_seconds=900.0,
        remote_dest=args.remote_sync_dest,
    )

    step, token_count = 0, 0
    resumed_phase_target_token_count: int | None = None
    ckpt_file = run_path / "latest.pt"
    if args.resume and ckpt_file.is_file():
        # Load the serialized payload on CPU first.  Mapping the entire
        # checkpoint directly to CUDA leaves a duplicate model tensor tree
        # resident beside the live model while the first resumed batch runs.
        # That duplicate is enough to turn CSA/mHC's temporary workspace into
        # an OOM on a 24-GiB card.
        payload = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if missing or unexpected:
            print(f"[Resume] Model migration: missing={missing}, unexpected={unexpected}")
        if payload.get("optimizer") and not args.reset_optimizer:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except (ValueError, RuntimeError) as exc:
                print(f"[Resume] Optimizer migration required; starting fresh optimizer state ({exc})")
        elif args.reset_optimizer:
            print("[Resume] Intentionally starting fresh optimizer state")
        step = int(payload.get("step", 0))
        token_count = int(payload.get("token_count", 0))
        resumed_data_state = payload.get("data_state")
        saved_phase_target = payload.get("phase_target_token_count")
        if saved_phase_target is not None:
            resumed_phase_target_token_count = int(saved_phase_target)
        print(f"[Resume] Resuming from step {step:,}, tokens {token_count:,}")
        del payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        resumed_data_state = None

    data_cfg = config.get("data", {})
    primary_cfg = data_cfg.get("primary", {})
    transition_cfg = data_cfg.get("after_tokens", data_cfg.get("interleave", {}))
    primary_dir = args.data_dir or str(
        primary_cfg.get("dir", "data/streamed_fineweb")
    )
    runtime_control = RuntimeControl(run_path / "runtime_control.json")
    runtime_values: dict[str, object] = {}
    runtime_lr_scale = 1.0
    runtime_warmup_tokens: int | None = None
    runtime_min_lr_ratio: float | None = None
    runtime_grad_clip = float(config.get("optimizer", {}).get("grad_clip", 1.0))
    force_checkpoint = False
    stop_after_checkpoint = False
    switch_tokens = int(transition_cfg.get("token_count", 0))

    def _phase_batch_size(phase: str) -> int:
        phase_cfg = primary_cfg if phase == "primary" else transition_cfg
        return int(phase_cfg.get("batch_size", batch_size))

    def _phase_context_length(phase: str) -> int:
        phase_cfg = primary_cfg if phase == "primary" else transition_cfg
        return int(phase_cfg.get("context_length", context_length))

    def _make_streamer(phase: str, start_tokens: int):
        phase_cfg = primary_cfg if phase == "primary" else transition_cfg
        if str(phase_cfg.get("loader", "")).lower() == "distillation":
            tokenizer_name = str(
                phase_cfg.get("tokenizer", "Qwen/Qwen3.5-0.8B-Base")
            )
            from data.tokenize import load_qwen35_tokenizer

            tokenizer = load_qwen35_tokenizer(tokenizer_name)
            state = resumed_data_state if isinstance(resumed_data_state, dict) else None
            return DistillationStreamingLoader(
                dataset_name=str(
                    phase_cfg.get(
                        "dataset_name",
                        "saidutta69/qwen-glm-kimi-distillation-clean",
                    )
                ),
                dataset_config=str(phase_cfg.get("dataset_config", "sft")),
                split=str(phase_cfg.get("split", "train")),
                tokenizer=tokenizer,
                max_length=_phase_context_length(phase),
                seed=int(phase_cfg.get("seed", config.get("seed", 1337))),
                state=state if state and state.get("loader") == "distillation" else None,
                shuffle_buffer=int(phase_cfg.get("shuffle_buffer", 10000)),
                fallback_datasets=(
                    phase_cfg.get("fallback_datasets", [])
                    if isinstance(phase_cfg.get("fallback_datasets", []), list)
                    else []
                ),
            )
        sources = phase_cfg.get("sources")
        if isinstance(sources, list) and sources:
            print(
                "[Data] Joint multi-source batching: "
                + ", ".join(str(source.get('name', '?')) for source in sources),
                flush=True,
            )
            start_batches = None
            if isinstance(resumed_data_state, dict):
                start_batches = resumed_data_state.get("batches_by_source")
            return JointMultiSourceLoader(
                sources,
                window_size=_phase_context_length(phase),
                batch_size=_phase_batch_size(phase),
                seed=int(phase_cfg.get("seed", config.get("seed", 1337))),
                start_batches=start_batches,
            )
        return ConcurrentStreamingLoader(
            str(phase_cfg.get("dir", primary_dir)),
            window_size=_phase_context_length(phase),
            batch_size=_phase_batch_size(phase),
            start_tokens=start_tokens,
            dataset_name=str(
                phase_cfg.get("dataset_name", "HuggingFaceFW/fineweb-edu")
            ),
            dataset_config=str(
                phase_cfg.get("dataset_config", "CC-MAIN-2024-10")
            ),
        )

    data_phase = (
        "after_tokens"
        if switch_tokens > 0 and token_count >= switch_tokens
        else "primary"
    )
    phase_token_budget = int(
        config.get("training", {}).get("phase_token_budget", 0)
    )
    phase_target_token_count: int | None = resumed_phase_target_token_count
    if data_phase == "after_tokens" and phase_token_budget > 0:
        if phase_target_token_count is None:
            phase_target_token_count = token_count + phase_token_budget
        token_budget = min(token_budget, phase_target_token_count)
    active_batch_size = _phase_batch_size(data_phase)
    active_context_length = _phase_context_length(data_phase)
    tokens_per_step = active_batch_size * active_context_length * grad_accum_steps
    streamer = _make_streamer(
        data_phase,
        0 if data_phase == "after_tokens" else token_count,
    )
    async_loader = AsyncCUDAPrefetcher(streamer, device=device)

    print(
        f"[Data] phase={data_phase} dir={streamer.shard_dir} "
        f"dataset={streamer.dataset_name}/{streamer.dataset_config}"
    )

    wandb_cfg = config.get("wandb", {})
    try:
        telemetry = WandbTelemetry(
            run_path / "events.jsonl",
            run_dir=run_path,
            config={**config, "parameter_report": model.parameter_report(), "optimizer_partition": partition},
            model=model,
            project=str(wandb_cfg.get("project", "hybrid-2b-looped-v0")),
            entity=wandb_cfg.get("entity"),
            run_name=wandb_cfg.get("run_name", "v0-a5000-ultimate-streaming"),
            group=wandb_cfg.get("group", "v0"),
            tags=list(wandb_cfg.get("tags", [])) or ["v0", "a5000", "streaming", "triton"],
            online_required=bool(wandb_cfg.get("online_required", False)),
            watch=bool(wandb_cfg.get("watch", False)),
        )
    except Exception as exc:
        print(f"[Telemetry] WandB unavailable ({exc}), falling back to local JsonlTelemetry")
        from train.telemetry import JsonlTelemetry
        telemetry = JsonlTelemetry(run_path / "events.jsonl")

    dashboard = TerminalDashboard(token_budget=token_budget)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(not use_bf16))

    import signal
    shutdown_requested = False
    def _sig_handler(signum, frame):
        nonlocal shutdown_requested
        print(f"\n[Shutdown] Caught signal {signum}, gracefully completing step and saving latest.pt...")
        shutdown_requested = True
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    print(f"=== [Pretraining Active] Processing {tokens_per_step:,} tokens per step (Native BF16: {use_bf16}) ===")

    while token_count < token_budget and not shutdown_requested:
        changed, new_runtime_values = runtime_control.poll()
        if changed:
            runtime_values = new_runtime_values
            runtime_lr_scale = float(runtime_values.get("lr_scale", 1.0))
            scheduler_values = runtime_values.get("scheduler", {})
            if not isinstance(scheduler_values, dict):
                scheduler_values = {}
            runtime_warmup_tokens = (
                int(scheduler_values["warmup_tokens"])
                if "warmup_tokens" in scheduler_values else None
            )
            runtime_min_lr_ratio = (
                float(scheduler_values["min_lr_ratio"])
                if "min_lr_ratio" in scheduler_values else None
            )
            runtime_grad_clip = float(runtime_values.get("grad_clip", runtime_grad_clip))
            base_lrs = runtime_values.get("base_lrs", {})
            if isinstance(base_lrs, dict):
                for group in optimizer.param_groups:
                    group_name = str(group.get("group_name", ""))
                    if group_name in base_lrs:
                        group["base_lr"] = float(base_lrs[group_name])
            adaptive_values = runtime_values.get("adaptive_loops", {})
            if isinstance(adaptive_values, dict):
                adaptive_cfg = model.config.adaptive_loops
                if "enabled" in adaptive_values:
                    adaptive_cfg.enabled = bool(adaptive_values["enabled"])
                if "learned_halting" in adaptive_values:
                    adaptive_cfg.learned_halting = bool(adaptive_values["learned_halting"])
                if "min_loops" in adaptive_values:
                    adaptive_cfg.min_loops = int(adaptive_values["min_loops"])
                if "max_loops" in adaptive_values:
                    adaptive_cfg.max_loops = min(
                        int(adaptive_values["max_loops"]), model.config.num_loops
                    )
                if "halt_penalty" in adaptive_values:
                    adaptive_cfg.halt_penalty = max(0.0, float(adaptive_values["halt_penalty"]))
                if "halt_threshold" in adaptive_values:
                    adaptive_cfg.halt_threshold = min(
                        1.0, max(0.0, float(adaptive_values["halt_threshold"]))
                    )
                if "halt_temperature" in adaptive_values:
                    adaptive_cfg.halt_temperature = max(
                        1e-4, float(adaptive_values["halt_temperature"])
                    )
                if "backprop_loops" in adaptive_values:
                    adaptive_cfg.backprop_loops = int(adaptive_values["backprop_loops"])
                if not 0 <= adaptive_cfg.backprop_loops <= model.config.num_loops:
                    raise ValueError("backprop_loops must be 0..num_loops in runtime_control.json")
                if not 1 <= adaptive_cfg.min_loops <= adaptive_cfg.max_loops:
                    raise ValueError("invalid adaptive loop range in runtime_control.json")
            if "batch_size" in runtime_values or "context_length" in runtime_values:
                print(
                    "[RuntimeControl] batch_size/context_length changes require a safe restart; "
                    "the live values were not changed.", flush=True
                )
            stop_after_checkpoint = stop_after_checkpoint or bool(
                runtime_values.get("shutdown_after_checkpoint", False)
            )
            force_checkpoint = force_checkpoint or bool(runtime_values.get("force_checkpoint", False))
            force_checkpoint = force_checkpoint or stop_after_checkpoint
            print(
                f"[RuntimeControl] applied lr_scale={runtime_lr_scale:g} "
                f"warmup_tokens={runtime_warmup_tokens} min_lr_ratio={runtime_min_lr_ratio} "
                f"grad_clip={runtime_grad_clip:g} "
                f"adaptive_loops={model.config.adaptive_loops.min_loops}-"
                f"{model.config.adaptive_loops.max_loops} "
                f"enabled={model.config.adaptive_loops.enabled} "
                f"learned={model.config.adaptive_loops.learned_halting} "
                f"penalty={model.config.adaptive_loops.halt_penalty:g} "
                f"backprop_loops={model.config.adaptive_loops.backprop_loops or model.config.num_loops}", flush=True
            )

        if (
            data_phase == "primary"
            and switch_tokens > 0
            and token_count >= switch_tokens
        ):
            streamer.stop()
            # The old CUDA prefetcher owns a GPU-resident next batch.  Release
            # it, along with the just-finished batch/graph, before constructing
            # the next phase's prefetcher or the transition can transiently
            # hold three full batches and OOM on an otherwise safe run.
            async_loader = None
            # Deliberately use explicit local deletion; mutating the dict
            # returned by locals() does not reliably clear fast locals.
            try:
                del batch
            except UnboundLocalError:
                pass
            try:
                del input_ids
            except UnboundLocalError:
                pass
            try:
                del labels
            except UnboundLocalError:
                pass
            try:
                del segments
            except UnboundLocalError:
                pass
            try:
                del reset
            except UnboundLocalError:
                pass
            try:
                del output
            except UnboundLocalError:
                pass
            try:
                del loss
            except UnboundLocalError:
                pass
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            data_phase = "after_tokens"
            if phase_token_budget > 0 and phase_target_token_count is None:
                phase_target_token_count = token_count + phase_token_budget
                token_budget = min(token_budget, phase_target_token_count)
            active_batch_size = _phase_batch_size(data_phase)
            active_context_length = _phase_context_length(data_phase)
            tokens_per_step = active_batch_size * active_context_length * grad_accum_steps
            streamer = _make_streamer(data_phase, 0)
            async_loader = AsyncCUDAPrefetcher(streamer, device=device)
            print(
                f"[Data] Switched at {token_count:,} tokens: "
                f"dir={streamer.shard_dir} "
                f"dataset={streamer.dataset_name}/{streamer.dataset_config} "
                f"batch_size={active_batch_size}",
                flush=True,
            )

        t0_step = time.time()

        accum_loss = 0.0
        loss_accum_t = torch.zeros((), device=device)
        t_data_total = 0.0
        t_fwd_total = 0.0
        t_bwd_total = 0.0
        losses_by_source: dict[str, float] = {}
        step_token_count = 0

        for _ in range(grad_accum_steps):
            _t0 = time.time()
            batch = next(async_loader)
            t_data_total += time.time() - _t0

            source_name = str(batch.get("_source_name", data_phase))
            source_names = batch.get("_source_names")
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            segments = batch["segment_ids"]
            reset = batch["reset_mask"]
            attention_mask = batch.get("attention_mask")
            if attention_mask is None:
                batch_tokens = int(input_ids.numel())
            else:
                batch_tokens = int(attention_mask.sum().item())
            step_token_count += batch_tokens

            # Most distillation rows are short enough to stay entirely on the
            # GPU.  Only offload saved activations for the rare long rows that
            # can approach the 64k context ceiling; applying save_on_cpu to
            # every row turns a memory safeguard into a global throughput cap.
            offload_this_batch = (
                resume_activation_offload
                and int(input_ids.shape[1]) >= activation_offload_min_tokens
            )
            long_sequence = int(input_ids.shape[1]) > long_sequence_loop_threshold
            loop_budget_override = (
                long_sequence and long_sequence_max_loops or None
            )
            activation_context = (
                torch.autograd.graph.save_on_cpu(pin_memory=False)
                if offload_this_batch
                else nullcontext()
            )
            with activation_context:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    _t0 = time.time()
                    output = model(
                        input_ids,
                        labels=labels,
                        num_loops=loop_budget_override,
                        segment_ids=segments,
                        reset_mask=reset,
                        batch_has_reset=batch.get("_has_reset"),
                        collect_diagnostics=False,
                        collect_router_diagnostics=((step + 1) % router_health_every == 0),
                    )
                    last_diagnostics = getattr(output, "diagnostics", {})
                    row_losses = last_diagnostics.get("row_losses")
                    if source_names and row_losses and len(source_names) == len(row_losses):
                        for s_name, r_l in zip(source_names, row_losses):
                            losses_by_source[s_name] = float(r_l)
                    loss = output.loss / grad_accum_steps
                    t_fwd_total += time.time() - _t0

            _t0 = time.time()
            if use_bf16:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            t_bwd_total += time.time() - _t0
            loss_accum_t = loss_accum_t + loss.detach()
            del loss, batch

        accum_loss = float(loss_accum_t) * grad_accum_steps
        _t0 = time.time()
        grad_norm, grad_norms = _capture_gradient_norms(optimizer)
        grad_clip = float(runtime_grad_clip)
        if use_bf16:
            for group in optimizer.param_groups:
                parameters = [parameter for parameter in group.get("params", []) if parameter.grad is not None]
                if parameters:
                    clip_grad_norm_(parameters, grad_clip)
            optimizer.step()
        else:
            scaler.unscale_(optimizer)
            for group in optimizer.param_groups:
                parameters = [parameter for parameter in group.get("params", []) if parameter.grad is not None]
                if parameters:
                    clip_grad_norm_(parameters, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        t_opt_total = time.time() - _t0

        step += 1
        # Distillation rows are variable-length.  Count the actual non-padding
        # tokens so scheduler progress, checkpoints, and W&B throughput remain
        # correct instead of pretending every row is a full 65,536 tokens.
        effective_step_tokens = step_token_count if step_token_count > 0 else tokens_per_step
        token_count += effective_step_tokens
        step_dt = max(time.time() - t0_step, 1e-6)
        if (step % 5 == 0):
            print(f"[Timing] Step {step}: total={step_dt*1000:.0f}ms (data={t_data_total*1000:.0f}ms, fwd={t_fwd_total*1000:.0f}ms, bwd={t_bwd_total*1000:.0f}ms, opt={t_opt_total*1000:.0f}ms)", flush=True)

        warmup_tokens = max(
            1,
            runtime_warmup_tokens
            if runtime_warmup_tokens is not None
            else int(token_budget * float(config.get("schedule", {}).get("warmup_token_fraction", 0.02))),
        )
        min_lr_ratio = (
            runtime_min_lr_ratio
            if runtime_min_lr_ratio is not None
            else float(config.get("schedule", {}).get("min_lr_ratio", 0.1))
        )
        if token_count < warmup_tokens:
            scale = token_count / warmup_tokens
        else:
            progress = min(1.0, (token_count - warmup_tokens) / max(1, token_budget - warmup_tokens))
            scale = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()

        for group in optimizer.param_groups:
            group["lr"] = (
                float(group.get("base_lr", optim_cfg.get("mud_lr", 1e-3)))
                * scale * runtime_lr_scale
            )

        curr_lr = optimizer.param_groups[0]["lr"]

        telemetry.log_step(
            step=step, token_count=token_count, loss=accum_loss,
            diagnostics=last_diagnostics, optimizer=optimizer,
            device=device, batch_tokens=effective_step_tokens, model=model,
            grad_norm=grad_norm, grad_norms=grad_norms,
            source_name=source_name,
            losses_by_source=losses_by_source if losses_by_source else None,
        )
        try:
            del output
        except UnboundLocalError:
            pass


        checkpoint_every = int(config.get("training", {}).get("checkpoint_every", 0))
        checkpoint_due = checkpoint_every > 0 and step % checkpoint_every == 0
        if force_checkpoint or sync_manager.should_sync() or checkpoint_due:
            sync_manager.trigger_async_sync(
                model=model, optimizer=optimizer, step=step,
                token_count=token_count, telemetry=last_diagnostics if 'last_diagnostics' in locals() else {},
                data_state=(streamer.state_dict() if hasattr(streamer, "state_dict") else None),
                config=config,
            )
            force_checkpoint = False
            if stop_after_checkpoint:
                shutdown_requested = True
                stop_after_checkpoint = False

        dashboard.log_step(
            step=step,
            token_count=token_count,
            step_tokens=effective_step_tokens,
            loss=accum_loss,
            lr=curr_lr,
            step_dt=step_dt,
            device=device,
            shards_downloaded=streamer.total_downloaded_shards,
            sync_status=sync_manager.last_sync_status,
            seconds_to_sync=sync_manager.seconds_until_next_sync(),
        )

    streamer.stop()
    print(f"[Shutdown] Saving final checkpoint at step {step:,}, tokens {token_count:,}...")
    torch.save({
        "format_version": 2,
        "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "step": int(step),
            "token_count": int(token_count),
            "phase_target_token_count": phase_target_token_count,
            "timestamp": time.time(),
        "telemetry": last_diagnostics if 'last_diagnostics' in locals() else {},
        "data_state": streamer.state_dict() if hasattr(streamer, "state_dict") else None,
        "config": config,
    }, run_path / "latest.pt")
    print("[Shutdown] Saved latest.pt successfully.")
    if token_count >= token_budget:
        print("=== [Completed] 1 Billion Tokens Pretraining Finished! ===")
    else:
        print(f"=== [Paused] Checkpointed at {token_count:,} tokens for restart. ===")



if __name__ == "__main__":
    main()
