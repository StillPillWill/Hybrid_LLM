"""Structured run telemetry with explicit architecture health metrics."""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


_NVIDIA_SMI_CACHE: dict[tuple[int, str], tuple[float, float | None]] = {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            return str(value)
    return value


class JsonlTelemetry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path = self.path.with_name("status.json")

    def emit(self, event: str, **fields: Any) -> None:
        payload = {"time": time.time(), "event": event, **fields}
        safe_payload = _json_safe(payload)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe_payload, sort_keys=True, default=str) + "\n")
        status = {
            "time": safe_payload["time"], "event": event,
            "step": safe_payload.get("step"), "token_count": safe_payload.get("token_count"),
            "loss": safe_payload.get("loss"), "status": safe_payload.get("status"),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.status_path.parent,
            prefix="status-", suffix=".tmp", delete=False,
        ) as handle:
            handle.write(json.dumps(status, sort_keys=True) + "\n")
            temporary = Path(handle.name)
        os.replace(temporary, self.status_path)

    def heartbeat(self, *, step: int, token_count: int, model: Any,
                  resource_reader: Callable[[], dict[str, Any]] | None = None) -> None:
        diagnostics: dict[str, Any] = {}
        if hasattr(model, "parameter_report"):
            diagnostics["parameter_report"] = model.parameter_report()
        resources = resource_reader() if resource_reader is not None else {}
        self.emit("heartbeat", step=step, token_count=token_count,
                  diagnostics=diagnostics, resources=resources, pid=os.getpid())

    def health(self, *, step: int, diagnostics: dict[str, Any]) -> None:
        self.emit("architecture_health", step=step, diagnostics=diagnostics)

    def log_step(self, *, step: int, token_count: int, loss: Any,
                 diagnostics: dict[str, Any], optimizer: Any = None,
                 device: Any = None, batch_tokens: int = 0, model: Any = None,
                 grad_norm: float | None = None,
                 grad_norms: dict[str, float] | None = None,
                 source_name: str | None = None) -> None:
        now = time.time()
        elapsed = max(now - getattr(self, "started_at", now), 1e-6)
        metrics: dict[str, Any] = {
            "training/step": int(step),
            "training/token_count": int(token_count),
            "training/loss": float(loss),
            "training/tokens_per_second_total": float(token_count / elapsed),
            "training/tokens_per_second_step": float(batch_tokens / max(now - getattr(self, "last_step_time", now), 1e-6)),
            "diagnostics": diagnostics,
        }
        metrics.update(optimizer_scalars(optimizer, grad_norms=grad_norms) if optimizer is not None else {})
        if grad_norm is not None:
            metrics["optimizer/grad_norm"] = float(grad_norm)
        if source_name:
            safe_source = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in str(source_name)
            )
            metrics[f"training/loss_by_source/{safe_source}"] = float(loss)
        self.emit("optimizer_step", **metrics)
        self.last_step_time = now


def flatten_scalars(value: Any, *, prefix: str = "") -> dict[str, float | int | bool | str]:
    """Flatten nested diagnostics into W&B-safe scalar fields."""
    result: dict[str, float | int | bool | str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            result.update(flatten_scalars(item, prefix=child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(flatten_scalars(item, prefix=f"{prefix}/{index}"))
    elif isinstance(value, (int, float, bool, str)):
        if isinstance(value, float) and not math.isfinite(value):
            return result
        result[prefix] = value
    elif hasattr(value, "ndim") and hasattr(value, "numel"):
        # Preserve scalar tensors but never upload an entire activation or
        # routing tensor as a history field.
        if int(value.numel()) == 1:
            scalar = value.detach().float().item()
            if math.isfinite(scalar):
                result[prefix] = scalar
    return result


def resource_snapshot(device: Any = None) -> dict[str, float | int | str | bool]:
    """Collect host/PyTorch/GPU resource values without requiring a GPU."""
    result: dict[str, float | int | str | bool] = {"pid": os.getpid()}
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory = psutil.virtual_memory()
        result.update({
            "host_ram_used_bytes": int(memory.used),
            "host_ram_available_bytes": int(memory.available),
            "host_ram_fraction": float(memory.percent / 100.0),
            "process_rss_bytes": int(process.memory_info().rss),
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
        })
    except Exception:
        pass
    try:
        import torch

        if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
            index = torch.device(device).index or torch.cuda.current_device()
            result.update({
                "cuda_device": int(index),
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(index)),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
                "cuda_temperature_c": _nvidia_smi_value(index, "temperature.gpu"),
                "cuda_utilization_percent": _nvidia_smi_value(index, "utilization.gpu"),
            })
    except Exception:
        pass
    return result


def _nvidia_smi_value(index: int, field: str) -> float | None:
    cache_key = (int(index), str(field))
    now = time.monotonic()
    cached = _NVIDIA_SMI_CACHE.get(cache_key)
    if cached is not None and now - cached[0] < 5.0:
        return cached[1]
    value: float | None = None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--id", str(index), "--query-gpu=" + field, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if completed.returncode == 0:
            value = float(completed.stdout.strip().splitlines()[0])
    except (FileNotFoundError, OSError, ValueError, subprocess.TimeoutExpired):
        value = None
    _NVIDIA_SMI_CACHE[cache_key] = (now, value)
    return value


def optimizer_scalars(optimizer: Any, *, grad_norms: dict[str, float] | None = None) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for index, group in enumerate(getattr(optimizer, "param_groups", [])):
        name = str(group.get("group_name", index))
        if grad_norms is None:
            gradients = [parameter.grad.detach() for parameter in group.get("params", []) if parameter.grad is not None]
            grad_norm = math.sqrt(sum(float(gradient.float().pow(2).sum()) for gradient in gradients)) if gradients else 0.0
        else:
            grad_norm = float(grad_norms.get(name, 0.0))
        result[f"optimizer/{name}/lr"] = float(group.get("lr", 0.0))
        result[f"optimizer/{name}/parameters"] = int(sum(parameter.numel() for parameter in group.get("params", [])))
        result[f"optimizer/{name}/grad_norm"] = grad_norm
    return result


def state_scalars(state: Any) -> dict[str, float | int]:
    """Summarize recurrent state without serializing activations to W&B."""
    result: dict[str, float | int] = {}
    streams = getattr(state, "streams", None)
    if streams is not None:
        value = streams.detach().float()
        result["state/streams_rms"] = float(value.pow(2).mean().sqrt())
        centered = value - value.mean(dim=-2, keepdim=True)
        result["state/streams_distinctness_rms"] = float(centered.pow(2).mean().sqrt())
        result["state/stream_count"] = int(streams.shape[-2])
    for name, tensor in getattr(state, "gdn_states", {}).items():
        result[f"state/gdn/{name}/rms"] = float(tensor.detach().float().pow(2).mean().sqrt())
    for family in ("csa_memories", "hca_memories"):
        for name, memory in getattr(state, family, {}).items():
            keys = getattr(memory, "keys", None)
            result[f"state/{family}/{name}/entries"] = int(keys.shape[1]) if keys is not None else 0
    result["state/loop_endpoint_count"] = int(len(getattr(state, "loop_endpoints", [])))
    return result


class WandbTelemetry(JsonlTelemetry):
    """Required online W&B logger with a local lossless event mirror."""

    _RECURSION_DIAGNOSTICS = (
        "effective_loop_count",
        "executed_loop_count",
        "num_loops",
    )

    def __init__(self, path: str | Path, *, run_dir: str | Path, config: dict[str, Any],
                 model: Any = None, project: str = "hybrid-2b-looped-v0",
                 entity: str | None = None, run_name: str | None = None,
                 group: str | None = None, tags: list[str] | None = None,
                 online_required: bool = True, watch: bool = False) -> None:
        super().__init__(path)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.started_at = time.time()
        self.last_step_time = self.started_at
        self.last_heartbeat = 0.0
        self.pending_error: str | None = None
        self._wandb = None
        self._source_loss_history: dict[str, list[tuple[int, float]]] = {}
        wandb_cfg = config.get("wandb", {}) if isinstance(config, dict) else {}
        self._source_chart_every = max(
            1, int(wandb_cfg.get("source_loss_chart_every", 100))
        )
        self._source_chart_points = max(
            10, int(wandb_cfg.get("source_loss_chart_points", 1000))
        )
        self._last_source_chart_step = -1
        self._restore_source_loss_history()
        try:
            import wandb
        except Exception as exc:
            if online_required:
                raise RuntimeError("W&B is required for V0 but the wandb package is unavailable") from exc
            wandb = None
        self.run = None
        if wandb is not None:
            self._wandb = wandb
            mode = "online" if online_required else os.environ.get("WANDB_MODE", "offline")
            os.environ["WANDB_MODE"] = mode
            os.environ["WANDB_SILENT"] = "true"
            flat_config = flatten_scalars(_json_safe(config))
            flat_config["execution/pid"] = os.getpid()
            flat_config["execution/python"] = os.sys.executable
            flat_config["execution/started_unix"] = self.started_at
            # Matching W&B names alone creates a new run on every restart.
            # Reuse the persisted id so CUDA debloat restarts remain one run.
            existing_run_id = None
            run_record_path = self.run_dir / "wandb_run.json"
            try:
                if run_record_path.is_file():
                    record = json.loads(run_record_path.read_text(encoding="utf-8"))
                    existing_run_id = record.get("id") or None
            except (OSError, ValueError, TypeError):
                existing_run_id = None
            resume_mode = "must" if existing_run_id else "allow"
            try:
                self.run = wandb.init(
                    project=project, entity=entity, name=run_name,
                    group=group, job_type="pretrain", mode=mode,
                    config=flat_config, tags=tags or ["v0", "looped-hybrid-moe", "qwen", "resumable"],
                    id=existing_run_id, resume=resume_mode,
                    reinit="finish_previous", save_code=False,
                )
            except Exception as exc:
                if online_required:
                    raise
                print(f"[Telemetry] WandB offline/fallback activated: {exc}")
                self.run = None
            if online_required and (self.run is None or bool(getattr(self.run, "offline", False))):
                raise RuntimeError("W&B did not create an online run; refusing to train without telemetry")
            if self.run is not None:
                self.run.define_metric("training/step")
                self.run.define_metric("*", step_metric="training/step")
                self.run.define_metric("training/recursions/*", step_metric="training/step")
                # These legacy scalar keys created one auto-panel per source.
                # Keep any historical values queryable, but hide those panels;
                # the single custom chart below is the canonical view.
                self.run.define_metric("training/loss_by_source/*", step_metric="training/step")
                self.run.define_metric("training/loss_ema_by_source/*", step_metric="training/step")
                summary_scalars = flatten_scalars(
                    {"model": config.get("parameter_report", {}),
                     "optimizer_partition": config.get("optimizer_partition", {})}
                )
                self.run.summary.update(summary_scalars)
                if watch and model is not None:
                    self.run.watch(model, log="all", log_freq=100, log_graph=False)
        record = {
            "id": getattr(self.run, "id", None), "url": str(getattr(self.run, "url", "")).split("?", 1)[0],
            "project": project, "entity": entity, "name": run_name,
            "group": group, "online": self.run is not None and not bool(getattr(self.run, "offline", False)),
        }
        (self.run_dir / "wandb_run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        with (self.run_dir / "wandb_runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.emit("wandb_started", wandb=record)

    def emit(self, event: str, **fields: Any) -> None:
        super().emit(event, **fields)
        remote = flatten_scalars(fields, prefix=f"events/{event}")
        if self.run is not None and remote:
            self._remote_log(remote)

    def _restore_source_loss_history(self) -> None:
        """Restore only the current W&B run's raw source points for the chart."""
        if not self.path.is_file():
            return
        current_run_start = -1
        records: list[dict[str, Any]] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if record.get("event") == "wandb_started":
                        current_run_start = len(records)
                    records.append(record)
        except OSError:
            return
        for record in records[current_run_start + 1:]:
            if record.get("event") != "optimizer_step":
                continue
            step = record.get("step")
            if not isinstance(step, int):
                continue
            for key, value in record.items():
                prefix = "training/loss_by_source/"
                if not key.startswith(prefix) or not isinstance(value, (int, float)):
                    continue
                source = key[len(prefix):]
                history = self._source_loss_history.setdefault(source, [])
                history.append((step, float(value)))
        for source, history in self._source_loss_history.items():
            self._source_loss_history[source] = history[-self._source_chart_points:]

    def _record_source_loss(self, *, step: int, source_name: str, loss: float) -> None:
        history = self._source_loss_history.setdefault(source_name, [])
        history.append((int(step), float(loss)))
        del history[:-self._source_chart_points]

    def _log_source_loss_chart(self, *, step: int) -> None:
        """Log one multi-line chart instead of one W&B panel per source."""
        if (
            self.run is None
            or self._wandb is None
            or not self._source_loss_history
            or step % self._source_chart_every != 0
            or step == self._last_source_chart_step
        ):
            return
        series = [
            (source, points)
            for source, points in sorted(self._source_loss_history.items())
            if points
        ]
        if not series:
            return
        xs = [[point[0] for point in points] for _, points in series]
        ys = [[point[1] for point in points] for _, points in series]
        keys = [source for source, _ in series]
        try:
            chart = self._wandb.plot.line_series(
                xs,
                ys,
                keys=keys,
                title="Raw loss by data source",
                xname="training step",
                split_table=False,
            )
            self.run.log(
                {"training/source_loss_curves": chart, "training/step": int(step)},
                commit=True,
            )
            self._last_source_chart_step = int(step)
        except Exception as exc:
            # The scalar training telemetry must remain reliable even if a
            # custom-chart upload is rejected by a W&B client/server version.
            print(f"[Telemetry] source-loss chart upload skipped: {exc}", flush=True)

    def log_step(self, *, step: int, token_count: int, loss: Any,
                 diagnostics: dict[str, Any], optimizer: Any = None,
                 device: Any = None, batch_tokens: int = 0, model: Any = None,
                 grad_norm: float | None = None,
                 grad_norms: dict[str, float] | None = None,
                 source_name: str | None = None,
                 losses_by_source: dict[str, float] | None = None) -> None:
        now = time.time()
        elapsed = max(now - self.started_at, 1e-6)
        metrics: dict[str, Any] = {
            "training/step": int(step), "training/token_count": int(token_count),
            "training/loss": float(loss), "training/tokens_per_second_total": float(token_count / elapsed),
            "training/tokens_per_second_step": float(batch_tokens / max(now - getattr(self, "last_step_time", now), 1e-6)),
            "resources": resource_snapshot(device) if int(step) % 50 == 0 else {},
            "diagnostics": diagnostics,
        }
        metrics.update(optimizer_scalars(optimizer, grad_norms=grad_norms) if optimizer is not None else {})
        if grad_norm is not None:
            metrics["optimizer/grad_norm"] = float(grad_norm)
        for diagnostic_name, metric_name in (
            ("effective_loop_count", "effective"),
            ("executed_loop_count", "executed"),
            ("num_loops", "max"),
        ):
            value = diagnostics.get(diagnostic_name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[f"training/recursions/{metric_name}"] = float(value)
        if losses_by_source:
            for s_name, s_loss in losses_by_source.items():
                safe_source = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in str(s_name)
                )
                metrics[f"training/loss_by_source/{safe_source}"] = float(s_loss)
                self._record_source_loss(step=step, source_name=safe_source, loss=float(s_loss))
        elif source_name:
            safe_source = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in str(source_name)
            )
            metrics[f"training/loss_by_source/{safe_source}"] = float(loss)
            self._record_source_loss(step=step, source_name=safe_source, loss=float(loss))
        local_metrics = dict(metrics)
        JsonlTelemetry.emit(self, "optimizer_step", **local_metrics)
        self.last_step_time = now
        if self.run is not None:
            remote_metrics = dict(metrics)
            remote_diagnostics = dict(diagnostics)
            for diagnostic_name in self._RECURSION_DIAGNOSTICS:
                remote_diagnostics.pop(diagnostic_name, None)
            remote_metrics["diagnostics"] = remote_diagnostics
            remote = flatten_scalars(remote_metrics)
            self._remote_log(remote, step=step)
            self._log_source_loss_chart(step=step)
        if now - self.last_heartbeat >= float(self.config.get("wandb", {}).get("heartbeat_seconds", 5)):
            self.heartbeat(step=step, token_count=token_count, model=model, device=device)

    def heartbeat(self, *, step: int, token_count: int, model: Any = None,
                  resource_reader: Callable[[], dict[str, Any]] | None = None,
                  device: Any = None) -> None:
        resources = resource_reader() if resource_reader is not None else resource_snapshot(device)
        diagnostics: dict[str, Any] = {}
        if model is not None and hasattr(model, "parameter_report"):
            diagnostics["parameter_report"] = model.parameter_report()
        self.last_heartbeat = time.time()
        self.emit("heartbeat", step=step, token_count=token_count,
                  diagnostics=diagnostics, resources=resources)

    def checkpoint(self, *, step: int, token_count: int, path: str | Path) -> None:
        self.emit("checkpoint", step=step, token_count=token_count, path=str(path), checkpoint_saved=True)

    def evaluation(self, *, split: str, metrics: dict[str, Any], step: int | None = None) -> None:
        values = {f"evaluation/{split}/{key}": value for key, value in metrics.items()}
        JsonlTelemetry.emit(self, "evaluation", split=split, step=step, metrics=metrics)
        if self.run is not None:
            self._remote_log(flatten_scalars(values), step=step)

    def _remote_log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        try:
            if step is None:
                self.run.log(metrics, commit=True)
            else:
                # Keep the logical step in a named metric; W&B's internal step
                # remains monotonic even when a heartbeat races a train step.
                self.run.log(metrics, commit=True)
        except Exception as exc:
            self.pending_error = str(exc)
            raise RuntimeError(f"W&B telemetry upload failed at step {step}: {exc}") from exc

    def finish(self, *, status: str = "completed", error: BaseException | None = None) -> None:
        summary = {"execution/status": status, "execution/finished_unix": time.time(),
                   "execution/elapsed_seconds": time.time() - self.started_at,
                   "execution/telemetry_pending": self.pending_error is not None}
        if error is not None:
            summary["execution/error"] = repr(error)
        fields: dict[str, Any] = {"status": status, "summary": summary}
        if error is not None:
            fields["error"] = repr(error)
        self.emit("run_finished", **fields)
        if self.run is not None:
            self.run.summary.update(summary)
            self.run.finish(exit_code=0 if status == "completed" else 1)
