"""Real-time Multi-Metric Terminal Dashboard for Pretraining."""
from __future__ import annotations

import time
import torch

class TerminalDashboard:
    """Prints training telemetry to stdout."""

    def __init__(self, token_budget: int = 1_000_000_000) -> None:
        self.token_budget = token_budget
        self.t_start = time.time()
        self.step_times = []
        self.window = 50
        self._last_vram_allocated_mb = 0.0
        self._last_vram_reserved_mb = 0.0

    def log_step(
        self,
        *,
        step: int,
        token_count: int,
        step_tokens: int,
        loss: float,
        lr: float,
        step_dt: float,
        device: torch.device,
        shards_downloaded: int,
        sync_status: str,
        seconds_to_sync: float,
    ) -> None:
        self.step_times.append(step_dt)
        if len(self.step_times) > self.window:
            self.step_times.pop(0)

        rolling_tps = (len(self.step_times) * step_tokens) / sum(self.step_times)
        instant_tps = step_tokens / max(step_dt, 1e-6)

        pct = (token_count / self.token_budget) * 100.0
        remaining_tokens = max(0, self.token_budget - token_count)
        eta_seconds = remaining_tokens / max(rolling_tps, 1.0)
        eta_str = self._format_duration(eta_seconds)
        elapsed_str = self._format_duration(time.time() - self.t_start)

        if device.type == "cuda" and (step % 5 == 0 or not self._last_vram_allocated_mb):
            self._last_vram_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            self._last_vram_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
        vram_allocated_mb = self._last_vram_allocated_mb
        vram_reserved_mb = self._last_vram_reserved_mb
        sync_min = int(seconds_to_sync // 60)
        sync_sec = int(seconds_to_sync % 60)

        print(
            f"[Step {step:6d}] {token_count:10,d}/{self.token_budget:,} ({pct:5.2f}%) "
            f"loss={loss:7.4f} lr={lr:.2e} speed={instant_tps:6.1f} "
            f"avg={rolling_tps:6.1f} tok/s dt={step_dt*1000:5.1f}ms "
            f"vram={vram_allocated_mb:,.0f}/{vram_reserved_mb:,.0f}MB "
            f"shards={shards_downloaded} sync={sync_status} "
            f"next={sync_min:02d}m {sync_sec:02d}s elapsed={elapsed_str} eta={eta_str}",
            flush=True,
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        s = int(seconds)
        days = s // 86400
        hours = (s % 86400) // 3600
        mins = (s % 3600) // 60
        secs = s % 60
        if days > 0:
            return f"{days}d {hours:02d}h {mins:02d}m"
        elif hours > 0:
            return f"{hours}h {mins:02d}m"
        else:
            return f"{mins:02d}m {secs:02d}s"
