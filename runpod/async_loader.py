"""Asynchronous double-buffered CUDA stream prefetcher."""
from __future__ import annotations
import torch

class AsyncCUDAPrefetcher:
    """Overlaps PCIe host-to-device batch transfers with GPU backward pass."""
    def __init__(self, loader, device: torch.device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        def _to_device(tensor: torch.Tensor) -> torch.Tensor:
            return tensor if tensor.is_cuda else tensor.to(self.device, non_blocking=True)

        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: _to_device(v) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self._preload()
        return batch

    def __iter__(self):
        return self
