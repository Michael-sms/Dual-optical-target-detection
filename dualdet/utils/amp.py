"""AMP helpers compatible with PyTorch 2.x."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch


def autocast_context(device: torch.device, *, enabled: bool) -> AbstractContextManager[object]:
    """Return a device-aware autocast context manager."""

    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def create_grad_scaler(device: torch.device, *, enabled: bool) -> torch.cuda.amp.GradScaler:
    """Create a CUDA grad scaler when AMP is enabled."""

    return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")
