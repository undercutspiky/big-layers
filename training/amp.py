"""Small compatibility layer for mixed precision across PyTorch releases."""

from contextlib import nullcontext

import torch


def autocast_context(enabled, dtype, device_type="cuda"):
    """Return an autocast context without tying the training code to one AMP namespace."""
    if not enabled or device_type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


def make_grad_scaler(enabled):
    """Construct the CUDA GradScaler using the API available in the installed PyTorch version."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)
