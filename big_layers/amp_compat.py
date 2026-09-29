"""Compatibility helpers for custom autograd functions across PyTorch AMP APIs."""

import torch

try:
    from torch.amp import custom_bwd as _custom_bwd
    from torch.amp import custom_fwd as _custom_fwd

    _USES_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import custom_bwd as _custom_bwd
    from torch.cuda.amp import custom_fwd as _custom_fwd

    _USES_DEVICE_TYPE = False


def custom_fwd(func=None, *, device_type="cuda", cast_inputs=None):
    """Return the custom_fwd decorator supported by the installed PyTorch version."""
    if _USES_DEVICE_TYPE:
        decorator = _custom_fwd(device_type=device_type, cast_inputs=cast_inputs)
    else:
        decorator = _custom_fwd(cast_inputs=cast_inputs)
    return decorator(func) if func is not None else decorator


def custom_bwd(func=None, *, device_type="cuda"):
    """Return the custom_bwd decorator supported by the installed PyTorch version."""
    if _USES_DEVICE_TYPE:
        decorator = _custom_bwd(device_type=device_type)
    else:
        decorator = _custom_bwd
    return decorator(func) if func is not None else decorator


def autocast_enabled(device_type="cuda"):
    """Check autocast state without assuming the newest torch.is_autocast_enabled signature."""
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:
        return torch.is_autocast_enabled()


def autocast_dtype(device_type="cuda"):
    """Get the current autocast dtype using the API available in this PyTorch version."""
    if hasattr(torch, "get_autocast_dtype"):
        return torch.get_autocast_dtype(device_type)
    if device_type == "cuda" and hasattr(torch, "get_autocast_gpu_dtype"):
        return torch.get_autocast_gpu_dtype()
    raise RuntimeError(f"Autocast dtype query is unavailable for device type {device_type!r}.")
