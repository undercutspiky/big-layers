"""Execution settings for the layer-wise implementation; no GPU allocation happens at import."""
import torch

cuda_device = torch.device('cuda', 0)
host_device = torch.device('cpu')
amp_enabled = True
_num_streams = 4
_streams = {}


def configure(device='cuda:0', use_amp=True, num_streams=4):
    """Select one compute device before constructing the model. CPU is useful for small checks."""
    global cuda_device, amp_enabled, _num_streams
    cuda_device = torch.device(device)
    if cuda_device.type not in ('cuda', 'cpu'):
        raise ValueError('Big Layers supports CUDA execution and small CPU checks, not this device type.')
    if cuda_device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install a CUDA-enabled PyTorch build or select device=cpu.')
    if num_streams < 1:
        raise ValueError('num_streams must be positive.')
    amp_enabled = bool(use_amp and cuda_device.type == 'cuda')
    _num_streams = int(num_streams)


def get_cuda_streams(device=None):
    """Lazily create streams on the requested compute device, never on an assumed second GPU."""
    device = torch.device(device or cuda_device)
    if device.type != 'cuda':
        return []
    key = (device.index if device.index is not None else torch.cuda.current_device(), _num_streams)
    if key not in _streams:
        _streams[key] = [torch.cuda.Stream(device=device) for _ in range(_num_streams)]
    return _streams[key]


def host_buffer(buffer, shape, dtype, pin_memory):
    """Reuse a channels-last host allocation when it fits; return storage without autograd history."""
    shape = tuple(shape)
    if buffer is None or buffer.dtype != dtype or any(a < b for a, b in zip(buffer.shape, shape)):
        buffer = torch.empty(shape, dtype=dtype, device='cpu', pin_memory=pin_memory,
                             memory_format=torch.channels_last)
    view = buffer.detach()[tuple(slice(0, size) for size in shape)]
    return buffer, view
