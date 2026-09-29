"""Execution settings and cache registries for the experimental hot-potato path."""
import torch

cuda_device = torch.device('cuda:0')
host_device = torch.device('cpu')
amp_enabled = True
GPU_CACHE_REGISTRY = {}
BWD_ACCELERATOR_REGISTRY = {}
_streams = None


def configure(device='cuda:0', use_amp=True):
    global cuda_device, amp_enabled, _streams
    cuda_device = torch.device(device)
    amp_enabled = use_amp
    _streams = None


def get_cuda_streams():
    global _streams
    if _streams is None:
        _streams = [torch.cuda.Stream(device=cuda_device) for _ in range(4)]
    return _streams


def clear_registries():
    GPU_CACHE_REGISTRY.clear()
    BWD_ACCELERATOR_REGISTRY.clear()
