"""Layer-wise, host-offloaded layers used by the paper's CNN training path."""
from .big_conv import BigConv2d
from .big_batch_norm import BigBatchNorm
from .resnet import BigResNet, resnet18

__all__ = ['BigConv2d', 'BigBatchNorm', 'BigResNet', 'resnet18']
