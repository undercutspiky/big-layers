"""ResNet-18 using Big Layers for the memory-heavy stem and optional early residual stages."""
import torch
from torch import nn

from . import cuda_config as cc
from .big_conv import BigConv2d
from .big_batch_norm import BigBatchNorm


class Conv2d(nn.Conv2d):
    """A native convolution marks the boundary between host-resident and device-resident stages."""
    def forward(self, x):
        return super().forward(x.to(self.weight.device, non_blocking=True))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, big=False, conv_elements=60_000_000, bn_elements=6_000_000):
        super().__init__()
        conv = BigConv2d if big else Conv2d
        norm = BigBatchNorm if big else nn.BatchNorm2d
        conv_kw = {'max_elements': conv_elements} if big else {'bias': False}
        norm_kw = {'max_elements': bn_elements} if big else {}
        self.conv1 = conv(inplanes, planes, 3, stride=stride, padding=1, **conv_kw)
        self.bn1 = norm(planes, **norm_kw)
        self.relu1, self.relu2 = nn.ReLU(), nn.ReLU()
        self.conv2 = conv(planes, planes, 3, padding=1, **conv_kw)
        self.bn2 = norm(planes, **norm_kw)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(conv(inplanes, planes, 1, stride=stride, **conv_kw),
                                            norm(planes, **norm_kw))
        self.stride = stride

    def forward(self, x):
        """Run the two convolutions and the unchanged residual branch; align devices only at addition."""
        identity = x
        out = self.bn1(self.conv1(x))
        out = self.relu1(out if out.is_cuda else out.float())
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        device = out.device if out.is_cuda else identity.device
        out = out.to(device) + identity.to(device)
        return self.relu2(out if out.is_cuda else out.float())


class BigResNet(nn.Module):
    """ResNet-18 with a partitioned stem and optionally partitioned early residual stages.

    num_big_stages=0 partitions the stem only; 1..4 extends partitioning through those stages.
    No layers are removed, frozen, widened or introduced progressively during training.
    """
    def __init__(self, num_big_stages=0, pooling='avg', conv_elements=60_000_000, bn_elements=6_000_000):
        super().__init__()
        if num_big_stages not in range(5) or pooling not in ('avg', 'flatten'):
            raise ValueError('Choose num_big_stages in 0..4 and pooling="avg" or "flatten".')
        self.pooling = pooling
        self.conv1 = BigConv2d(3, 64, 7, stride=2, padding=0, max_elements=conv_elements)
        self.bn1 = BigBatchNorm(64, max_elements=bn_elements)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=0)
        inplanes = 64
        for index, planes in enumerate((64, 128, 256, 512)):
            stride = 1 if index == 0 else 2
            blocks = [BasicBlock(inplanes, planes, stride, index < num_big_stages, conv_elements, bn_elements),
                      BasicBlock(planes, planes, 1, index < num_big_stages, conv_elements, bn_elements)]
            setattr(self, f'layer{index + 1}', nn.Sequential(*blocks))
            inplanes = planes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, BigConv2d)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, (nn.BatchNorm2d, BigBatchNorm)):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.to(cc.cuda_device)

    def forward_features(self, x):
        """Return layer4's spatial feature map; early cheap operations remain on the host."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x)).float()))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, x):
        x = self.forward_features(x)
        if self.pooling == 'avg':
            x = self.avgpool(x)
        return x.flatten(1)

    def load_imagenet_weights(self, weights=None):
        """Load ImageNet ResNet-18 parameters, omitting only the unused ImageNet classifier."""
        if weights is None:
            from torchvision.models import ResNet18_Weights
            state = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
        else:
            state = torch.load(weights, map_location='cpu', weights_only=True)
        state = {key: value for key, value in state.items() if not key.startswith('fc.')}
        self.load_state_dict(state, strict=True)
        return self


def resnet18(pretrained=True, weights=None, **kwargs):
    model = BigResNet(**kwargs)
    return model.load_imagenet_weights(weights) if pretrained or weights is not None else model
