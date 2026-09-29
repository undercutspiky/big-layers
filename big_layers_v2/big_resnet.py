"""EXPERIMENTAL: archived hot-potato path, not used by the paper training entry points."""
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models.resnet import Bottleneck, BasicBlock

from . import cuda_config as cc
from .big_resnet_components_BN_fused import BigBottleneck, BigBlockJunctionFunction, BigFusedBNReLU, BigFusedBNAddReLU
from .big_resnet_components_fused import BigConv2dStats, BigFusedBNReLUMaxPool, BigDS


def load_pretrained_big_resnet(big_model, model_type='resnet18'):
    """
    Downloads standard torchvision weights and initializes a BigResNet
    instance with them using procedural mapping.
    """
    print(f"[*] Downloading pre-trained {model_type} weights...")

    # 1. Initialize temporary reference model with pre-trained weights
    # Note: Using 'weights' instead of 'pretrained' to follow modern torchvision API
    weights_enum = getattr(tv_models, f"{model_type.capitalize()}_Weights")
    ref_model = getattr(tv_models, model_type)(weights=weights_enum.DEFAULT).to(cc.cuda_device).eval()

    # 2. Use your existing sync logic to move weights
    print(f"[*] Mapping weights to BigResNet architecture...")
    sync_weights(big_model, ref_model)

    # 3. Cleanup
    del ref_model
    torch.cuda.empty_cache()
    print(f"[SUCCESS] BigResNet initialized with pre-trained {model_type} weights.")


def sync_weights(big_model, ref_model):
    """Deep synchronization including all weights, BN running stats, and FC layer."""
    with torch.no_grad():
        # 1. Sync Stem
        ref_model.conv1.weight.copy_(big_model.conv1.weight)
        ref_model.bn1.weight.copy_(big_model.bn_pool.weight)
        ref_model.bn1.bias.copy_(big_model.bn_pool.bias)
        ref_model.bn1.running_mean.copy_(big_model.bn_pool.running_mean)
        ref_model.bn1.running_var.copy_(big_model.bn_pool.running_var)

        # 2. Sync Stages
        for s_idx in range(4):
            big_stage = big_model.stages[s_idx]
            ref_stage = getattr(ref_model, f"layer{s_idx + 1}")

            for bb, rb in zip(big_stage, ref_stage):
                # Standard attributes common to all blocks
                rb.conv1.weight.copy_(bb.conv1.weight)
                rb.bn1.weight.copy_(bb.bn1.weight)
                rb.bn1.bias.copy_(bb.bn1.bias)
                rb.bn1.running_mean.copy_(bb.bn1.running_mean)
                rb.bn1.running_var.copy_(bb.bn1.running_var)
                rb.conv2.weight.copy_(bb.conv2.weight)

                # Handle Big vs Standard and Basic vs Bottleneck mapping
                if hasattr(bb, 'tail'):  # CUSTOM BIG BLOCK
                    if hasattr(bb, 'conv3'):  # BigBottleneck (ResNet-50)
                        # Sync intermediate BN2 for ResNet-50 body
                        rb.bn2.weight.copy_(bb.bn2.weight)
                        rb.bn2.bias.copy_(bb.bn2.bias)
                        rb.bn2.running_mean.copy_(bb.bn2.running_mean)
                        rb.bn2.running_var.copy_(bb.bn2.running_var)
                        rb.conv3.weight.copy_(bb.conv3.weight)
                        # Tail is BN3 in ResNet-50
                        rb.bn3.weight.copy_(bb.tail.weight)
                        rb.bn3.bias.copy_(bb.tail.bias)
                        rb.bn3.running_mean.copy_(bb.tail.running_mean)
                        rb.bn3.running_var.copy_(bb.tail.running_var)
                    else:  # BigBasicBlock (ResNet-18)
                        # Tail is BN2 in ResNet-18
                        rb.bn2.weight.copy_(bb.tail.weight)
                        rb.bn2.bias.copy_(bb.tail.bias)
                        rb.bn2.running_mean.copy_(bb.tail.running_mean)
                        rb.bn2.running_var.copy_(bb.tail.running_var)
                else:  # STANDARD BLOCK (Stages 2-4 if num_big_stages=1)
                    rb.bn2.weight.copy_(bb.bn2.weight)
                    rb.bn2.bias.copy_(bb.bn2.bias)
                    rb.bn2.running_mean.copy_(bb.bn2.running_mean)
                    rb.bn2.running_var.copy_(bb.bn2.running_var)
                    if hasattr(bb, 'bn3'):  # Standard Bottleneck
                        rb.conv3.weight.copy_(bb.conv3.weight)
                        rb.bn3.weight.copy_(bb.bn3.weight)
                        rb.bn3.bias.copy_(bb.bn3.bias)
                        rb.bn3.running_mean.copy_(bb.bn3.running_mean)
                        rb.bn3.running_var.copy_(bb.bn3.running_var)

                # 3. Sync Downsample
                if rb.downsample is not None:
                    if hasattr(bb.downsample, 'conv'):  # BigDS
                        rb.downsample[0].weight.copy_(bb.downsample.conv.weight)
                        rb.downsample[1].weight.copy_(bb.downsample.bn.weight)
                        rb.downsample[1].bias.copy_(bb.downsample.bn.bias)
                        rb.downsample[1].running_mean.copy_(bb.downsample.bn.running_mean)
                        rb.downsample[1].running_var.copy_(bb.downsample.bn.running_var)
                    else:  # Standard Sequential
                        rb.downsample[0].weight.copy_(bb.downsample[0].weight)
                        rb.downsample[1].weight.copy_(bb.downsample[1].weight)
                        rb.downsample[1].bias.copy_(bb.downsample[1].bias)
                        rb.downsample[1].running_mean.copy_(bb.downsample[1].running_mean)
                        rb.downsample[1].running_var.copy_(bb.downsample[1].running_var)

        # 4. Sync final Fully Connected layer
        ref_model.fc.weight.copy_(big_model.fc.weight)
        ref_model.fc.bias.copy_(big_model.fc.bias)


class BigBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample_module=None, last_block=False, max_elements=6e6):
        super().__init__()
        self.max_elements = max_elements
        self.conv1 = BigConv2dStats(inplanes, planes, 3, stride, 1, out_on_gpu=True, max_elements=max_elements)
        self.bn1 = BigFusedBNReLU(planes, out_on_gpu=True, max_elements=max_elements)
        self.conv2 = BigConv2dStats(planes, planes, 3, 1, 1, out_on_gpu=True, max_elements=max_elements)
        self.tail = BigFusedBNAddReLU(planes, out_on_gpu=True, max_elements=max_elements)
        self.downsample = downsample_module
        self.last_block = last_block  # This flag can be used for Stage 4 logic
        self.junction_main = []
        self.junction_id = []
        # 2. Register them
        cc.BWD_ACCELERATOR_REGISTRY[id(self.junction_main)] = self.junction_main
        cc.BWD_ACCELERATOR_REGISTRY[id(self.junction_id)] = self.junction_id

    def forward(self, x, hot_potato=True):
        self.junction_main.clear()
        self.junction_id.clear()

        # RULE: Hold a local reference so branch deletion doesn't kill the underlying tensor
        cache_ref = getattr(x, '_big_gpu_cache', None)
        prev_id = getattr(x, '_prev_module_id', None)

        # Apply Junction (returns a view of x)
        x = BigBlockJunctionFunction.apply(x, id(self.junction_main), id(self.junction_id), prev_id)

        # Branch 1: Main Path
        x_main = x.view_as(x)
        x_main._prev_module_id = id(self.junction_main)
        if cache_ref is not None:
            x_main._big_gpu_cache = cache_ref  # conv1 will delete this from x_main

        out, m1, v1 = self.conv1(x_main, return_hot_potato=True)
        out._prev_module_id = id(self.conv1)
        out = self.bn1(out, m1, v1, hot_potato=True)
        out._prev_module_id = id(self.bn1)
        out, m2, v2 = self.conv2(out, return_hot_potato=True)

        # Branch 2: Identity Path
        x_id = x.view_as(x)
        x_id._prev_module_id = id(self.junction_id)
        if cache_ref is not None:
            x_id._big_gpu_cache = cache_ref  # Identity branch will use and delete this

        if self.downsample is not None:
            identity = self.downsample(x_id, hot_potato=True)
            identity_id = id(self.downsample.bn)
        else:
            identity = x_id
            identity_id = id(self.junction_id)

        # Routing to tail
        setattr(out, '_main_id', id(self.conv2))
        setattr(identity, '_identity_id', identity_id)

        # RULE: Cleanup local reference before entering Tail to free memory
        del cache_ref

        # Respect the passed hot_potato flag for the boundary exit
        return self.tail(out, identity, m2, v2, hot_potato=hot_potato)


class BigBasicBlockSequential(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample_module=None, last_block=False, max_elements=6e6):
        super().__init__()
        self.max_elements = max_elements
        self.last_block = last_block  # Keeps the signature identical

        # Path: Conv -> BN/ReLU -> Conv -> BN/ReLU
        # We ignore downsample_module to keep it purely sequential
        self.conv1 = BigConv2dStats(inplanes, planes, 3, stride, 1, out_on_gpu=True, max_elements=max_elements)
        self.bn1 = BigFusedBNReLU(planes, out_on_gpu=True, max_elements=max_elements)
        self.conv2 = BigConv2dStats(planes, planes, 3, 1, 1, out_on_gpu=True, max_elements=max_elements)
        # self.bn2 = BigFusedBNReLU(planes, out_on_gpu=True, max_elements=max_elements)
        self.bn2 = BigFusedBNAddReLU(planes, out_on_gpu=True, max_elements=max_elements)

    def forward(self, x, hot_potato=True):
        # 1. Conv1 -> BN1
        # Each layer automatically pops the previous _big_gpu_cache
        out, m1, v1 = self.conv1(x, return_hot_potato=hot_potato)
        out = self.bn1(out, m1, v1, hot_potato=hot_potato)

        # 2. Conv2 -> BN2 (The Exit Gate)
        out, m2, v2 = self.conv2(out, return_hot_potato=hot_potato)

        # If hot_potato=False (at the end of Stage 1), bn2 returns a standard
        # GPU tensor to Stage 2. If True, it continues the highway.
        return self.bn2(out, torch.zeros_like(out), m2, v2, hot_potato=hot_potato)


class BigResNet(nn.Module):
    def __init__(self, encoder_name='resnet18', num_big_stages=4, max_elements=6e6):
        super().__init__()
        self.max_elements = max_elements
        self.num_big_stages = num_big_stages
        self.inplanes = 64
        self.block_cls = BigBottleneck if encoder_name == 'resnet50' else BigBasicBlock
        self.layers = [3, 4, 6, 3] if encoder_name == 'resnet50' else [2, 2, 2, 2]
        self.expansion = 4 if encoder_name == 'resnet50' else 1
        self.conv1 = BigConv2dStats(3, 64, 7, 2, 3, out_on_gpu=True, max_elements=max_elements)
        self.bn_pool = BigFusedBNReLUMaxPool(64, out_on_gpu=True, max_elements=max_elements)
        self.stages = nn.ModuleList()
        curr_planes = 64
        for i in range(4):
            stride = 1 if i == 0 else 2
            if i < num_big_stages:
                self.stages.append(self._make_big_layer(curr_planes, self.layers[i], stride))
            else:
                self.stages.append(self._make_standard_layer(curr_planes, self.layers[i], stride, encoder_name))
            curr_planes *= 2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(curr_planes // 2 * self.expansion, 1000)

    def _make_big_layer(self, planes, blocks, stride):
        ds = BigDS(self.inplanes, planes * self.expansion, stride,
                   max_elements=self.max_elements) if stride != 1 or self.inplanes != planes * self.expansion else None
        layers = [self.block_cls(self.inplanes, planes, stride, ds, max_elements=self.max_elements)]
        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(self.block_cls(self.inplanes, planes, max_elements=self.max_elements))
        return nn.Sequential(*layers)

    def _make_standard_layer(self, planes, blocks, stride, name):
        t_block = Bottleneck if name == 'resnet50' else BasicBlock
        ds = nn.Sequential(nn.Conv2d(self.inplanes, planes * self.expansion, 1, stride, bias=False),
                           nn.BatchNorm2d(planes * self.expansion)).to(
            cc.cuda_device) if stride != 1 or self.inplanes != planes * self.expansion else None
        layers = [t_block(self.inplanes, planes, stride, ds)]
        self.inplanes = planes * self.expansion
        for _ in range(1, blocks): layers.append(t_block(self.inplanes, planes))
        return nn.Sequential(*layers).to(cc.cuda_device)

    def set_gpu_config(self, num_cpu_blocks):
        self.conv1.out_on_gpu = (num_cpu_blocks <= 0)
        self.bn_pool.out_on_gpu = (num_cpu_blocks <= 0)
        count = 1
        for i, stage in enumerate(self.stages):
            if i >= self.num_big_stages: break
            for block in stage:
                bg = (count >= num_cpu_blocks)
                for m in block.modules():
                    if hasattr(m, 'out_on_gpu'): m.out_on_gpu = bg
                count += 1

    def forward(self, x):
        stem_hp = (self.num_big_stages > 0)
        x, m, v = self.conv1(x, return_hot_potato=stem_hp)
        x = self.bn_pool(x, mean=m, var=v, hot_potato=stem_hp)
        for i in range(self.num_big_stages):
            stage, is_last_stg = self.stages[i], (i == self.num_big_stages - 1)
            for j, block in enumerate(stage):
                is_exit = is_last_stg and (j == len(stage) - 1)
                x = block(x, hot_potato=not is_exit)
        if hasattr(x, '_big_gpu_cache'):
            del x._big_gpu_cache  # Free the last activation before entering standard stages
        for i in range(self.num_big_stages, 4):
            if x.device != cc.cuda_device: x = x.to(cc.cuda_device)
            x = self.stages[i](x)
        embedding = torch.flatten(self.avgpool(x), 1)
        embedding = embedding.to(self.fc.weight.device)
        return self.fc(embedding)
