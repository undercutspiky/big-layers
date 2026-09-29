"""Image encoders and feature conventions used by the training and feature-extraction scripts."""

import json
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

from big_layers import resnet18 as big_resnet18

FM_NAMES = ("h0-mini", "uni2-h", "h-optimus-1", "prov-gigapath")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CAM_MEAN = (0.756786, 0.653295, 0.762873)
CAM_STD = (0.221554, 0.279361, 0.201079)


# -----------------------------------------------------------------------------
# ResNet-18
# -----------------------------------------------------------------------------


class RN18Encoder(nn.Module):
    """RN18 encoder with the feature output used for each dataset in the experiments."""

    def __init__(self, dataset, end_to_end, weights=None, pretrained=True, num_big_stages=0,
                 conv_elements=60_000_000, bn_elements=6_000_000):
        super().__init__()
        self.pooling = "flatten" if dataset == "panda" else "avg"
        self.end_to_end = end_to_end
        self.feature_dim = 32768 if dataset == "panda" else 512

        if end_to_end:
            self.encoder = big_resnet18(
                pretrained,
                weights,
                pooling=self.pooling,
                num_big_stages=num_big_stages,
                conv_elements=conv_elements,
                bn_elements=bn_elements,
            )
        else:
            default_weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained and not weights else None
            self.encoder = models.resnet18(weights=default_weights)
            if weights:
                state = torch.load(weights, map_location="cpu", weights_only=True)
                self.encoder.load_state_dict(state, strict=True)
            self.encoder.fc = nn.Identity()

    def forward(self, x):
        if self.end_to_end:
            return self.encoder(x)

        # Frozen RN18 follows the native torchvision layers and returns the same feature convention as BigResNet.
        x = x.to(next(self.encoder.parameters()).device, non_blocking=True)
        model = self.encoder
        x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
        x = model.avgpool(x) if self.pooling == "avg" else x
        return x.flatten(1)


# -----------------------------------------------------------------------------
# Pathology foundation models
# -----------------------------------------------------------------------------


class H0MiniEncoder(nn.Module):
    """Return H0-mini's concatenated CLS and mean patch-token representation."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feature_dim = 1536

    def forward(self, x):
        tokens = self.model.forward_features(x)
        if tokens.ndim != 3 or tokens.shape[1] <= 5:
            raise ValueError("H0-mini must provide CLS, four register tokens, and patch tokens.")
        return torch.cat((tokens[:, 0], tokens[:, 5:].mean(1)), dim=-1)


def _load_state_dict(path):
    """Load a local PyTorch or safetensors checkpoint."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_foundation_encoder(name, weights, device="cuda:0"):
    """Build a supported foundation encoder and load its local checkpoint plus published preprocessing."""
    import timm

    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(path)

    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    if name == "h0-mini":
        config = json.loads(path.with_name("config.json").read_text())
        architecture = config.get("architecture", "vit_base_patch14_reg4_dinov2")
        model = timm.create_model(
            architecture,
            pretrained=False,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=nn.SiLU,
            **config.get("model_args", {}),
        )
        mean = config["pretrained_cfg"]["mean"]
        std = config["pretrained_cfg"]["std"]
        resize = [transforms.Resize(224), transforms.CenterCrop(224)]

    elif name == "uni2-h":
        model = timm.create_model(
            "vit_giant_patch14_224",
            pretrained=False,
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
        resize = [transforms.Resize(224)]

    elif name == "prov-gigapath":
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
        resize = [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                  transforms.CenterCrop(224)]

    elif name == "h-optimus-1":
        model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=False,
                                  init_values=1e-5, dynamic_img_size=False)
        mean = (0.707223, 0.578729, 0.703617)
        std = (0.211883, 0.230117, 0.177517)
        resize = [transforms.CenterCrop(224)]

    else:
        raise ValueError(f"Choose one of {FM_NAMES}, got {name!r}.")

    state = _load_state_dict(path)
    model.load_state_dict(state.get("model", state), strict=True)
    model = H0MiniEncoder(model) if name == "h0-mini" else model
    model.feature_dim = 1536
    model.to(device).eval()

    preprocessing = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return model, preprocessing, (mean, std)
