"""WSI-level augmentation: sample one transform and apply it to every tile in the bag."""

import random

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms.autoaugment import TrivialAugmentWide, _apply_op


# -----------------------------------------------------------------------------
# RN18 augmentation
# -----------------------------------------------------------------------------


class FixedBatchAugment:
    """Apply the same photometric and geometric augmentation parameters to every tile in one WSI."""

    def __init__(self, blur_kernel=3, blur_sigma=(0.1, 2.0), brightness=(0.5, 1.5), contrast=(0.5, 1.5),
                 saturation=(0.5, 1.5), hue=(-0.3, 0.3), rotation=90):
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.rotation = rotation

    def generate_params(self):
        """Sample one parameter set for a WSI bag."""
        sigma = random.uniform(*self.blur_sigma) if isinstance(self.blur_sigma, (tuple, list)) else self.blur_sigma
        return {
            "sigma": sigma,
            "brightness": random.uniform(*self.brightness),
            "contrast": random.uniform(*self.contrast),
            "saturation": random.uniform(*self.saturation),
            "hue": random.uniform(*self.hue),
            "do_hflip": random.random() > 0.5,
            "do_vflip": random.random() > 0.5,
            "angle": random.choice([0, self.rotation, -self.rotation]),
        }

    def __call__(self, image, params):
        image = TF.gaussian_blur(image, kernel_size=self.blur_kernel, sigma=params["sigma"])
        image = TF.adjust_brightness(image, params["brightness"])
        image = TF.adjust_contrast(image, params["contrast"])
        image = TF.adjust_saturation(image, params["saturation"])
        image = TF.adjust_hue(image, params["hue"])

        if params["do_hflip"]:
            image = TF.hflip(image)
        if params["do_vflip"]:
            image = TF.vflip(image)
        return TF.rotate(image, params["angle"])


class TrainTransforms:
    """Apply a fixed WSI-level augmentation followed by the encoder preprocessing pipeline."""

    def __init__(self, augmentation, additional_transforms):
        self.augmentation = augmentation
        self.compose = transforms.Compose(additional_transforms)

    def generate_params(self):
        return self.augmentation.generate_params()

    def __call__(self, image, params):
        return self.compose(self.augmentation(image, params))


# -----------------------------------------------------------------------------
# H0-mini augmentation
# -----------------------------------------------------------------------------


class FixedTrivialAugmentWide(TrivialAugmentWide):
    """TrivialAugmentWide with one sampled operation and magnitude shared by the whole WSI bag."""

    def __init__(self, resize_size=224, num_magnitude_bins=31, interpolation=InterpolationMode.NEAREST, fill=None,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__(num_magnitude_bins=num_magnitude_bins, interpolation=interpolation, fill=fill)
        self.resize = transforms.Resize(resize_size)
        self.final_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    def generate_params(self):
        """Sample one TrivialAugmentWide operation and magnitude."""
        operation_space = self._augmentation_space(self.num_magnitude_bins)
        operation_index = int(torch.randint(len(operation_space), (1,)).item())
        operation_name = list(operation_space.keys())[operation_index]
        magnitudes, signed = operation_space[operation_name]

        if magnitudes.ndim > 0:
            magnitude_index = torch.randint(len(magnitudes), (1,), dtype=torch.long).item()
            magnitude = float(magnitudes[magnitude_index])
        else:
            magnitude = float(magnitudes)

        if signed and bool(torch.randint(2, (1,)).item()):
            magnitude *= -1.0
        return {"op_name": operation_name, "magnitude": magnitude}

    def __call__(self, image, params=None):
        if params is None:
            params = self.generate_params()

        image = self.resize(image)
        fill = self.fill
        if isinstance(image, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(image)
            elif fill is not None:
                fill = [float(value) for value in fill]

        image = _apply_op(
            image,
            params["op_name"],
            params["magnitude"],
            interpolation=self.interpolation,
            fill=fill,
        )
        return self.final_transform(image)
