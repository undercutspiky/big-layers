"""CNN and precomputed-feature wrapper around an MIL head."""

import torch
from torch import nn


class MILWrapper(nn.Module):
    """Connect a tile encoder to an MIL head without splitting trainable CNN BatchNorm populations."""

    def __init__(self, encoder, mil, freeze_encoder=False, inference_batch_size=32):
        super().__init__()
        if inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive.")

        self.encoder = encoder
        self.mil = mil
        self.freeze_encoder = freeze_encoder
        self.inference_batch_size = inference_batch_size

        if freeze_encoder:
            self.encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            # Frozen weights are not enough: BatchNorm running statistics must also remain frozen.
            self.encoder.eval()
        return self

    def forward(self, x):
        """Encode one complete training bag, or stream independent chunks when the encoder is frozen."""
        if x.ndim == 3:
            return self.mil(x.to(next(self.mil.parameters()).device))

        batch_size, bag_size, channels, height, width = x.shape
        images = x.reshape(-1, channels, height, width)

        if self.training and not self.freeze_encoder:
            # Big Layers partitions inside the CNN, so the encoder still sees the complete training population.
            features = self.encoder(images)
        else:
            with torch.no_grad():
                chunks = [self.encoder(chunk).clone() for chunk in images.split(self.inference_batch_size)]
            features = torch.cat(chunks)

        features = features.reshape(batch_size, bag_size, -1).to(next(self.mil.parameters()).device)
        return self.mil(features)
