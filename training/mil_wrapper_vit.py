"""Full-bag MIL with encoder micro-batch recomputation for patch-independent ViT encoders."""

import torch
from torch import nn

from training.amp import autocast_context


class MILWrapperViT(nn.Module):
    """Keep bag embeddings for the MIL loss and reconstruct encoder activations during backward.

    Training has three phases:
      1. Encode tiles in micro-batches without retaining encoder graphs.
      2. Run the MIL head on the complete bag and backpropagate to the detached embeddings.
      3. Replay each encoder micro-batch with its dL/dE slice and accumulate encoder gradients.

    The encoder must process images independently. The wrapper also saves/replays RNG state so dropout
    and stochastic-depth layers use the same random draws in phases 1 and 3.
    """

    def __init__(self, encoder, mil, micro_batch_size, inf_batch_size=None, use_amp=True, amp_dtype=torch.float16):
        super().__init__()
        if micro_batch_size < 1 or (inf_batch_size is not None and inf_batch_size < 1):
            raise ValueError("Encoder micro-batch sizes must be positive.")

        self.encoder = encoder
        self.mil = mil
        self.micro_batch_size = micro_batch_size
        self.inf_batch_size = inf_batch_size or micro_batch_size
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self._pending = None

    def _encode(self, images):
        """Run one encoder micro-batch with the same autocast policy in the forward and replay passes."""
        device = next(self.encoder.parameters()).device
        with autocast_context(self.use_amp, self.amp_dtype, device.type):
            output = self.encoder(images.to(device, non_blocking=True))
        if output.ndim != 2:
            raise ValueError("The encoder must return one feature vector per image (N x D).")
        return output

    def forward(self, x):
        """Run phases 1 and 2: detached encoder micro-batches followed by one full-bag MIL computation."""
        if self._pending is not None:
            raise RuntimeError("Call backward_encoder() before the next training forward.")

        batch_size, bag_size, channels, height, width = x.shape
        images = x.reshape(-1, channels, height, width)
        device = next(self.encoder.parameters()).device
        update_encoder = self.training and torch.is_grad_enabled() and any(
            parameter.requires_grad for parameter in self.encoder.parameters()
        )

        # Training-mode BatchNorm changes state and couples images, so this recomputation path cannot preserve it.
        if update_encoder and any(isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
                                  for module in self.encoder.modules()):
            raise ValueError("ViT recomputation requires a patch-independent encoder without training BatchNorm.")

        chunk_size = self.micro_batch_size if update_encoder else self.inf_batch_size
        states = []
        chunks = []

        with torch.no_grad():
            for start in range(0, len(images), chunk_size):
                if update_encoder:
                    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                    states.append((torch.get_rng_state(), cuda_state))
                chunks.append(self._encode(images[start:start + chunk_size]))

        features = torch.cat(chunks, dim=0).detach().requires_grad_(update_encoder)
        logits, auxiliary = self.mil(features.reshape(batch_size, bag_size, -1))

        if update_encoder:
            self._pending = (images, features, states)
        return logits, auxiliary, features

    def backward_encoder(self):
        """Run phase 3 by replaying each encoder micro-batch with its gradient from the full-bag loss."""
        if self._pending is None:
            return

        images, features, states = self._pending
        if features.grad is None:
            raise RuntimeError("Backpropagate the MIL/LwF loss before calling backward_encoder().")

        device = next(self.encoder.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] \
            if device.type == "cuda" else []

        try:
            for index, start in enumerate(range(0, len(images), self.micro_batch_size)):
                cpu_state, cuda_state = states[index]
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(cpu_state)
                    if cuda_state is not None:
                        torch.cuda.set_rng_state(cuda_state, device)

                    replay = self._encode(images[start:start + self.micro_batch_size])
                    replay.backward(features.grad[start:start + self.micro_batch_size])
        finally:
            self._pending = None
