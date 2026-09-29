"""AdamW with DEMON decay of the first-moment coefficient."""

import torch


class DemonAdamW(torch.optim.AdamW):
    """Decay beta1 from its initial value to zero over a fixed number of optimiser steps."""

    def __init__(self, params, lr, betas=(0.9, 0.999), total_steps=None, **kwargs):
        if total_steps is None or total_steps < 1:
            raise ValueError("total_steps must be a positive integer.")

        super().__init__(params, lr=lr, betas=betas, **kwargs)
        self.beta1_init = betas[0]
        self.beta2 = betas[1]
        self.total_steps = total_steps
        self._step_count = 0

    def step(self, closure=None):
        """Update beta1 according to DEMON, then run the normal AdamW step."""
        self._step_count += 1
        step = min(self._step_count, self.total_steps)
        fraction = 1.0 - step / self.total_steps
        numerator = self.beta1_init * fraction
        denominator = (1.0 - self.beta1_init) + self.beta1_init * fraction
        beta1 = numerator / denominator

        for group in self.param_groups:
            group["betas"] = (beta1, self.beta2)
        return super().step(closure)

    def state_dict(self):
        state = super().state_dict()
        state["demon"] = {
            "step_count": self._step_count,
            "total_steps": self.total_steps,
            "beta1_init": self.beta1_init,
            "beta2": self.beta2,
        }
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        demon = state_dict.pop("demon")
        super().load_state_dict(state_dict)
        self._step_count = demon["step_count"]
        self.total_steps = demon["total_steps"]
        self.beta1_init = demon["beta1_init"]
        self.beta2 = demon["beta2"]
