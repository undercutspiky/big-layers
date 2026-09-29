"""Full-batch BatchNorm with host-resident activations and chunked device computation.

Chunking changes storage and execution, not the population used to compute BatchNorm statistics.
Parameters and running statistics stay on the compute device; large activations stay in CPU RAM.
"""
import torch
from .amp_compat import custom_bwd, custom_fwd
from torch.autograd.function import once_differentiable

from . import cuda_config as cc


def _chunks(x, max_elements):
    """Yield non-overlapping (batch, height) slices, retaining full channel and width dimensions."""
    n, c, h, w = x.shape
    batch_size = max(1, min(n, int(max_elements) // (c * h * w)))
    for b0 in range(0, n, batch_size):
        b1 = min(n, b0 + batch_size)
        rows = max(1, min(h, int(max_elements) // ((b1 - b0) * c * w)))
        for y0 in range(0, h, rows):
            yield b0, b1, y0, min(h, y0 + rows)


class BigBatchNormFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, X, weight, bias, running_mean, running_var, momentum, eps, output, X_grad,
                training, relu, max_elements, out_on_gpu):
        """Compute global statistics in pass 1; normalise every chunk with those statistics in pass 2."""
        source = X.detach().to('cpu') if X.is_cuda else X.detach()
        device = weight.device
        acc_dtype = torch.float64 if X.dtype == torch.float64 else torch.float32
        if training:
            mean, variance, count = BigBatchNormFunction.compute_welford_mean_and_variance(
                source, max_elements, device, acc_dtype)
            # Training uses var_biased = M2 / m. The running estimate uses M2 / (m - 1).
            running_mean.lerp_(mean.to(running_mean.dtype), momentum)
            running_var.lerp_((variance * (count / (count - 1))).to(running_var.dtype), momentum)
        else:
            # Clone: a later running-stat update must not change this forward's saved statistics.
            mean, variance = running_mean.to(acc_dtype).clone(), running_var.to(acc_dtype).clone()

        invstd = torch.rsqrt(variance + eps)  # eps belongs inside the square root.
        ctx.save_for_backward(source, weight, bias, mean, invstd)
        ctx.input_device, ctx.input_dtype = X.device, X.dtype
        ctx.X_grad, ctx.max_elements = X_grad, max_elements
        ctx.training, ctx.relu, ctx.acc_dtype = bool(training), bool(relu), acc_dtype
        for b0, b1, y0, y1 in _chunks(source, max_elements):
            x = source[b0:b1, :, y0:y1, :].to(device=device, dtype=acc_dtype)
            # x_hat = (x - mu) / sqrt(var + eps); y = gamma * x_hat + beta.
            xhat = (x - mean[None, :, None, None]) * invstd[None, :, None, None]
            y = xhat * weight[None, :, None, None] + bias[None, :, None, None]
            if relu:
                y = y.relu()
            output[b0:b1, :, y0:y1, :].copy_(y)  # Blocking: the next host layer consumes this output.
        return output.to(device) if out_on_gpu else output

    @staticmethod
    def compute_welford_mean_and_variance(X, max_elements, device, dtype):
        """Merge per-chunk means and centered sums of squares without subtracting large raw moments."""
        mean = torch.zeros(X.shape[1], device=device, dtype=dtype)
        M2 = torch.zeros_like(mean)
        count = 0
        for b0, b1, y0, y1 in _chunks(X, max_elements):
            x = X[b0:b1, :, y0:y1, :].to(device=device, dtype=dtype)
            chunk_count = x.shape[0] * x.shape[2] * x.shape[3]
            chunk_mean = x.mean((0, 2, 3))
            chunk_M2 = ((x - chunk_mean[None, :, None, None]) ** 2).sum((0, 2, 3))
            total = count + chunk_count
            delta = chunk_mean - mean
            # Parallel Welford merge, channel by channel:
            # mu' = mu + delta * n_chunk / n'; M2' = M2 + M2_chunk + delta^2 * n*n_chunk/n'.
            M2.add_(chunk_M2 + delta.square() * (count * chunk_count / total))
            mean.add_(delta * (chunk_count / total))
            count = total
        return mean, M2 / count, count

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_out):
        """Reduce d_beta and d_gamma globally, then evaluate dX with those two global reductions.

        For m = N*H*W and d = dL/dy:
            d_beta = sum(d), d_gamma = sum(d*x_hat)
            dX = gamma*invstd * (d - d_beta/m - x_hat*d_gamma/m).
        Using per-chunk reductions here would train a different normalisation layer.
        """
        X, weight, bias, mean, invstd = ctx.saved_tensors
        device = weight.device
        grad_bias = torch.zeros_like(mean)
        grad_weight = torch.zeros_like(mean)
        for b0, b1, y0, y1 in _chunks(X, ctx.max_elements):
            x = X[b0:b1, :, y0:y1, :].to(device=device, dtype=ctx.acc_dtype)
            dy = grad_out[b0:b1, :, y0:y1, :].to(device=device, dtype=ctx.acc_dtype)
            xhat = (x - mean[None, :, None, None]) * invstd[None, :, None, None]
            if ctx.relu:
                # Differentiate the optional fused ReLU using the BN output, not the raw input.
                pre_relu = xhat * weight[None, :, None, None] + bias[None, :, None, None]
                dy = dy * (pre_relu > 0)
            grad_bias.add_(dy.sum((0, 2, 3)))
            grad_weight.add_((dy * xhat).sum((0, 2, 3)))

        grad_X = None
        if ctx.needs_input_grad[0]:
            grad_X = ctx.X_grad
            m = X.shape[0] * X.shape[2] * X.shape[3]
            scale = weight * invstd
            for b0, b1, y0, y1 in _chunks(X, ctx.max_elements):
                x = X[b0:b1, :, y0:y1, :].to(device=device, dtype=ctx.acc_dtype)
                dy = grad_out[b0:b1, :, y0:y1, :].to(device=device, dtype=ctx.acc_dtype)
                xhat = (x - mean[None, :, None, None]) * invstd[None, :, None, None]
                if ctx.relu:
                    pre_relu = xhat * weight[None, :, None, None] + bias[None, :, None, None]
                    dy = dy * (pre_relu > 0)
                if ctx.training:
                    dy = dy - grad_bias[None, :, None, None] / m - xhat * grad_weight[None, :, None, None] / m
                # In evaluation mode mu/var are constants, so dX = gamma*invstd*d (no batch terms).
                grad_X[b0:b1, :, y0:y1, :].copy_(scale[None, :, None, None] * dy)
            grad_X = grad_X.to(ctx.input_device)
        dw = grad_weight.to(weight.dtype) if ctx.needs_input_grad[1] else None
        db = grad_bias.to(bias.dtype) if ctx.needs_input_grad[2] else None
        return grad_X, dw, db, None, None, None, None, None, None, None, None, None, None


class BigBatchNorm(torch.nn.Module):
    """Affine BatchNorm2d with global N,H,W statistics and reusable CPU buffers.

    Like BigConv2d, this module expects one outstanding forward/backward per instance.
    max_elements bounds a chunk's input size; parameters and reduction workspaces also use memory.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, relu=False, out_on_gpu=False,
                 max_elements=6_000_000, device=None):
        super().__init__()
        if num_features < 1 or eps <= 0 or not 0 <= momentum <= 1 or max_elements < 1:
            raise ValueError('Invalid BatchNorm size, epsilon, momentum or chunk budget.')
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.relu, self.out_on_gpu, self.max_elements = relu, out_on_gpu, int(max_elements)
        device = device or cc.cuda_device
        self.weight = torch.nn.Parameter(torch.ones(num_features, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(num_features, device=device))
        self.register_buffer('running_mean', torch.zeros(num_features, device=device))
        self.register_buffer('running_var', torch.ones(num_features, device=device))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=device))
        self.output, self.X_grad = None, None

    def forward(self, X):
        if X.ndim != 4 or X.shape[1] != self.num_features or min(X.shape) < 1:
            raise ValueError(f'Expected nonempty NCHW input with {self.num_features} channels, got {X.shape}.')
        if self.training and X.shape[0] * X.shape[2] * X.shape[3] <= 1:
            raise ValueError('BatchNorm training needs more than one value per channel.')
        if self.training:
            self.num_batches_tracked.add_(1)
        self.output, output = cc.host_buffer(self.output, X.shape, X.dtype, self.weight.is_cuda)
        grad_X = None
        if X.requires_grad:
            self.X_grad, grad_X = cc.host_buffer(self.X_grad, X.shape, X.dtype, self.weight.is_cuda)
        return BigBatchNormFunction.apply(X, self.weight, self.bias, self.running_mean, self.running_var,
                                         self.momentum, self.eps, output, grad_X, self.training, self.relu,
                                         self.max_elements, self.out_on_gpu)

    def extra_repr(self):
        return f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, max_elements={self.max_elements}'
