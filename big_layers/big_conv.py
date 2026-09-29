"""Height-partitioned convolution with host-resident activations and exact first-order gradients."""
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from .amp_compat import autocast_dtype, autocast_enabled, custom_bwd, custom_fwd
from torch.autograd.function import once_differentiable

from . import cuda_config as cc


def _partitions(shape, kernel_height, stride, padding, dilation, max_elements):
    """Yield disjoint output-row ranges and their overlapping input halos.

    An output interval [o0,o1) needs input rows [o0*s-p, (o1-1)*s-p+k_eff).
    Only the first/last interval may need explicit top/bottom padding. Width remains whole.
    """
    n, c, h, w = shape
    k_eff = dilation * (kernel_height - 1) + 1
    out_h = (h + 2 * padding - k_eff) // stride + 1
    batch_size = max(1, min(n, int(max_elements) // (c * h * w)))
    for b0 in range(0, n, batch_size):
        b1 = min(b0 + batch_size, n)
        patch_h = max(k_eff, int(max_elements) // ((b1 - b0) * c * w))
        output_rows = max(1, (patch_h - k_eff) // stride + 1)
        for o0 in range(0, out_h, output_rows):
            o1 = min(o0 + output_rows, out_h)
            y0, y1 = o0 * stride - padding, (o1 - 1) * stride - padding + k_eff
            yield b0, b1, o0, o1, max(y0, 0), min(y1, h), max(-y0, 0), max(y1 - h, 0)


class BigConv2dFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, X, weight, stride, padding, dilation, output, X_grad, max_elements, out_on_gpu):
        """Copy one input halo to the compute device, convolve, and assemble disjoint output rows."""
        device = weight.device
        source = X.detach().to('cpu') if X.is_cuda else X.detach()
        ctx.save_for_backward(source, weight)
        ctx.input_device, ctx.input_dtype = X.device, X.dtype
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        ctx.max_elements, ctx.X_grad = max_elements, X_grad
        ctx.compute_dtype = output.dtype
        streams = cc.get_cuda_streams(device)
        if streams:
            caller = torch.cuda.current_stream(device)
            for stream in streams:
                stream.wait_stream(caller)

        parts = _partitions(X.shape, weight.shape[2], stride, padding, dilation, max_elements)
        for i, (b0, b1, o0, o1, y0, y1, top, bottom) in enumerate(parts):
            context = torch.cuda.stream(streams[i % len(streams)]) if streams else nullcontext()
            with context:
                im = source[b0:b1, :, y0:y1, :].to(device, non_blocking=True)
                if top or bottom:
                    im = F.pad(im, (0, 0, top, bottom))
                # Height padding is local; width padding is the ordinary convolution padding.
                out = F.conv2d(im, weight, stride=stride, padding=(0, padding), dilation=dilation)
                output[b0:b1, :, o0:o1, :].copy_(out, non_blocking=bool(streams))
        for stream in streams:
            stream.synchronize()  # Host consumers must see completed device-to-host copies.
        return output.to(device) if out_on_gpu else output

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_out):
        """Replay each convolution; sum weight gradients and scatter-add input-halo gradients.

        dW = sum_j dW_j. Input halos overlap, so dX uses addition, not assignment.
        One compute stream serialises these reductions and avoids concurrent writes to dW.
        """
        X, weight = ctx.saved_tensors
        device = weight.device
        need_x, need_w = ctx.needs_input_grad[:2]
        grad_X = ctx.X_grad if need_x else None
        if grad_X is not None:
            grad_X.zero_()
        grad_weight = torch.zeros_like(weight) if need_w else None
        parts = _partitions(X.shape, weight.shape[2], ctx.stride, ctx.padding, ctx.dilation, ctx.max_elements)
        for b0, b1, o0, o1, y0, y1, top, bottom in parts:
            im = X[b0:b1, :, y0:y1, :].to(device=device, dtype=ctx.compute_dtype)
            if top or bottom:
                im = F.pad(im, (0, 0, top, bottom))
            grads = grad_out[b0:b1, :, o0:o1, :].to(device=device, dtype=ctx.compute_dtype).contiguous()
            # This is exactly the operator called by the original conv_backward_new.cpp bridge.
            grad_input, grad_w, _ = torch.ops.aten.convolution_backward.default(
                grads, im, weight.to(ctx.compute_dtype), None, [ctx.stride] * 2, [0, ctx.padding],
                [ctx.dilation] * 2, False, [0, 0], 1, [need_x, need_w, False])
            if need_w:
                grad_weight.add_(grad_w.to(weight.dtype))
            if need_x:
                # Crop only the explicit top/bottom padding; retain every genuine halo contribution.
                local_dx = grad_input[:, :, top:top + y1 - y0, :].to('cpu', dtype=ctx.input_dtype)
                grad_X[b0:b1, :, y0:y1, :].add_(local_dx)
        if need_x:
            grad_X = grad_X.to(ctx.input_device)
        return grad_X, grad_weight, None, None, None, None, None, None, None


class BigConv2d(torch.nn.Module):
    """Convolve NCHW inputs in bounded chunks. Parameters live on the selected compute device.

    max_elements bounds input elements per chunk, not total CUDA memory or convolution workspace.
    Reused host buffers assume one outstanding forward/backward per layer instance, as in ResNet.
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, dilation=1,
                 out_on_gpu=False, dtype=None, max_elements=60_000_000, device=None):
        super().__init__()
        self.input_channels, self.output_channels = input_channels, output_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        if len(self.kernel_size) != 2 or any(not isinstance(k, int) or k < 1 for k in self.kernel_size):
            raise ValueError('kernel_size must contain two positive integers.')
        if any(not isinstance(v, int) for v in (stride, padding, dilation)):
            raise TypeError('stride, padding and dilation must be integers.')
        if stride < 1 or dilation < 1 or padding < 0 or max_elements < 1:
            raise ValueError('Invalid stride, padding, dilation or max_elements.')
        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.max_elements, self.out_on_gpu, self._dtype = int(max_elements), bool(out_on_gpu), dtype
        self.weight = torch.nn.Parameter(torch.empty(output_channels, input_channels, *self.kernel_size,
                                                     device=device or cc.cuda_device))
        torch.nn.init.trunc_normal_(self.weight, std=0.02)
        self.output, self.X_grad = None, None

    def forward(self, X):
        if X.ndim != 4 or X.shape[1] != self.input_channels or min(X.shape) < 1:
            raise ValueError(f'Expected nonempty NCHW input with {self.input_channels} channels; got {X.shape}.')
        if X.dtype != self.weight.dtype:
            X = X.to(dtype=self.weight.dtype)
        kh, kw = self.kernel_size
        out_h = (X.shape[2] + 2 * self.padding - self.dilation * (kh - 1) - 1) // self.stride + 1
        out_w = (X.shape[3] + 2 * self.padding - self.dilation * (kw - 1) - 1) // self.stride + 1
        if min(out_h, out_w) < 1:
            raise ValueError('The effective kernel is larger than the padded input.')
        dtype = self._dtype or X.dtype
        if self.weight.is_cuda and autocast_enabled('cuda'):
            dtype = autocast_dtype('cuda')
        shape = (X.shape[0], self.output_channels, out_h, out_w)
        self.output, output = cc.host_buffer(self.output, shape, dtype, self.weight.is_cuda)
        X_grad = None
        if X.requires_grad:
            self.X_grad, X_grad = cc.host_buffer(self.X_grad, X.shape, X.dtype, self.weight.is_cuda)
        return BigConv2dFunction.apply(X, self.weight, self.stride, self.padding, self.dilation, output,
                                      X_grad, self.max_elements, self.out_on_gpu)

    def extra_repr(self):
        return (f'{self.input_channels}, {self.output_channels}, kernel_size={self.kernel_size}, '
                f'stride={self.stride}, padding={self.padding}, max_elements={self.max_elements}')
