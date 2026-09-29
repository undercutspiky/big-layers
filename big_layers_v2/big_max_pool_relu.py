"""EXPERIMENTAL: archived hot-potato path, not used by the paper training entry points."""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from big_layers.amp_compat import custom_bwd, custom_fwd

from . import cuda_config as cc


class BigMaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    @torch.no_grad()
    def forward(ctx, X, kernel_size, stride=1, padding=0, output=None, X_grad=None, out_on_gpu=False,
                memory_format=torch.channels_last):
        """
        Forward: run max_pool2d on X._big_gpu_out (GPU tensor).
        Copy the GPU result to the pinned host output and return gpu_out.
        We also copy the `indices` (for unpool) to CPU and stash in ctx for use in backward (which runs on CPU).
        """
        k = int(kernel_size)
        s = int(stride)
        p = int(padding)

        ctx.kernel_size = k
        ctx.stride = s
        ctx.padding = p
        ctx.X_grad = X_grad
        ctx.input_shape = X.shape

        # run full-tensor pooling on GPU (no slicing)
        gpu_out, indices_gpu = F.max_pool2d(X, kernel_size=k, stride=s, padding=p, return_indices=True)

        # copy gpu_out -> host pinned output (single copy)
        output.copy_(gpu_out, non_blocking=True)

        # store indices for CPU backward: copy indices to host (int64)
        # indices_gpu shape == gpu_out.shape; move to host device
        indices_cpu = indices_gpu.to(device=cc.host_device, non_blocking=True)
        ctx._saved_indices_cpu = indices_cpu  # CPU tensor
        ctx.output_shape = output.shape

        # return gpu_out (so wrapper can attach ._big_gpu_out on output)
        del X
        if out_on_gpu:
            return None, gpu_out
        return output, gpu_out

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_out, grad_gpu_out):
        """
        Backward runs on CPU only.
        grad_out: CPU tensor (host pinned). We use saved CPU indices and CPU grad_out to compute grad_input on CPU.
        """
        if grad_out is None or torch.sum(grad_out).item() == 0:
            grad_out, grad_gpu_out = grad_gpu_out, grad_out
        # Retrieve shapes and saved CPU indices
        k = ctx.kernel_size
        s = ctx.stride
        p = ctx.padding
        indices_cpu = ctx._saved_indices_cpu  # CPU int64
        input_shape = ctx.input_shape  # CPU shape (N,C,H,W)

        # prepare grad_X host buffer if provided
        grad_X = ctx.X_grad
        if grad_X is not None:
            grad_X.zero_()

        # If shapes mismatch (e.g., padded vs unpadded), pass output_size explicitly.
        out_size = (grad_out.shape[0], grad_out.shape[1], (input_shape[2] + 2 * p), input_shape[3])

        # Perform unpool on CPU
        if grad_out.is_cuda:
            grad_input = F.max_unpool2d(grad_out, indices_cpu.to(grad_out.device), kernel_size=k, stride=s,
                                        padding=p, output_size=out_size)
        else:
            # TODO: Implement max_unpool2d so that you can use grad_X to store output
            grad_input = F.max_unpool2d(grad_out.float(), indices_cpu, kernel_size=k, stride=s, padding=p,
                                        output_size=out_size)

        # Now crop the padded regions to original input size and add into grad_X if provided
        # forward may have padded; input_shape is original X.shape (without manual padding), so compute offsets
        # We need only the central region matching original X.shape[2].
        # If padding>0, we take the slice [p : p + H_original]
        H_orig = input_shape[2]
        pad = p
        h_start = pad
        h_end = pad + H_orig

        # grad_input_cpu shape: (N, C, out_size[2], W). Crop to original H_orig.
        cropped = grad_input[:, :, h_start:h_end, :]

        grad_X = cropped.contiguous()

        grad_X = grad_X.to(cc.cuda_device)
        return grad_X, None, None, None, None, None, None


class BigMaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, out_on_gpu=False, dtype=None):
        super(BigMaxPool2d, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else self.kernel_size
        self.padding = int(padding)
        self.output = None
        self.X_grad = None
        self._dtype = torch.float16 if cc.amp_enabled else torch.float32
        if dtype is not None:
            self._dtype = dtype
        self.out_on_gpu = out_on_gpu

    def forward(self, X: torch.Tensor):
        output = self._get_output_tensor(X)
        X_grad = self._get_input_grad_tensor(X) if X.requires_grad else None

        # call the Function: it will expect X._big_gpu_out to exist (or fall back to copying)
        out, gpu_out = BigMaxPool2dFunction.apply(X, self.kernel_size, self.stride, self.padding, output, X_grad,
                                                  self.out_on_gpu)
        if self.out_on_gpu:
            gpu_out._big_cpu_out = output
            return gpu_out
        # attach the GPU output on the host output so next layer can reuse it
        out._big_gpu_out = gpu_out
        return out

    def _get_output_tensor(self, X: torch.Tensor):
        input_height = X.shape[2] + 2 * self.padding
        input_width = X.shape[3] + 2 * self.padding
        out_h = (input_height - self.kernel_size) // self.stride + 1
        out_w = (input_width - self.kernel_size) // self.stride + 1

        if self.output is None:
            self.output = torch.empty((X.shape[0], X.shape[1], out_h, out_w), pin_memory=True,
                                      memory_format=torch.channels_last, device=cc.host_device,
                                      dtype=self._dtype, requires_grad=self.training)
            return self.output

        if self.output.shape[3] >= out_w and self.output.shape[2] >= out_h and self.output.shape[0] >= X.shape[0]:
            self.output.requires_grad_(False)
            sliced_output = self.output[:X.shape[0], :, :out_h, :out_w].contiguous(
                memory_format=torch.channels_last).pin_memory()
            sliced_output.requires_grad_(self.training)
            return sliced_output

        max_w = max(out_w, self.output.shape[3])
        max_h = max(out_h, self.output.shape[2])
        self.output.requires_grad_(False)
        self.output = self.output.resize_((X.shape[0], X.shape[1], max_h, max_w))
        sliced_output = self.output[:, :, :out_h, :out_w].contiguous(memory_format=torch.channels_last).pin_memory()
        # sliced_output.requires_grad_(True)
        sliced_output.requires_grad_(self.training)

        return sliced_output

    def _get_input_grad_tensor(self, X: torch.Tensor):
        if self.X_grad is None:
            self.X_grad = torch.zeros(X.shape, pin_memory=True, device=cc.host_device, dtype=self._dtype).to(
                memory_format=torch.channels_last)
            return self.X_grad
        if (self.X_grad.shape[2] >= X.shape[2] and self.X_grad.shape[3] >= X.shape[3] and
                self.X_grad.shape[0] >= X.shape[0]):
            return self.X_grad[:X.shape[0], :, :X.shape[2], :X.shape[3]].contiguous(
                memory_format=torch.channels_last).pin_memory()

        max_w = max(X.shape[3], self.X_grad.shape[3])
        max_h = max(X.shape[2], self.X_grad.shape[2])
        self.X_grad = self.X_grad.resize_((X.shape[0], X.shape[1], max_h, max_w))
        sliced = self.X_grad[:, :, :X.shape[2], :X.shape[3]].contiguous(memory_format=torch.channels_last).pin_memory()
        return sliced


class BigAddFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    @torch.no_grad()
    def forward(ctx, X1, X2, output, relu=False):
        X1_gpu = X1
        X2_gpu = X2
        ctx.return_cpu_grad1 = not X1_gpu.is_cuda
        ctx.return_cpu_grad2 = not X2_gpu.is_cuda
        if hasattr(X1, '_big_gpu_out'):
            X1_gpu = X1._big_gpu_out
            ctx.return_cpu_grad1 = False
        if hasattr(X2, '_big_gpu_out'):
            X2_gpu = X2._big_gpu_out
            ctx.return_cpu_grad2 = False

        if not X1_gpu.is_cuda:
            X1_gpu = X1_gpu.to(cc.cuda_device)
        if not X2_gpu.is_cuda:
            X2_gpu = X2_gpu.to(cc.cuda_device)

        gpu_out = X1_gpu + X2_gpu
        del X1_gpu, X2_gpu

        if bool(relu):
            gpu_out = F.relu(gpu_out)
            output.copy_(gpu_out, non_blocking=True)  # CPU copy of output

        ctx._saved_output_cpu = output
        ctx.relu = bool(relu)

        del X1, X2
        return output, gpu_out

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_out, grad_gpu_out):
        """
        Backward runs on CPU only.
        grad_out: CPU tensor (host pinned). We use saved CPU indices and CPU grad_out to compute grad_input on CPU.
        """
        if ctx.relu:
            output = ctx._saved_output_cpu
            mask = (output > 0).to(cc.cuda_device, dtype=grad_out.dtype)
            grad_gpu_out = grad_gpu_out * mask

        grad1 = grad_gpu_out
        grad2 = grad_gpu_out
        if ctx.return_cpu_grad1 and grad1.is_cuda:
            grad1 = grad1.to(cc.host_device)
        if ctx.return_cpu_grad2 and grad2.is_cuda:
            grad2 = grad2.to(cc.host_device)

        return grad1, grad2, None, None


class BigAdd(torch.nn.Module):
    def __init__(self, relu=False, dtype=None):
        super(BigAdd, self).__init__()
        self.relu = relu
        self.output = None
        self._dtype = torch.float16 if cc.amp_enabled else torch.float32
        if dtype is not None:
            self._dtype = dtype

    def forward(self, X1: torch.Tensor, X2: torch.Tensor):
        output = self._get_output_tensor(X1)
        output, gpu_out = BigAddFunction.apply(X1, X2, output, self.relu)
        gpu_out._big_cpu_out = output
        return gpu_out

    def _get_output_tensor(self, X: torch.Tensor):
        if self.output is None:
            self.output = torch.empty(X.shape,
                pin_memory=True, memory_format=torch.channels_last, device=cc.host_device,
                                    dtype=self._dtype)
            return self.output

        if (self.output.shape[3] >= X.shape[3] and self.output.shape[2] >= X.shape[2] and
            self.output.shape[0] >= X.shape[0]):
            sliced_output = self.output[:X.shape[0], :, :X.shape[2], :X.shape[3]].contiguous(
                memory_format=torch.channels_last).pin_memory()
            return sliced_output

        max_w = max(X.shape[3], self.output.shape[3])
        max_h = max(X.shape[2], self.output.shape[2])
        self.output = self.output.resize_((X.shape[0], X.shape[1], max_h, max_w))
        sliced_output = self.output[:, :, :X.shape[2], :X.shape[3]].contiguous(
            memory_format=torch.channels_last).pin_memory()

        return sliced_output


# -------------------------
# BigReLU (GPU forward using X._big_gpu_out, CPU backward)
# -------------------------
class BigReLUFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    @torch.no_grad()
    def forward(ctx, X, output=None, X_grad=None, memory_format=torch.channels_last):
        """
        Forward: run ReLU on X._big_gpu_out (GPU). Copy gpu result to output (host). Save CPU mask for backward.
        """
        ctx.X_grad = X_grad
        ctx.input_shape = X.shape

        gpu_out = F.relu(X._big_gpu_out)
        # mask = (gpu_out > 0)  # or (gpu_in > 0) - either works
        mask_gpu = (gpu_out > 0).to(dtype=torch.uint8)  # small memory

        # prepare output host
        h, w = X.shape[2], X.shape[3]
        if output is None:
            output = torch.empty((X.shape[0], X.shape[1], h, w), pin_memory=True,
                                 memory_format=memory_format, device=cc.host_device,
                                 dtype=gpu_out.dtype)

        # copy to host
        output.copy_(gpu_out, non_blocking=True)

        # move mask to CPU for CPU backward
        ctx._saved_mask_cpu = mask_gpu.to(device=cc.host_device, non_blocking=True)

        del X._big_gpu_out
        return gpu_out

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_out):
        """
        Backward runs on CPU: multiply CPU grad_out by CPU mask and write into preallocated grad_X if present.
        """
        mask_cpu = ctx._saved_mask_cpu  # uint8 cpu
        grad_X = ctx.X_grad
        if grad_X is not None:
            grad_X.zero_()

        # convert mask to grad dtype and multiply on CPU
        grad_in_cpu = grad_out * mask_cpu.to(dtype=grad_out.dtype)

        if grad_X is not None:
            grad_X += grad_in_cpu
        else:
            grad_X = grad_in_cpu.contiguous()

        # forward args: X, output, X_grad
        return grad_X, None, None


class BigReLU(torch.nn.Module):
    def __init__(self, dtype=None):
        super(BigReLU, self).__init__()
        self.output = None
        self.X_grad = None
        self._dtype = torch.float16 if cc.amp_enabled else torch.float32
        if dtype is not None:
            self._dtype = dtype

    def forward(self, X: torch.Tensor):
        output = self._get_output_tensor(X)
        X_grad = self._get_input_grad_tensor(X) if X.requires_grad else None
        gpu_out = BigReLUFunction.apply(X, output, X_grad)
        output._big_gpu_out = gpu_out
        return output

    def _get_output_tensor(self, X: torch.Tensor):
        h, w = X.shape[2], X.shape[3]
        if self.output is None:
            self.output = torch.empty((X.shape[0], X.shape[1], h, w), pin_memory=True,
                                      memory_format=torch.channels_last, device=cc.host_device,
                                      dtype=self._dtype, requires_grad=True)
            return self.output
        if (self.output.shape[2] >= h and self.output.shape[3] >= w and self.output.shape[0] >= X.shape[0]):
            return self.output[:X.shape[0], :, :h, :w].contiguous(memory_format=torch.channels_last).pin_memory()

        max_w = max(w, self.output.shape[3])
        max_h = max(h, self.output.shape[2])
        self.output = self.output.resize_((X.shape[0], X.shape[1], max_h, max_w))
        sliced = self.output[:, :, :h, :w].contiguous(memory_format=torch.channels_last).pin_memory()
        sliced.requires_grad_(True)
        return sliced

    def _get_input_grad_tensor(self, X: torch.Tensor):
        if self.X_grad is None:
            self.X_grad = torch.zeros(X.shape, pin_memory=True, device=cc.host_device, dtype=self._dtype).to(
                memory_format=torch.channels_last)
            return self.X_grad
        if (self.X_grad.shape[2] >= X.shape[2] and self.X_grad.shape[3] >= X.shape[3] and
                self.X_grad.shape[0] >= X.shape[0]):
            return self.X_grad[:X.shape[0], :, :X.shape[2], :X.shape[3]].contiguous(
                memory_format=torch.channels_last).pin_memory()

        max_w = max(X.shape[3], self.X_grad.shape[3])
        max_h = max(X.shape[2], self.X_grad.shape[2])
        self.X_grad = self.X_grad.resize_((X.shape[0], X.shape[1], max_h, max_w))
        sliced = self.X_grad[:, :, :X.shape[2], :X.shape[3]].contiguous(memory_format=torch.channels_last).pin_memory()
        return sliced
