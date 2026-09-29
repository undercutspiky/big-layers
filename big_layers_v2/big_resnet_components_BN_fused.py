"""EXPERIMENTAL: archived hot-potato path, not used by the paper training entry points."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from big_layers.amp_compat import custom_bwd, custom_fwd

from . import cuda_config as cc
from .cuda_config import GPU_CACHE_REGISTRY, BWD_ACCELERATOR_REGISTRY
from .big_resnet_components_fused import BigBatchNormFunction, BigConv2dStats

DEBUG = False  # TODO: Set to False for real usage
# GLOBAL AUDIT LOG
# Keys will be things like "Conv1_DX", "BN2_DX", etc.
audit_log = {}


def log_grad(name, tensor):
    if tensor is not None:
        audit_log[name] = tensor.detach().abs().mean().item()


class BigFusedBNReLUFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    @torch.no_grad()
    def forward(ctx, X, weight, bias, running_mean, running_var, momentum,
                output=None, X_grad=None, training=True,
                out_on_gpu=True, hot_potato=True,
                max_elements=6e6, memory_format=torch.channels_last,
                module_id=None, prev_module_id=None,
                external_mean=None, external_var=None):

        device = weight.device
        X_gpu_cache = getattr(X, '_big_gpu_cache', None)
        if X_gpu_cache is not None:
            del X._big_gpu_cache
            X_input = X_gpu_cache
        else:
            X_input = X

        # prev_bwd_accelerator = getattr(X, '_bwd_accelerator', None)
        # Logic: If we are outputting to GPU, we generally want to return
        # a GPU gradient (ctx.return_cpu_grad = False).
        # EXCEPT: If we are at the very start of the model (Stem) and
        # the original input was CPU, we must eventually return CPU.
        if out_on_gpu:
            if prev_module_id is not None:
                # Middle layer in an accelerated chain: Stay on GPU
                ctx.return_cpu_grad = False
            else:
                # Exit or Standalone layer: Return what the input X was
                ctx.return_cpu_grad = not X.is_cuda
        else:
            # Slow path: Always match the input X
            ctx.return_cpu_grad = not X.is_cuda

        if training:
            if external_mean is not None and external_var is not None:
                mean = external_mean
                # FIX: Convert Unbiased Var (from ConvStats) to Biased Var (for BN)
                # ConvStats uses N-1. BN expects N.
                # Biased = Unbiased * (N-1) / N
                n = X.numel() / X.size(1)
                variance = external_var #* ((n - 1) / n)
                running_mean = momentum * mean + (1 - momentum) * running_mean
                running_var = momentum * variance + (1 - momentum) * running_var
            else:
                mean, m2 = BigBatchNormFunction.compute_stats_welford_algo(X_input, max_elements)
                n_total = X.numel() / X.size(1)
                variance = m2 / n_total  # Biased variance by default from Welford m2/n

                # Update running stats (requires unbiased)
                unbiased_var = m2 / (n_total - 1)
                running_mean = momentum * mean + (1 - momentum) * running_mean
                running_var = momentum * unbiased_var + (1 - momentum) * running_var
        else:
            mean = running_mean
            variance = running_var

        ctx.x_device = X.device
        ctx.save_for_backward(X if not X.is_cuda else X.cpu(), weight, bias)
        ctx.mean = mean
        ctx.variance = variance
        ctx.X_grad = X_grad
        ctx.max_elements = max_elements
        ctx.memory_format = memory_format
        ctx.prev_module_id = prev_module_id
        ctx.module_id = module_id

        B, C, H, W = X.shape
        inv_std = torch.rsqrt(variance + 1e-5)
        scale = weight * inv_std
        shift = bias - mean * scale

        gpu_out = None
        if out_on_gpu:
            gpu_out = torch.empty(output.shape, device=device)

        max_batch = max(1, min(B, int(max_elements // (C * H * W))))
        streams = cc.get_cuda_streams()

        batch_start = 0
        while batch_start < B:
            batch_end = min(batch_start + max_batch, B)
            if X_input.is_cuda:
                sliced_X = X_input[batch_start:batch_end]
            else:
                sliced_X = X_input[batch_start:batch_end]

            sliced_out_cpu = output[batch_start:batch_end]

            iy, i = 0, 0
            while iy < H:
                ph = min(max(1, int(max_elements // (sliced_X.size(0) * C * W))), H - iy)
                with torch.cuda.stream(streams[i % len(streams)]):
                    if sliced_X.is_cuda:
                        im = sliced_X[:, :, iy:iy + ph, :]
                    else:
                        im = sliced_X[:, :, iy:iy + ph, :].to(device, non_blocking=True)

                    im = im * scale[None, :, None, None] + shift[None, :, None, None]
                    im = F.relu(im)

                    if out_on_gpu:
                        gpu_out[batch_start:batch_end, :, iy:iy + ph, :].copy_(im)
                    sliced_out_cpu[:, :, iy:iy + ph, :].copy_(im, non_blocking=True)
                iy += ph
                i += 1
            batch_start = batch_end

        for s in streams:
            s.synchronize()

        if out_on_gpu:
            if hot_potato:
                if module_id is not None:
                    GPU_CACHE_REGISTRY[module_id].append(gpu_out)
                return output, running_mean, running_var
            else:
                # Highway Exit: Return the GPU tensor directly to standard Autograd
                gpu_out.requires_grad_(True)
                return gpu_out, running_mean, running_var
        return output, running_mean, running_var

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_out, grad_rm, grad_rv):
        X, weight, bias = ctx.saved_tensors
        if ctx.module_id and len(BWD_ACCELERATOR_REGISTRY[ctx.module_id]) > 0:
            grad_out_gpu = BWD_ACCELERATOR_REGISTRY[ctx.module_id].pop()
            while len(BWD_ACCELERATOR_REGISTRY[ctx.module_id]) > 0:
                grad_out_gpu = grad_out_gpu + BWD_ACCELERATOR_REGISTRY[ctx.module_id].pop()
            use_gpu_grad = True
        else:
            grad_out_gpu = None
            use_gpu_grad = False

        X, weight, bias = ctx.saved_tensors
        mean, variance = ctx.mean, ctx.variance
        N, C, H, W = X.shape
        device = weight.device

        grad_bias = torch.zeros(C, device=device)
        grad_weight = torch.zeros(C, device=device)

        grad_X_cpu = ctx.X_grad
        # if grad_X_cpu is not None:
        #     grad_X_cpu.zero_()

        # FIX: Ensure CPU buffer allocated if returning CPU grad
        if grad_X_cpu is None and ctx.return_cpu_grad:
            grad_X_cpu = torch.zeros(X.shape, device=cc.host_device, pin_memory=torch.cuda.is_available()).to(
                memory_format=torch.channels_last)

        need_gpu_buffer = (ctx.prev_module_id is not None)
        grad_X_gpu = torch.zeros(X.shape, device=device) if need_gpu_buffer else None

        max_batch = max(1, min(N, int(ctx.max_elements // (C * H * W))))
        inv_std = torch.rsqrt(variance + 1e-5)
        scale = weight * inv_std
        shift = bias - mean * scale

        streams = cc.get_cuda_streams()

        batch_start = 0
        while batch_start < N:
            batch_end = min(batch_start + max_batch, N)

            if X.is_cuda:
                x_batch = X[batch_start:batch_end]
            else:
                x_batch = X[batch_start:batch_end]

            if use_gpu_grad:
                g_batch = grad_out_gpu[batch_start:batch_end]
            else:
                g_batch = grad_out[batch_start:batch_end]

            iy, i = 0, 0
            while iy < H:
                ph = min(max(1, int(ctx.max_elements // (x_batch.size(0) * C * W))), H - iy)
                with torch.cuda.stream(streams[i % len(streams)]):
                    if x_batch.is_cuda:
                        im = x_batch[:, :, iy:iy + ph, :]
                    else:
                        im = x_batch[:, :, iy:iy + ph, :].to(device, non_blocking=True)

                    if use_gpu_grad:
                        dy = g_batch[:, :, iy:iy + ph, :]
                    else:
                        dy = g_batch[:, :, iy:iy + ph, :].to(device, non_blocking=True)

                    val = im * scale[None, :, None, None] + shift[None, :, None, None]
                    mask = (val > 0).float()
                    dy = dy * mask

                    grad_bias += dy.sum(dim=(0, 2, 3))
                    im_centered = im - mean[None, :, None, None]
                    x_hat = im_centered * inv_std[None, :, None, None]
                    grad_weight += (dy * x_hat).sum(dim=(0, 2, 3))
                iy += ph
                i += 1
            batch_start = batch_end
        for s in streams: s.synchronize()

        N_elements = N * H * W
        term1 = (weight * inv_std)[None, :, None, None]
        sum_dy = grad_bias[None, :, None, None]
        sum_dy_xhat = grad_weight[None, :, None, None]

        batch_start = 0
        while batch_start < N:
            batch_end = min(batch_start + max_batch, N)

            if X.is_cuda:
                x_batch = X[batch_start:batch_end]
            else:
                x_batch = X[batch_start:batch_end]

            if use_gpu_grad:
                g_batch = grad_out_gpu[batch_start:batch_end]
            else:
                g_batch = grad_out[batch_start:batch_end]

            if grad_X_cpu is not None: sliced_grad_X_cpu = grad_X_cpu[batch_start:batch_end]

            iy, i = 0, 0
            while iy < H:
                ph = min(max(1, int(ctx.max_elements // (x_batch.size(0) * C * W))), H - iy)
                with torch.cuda.stream(streams[i % len(streams)]):
                    if x_batch.is_cuda:
                        im = x_batch[:, :, iy:iy + ph, :]
                    else:
                        im = x_batch[:, :, iy:iy + ph, :].to(device, non_blocking=True)

                    if use_gpu_grad:
                        dy = g_batch[:, :, iy:iy + ph, :]
                    else:
                        dy = g_batch[:, :, iy:iy + ph, :].to(device, non_blocking=True)

                    val = im * scale[None, :, None, None] + shift[None, :, None, None]
                    mask = (val > 0).float()
                    dy = dy * mask

                    x_hat = (im - mean[None, :, None, None]) * inv_std[None, :, None, None]
                    d_input = (term1 / N_elements) * (N_elements * dy - sum_dy - x_hat * sum_dy_xhat)

                    if grad_X_gpu is not None:
                        grad_X_gpu[batch_start:batch_end, :, iy:iy + ph, :].copy_(d_input)
                    elif grad_X_cpu is not None:
                        sliced_grad_X_cpu[:, :, iy:iy + ph, :].copy_(d_input, non_blocking=True)
                    if DEBUG:
                        sliced_grad_X_cpu[:, :, iy:iy + ph, :].copy_(d_input, non_blocking=True)

                iy += ph
                i += 1
            batch_start = batch_end

        torch.cuda.synchronize()

        if ctx.prev_module_id is not None:
            BWD_ACCELERATOR_REGISTRY[ctx.prev_module_id].append(grad_X_gpu)
            if DEBUG:
                print(f"[{ctx.__class__.__name__}] Returning None to Prev Layer")
            ghost_grad = torch.zeros(X.shape, device=ctx.x_device, dtype=X.dtype)
            return ghost_grad, grad_weight, grad_bias, *([None] * 14)

        grad_ret = grad_X_cpu if ctx.return_cpu_grad else grad_X_gpu
        if DEBUG:
            ret_mag = grad_ret.abs().mean().item() if grad_ret is not None else -1.0
            print(f"[{ctx.__class__.__name__}] Returning Grad to Prev Layer | Mag: {ret_mag:.6e}")
        if grad_ret is None:
            grad_ret = torch.zeros(X.shape, device=ctx.x_device, dtype=X.dtype)
        return grad_ret, grad_weight, grad_bias, *([None] * 14)


class BigFusedBNReLU(nn.Module):
    def __init__(self, num_features, momentum=0.1, out_on_gpu=True, dtype=None, max_elements=6e6):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.out_on_gpu = out_on_gpu
        self._dtype = dtype if dtype else (torch.float16 if cc.amp_enabled else torch.float32)
        self.max_elements = max_elements

        self.weight = nn.Parameter(torch.ones(num_features, device=cc.cuda_device))
        self.bias = nn.Parameter(torch.zeros(num_features, device=cc.cuda_device))
        self.register_buffer('running_mean', torch.zeros(num_features, device=cc.cuda_device))
        self.register_buffer('running_var', torch.ones(num_features, device=cc.cuda_device))

        self.output = None
        self.X_grad = None
        self.gpu_cache = [] if out_on_gpu else None
        self.bwd_accelerator = []
        # Register this module's buckets
        GPU_CACHE_REGISTRY[id(self)] = self.gpu_cache
        BWD_ACCELERATOR_REGISTRY[id(self)] = self.bwd_accelerator

    def _get_output_tensor(self, X):
        B, C, H, W = X.shape
        if self.output is None:
            self.output = torch.empty(X.shape, device=cc.host_device, pin_memory=torch.cuda.is_available(),
                                      memory_format=torch.channels_last, dtype=self._dtype)
            return self.output

        cur_B, _, cur_H, cur_W = self.output.shape
        if B > cur_B or H > cur_H or W > cur_W:
            max_B, max_H, max_W = max(B, cur_B), max(H, cur_H), max(W, cur_W)
            # self.output.requires_grad_(False)
            self.output.resize_((max_B, self.num_features, max_H, max_W)).contiguous(
                memory_format=torch.channels_last)
            if not self.output.is_pinned():
                self.output.pin_memory()

        sliced = self.output[:B, :, :H, :W]
        # sliced.requires_grad_(self.training)
        return sliced

    def _get_input_grad_tensor(self, X):
        B, C, H, W = X.shape
        if self.X_grad is None:
            self.X_grad = torch.zeros(X.shape, pin_memory=torch.cuda.is_available(), device=cc.host_device,
                                      dtype=self._dtype).to(memory_format=torch.channels_last)
            return self.X_grad

        cur_B, _, cur_H, cur_W = self.X_grad.shape
        if B > cur_B or H > cur_H or W > cur_W:
            max_B, max_H, max_W = max(B, cur_B), max(H, cur_H), max(W, cur_W)
            self.X_grad.resize_((max_B, self.num_features, max_H, max_W)).contiguous(
                memory_format=torch.channels_last)
            if not self.X_grad.is_pinned():
                self.X_grad.pin_memory()

        return self.X_grad[:B, :, :H, :W]

    def forward(self, x, mean=None, var=None, hot_potato=True):
        out = self._get_output_tensor(x)
        x_grad = out  # self._get_input_grad_tensor(x)
        if x_grad is not None:
            x_grad.zero_()

        if self.gpu_cache: self.gpu_cache.clear()
        if self.bwd_accelerator: self.bwd_accelerator.clear()

        prev_id = getattr(x, '_prev_module_id', None)

        out, rm, rv = BigFusedBNReLUFunction.apply(
            x, self.weight, self.bias, self.running_mean, self.running_var, self.momentum,
            out, x_grad, self.training, self.out_on_gpu, hot_potato,
            self.max_elements, torch.channels_last,
            id(self), prev_id,
            mean, var
        )

        if self.training:
            self.running_mean.copy_(rm.detach())
            self.running_var.copy_(rv.detach())

        if hot_potato:
            if self.out_on_gpu and self.gpu_cache:
                out._big_gpu_cache = self.gpu_cache.pop()
                out._prev_module_id = id(self)
        else:
            # Ensure no stale attributes exist
            if hasattr(out, '_bwd_accelerator'): del out._bwd_accelerator
            if hasattr(out, '_big_gpu_cache'): del out._big_gpu_cache
        return out


class BigFusedBNAddReLUFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    @torch.no_grad()
    def forward(ctx, X, identity, weight, bias, running_mean, running_var, momentum,
                output=None, X_grad=None, identity_grad=None, training=True,
                out_on_gpu=True, hot_potato=True,
                max_elements=6e6, memory_format=torch.channels_last,
                module_id=None, main_id=None, identity_id=None,
                external_mean=None, external_var=None):

        device = weight.device
        X_gpu_cache = getattr(X, '_big_gpu_cache', None)
        if X_gpu_cache is not None:
            del X._big_gpu_cache
            X_input = X_gpu_cache
        else:
            X_input = X

        Identity_gpu_cache = getattr(identity, '_big_gpu_cache', None)
        if Identity_gpu_cache is not None:
            del identity._big_gpu_cache
            I_input = Identity_gpu_cache
            identity_is_gpu = True
        else:
            I_input = identity
            identity_is_gpu = identity.is_cuda

        # prev_bwd_accelerator = getattr(X, '_bwd_accelerator', None)
        # Logic: If we are outputting to GPU, we generally want to return
        # a GPU gradient (ctx.return_cpu_grad = False).
        # EXCEPT: If we are at the very start of the model (Stem) and
        # the original input was CPU, we must eventually return CPU.
        if out_on_gpu:
            if main_id is not None:
                # Middle layer in an accelerated chain: Stay on GPU
                ctx.return_cpu_grad = False
            else:
                # Exit or Standalone layer: Return what the input X was
                ctx.return_cpu_grad = not X.is_cuda
        else:
            # Slow path: Always match the input X
            ctx.return_cpu_grad = not X.is_cuda

        if training:
            if external_mean is not None and external_var is not None:
                mean = external_mean
                # FIX: Convert Unbiased Var (N-1) to Biased Var (N)
                # n = X.numel() / X.size(1)
                variance = external_var #* ((n - 1) / n)
                running_mean = momentum * mean + (1 - momentum) * running_mean
                running_var = momentum * variance + (1 - momentum) * running_var
            else:
                mean, m2 = BigBatchNormFunction.compute_stats_welford_algo(X_input, max_elements)
                n_total = X.numel() / X.size(1)
                variance = m2 / n_total  # Biased

                unbiased_var = m2 / (n_total - 1)
                running_mean = momentum * mean + (1 - momentum) * running_mean
                running_var = momentum * unbiased_var + (1 - momentum) * running_var
        else:
            mean = running_mean
            variance = running_var

        ctx.x_device = X.device
        ctx.id_device = identity.device
        ctx.save_for_backward(X if not X.is_cuda else X.cpu(), identity if not identity.is_cuda else identity.cpu(),
                              weight, bias)
        ctx.mean = mean
        ctx.variance = variance
        ctx.X_grad = X_grad
        ctx.identity_grad = identity_grad
        ctx.max_elements = max_elements
        ctx.memory_format = memory_format
        # ctx.prev_bwd_accelerator = prev_bwd_accelerator
        ctx.main_id = main_id
        ctx.identity_id = identity_id
        ctx.module_id = module_id
        ctx.identity_requires_grad = identity.requires_grad

        B, C, H, W = X.shape
        inv_std = torch.rsqrt(variance + 1e-5)
        scale = weight * inv_std
        shift = bias - mean * scale

        gpu_out = None
        if out_on_gpu:
            gpu_out = torch.empty(output.shape, device=device)

        max_batch = max(1, min(B, int(max_elements // (C * H * W))))
        streams = cc.get_cuda_streams()

        batch_start = 0
        while batch_start < B:
            batch_end = min(batch_start + max_batch, B)

            if X_input.is_cuda:
                sliced_X = X_input[batch_start:batch_end]
            else:
                sliced_X = X_input[batch_start:batch_end]

            if identity_is_gpu:
                sliced_I = I_input[batch_start:batch_end]
            else:
                sliced_I = I_input[batch_start:batch_end]

            sliced_out_cpu = output[batch_start:batch_end]

            iy, i = 0, 0
            while iy < H:
                ph = min(max(1, int(max_elements // (sliced_X.size(0) * C * W))), H - iy)
                with torch.cuda.stream(streams[i % len(streams)]):
                    if sliced_X.is_cuda:
                        im = sliced_X[:, :, iy:iy + ph, :]
                    else:
                        im = sliced_X[:, :, iy:iy + ph, :].to(device, non_blocking=True)

                    if identity_is_gpu:
                        id_tile = sliced_I[:, :, iy:iy + ph, :]
                    else:
                        id_tile = sliced_I[:, :, iy:iy + ph, :].to(device, non_blocking=True)

                    # Normalize
                    im = im * scale[None, :, None, None] + shift[None, :, None, None]
                    # Add
                    im += id_tile
                    # ReLU
                    im = F.relu(im)

                    if out_on_gpu:
                        gpu_out[batch_start:batch_end, :, iy:iy + ph, :].copy_(im)
                    sliced_out_cpu[:, :, iy:iy + ph, :].copy_(im, non_blocking=True)

                iy += ph
                i += 1
            batch_start = batch_end

        for s in streams:
            s.synchronize()

        if out_on_gpu:
            if hot_potato:
                if module_id is not None:
                    GPU_CACHE_REGISTRY[module_id].append(gpu_out)
                return output, running_mean, running_var
            else:
                gpu_out.requires_grad_(True)
                return gpu_out, running_mean, running_var
        return output, running_mean, running_var

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_out, grad_rm, grad_rv):
        X, Identity, weight, bias = ctx.saved_tensors
        # if ctx.main_id is not None:
        # else:
        #
        # if ctx.identity_id is not None:
        # else:

        if ctx.module_id and len(BWD_ACCELERATOR_REGISTRY[ctx.module_id]) > 0:
            grad_out_gpu = BWD_ACCELERATOR_REGISTRY[ctx.module_id].pop()
            while len(BWD_ACCELERATOR_REGISTRY[ctx.module_id]) > 0:
                grad_out_gpu = grad_out_gpu + BWD_ACCELERATOR_REGISTRY[ctx.module_id].pop()
            use_gpu_grad = True
        else:
            grad_out_gpu = None
            use_gpu_grad = False

        X, Identity, weight, bias = ctx.saved_tensors
        mean, variance = ctx.mean, ctx.variance
        N, C, H, W = X.shape
        device = weight.device

        grad_bias = torch.zeros(C, device=device)
        grad_weight = torch.zeros(C, device=device)

        grad_X_cpu = ctx.X_grad
        grad_I_cpu = ctx.identity_grad
        # if grad_X_cpu is not None:
        #     grad_X_cpu.zero_()

        # FIX: Ensure CPU buffer allocated if returning CPU grad
        if grad_X_cpu is None and ctx.return_cpu_grad:
            grad_X_cpu = torch.zeros(X.shape, device=cc.host_device, pin_memory=torch.cuda.is_available()).to(
                memory_format=torch.channels_last)

        need_gpu_buffer = ctx.main_id is not None
        grad_X_gpu = torch.zeros(X.shape, device=device) if need_gpu_buffer else None

        # Add a GPU buffer for the shortcut gradient
        # need_identity_gpu_buffer = (ctx.identity_bwd_accelerator is not None) or \
        #                            (not ctx.return_cpu_grad and ctx.identity_requires_grad)
        # grad_I_gpu = torch.zeros(X.shape, device=device) if need_identity_gpu_buffer else None
        need_id_gpu = (ctx.identity_id is not None) or (Identity.is_cuda)
        grad_I_gpu = torch.zeros(X.shape, device=device) if need_id_gpu else None
        # grad_I_cpu = torch.zeros_like(Identity) if (
        #             not Identity.is_cuda and ctx.identity_id is None) else None

        if DEBUG:
            log_grad("TAIL_INCOMING", grad_out_gpu if use_gpu_grad else grad_out)

        max_batch = max(1, min(N, int(ctx.max_elements // (C * H * W))))
        inv_std = torch.rsqrt(variance + 1e-5)
        scale = weight * inv_std
        shift = bias - mean * scale

        streams = cc.get_cuda_streams()

        # --- PASS 1: GLOBAL ACCUMULATION ---
        batch_start = 0
        while batch_start < N:
            batch_end = min(batch_start + max_batch, N)

            if X.is_cuda:
                x_batch = X[batch_start:batch_end]
            else:
                x_batch = X[batch_start:batch_end]

            if Identity.is_cuda:
                i_batch = Identity[batch_start:batch_end]
            else:
                i_batch = Identity[batch_start:batch_end]

            if use_gpu_grad:
                g_batch = grad_out_gpu[batch_start:batch_end]
            else:
                g_batch = grad_out[batch_start:batch_end]

            if grad_I_cpu is not None:
                g_I_slice = grad_I_cpu[batch_start:batch_end]

            iy, i = 0, 0
            while iy < H:
                ph = min(max(1, int(ctx.max_elements // (x_batch.size(0) * C * W))), H - iy)
                with torch.cuda.stream(streams[i % len(streams)]):
                    if x_batch.is_cuda:
                        im = x_batch[:, :, iy:iy + ph, :]
                    else:
                        im = x_batch[:, :, iy:iy + ph, :].to(device, non_blocking=True)

                    if i_batch.is_cuda:
                        id_tile = i_batch[:, :, iy:iy + ph, :]
                    else:
                        id_tile = i_batch[:, :, iy:iy + ph, :].to(device, non_blocking=True)

                    if use_gpu_grad:
                        dy = g_batch[:, :, iy:iy + ph, :]
                    else:
                        dy = g_batch[:, :, iy:iy + ph, :].to(device, non_blocking=True)

                    val = im * scale[None, :, None, None] + shift[None, :, None, None]
                    pre_relu = val + id_tile
                    mask = (pre_relu > 0).float()

                    d_pre = dy * mask

                    if grad_I_cpu is not None:
                        g_I_slice[:, :, iy:iy + ph, :].copy_(d_pre, non_blocking=True)

                    if grad_I_gpu is not None:
                        grad_I_gpu[batch_start:batch_end, :, iy:iy + ph, :].copy_(d_pre)

                    grad_bias += d_pre.sum(dim=(0, 2, 3))
                    im_centered = im - mean[None, :, None, None]
                    x_hat = im_centered * inv_std[None, :, None, None]
                    grad_weight += (d_pre * x_hat).sum(dim=(0, 2, 3))

                iy += ph
                i += 1
            batch_start = batch_end
        for s in streams: s.synchronize()

        # --- PASS 2: COMPUTE DX (Main Path) ---
        N_elements = N * H * W
        term1 = (weight * inv_std)[None, :, None, None]
        sum_dy = grad_bias[None, :, None, None]
        sum_dy_xhat = grad_weight[None, :, None, None]

        batch_start = 0
        while batch_start < N:
            batch_end = min(batch_start + max_batch, N)

            if X.is_cuda:
                x_batch = X[batch_start:batch_end]
            else:
                x_batch = X[batch_start:batch_end]

            if Identity.is_cuda:
                i_batch = Identity[batch_start:batch_end]
            else:
                i_batch = Identity[batch_start:batch_end]

            if use_gpu_grad:
                g_batch = grad_out_gpu[batch_start:batch_end]
            else:
                g_batch = grad_out[batch_start:batch_end]

            if grad_X_cpu is not None: sliced_grad_X_cpu = grad_X_cpu[batch_start:batch_end]

            iy, i = 0, 0
            while iy < H:
                ph = min(max(1, int(ctx.max_elements // (x_batch.size(0) * C * W))), H - iy)
                with torch.cuda.stream(streams[i % len(streams)]):
                    if x_batch.is_cuda:
                        im = x_batch[:, :, iy:iy + ph, :]
                    else:
                        im = x_batch[:, :, iy:iy + ph, :].to(device, non_blocking=True)

                    if i_batch.is_cuda:
                        id_tile = i_batch[:, :, iy:iy + ph, :]
                    else:
                        id_tile = i_batch[:, :, iy:iy + ph, :].to(device, non_blocking=True)

                    if use_gpu_grad:
                        dy = g_batch[:, :, iy:iy + ph, :]
                    else:
                        dy = g_batch[:, :, iy:iy + ph, :].to(device, non_blocking=True)

                    val = im * scale[None, :, None, None] + shift[None, :, None, None]
                    mask = ((val + id_tile) > 0).float()
                    d_pre = dy * mask

                    x_hat = (im - mean[None, :, None, None]) * inv_std[None, :, None, None]
                    d_input = (term1 / N_elements) * (N_elements * d_pre - sum_dy - x_hat * sum_dy_xhat)

                    # if DEBUG:
                    #     # This reveals the exact signal magnitude being returned to the previous layer
                    # f"  [Pass 2 Debug] Tile iy={iy} | d_pre (from ReLU) mean: {d_pre.abs().mean().item():.4e}")
                    # f"  [Pass 2 Debug] Tile iy={iy} | d_input (Main Path) mean: {d_input.abs().mean().item():.4e}")

                    if grad_X_gpu is not None:
                        grad_X_gpu[batch_start:batch_end, :, iy:iy + ph, :].copy_(d_input)
                    elif grad_X_cpu is not None:
                        sliced_grad_X_cpu[:, :, iy:iy + ph, :].copy_(d_input, non_blocking=True)
                    # if DEBUG:
                    #     sliced_grad_X_cpu[:, :, iy:iy + ph, :].copy_(d_input, non_blocking=True)

                iy += ph
                i += 1
            batch_start = batch_end

        torch.cuda.synchronize()

        # Handle Main Path Bucket: ALWAYS push GPU, return None to PyTorch
        if ctx.main_id is not None:
            BWD_ACCELERATOR_REGISTRY[ctx.main_id].append(grad_X_gpu)  # Push the GPU buffer, NOT the CPU grad_ret
            final_main = None
        else:
            # Stem Exit: Hand off to PyTorch in the requested format
            final_main = grad_X_cpu if ctx.return_cpu_grad else grad_X_gpu

        # Handle Identity Path Bucket: ALWAYS push GPU, return None to PyTorch
        if ctx.identity_id is not None:
            BWD_ACCELERATOR_REGISTRY[ctx.identity_id].append(grad_I_gpu)  # Push the GPU buffer
            final_id = None
        else:
            # Exit to standard Autograd
            final_id = grad_I_cpu if ctx.return_cpu_grad else grad_I_gpu

        # Tagging for audit
        if final_id is not None: final_id._origin = "TAIL_IDENTITY_PATH"
        if final_main is not None: final_main._origin = "TAIL_MAIN_PATH"

        if DEBUG:
            # PROBE 1: The ReLU Mask Signal
            print(f"[PROBE TAIL] d_pre Mag: {d_pre.abs().mean().item():.6e}")
            # PROBE 2: The Main Path result
            if grad_X_gpu is not None:
                print(f"[PROBE TAIL] grad_X_gpu Mag: {grad_X_gpu.abs().mean().item():.6e}")
            log_grad("TAIL_MAIN_OUT", d_input)
            log_grad("TAIL_ID_OUT", d_pre)
            if final_main is not None:
                ret_mag = final_main.abs().mean().item() if final_main is not None else -1.0
                print(f"[{ctx.__class__.__name__}] Returning final_main Grad to Prev Layer | Mag: {ret_mag:.6e}")
            if final_id is not None:
                ret_mag = final_id.abs().mean().item() if final_id is not None else -1.0
                print(f"[{ctx.__class__.__name__}] Returning final_id Grad to Prev Layer | Mag: {ret_mag:.6e}")

        if final_main is None:
            final_main = torch.zeros(X.shape, device=ctx.x_device, dtype=X.dtype)
        if final_id is None:
            final_id = torch.zeros(Identity.shape, device=ctx.id_device, dtype=Identity.dtype)

        return final_main, final_id, grad_weight, grad_bias, *([None] * 16)


class BigFusedBNAddReLU(nn.Module):
    def __init__(self, num_features, momentum=0.1, out_on_gpu=True, dtype=None, max_elements=6e6):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.out_on_gpu = out_on_gpu
        self._dtype = dtype if dtype else (torch.float16 if cc.amp_enabled else torch.float32)
        self.max_elements = max_elements

        self.weight = nn.Parameter(torch.ones(num_features, device=cc.cuda_device))
        self.bias = nn.Parameter(torch.zeros(num_features, device=cc.cuda_device))
        self.register_buffer('running_mean', torch.zeros(num_features, device=cc.cuda_device))
        self.register_buffer('running_var', torch.ones(num_features, device=cc.cuda_device))

        self.output = None
        self.X_grad = None
        self.identity_grad = None
        self.gpu_cache = [] if out_on_gpu else None
        self.bwd_accelerator = []
        # Register this module's buckets
        GPU_CACHE_REGISTRY[id(self)] = self.gpu_cache
        BWD_ACCELERATOR_REGISTRY[id(self)] = self.bwd_accelerator

    def _get_cpu_tensor_buffer(self, X, buffer):
        B, C, H, W = X.shape
        if buffer is None:
            buffer = torch.zeros(X.shape, pin_memory=torch.cuda.is_available(), device=cc.host_device,
                                 dtype=self._dtype).to(memory_format=torch.channels_last)
            return buffer

        cur_B, _, cur_H, cur_W = buffer.shape
        if B > cur_B or H > cur_H or W > cur_W:
            max_B, max_H, max_W = max(B, cur_B), max(H, cur_H), max(W, cur_W)
            buffer.resize_((max_B, self.num_features, max_H, max_W)).contiguous(
                memory_format=torch.channels_last)
            if not buffer.is_pinned():
                buffer.pin_memory()

        return buffer[:B, :, :H, :W]

    def forward(self, x, identity, mean=None, var=None, hot_potato=True):
        out = self._get_cpu_tensor_buffer(x, self.output)
        x_grad = out  # self._get_input_grad_tensor(x, self.X_grad)
        identity_grad = self._get_cpu_tensor_buffer(x, self.identity_grad)
        if x_grad is not None:
            x_grad.zero_()

        if self.gpu_cache: self.gpu_cache.clear()
        if self.bwd_accelerator: self.bwd_accelerator.clear()

        main_id = getattr(x, '_main_id', None)
        identity_id = getattr(identity, '_identity_id', None)

        # if getattr(x, '_big_gpu_cache', None) is not None:
        # else:
        out, rm, rv = BigFusedBNAddReLUFunction.apply(
            x, identity, self.weight, self.bias, self.running_mean, self.running_var, self.momentum,
            out, x_grad, identity_grad, self.training, self.out_on_gpu, hot_potato,
            self.max_elements, torch.channels_last,
            id(self), main_id, identity_id,
            mean, var
        )

        if self.training:
            self.running_mean.copy_(rm.detach())
            self.running_var.copy_(rv.detach())

        if hot_potato:
            if self.out_on_gpu and self.gpu_cache:
                out._big_gpu_cache = self.gpu_cache.pop()
                out._prev_module_id = id(self)
        else:
            # Ensure no stale attributes exist
            if hasattr(out, '_bwd_accelerator'): del out._bwd_accelerator
            if hasattr(out, '_big_gpu_cache'): del out._big_gpu_cache
        return out


class BigBlockJunctionFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    @torch.no_grad()
    def forward(ctx, x, main_id, identity_id, prev_module_id):
        # 1. Clean up the incoming potato BEFORE making views
        # Store the original accelerator from the input
        ctx.main_id = main_id
        ctx.identity_id = identity_id
        ctx.prev_module_id = prev_module_id
        ctx.x_device = x.device

        # Create a clean view for the branches
        x_view = x.view_as(x)
        # if hasattr(x_view, '_bwd_accelerator'):
        #     del x_view._bwd_accelerator
        # if hasattr(x_view, '_big_gpu_cache'):
        #     del x_view._big_gpu_cache
        # TRANSFER THESE
        if hasattr(x, '_big_gpu_cache'):
            x_view._big_gpu_cache = x._big_gpu_cache
        if hasattr(x, '_prev_module_id'):
            x_view._prev_module_id = x._prev_module_id

        return x_view

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_out):
        main_mailbox = BWD_ACCELERATOR_REGISTRY[ctx.main_id]
        id_mailbox = BWD_ACCELERATOR_REGISTRY[ctx.identity_id]
        # if grad_out is not None:

        grad_sum = None
        # 1. Main Path
        while len(main_mailbox) > 0:
            val = main_mailbox.pop()
            if grad_sum is None:
                grad_sum = val
            else:
                grad_sum.add_(val)

        # 2. Identity Path
        while len(id_mailbox) > 0:
            val = id_mailbox.pop()
            if grad_sum is None:
                grad_sum = val
            else:
                grad_sum.add_(val)

        # If we found gradients in the buckets, prioritize them
        if grad_sum is not None:
            # ONLY add grad_out if it is already on the GPU (meaning a real layer used it)
            # If it's on CPU, it's a Ghost. Adding a 0-sum is a no-op,
            # but materializing it to GPU costs 8GB. SKIP IT.
            if grad_out is not None and grad_out.device.type == 'cuda':
                grad_sum.add_(grad_out)

            if ctx.prev_module_id is not None:
                BWD_ACCELERATOR_REGISTRY[ctx.prev_module_id].append(grad_sum)
                ghost = torch.zeros(grad_out.shape, device=grad_out.device)
                return ghost, *([None] * 3)  # Kill highway exit
            return grad_sum.to(ctx.x_device), *([None] * 3)
        # If buckets were empty (CPU path), return the standard grad_out
        return grad_out, *([None] * 3)


class BigBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample_module=None, groups=1,
                 base_width=64, dilation=1, out_on_gpu=True, max_elements=6e6):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.stride = stride
        self.out_on_gpu = out_on_gpu

        # 1. Head: Conv1x1 -> BN+ReLU
        # We need Conv1 to compute stats for BN1
        self.conv1 = BigConv2dStats(inplanes, width, kernel_size=1, stride=1, padding=0, out_on_gpu=out_on_gpu,
                                    max_elements=max_elements)
        self.bn1 = BigFusedBNReLU(width, out_on_gpu=out_on_gpu, max_elements=max_elements)

        # 2. Body: Conv3x3 -> BN+ReLU
        self.conv2 = BigConv2dStats(width, width, kernel_size=3, stride=stride, padding=dilation,
                                    dilation=dilation, out_on_gpu=out_on_gpu, max_elements=max_elements)
        self.bn2 = BigFusedBNReLU(width, out_on_gpu=out_on_gpu, max_elements=max_elements)

        # 3. Tail: Conv1x1 -> BN+Add+ReLU
        self.conv3 = BigConv2dStats(width, planes * self.expansion, kernel_size=1, stride=1, padding=0,
                                    out_on_gpu=out_on_gpu, max_elements=max_elements)
        self.tail = BigFusedBNAddReLU(planes * self.expansion, out_on_gpu=out_on_gpu, max_elements=max_elements)

        self.downsample = downsample_module  # Optional

        self.junction_main = []
        self.junction_id = []
        # REGISTER
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

        # --- MAIN PATH STITCHING ---
        out, m1, v1 = self.conv1(x_main, return_hot_potato=True)
        out._prev_module_id = id(self.conv1)
        out = self.bn1(out, m1, v1, hot_potato=True)
        out._prev_module_id = id(self.bn1)
        out, m2, v2 = self.conv2(out, return_hot_potato=True)
        out._prev_module_id = id(self.conv2)
        out = self.bn2(out, m2, v2, hot_potato=True)
        out._prev_module_id = id(self.bn2)
        out, m3, v3 = self.conv3(out, return_hot_potato=True)

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
        setattr(out, '_main_id', id(self.conv3))
        setattr(identity, '_identity_id', identity_id)

        # RULE: Cleanup local reference before entering Tail to free memory
        del cache_ref

        return self.tail(out, identity, m3, v3, hot_potato=hot_potato)
