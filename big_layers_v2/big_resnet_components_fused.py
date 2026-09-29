"""EXPERIMENTAL: archived hot-potato path, not used by the paper training entry points."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from big_layers.amp_compat import custom_bwd, custom_fwd
from . import cuda_config as cc
from .cuda_config import GPU_CACHE_REGISTRY, BWD_ACCELERATOR_REGISTRY

from .convolution_bridge import cpp_conv
USE_NEW = True

DEBUG = False  # TODO: Set to False for real usage
audit_log = {}


def log_grad(name, tensor):
    if tensor is not None:
        audit_log[name] = tensor.detach().abs().mean().item()
# ==============================================================================
# 1. BigConv2dStats (Conv + Welford Stats)
# ==============================================================================

class BigConv2dStatsFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    @torch.no_grad()
    def forward(ctx, X, weight, bias=None, stride=1, padding=0, dilation=1,
                output=None, X_grad=None,
                module_id=None, prev_module_id=None,
                max_elements=6e7, out_on_gpu=False, training_stats=True,
                hot_potato=True):

        # VISIBILITY CHECK
        # if prev_module_id is not None:
        # else:

        device = weight.device
        X_gpu_cache = getattr(X, '_big_gpu_cache', None)
        if X_gpu_cache is not None:
            del X._big_gpu_cache
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
        ctx.x_device = X.device
        saved_X = X._big_cpu_out if hasattr(X, '_big_cpu_out') else X
        ctx.save_for_backward(saved_X, weight, bias)
        ctx.X_grad = X_grad
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.max_elements = max_elements
        ctx.return_hot_potato = hot_potato
        ctx.grad_reqs = (X.requires_grad, weight.requires_grad)
        ctx.module_id = module_id
        ctx.prev_module_id = prev_module_id

        gpu_out = None
        if out_on_gpu:
            gpu_out = torch.empty(output.shape, device=device)

        mean = torch.zeros(weight.shape[0], device=device)
        m2 = torch.zeros(weight.shape[0], device=device)
        n_count = 0

        max_batch_size = max(1, int(max_elements // X[0].numel()))
        max_batch_size = min(max_batch_size, X.shape[0])
        input_height = X.shape[2] + (2 * padding)
        effective_kernel_height = dilation * (weight.shape[2] - 1) + 1
        max_patch_height = max(1, int(max_elements // (max_batch_size * X.shape[1] * X.shape[3])))
        max_patch_height = max(effective_kernel_height, min(max_patch_height, input_height))

        streams = cc.get_cuda_streams()

        batch_start = 0
        while batch_start < X.shape[0]:
            batch_end = min(batch_start + max_batch_size, X.shape[0])

            if X_gpu_cache is not None:
                sliced_X = X_gpu_cache[batch_start:batch_end]
            else:
                sliced_X = X[batch_start:batch_end]

            slided_output = output[batch_start:batch_end]
            slided_gpu_out = gpu_out[batch_start:batch_end] if out_on_gpu else None

            output_offset_y = 0
            iy, out_i = 0, 0

            while iy < input_height - effective_kernel_height + 1:
                input_offset_y = iy - padding
                im_offset_y = 0
                patch_output_height_temp = ((max_patch_height - effective_kernel_height) // stride) + 1
                curr_patch_h = ((patch_output_height_temp - 1) * stride) + effective_kernel_height
                input_rows_required = min(curr_patch_h, X.shape[2])
                im_y_limit = curr_patch_h

                if iy < padding:
                    input_offset_y = 0
                    im_offset_y = padding - iy
                    input_rows_required = curr_patch_h - (padding - iy)
                elif iy - padding + curr_patch_h > X.shape[2]:
                    input_rows_required = X.shape[2] - (iy - padding)
                    im_y_limit = input_rows_required
                    if iy - padding + curr_patch_h > X.shape[2] + padding:
                        curr_patch_h = X.shape[2] + (2 * padding) - iy

                input_rows_required = min(input_rows_required, X.shape[2])
                im_y_limit = min(im_y_limit, im_offset_y + X.shape[2])
                curr_output_height = ((curr_patch_h - effective_kernel_height) // stride) + 1

                with torch.cuda.stream(streams[out_i % len(streams)]):
                    if im_y_limit - im_offset_y != curr_patch_h:
                        im = torch.zeros((sliced_X.shape[0], X.shape[1], curr_patch_h, X.shape[3]),
                                         device=device)
                        im[:, :, im_offset_y:im_y_limit, :].copy_(
                            sliced_X[:, :, input_offset_y:(input_offset_y + input_rows_required), :], non_blocking=True)
                    else:
                        if X_gpu_cache is not None:
                            im = sliced_X[:, :, input_offset_y:(input_offset_y + input_rows_required), :]
                        else:
                            im = sliced_X[:, :, input_offset_y:(input_offset_y + input_rows_required), :].to(
                                device=device, non_blocking=True, dtype=weight.dtype)

                    im = im.to(weight.dtype)
                    conv_res = F.conv2d(im, weight, bias, stride, padding=(0, padding), dilation=dilation)

                    if training_stats:
                        flat_res = conv_res.permute(1, 0, 2, 3).flatten(1)
                        new_n = flat_res.shape[1]
                        new_mean = flat_res.mean(dim=1)
                        new_m2 = ((flat_res - new_mean.unsqueeze(1)) ** 2).sum(dim=1)
                        delta = new_mean - mean
                        total_n = n_count + new_n
                        mean += delta * (new_n / total_n)
                        m2 += new_m2 + delta ** 2 * (n_count * new_n / total_n)
                        n_count = total_n

                    if out_on_gpu:
                        slided_gpu_out[:, :, output_offset_y:(output_offset_y + curr_output_height), :].copy_(conv_res)
                    slided_output[:, :, output_offset_y:(output_offset_y + curr_output_height), :].copy_(conv_res,
                                non_blocking=True)

                output_offset_y += curr_output_height
                out_i += 1
                iy += curr_patch_h - effective_kernel_height + stride

            batch_start += max_batch_size

        if torch.cuda.is_available():
            for s in streams: s.synchronize()

        var = m2 / n_count if n_count > 1 else torch.zeros_like(mean)

        if out_on_gpu:
            if hot_potato:
                if module_id is not None:
                    GPU_CACHE_REGISTRY[module_id].append(gpu_out)
                # else:
                return output, mean, var
            else:
                gpu_out.requires_grad_(True)
                return gpu_out, mean, var

        return output, mean, var

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_out, grad_mean, grad_var):
        X, weight, bias = ctx.saved_tensors
        if DEBUG:
            # --- DIAGNOSTIC AUDIT ---
            std_mag = grad_out.abs().mean().item() if hasattr(grad_out, 'abs') else 0.0
            acc_mag = 0.0
            if ctx.my_bwd_accelerator and len(ctx.my_bwd_accelerator) > 0:
                acc_mag = ctx.my_bwd_accelerator[-1].abs().mean().item()

            print(f"[AUDIT CONV] Std Path Mag: {std_mag:.4e} | Bucket Mag: {acc_mag:.4e}")
            if std_mag > 1e-8 and acc_mag > 1e-8:
                print(f"  !! WARNING: Potential Double Counting detected in Conv !!")
            # ------------------------
            # --- AUDIT LOG ---
            if ctx.my_bwd_accelerator is not None:
                print(
                    f"[CONSUME AUDIT: CONV] Bucket ID: {id(ctx.my_bwd_accelerator)}"
                    f" | Size: {len(ctx.my_bwd_accelerator)}")
                if len(ctx.my_bwd_accelerator) > 0:
                    top_val = ctx.my_bwd_accelerator[-1].abs().mean().item()
                    print(f"  - Found Potato! Value: {top_val:.4e} | Potato ID: {id(ctx.my_bwd_accelerator[-1])}")
            origin = getattr(grad_out, '_origin', 'Autograd_Standard')
            bucket_len = len(ctx.my_bwd_accelerator) if ctx.my_bwd_accelerator is not None else 0
            print(
                f"[AUDIT: CONV_LAYER] Entry | Origin: {origin} | Bucket Size: {bucket_len}"
                f" | Mean: {grad_out.abs().mean().item():.4e}")
            # -----------------

        grad_out_bucket = None
        if ctx.module_id and len(BWD_ACCELERATOR_REGISTRY[ctx.module_id]) > 0:
            grad_out_bucket = BWD_ACCELERATOR_REGISTRY[ctx.module_id].pop()
            while len(BWD_ACCELERATOR_REGISTRY[ctx.module_id]) > 0:
                grad_out_bucket.add_(BWD_ACCELERATOR_REGISTRY[ctx.module_id].pop())

        if grad_out_bucket is not None:
            grad = grad_out_bucket
            grad_is_cuda = True
        elif hasattr(grad_out, 'is_cuda') and grad_out.is_cuda:
            grad = grad_out
            grad_is_cuda = True
        else:
            grad = grad_out
            grad_is_cuda = False

        X, weight, bias = ctx.saved_tensors
        B, C_in, H_in, W_in = X.shape
        needs_input_grad, needs_weight_grad = ctx.grad_reqs
        device = weight.device

        if DEBUG:
            k = weight.shape[2]
            name = f"CONV_{k}x{k}"
            log_grad(f"{name}_INCOMING", grad if grad_is_cuda else grad_out)

        kernel_height, kernel_width = weight.shape[2:]
        effective_kernel_height = ctx.dilation * (kernel_height - 1) + 1
        input_height = H_in + (2 * ctx.padding)
        output_height = ((input_height - effective_kernel_height) // ctx.stride) + 1

        max_batch_size = max(1, int(ctx.max_elements // X[0].numel()))
        max_batch_size = min(max_batch_size, B)
        max_patch_height = int(ctx.max_elements // (max_batch_size * C_in * W_in))
        max_patch_height = max(effective_kernel_height, min(max_patch_height, input_height))
        patch_out_h = max(1, ((max_patch_height - effective_kernel_height) // ctx.stride) + 1)

        # Init Grads
        grad_X = ctx.X_grad
        # if grad_X is not None:
        #     grad_X.zero_()

        # Always allocate if CPU return is needed
        if grad_X is None and ctx.return_cpu_grad and needs_input_grad:
            grad_X = torch.zeros(X.shape, device=cc.host_device, pin_memory=torch.cuda.is_available())

        # GPU Buffer for Accumulation
        # We need it if calculating input grad
        need_gpu_buf = (needs_input_grad and ((not ctx.return_cpu_grad) or (ctx.prev_module_id is not None)))
        grad_X_gpu = torch.zeros(X.shape, device=device) if need_gpu_buf else None

        grad_weight = torch.zeros(weight.shape, device=device) if needs_weight_grad else None
        grad_bias = torch.zeros(weight.shape[0], device=device) if bias is not None else None
        if grad_weight is None: grad_weight = torch.zeros_like(weight)
        target_dtype = grad.dtype
        weight_aligned = weight.to(target_dtype)

        streams = cc.get_cuda_streams()

        batch_start = 0
        while batch_start < B:
            batch_end = min(batch_start + max_batch_size, B)
            sliced_X = X[batch_start:batch_end]
            sliced_grad = grad[batch_start:batch_end]

            output_offset_y = 0
            iy, out_i = 0, 0
            while iy < input_height - effective_kernel_height + 1:
                input_offset_y = iy - ctx.padding
                im_offset_y = 0

                # Calculate how many output rows this patch produces
                patch_output_height_temp = ((max_patch_height - effective_kernel_height) // ctx.stride) + 1
                curr_patch_h = ((patch_output_height_temp - 1) * ctx.stride) + effective_kernel_height

                input_rows_required = min(curr_patch_h, H_in)
                im_y_limit = curr_patch_h

                # Padding Boundary Handling
                if iy < ctx.padding:
                    input_offset_y = 0
                    im_offset_y = ctx.padding - iy
                    input_rows_required = curr_patch_h - (ctx.padding - iy)
                elif iy - ctx.padding + curr_patch_h > H_in:
                    input_rows_required = H_in - (iy - ctx.padding)
                    im_y_limit = input_rows_required
                    if iy - ctx.padding + curr_patch_h > H_in + ctx.padding:
                        curr_patch_h = H_in + (2 * ctx.padding) - iy

                input_rows_required = min(input_rows_required, H_in)
                im_y_limit = min(im_y_limit, im_offset_y + H_in)

                # This must match the height of 'grads' we slice
                curr_output_height = ((curr_patch_h - effective_kernel_height) // ctx.stride) + 1

                mask = (needs_input_grad, needs_weight_grad)

                with torch.cuda.stream(streams[out_i % len(streams)]):
                    # Load Input
                    if im_y_limit - im_offset_y != curr_patch_h:
                        im = torch.zeros((sliced_X.shape[0], C_in, curr_patch_h, W_in), device=device)
                        if input_rows_required > 0:
                            im[:, :, im_offset_y:im_y_limit, :].copy_(
                        sliced_X[:, :, input_offset_y:input_offset_y + input_rows_required, :], non_blocking=True)
                    else:
                        im = torch.zeros((sliced_X.shape[0], C_in, curr_patch_h, W_in), device=device)
                        im.copy_(sliced_X[:, :, input_offset_y:input_offset_y + input_rows_required, :],
                                 non_blocking=True)

                    # Load Grad
                    if grad_is_cuda:
                        grads = sliced_grad[:, :, output_offset_y:(output_offset_y + curr_output_height), :]
                    else:
                        grads = torch.zeros((sliced_grad.shape[0], grad.shape[1], curr_output_height, grad.shape[3]),
                                            device=device)
                        grads.copy_(sliced_grad[:, :, output_offset_y:(output_offset_y + curr_output_height), :],
                                    non_blocking=True)

                    with torch.autocast(device_type='cuda', dtype=im.dtype, enabled=cc.amp_enabled):
                        im = im.to(weight_aligned.dtype)
                        grads = grads.to(weight_aligned.dtype)
                        if grad_bias is not None:
                            grad_bias += grads.to(torch.float32).sum(dim=(0, 2, 3))

                        if USE_NEW:
                            grad_input, grad_w, _ = cpp_conv.convolution_backward(
                                im, weight_aligned, grads, (ctx.stride, ctx.stride), (0, ctx.padding), (0, 0),
                                (ctx.dilation, ctx.dilation), 1, False, True, True, (mask[0], mask[1], False))
                        else:
                            grad_input, grad_w = cpp_conv.convolution_backward(
                                im, weight_aligned, grads, (ctx.stride, ctx.stride), (0, ctx.padding),
                                (ctx.dilation, ctx.dilation), 1, False, True, True, mask)

                    if needs_weight_grad:
                        grad_weight.add_(grad_w)

                    # Save Input Grad - MUST ACCUMULATE
                    if needs_input_grad:
                        if grad_X_gpu is not None:
                            # Use add_ because tiles overlap
                            grad_X_gpu[batch_start:batch_end, :, input_offset_y:(input_offset_y + input_rows_required),
                            :].add_(grad_input[:, :, im_offset_y:im_y_limit, :])

                        if grad_X is not None and not need_gpu_buf:
                            tmp = torch.empty_like(grad_input[:, :, im_offset_y:im_y_limit, :], device='cpu')
                            tmp.copy_(grad_input[:, :, im_offset_y:im_y_limit, :])
                            # Use += because tiles overlap
                            grad_X[batch_start:batch_end, :, input_offset_y:(input_offset_y + input_rows_required),
                            :] += tmp

                        # if grad_X is not None and DEBUG:
                        #     grad_X[batch_start:batch_end, ...].add_(grad_input[...].to('cpu', non_blocking=True))

                    if DEBUG:
                        # Safely probe the magnitude
                        mag = grad_X_gpu.abs().mean().item() if grad_X_gpu is not None else 0.0
                        print(f"[PROBE CONV] kernel dims = {kernel_height, kernel_width} grad_X_gpu Mag: {mag:.6e}")

                output_offset_y += curr_output_height
                out_i += 1
                iy += curr_patch_h - effective_kernel_height + ctx.stride

            batch_start += max_batch_size

        streams[0].synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 1. Start with the internal GPU buffer we just calculated
        # This is always on GPU because convolution_backward requires it
        grad_for_highway = grad_X_gpu

        # 2. Consume MY bucket (incoming high-precision signal)
        # if ctx.my_bwd_accelerator and len(ctx.my_bwd_accelerator) > 0:
        #     while len(ctx.my_bwd_accelerator) > 0:
        #         acc_grad = ctx.my_bwd_accelerator.pop().to(device)
        #         if grad_for_highway is None:
        #             grad_for_highway = acc_grad
        #         else:
        #             grad_for_highway = grad_for_highway + acc_grad

        if DEBUG:
            k = weight.shape[2]
            name = f"CONV_{k}x{k}"
            log_grad(f"{name}_DX", grad_input)

        # 3. Middle-Layer Logic: Push to the highway and stop standard Autograd
        if ctx.prev_module_id is not None:
            # Ensure we ONLY push the GPU tensor to the next bucket
            BWD_ACCELERATOR_REGISTRY[ctx.prev_module_id].append(grad_for_highway)
            # Return None to PyTorch to kill the "Ghost Path" and prevent double-counting
            if DEBUG:
                print(f"[{ctx.__class__.__name__}] Returning None to Prev Layer")
            target_device = ctx.x_device
            ghost_grad = torch.zeros(X.shape, device=target_device, dtype=X.dtype)
            return ghost_grad, grad_weight, grad_bias, *([None] * 11)

        # 4. Stem-Layer Logic: Hand-off to PyTorch
        # We only reach here if ctx.prev_bwd_accelerator is None
        if ctx.return_cpu_grad:
            # Use the CPU buffer provided by the module
            grad_ret = grad_X
            if grad_ret is not None and grad_for_highway is not None:
                # Final non-blocking copy back to Host
                grad_ret.copy_(grad_for_highway, non_blocking=True)
                torch.cuda.current_stream().synchronize()
            if DEBUG:
                ret_mag = grad_ret.abs().mean().item() if grad_ret is not None else -1.0
                print(f"[{ctx.__class__.__name__}] Returning Grad to Prev Layer | Mag: {ret_mag:.6e}")
            if grad_ret is None:
                grad_ret = torch.zeros(X.shape, device=ctx.x_device, dtype=X.dtype)
            return grad_ret, grad_weight, grad_bias, *([None] * 11)
        if DEBUG:
            ret_mag = grad_for_highway.abs().mean().item() if grad_for_highway is not None else -1.0
            print(f"[{ctx.__class__.__name__}] Returning Grad to Prev Layer | Mag: {ret_mag:.6e}")
        if grad_for_highway is None:
            target_device = ctx.x_device
            grad_for_highway = torch.zeros(X.shape, device=target_device, dtype=X.dtype)
        return grad_for_highway, grad_weight, grad_bias, *([None] * 11)


class BigConv2dStats(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=7, stride=2, padding=3, dilation=1, out_on_gpu=True,
                 max_elements=6e6, bias=False):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.out_on_gpu = out_on_gpu
        self.max_elements = max_elements
        self._dtype = torch.float16 if cc.amp_enabled else torch.float32

        self.weight = nn.Parameter(torch.empty((out_chans, in_chans, kernel_size, kernel_size), device=cc.cuda_device))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_chans, device=cc.cuda_device)) if bias else None

        self.output = None
        self.X_grad = None
        self.gpu_cache = [] if out_on_gpu else None
        self.bwd_accelerator = []

        # Register this module's buckets
        GPU_CACHE_REGISTRY[id(self)] = self.gpu_cache
        BWD_ACCELERATOR_REGISTRY[id(self)] = self.bwd_accelerator

    def _get_output_dimensions(self, X: torch.Tensor):
        kernel_height = self.weight.shape[2]
        kernel_width = self.weight.shape[3]
        effective_kernel_height = self.dilation * (kernel_height - 1) + 1
        effective_kernel_width = self.dilation * (kernel_width - 1) + 1
        input_height = X.shape[2] + (2 * self.padding)
        input_width = X.shape[3] + (2 * self.padding)
        output_height = ((input_height - effective_kernel_height) // self.stride) + 1
        output_width = ((input_width - effective_kernel_width) // self.stride) + 1
        return output_height, output_width

    def _get_output_tensor(self, X: torch.Tensor):
        h_out, w_out = self._get_output_dimensions(X)
        B = X.shape[0]

        if self.output is None:
            self.output = torch.empty((B, self.out_chans, h_out, w_out),
                                      device=cc.host_device, pin_memory=torch.cuda.is_available(),
                                      memory_format=torch.channels_last,
                                      dtype=self._dtype)
            self.output._big_gpu_cache = None
            return self.output

        cur_B, _, cur_h, cur_w = self.output.shape

        if cur_B >= B and cur_h >= h_out and cur_w >= w_out:
            # self.output.requires_grad_(False)
            sliced_output = self.output[:B, :, :h_out, :w_out]
            # sliced_output.requires_grad_(self.training)
            sliced_output._big_gpu_cache = None
            return sliced_output

        max_B = max(cur_B, B)
        max_h = max(cur_h, h_out)
        max_w = max(cur_w, w_out)

        # self.output.requires_grad_(False)
        self.output.resize_((max_B, self.out_chans, max_h, max_w)).contiguous()
        if not self.output.is_pinned():
            self.output.pin_memory()

        sliced_output = self.output[:B, :, :h_out, :w_out]
        # sliced_output.requires_grad_(self.training)
        sliced_output._big_gpu_cache = None
        return sliced_output

    def _get_input_grad_tensor(self, X: torch.Tensor):
        B, C, H, W = X.shape

        if self.X_grad is None:
            self.X_grad = torch.zeros(X.shape,
                pin_memory=torch.cuda.is_available(), device=cc.host_device, dtype=self._dtype).to(
                memory_format=torch.channels_last)
            return self.X_grad

        cur_B, cur_C, cur_H, cur_W = self.X_grad.shape

        if cur_B >= B and cur_H >= H and cur_W >= W:
            return self.X_grad[:B, :, :H, :W]

        max_B = max(cur_B, B)
        max_H = max(cur_H, H)
        max_W = max(cur_W, W)

        self.X_grad.resize_((max_B, C, max_H, max_W)).contiguous()
        if not self.X_grad.is_pinned():
            self.X_grad.pin_memory()
        return self.X_grad[:B, :, :H, :W]

    def forward(self, x, return_hot_potato=True):
        out = self._get_output_tensor(x)
        # Fix: Disable persistent buffer passing to ensure backward allocates fresh tensor
        x_grad = self._get_input_grad_tensor(x) if x.requires_grad and not self.out_on_gpu else None
        if x_grad is not None:
            x_grad.zero_()

        if self.gpu_cache: self.gpu_cache.clear()
        if self.bwd_accelerator: self.bwd_accelerator.clear()

        prev_id = getattr(x, '_prev_module_id', None)

        res, mean, var = BigConv2dStatsFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation,
            out, x_grad, id(self), prev_id,
            self.max_elements, self.out_on_gpu, self.training, return_hot_potato
        )

        if return_hot_potato:
            if self.out_on_gpu and self.gpu_cache:
                res._big_gpu_cache = self.gpu_cache.pop()
                res._prev_module_id = id(self)
            # else:
        else:
            #       f'return_hot_potato={return_hot_potato} for weight {self.weight.shape} '
            #       f'and input shape {x.shape}', flush=True)
            # Ensure no stale attributes exist
            if hasattr(res, '_bwd_accelerator'): del res._bwd_accelerator
            if hasattr(res, '_big_gpu_cache'): del res._big_gpu_cache

        return res, mean, var


class BigBatchNormFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    @torch.no_grad()
    def forward(ctx, X, weight, bias, running_mean, running_var, momentum,
                output=None, X_grad=None, training=True, relu=False,
                out_on_gpu=False, max_elements=6e7, memory_format=torch.channels_last,
                module_id=None, prev_module_id=None, external_mean=None, external_var=None):

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
                # Note: Assuming input variance is Biased (div N) or converting if Unbiased (div N-1)
                # BN Standard expects Biased variance for its calculations.
                # However, for generic BN (not fused with ConvStats), we often just use Welford.
                variance = external_var
                running_mean = momentum * mean + (1 - momentum) * running_mean
                running_var = momentum * variance + (1 - momentum) * running_var
            else:
                mean, m2 = BigBatchNormFunction.compute_stats_welford_algo(X_input, max_elements, weight.device)
                n_total = X.numel() / X.size(1)
                variance = m2 / n_total

                unbiased_var = m2 / (n_total - 1)
                running_mean = momentum * mean + (1 - momentum) * running_mean
                running_var = momentum * unbiased_var + (1 - momentum) * running_var
        else:
            mean = running_mean
            variance = running_var

        ctx.x_device = X.device
        ctx.save_for_backward(X if not X.is_cuda else X.cpu(), weight)
        ctx.mean = mean
        ctx.variance = variance
        ctx.X_grad = X_grad
        ctx.memory_format = memory_format
        ctx.max_elements = max_elements
        ctx.relu = bool(relu)
        ctx.module_id = module_id
        ctx.prev_module_id = prev_module_id

        gpu_out = BigBatchNormFunction.normalise(X_input, mean, variance, weight, bias, output, relu, max_elements,
                                                 out_on_gpu, memory_format)

        torch.cuda.empty_cache()

        # if out_on_gpu and hot_potato is not None:
        #     hot_potato.append(gpu_out)
        #
        # return output, running_mean, running_var

        if out_on_gpu:
            if module_id is not None:
                GPU_CACHE_REGISTRY[module_id].append(gpu_out)
                return output, running_mean, running_var
            else:
                gpu_out.requires_grad_(True)
                return gpu_out, running_mean, running_var

        return output, running_mean, running_var

    @staticmethod
    @torch.no_grad()
    def compute_stats_welford_algo(X, max_elements, device):
        # ... (Welford implementation same as before) ...
        max_batch_size = max(1, int(max_elements // X[0].numel()))
        max_batch_size = min(max_batch_size, X.shape[0])

        global_mean = torch.zeros(X.shape[1], device=device)
        global_m2 = torch.zeros(X.shape[1], device=device)
        global_n = 0.0

        streams = cc.get_cuda_streams()

        batch_start = 0
        while batch_start < X.shape[0]:
            batch_end = min(batch_start + max_batch_size, X.shape[0])
            if X.is_cuda:
                sliced_X = X[batch_start:batch_end]
            else:
                sliced_X = X[batch_start:batch_end]

            iy, i = 0, 0
            while iy < X.shape[2]:
                ph = int(max_elements // (sliced_X.size(0) * X.size(1) * X.size(3)))
                patch_height = min(max(1, ph), X.shape[2] - iy)
                with torch.cuda.stream(streams[i % len(streams)]):
                    if sliced_X.is_cuda:
                        im = sliced_X[:, :, iy:(iy + patch_height), :]
                    else:
                        im = sliced_X[:, :, iy:(iy + patch_height), :].to(device=device, non_blocking=True)

                    im_flat = im.view(im.size(0), im.size(1), -1)
                    n_chunk = im_flat.shape[0] * im_flat.shape[2]
                    chunk_mean = im_flat.mean(dim=(0, 2))
                    chunk_m2 = ((im_flat - chunk_mean.view(1, -1, 1)) ** 2).sum(dim=(0, 2))

                    if global_n == 0:
                        global_mean = chunk_mean
                        global_m2 = chunk_m2
                        global_n = n_chunk
                    else:
                        delta = chunk_mean - global_mean
                        new_n = global_n + n_chunk
                        global_m2 += chunk_m2 + (delta ** 2) * (global_n * n_chunk / new_n)
                        global_mean += delta * (n_chunk / new_n)
                        global_n = new_n
                iy += patch_height
                i += 1
            batch_start = batch_end
        for s in streams: s.synchronize()
        return global_mean, global_m2

    @staticmethod
    @torch.no_grad()
    def normalise(X, mean, variance, weight, bias, output, relu, max_elements, out_on_gpu, memory_format):
        device = weight.device
        max_batch_size = max(1, int(max_elements // X[0].numel()))
        max_batch_size = min(max_batch_size, X.shape[0])

        if output is None:
            output = torch.empty(X.shape, device='cpu', memory_format=memory_format,
                                 pin_memory=torch.cuda.is_available())

        inv_std = torch.rsqrt(variance + 1e-5)
        scale = weight * inv_std
        shift = bias - mean * scale

        gpu_out = torch.empty(X.shape, device=device) if out_on_gpu else None
        streams = cc.get_cuda_streams()

        batch_start = 0
        while batch_start < X.shape[0]:
            batch_end = min(batch_start + max_batch_size, X.shape[0])

            if X.is_cuda:
                sliced_X = X[batch_start:batch_end]
            else:
                sliced_X = X[batch_start:batch_end]
            sliced_output = output[batch_start:batch_end]

            iy, i = 0, 0
            while iy < X.shape[2]:
                ph = int(max_elements // (sliced_X.size(0) * X.size(1) * X.size(3)))
                patch_height = min(max(1, ph), X.shape[2] - iy)
                with torch.cuda.stream(streams[i % len(streams)]):
                    if sliced_X.is_cuda:
                        im = sliced_X[:, :, iy:(iy + patch_height), :]
                    else:
                        im = sliced_X[:, :, iy:(iy + patch_height), :].to(device=device, non_blocking=True)

                    im = im * scale[None, :, None, None] + shift[None, :, None, None]

                    if bool(relu): im = F.relu(im)

                    sliced_output[:, :, iy:(iy + patch_height), :].copy_(im, non_blocking=True)
                    if out_on_gpu: gpu_out[batch_start: batch_end, :, iy:(iy + patch_height), :].copy_(im)
                iy += patch_height
                i += 1
            batch_start = batch_end
        for s in streams: s.synchronize()
        if gpu_out is not None: gpu_out.requires_grad_(True)
        return gpu_out

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_out, grad_rm, grad_rv):
        X, weight = ctx.saved_tensors
        if ctx.module_id and len(BWD_ACCELERATOR_REGISTRY[ctx.module_id]) > 0:
            grad_out_gpu = BWD_ACCELERATOR_REGISTRY[ctx.module_id].pop()
            while len(BWD_ACCELERATOR_REGISTRY[ctx.module_id]) > 0:
                grad_out_gpu = grad_out_gpu + BWD_ACCELERATOR_REGISTRY[ctx.module_id].pop()
            use_gpu_grad = True
        else:
            grad_out_gpu = None
            use_gpu_grad = False

        X, weight = ctx.saved_tensors
        mean, variance = ctx.mean, ctx.variance
        N, C, H, W = X.shape
        device = weight.device

        grad_bias = torch.zeros(C, device=device)
        grad_weight = torch.zeros(C, device=device)

        grad_X_cpu = ctx.X_grad
        # if grad_X_cpu is not None:
        #     grad_X_cpu.zero_()

        # Allocate fresh buffer if needed
        if grad_X_cpu is None and ctx.return_cpu_grad:
            grad_X_cpu = torch.zeros(X.shape, device=cc.host_device,
                pin_memory=torch.cuda.is_available()).to(memory_format=ctx.memory_format)

        need_full_gpu_buffer = ctx.prev_module_id is not None
        grad_X_gpu = torch.empty(X.shape, device=device,
                                 memory_format=ctx.memory_format) if need_full_gpu_buffer else None

        max_batch = max(1, int(ctx.max_elements // X[0].numel()))
        max_batch = min(max_batch, N)
        streams = cc.get_cuda_streams()

        inv_std = torch.rsqrt(variance + 1e-5)

        # --- PASS 1: GLOBAL ACCUMULATION ---
        batch_start = 0
        while batch_start < N:
            batch_end = min(batch_start + max_batch, N)

            if X.is_cuda:
                sliced_X = X[batch_start: batch_end]
            else:
                sliced_X = X[batch_start: batch_end]

            if use_gpu_grad:
                sliced_grad_out = grad_out_gpu[batch_start: batch_end]
            else:
                sliced_grad_out = grad_out[batch_start: batch_end]

            iy, i = 0, 0
            while iy < H:
                patch_height = min(max(1, int(ctx.max_elements // (sliced_X.size(0) * C * W))), H - iy)
                with torch.cuda.stream(streams[i % len(streams)]):
                    if sliced_X.is_cuda:
                        im = sliced_X[:, :, iy:(iy + patch_height), :]
                    else:
                        im = sliced_X[:, :, iy:(iy + patch_height), :].to(device=device, non_blocking=True)

                    if use_gpu_grad:
                        dy = sliced_grad_out[:, :, iy:(iy + patch_height), :]
                    else:
                        dy = sliced_grad_out[:, :, iy:(iy + patch_height), :].to(device=device,
                                                                                 non_blocking=True)

                    if ctx.relu:
                        scale = weight * inv_std
                        # No bias saved, assume 0 shift relative to mean?
                        # Actually standard BN backward doesn't need re-masking if dy already masked.
                        # But here dy comes from next layer.
                        # We lack saved bias to reconstruct mask exactly if it was fused.
                        # Ideally BigFusedBNReLU handles relu logic.
                        pass

                    grad_bias += dy.sum(dim=(0, 2, 3))

                    im_centered = im - mean[None, :, None, None]
                    x_hat = im_centered * inv_std[None, :, None, None]
                    grad_weight += (dy * x_hat).sum(dim=(0, 2, 3))

                iy += patch_height
                i += 1
            batch_start = batch_end
        for s in streams:
            s.synchronize()

        # --- PASS 2: COMPUTE DX ---
        N_elements = N * H * W
        term1 = (weight * inv_std)[None, :, None, None]
        sum_dy = grad_bias[None, :, None, None]
        sum_dy_xhat = grad_weight[None, :, None, None]

        batch_start = 0
        while batch_start < N:
            batch_end = min(batch_start + max_batch, N)

            if X.is_cuda:
                sliced_X = X[batch_start: batch_end]
            else:
                sliced_X = X[batch_start: batch_end]

            if use_gpu_grad:
                sliced_grad_out = grad_out_gpu[batch_start: batch_end]
            else:
                sliced_grad_out = grad_out[batch_start: batch_end]

            if grad_X_cpu is not None: sliced_grad_X_cpu = grad_X_cpu[batch_start: batch_end]

            iy, i = 0, 0
            while iy < H:
                patch_height = min(max(1, int(ctx.max_elements // (sliced_X.size(0) * C * W))), H - iy)
                with torch.cuda.stream(streams[i % len(streams)]):
                    if sliced_X.is_cuda:
                        im = sliced_X[:, :, iy:(iy + patch_height), :]
                    else:
                        im = sliced_X[:, :, iy:(iy + patch_height), :].to(device=device, non_blocking=True)

                    if use_gpu_grad:
                        dy = sliced_grad_out[:, :, iy:(iy + patch_height), :]
                    else:
                        dy = sliced_grad_out[:, :, iy:(iy + patch_height), :].to(device=device,
                                                                                 non_blocking=True)

                    x_hat = (im - mean[None, :, None, None]) * inv_std[None, :, None, None]

                    res = (term1 / N_elements) * (N_elements * dy - sum_dy - x_hat * sum_dy_xhat)

                    if need_full_gpu_buffer:
                        grad_X_gpu[batch_start:batch_end, :, iy:(iy + patch_height), :].copy_(res)
                    elif grad_X_cpu is not None:
                        sliced_grad_X_cpu[:, :, iy:(iy + patch_height), :].copy_(res, non_blocking=True)
                    if DEBUG:
                        sliced_grad_X_cpu[:, :, iy:(iy + patch_height), :].copy_(res, non_blocking=True)

                iy += patch_height
                i += 1
            batch_start = batch_end

        for s in streams:
            s.synchronize()

        if ctx.prev_module_id is not None and grad_X_gpu is not None:
            BWD_ACCELERATOR_REGISTRY[ctx.prev_module_id].append(grad_X_gpu)
            if DEBUG:
                print(f"[{ctx.__class__.__name__}] Returning None to Prev Layer")
            target_device = ctx.x_device
            ghost_grad = torch.zeros(X.shape, device=target_device, dtype=X.dtype)
            return ghost_grad, grad_weight, grad_bias, *([None] * 14)

        grad_to_return = grad_X_cpu if ctx.return_cpu_grad else grad_X_gpu
        grad_to_return._origin = "BN_Backward_Output"
        if DEBUG:
            ret_mag = grad_to_return.abs().mean().item() if grad_to_return is not None else -1.0
            print(f"[{ctx.__class__.__name__}] Returning Grad to Prev Layer | Mag: {ret_mag:.6e}")
        if grad_to_return is None:
            target_device = ctx.x_device
            grad_to_return = torch.zeros(X.shape, device=target_device, dtype=X.dtype)
        return grad_to_return, grad_weight, grad_bias, *([None] * 14)


class BigBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, dtype=None, out_on_gpu=True, relu=False, max_elements=6e6):
        super(BigBatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features, device=cc.cuda_device))
        self.bias = nn.Parameter(torch.zeros(num_features, device=cc.cuda_device))
        self.register_buffer('running_mean', torch.zeros(num_features, device=cc.cuda_device))
        self.register_buffer('running_var', torch.ones(num_features, device=cc.cuda_device))

        self.output = None
        self.X_grad = None
        self._dtype = dtype if dtype else (torch.float16 if cc.amp_enabled else torch.float32)
        self.X_grad_on_gpu = False
        self.out_on_gpu = out_on_gpu
        self.relu = relu
        self.max_elements = max_elements
        self.gpu_cache = [] if out_on_gpu else None
        self.bwd_accelerator = []
        # Register this module's buckets
        GPU_CACHE_REGISTRY[id(self)] = self.gpu_cache
        BWD_ACCELERATOR_REGISTRY[id(self)] = self.bwd_accelerator

    def forward(self, X, mean=None, var=None, return_hot_potato=False):
        output = self._get_output_tensor(X)
        X_grad = output  # self._get_input_grad_tensor(X)
        if X_grad is not None:
            X_grad.zero_()

        if self.gpu_cache: self.gpu_cache.clear()
        if self.bwd_accelerator: self.bwd_accelerator.clear()

        prev_id = getattr(X, '_prev_module_id', None)

        output, rm, rv = BigBatchNormFunction.apply(
            X, self.weight, self.bias, self.running_mean, self.running_var, self.momentum,
            output, X_grad, self.training, self.relu, self.out_on_gpu, self.max_elements, torch.channels_last,
            id(self), prev_id, mean, var
        )

        if self.training:
            self.running_mean.copy_(rm.detach())
            self.running_var.copy_(rv.detach())

        if return_hot_potato:
            if self.out_on_gpu and self.gpu_cache:
                output._big_gpu_cache = self.gpu_cache.pop()
                output._prev_module_id = id(self)
        else:
            # Ensure no stale attributes exist
            if hasattr(output, '_bwd_accelerator'): del output._bwd_accelerator
            if hasattr(output, '_big_gpu_cache'): del output._big_gpu_cache
        return output

    def _get_output_tensor(self, X):
        B, C, H, W = X.shape
        if self.output is None:
            self.output = torch.empty(X.shape, pin_memory=torch.cuda.is_available(), memory_format=torch.channels_last,
                                      device=cc.host_device, dtype=self._dtype)
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
                                      dtype=self._dtype).to(
                memory_format=torch.channels_last)
            return self.X_grad
        cur_B, _, cur_H, cur_W = self.X_grad.shape
        if B > cur_B or H > cur_H or W > cur_W:
            max_B, max_H, max_W = max(B, cur_B), max(H, cur_H), max(W, cur_W)
            self.X_grad.resize_((max_B, self.num_features, max_H, max_W)).contiguous(
                memory_format=torch.channels_last)
            if not self.X_grad.is_pinned():
                self.X_grad.pin_memory()
        return self.X_grad[:B, :, :H, :W]


class BigFusedBNReLUMaxPoolFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float16)
    @torch.no_grad()
    def forward(ctx, X, weight, bias, running_mean, running_var, momentum,
                pool_k, pool_s, pool_p,
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

        # --- STATS ---
        if training:
            if external_mean is not None and external_var is not None:
                mean = external_mean
                variance = external_var
                running_mean = momentum * mean + (1 - momentum) * running_mean
                running_var = momentum * variance + (1 - momentum) * running_var
            else:
                mean, m2 = BigFusedBNReLUMaxPoolFunction.compute_welford_stats(X_input, max_elements, weight.device)
                n_total = X.numel() / X.size(1)
                variance = m2 / n_total

                unbiased_var = m2 / (n_total - 1)
                running_mean = momentum * mean + (1 - momentum) * running_mean
                running_var = momentum * unbiased_var + (1 - momentum) * running_var
        else:
            mean = running_mean
            variance = running_var

        # --- SAVE CONTEXT ---
        # We need BIAS for exact ReLU mask reconstruction in backward
        ctx.x_device = X.device
        ctx.save_for_backward(X if not X.is_cuda else X.cpu(), weight, bias)
        ctx.mean = mean
        ctx.variance = variance
        ctx.X_grad = X_grad
        ctx.pool_params = (pool_k, pool_s, pool_p)
        ctx.max_elements = max_elements
        ctx.memory_format = memory_format

        ctx.prev_module_id = prev_module_id
        ctx.module_id = module_id

        # --- COMPUTE ---
        B, C, H_in, W_in = X.shape
        H_out, W_out = output.shape[2], output.shape[3]

        gpu_out = None
        if out_on_gpu:
            gpu_out = torch.empty(output.shape, device=device)

        inv_std = torch.rsqrt(variance + 1e-5)
        scale = weight * inv_std
        shift = bias - mean * scale

        max_batch = max(1, min(B, int(max_elements // (C * H_out * W_out))))
        max_rows = max(1, int(max_elements // (max_batch * C * W_out)))
        max_rows = min(max_rows, H_out)
        streams = cc.get_cuda_streams()

        batch_start = 0
        while batch_start < B:
            batch_end = min(batch_start + max_batch, B)
            if X_input.is_cuda:
                x_batch = X_input[batch_start:batch_end]
            else:
                x_batch = X_input[batch_start:batch_end]
            out_batch_cpu = output[batch_start:batch_end]

            out_y = 0
            while out_y < H_out:
                with torch.cuda.stream(streams[0]):
                    out_y_end = min(out_y + max_rows, H_out)
                    in_y_start_raw = out_y * pool_s - pool_p
                    in_y_end_raw = (out_y_end - 1) * pool_s + pool_k - pool_p
                    in_y_start = max(0, in_y_start_raw)
                    in_y_end = min(H_in, in_y_end_raw)
                    pad_top = in_y_start - in_y_start_raw
                    pad_bottom = in_y_end_raw - in_y_end

                    if x_batch.is_cuda:
                        x_tile = x_batch[:, :, in_y_start:in_y_end, :]
                    else:
                        x_tile = x_batch[:, :, in_y_start:in_y_end, :].to(device, non_blocking=True)

                    x_norm = x_tile * scale[None, :, None, None] + shift[None, :, None, None]
                    x_relu = F.relu(x_norm)
                    x_padded = F.pad(x_relu, (pool_p, pool_p, pad_top, pad_bottom), value=float('-inf'))
                    pooled = F.max_pool2d(x_padded, kernel_size=pool_k, stride=pool_s, padding=0)

                    h_needed = out_y_end - out_y
                    if pooled.shape[2] != h_needed: pooled = pooled[:, :, :h_needed, :]

                    if out_on_gpu:
                        gpu_out[batch_start:batch_end, :, out_y:out_y_end, :].copy_(pooled)
                    out_batch_cpu[:, :, out_y:out_y_end, :].copy_(pooled, non_blocking=True)

                out_y = out_y_end
            batch_start = batch_end

        if out_on_gpu and not hot_potato:
            gpu_out.requires_grad_(True)
            return gpu_out, running_mean, running_var

        if out_on_gpu and module_id is not None:
            GPU_CACHE_REGISTRY[module_id].append(gpu_out)

        return output, running_mean, running_var

    @staticmethod
    @torch.no_grad()
    def compute_welford_stats(X, max_elements, device):
        # Same Welford implementation as before...
        max_batch = max(1, min(X.shape[0], int(max_elements // X[0].numel())))
        global_mean = torch.zeros(X.shape[1], device=device)
        global_m2 = torch.zeros(X.shape[1], device=device)
        streams = cc.get_cuda_streams()
        global_n = 0.0
        batch_start = 0
        while batch_start < X.shape[0]:
            batch_end = min(batch_start + max_batch, X.shape[0])
            if X.is_cuda:
                sl = X[batch_start:batch_end]
            else:
                sl = X[batch_start:batch_end]
            iy, i = 0, 0
            while iy < X.shape[2]:
                ph = min(max(1, int(max_elements // (sl.size(0) * X.size(1) * X.size(3)))), X.shape[2] - iy)
                with torch.cuda.stream(streams[i % len(streams)]):
                    if sl.is_cuda:
                        im = sl[:, :, iy:iy + ph, :]
                    else:
                        im = sl[:, :, iy:iy + ph, :].to(device, non_blocking=True)
                    im_flat = im.view(im.size(0), im.size(1), -1)
                    n_chunk = im_flat.shape[0] * im_flat.shape[2]
                    c_mean = im_flat.mean((0, 2))
                    c_m2 = ((im_flat - c_mean.view(1, -1, 1)) ** 2).sum((0, 2))
                    if global_n == 0:
                        global_mean, global_m2, global_n = c_mean, c_m2, n_chunk
                    else:
                        delta = c_mean - global_mean
                        new_n = global_n + n_chunk
                        global_m2 += c_m2 + (delta ** 2) * (global_n * n_chunk / new_n)
                        global_mean += delta * (n_chunk / new_n)
                        global_n = new_n
                iy += ph
                i += 1
            batch_start = batch_end
        for s in streams: s.synchronize()
        return global_mean, global_m2

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_out, grad_rm, grad_rv):
        X, weight, bias = ctx.saved_tensors
        if grad_out.is_cuda:
            grad_out_gpu = grad_out
            use_gpu_grad = True
        elif ctx.module_id and len(BWD_ACCELERATOR_REGISTRY[ctx.module_id]) > 0:
            grad_out_gpu = BWD_ACCELERATOR_REGISTRY[ctx.module_id].pop()
            while len(BWD_ACCELERATOR_REGISTRY[ctx.module_id]) > 0:
                grad_out_gpu = grad_out_gpu + BWD_ACCELERATOR_REGISTRY[ctx.module_id].pop()
            use_gpu_grad = True
        else:
            grad_out_gpu = None
            use_gpu_grad = False

        X, weight, bias = ctx.saved_tensors
        mean, variance = ctx.mean, ctx.variance
        k, s, p = ctx.pool_params
        N, C, H_in, W_in = X.shape
        H_out, W_out = grad_out.shape[2], grad_out.shape[3]
        device = weight.device

        grad_bias = torch.zeros(C, device=device)
        grad_weight = torch.zeros(C, device=device)

        grad_X_cpu = ctx.X_grad
        # if grad_X_cpu is not None:
        #     grad_X_cpu.zero_()

        need_gpu_buffer = ctx.prev_module_id is not None
        grad_X_gpu = torch.zeros(X.shape, device=device) if need_gpu_buffer else None

        max_batch = max(1, min(N, int(ctx.max_elements // (C * H_out * W_out))))
        max_rows = max(1, int(ctx.max_elements // (max_batch * C * W_out)))
        max_rows = min(max_rows, H_out)
        streams = cc.get_cuda_streams()

        inv_std = torch.rsqrt(variance + 1e-5)
        # Constants for tile calculation
        scale = weight * inv_std
        shift = bias - mean * scale

        # --- PASS 1: GLOBAL ACCUMULATION ---
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

            out_y, i = 0, 0
            while out_y < H_out:
                with torch.cuda.stream(streams[0]):
                    out_y_end = min(out_y + max_rows, H_out)
                    in_y_start_raw = out_y * s - p
                    in_y_end_raw = (out_y_end - 1) * s + k - p
                    in_y_start = max(0, in_y_start_raw)
                    in_y_end = min(H_in, in_y_end_raw)
                    pad_top = in_y_start - in_y_start_raw
                    pad_bottom = in_y_end_raw - in_y_end

                    if x_batch.is_cuda:
                        x_tile = x_batch[:, :, in_y_start:in_y_end, :]
                    else:
                        x_tile = x_batch[:, :, in_y_start:in_y_end, :].to(device, non_blocking=True)

                    if use_gpu_grad:
                        g_tile = g_batch[:, :, out_y:out_y_end, :]
                    else:
                        g_tile = g_batch[:, :, out_y:out_y_end, :].to(device, non_blocking=True)

                    # --- PROCEDURAL GRADIENT ---
                    # 1. Manually Re-Normalize (No Autograd overhead)
                    x_norm = x_tile * scale[None, :, None, None] + shift[None, :, None, None]

                    # 2. Local Autograd for ReLU + Pool
                    with torch.enable_grad():
                        x_norm.detach_()
                        x_norm.requires_grad_(True)

                        x_relu = F.relu(x_norm)
                        x_padded = F.pad(x_relu, (p, p, pad_top, pad_bottom), value=float('-inf'))
                        pooled = F.max_pool2d(x_padded, k, s, padding=0)

                        h_needed = out_y_end - out_y
                        if pooled.shape[2] != h_needed:
                            pooled = pooled[:, :, :h_needed, :]

                        pooled.backward(g_tile)
                        g_norm = x_norm.grad

                    # 3. Accumulate BN Grads
                    grad_bias += g_norm.sum(dim=(0, 2, 3))
                    im_centered = x_tile - mean[None, :, None, None]
                    x_hat = im_centered * inv_std[None, :, None, None]
                    grad_weight += (g_norm * x_hat).sum(dim=(0, 2, 3))

                out_y = out_y_end
            batch_start = batch_end

        for stream in streams:
            stream.synchronize()

        # --- PASS 2: COMPUTE DX ---
        N_elements = N * H_in * W_in
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

            if grad_X_cpu is not None: g_x_cpu_slice = grad_X_cpu[batch_start:batch_end]

            out_y, i = 0, 0
            while out_y < H_out:
                with torch.cuda.stream(streams[0]):
                    out_y_end = min(out_y + max_rows, H_out)
                    in_y_start_raw = out_y * s - p
                    in_y_end_raw = (out_y_end - 1) * s + k - p
                    in_y_start = max(0, in_y_start_raw)
                    in_y_end = min(H_in, in_y_end_raw)
                    pad_top = in_y_start - in_y_start_raw
                    pad_bottom = in_y_end_raw - in_y_end

                    if x_batch.is_cuda:
                        x_tile = x_batch[:, :, in_y_start:in_y_end, :]
                    else:
                        x_tile = x_batch[:, :, in_y_start:in_y_end, :].to(device, non_blocking=True)

                    if use_gpu_grad:
                        g_tile = g_batch[:, :, out_y:out_y_end, :]
                    else:
                        g_tile = g_batch[:, :, out_y:out_y_end, :].to(device, non_blocking=True)

                    # --- PROCEDURAL GRADIENT (RE-RUN) ---
                    # 1. Manual Re-Normalize
                    x_norm = x_tile * scale[None, :, None, None] + shift[None, :, None, None]

                    # 2. Local Autograd
                    with torch.enable_grad():
                        x_norm.detach_()
                        x_norm.requires_grad_(True)

                        x_relu = F.relu(x_norm)
                        x_padded = F.pad(x_relu, (p, p, pad_top, pad_bottom), value=float('-inf'))
                        pooled = F.max_pool2d(x_padded, k, s, padding=0)

                        h_needed = out_y_end - out_y
                        if pooled.shape[2] != h_needed: pooled = pooled[:, :, :h_needed, :]

                        pooled.backward(g_tile)
                        g_norm = x_norm.grad

                    # 3. BN Backward Formula
                    # dx = term1 * (g_norm - (sum_dy + x_hat * sum_dy_xhat) / N_elements)
                    x_hat = (x_tile - mean[None, :, None, None]) * inv_std[None, :, None, None]
                    d_input = term1 * (g_norm - (sum_dy + x_hat * sum_dy_xhat) / N_elements)

                    # 4. Accumulate
                    if grad_X_gpu is not None:
                        grad_X_gpu[batch_start:batch_end, :, in_y_start:in_y_end, :].add_(d_input)

                    if ctx.return_cpu_grad and grad_X_cpu is not None:
                        _tmp = d_input.to('cpu')
                        g_x_cpu_slice[:, :, in_y_start:in_y_end, :].add_(_tmp)

                out_y = out_y_end
            batch_start = batch_end
        for s in streams: s.synchronize()

        if ctx.prev_module_id is not None and grad_X_gpu is not None:
            BWD_ACCELERATOR_REGISTRY[ctx.prev_module_id].append(grad_X_gpu)
            if DEBUG:
                print(f"[{ctx.__class__.__name__}] Returning None to Prev Layer")
            target_device = ctx.x_device
            ghost_grad = torch.zeros(X.shape, device=target_device, dtype=X.dtype)
            return ghost_grad, grad_weight, grad_bias, *([None] * 17)

        grad_to_return = grad_X_cpu if ctx.return_cpu_grad else grad_X_gpu
        if DEBUG:
            ret_mag = grad_to_return.abs().mean().item() if grad_to_return is not None else -1.0
            print(f"[{ctx.__class__.__name__}] Returning Grad to Prev Layer | Mag: {ret_mag:.6e}")
        if grad_to_return is None:
            target_device = ctx.x_device
            grad_to_return = torch.zeros(X.shape, device=target_device, dtype=X.dtype).expand(X.shape)
        return grad_to_return, grad_weight, grad_bias, *([None] * 17)


class BigFusedBNReLUMaxPool(nn.Module):
    def __init__(self, num_features, momentum=0.1, dtype=None, out_on_gpu=True, max_elements=6e6):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features, device=cc.cuda_device))
        self.bias = nn.Parameter(torch.zeros(num_features, device=cc.cuda_device))
        self.register_buffer('running_mean', torch.zeros(num_features, device=cc.cuda_device))
        self.register_buffer('running_var', torch.ones(num_features, device=cc.cuda_device))

        self.output = None
        self.X_grad = None
        self._dtype = dtype if dtype else (torch.float16 if cc.amp_enabled else torch.float32)
        self.out_on_gpu = out_on_gpu
        self.gpu_cache = [] if out_on_gpu else None
        self.bwd_accelerator = []
        self.max_elements = max_elements
        # Register this module's buckets
        GPU_CACHE_REGISTRY[id(self)] = self.gpu_cache
        BWD_ACCELERATOR_REGISTRY[id(self)] = self.bwd_accelerator

    def forward(self, X, mean=None, var=None, hot_potato=True):
        output = self._get_output_tensor(X)
        X_grad = None
        if X.requires_grad and not self.out_on_gpu:
            X_grad = self._get_input_grad_tensor(X)
        if X_grad is not None:
            X_grad.zero_()

        if self.gpu_cache: self.gpu_cache.clear()
        if self.bwd_accelerator: self.bwd_accelerator.clear()

        prev_id = getattr(X, '_prev_module_id', None)

        out, rm, rv = BigFusedBNReLUMaxPoolFunction.apply(
            X, self.weight, self.bias, self.running_mean, self.running_var, self.momentum,
            3, 2, 1,  # k, s, p
            output, X_grad, self.training, self.out_on_gpu, hot_potato, self.max_elements, torch.channels_last,
            id(self), prev_id, mean, var
        )
        if self.out_on_gpu and not hot_potato:
            return out

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

    def _get_output_tensor(self, X):
        B, C, H, W = X.shape
        H_out = ((H + 2 * 1 - 3) // 2) + 1
        W_out = ((W + 2 * 1 - 3) // 2) + 1
        if self.output is None:
            self.output = torch.empty((B, C, H_out, W_out),
                pin_memory=torch.cuda.is_available(), memory_format=torch.channels_last,
                                      device=cc.host_device, dtype=self._dtype)
            return self.output
        cur_B, _, cur_H, cur_W = self.output.shape
        if B > cur_B or H_out > cur_H or W_out > cur_W:
            max_B, max_H, max_W = max(B, cur_B), max(H_out, cur_H), max(W_out, cur_W)
            # self.output.requires_grad_(False)
            self.output.resize_((max_B, self.num_features, max_H, max_W)).contiguous(
                memory_format=torch.channels_last)
            if not self.output.is_pinned():
                self.output.pin_memory()
        sliced = self.output[:B, :, :H_out, :W_out]
        # sliced.requires_grad_(self.training)
        return sliced

    def _get_input_grad_tensor(self, X):
        B, C, H, W = X.shape
        if self.X_grad is None:
            self.X_grad = torch.zeros(X.shape,
                pin_memory=torch.cuda.is_available(), device=cc.host_device, dtype=self._dtype).to(
                memory_format=torch.channels_last)
            return self.X_grad
        cur_B, _, cur_H, cur_W = self.X_grad.shape
        if B > cur_B or H > cur_H or W > cur_W:
            max_B, max_H, max_W = max(B, cur_B), max(H, cur_H), max(W, cur_W)
            self.X_grad.resize_((max_B, self.num_features, max_H, max_W)).contiguous(
                memory_format=torch.channels_last)
            if not self.X_grad.is_pinned():
                self.X_grad.pin_memory()
        return self.X_grad[:B, :, :H, :W]


class BigDS(nn.Module):
    """Helper for the Downsample/Shortcut branch in ResNet stages."""
    def __init__(self, c_in, c_out, stride=1, out_on_gpu=True, max_elements=6e6):
        super().__init__()
        self.out_on_gpu = out_on_gpu
        self.conv = BigConv2dStats(c_in, c_out, kernel_size=1, stride=stride, padding=0, out_on_gpu=out_on_gpu,
            max_elements=max_elements)
        self.bn = BigBatchNorm(c_out, out_on_gpu=out_on_gpu, max_elements=max_elements)

    def forward(self, x, hot_potato=True):
        # Propagation of hot_potato flag
        out, m, v = self.conv(x, return_hot_potato=hot_potato)
        if hot_potato:
            out._prev_module_id = id(self.conv)
        else:
            # Explicitly strip any inherited accelerators
            if hasattr(out, '_bwd_accelerator'): del out._bwd_accelerator
        return self.bn(out, mean=m, var=v, return_hot_potato=hot_potato)
