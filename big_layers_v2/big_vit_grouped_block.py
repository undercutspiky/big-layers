"""EXPERIMENTAL: archived hot-potato path, not used by the paper training entry points."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from big_layers.amp_compat import custom_bwd, custom_fwd
from . import cuda_config as cc


# ==============================================================================
# 1. Helper: Single Block Logic
# ==============================================================================

def vit_block_forward(x, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b,
                      ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b,
                      num_heads, mlp_ratio):
    B, L, D = x.shape

    # 1. Attn
    norm1 = F.layer_norm(x, (D,), ln1_w, ln1_b, eps=1e-6)
    qkv = F.linear(norm1, qkv_w, qkv_b)
    qkv = qkv.reshape(B, L, 3, num_heads, D // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
    attn_out = F.linear(attn_out, proj_w, proj_b)
    x = x + attn_out

    # 2. MLP
    norm2 = F.layer_norm(x, (D,), ln2_w, ln2_b, eps=1e-6)
    mid = F.linear(norm2, fc1_w, fc1_b)
    mid = F.gelu(mid)
    out = F.linear(mid, fc2_w, fc2_b)
    x = x + out

    return x


# OPTIMIZATION 1: Compile the inner loop logic
# This fuses the LN+Linear+GELU ops for the chunk
try:
    compiled_block_fwd = torch.compile(vit_block_forward, mode="reduce-overhead")
except:
    compiled_block_fwd = vit_block_forward


# ==============================================================================
# 2. Autograd Function: Handles K blocks at once with ASYNC PREFETCH
# ==============================================================================

class BigGroupedViTBlockFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    @torch.no_grad()
    def forward(ctx, x, num_heads, mlp_ratio, output, x_grad, gpu_cache, bwd_accelerator,
                max_elements, out_on_gpu, hot_potato, *all_weights):

        # --- INPUT HANDLING ---
        X_gpu_cache = getattr(x, '_big_gpu_cache', None)
        if X_gpu_cache is not None:
            del x._big_gpu_cache
            x_in = X_gpu_cache
        else:
            x_in = x

        # Save context
        ctx.save_for_backward(x if not x.is_cuda else x.cpu(), *all_weights)
        ctx.config = (num_heads, mlp_ratio)
        ctx.max_elements = max_elements
        ctx.x_grad_buffer = x_grad
        ctx.prev_bwd_accelerator = getattr(x, '_bwd_accelerator', None)
        ctx.my_bwd_accelerator = bwd_accelerator
        ctx.return_cpu_grad = not x.is_cuda

        B, L, D = x.shape
        num_weights_per_block = 12
        num_blocks = len(all_weights) // num_weights_per_block
        ctx.num_blocks = num_blocks

        # Output Setup
        gpu_out = None
        if out_on_gpu:
            gpu_out = torch.empty((B, L, D), device=cc.cuda_device, requires_grad=False)

        # Tiling (Batch dimension)
        max_batch = max(1, int(max_elements // (L * D)))

        # --- ASYNC PIPELINE SETUP ---
        compute_stream = torch.cuda.current_stream()
        transfer_stream = torch.cuda.Stream()

        # Prefetch Buffers
        next_x_gpu = None

        # 1. Prefetch First Tile
        b_start = 0
        b_end = min(max_batch, B)

        with torch.cuda.stream(transfer_stream):
            # Non-blocking copy
            if x_in.is_cuda:
                next_x_gpu = x_in[b_start:b_end]
            else:
                next_x_gpu = x_in[b_start:b_end].to(cc.cuda_device, non_blocking=True)

        # Loop
        while b_start < B:
            # A. Wait for Transfer
            compute_stream.wait_stream(transfer_stream)
            curr_x = next_x_gpu
            curr_start, curr_end = b_start, b_end

            # B. Launch Next Transfer
            next_start = b_end
            if next_start < B:
                next_end = min(next_start + max_batch, B)
                with torch.cuda.stream(transfer_stream):
                    if x_in.is_cuda:
                        next_x_gpu = x_in[next_start:next_end]
                    else:
                        next_x_gpu = x_in[next_start:next_end].to(cc.cuda_device, non_blocking=True)

            # C. Compute (Fused Group)
            # Use compiled kernel if possible
            for k in range(num_blocks):
                off = k * num_weights_per_block
                w_k = all_weights[off: off + num_weights_per_block]
                curr_x = compiled_block_fwd(curr_x, *w_k, num_heads, mlp_ratio)

            # D. Store Output
            # Note: CPU store must be synchronized or use pinned memory carefully.
            # Here we issue the copy on the compute stream.
            if out_on_gpu:
                gpu_out[curr_start:curr_end].copy_(curr_x)

            # Async copy to CPU output
            output[curr_start:curr_end].copy_(curr_x, non_blocking=True)

            # Advance
            b_start = next_start
            b_end = next_end if b_start < B else B

        torch.cuda.synchronize()

        if out_on_gpu:
            if not hot_potato: return gpu_out
            if gpu_cache is not None: gpu_cache.append(gpu_out)

        return output

    @staticmethod
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_out):
        if ctx.my_bwd_accelerator and len(ctx.my_bwd_accelerator) > 0:
            grad_out_gpu = ctx.my_bwd_accelerator.pop()
            use_gpu_grad = True
        else:
            grad_out_gpu = None
            use_gpu_grad = False

        # Unpack
        saved = ctx.saved_tensors
        x_cpu = saved[0]
        all_weights = saved[1:]
        num_heads, mlp_ratio = ctx.config
        num_blocks = ctx.num_blocks
        num_weights_per_block = 12

        # Init Grads
        grad_weights = [torch.zeros_like(w) for w in all_weights]

        grad_x_cpu_buf = ctx.x_grad_buffer
        if grad_x_cpu_buf is not None: grad_x_cpu_buf.zero_()

        need_gpu_buf = (ctx.prev_bwd_accelerator is not None)
        grad_x_gpu = torch.zeros(x_cpu.shape, device=cc.cuda_device) if need_gpu_buf else None

        B, L, D = x_cpu.shape
        max_batch = max(1, int(ctx.max_elements // (L * D)))

        # --- ASYNC PIPELINE SETUP ---
        compute_stream = torch.cuda.current_stream()
        transfer_stream = torch.cuda.Stream()

        # Prefetch Buffers
        next_x_tile = None
        next_g_tile = None

        # 1. Prefetch First Tile
        b_start = 0
        b_end = min(max_batch, B)

        with torch.cuda.stream(transfer_stream):
            # Load Input
            if x_cpu.is_cuda:
                next_x_tile = x_cpu[b_start:b_end]
            else:
                next_x_tile = x_cpu[b_start:b_end].to(cc.cuda_device, non_blocking=True)

            # Load Grad
            if use_gpu_grad:
                next_g_tile = grad_out_gpu[b_start:b_end]
            else:
                next_g_tile = grad_out[b_start:b_end].to(cc.cuda_device, non_blocking=True)

        # Loop
        while b_start < B:
            # A. Wait for Transfer
            compute_stream.wait_stream(transfer_stream)
            x_tile = next_x_tile
            g_tile = next_g_tile
            curr_start, curr_end = b_start, b_end

            # B. Launch Next Transfer
            next_start = b_end
            if next_start < B:
                next_end = min(next_start + max_batch, B)
                with torch.cuda.stream(transfer_stream):
                    if x_cpu.is_cuda:
                        next_x_tile = x_cpu[next_start:next_end]
                    else:
                        next_x_tile = x_cpu[next_start:next_end].to(cc.cuda_device, non_blocking=True)

                    if use_gpu_grad:
                        next_g_tile = grad_out_gpu[next_start:next_end]
                    else:
                        next_g_tile = grad_out[next_start:next_end].to(cc.cuda_device, non_blocking=True)

            # C. Compute (Recompute + Backward)
            with torch.enable_grad():
                x_tile.detach_()
                x_tile.requires_grad_(True)

                # Weights for graph
                temp_weights = []
                for w in all_weights:
                    temp_weights.append(w.detach().requires_grad_(True))

                # Forward Chain
                curr_x = x_tile
                for k in range(num_blocks):
                    off = k * num_weights_per_block
                    w_k = temp_weights[off: off + num_weights_per_block]
                    curr_x = compiled_block_fwd(curr_x, *w_k, num_heads, mlp_ratio)

                # Backward
                curr_x.backward(g_tile)

                # 1. Accumulate Input Grads
                # We do this on compute stream.
                # If target is CPU, we should queue copy.
                if grad_x_gpu is not None:
                    grad_x_gpu[curr_start:curr_end].add_(x_tile.grad)
                if ctx.return_cpu_grad and grad_x_cpu_buf is not None:
                    # Blocking copy or async?
                    # Since we reuse x_tile.grad buffer in next iter (autograd internals),
                    # strictly safer to block or clone. But .to() creates copy.
                    grad_x_cpu_buf[curr_start:curr_end].add_(x_tile.grad.to('cpu', non_blocking=True))

                # 2. Accumulate Weight Grads
                for i, tw in enumerate(temp_weights):
                    if tw.grad is not None:
                        grad_weights[i] += tw.grad

            # D. Advance
            b_start = next_start
            b_end = next_end if b_start < B else B

        torch.cuda.synchronize()

        if ctx.prev_bwd_accelerator is not None and grad_x_gpu is not None:
            ctx.prev_bwd_accelerator.append(grad_x_gpu)

        ret_grad_x = grad_x_cpu_buf if ctx.return_cpu_grad else grad_x_gpu

        return ret_grad_x, *([None]*9), *grad_weights


# ==============================================================================
# 3. Module: BigGroupedViTBlock
# ==============================================================================

class BigGroupedViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., num_blocks=1, out_on_gpu=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_blocks = num_blocks
        self.out_on_gpu = out_on_gpu
        self._dtype = torch.float32

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            blk = nn.Module()
            blk.ln1 = nn.LayerNorm(dim)
            blk.attn_qkv = nn.Linear(dim, dim * 3)
            blk.attn_proj = nn.Linear(dim, dim)
            blk.ln2 = nn.LayerNorm(dim)
            blk.mlp_fc1 = nn.Linear(dim, int(dim * mlp_ratio))
            blk.mlp_fc2 = nn.Linear(int(dim * mlp_ratio), dim)
            self.blocks.append(blk)

        self.output = None
        self.X_grad = None
        self.gpu_cache = [] if out_on_gpu else None
        self.bwd_accelerator = []
        self.max_elements = 6e6

    def _get_output_tensor(self, B, L, D):
        if self.output is None:
            self.output = torch.empty((max(B, 1), max(L, 1024), D), device='cpu',
                                      pin_memory=torch.cuda.is_available(), dtype=self._dtype)
        cur_B, cur_L, _ = self.output.shape
        if cur_B < B or cur_L < L:
            self.output = torch.empty((max(B, cur_B), max(L, cur_L), D), device='cpu',
                                      pin_memory=torch.cuda.is_available(), dtype=self._dtype)
        return self.output[:B, :L]

    def _get_input_grad_tensor(self, X):
        B, L, D = X.shape
        if self.X_grad is None:
            self.X_grad = torch.zeros(X.shape, device='cpu', pin_memory=torch.cuda.is_available(), dtype=self._dtype)
        cur_B, cur_L, _ = self.X_grad.shape
        if cur_B < B or cur_L < L:
            self.X_grad = torch.zeros((max(B, cur_B), max(L, cur_L), D), device='cpu',
                                      pin_memory=torch.cuda.is_available(), dtype=self._dtype)
        return self.X_grad[:B, :L]

    def forward(self, x, hot_potato=True):
        B, L, D = x.shape
        out = self._get_output_tensor(B, L, D)

        x_grad = None
        if x.requires_grad:
            x_grad = self._get_input_grad_tensor(x)

        if self.gpu_cache: self.gpu_cache.clear()

        all_weights = []
        for blk in self.blocks:
            all_weights.extend([
                blk.ln1.weight, blk.ln1.bias,
                blk.attn_qkv.weight, blk.attn_qkv.bias, blk.attn_proj.weight, blk.attn_proj.bias,
                blk.ln2.weight, blk.ln2.bias,
                blk.mlp_fc1.weight, blk.mlp_fc1.bias, blk.mlp_fc2.weight, blk.mlp_fc2.bias
            ])

        out = BigGroupedViTBlockFunction.apply(
            x, self.num_heads, self.mlp_ratio,
            out, x_grad,
            self.gpu_cache, self.bwd_accelerator,
            self.max_elements, self.out_on_gpu, hot_potato,
            *all_weights
        )

        if self.out_on_gpu and hot_potato and self.gpu_cache:
            out._big_gpu_cache = self.gpu_cache.pop()

        out._bwd_accelerator = self.bwd_accelerator
        return out
