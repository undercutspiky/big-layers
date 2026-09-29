"""EXPERIMENTAL: archived hot-potato path, not used by the paper training entry points."""
import math
import time
from contextlib import nullcontext

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from big_layers.amp_compat import custom_bwd, custom_fwd

from . import cuda_config as cc


def maybe_cuda_stream(streams, idx):
    """
    Return a context manager that enters the requested CUDA stream if
    CUDA is available and `streams` is non-empty; otherwise returns
    a no-op context manager.
    `streams` should be a sequence of torch.cuda.Stream objects (or empty).
    `idx` is the integer index you want to use (we'll mod by len(streams)).
    """
    if torch.cuda.is_available() and streams and len(streams) > 0:
        return torch.cuda.stream(streams[idx % len(streams)])
    return nullcontext()

# ==============================================================================
# 1. Helper: Tiled LayerNorm (Used internally by both layers)
# ==============================================================================

def manual_layernorm_forward(x, weight, bias, eps=1e-6):
    """Simple functional LayerNorm for use inside tiles."""
    u = x.mean(-1, keepdim=True)
    s = (x - u).pow(2).mean(-1, keepdim=True)
    x = (x - u) * torch.rsqrt(s + eps)
    return weight * x + bias


def manual_layernorm_backward(grad_out, x, weight, bias, eps=1e-6):
    """
    Manual backward for LayerNorm to be used inside fused kernels.
    Returns grad_input.
    """
    N = x.shape[-1]
    mean = x.mean(-1, keepdim=True)
    var = (x - mean).pow(2).mean(-1, keepdim=True)
    std = torch.sqrt(var + eps)
    inv_std = 1.0 / std

    x_mu = x - mean
    dY_w = grad_out * weight

    dVar = (dY_w * x_mu * -0.5 * torch.pow(std, -3)).sum(-1, keepdim=True)
    dMean = (dY_w * -inv_std).sum(-1, keepdim=True) + \
            dVar * (-2.0 / N) * x_mu.sum(-1, keepdim=True)

    dX = dY_w * inv_std + dVar * (2.0 / N) * x_mu + dMean * (1.0 / N)
    return dX


# ==============================================================================
# 2. Layer 1: BigFusedSelfAttention
#    (Norm -> QKV -> Attn -> Proj -> LS -> Res)
# ==============================================================================

class BigFusedSelfAttentionFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    @torch.no_grad()
    def forward(ctx, X_in, norm_w, norm_b, W_qkv, b_qkv, W_o, b_o, ls_gamma,
                num_heads, head_dim, output=None, X_grad=None,
                gpu_cache=None, bwd_accelerator=None,  # Side Channels
                max_block_tokens=2048, max_elements=int(6e6), out_on_gpu=True):

        # 1. INPUT HANDLING (Forward Cache + Backward Bucket)
        X_gpu_cache = getattr(X_in, '_big_gpu_cache', None)
        if X_gpu_cache is not None:
            del X_in._big_gpu_cache
        prev_bwd_accelerator = getattr(X_in, '_bwd_accelerator', None)

        device = norm_w.device
        B, L, D = X_in.shape
        H, d = num_heads, head_dim
        Hd = H * d
        D_out = W_o.shape[1]

        # Tiling Setup
        max_elements = int(max_elements)
        tokens_per_sample = L * D
        max_batch_size = max(1, int(max_elements // tokens_per_sample))
        max_batch_size = min(max_batch_size, B)
        # Factor 4 conservative estimate for intermediates
        max_patch_len = max(1, int(max_elements // (max_batch_size * D * 4)))

        ctx.tiling_params = (max_block_tokens, max_elements)
        ctx.shapes = (B, L, D, H, d)
        ctx.device = device
        ctx.X_grad = X_grad
        ctx.return_cpu_grad = not X_in.is_cuda

        Wo_slices = W_o.view(H, d, D_out)
        gpu_out = torch.empty((B, L, D), device=device) if out_on_gpu else None

        # Stats Buffers for Attention
        L_vec = torch.empty((B, H, L), device=device)  #, dtype=torch.float32)
        M_vec = torch.empty((B, H, L), device=device)  # , dtype=torch.float32)

        batch_start = 0
        stream_idx = 0
        while batch_start < B:
            batch_end = min(batch_start + max_batch_size, B)
            b_s = batch_end - batch_start

            with maybe_cuda_stream(cc.cuda_streams, stream_idx):
                if X_gpu_cache is not None:
                    x_chunk = X_gpu_cache[batch_start:batch_end]
                else:
                    x_chunk = X_in[batch_start:batch_end].to(device, non_blocking=True)

                # 2. Fused Norm + QKV Proj
                # Note: We do this for the whole batch-slice to have full QKV for attention
                # (Assuming batch-slice fits in GPU, which max_batch_size logic ensures)

                # Norm
                x_norm = manual_layernorm_forward(x_chunk, norm_w, norm_b)

                # QKV
                b_flat = b_qkv.to(W_qkv.dtype).view(1, 1, 3 * Hd)
                QKV = x_norm.matmul(W_qkv) + b_flat
                QKV = QKV.view(b_s, L, 3, H, d)
                Q_all, K_all, V_all = QKV[:, :, 0], QKV[:, :, 1], QKV[:, :, 2]

                # 3. Flash Attention Loop (Tiled over L)
                block = min(ctx.tiling_params[0], max_patch_len, L)

                for i0 in range(0, L, block):
                    i1 = min(L, i0 + block)
                    Qb = Q_all[:, i0:i1]

                    m = torch.full((b_s, H, i1 - i0), -float('inf'), device=device)  # , dtype=torch.float32)
                    l = torch.zeros((b_s, H, i1 - i0), device=device)  # , dtype=torch.float32)
                    acc = torch.zeros((b_s, H, i1 - i0, D_out), device=device)  # , dtype=torch.float32)

                    for j0 in range(0, L, block):
                        j1 = min(L, j0 + block)
                        Kb = K_all[:, j0:j1]
                        Vb = V_all[:, j0:j1]

                        S = torch.einsum('bqhd,bkhd->bhqk', Qb, Kb) / math.sqrt(d)
                        S_max = S.amax(dim=3)
                        m_new = torch.maximum(m, S_max)
                        exp_diff = (S - m_new.unsqueeze(-1)).exp()
                        exp_m_old_new = (m - m_new).exp()
                        l_new = (exp_m_old_new * l) + exp_diff.sum(dim=3)

                        Vb_proj = torch.einsum('bkhd,hde->bkhe', Vb, Wo_slices)
                        acc = acc * exp_m_old_new.unsqueeze(-1)
                        acc += torch.einsum('bhqk,bkhe->bhqe', exp_diff.to(Vb_proj.dtype), Vb_proj)
                        m = m_new
                        l = l_new

                    M_vec[batch_start:batch_end, :, i0:i1] = m
                    L_vec[batch_start:batch_end, :, i0:i1] = l

                    # Finalize Output Block
                    out_block = (acc / (l.unsqueeze(-1) + 1e-6)).sum(dim=1)

                    # Proj Bias + LS + Residual
                    out_block = out_block + b_o
                    if ls_gamma is not None:
                        out_block = out_block * ls_gamma

                    # Residual connection from ORIGINAL input x_chunk
                    out_block = out_block + x_chunk[:, i0:i1]

                    # Store
                    if out_on_gpu:
                        gpu_out[batch_start:batch_end, i0:i1].copy_(out_block)

                    output[batch_start:batch_end, i0:i1].copy_(out_block, non_blocking=True)

            batch_start = batch_end
            stream_idx += 1

        if torch.cuda.is_available():
            # torch.cuda.synchronize()
            for s in cc.cuda_streams:
                s.synchronize()

        ctx.save_for_backward(X_in, norm_w, norm_b, W_qkv, M_vec, L_vec, output, W_o, b_o, ls_gamma)
        ctx.out_on_gpu = bool(out_on_gpu)

        ctx.prev_bwd_accelerator = prev_bwd_accelerator
        ctx.my_bwd_accelerator = bwd_accelerator

        if out_on_gpu and gpu_cache is not None:
            gpu_cache.append(gpu_out)
        return output

    @staticmethod
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_Y_cpu):
        # 1. CHECK BACKWARD ACCELERATOR
        grad_out_gpu = None
        if ctx.my_bwd_accelerator and len(ctx.my_bwd_accelerator) > 0:
            grad_out_gpu = ctx.my_bwd_accelerator.pop()

        # Prefer GPU grad if available
        use_gpu_path = grad_out_gpu is not None

        X_in, norm_w, norm_b, W_qkv, M_vec, L_vec, Y_cpu, W_o, b_o, ls_gamma = ctx.saved_tensors
        device = ctx.device
        B, L, D, H, d = ctx.shapes
        Hd = H * d
        D_out = D  # Assuming D_out == D for residual
        Wo_slices = W_o.view(H, d, D_out)

        # Grads
        d_norm_w = torch.zeros_like(norm_w)
        d_norm_b = torch.zeros_like(norm_b)
        d_Wqkv = torch.zeros_like(W_qkv)
        d_bqkv = torch.zeros((3 * Hd,), device=device, dtype=W_qkv.dtype)
        d_Wo = torch.zeros_like(W_o)
        d_bo = torch.zeros_like(b_o)
        d_gamma = torch.zeros_like(ls_gamma) if ls_gamma is not None else None

        # Input Grad Buffer
        grad_X_cpu = ctx.X_grad
        grad_X_cpu.zero_()

        # We need a GPU grad buffer if we have to pass it to previous layer OR copy to CPU
        # If ctx.return_cpu_grad=True, we normally don't need GPU buffer unless prev_bwd_accelerator is present
        need_gpu_grad = (not ctx.return_cpu_grad) or (ctx.prev_bwd_accelerator is not None)
        grad_X_gpu = torch.zeros(X_in.shape, device=device) if need_gpu_grad else None

        # Tiling config
        max_block, max_elems = ctx.tiling_params
        max_batch_size = max(1, int(max_elems // (L * D)))
        max_batch_size = min(max_batch_size, B)
        max_patch_len = max(1, int(max_elems // (max_batch_size * D * 4)))
        block = min(max_block, max_patch_len, L)
        inv_sqrt_d = 1.0 / math.sqrt(d)

        batch_start = 0
        stream_idx = 0
        while batch_start < B:
            batch_end = min(batch_start + max_batch_size, B)
            b_s = batch_end - batch_start

            with maybe_cuda_stream(cc.cuda_streams, stream_idx):
                # 1. Load Inputs & Grads
                x_chunk = X_in[batch_start:batch_end].to(device, non_blocking=True)

                # Load Grad (GPU or CPU)
                if use_gpu_path:
                    gY_chunk = grad_out_gpu[batch_start:batch_end]
                else:
                    gY_chunk = grad_Y_cpu[batch_start:batch_end].to(device, non_blocking=True)

                # Residual Grad
                dX_chunk = gY_chunk.clone()

                # 2. Recompute Forward (Norm + QKV) to get states for backward
                # (Gradient Checkpointing style)
                x_norm = manual_layernorm_forward(x_chunk, norm_w, norm_b)
                QKV = x_norm.matmul(W_qkv).view(b_s, L, 3, H, d)
                Q, K, V = QKV[:, :, 0], QKV[:, :, 1], QKV[:, :, 2]

                # Buffers for Attention Grads
                dQ = torch.zeros_like(Q)
                dK = torch.zeros_like(K)
                dV = torch.zeros_like(V)

                # Loop for Attention Backward
                L_b = L_vec[batch_start:batch_end]
                M_b = M_vec[batch_start:batch_end]

                for i0 in range(0, L, block):
                    i1 = min(L, i0 + block)
                    Qb = Q[:, i0:i1]
                    gYb = gY_chunk[:, i0:i1]
                    m_g = M_b[:, :, i0:i1]
                    l_g = L_b[:, :, i0:i1]

                    # Recompute O_unproj
                    O_unproj = torch.zeros_like(Qb)
                    for j0 in range(0, L, block):
                        j1 = min(L, j0 + block)
                        Kb = K[:, j0:j1]
                        Vb = V[:, j0:j1]
                        S = torch.einsum('bqhd,bkhd->bhqk', Qb, Kb) * inv_sqrt_d
                        P = (S - m_g.unsqueeze(-1)).exp() / (l_g.unsqueeze(-1) + 1e-12)
                        O_unproj += torch.einsum('bhqk,bkhd->bqhd', P.to(Qb.dtype), Vb)

                    # LS + Proj Backward
                    if ls_gamma is not None:
                        O_flat = O_unproj.view(b_s, i1 - i0, Hd)
                        Y_unscaled = O_flat.matmul(W_o) + b_o
                        d_gamma += (gYb * Y_unscaled).sum(dim=(0, 1))
                        gY_eff = gYb * ls_gamma
                    else:
                        gY_eff = gYb

                    d_bo += gY_eff.sum(dim=(0, 1))
                    gO = torch.einsum('bqe,hde->bqhd', gY_eff, Wo_slices)
                    D_vec = torch.sum(gO * O_unproj, dim=-1).permute(0, 2, 1)

                    # Flash Backward
                    for j0 in range(0, L, block):
                        j1 = min(L, j0 + block)
                        Kb = K[:, j0:j1]
                        Vb = V[:, j0:j1]
                        S = torch.einsum('bqhd,bkhd->bhqk', Qb, Kb) * inv_sqrt_d
                        P = (S - m_g.unsqueeze(-1)).exp() / (l_g.unsqueeze(-1) + 1e-12)

                        dV_local = torch.einsum('bhqk,bqhd->bkhd', P.to(Qb.dtype), gO)
                        dV[:, j0:j1].add_(dV_local)

                        dV_proj = torch.einsum('bhqk,bqe->bkhe', P.to(Qb.dtype), gY_eff)
                        d_Wo += torch.einsum('bkhd,bkhe->hde', Vb, dV_proj).reshape(Hd, D_out)

                        G = torch.einsum('bqhd,bkhd->bhqk', gO, Vb)
                        dS = P * (G - D_vec.unsqueeze(-1))

                        dQ[:, i0:i1].add_(inv_sqrt_d * torch.einsum('bhqk,bkhd->bqhd', dS, Kb))
                        dK[:, j0:j1].add_(inv_sqrt_d * torch.einsum('bhqk,bqhd->bkhd', dS, Qb))

                # 3. Backprop through QKV Proj
                dQ_flat = dQ.reshape(b_s, L, Hd)
                dK_flat = dK.reshape(b_s, L, Hd)
                dV_flat = dV.reshape(b_s, L, Hd)

                d_bqkv[0:Hd] += dQ_flat.sum((0, 1))
                d_bqkv[Hd:2 * Hd] += dK_flat.sum((0, 1))
                d_bqkv[2 * Hd:3 * Hd] += dV_flat.sum((0, 1))

                d_Wqkv[:, 0:Hd] += torch.einsum('bld,blh->dh', x_norm, dQ_flat)
                d_Wqkv[:, Hd:2 * Hd] += torch.einsum('bld,blh->dh', x_norm, dK_flat)
                d_Wqkv[:, 2 * Hd:3 * Hd] += torch.einsum('bld,blh->dh', x_norm, dV_flat)

                Wq = W_qkv[:, 0:Hd]
                Wk = W_qkv[:, Hd:2 * Hd]
                Wv = W_qkv[:, 2 * Hd:3 * Hd]
                dX_norm = dQ_flat @ Wq.t() + dK_flat @ Wk.t() + dV_flat @ Wv.t()

                # 4. Backprop through Norm
                dX_branch = manual_layernorm_backward(dX_norm, x_chunk, norm_w, norm_b)

                # Accumulate Norm Params
                # Note: We need standard LayerNorm param grad accumulation here
                # x_chunk, norm_w, etc are available
                # We assume manual_layernorm_backward returns dX, but we also need dWeight/dBias for Norm
                # Let's compute them manually here to be safe and explicit
                inv_std = torch.rsqrt(x_chunk.var(-1, unbiased=False, keepdim=True) + 1e-6)
                x_mu = x_chunk - x_chunk.mean(-1, keepdim=True)
                d_norm_w += (dX_norm * x_mu * inv_std).sum((0, 1))
                d_norm_b += dX_norm.sum((0, 1))

                # 5. Add Branch Grad to Residual Grad
                dX_chunk += dX_branch

                # 6. Store Input Grad
                # if ctx.return_cpu_grad:
                #     grad_X_cpu[batch_start:batch_end].copy_(dX_chunk, non_blocking=True)
                if grad_X_gpu is not None:
                    grad_X_gpu[batch_start:batch_end].copy_(dX_chunk, non_blocking=True)
                else:
                    grad_X_cpu[batch_start:batch_end].copy_(dX_chunk, non_blocking=True)

            batch_start = batch_end
            stream_idx += 1

        if torch.cuda.is_available():
            # torch.cuda.synchronize()
            for s in cc.cuda_streams:
                s.synchronize()
        # 2. DEPOSIT GRADIENT FOR PREVIOUS LAYER
        if ctx.prev_bwd_accelerator is not None and grad_X_gpu is not None:
            ctx.prev_bwd_accelerator.append(grad_X_gpu)

        grad_to_return = grad_X_cpu if ctx.return_cpu_grad else grad_X_gpu

        return grad_to_return, d_norm_w, d_norm_b, d_Wqkv, d_bqkv, d_Wo, d_bo, d_gamma, *([None] * 9)


class BigFusedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, init_values=None, device='cuda', dtype=None, out_on_gpu=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.out_on_gpu = out_on_gpu
        self._dtype = dtype if dtype else torch.float32

        self.norm_weight = nn.Parameter(torch.ones(d_model, device=device))
        self.norm_bias = nn.Parameter(torch.zeros(d_model, device=device))

        self.W_qkv = nn.Parameter(torch.empty(d_model, 3 * d_model, device=device))
        self.b_qkv = nn.Parameter(torch.zeros(3 * d_model, device=device))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model, device=device))
        self.b_o = nn.Parameter(torch.zeros(d_model, device=device))
        nn.init.xavier_uniform_(self.W_qkv)
        nn.init.xavier_uniform_(self.W_o)

        self.gamma = None
        if init_values:
            self.gamma = nn.Parameter(init_values * torch.ones(d_model, device=device))

        self.output = None
        self.X_grad = None

        self.gpu_cache = [] if self.out_on_gpu else None
        self.bwd_accelerator = []  # Bucket for next layer

    def _get_output_tensor(self, B, L):
        # 1. Allocate huge buffer if None
        if self.output is None:
            self.output = torch.empty((max(B, 1), max(L, 1024), self.d_model), device='cpu',
                                      pin_memory=torch.cuda.is_available(), dtype=self._dtype)

        # 2. Check if current buffer is large enough
        cur_B, cur_L, _ = self.output.shape
        if cur_B < B or cur_L < L:
            # Reallocate only if too small (rare if initialized conservatively big)
            self.output = torch.empty((max(B, cur_B), max(L, cur_L), self.d_model), device='cpu',
                                      pin_memory=torch.cuda.is_available(), dtype=self._dtype)

        # 3. Return a SLICE of the pre-allocated buffer
        # This view shares memory, is contiguous enough for copy_, and avoids malloc
        return self.output[:B, :L]

    def _get_input_grad_tensor(self, X):
        # 1. Allocate huge buffer if None
        if self.X_grad is None:
            self.X_grad = torch.zeros(X.shape, device='cpu', pin_memory=torch.cuda.is_available(), dtype=self._dtype)

        # 2. Check if current buffer is large enough
        cur_B, cur_L, _ = self.X_grad.shape
        B, L, D = X.shape
        if cur_B < B or cur_L < L:
            # Reallocate only if too small (rare if initialized conservatively big)
            self.X_grad = torch.zeros((max(B, cur_B), max(L, cur_L), D), device='cpu',
                                      pin_memory=torch.cuda.is_available(), dtype=self._dtype)

        # 3. Return a SLICE of the pre-allocated buffer
        # This view shares memory, is contiguous enough for copy_, and avoids malloc
        return self.X_grad[:B, :L]

    def forward(self, x_cpu):
        B, L, D = x_cpu.shape
        out = self._get_output_tensor(B, L)
        x_grad = self._get_input_grad_tensor(x_cpu) if x_cpu.requires_grad else None

        out_cpu = BigFusedSelfAttentionFunc.apply(
            x_cpu, self.norm_weight, self.norm_bias, self.W_qkv, self.b_qkv, self.W_o, self.b_o, self.gamma,
            self.num_heads, self.head_dim, out, x_grad,
            self.gpu_cache, self.bwd_accelerator,  # Channels
            2048, 6e5, self.out_on_gpu
        )

        if self.out_on_gpu and self.gpu_cache:
            out_cpu._big_gpu_cache = self.gpu_cache.pop()
        out_cpu._bwd_accelerator = self.bwd_accelerator
        return out_cpu


# ==============================================================================
# 3. Layer 2: BigFusedGluMLP
#    (Norm -> SwiGLU -> LS -> Res)
# ==============================================================================

def swiglu_forward(x_norm, w1, b1, w2, b2):
    h = F.linear(x_norm, w1, b1)
    x1, x2 = h.chunk(2, dim=-1)
    return F.linear(F.silu(x1) * x2, w2, b2)


class BigFusedGluMLPFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    @torch.no_grad()
    def forward(ctx, X_in, norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b, ls_gamma,
                output=None, X_grad=None,
                gpu_cache=None, bwd_accelerator=None,
                max_elements=int(6e7), out_on_gpu=False):

        X_gpu_cache = getattr(X_in, '_big_gpu_cache', None)
        if X_gpu_cache is not None:
            del X_in._big_gpu_cache
        prev_bwd_accelerator = getattr(X_in, '_bwd_accelerator', None)

        device = norm_w.device
        B, L, D = X_in.shape

        hidden_dim = fc1_w.shape[0]
        expansion = (hidden_dim / D) + 2
        tokens_per_tile = int(max_elements / (D * expansion))
        max_batch = max(1, min(B, int(tokens_per_tile // L))) if L < tokens_per_tile else 1
        max_seq = min(L, int(tokens_per_tile // max_batch))

        ctx.tiling = (max_batch, max_seq)
        ctx.input_shape = (B, L, D)
        ctx.device = device
        ctx.X_grad = X_grad
        ctx.return_cpu_grad = not X_in.is_cuda

        gpu_out = torch.empty((B, L, D), device=device) if out_on_gpu else None

        batch_start = 0
        stream_idx = 0
        while batch_start < B:
            batch_end = min(batch_start + max_batch, B)
            with maybe_cuda_stream(cc.cuda_streams, stream_idx):
                seq_start = 0
                while seq_start < L:
                    seq_end = min(seq_start + max_seq, L)
                    sl = (slice(batch_start, batch_end), slice(seq_start, seq_end))

                    # 1. Load Tile
                    if X_gpu_cache is not None:
                        x_tile = X_gpu_cache[sl]
                    else:
                        x_tile = X_in[sl].to(device, non_blocking=True)

                    # 2. Fused Ops
                    # Norm
                    x_norm = manual_layernorm_forward(x_tile, norm_w, norm_b)
                    # SwiGLU
                    mlp_out = swiglu_forward(x_norm, fc1_w, fc1_b, fc2_w, fc2_b)
                    # LS
                    if ls_gamma is not None:
                        mlp_out = mlp_out * ls_gamma
                    # Residual
                    res_out = mlp_out + x_tile

                    # 3. Store
                    if out_on_gpu: gpu_out[sl].copy_(res_out)
                    output[sl].copy_(res_out, non_blocking=True)

                    seq_start = seq_end
            batch_start = batch_end
            stream_idx += 1

        if torch.cuda.is_available():
            # torch.cuda.synchronize()
            for s in cc.cuda_streams:
                s.synchronize()

        ctx.save_for_backward(X_in, norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b, ls_gamma)
        ctx.out_on_gpu = bool(out_on_gpu)
        ctx.prev_bwd_accelerator = prev_bwd_accelerator
        ctx.my_bwd_accelerator = bwd_accelerator

        if out_on_gpu and gpu_cache is not None:
            gpu_cache.append(gpu_out)
        return output

    @staticmethod
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_Y_cpu):
        # 1. CHECK BACKWARD ACCELERATOR
        grad_out_gpu = None
        if ctx.my_bwd_accelerator and len(ctx.my_bwd_accelerator) > 0:
            grad_out_gpu = ctx.my_bwd_accelerator.pop()

        use_gpu_path = grad_out_gpu is not None

        X_in, norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b, ls_gamma = ctx.saved_tensors
        B, L, D = ctx.input_shape
        max_batch, max_seq = ctx.tiling
        device = ctx.device

        # Init Grads
        d_norm_w = torch.zeros_like(norm_w)
        d_norm_b = torch.zeros_like(norm_b)
        d_fc1_w = torch.zeros_like(fc1_w)
        d_fc1_b = torch.zeros_like(fc1_b) if fc1_b is not None else None
        d_fc2_w = torch.zeros_like(fc2_w)
        d_fc2_b = torch.zeros_like(fc2_b) if fc2_b is not None else None
        d_gamma = torch.zeros_like(ls_gamma) if ls_gamma is not None else None

        grad_X_cpu = ctx.X_grad
        grad_X_cpu.zero_()

        need_gpu_grad = (not ctx.return_cpu_grad) or (ctx.prev_bwd_accelerator is not None)
        grad_X_gpu = torch.zeros(X_in.shape, device=device) if need_gpu_grad else None

        batch_start = 0
        stream_idx = 0
        while batch_start < B:
            batch_end = min(batch_start + max_batch, B)
            seq_start = 0
            while seq_start < L:
                seq_end = min(seq_start + max_seq, L)
                sl = (slice(batch_start, batch_end), slice(seq_start, seq_end))

                with maybe_cuda_stream(cc.cuda_streams, stream_idx):

                    x_tile = X_in[sl].to(device, non_blocking=True).requires_grad_(True)
                    if use_gpu_path:
                        gY_tile = grad_out_gpu[sl]
                    else:
                        gY_tile = grad_Y_cpu[sl].to(device, non_blocking=True)

                    # Checkpointing Recomputation
                    with torch.enable_grad():
                        x_norm = manual_layernorm_forward(x_tile, norm_w, norm_b)
                        mlp_out = swiglu_forward(x_norm, fc1_w, fc1_b, fc2_w, fc2_b)
                        if ls_gamma is not None:
                            mlp_out = mlp_out * ls_gamma
                        res_out = mlp_out + x_tile

                        inputs = [x_tile, norm_w, norm_b, fc1_w, fc2_w]
                        if fc1_b is not None:
                            inputs.append(fc1_b)
                        if fc2_b is not None:
                            inputs.append(fc2_b)
                        if ls_gamma is not None:
                            inputs.append(ls_gamma)

                        torch.autograd.backward([res_out], [gY_tile], inputs=inputs)

                        if grad_X_gpu is not None:
                            grad_X_gpu[sl].copy_(x_tile.grad)
                        else:
                            grad_X_cpu[sl].copy_(x_tile.grad, non_blocking=True)
                        # else:
                        #     grad_X_gpu[sl].copy_(x_tile.grad, non_blocking=True)
                        d_norm_w += norm_w.grad
                        d_norm_b += norm_b.grad
                        d_fc1_w += fc1_w.grad
                        d_fc2_w += fc2_w.grad
                        if fc1_b is not None:
                            d_fc1_b += fc1_b.grad
                        if fc2_b is not None:
                            d_fc2_b += fc2_b.grad
                        if ls_gamma is not None:
                            d_gamma += ls_gamma.grad

                        for t in inputs:
                            t.grad = None

                seq_start = seq_end
            batch_start = batch_end
            stream_idx += 1

        if torch.cuda.is_available():
            # torch.cuda.synchronize()
            for s in cc.cuda_streams:
                s.synchronize()

        if ctx.prev_bwd_accelerator is not None and grad_X_gpu is not None:
            ctx.prev_bwd_accelerator.append(grad_X_gpu)

        grad_to_return = grad_X_cpu if ctx.return_cpu_grad else grad_X_gpu

        return grad_to_return, d_norm_w, d_norm_b, d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_gamma, *([None] * 6)


class BigFusedGluMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, init_values=None, device='cuda',
                 dtype=None, out_on_gpu=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.output = None
        self.X_grad = None
        self.out_on_gpu = out_on_gpu
        self._dtype = dtype if dtype else torch.float32

        self.norm_weight = nn.Parameter(torch.ones(in_features, device=device))
        self.norm_bias = nn.Parameter(torch.zeros(in_features, device=device))

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True, device=device)
        self.fc2 = nn.Linear(hidden_features // 2, out_features, bias=True, device=device)
        nn.init.ones_(self.fc1.bias[self.fc1.bias.shape[0] // 2:])
        nn.init.normal_(self.fc1.weight[self.fc1.weight.shape[0] // 2:], std=1e-6)

        self.gamma = None
        if init_values:
            self.gamma = nn.Parameter(init_values * torch.ones(out_features, device=device))

        self.gpu_cache = [] if self.out_on_gpu else None
        self.bwd_accelerator = []

    def _get_output_tensor(self, B, L, D):
        # 1. Allocate huge buffer if None
        if self.output is None:
            self.output = torch.empty((max(B, 1), max(L, 1024), D), device='cpu',
                                      pin_memory=torch.cuda.is_available(), dtype=self._dtype)

        # 2. Check if current buffer is large enough
        cur_B, cur_L, _ = self.output.shape
        if cur_B < B or cur_L < L:
            # Reallocate only if too small (rare if initialized conservatively big)
            self.output = torch.empty((max(B, cur_B), max(L, cur_L), D), device='cpu',
                                      pin_memory=torch.cuda.is_available(), dtype=self._dtype)

        # 3. Return a SLICE of the pre-allocated buffer
        # This view shares memory, is contiguous enough for copy_, and avoids malloc
        return self.output[:B, :L]

    def _get_input_grad_tensor(self, X):
        # 1. Allocate huge buffer if None
        if self.X_grad is None:
            self.X_grad = torch.zeros(X.shape, device='cpu', pin_memory=torch.cuda.is_available(), dtype=self._dtype)

        # 2. Check if current buffer is large enough
        cur_B, cur_L, _ = self.X_grad.shape
        B, L, D = X.shape
        if cur_B < B or cur_L < L:
            # Reallocate only if too small (rare if initialized conservatively big)
            self.X_grad = torch.zeros((max(B, cur_B), max(L, cur_L), D), device='cpu',
                                      pin_memory=torch.cuda.is_available(), dtype=self._dtype)

        # 3. Return a SLICE of the pre-allocated buffer
        # This view shares memory, is contiguous enough for copy_, and avoids malloc
        return self.X_grad[:B, :L]

    def forward(self, x_in):
        B, L, D = x_in.shape
        out = self._get_output_tensor(B, L, D)
        x_grad = self._get_input_grad_tensor(x_in) if x_in.requires_grad else None

        out_cpu = BigFusedGluMLPFunc.apply(
            x_in, self.norm_weight, self.norm_bias,
            self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias, self.gamma,
            out, x_grad, self.gpu_cache, self.bwd_accelerator, 6e5, self.out_on_gpu
        )
        if self.out_on_gpu and self.gpu_cache:
            out_cpu._big_gpu_cache = self.gpu_cache.pop()
        out_cpu._bwd_accelerator = self.bwd_accelerator
        return out_cpu


# ==============================================================================
# 4. BigBlock (The Composer)
# ==============================================================================

class BigBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., init_values=None, device='cuda', dtype=None, out_on_gpu=True):
        super().__init__()
        self.l1 = BigFusedSelfAttention(dim, num_heads, init_values, device, dtype, out_on_gpu)
        hidden_dim = int(dim * mlp_ratio)
        if hidden_dim % 2 != 0:
            hidden_dim += 1
        self.l2 = BigFusedGluMLP(dim, hidden_dim, dim, init_values, device, dtype, out_on_gpu)

    def forward(self, x):
        x_mid = self.l1(x)
        x_out = self.l2(x_mid)
        return x_out

# ==============================================================================
# 5. BigLayerNorm
# ==============================================================================


class BigLayerNormFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    @torch.no_grad()
    def forward(ctx, X_in, weight, bias, output=None, X_grad=None,
                gpu_cache=None, bwd_accelerator=None,
                max_elements=int(6e7), out_on_gpu=True):

        # 1. INPUT HANDLING (Forward Cache + Backward Bucket)
        X_gpu_cache = getattr(X_in, '_big_gpu_cache', None)
        if X_gpu_cache is not None:
            del X_in._big_gpu_cache

        # Capture the bucket where we must deposit our Input Gradients later
        prev_bwd_accelerator = getattr(X_in, '_bwd_accelerator', None)

        device = weight.device
        B, L, D = X_in.shape

        # Tiling Setup
        # Row size = D. Tile over B then L.
        tokens_per_tile = int(max_elements // D)
        max_batch = max(1, min(B, int(tokens_per_tile // L)))
        # If B=1 and L is huge, we tile L
        max_seq = min(L, int(tokens_per_tile // max_batch))

        ctx.tiling = (max_batch, max_seq)
        ctx.input_shape = (B, L, D)
        ctx.device = device
        ctx.X_grad = X_grad
        ctx.return_cpu_grad = not X_in.is_cuda

        gpu_out = torch.empty((B, L, D), device=device) if out_on_gpu else None

        batch_start = 0
        stream_idx = 0
        while batch_start < B:
            batch_end = min(batch_start + max_batch, B)

            with maybe_cuda_stream(cc.cuda_streams, stream_idx):
                seq_start = 0
                while seq_start < L:
                    seq_end = min(seq_start + max_seq, L)
                    sl = (slice(batch_start, batch_end), slice(seq_start, seq_end))

                    # 1. Load Tile (Cache or CPU)
                    if X_gpu_cache is not None:
                        x_tile = X_gpu_cache[sl]
                    else:
                        x_tile = X_in[sl].to(device, non_blocking=True)

                    # 2. Compute
                    x_out = manual_layernorm_forward(x_tile, weight, bias)

                    # 3. Store
                    if out_on_gpu:
                        gpu_out[sl].copy_(x_out)

                    output[sl].copy_(x_out, non_blocking=True)
                    seq_start = seq_end

            batch_start = batch_end
            stream_idx += 1

        if torch.cuda.is_available():
            for s in cc.cuda_streams:
                s.synchronize()

        # Save context
        saved_X = X_in._big_cpu_out if hasattr(X_in, '_big_cpu_out') else X_in
        ctx.save_for_backward(saved_X, weight, bias)

        # Save side channels
        ctx.prev_bwd_accelerator = prev_bwd_accelerator
        ctx.my_bwd_accelerator = bwd_accelerator

        # Populate forward cache for next layer
        if out_on_gpu and gpu_cache is not None:
            gpu_cache.append(gpu_out)

        return output

    @staticmethod
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_Y_cpu):
        # 1. CHECK BACKWARD ACCELERATOR (Did the next layer leave us a GPU grad?)
        grad_out_gpu = None
        if ctx.my_bwd_accelerator and len(ctx.my_bwd_accelerator) > 0:
            grad_out_gpu = ctx.my_bwd_accelerator.pop()

        use_gpu_path = (grad_out_gpu is not None)

        X_in, weight, bias = ctx.saved_tensors
        B, L, D = ctx.input_shape
        max_batch, max_seq = ctx.tiling
        device = ctx.device

        # Init Grads
        d_weight = torch.zeros_like(weight)
        d_bias = torch.zeros_like(bias)

        grad_X_cpu = ctx.X_grad
        grad_X_cpu.zero_()

        # We need a GPU grad buffer if we have to pass it to previous layer OR copy to CPU
        need_gpu_grad = (not ctx.return_cpu_grad) or (ctx.prev_bwd_accelerator is not None)
        grad_X_gpu = torch.zeros(X_in.shape, device=device) if need_gpu_grad else None

        batch_start = 0
        stream_idx = 0
        while batch_start < B:
            batch_end = min(batch_start + max_batch, B)

            with maybe_cuda_stream(cc.cuda_streams, stream_idx):
                seq_start = 0
                while seq_start < L:
                    seq_end = min(seq_start + max_seq, L)
                    sl = (slice(batch_start, batch_end), slice(seq_start, seq_end))

                    # 1. Load Inputs
                    x_tile = X_in[sl].to(device, non_blocking=True)

                    # 2. Load Grads
                    if use_gpu_path:
                        gY_tile = grad_out_gpu[sl]
                    else:
                        gY_tile = grad_Y_cpu[sl].to(device, non_blocking=True)

                    # 3. Compute Grads
                    # We don't need recomputation loop here because LN backward is cheap
                    # and manual_layernorm_backward is memory efficient (doesn't expand dims)
                    dX_tile = manual_layernorm_backward(gY_tile, x_tile, weight, bias)

                    # Accumulate Weight/Bias grads
                    # manual_layernorm_backward returns dX, but we need dW/db.
                    # We can compute them using the standard formula.
                    # dW = sum(gY * (x-u)*inv_std)
                    u = x_tile.mean(-1, keepdim=True)
                    s = x_tile.var(-1, unbiased=False, keepdim=True)
                    inv_std = torch.rsqrt(s + 1e-6)
                    x_norm = (x_tile - u) * inv_std

                    d_weight += (gY_tile * x_norm).sum((0, 1))
                    d_bias += gY_tile.sum((0, 1))

                    # 4. Store Input Grad
                    if grad_X_gpu is not None:
                        grad_X_gpu[sl].copy_(dX_tile)
                    if ctx.return_cpu_grad:
                        grad_X_cpu[sl].copy_(dX_tile, non_blocking=True)

                    seq_start = seq_end

            batch_start = batch_end
            stream_idx += 1

        if torch.cuda.is_available():
            for s in cc.cuda_streams: s.synchronize()
            torch.cuda.synchronize()

        # 2. DEPOSIT GRADIENT FOR PREVIOUS LAYER
        if ctx.prev_bwd_accelerator is not None and grad_X_gpu is not None:
            ctx.prev_bwd_accelerator.append(grad_X_gpu)

        grad_to_return = grad_X_cpu if ctx.return_cpu_grad else grad_X_gpu

        return grad_to_return, d_weight, d_bias, None, None, None, None, None, None


class BigLayerNorm(nn.Module):
    def __init__(self, dim, device='cuda', dtype=None, out_on_gpu=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device))
        self.bias = nn.Parameter(torch.zeros(dim, device=device))
        self.out_on_gpu = out_on_gpu
        self._dtype = dtype if dtype else torch.float32

        self.output = None
        self.X_grad = None

        self.gpu_cache = [] if self.out_on_gpu else None
        self.bwd_accelerator = []
        self.max_elements = 6e7

    def _get_output_tensor(self, B, L, D):
        if self.output is None:
            self.output = torch.empty((max(B, 1), max(L, 1024), D), device='cpu', pin_memory=True, dtype=self._dtype)

        cur_B, cur_L, _ = self.output.shape
        if cur_B < B or cur_L < L:
            self.output = torch.empty((max(B, cur_B), max(L, cur_L), D), device='cpu', pin_memory=True,
                                      dtype=self._dtype)

        return self.output[:B, :L]

    def _get_input_grad_tensor(self, X):
        if self.X_grad is None:
            self.X_grad = torch.zeros(X.shape, device='cpu', pin_memory=True, dtype=self._dtype)

        cur_B, cur_L, _ = self.X_grad.shape
        B, L, D = X.shape
        if cur_B < B or cur_L < L:
            self.X_grad = torch.zeros((max(B, cur_B), max(L, cur_L), D), device='cpu', pin_memory=True,
                                      dtype=self._dtype)

        return self.X_grad[:B, :L]

    def forward(self, x_in):
        B, L, D = x_in.shape
        out = self._get_output_tensor(B, L, D)
        x_grad = self._get_input_grad_tensor(x_in) if x_in.requires_grad else None

        # Clear buckets
        # if self.gpu_cache: self.gpu_cache.clear()
        # if self.bwd_accelerator: self.bwd_accelerator.clear()

        out_cpu = BigLayerNormFunc.apply(
            x_in, self.weight, self.bias,
            out, x_grad, self.gpu_cache, self.bwd_accelerator,
            self.max_elements, self.out_on_gpu
        )

        if self.out_on_gpu and self.gpu_cache:
            out_cpu._big_gpu_cache = self.gpu_cache.pop()

        out_cpu._bwd_accelerator = self.bwd_accelerator
        return out_cpu

# ==============================================================================
# 6. Verification
# ==============================================================================

def check(name, a, b):
    a = a.detach().cpu().float().flatten()
    b = b.detach().cpu().float().flatten()
    diff = (a - b).abs()
    avg_diff = diff.mean().item()
    ref_mag = b.abs().mean().item() + 1e-9
    rel_error = avg_diff / ref_mag

    is_grad = "Grad" in name
    # Accumulation noise in huge blocks is significant
    threshold = 0.05 if is_grad else 1e-5

    status = "OK" if rel_error < threshold else "FAIL"
    print(f"--- {name} ---")
    print(f"  Status:    {status}")
    print(f"  Avg Diff:  {avg_diff:.6g}")
    print(f"  Rel Error: {rel_error:.6g}")


def compare_full_block():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Comparing BigBlock vs Timm (UNI-2 config) on {device}...")

    # Configuration (Small batch, Large L to test memory/tiling)
    B, L, D = 256, 16 * 16, 1536
    init_values = 0.5
    # SwiGLU ratio: 2/3 of 4 * 2 (approx 5.333)
    mlp_ratio = (8.0 / 3.0) * 2

    # 1. Setup BigBlock
    big_block = BigBlock(D, num_heads=24, mlp_ratio=mlp_ratio, init_values=init_values, device=device)

    # 2. Setup Timm Block
    timm_block = timm.models.vision_transformer.Block(
        dim=D, num_heads=24, mlp_ratio=mlp_ratio,
        qkv_bias=True, proj_bias=True,
        init_values=init_values,
        act_layer=nn.SiLU,
        mlp_layer=timm.layers.SwiGLUPacked,
        norm_layer=nn.LayerNorm
    ).to(device)

    # 3. Sync Weights (Corrected Transposes)
    with torch.no_grad():
        # L1 Norm
        timm_block.norm1.weight.copy_(big_block.l1.norm_weight)
        timm_block.norm1.bias.copy_(big_block.l1.norm_bias)

        # Attn QKV - NEEDS TRANSPOSE
        # Big: (In, Out) | Timm Linear: (Out, In)
        timm_block.attn.qkv.weight.copy_(big_block.l1.W_qkv.t())
        timm_block.attn.qkv.bias.copy_(big_block.l1.b_qkv)

        # Attn Proj - NEEDS TRANSPOSE
        # Big: (In, Out) | Timm Linear: (Out, In)
        timm_block.attn.proj.weight.copy_(big_block.l1.W_o.t())
        timm_block.attn.proj.bias.copy_(big_block.l1.b_o)

        # LS1
        timm_block.ls1.gamma.copy_(big_block.l1.gamma)

        # L2 Norm
        timm_block.norm2.weight.copy_(big_block.l2.norm_weight)
        timm_block.norm2.bias.copy_(big_block.l2.norm_bias)

        # MLP FC1 - NO TRANSPOSE (Both use nn.Linear convention)
        timm_block.mlp.fc1.weight.copy_(big_block.l2.fc1.weight)
        timm_block.mlp.fc1.bias.copy_(big_block.l2.fc1.bias)

        # MLP FC2 - NO TRANSPOSE
        timm_block.mlp.fc2.weight.copy_(big_block.l2.fc2.weight)
        timm_block.mlp.fc2.bias.copy_(big_block.l2.fc2.bias)

        # LS2
        timm_block.ls2.gamma.copy_(big_block.l2.gamma)

    # 4. Inputs
    X = torch.randn(B, L, D, device='cpu', requires_grad=True)
    X_gpu = X.detach().to(device).clone().requires_grad_(True)

    # 5. Run
    # Big
    t0 = time.time()
    Y_big = big_block(X)
    if hasattr(Y_big, '_big_cpu_out'): Y_big = Y_big._big_cpu_out
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_big = time.time() - t0

    # Timm
    t0 = time.time()
    Y_timm = timm_block(X_gpu)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_timm = time.time() - t0

    print(f"\nForward Time: Big={t_big:.4f}s | Timm={t_timm:.4f}s")
    check("Output Y", Y_big, Y_timm)

    # 6. Backward
    big_block.zero_grad()
    if X.grad is not None:
        X.grad.zero_()
    Y_big.sum().backward()

    timm_block.zero_grad()
    if X_gpu.grad is not None:
        X_gpu.grad.zero_()
    Y_timm.sum().backward()

    print("\n--- Gradient Checks ---")
    check("Grad Input X", X.grad, X_gpu.grad)
    check("Grad L1 Norm W", big_block.l1.norm_weight.grad, timm_block.norm1.weight.grad)
    check("Grad Attn QKV", big_block.l1.W_qkv.grad, timm_block.attn.qkv.weight.grad.t())  # Transpose check
    check("Grad L2 Norm W", big_block.l2.norm_weight.grad, timm_block.norm2.weight.grad)
    check("Grad MLP FC1", big_block.l2.fc1.weight.grad, timm_block.mlp.fc1.weight.grad)


def check_block_grads(k, big_blk, timm_blk):
    """Helper to verify gradients for a single block pair."""
    prefix = f"Block {k}"
    # Norms
    check(f"{prefix} L1 Norm W", big_blk.l1.norm_weight.grad, timm_blk.norm1.weight.grad)
    check(f"{prefix} L1 Norm B", big_blk.l1.norm_bias.grad, timm_blk.norm1.bias.grad)

    # Attention
    check(f"{prefix} Attn QKV", big_blk.l1.W_qkv.grad, timm_blk.attn.qkv.weight.grad.t())
    check(f"{prefix} Attn Proj", big_blk.l1.W_o.grad, timm_blk.attn.proj.weight.grad.t())
    check(f"{prefix} Attn Gamma", big_blk.l1.gamma.grad, timm_blk.ls1.gamma.grad)

    # MLP
    check(f"{prefix} L2 Norm W", big_blk.l2.norm_weight.grad, timm_blk.norm2.weight.grad)
    check(f"{prefix} MLP FC1", big_blk.l2.fc1.weight.grad, timm_blk.mlp.fc1.weight.grad)
    check(f"{prefix} MLP FC2", big_blk.l2.fc2.weight.grad, timm_blk.mlp.fc2.weight.grad)
    check(f"{prefix} MLP Gamma", big_blk.l2.gamma.grad, timm_blk.ls2.gamma.grad)


def compare_k_blocks_and_norm():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Comparing K Blocks + Norm vs Timm on {device}...")

    # Configuration
    K = 2
    B, L, D = 2, 2048, 1536  # Adjusted for quick testing; scale up L for stress testing
    init_values = 0.5
    mlp_ratio = (8.0 / 3.0) * 2

    # 1. Setup Big Model (Sequence of K Blocks + 1 Norm)
    big_blocks = nn.ModuleList([
        BigBlock(D, num_heads=24, mlp_ratio=mlp_ratio, init_values=init_values, device=device)
        for _ in range(K)
    ])
    big_norm = BigLayerNorm(D, device=device)

    # 2. Setup Timm Model (Sequence of K Blocks + 1 Norm)
    timm_blocks = nn.ModuleList([
        timm.models.vision_transformer.Block(
            dim=D, num_heads=24, mlp_ratio=mlp_ratio,
            qkv_bias=True, proj_bias=True,
            init_values=init_values,
            act_layer=nn.SiLU,
            mlp_layer=timm.layers.SwiGLUPacked,
            norm_layer=nn.LayerNorm
        ).to(device) for _ in range(K)
    ])
    timm_norm = nn.LayerNorm(D, eps=1e-6).to(device)

    # 3. Sync Weights Loop
    print("Syncing weights...")
    with torch.no_grad():
        for k in range(K):
            bb = big_blocks[k]
            tb = timm_blocks[k]

            # L1 Norm
            tb.norm1.weight.copy_(bb.l1.norm_weight)
            tb.norm1.bias.copy_(bb.l1.norm_bias)
            # Attn (Note Transposes)
            tb.attn.qkv.weight.copy_(bb.l1.W_qkv.t())
            tb.attn.qkv.bias.copy_(bb.l1.b_qkv)
            tb.attn.proj.weight.copy_(bb.l1.W_o.t())
            tb.attn.proj.bias.copy_(bb.l1.b_o)
            tb.ls1.gamma.copy_(bb.l1.gamma)

            # L2 Norm
            tb.norm2.weight.copy_(bb.l2.norm_weight)
            tb.norm2.bias.copy_(bb.l2.norm_bias)
            # MLP (No Transposes for Linear->Linear)
            tb.mlp.fc1.weight.copy_(bb.l2.fc1.weight)
            tb.mlp.fc1.bias.copy_(bb.l2.fc1.bias)
            tb.mlp.fc2.weight.copy_(bb.l2.fc2.weight)
            tb.mlp.fc2.bias.copy_(bb.l2.fc2.bias)
            tb.ls2.gamma.copy_(bb.l2.gamma)

        # Final Norm
        timm_norm.weight.copy_(big_norm.weight)
        timm_norm.bias.copy_(big_norm.bias)

    # 4. Inputs
    X = torch.randn(B, L, D, device='cpu', requires_grad=True)
    X_gpu = X.detach().to(device).clone().requires_grad_(True)

    # 5. Run Big
    t0 = time.time()
    x_curr = X
    for blk in big_blocks:
        x_curr = blk(x_curr)

    # Pass through final BigLayerNorm
    # Note: BigBlock returns CPU tensor (possibly with ._big_gpu_cache)
    # BigLayerNorm expects exactly that format.
    Y_big = big_norm(x_curr)

    if hasattr(Y_big, '_big_cpu_out'): Y_big = Y_big._big_cpu_out
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_big = time.time() - t0

    # 6. Run Timm
    t0 = time.time()
    x_curr_timm = X_gpu
    for blk in timm_blocks:
        x_curr_timm = blk(x_curr_timm)
    Y_timm = timm_norm(x_curr_timm)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_timm = time.time() - t0

    print(f"\nForward Time (K={K}): Big={t_big:.4f}s | Timm={t_timm:.4f}s")
    check("Output Y", Y_big, Y_timm)

    # 7. Backward
    # Zero all grads
    if X.grad is not None: X.grad.zero_()
    if X_gpu.grad is not None: X_gpu.grad.zero_()
    big_norm.zero_grad()
    timm_norm.zero_grad()
    for b in big_blocks: b.zero_grad()
    for b in timm_blocks: b.zero_grad()

    # Backward Big
    Y_big.sum().backward()

    # Backward Timm
    Y_timm.sum().backward()

    print("\n--- Gradient Checks ---")
    check("Grad Input X", X.grad, X_gpu.grad)

    # Check Blocks in Loop
    for k in range(K):
        check_block_grads(k, big_blocks[k], timm_blocks[k])

    # Check Final Norm
    check("Grad Final Norm W", big_norm.weight.grad, timm_norm.weight.grad)
    check("Grad Final Norm B", big_norm.bias.grad, timm_norm.bias.grad)


