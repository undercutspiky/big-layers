"""EXPERIMENTAL: archived hot-potato path, not used by the paper training entry points."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from big_layers.amp_compat import custom_bwd, custom_fwd
from contextlib import nullcontext
from . import cuda_config as cc


def maybe_cuda_stream(streams, idx):
    if torch.cuda.is_available() and streams and len(streams) > 0:
        return torch.cuda.stream(streams[idx % len(streams)])
    return nullcontext()


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
# 1. BigLayerNorm (Full Implementation with 2-Way Hot Potato)
# ==============================================================================

class BigLayerNormFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    @torch.no_grad()
    def forward(ctx, X_in, weight, bias, output=None, X_grad=None,
                gpu_cache=None, bwd_accelerator=None,  # Output side-channels
                max_elements=int(6e7), out_on_gpu=True):

        # 1. INPUT HOT POTATO (Forward Cache)
        # Check input for GPU cache
        X_gpu_cache = getattr(X_in, '_big_gpu_cache', None)
        if X_gpu_cache is not None: del X_in._big_gpu_cache

        # 2. INPUT BACKWARD ACCELERATOR (Capture from previous layer)
        # If the previous layer attached a bucket for us to drop grads into, grab it.
        prev_bwd_accelerator = getattr(X_in, '_bwd_accelerator', None)
        # We don't delete this attribute because multiple consumers might need to know 
        # (though in linear ViT, it's 1-to-1).

        device = weight.device
        B, L, D = X_in.shape

        # Tiling
        # Row size = D. Tile over B then L.
        tokens_per_tile = int(max_elements // D)
        max_batch = max(1, min(B, int(tokens_per_tile // L))) if L < tokens_per_tile else 1
        max_seq = min(L, int(tokens_per_tile // max_batch))

        ctx.tiling = (max_batch, max_seq)
        ctx.input_shape = (B, L, D)
        ctx.device = device
        ctx.X_grad = X_grad
        ctx.return_cpu_grad = not X_in.is_cuda

        if output is None: output = torch.empty((B, L, D), device='cpu', pin_memory=True)
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

                    # Load (Cache or CPU)
                    if X_gpu_cache is not None:
                        x_tile = X_gpu_cache[sl]
                    else:
                        x_tile = X_in[sl].to(device, non_blocking=True)

                    # Compute
                    # Standard LN: (x - u) / s * w + b
                    u = x_tile.mean(-1, keepdim=True)
                    s = x_tile.var(-1, unbiased=False, keepdim=True)
                    x_out = (x_tile - u) * torch.rsqrt(s + 1e-5) * weight + bias

                    # Store
                    if out_on_gpu: gpu_out[sl].copy_(x_out)
                    output[sl].copy_(x_out, non_blocking=True)
                    seq_start = seq_end
            batch_start = batch_end
            stream_idx += 1

        if torch.cuda.is_available():
            for s in cc.cuda_streams: s.synchronize()
            if not out_on_gpu: torch.cuda.synchronize()

        # Save for Backward
        saved_X = X_in._big_cpu_out if hasattr(X_in, '_big_cpu_out') else X_in
        ctx.save_for_backward(saved_X, weight, bias)

        # Save the bucket where we must deposit our Input Gradients later
        ctx.prev_bwd_accelerator = prev_bwd_accelerator

        # Save the bucket we created for the Next layer to deposit Gradients into
        ctx.my_bwd_accelerator = bwd_accelerator

        # Output Side Channels
        if out_on_gpu and gpu_cache is not None: gpu_cache.append(gpu_out)

        return output

    @staticmethod
    @custom_bwd(device_type='cuda')
    @torch.no_grad()
    def backward(ctx, grad_out_cpu):
        # 1. CHECK BACKWARD ACCELERATOR (Did the next layer leave us a GPU grad?)
        # If next layer was "Big", it put grad in ctx.my_bwd_accelerator
        grad_out_gpu = None
        if ctx.my_bwd_accelerator and len(ctx.my_bwd_accelerator) > 0:
            grad_out_gpu = ctx.my_bwd_accelerator.pop()  # Retrieve and clear

        # Decide source
        grad_Y = grad_out_gpu if grad_out_gpu is not None else grad_out_cpu

        X_in, weight, bias = ctx.saved_tensors
        B, L, D = ctx.input_shape
        max_batch, max_seq = ctx.tiling
        device = ctx.device

        d_weight = torch.zeros_like(weight)
        d_bias = torch.zeros_like(bias)

        grad_X_cpu = ctx.X_grad
        grad_X_cpu.zero_()
        grad_X_gpu = torch.zeros(X_in.shape, device=device) if (
                    not ctx.return_cpu_grad or ctx.prev_bwd_accelerator is not None) else None

        batch_start = 0
        stream_idx = 0
        while batch_start < B:
            batch_end = min(batch_start + max_batch, B)
            seq_start = 0
            while seq_start < L:
                seq_end = min(seq_start + max_seq, L)
                sl = (slice(batch_start, batch_end), slice(seq_start, seq_end))

                with maybe_cuda_stream(cc.cuda_streams, stream_idx):
                    x_tile = X_in[sl].to(device, non_blocking=True)
                    gY_tile = grad_Y[sl].to(device, non_blocking=True)

                    # Backward Math
                    N = D
                    u = x_tile.mean(-1, keepdim=True)
                    s = x_tile.var(-1, unbiased=False, keepdim=True)
                    std = torch.sqrt(s + 1e-5)
                    inv_std = 1.0 / std
                    x_mu = x_tile - u
                    dY_w = gY_tile * weight

                    dVar = (dY_w * x_mu * -0.5 * torch.pow(std, -3)).sum(-1, keepdim=True)
                    dMean = (dY_w * -inv_std).sum(-1, keepdim=True) + dVar * (-2.0 / N) * x_mu.sum(-1, keepdim=True)
                    dX = dY_w * inv_std + dVar * (2.0 / N) * x_mu + dMean * (1.0 / N)

                    d_weight += (gY_tile * x_mu * inv_std).sum((0, 1))
                    d_bias += gY_tile.sum((0, 1))

                    if grad_X_gpu is not None: grad_X_gpu[sl].copy_(dX)
                    if ctx.return_cpu_grad: grad_X_cpu[sl].copy_(dX, non_blocking=True)

                seq_start = seq_end
            batch_start = batch_end
            stream_idx += 1

        if torch.cuda.is_available():
            for s in cc.cuda_streams: s.synchronize()
            torch.cuda.synchronize()

        # 2. DEPOSIT GRADIENT FOR PREVIOUS LAYER
        if ctx.prev_bwd_accelerator is not None and grad_X_gpu is not None:
            ctx.prev_bwd_accelerator.append(grad_X_gpu)

        return (grad_X_cpu if ctx.return_cpu_grad else grad_X_gpu), d_weight, d_bias, None, None, None, None, None, None


class BigLayerNorm(nn.Module):
    def __init__(self, dim, device='cuda', dtype=None, out_on_gpu=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device))
        self.bias = nn.Parameter(torch.zeros(dim, device=device))
        self.out_on_gpu = out_on_gpu
        self._dtype = dtype if dtype else torch.float32
        self.output = None
        self.X_grad = None

    def _get_output_tensor(self, B, L, D):
        if self.output is None or self.output.shape[:2] != (B, L):
            self.output = torch.empty((B, L, D), device='cpu', pin_memory=True, dtype=self._dtype)
        return self.output

    def _get_input_grad_tensor(self, X):
        if self.X_grad is None or self.X_grad.shape != X.shape:
            self.X_grad = torch.zeros(X.shape, device='cpu', pin_memory=True, dtype=self._dtype)
        return self.X_grad

    def forward(self, x_in):
        B, L, D = x_in.shape
        out = self._get_output_tensor(B, L, D)
        x_grad = self._get_input_grad_tensor(x_in) if x_in.requires_grad else None

        gpu_cache = [] if self.out_on_gpu else None
        bwd_accelerator = []  # Bucket for next layer to drop grads into

        out_cpu = BigLayerNormFunc.apply(
            x_in, self.weight, self.bias,
            out, x_grad, gpu_cache, bwd_accelerator,  # Side channels
            6e7, self.out_on_gpu
        )

        # Attach side channels to output
        if self.out_on_gpu and gpu_cache:
            out_cpu._big_gpu_cache = gpu_cache[0]

        # Attach the backward bucket so next layer can find it
        out_cpu._bwd_accelerator = bwd_accelerator

        return out_cpu


# ==============================================================================
# 2. BigFusedSelfAttention (With 2-Way Hot Potato)
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
        D_out = D

        max_elements = int(max_elements)
        tokens_per_sample = L * D
        max_batch_size = max(1, min(B, int(max_elements // tokens_per_sample)))
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

                # Norm + QKV
                u = x_chunk.mean(-1, keepdim=True)
                s = x_chunk.var(-1, unbiased=False, keepdim=True)
                x_norm = (x_chunk - u) * torch.rsqrt(s + 1e-6) * norm_w + norm_b

                QKV = x_norm.matmul(W_qkv) + b_qkv.view(1, 1, 3 * Hd)
                QKV = QKV.view(b_s, L, 3, H, d)
                Q_all, K_all, V_all = QKV[:, :, 0], QKV[:, :, 1], QKV[:, :, 2]

                block = min(ctx.tiling_params[0], max_patch_len, L)
                for i0 in range(0, L, block):
                    i1 = min(L, i0 + block)
                    Qb = Q_all[:, i0:i1]
                    m = torch.full((b_s, H, i1 - i0), -float('inf'), device=device)
                    l = torch.zeros((b_s, H, i1 - i0), device=device)
                    acc = torch.zeros((b_s, H, i1 - i0, D_out), device=device)

                    for j0 in range(0, L, block):
                        j1 = min(L, j0 + block)
                        S = torch.einsum('bqhd,bkhd->bhqk', Qb, K_all[:, j0:j1]) / math.sqrt(d)
                        m_new = torch.maximum(m, S.amax(3))
                        exp_diff = (S - m_new.unsqueeze(-1)).exp()
                        l_new = (m - m_new).exp() * l + exp_diff.sum(3)
                        Vb_proj = torch.einsum('bkhd,hde->bkhe', V_all[:, j0:j1], Wo_slices)
                        acc = (m - m_new).exp().unsqueeze(-1) * acc + torch.einsum('bhqk,bkhe->bhqe',
                                                                                   exp_diff.to(Vb_proj.dtype), Vb_proj)
                        m = m_new
                        l = l_new

                    M_vec[batch_start:batch_end, :, i0:i1] = m
                    L_vec[batch_start:batch_end, :, i0:i1] = l

                    out_block = (acc / (l.unsqueeze(-1) + 1e-6)).sum(1) + b_o
                    if ls_gamma is not None: out_block = out_block * ls_gamma
                    out_block = out_block + x_chunk[:, i0:i1]  # Residual

                    if out_on_gpu: gpu_out[batch_start:batch_end, i0:i1].copy_(out_block)
                    output[batch_start:batch_end, i0:i1].copy_(out_block, non_blocking=True)

            batch_start = batch_end
            stream_idx += 1

        if torch.cuda.is_available():
            for s in cc.cuda_streams:
                s.synchronize()
            if not out_on_gpu:
                torch.cuda.synchronize()

        saved_X = X_in._big_cpu_out if hasattr(X_in, '_big_cpu_out') else X_in
        ctx.save_for_backward(saved_X, norm_w, norm_b, W_qkv, M_vec, L_vec, output, W_o, b_o, ls_gamma)

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
        use_gpu_path = (grad_out_gpu is not None)

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

        max_block, max_elems = ctx.tiling_params
        max_batch_size = max(1, min(B, int(max_elems // (L * D))))
        max_patch_len = max(1, int(max_elems // (max_batch_size * D * 4)))
        block = min(max_block, max_patch_len, L)
        inv_sqrt_d = 1.0 / math.sqrt(d)

        batch_start = 0
        stream_idx = 0
        while batch_start < B:
            batch_end = min(batch_start + max_batch_size, B)
            b_s = batch_end - batch_start

            with maybe_cuda_stream(cc.cuda_streams, stream_idx):
                x_chunk = X_in[batch_start:batch_end].to(device, non_blocking=True)

                # Load Grad (GPU or CPU)
                if use_gpu_path:
                    gY_chunk = grad_out_gpu[batch_start:batch_end]
                else:
                    gY_chunk = grad_Y_cpu[batch_start:batch_end].to(device, non_blocking=True)

                # Residual Grad
                dX_chunk = gY_chunk.clone()

                # Recompute Forward
                u = x_chunk.mean(-1, keepdim=True)
                s = x_chunk.var(-1, unbiased=False, keepdim=True)
                x_norm = (x_chunk - u) * torch.rsqrt(s + 1e-6) * norm_w + norm_b
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
                        S = torch.einsum('bqhd,bkhd->bhqk', Qb, K[:, j0:j1]) * inv_sqrt_d
                        P = (S - m_g.unsqueeze(-1)).exp() / (l_g.unsqueeze(-1) + 1e-12)
                        O_unproj += torch.einsum('bhqk,bkhd->bqhd', P.to(Qb.dtype), V[:, j0:j1])

                    if ls_gamma is not None:
                        Y_unscaled = O_unproj.view(b_s, i1 - i0, Hd).matmul(W_o) + b_o
                        d_gamma += (gYb * Y_unscaled).sum((0, 1))
                        gY_eff = gYb * ls_gamma
                    else:
                        gY_eff = gYb

                    d_bo += gY_eff.sum((0, 1))
                    gO = torch.einsum('bqe,hde->bqhd', gY_eff, Wo_slices)
                    D_vec = torch.sum(gO * O_unproj, -1).permute(0, 2, 1)

                    for j0 in range(0, L, block):
                        j1 = min(L, j0 + block)
                        S = torch.einsum('bqhd,bkhd->bhqk', Qb, K[:, j0:j1]) * inv_sqrt_d
                        P = (S - m_g.unsqueeze(-1)).exp() / (l_g.unsqueeze(-1) + 1e-12)
                        dV[:, j0:j1].add_(torch.einsum('bhqk,bqhd->bkhd', P.to(Qb.dtype), gO))
                        d_Wo += torch.einsum('bkhd,bkhe->hde', V[:, j0:j1],
                                             torch.einsum('bhqk,bqe->bkhe', P.to(Qb.dtype), gY_eff)).reshape(Hd, D_out)
                        dS = P * (torch.einsum('bqhd,bkhd->bhqk', gO, V[:, j0:j1]) - D_vec.unsqueeze(-1))
                        dQ[:, i0:i1].add_(inv_sqrt_d * torch.einsum('bhqk,bkhd->bqhd', dS, K[:, j0:j1]))
                        dK[:, j0:j1].add_(inv_sqrt_d * torch.einsum('bhqk,bqhd->bkhd', dS, Qb))

                dQ_flat, dK_flat, dV_flat = dQ.reshape(b_s, L, Hd), dK.reshape(b_s, L, Hd), dV.reshape(b_s, L, Hd)
                d_bqkv[0:Hd] += dQ_flat.sum((0, 1))
                d_bqkv[Hd:2 * Hd] += dK_flat.sum((0, 1))
                d_bqkv[2 * Hd:3 * Hd] += dV_flat.sum((0, 1))
                d_Wqkv[:, 0:Hd] += torch.einsum('bld,blh->dh', x_norm, dQ_flat)
                d_Wqkv[:, Hd:2 * Hd] += torch.einsum('bld,blh->dh', x_norm, dK_flat)
                d_Wqkv[:, 2 * Hd:3 * Hd] += torch.einsum('bld,blh->dh', x_norm, dV_flat)

                dX_norm = dQ_flat @ W_qkv[:, 0:Hd].t() + dK_flat @ W_qkv[:, Hd:2 * Hd].t() + dV_flat @ W_qkv[:,
                        2 * Hd:3 * Hd].t()
                dX_chunk += manual_layernorm_backward(dX_norm, x_chunk, norm_w, norm_b)

                # Norm Grads
                inv_std = torch.rsqrt(x_chunk.var(-1, unbiased=False, keepdim=True) + 1e-6)
                d_norm_w += (dX_norm * (x_chunk - x_chunk.mean(-1, keepdim=True)) * inv_std).sum((0, 1))
                d_norm_b += dX_norm.sum((0, 1))

                # Store Input Grad
                if grad_X_gpu is not None:
                    grad_X_gpu[batch_start:batch_end].copy_(dX_chunk)
                if ctx.return_cpu_grad:
                    grad_X_cpu[batch_start:batch_end].copy_(dX_chunk, non_blocking=True)

            batch_start = batch_end
            stream_idx += 1

        if torch.cuda.is_available():
            for s in cc.cuda_streams: s.synchronize()
            torch.cuda.synchronize()

        # 2. DEPOSIT GRADIENT FOR PREVIOUS LAYER
        if ctx.prev_bwd_accelerator is not None and grad_X_gpu is not None:
            ctx.prev_bwd_accelerator.append(grad_X_gpu)

        return (grad_X_cpu if ctx.return_cpu_grad else grad_X_gpu,
                d_norm_w, d_norm_b, d_Wqkv, d_bqkv, d_Wo, d_bo, d_gamma, *([None] * 9))


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

    def _get_output_tensor(self, B, L):
        if self.output is None or self.output.shape[:2] != (B, L):
            self.output = torch.empty((max(B, 1), max(L, 1024), self.d_model), device='cpu', pin_memory=True,
                                      dtype=self._dtype)
        return self.output[:B, :L]

    def _get_input_grad_tensor(self, X):
        if self.X_grad is None or self.X_grad.shape[0] < X.shape[0] or self.X_grad.shape[1] < X.shape[1]:
            self.X_grad = torch.zeros((max(X.shape[0], 1), max(X.shape[1], 1024), self.d_model), device='cpu',
                                      pin_memory=True, dtype=self._dtype)
        return self.X_grad[:X.shape[0], :X.shape[1]]

    def forward(self, x_cpu):
        out = self._get_output_tensor(*x_cpu.shape[:2])
        x_grad = self._get_input_grad_tensor(x_cpu) if x_cpu.requires_grad else None

        gpu_cache = [] if self.out_on_gpu else None
        bwd_accelerator = []  # Bucket for next layer

        out_cpu = BigFusedSelfAttentionFunc.apply(
            x_cpu, self.norm_weight, self.norm_bias, self.W_qkv, self.b_qkv, self.W_o, self.b_o, self.gamma,
            self.num_heads, self.head_dim, out, x_grad,
            gpu_cache, bwd_accelerator,  # Channels
            2048, 6e5, self.out_on_gpu
        )

        if self.out_on_gpu and gpu_cache:
            out_cpu._big_gpu_cache = gpu_cache[0]
        out_cpu._bwd_accelerator = bwd_accelerator
        return out_cpu


# ==============================================================================
# 3. Layer 2: BigFusedGluMLP
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
        expansion = (fc1_w.shape[0] / D) + 2
        tokens_per_tile = int(max_elements / (D * expansion))
        max_batch = max(1, min(B, int(tokens_per_tile // L)))
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

                    if X_gpu_cache is not None:
                        x_tile = X_gpu_cache[sl]
                    else:
                        x_tile = X_in[sl].to(device, non_blocking=True)

                    u = x_tile.mean(-1, keepdim=True)
                    s = x_tile.var(-1, unbiased=False, keepdim=True)
                    x_norm = (x_tile - u) * torch.rsqrt(s + 1e-6) * norm_w + norm_b
                    mlp_out = swiglu_forward(x_norm, fc1_w, fc1_b, fc2_w, fc2_b)
                    if ls_gamma is not None: mlp_out = mlp_out * ls_gamma
                    res_out = mlp_out + x_tile

                    if out_on_gpu: gpu_out[sl].copy_(res_out)
                    output[sl].copy_(res_out, non_blocking=True)
                    seq_start = seq_end
            batch_start = batch_end
            stream_idx += 1

        if torch.cuda.is_available():
            for s in cc.cuda_streams:
                s.synchronize()
            if not out_on_gpu:
                torch.cuda.synchronize()

        saved_X = X_in._big_cpu_out if hasattr(X_in, '_big_cpu_out') else X_in
        ctx.save_for_backward(saved_X, norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b, ls_gamma)
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

                    with torch.enable_grad():
                        # Simple LN for backward calc
                        u = x_tile.mean(-1, keepdim=True)
                        s = x_tile.var(-1, unbiased=False, keepdim=True)
                        x_norm = (x_tile - u) * torch.rsqrt(s + 1e-6) * norm_w + norm_b

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
                        if ctx.return_cpu_grad:
                            grad_X_cpu[sl].copy_(x_tile.grad, non_blocking=True)

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
            for s in cc.cuda_streams:
                s.synchronize()
            torch.cuda.synchronize()

        if ctx.prev_bwd_accelerator is not None and grad_X_gpu is not None:
            ctx.prev_bwd_accelerator.append(grad_X_gpu)

        return (grad_X_cpu if ctx.return_cpu_grad else grad_X_gpu,
                d_norm_w, d_norm_b, d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_gamma, *([None] * 6))


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

    def _get_output_tensor(self, B, L, D):
        if self.output is None or self.output.shape[0] < B or self.output.shape[1] < L:
            self.output = torch.empty((max(B, 1), max(L, 1024), D), device='cpu', pin_memory=True, dtype=self._dtype)
        return self.output[:B, :L]

    def _get_input_grad_tensor(self, X):
        if self.X_grad is None or self.X_grad.shape[0] < X.shape[0] or self.X_grad.shape[1] < X.shape[1]:
            self.X_grad = torch.zeros((max(X.shape[0], 1), max(X.shape[1], 1024), X.shape[2]), device='cpu',
                                      pin_memory=True, dtype=self._dtype)
        return self.X_grad[:X.shape[0], :X.shape[1]]

    def forward(self, x_in):
        out = self._get_output_tensor(*x_in.shape[:2])
        x_grad = self._get_input_grad_tensor(x_in) if x_in.requires_grad else None

        gpu_cache = [] if self.out_on_gpu else None
        bwd_accelerator = []

        out_cpu = BigFusedGluMLPFunc.apply(
            x_in, self.norm_weight, self.norm_bias,
            self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias, self.gamma,
            out, x_grad, gpu_cache, bwd_accelerator, 6e5, self.out_on_gpu
        )
        if self.out_on_gpu and gpu_cache:
            out_cpu._big_gpu_cache = gpu_cache[0]
        out_cpu._bwd_accelerator = bwd_accelerator
        return out_cpu
