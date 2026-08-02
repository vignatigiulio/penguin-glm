"""
sdpa_fp32.py — drop-in fp32 replacement for flash_attn_func.

WHY THIS EXISTS
`fla.layers.attn.Attention` calls flash_attn_func, which accepts fp16/bf16 ONLY.
On CrossDNA 71.6M and 519M that forces half precision, where the activations
overflow and the forward pass returns NaN/Inf. This module provides a function
with the SAME signature and semantics in fp32, built on
torch.nn.functional.scaled_dot_product_attention (math backend, which does
support fp32).

RoPE is already applied upstream by fla (`self.rotary`), so it must NOT be
touched here: this reproduces only softmax(QK^T / sqrt(d) + mask) V.

The semantics replicated are those of fla's actual call site:

    flash_attn_func(q, k, v, causal=True,
                    window_size=(w-1, 0))   # sliding window: w-1 left, 0 right

flash-attn layout: q, k, v = [batch, seqlen, heads, head_dim].

Usage:
    import sdpa_fp32; sdpa_fp32.install()   # BEFORE loading the model
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def flash_attn_func_fp32(q, k, v, causal=False, window_size=(-1, -1),
                         softmax_scale=None, **kwargs):
    """fp32 drop-in for flash_attn_func, encoder case with no padding or cache.

    q, k, v: [B, T, H, D] (flash-attn layout). Returns [B, T, H, D].
    """
    B, T, H, D = q.shape
    Hk = k.shape[2]
    # GQA/MQA: replicate KV heads if there are fewer than query heads.
    # (In this configuration num_kv == num_heads, so this is a no-op.)
    if Hk != H:
        rep = H // Hk
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)

    # -> [B, H, T, D] for SDPA, in fp32
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()

    scale = softmax_scale if softmax_scale is not None else (1.0 / (D ** 0.5))

    # additive mask [T, T]; True = ALLOWED
    left = window_size[0]   # positions to the left (w-1); -1 = unbounded
    right = window_size[1]  # positions to the right; 0 for causal
    idx = torch.arange(T, device=q.device)
    diff = idx[None, :] - idx[:, None]  # diff[i, j] = j - i
    allowed = torch.ones(T, T, dtype=torch.bool, device=q.device)
    if causal:
        allowed &= (diff <= 0)      # j <= i, no future
    if right is not None and right >= 0:
        allowed &= (diff <= right)
    if left is not None and left >= 0:
        allowed &= (diff >= -left)  # j >= i - left, the left window
    attn_mask = torch.zeros(T, T, dtype=torch.float32, device=q.device)
    attn_mask.masked_fill_(~allowed, float("-inf"))

    out = F.scaled_dot_product_attention(
        qh, kh, vh, attn_mask=attn_mask[None, None], scale=scale)
    # back to [B, T, H, D], in q's original dtype so nothing else is disturbed
    return out.permute(0, 2, 1, 3).to(q.dtype)


_ORIG = {}


def install():
    """Replace flash_attn_func in the fla.layers.attn namespace.

    torch.compile is neutralised BEFORE fla is imported: the inductor shipped
    with this torch is incompatible with Triton 3.2.0 and would break the import
    outright. DISABLE_TORCH_COMPILE alone is not sufficient because
    fla/ops/attn/parallel.py applies the decorator at import time.
    """
    import os

    os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")
    import torch

    if hasattr(torch, "compile"):
        torch.compile = (lambda fn=None, *a, **k:
                         (fn if fn is not None else (lambda g: g)))
    import fla.layers.attn as A

    if "flash_attn_func" not in _ORIG:
        _ORIG["flash_attn_func"] = getattr(A, "flash_attn_func", None)
    A.flash_attn_func = flash_attn_func_fp32
    print("  [sdpa-fp32] flash_attn_func -> fp32 replacement (SDPA math)")


def uninstall():
    import fla.layers.attn as A

    if _ORIG.get("flash_attn_func") is not None:
        A.flash_attn_func = _ORIG["flash_attn_func"]
        print("  [sdpa-fp32] flash_attn_func restored to the original")
