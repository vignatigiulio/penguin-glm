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

install(force_dtype=None) installs the SAME replacement without promoting the
precision, so attention runs through SDPA at the input dtype. That mode exists
only for diagnosis, and it OVERFLOWS on checkpoints flash-attn handles cleanly —
which is the point. FlashAttention is not a bf16 kernel: it takes bf16 in and
returns bf16, but accumulates the softmax statistics and the output in fp32
inside the kernel. The fp32 promotion here is therefore not a precaution, it is
what makes the replacement equivalent. Production runs use the default, fp32.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Bumped on every change to the masking or precision logic, and printed by
# install(). Without it there is no way to tell from a log which copy of this
# file a remote machine actually loaded.
__revision__ = 4


# Working precision of the replacement. fp32 is the DEFAULT, not merely what
# install() happens to set: calling flash_attn_func_fp32 directly without
# install() must not silently downgrade to the input dtype.
_FORCE_DTYPE = torch.float32


def flash_attn_func_fp32(q, k, v, causal=False, window_size=(-1, -1),
                         softmax_scale=None, **kwargs):
    """Drop-in for flash_attn_func, encoder case with no padding or cache.

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

    # -> [B, H, T, D] for SDPA, at the working precision
    wd = _FORCE_DTYPE if _FORCE_DTYPE is not None else q.dtype
    qh = q.permute(0, 2, 1, 3).to(wd)
    kh = k.permute(0, 2, 1, 3).to(wd)
    vh = v.permute(0, 2, 1, 3).to(wd)

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

    # THE MASK MUST BE BOOLEAN, NOT AN ADDITIVE FLOAT MASK.
    # A float mask in bf16 makes scaled_dot_product_attention return NaN — on
    # every backend, and with -inf and finfo.min alike. Measured on H100 with
    # T=1000 and window (511, 0): 36% of the output non-finite, confined to
    # query rows 639..999, independent of the input magnitude. In fp32 the float
    # mask is fine, which is why the failure only appears when the surrounding
    # model runs in bf16. A boolean mask (True = participate) works in both
    # precisions on every backend, and lets PyTorch choose its own sentinel.
    # tools/test_sdpa_mask.py reproduces the whole table in a few seconds.
    attn_mask = None if bool(allowed.all()) else allowed[None, None]

    # AUTOCAST MUST BE DISABLED AROUND THIS CALL.
    # scaled_dot_product_attention is on PyTorch's autocast list, so inside an
    # autocast(bf16) region it silently casts the fp32 tensors above straight
    # back down to bf16 — undoing the whole point of the replacement and
    # reproducing the overflow it exists to prevent. Without this guard the
    # patch only works when the surrounding model is already in fp32, which
    # hides the failure in exactly the configuration used in production.
    with torch.autocast(device_type=q.device.type, enabled=False):
        out = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=attn_mask, scale=scale)
    # back to [B, T, H, D], in q's original dtype so nothing else is disturbed
    return out.permute(0, 2, 1, 3).to(q.dtype)


_ORIG = {}


def install(force_dtype="fp32"):
    """Replace flash_attn_func in the fla.layers.attn namespace.

    force_dtype: "fp32" (default, production) or None (diagnostic: keep the
    input dtype, so only the implementation changes).

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

    global _FORCE_DTYPE
    _FORCE_DTYPE = torch.float32 if force_dtype == "fp32" else None

    if "flash_attn_func" not in _ORIG:
        _ORIG["flash_attn_func"] = getattr(A, "flash_attn_func", None)
    A.flash_attn_func = flash_attn_func_fp32
    label = "fp32" if _FORCE_DTYPE is not None else "same-dtype (diagnostic)"
    print(f"  [sdpa-fp32 rev{__revision__}] flash_attn_func -> SDPA replacement, "
          f"{label}")


def uninstall():
    import fla.layers.attn as A

    if _ORIG.get("flash_attn_func") is not None:
        A.flash_attn_func = _ORIG["flash_attn_func"]
        print("  [sdpa-fp32] flash_attn_func restored to the original")
