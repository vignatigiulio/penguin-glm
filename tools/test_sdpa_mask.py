#!/usr/bin/env python
"""
test_sdpa_mask.py — isolate why the SDPA replacement produces NaN, without
loading any model.

Builds random tensors of the exact shape CrossDNA uses ([B, T, H, D] with the
sliding-window mask fla asks for) and runs scaled_dot_product_attention through
every combination of:

    mask encoding : -inf | finfo.min | boolean | none
    backend       : math | mem_efficient | flash | auto
    dtype         : fp32 | bf16

against a reference softmax computed explicitly in float64. For each it reports
the number of non-finite outputs, WHICH query rows are affected, and the error
against the reference.

This answers the question the model-level diagnostic could only pose: it is the
mask encoding, the backend, the precision, or the window arithmetic?

Usage:
    python test_sdpa_mask.py
    python test_sdpa_mask.py --seqlen 1000 --window 511
"""
from __future__ import annotations

import argparse
import itertools

import torch
import torch.nn.functional as F


def build_allowed(T, causal, left, right, device):
    idx = torch.arange(T, device=device)
    diff = idx[None, :] - idx[:, None]
    allowed = torch.ones(T, T, dtype=torch.bool, device=device)
    if causal:
        allowed &= (diff <= 0)
    if right is not None and right >= 0:
        allowed &= (diff <= right)
    if left is not None and left >= 0:
        allowed &= (diff >= -left)
    return allowed


def reference(q, k, v, allowed, scale):
    """Explicit attention in float64. Slow, but unambiguous."""
    qh = q.permute(0, 2, 1, 3).double()
    kh = k.permute(0, 2, 1, 3).double()
    vh = v.permute(0, 2, 1, 3).double()
    scores = (qh @ kh.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    w = torch.softmax(scores, dim=-1)
    return (w @ vh).permute(0, 2, 1, 3)


def make_mask(kind, allowed, dtype, device):
    if kind == "none":
        return None
    if kind == "bool":
        return allowed[None, None]
    val = float("-inf") if kind == "-inf" else torch.finfo(dtype).min
    m = torch.zeros(allowed.shape, dtype=dtype, device=device)
    m.masked_fill_(~allowed, val)
    return m[None, None]


def run(q, k, v, mask, scale, backend):
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    flags = dict(enable_flash=False, enable_mem_efficient=False, enable_math=False)
    if backend == "math":
        flags["enable_math"] = True
    elif backend == "mem_efficient":
        flags["enable_mem_efficient"] = True
    elif backend == "flash":
        flags["enable_flash"] = True
    else:
        flags = None

    if flags is not None:
        torch.backends.cuda.enable_flash_sdp(flags["enable_flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(flags["enable_mem_efficient"])
        torch.backends.cuda.enable_math_sdp(flags["enable_math"])
    else:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale)
    return out.permute(0, 2, 1, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seqlen", type=int, default=1000)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--headdim", type=int, default=16)
    ap.add_argument("--window", type=int, default=511, help="left window (w-1)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    B, T, H, D = args.batch, args.seqlen, args.heads, args.headdim
    scale = 1.0 / (D ** 0.5)
    allowed = build_allowed(T, True, args.window, 0, dev)

    print(f"shape [B,T,H,D] = {(B, T, H, D)}  scale={scale:.4f}  "
          f"causal=True window=({args.window}, 0)  device={dev}")
    per_row = allowed.sum(dim=1)
    print(f"allowed keys per query row: min={int(per_row.min())} "
          f"max={int(per_row.max())}  rows with zero keys: "
          f"{int((per_row == 0).sum())}")
    print()

    q32 = torch.randn(B, T, H, D, device=dev)
    k32 = torch.randn(B, T, H, D, device=dev)
    v32 = torch.randn(B, T, H, D, device=dev)
    ref = reference(q32, k32, v32, allowed, scale)

    hdr = (f"{'dtype':>6} {'mask':>10} {'backend':>14} {'non-finite':>12} "
           f"{'bad rows':>9} {'max err':>10}  note")
    print(hdr)
    print("-" * len(hdr))

    for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        q, k, v = q32.to(dt), k32.to(dt), v32.to(dt)
        for mask_kind, backend in itertools.product(
                ("-inf", "finfo.min", "bool", "none"),
                ("math", "mem_efficient", "flash", "auto")):
            mask = make_mask(mask_kind, allowed, dt, dev)
            note = ""
            try:
                out = run(q, k, v, mask, scale, backend)
                fin = torch.isfinite(out)
                nnf = int((~fin).sum())
                # which query rows are affected
                bad = (~fin).any(dim=-1).any(dim=-1).any(dim=0)  # over T
                nbad = int(bad.sum())
                if nnf == 0:
                    err = float((out.double() - ref).abs().max())
                    errs = f"{err:10.3e}"
                    if mask_kind == "none":
                        note = "no mask: expected to differ from the reference"
                else:
                    errs = " " * 10
                    idx = torch.nonzero(bad).flatten()
                    note = (f"rows {int(idx[0])}..{int(idx[-1])}"
                            if len(idx) else "")
            except Exception as e:
                nnf, nbad, errs = -1, -1, " " * 10
                note = f"{type(e).__name__}: {str(e)[:60]}"
            print(f"{dt_name:>6} {mask_kind:>10} {backend:>14} "
                  f"{nnf:>12} {nbad:>9} {errs}  {note}")
        print()

    print("Read the table like this:")
    print("  * a combination with 0 non-finite and a small max err is CORRECT.")
    print("  * if '-inf' fails where 'finfo.min' succeeds, the mask encoding is "
          "the bug.")
    print("  * if every masked variant fails on one backend only, the backend "
          "is the bug and must be pinned.")
    print("  * if the 'bad rows' range matches the window boundary, the window "
          "arithmetic is the bug.")


if __name__ == "__main__":
    main()
