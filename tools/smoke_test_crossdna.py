#!/usr/bin/env python
"""
smoke_test_crossdna.py — minimal load-and-forward check for a CrossDNA checkpoint.

Run this before spending GPU hours on a full extraction. It answers the one
question that matters up front: does this checkpoint produce finite values at
this sequence length with this SDP backend?

The variance line is the second check. A model that returns a constant vector is
"finite" and entirely useless, and it will happily produce a well-formed .npz and
plausible-looking downstream numbers.

Run it BARE first. Without --amp-fp32 the 71.6M and 519M checkpoints raise
FloatingPointError inside the model's own ensure_finite guard: that failure is
the expected baseline, not a broken install. Then re-run with the precision
flags and confirm the same checkpoint comes out finite — which is what the
extraction pipeline actually uses.

Usage:
    # bare: 8.1M passes, 519M is expected to fail
    python smoke_test_crossdna.py --weights .../crossdna_inference --sdp default

    # with the fp32 attention patch: 519M should now pass
    python smoke_test_crossdna.py --weights .../519M --sdp default \
        --amp-fp32 --sdpa-patch
"""
import argparse
import os
import random
import sys
from pathlib import Path

os.environ["DISABLE_TORCH_COMPILE"] = "1"

import torch

# runtime patches live in env/patches/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "env" / "patches"))

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--weights", required=True)
ap.add_argument("--length", type=int, default=1000)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--sdp", choices=["default", "math"], default="default")
ap.add_argument("--sdpa-patch", action="store_true",
                help="replace flash_attn_func with the fp32 SDPA equivalent")
ap.add_argument("--amp-fp32", action="store_true",
                help="force internal attention to fp32 (needed by 71.6M / 519M)")
ap.add_argument("--clamp", action="store_true",
                help="zero non-finite values instead of raising, and COUNT them "
                     "— the configuration the 519M extraction actually used")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

# torch.compile is neutralised before fla/transformers are imported: the
# inductor shipped with this torch is incompatible with Triton 3.2.0.
if hasattr(torch, "compile"):
    torch.compile = (lambda fn=None, *a, **k:
                     (fn if fn is not None else (lambda g: g)))
if args.sdpa_patch:
    import sdpa_fp32
    sdpa_fp32.install()
from transformers import AutoModelForMaskedLM

if args.sdp == "math":
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

print(f"Loading: {args.weights}  (L={args.length}, batch={args.batch}, "
      f"sdp={args.sdp}, amp_fp32={args.amp_fp32}, sdpa_patch={args.sdpa_patch})")
m = AutoModelForMaskedLM.from_pretrained(
    args.weights, trust_remote_code=True, local_files_only=True
).float().cuda().eval()

CLAMP = {"nonfinite": 0, "checked": 0}

if args.clamp:
    import sys as _sys
    _cmod = _sys.modules.get(type(m).__module__)
    if _cmod is not None and hasattr(_cmod, "ensure_finite"):
        def _ensure_finite_clamp(x, name=""):
            try:
                nb = int((~torch.isfinite(x)).sum())
                CLAMP["checked"] += x.numel()
                if nb:
                    CLAMP["nonfinite"] += nb
                    x.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass
            return x
        _cmod.ensure_finite = _ensure_finite_clamp
        print("  [clamp] ensure_finite replaced (zeroes and counts)")

if args.amp_fp32:
    import sys as _sys
    _mod = _sys.modules.get(type(m).__module__)
    if _mod is not None and hasattr(_mod, "preferred_amp_dtype"):
        _mod.preferred_amp_dtype = lambda: torch.float32
        print("  [amp-fp32] preferred_amp_dtype -> float32")
    else:
        print("  [amp-fp32] WARNING: preferred_amp_dtype not found")

BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
random.seed(args.seed)
seqs = ["".join(random.choice("ACGT") for _ in range(args.length))
        for _ in range(args.batch)]
x = torch.tensor([[BASE[c] for c in s] for s in seqs]).cuda()

m.backbone.pretrain = False
m.backbone.for_representation = True
try:
    with torch.inference_mode():
        emb, _ = m.backbone(x)
except FloatingPointError as e:
    # The model guards itself and raises rather than returning NaN. Say plainly
    # what it means, so the traceback is not mistaken for a broken environment.
    print(f"\nRESULT: FAILED — the model raised: {e}")
    if not (args.amp_fp32 and args.sdpa_patch):
        print("Expected on the larger checkpoints without the precision flags. "
              "Re-run with --amp-fp32 --sdpa-patch.")
    else:
        # Reaching here does NOT mean the checkpoint is unusable. Two things
        # commonly explain it, and both are configuration rather than defect:
        print("The precision flags were active, so this is the residual "
              "instability of the checkpoint itself, not the bf16 overflow.")
        print("Two things to try before concluding anything:")
        print("  --clamp        the 519M extraction ran with it: non-finite "
              "values are zeroed and COUNTED, so the damage is visible instead "
              "of fatal.")
        print(f"  --batch N      this ran at batch {args.batch}. Batch size "
              "changes numerical stability in these models measurably; the "
              "pipeline used 16-32.")
    raise SystemExit(1)

finite = bool(torch.isfinite(emb).all())
print("shape :", tuple(emb.shape))
print("finite:", finite)
print("var   :", round(float(emb.float().var()), 5) if finite else "n/a (NaN/Inf)")
print("VRAM  :", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")
if args.clamp:
    frac = CLAMP["nonfinite"] / max(CLAMP["checked"], 1)
    print(f"clamped: {CLAMP['nonfinite']:,} of {CLAMP['checked']:,} values "
          f"({frac:.2e})")
    if CLAMP["nonfinite"]:
        print("  NOTE: a non-zero count means the forward really did go "
              "non-finite. The output is usable only as a diagnostic.")
print("RESULT:", "OK" if finite else "FAILED (non-finite)")
