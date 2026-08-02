#!/usr/bin/env python
"""
smoke_test_crossdna.py — minimal load-and-forward check for a CrossDNA checkpoint.

Run this before spending GPU hours on a full extraction. It answers the one
question that matters up front: does this checkpoint produce finite values at
this sequence length with this SDP backend?

The variance line is the second check. A model that returns a constant vector is
"finite" and entirely useless, and it will happily produce a well-formed .npz and
plausible-looking downstream numbers.

Usage:
    python smoke_test_crossdna.py --weights $PENGUIN_ROOT/weights/519M --sdp default
"""
import argparse
import os
import random

os.environ["DISABLE_TORCH_COMPILE"] = "1"

import torch
from transformers import AutoModelForMaskedLM

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--weights", required=True)
ap.add_argument("--length", type=int, default=1000)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--sdp", choices=["default", "math"], default="default")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

if args.sdp == "math":
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

print(f"Loading: {args.weights}  (L={args.length}, batch={args.batch}, sdp={args.sdp})")
m = AutoModelForMaskedLM.from_pretrained(
    args.weights, trust_remote_code=True, local_files_only=True
).float().cuda().eval()

BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
random.seed(args.seed)
seqs = ["".join(random.choice("ACGT") for _ in range(args.length))
        for _ in range(args.batch)]
x = torch.tensor([[BASE[c] for c in s] for s in seqs]).cuda()

m.backbone.pretrain = False
m.backbone.for_representation = True
with torch.inference_mode():
    emb, _ = m.backbone(x)

finite = bool(torch.isfinite(emb).all())
print("shape :", tuple(emb.shape))
print("finite:", finite)
print("var   :", round(float(emb.float().var()), 5) if finite else "n/a (NaN/Inf)")
print("VRAM  :", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")
print("RESULT:", "OK" if finite else "FAILED (non-finite)")
