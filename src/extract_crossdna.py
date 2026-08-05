#!/usr/bin/env python
"""
extract_crossdna.py — extract frozen CrossDNA embeddings for every window.

The backbone is never updated: this is the zero-shot representation. Output is
the canonical .npz (ids + X + meta) consumed by everything downstream.

NUMERICAL PRECISION IS NOT OPTIONAL HERE. CrossDNA overflows under bf16 on some
checkpoints, and the failure is silent in the sense that it produces non-finite
values rather than an exception. Three independent controls exist:

  --sdp default   use PyTorch's default SDP kernel selection. The `math`
                  implementation produced ~15% non-finite values on the 519M
                  checkpoint, so `default` is what the reported results use,
                  even though `math` sounds like the safer choice.
  --amp-fp32      override the model's internal preferred_amp_dtype so the
                  attention runs in full precision. Required for the
                  from-scratch checkpoints.
  --clamp         last resort: zero out non-finite values instead of letting
                  them propagate, and COUNT them so the damage is visible in
                  the log rather than hidden in the embeddings.

Always read the non-finite count printed at the end. A run with a non-zero
count is not a valid run; it is a diagnostic.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

# Must be set before torch is imported.
os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import save_embeddings, set_seed

# runtime patches live in env/patches/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "env" / "patches"))

BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
CLAMP = {"nonfinite": 0, "checked": 0, "calls": 0}
# Per-tensor accounting. An aggregate percentage cannot distinguish scattered
# rounding noise from one specific intermediate dying completely and always —
# and those call for very different statements about the results.
CLAMP_BY_NAME = {}


def force_fp32_amp(model):
    """Override preferred_amp_dtype() -> float32 so attention runs in full precision."""
    import torch

    mod = sys.modules.get(type(model).__module__)
    if mod is None or not hasattr(mod, "preferred_amp_dtype"):
        print("  [amp-fp32] preferred_amp_dtype not found")
        return
    mod.preferred_amp_dtype = lambda: torch.float32
    print("  [amp-fp32] preferred_amp_dtype -> float32 (prevents attention overflow)")


def install_clamp(model):
    """Replace ensure_finite so non-finite values are zeroed in place and counted.

    This is a fallback, not a fix. The count printed at the end is the point:
    it turns a silent corruption into a visible number.
    """
    import torch

    mod = sys.modules.get(type(model).__module__)
    if mod is None or not hasattr(mod, "ensure_finite"):
        print("  [clamp] ensure_finite not found")
        return

    def ensure_finite_clamp(x, name=""):
        try:
            nb = int((~torch.isfinite(x)).sum())
            CLAMP["calls"] += 1
            CLAMP["checked"] += x.numel()
            rec = CLAMP_BY_NAME.setdefault(
                name or "?", {"calls": 0, "hits": 0, "nonfinite": 0,
                              "checked": 0, "full": 0, "at": []})
            # Ordinal within this name. ensure_finite carries no layer index, so
            # the call order is the only handle on WHICH layer failed — and
            # "one attention layer out of 24 is dead" is a very different
            # statement from "1.3% of values are noisy".
            ordinal = rec["calls"]
            rec["calls"] += 1
            rec["checked"] += x.numel()
            if nb:
                CLAMP["nonfinite"] += nb
                rec["hits"] += 1
                rec["nonfinite"] += nb
                if nb == x.numel():
                    rec["full"] += 1   # the ENTIRE tensor was non-finite
                if len(rec["at"]) < 40:
                    rec["at"].append(ordinal)
                x.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
        return x

    mod.ensure_finite = ensure_finite_clamp
    print("  [clamp] ensure_finite replaced (zeroes and counts non-finite values)")


def encode(seqs, device):
    import torch

    L = len(seqs[0])
    arr = np.empty((len(seqs), L), dtype=np.int64)
    for r, s in enumerate(seqs):
        arr[r] = [BASE.get(c, 4) for c in s]
    return torch.from_numpy(arr).to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True,
                    help="HuggingFace-format weights directory")
    ap.add_argument("--windows", required=True, help="windows.parquet")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--pooling", default="mean", choices=["mean", "max"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N windows (smoke test)")
    ap.add_argument("--sdp", default="math", choices=["math", "default"],
                    help="SDP kernel selection; use 'default' for the 519M checkpoint")
    ap.add_argument("--sdpa-patch", action="store_true",
                    help="replace flash_attn_func with the fp32 SDPA equivalent")
    ap.add_argument("--amp-fp32", action="store_true",
                    help="force internal attention to fp32 (fixes 519M overflow)")
    ap.add_argument("--clamp", action="store_true",
                    help="zero non-finite values instead of propagating them")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    import torch

    # fla applies @torch.compile at import time and torch._inductor is broken
    # with triton 3.2.0, so DISABLE_TORCH_COMPILE alone is not enough: the
    # decorator has to be neutralised BEFORE transformers/fla are imported.
    if hasattr(torch, "compile"):
        torch.compile = (lambda fn=None, *a, **k: (fn if fn is not None else (lambda g: g)))
    if args.sdpa_patch:
        import sdpa_fp32
        sdpa_fp32.install()
    from transformers import AutoModelForMaskedLM

    if args.sdp == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    df = pd.read_parquet(args.windows, columns=["id", "seq"])
    if args.limit:
        df = df.iloc[: args.limit]
    ids = df["id"].tolist()
    seqs = df["seq"].tolist()
    L = len(seqs[0])
    print(f"[CrossDNA] windows={len(seqs)}  L={L}  batch={args.batch_size}  "
          f"pooling={args.pooling}  sdp={args.sdp}  amp_fp32={args.amp_fp32}  "
          f"clamp={args.clamp}")

    print(f"Loading weights from: {args.weights}")
    model = AutoModelForMaskedLM.from_pretrained(
        args.weights, trust_remote_code=True, local_files_only=True
    ).float().to(args.device).eval()
    model.backbone.pretrain = False
    model.backbone.for_representation = True
    if args.amp_fp32:
        force_fp32_amp(model)
    if args.clamp:
        install_clamp(model)

    feats, batch_times = [], []
    t0 = time.time()
    with torch.inference_mode():
        for i in tqdm(range(0, len(seqs), args.batch_size)):
            bt = time.time()
            batch = seqs[i: i + args.batch_size]
            x = encode(batch, args.device)
            emb, _ = model.backbone(x)
            emb = emb.float()
            vec = emb.max(dim=1).values if args.pooling == "max" else emb.mean(dim=1)
            feats.append(vec.detach().cpu().numpy())
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            batch_times.append(time.time() - bt)
    dt = time.time() - t0

    X = np.concatenate(feats, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = dict(model="crossdna", weights=args.weights, pooling=args.pooling,
                dtype="fp32", sdp=args.sdp, amp_fp32=bool(args.amp_fp32),
                sdpa_patch=bool(args.sdpa_patch), clamp=bool(args.clamp),
                dim=int(X.shape[1]), n=int(X.shape[0]))
    if args.clamp:
        # The COUNT belongs in the file, not only in the log. Recording that
        # clamping was enabled says nothing about whether it fired; without the
        # number, an .npz produced with a substantial fraction of its
        # intermediates zeroed is indistinguishable from a clean one, months
        # later, when the log is gone.
        meta["clamp_nonfinite"] = int(CLAMP["nonfinite"])
        meta["clamp_checked"] = int(CLAMP["checked"])
        meta["clamp_fraction"] = (CLAMP["nonfinite"] / CLAMP["checked"]
                                  if CLAMP["checked"] else 0.0)
    save_embeddings(args.out, ids, X, meta)

    # ---------------------------------------------------------------- #
    # Diagnostics. The variance check is the cheap guard against the
    # worst failure mode: a model that returns a constant vector still
    # produces a well-shaped .npz and plausible-looking downstream runs.
    # ---------------------------------------------------------------- #
    steady = statistics.median(batch_times[1:]) if len(batch_times) > 1 else batch_times[0]
    wps = args.batch_size / steady if steady > 0 else 0.0
    var = float(np.mean(np.var(X, axis=0)))
    print(f"\nWrote {args.out}  shape={X.shape}  ({dt/60:.1f} min)")
    print(f"first batch: {batch_times[0]:.1f}s (warmup)  |  steady state: "
          f"{steady:.2f}s/batch = {wps:.1f} windows/s")
    if args.device.startswith("cuda"):
        print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"mean embedding variance: {var:.4e} "
          f"({'ok' if var > 1e-6 else 'SUSPICIOUS, near zero'})")
    if args.clamp:
        frac = CLAMP["nonfinite"] / max(CLAMP["checked"], 1)
        print(f"[clamp] non-finite values zeroed: {CLAMP['nonfinite']:,} of "
              f"{CLAMP['checked']:,} ({frac:.2e})")
        hit = {k: v for k, v in CLAMP_BY_NAME.items() if v["hits"]}
        if hit:
            print(f"[clamp] {len(hit)} of {len(CLAMP_BY_NAME)} guarded tensors "
                  "were affected:")
            print(f"        {'tensor':28} {'calls':>7} {'hit':>7} {'fully NaN':>10} "
                  f"{'fraction':>10}")
            for k, v in sorted(hit.items(), key=lambda kv: -kv[1]["nonfinite"]):
                f = v["nonfinite"] / max(v["checked"], 1)
                print(f"        {k[:28]:28} {v['calls']:7d} {v['hits']:7d} "
                      f"{v['full']:10d} {f:10.2e}")
            for k, v in sorted(hit.items()):
                if v["at"]:
                    print(f"        {k}: failed at call ordinal(s) "
                          f"{v['at'][:20]} of {v['calls']} per forward")
            full = [k for k, v in hit.items() if v["full"] == v["calls"]]
            allfull = [k for k, v in hit.items() if v["full"] == v["hits"]]
            if allfull:
                print(f"        NOTE: {allfull} was non-finite in ITS ENTIRETY "
                      "whenever it was hit.")
                print("        A whole intermediate replaced by zeros is a "
                      "structural failure of one")
                print("        layer, not rounding noise, and any result from "
                      "this run has to say so.")
            if full:
                print(f"        NOTE: {full} failed on EVERY call.")
                print("        That is a structural failure of one intermediate, "
                      "not rounding noise,")
                print("        and any result from this run has to say so.")
    if args.limit and wps > 0:
        print(f"\nProjected time for 120,000 windows: ~{120000/wps/60:.1f} min "
              f"({120000/wps/3600:.2f} h)")


if __name__ == "__main__":
    main()
