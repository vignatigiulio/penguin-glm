#!/usr/bin/env python
"""
check_weights.py — inspect a checkpoint's parameters directly, without building
the model.

Written to answer one question: when a specific layer produces a wholly
non-finite activation on every input, are its WEIGHTS already broken?

Three ways a released checkpoint can carry the fault:

  1. non-finite parameters — NaN or Inf stored in the file. Decisive.
  2. finite but enormous parameters — a weight of 1e30 overflows the moment it
     meets an activation, and looks perfectly normal in a summary statistic
     taken over the whole model.
  3. dead parameters — an all-zero tensor, which produces a zero activation
     rather than a non-finite one, but is equally a broken layer.

Reports per tensor and flags outliers against the distribution of the rest, so
a single bad layer stands out instead of being averaged away.

Usage:
  python check_weights.py --weights $CROSSDNA_REPO/crossdna_inference
  python check_weights.py --weights .../519M --filter branchB
  python check_weights.py --weights .../519M --top 30
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def load_tensors(path):
    """Yield (name, tensor) from a HF directory or a single checkpoint file."""
    import torch

    if os.path.isdir(path):
        st = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if st:
            from safetensors.torch import load_file
            for f in st:
                for k, v in load_file(f).items():
                    yield k, v
            return
        bins = sorted(glob.glob(os.path.join(path, "*.bin")))
        if not bins:
            raise SystemExit(f"No .safetensors or .bin found in {path}")
        for f in bins:
            sd = torch.load(f, map_location="cpu")
            sd = sd.get("state_dict", sd)
            for k, v in sd.items():
                if hasattr(v, "dtype"):
                    yield k, v
        return

    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    for k, v in sd.items():
        if hasattr(v, "dtype"):
            yield k, v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True,
                    help="HF directory or checkpoint file")
    ap.add_argument("--filter", default="", help="only tensors matching this substring")
    ap.add_argument("--top", type=int, default=15,
                    help="how many extreme tensors to list")
    args = ap.parse_args()

    import torch

    rows = []
    for name, t in load_tensors(args.weights):
        if args.filter and args.filter not in name:
            continue
        if not torch.is_floating_point(t):
            continue
        f = t.float()
        finite = torch.isfinite(f)
        n_bad = int((~finite).sum())
        vals = f[finite]
        rows.append(dict(
            name=name,
            shape=tuple(t.shape),
            dtype=str(t.dtype).replace("torch.", ""),
            n=int(t.numel()),
            bad=n_bad,
            absmax=float(vals.abs().max()) if vals.numel() else float("nan"),
            std=float(vals.std()) if vals.numel() > 1 else float("nan"),
            zeros=int((f == 0).sum()),
        ))

    if not rows:
        raise SystemExit("No floating-point tensor matched.")

    total = sum(r["n"] for r in rows)
    bad_rows = [r for r in rows if r["bad"]]
    print(f"{len(rows)} floating-point tensors, {total:,} parameters\n")

    # ---- 1. non-finite parameters ----
    if bad_rows:
        print("*** NON-FINITE PARAMETERS IN THE CHECKPOINT ***")
        for r in sorted(bad_rows, key=lambda r: -r["bad"]):
            print(f"  {r['name']:60} {str(r['shape']):22} "
                  f"{r['bad']:,} of {r['n']:,}")
        print("  A checkpoint containing NaN or Inf is broken as distributed; "
              "no runtime setting can repair it.\n")
    else:
        print("[ok] no non-finite parameter\n")

    # ---- 2. magnitude outliers ----
    # Judged against the distribution of the other tensors, because what counts
    # as "large" depends on the model, not on an absolute number.
    amax = np.array([r["absmax"] for r in rows])
    med = float(np.median(amax))
    print(f"|w|max across tensors: median={med:.3e}  "
          f"max={float(amax.max()):.3e}")
    print(f"\nTop {args.top} tensors by |w|max:")
    print(f"  {'tensor':60} {'|w|max':>11} {'x median':>9} {'std':>11}")
    for r in sorted(rows, key=lambda r: -r["absmax"])[:args.top]:
        ratio = r["absmax"] / med if med else float("nan")
        flag = "  <-- OUTLIER" if ratio > 50 else ""
        print(f"  {r['name'][:60]:60} {r['absmax']:11.3e} {ratio:9.1f} "
              f"{r['std']:11.3e}{flag}")

    # ---- 3. dead tensors ----
    dead = [r for r in rows if r["zeros"] == r["n"]]
    if dead:
        print(f"\n{len(dead)} tensors are entirely zero:")
        for r in dead[:20]:
            print(f"  {r['name']:60} {str(r['shape']):22}")
        print("  An all-zero parameter is a dead layer: it produces no "
              "non-finite value, and no signal either.")
    else:
        print("\n[ok] no all-zero tensor")


if __name__ == "__main__":
    main()
