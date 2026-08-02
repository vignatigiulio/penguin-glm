#!/usr/bin/env python
"""
check_embeddings.py — automated checks on an .npz produced by either extractor.

Intended to validate a smoke run before committing GPU time to the full 120,000
windows. Exits non-zero on failure, so it can gate a shell pipeline.

Checks:
  - the file opens and contains the 'X' key
  - shape is coherent: (n_windows, hidden_size)
  - no NaN / Inf — the failure mode CrossDNA 519M and 71.6M exhibit in bf16
  - the rows are not all identical, which would mean the model is not reacting
    to its input at all (collapsed embedding, or wrong input_ids)
  - no dead dimensions with zero variance
  - the ids are unique and aligned with the embedding rows

The collapse check matters more than it looks. A constant embedding passes every
shape and finiteness test, saves without complaint, and produces downstream
metrics that sit near chance — which is easy to misread as a weak model rather
than a broken pipeline.

Usage:
  python check_embeddings.py --npz /path/to/embeddings_144.npz --expected-dim 144
"""
import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--expected-dim", type=int, default=None,
                        help="expected hidden_size (e.g. 144 for the "
                             "144_without_midattn variant)")
    args = parser.parse_args()

    ok = True

    try:
        data = np.load(args.npz, allow_pickle=True)
    except Exception as e:
        print(f"[FAIL] could not open the file: {e}")
        sys.exit(1)

    if "X" not in data:
        print(f"[FAIL] key 'X' absent. Keys found: {list(data.keys())}")
        sys.exit(1)

    emb = data["X"]
    n, dim = emb.shape
    print(f"[info] embeddings shape: {emb.shape}")

    if args.expected_dim is not None and dim != args.expected_dim:
        print(f"[FAIL] expected hidden_size={args.expected_dim}, found={dim}")
        ok = False
    else:
        print(f"[OK] hidden_size = {dim}")

    n_nan = np.isnan(emb).sum()
    n_inf = np.isinf(emb).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"[FAIL] found {n_nan} NaN and {n_inf} Inf out of {emb.size} values")
        ok = False
    else:
        print("[OK] no NaN/Inf")

    n_unique_rows = len(np.unique(emb, axis=0))
    if n_unique_rows < max(2, n * 0.5):
        print(f"[FAIL] only {n_unique_rows}/{n} distinct rows: the model may not "
              "be reacting to its input (collapsed embedding)")
        ok = False
    else:
        print(f"[OK] {n_unique_rows}/{n} distinct rows")

    per_dim_std = emb.std(axis=0)
    n_dead_dims = (per_dim_std < 1e-6).sum()
    if n_dead_dims > 0:
        print(f"[WARNING] {n_dead_dims}/{dim} dimensions with ~zero variance "
              "(not necessarily an error, but worth watching)")
    else:
        print("[OK] no zero-variance dimensions")

    print(f"[stats] mean={emb.mean():.4f} std={emb.std():.4f} "
          f"min={emb.min():.4f} max={emb.max():.4f}")

    other_keys = [k for k in data.keys() if k != "X"]
    if other_keys:
        print(f"[info] other keys in the file: {other_keys}")
        if "ids" in data.files:
            ids = data["ids"]
            print(f"    ids: dtype={ids.dtype} n={len(ids)} sample={ids[:3]}")
            if len(ids) != n:
                print(f"[FAIL] number of ids ({len(ids)}) != number of embedding "
                      f"rows ({n})")
                ok = False
            elif len(np.unique(ids)) != len(ids):
                print(f"[FAIL] duplicate ids: {len(ids)} total, "
                      f"{len(np.unique(ids))} unique")
                ok = False
            else:
                print(f"[OK] {len(ids)} unique ids, aligned with the embeddings")
        if "meta" in data.files:
            print(f"    meta: {str(data['meta'])[:200]}")

    if ok:
        print("\n=== SMOKE TEST PASSED ===")
        sys.exit(0)
    else:
        print("\n=== SMOKE TEST FAILED: do NOT launch the full extraction "
              "until this is resolved ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
