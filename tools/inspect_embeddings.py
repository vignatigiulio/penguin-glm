#!/usr/bin/env python3
"""
inspect_embeddings.py -- separate non-finite values from clamped values in an
embedding archive (written to check the CrossDNA-519M archives).

The two are different things: a value can be FINITE but huge, and be clamped
without ever having been NaN/Inf. This script reports them separately:

    non-finite  -> genuine NaN or Inf
    clamp       -> values zeroed or saturated by --clamp at extraction time

USAGE
    # 1. inspect one or more archives
    python inspect_embeddings.py $PENGUIN_ROOT/out/embeddings/*519*.npz

    # 2. clamp vs no-clamp comparison ON THE METRICS (not on the vectors)
    python inspect_embeddings.py emb_clamp.npz emb_noclamp.npz --compare

    # 3. as above, adding the linear probe (slower, ~1-3 min)
    python inspect_embeddings.py emb_clamp.npz emb_noclamp.npz --compare --probe

NOTE on the comparison: it reproduces the thesis protocol only approximately.
What matters is the DELTA between the two archives, not the absolute value.
"""

import argparse
import glob
import os
import sys

import numpy as np

FLOAT_KEYS = ("embedding", "emb", "x", "features", "hidden", "repr")
LABEL_KEYS = ("label", "labels", "y", "class", "classe")
GC_KEYS = ("gc", "gc_content", "gc_frac")
SPLIT_KEYS = ("split", "partition", "set")


def pick(z, candidates):
    """Return the first key that looks like one of the candidates."""
    keys = list(z.keys())
    for c in candidates:
        for k in keys:
            if k.lower() == c:
                return k
    for c in candidates:
        for k in keys:
            if c in k.lower():
                return k
    return None


def biggest_float(z):
    """The largest 2-D float array: almost always the embedding matrix."""
    best, best_size = None, -1
    for k in z.keys():
        try:
            a = z[k]
        except Exception:
            continue
        if getattr(a, "ndim", 0) == 2 and np.issubdtype(a.dtype, np.floating):
            if a.size > best_size:
                best, best_size = k, a.size
    return best


def describe(path, deep=True):
    print(f"\n{'='*92}\n{path}\n{'='*92}")
    if not os.path.isfile(path):
        print("  FILE MISSING")
        return None
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"  ERROR while loading: {e}")
        return None

    print(f"  keys: {list(z.keys())}\n")

    # -- metadata: often saved as a dict inside a 0-d array -----------------
    meta_found = False
    for k in z.keys():
        kl = k.lower()
        if any(t in kl for t in ("clamp", "meta", "config", "cfg", "info", "sdp", "amp")):
            try:
                v = z[k]
                v = v.item() if getattr(v, "ndim", 1) == 0 else v
            except Exception:
                v = "<unreadable>"
            print(f"  [META] {k} = {v}")
            meta_found = True
    if not meta_found:
        print("  [META] no recognised metadata key "
              "(search for 'clamp' in the extraction source)")

    ek = pick(z, FLOAT_KEYS) or biggest_float(z)
    if ek is None:
        print("\n  no float matrix found")
        return None

    X = z[ek]
    print(f"\n  matrix: '{ek}'  shape={X.shape}  dtype={X.dtype}")

    if not np.issubdtype(X.dtype, np.floating):
        print("  (not float: skipping the numeric statistics)")
        return dict(path=path, key=ek, X=None, z=z)

    Xf = X.astype(np.float64, copy=False)
    finite = np.isfinite(Xf)
    n_nonfinite = int((~finite).sum())
    n_nan = int(np.isnan(Xf).sum())
    n_inf = int(np.isinf(Xf).sum())
    tot = Xf.size

    print(f"\n  --- NON-FINITE VALUES ---")
    print(f"  NaN            : {n_nan}")
    print(f"  Inf            : {n_inf}")
    print(f"  total          : {n_nonfinite}  of {tot}  ({100*n_nonfinite/tot:.4f}%)")
    print(f"  rows affected  : {int((~finite).any(axis=1).sum())} of {X.shape[0]}")

    if n_nonfinite == 0:
        print("  -> no non-finite value.")
    else:
        print("  -> non-finite values present.")

    if deep and finite.any():
        vals = Xf[finite]
        n_zero = int((Xf == 0).sum())
        rows_with_zero = int((Xf == 0).any(axis=1).sum())
        print(f"\n  --- CLAMP FOOTPRINT ---")
        print(f"  exact zeros    : {n_zero}  ({100*n_zero/tot:.3f}% of the values)")
        print(f"  rows with zeros: {rows_with_zero}  ({100*rows_with_zero/X.shape[0]:.2f}%)")
        print(f"  |x| max        : {np.abs(vals).max():.4e}")
        print(f"  |x| p99.9      : {np.percentile(np.abs(vals), 99.9):.4e}")
        print(f"  mean / std     : {vals.mean():.4e} / {vals.std():.4e}")
        if 0.005 <= n_zero / tot <= 0.05:
            print("  -> exact zeros between 0.5% and 5% of the values:")
            print("     consistent with a clamp that zeroes, NOT with non-finite values.")
        elif n_zero == 0:
            print("  -> no exact zero: no trace of the clamp in this archive.")

    return dict(path=path, key=ek, X=X, z=z)


# --------------------------------------------------- comparison on the metrics ---

def metrics(X, z, probe=False, seed=0):
    """NMI (k-means, k=4) against classes and GC; optional MCC from a linear probe."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
        print("  sklearn not available: skipping the metrics")
        return {}

    out = {}
    Xc = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    Xs = StandardScaler().fit_transform(Xc)

    km = KMeans(n_clusters=4, n_init=10, random_state=seed).fit(Xs)
    lab_k = pick(z, LABEL_KEYS)
    gc_k = pick(z, GC_KEYS)

    if lab_k is not None:
        y = np.asarray(z[lab_k]).ravel()
        out["NMI_function"] = normalized_mutual_info_score(y, km.labels_)
    if gc_k is not None:
        gc = np.asarray(z[gc_k]).ravel().astype(float)
        gcb = np.digitize(gc, np.quantile(gc, [0.25, 0.5, 0.75]))
        out["NMI_GC"] = normalized_mutual_info_score(gcb, km.labels_)
    if "NMI_function" in out and out.get("NMI_GC"):
        out["ratio_function_GC"] = out["NMI_function"] / out["NMI_GC"]

    if probe and lab_k is not None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import matthews_corrcoef
        y = np.asarray(z[lab_k]).ravel()
        sp_k = pick(z, SPLIT_KEYS)
        if sp_k is not None:
            sp = np.asarray(z[sp_k]).ravel().astype(str)
            tr = np.char.startswith(np.char.lower(sp), "tr")
            te = np.char.startswith(np.char.lower(sp), "te")
            if te.sum() == 0:
                te = ~tr
        else:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(y))
            cut = int(0.8 * len(y))
            tr = np.zeros(len(y), bool); tr[idx[:cut]] = True
            te = ~tr
        clf = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr])
        out["MCC_linear_probe"] = matthews_corrcoef(y[te], clf.predict(Xs[te]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".npz files (shell globs accepted)")
    ap.add_argument("--compare", action="store_true",
                    help="compare the first two archives on the METRICS")
    ap.add_argument("--probe", action="store_true",
                    help="add the linear probe to the comparison (slower)")
    ap.add_argument("--quick", action="store_true", help="skip the heavy statistics")
    args = ap.parse_args()

    paths = []
    for p in args.paths:
        paths.extend(sorted(glob.glob(p)) or [p])

    loaded = [describe(p, deep=not args.quick) for p in paths]
    loaded = [d for d in loaded if d and d.get("X") is not None]

    if args.compare:
        if len(loaded) < 2:
            sys.exit("\n--compare needs two readable archives")
        a, b = loaded[0], loaded[1]
        print(f"\n{'='*92}\nCOMPARISON ON THE METRICS (what the argument rests on)\n{'='*92}")
        print(f"  A = {a['path']}\n  B = {b['path']}\n")

        if a["X"].shape == b["X"].shape:
            A = np.nan_to_num(a["X"].astype(np.float64))
            B = np.nan_to_num(b["X"].astype(np.float64))
            num = (A * B).sum(1)
            den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
            cos = num / den
            print(f"  row cosine: min={cos.min():.6f}  mean={cos.mean():.6f}")
            rel = np.abs(A - B).sum() / (np.abs(A).sum() + 1e-12)
            print(f"  relative difference: {100*rel:.2f}%\n")
        else:
            print("  different shapes: skipping the vector comparison\n")

        ma = metrics(a["X"], a["z"], probe=args.probe)
        mb = metrics(b["X"], b["z"], probe=args.probe)
        keys = sorted(set(ma) | set(mb))
        if keys:
            print(f"  {'metric':<20} {'A':>10} {'B':>10} {'delta':>10}")
            for k in keys:
                va, vb = ma.get(k), mb.get(k)
                d = f"{vb-va:+.4f}" if (va is not None and vb is not None) else "-"
                print(f"  {k:<20} {va if va is None else f'{va:.4f}':>10} "
                      f"{vb if vb is None else f'{vb:.4f}':>10} {d:>10}")
            print("\n  If the deltas are negligible, the clamp is cosmetic and should")
            print("  be reported as a diagnostic. If they move, every 519M result must")
            print("  report the clamp_fraction stored in the .npz metadata.")


if __name__ == "__main__":
    main()
