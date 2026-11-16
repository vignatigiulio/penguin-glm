#!/usr/bin/env python3
"""
gc_per_dimension.py: is a drop in NMI against GC a redistribution of variance,
or a suppression of the compositional signal?

For every model it measures three things, on the same embeddings used by the
clustering:

  1. per-dimension profile   |correlation| with GC of each of the D dimensions,
                             sorted from most to least correlated
  2. GC decodability         R^2 of a ridge regression GC ~ embedding, estimated
                             on held-out data (30%): how much information about
                             GC is still present, wherever it sits
  3. weight on the geometry  share of the total variance that falls along the
                             direction predicting GC (1/D is the share of an
                             arbitrary direction), and variance of PC1

Reading: if R^2 stays similar across models while the variance share along the
GC direction falls, GC is still fully in the representation but weighs less on
the geometry (redistribution). If R^2 drops clearly, the signal was removed
(suppression).

Measured on JanusDNA-144 (thesis): R^2 between 0.996 and 0.998 for every model,
so GC is never removed. The variance share along the GC direction is 1.3% for
the from-scratch baseline, 2.1% for stem S=4, 3.5% for the human checkpoint and
0.7% for step 1, the share of an arbitrary direction (1/144): it does not follow
the NMI, and a redistribution away from GC is documented only for step 1.

As in the clustering (src/cluster_embeddings.py), embeddings are standardised
with TRAIN statistics only; --scaling centred uses centred data only, as
analysis/analyze_embeddings.py does.

The thesis version of this script wrote Italian keys (r2_gc, quota_varianza_gc,
corr_media, ...); the JSON written here uses the English names listed in main().

Usage:
  python analysis/gc_per_dimension.py --windows $W \\
      --emb "human=$E/embeddings_janusdna144/embeddings_144_without_midattn.npz" \\
      --emb "baseline=$E/emb_janusdna_144_scratch_clean.npz" \\
      --emb "stem S=4=$E/embeddings_janusdna144_stem_s4/embeddings_144_stem_s4.npz" \\
      --emb "step 1=$E/embeddings_janusdna144_stem_s1_b4/embeddings_144_stem_s1_b4.npz" \\
      --out figures/gc_per_dimension.pdf --json $E/gc_per_dimension.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import align_to_ids, load_embeddings  # noqa: E402

PALETTE = ["#7a4fa3", "#4d4d4d", "#2b7bba", "#d1495b", "#3a9a4a", "#e08a1e", "#9cc6ea"]


def name_path_pairs(specs):
    """Turn ['LABEL=path', ...] into (label, path) tuples.

    Both the label ("stem S=4") and the path may contain '='. Every '=' is tried
    from the left and the first one whose right-hand side is an existing file is
    kept.
    """
    out = []
    for s in specs:
        cuts = [i for i, ch in enumerate(s) if ch == "="]
        if not cuts:
            raise SystemExit(f"expected LABEL=path, got: {s}")
        for i in cuts:
            name, path = s[:i], s[i + 1:]
            if name and os.path.isfile(path):
                out.append((name, path))
                break
        else:
            raise SystemExit(f"no existing file in: {s}")
    return out


def analyse(X, gc, is_tr, scaling, seed):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split

    if scaling == "standard":
        mu = X[is_tr].mean(0, keepdims=True)
        sd = X[is_tr].std(0, keepdims=True) + 1e-9
        Xs = (X - mu) / sd
    else:
        Xs = X - X.mean(0, keepdims=True)

    # 1. |corr| per dimension
    Z = (Xs - Xs.mean(0, keepdims=True)) / (Xs.std(0, keepdims=True) + 1e-9)
    g = (gc - gc.mean()) / (gc.std() + 1e-9)
    corr = np.abs(Z.T @ g / len(g))

    # 2. GC decodability on held-out data
    idx = np.arange(len(Xs))
    itr, ite = train_test_split(idx, test_size=0.3, random_state=seed)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xs[itr], gc[itr])
    r2 = float(ridge.score(Xs[ite], gc[ite]))

    # 3. variance share along the GC direction, and PCA profile
    Xc = Xs - Xs.mean(0, keepdims=True)
    u = ridge.coef_ / (np.linalg.norm(ridge.coef_) + 1e-12)
    var_tot = float(Xc.var(0).sum())
    share_gc = float((Xc @ u).var() / var_tot)
    pca = PCA(n_components=min(20, min(Xs.shape)), random_state=0).fit(Xc)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return dict(corr=corr, r2=r2, share_gc=share_gc, cum=cum,
                var_pc1=float(pca.explained_variance_ratio_[0]), dim=int(Xs.shape[1]),
                alpha=float(ridge.alpha_))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--emb", action="append", required=True, help="LABEL=path.npz (repeatable)")
    ap.add_argument("--out", default="figures/gc_per_dimension.pdf")
    ap.add_argument("--json", default=None, help="where to save the metrics")
    ap.add_argument("--scaling", choices=["standard", "centred"], default="standard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.windows, columns=["id", "gc", "split"])
    ids = df["id"].astype(str).tolist()
    gc = df["gc"].to_numpy().astype(float)
    is_tr = (df["split"] == "train").to_numpy()

    res = {}
    for name, path in name_path_pairs(args.emb):
        emb_ids, X, _ = load_embeddings(path)
        X = align_to_ids(list(emb_ids), X, ids).astype(np.float64)
        res[name] = analyse(X, gc, is_tr, args.scaling, args.seed)
        m = res[name]
        print(f"[{name}] mean |corr|={m['corr'].mean():.3f} max={m['corr'].max():.3f} "
              f"dims with |corr|>0.5: {(m['corr'] > 0.5).sum()}   R2(GC)={m['r2']:.3f}   "
              f"variance share of the GC direction={m['share_gc']:.4f} (1/D={1 / m['dim']:.4f})   "
              f"var PC1={m['var_pc1']:.3f}")

    names = list(res)
    ref = "baseline" if "baseline" in names else names[0]
    print(f"\ncompared with '{ref}':")
    for n in names:
        if n == ref:
            continue
        a, b = res[ref], res[n]
        print(f"  {n:14s} delta R2 = {b['r2'] - a['r2']:+.3f}   "
              f"GC share: {a['share_gc']:.4f} -> {b['share_gc']:.4f} "
              f"({(b['share_gc'] / a['share_gc'] - 1) * 100:+.0f}%)")
    print("  similar R2 and falling share -> redistribution; clearly falling R2 -> suppression")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0))
    colours = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}

    ax = axes[0, 0]
    for n in names:
        c = np.sort(res[n]["corr"])[::-1]
        ax.plot(np.arange(1, len(c) + 1), c, lw=1.4, color=colours[n], label=n)
    ax.set_title("(a) |correlation| with GC per dimension")
    ax.set_xlabel("dimension (decreasing order)")
    ax.set_ylabel("|r| with GC")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for n in names:
        cum = res[n]["cum"]
        ax.plot(np.arange(1, len(cum) + 1), cum, lw=1.4, marker="o", ms=2.5, color=colours[n])
    ax.axhline(0.8, color="#999999", lw=0.8, ls="--")
    ax.set_title("(b) cumulative variance, first 20 components")
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("explained variance")
    ax.grid(alpha=0.25)

    x = np.arange(len(names))
    ax = axes[1, 0]
    ax.bar(x, [res[n]["r2"] for n in names], color=[colours[n] for n in names])
    for i, n in enumerate(names):
        ax.text(i, res[n]["r2"] + 0.01, f"{res[n]['r2']:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_title("(c) GC decodable, held-out R²")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    shares = [res[n]["share_gc"] for n in names]
    ax.bar(x, shares, color=[colours[n] for n in names])
    ax.axhline(1 / res[names[0]]["dim"], color="#999999", lw=0.8, ls="--")
    ax.annotate("1/D", xy=(1.0, 1 / res[names[0]]["dim"]),
                xycoords=("axes fraction", "data"), xytext=(3, 0), textcoords="offset points",
                va="center", ha="left", fontsize=7, color="#666666")
    for i, q in enumerate(shares):
        ax.text(i, q, f"{q:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_title("(d) variance along the GC direction")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    if not args.no_png:
        fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200)
    plt.close(fig)
    print(f"[ok] {args.out}")

    if args.json:
        out = {n: {"corr_mean": float(m["corr"].mean()), "corr_max": float(m["corr"].max()),
                   "n_dims_corr_above_0.5": int((m["corr"] > 0.5).sum()),
                   "r2_gc": m["r2"], "ridge_alpha": m["alpha"], "gc_variance_share": m["share_gc"],
                   "var_pc1": m["var_pc1"], "cumulative_variance_20": [float(v) for v in m["cum"]],
                   "corr_per_dimension": [float(v) for v in m["corr"]]}
               for n, m in res.items()}
        out["_protocol"] = {"scaling": args.scaling, "seed": args.seed}
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[ok] {args.json}")


if __name__ == "__main__":
    main()
