#!/usr/bin/env python
"""
cluster_embeddings.py — unsupervised structure of the frozen embeddings.

Runs k-means with k = number of classes and measures how well the resulting
partition recovers three different things:

    NMI(class)   the biological annotation — what we hope to find
    NMI(GC)      nucleotide composition    — the confounder
    NMI(repeat)  soft-masked repeat content — the other confounder

Reporting all three is the whole point. A single NMI(class) figure looks
unremarkable in isolation; the comparison against the confounders is what turns
"clustering is weak" into the specific and much more interesting claim that the
latent space is organised by composition rather than function. On this dataset
NMI(GC) exceeds NMI(class) by roughly an order of magnitude for every model
tested.

The projection figure is drawn twice, coloured by class and by GC, for exactly
the same reason: the second panel is usually the one that explains the first.

Usage:
  python cluster_embeddings.py --windows out/windows.parquet \\
    --emb janus144=out/emb_janusdna_144.npz \\
    --emb crossdna81=out/emb_crossdna_81.npz \\
    --outdir out/cluster --umap
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from common import load_embeddings, align_to_ids


def bins(series, q=8):
    """Quantile binning, tolerant of repeated values.

    The continuous confounders have to be discretised before NMI can be
    computed against them.
    """
    try:
        return pd.qcut(series, q=q, labels=False, duplicates="drop").to_numpy()
    except Exception:
        return pd.cut(series, bins=min(q, 5), labels=False).to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--emb", action="append", required=True,
                    help="name=path.npz (repeatable)")
    ap.add_argument("--outdir", default="out/cluster")
    ap.add_argument("--umap", action="store_true", help="use UMAP (default: PCA)")
    ap.add_argument("--viz-n", type=int, default=8000, help="windows to plot")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (normalized_mutual_info_score as NMI,
                                 adjusted_rand_score as ARI,
                                 silhouette_score)

    df = pd.read_parquet(args.windows,
                         columns=["id", "label", "gc", "repeat_frac", "split"])
    ids = df["id"].tolist()
    y = df["label"].to_numpy()
    classes = sorted(pd.unique(y))
    k = len(classes)
    is_tr = (df["split"] == "train").to_numpy()
    gc_bin = bins(df["gc"])
    rep_bin = bins(df["repeat_frac"])
    print(f"windows={len(ids)}  classes={classes}  k={k}")

    results = []
    for spec in args.emb:
        name, path = spec.split("=", 1)
        emb_ids, X, meta = load_embeddings(path)
        X = align_to_ids(emb_ids, X, ids).astype(np.float32)

        # The scaler is fitted on TRAIN only, even though clustering is
        # unsupervised: fitting it on everything would leak test statistics
        # into the representation.
        scaler = StandardScaler().fit(X[is_tr])
        Xs = scaler.transform(X)

        km = KMeans(n_clusters=k, n_init=10, random_state=args.seed)
        cl = km.fit_predict(Xs)

        # silhouette is O(n^2), so it is computed on a subsample
        rng = np.random.default_rng(args.seed)
        sub = rng.choice(len(Xs), size=min(5000, len(Xs)), replace=False)
        sil = float(silhouette_score(Xs[sub], cl[sub]))

        row = dict(
            emb=name, dim=int(X.shape[1]),
            NMI_class=float(NMI(y, cl)),
            ARI_class=float(ARI(y, cl)),
            silhouette=sil,
            NMI_vs_GC=float(NMI(gc_bin, cl)),
            NMI_vs_repeat=float(NMI(rep_bin, cl)),
        )
        results.append(row)
        print(f"\n[{name}]  dim={row['dim']}")
        print(f"  NMI(class)    = {row['NMI_class']:.4f}   ARI = {row['ARI_class']:.4f}"
              f"   silhouette = {row['silhouette']:.4f}")
        print(f"  NMI(vs GC)    = {row['NMI_vs_GC']:.4f}   "
              f"NMI(vs repeat) = {row['NMI_vs_repeat']:.4f}")
        gap = row["NMI_class"] - max(row["NMI_vs_GC"], row["NMI_vs_repeat"])
        print("  -> " + ("GOOD signal (classes beat the confounders)" if gap > 0.05
                         else "WARNING: the confounders explain as much as the classes"))

        # ---- projection figure ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            vsub = rng.choice(len(Xs), size=min(args.viz_n, len(Xs)), replace=False)
            if args.umap:
                import umap

                emb2 = umap.UMAP(n_neighbors=30, min_dist=0.1,
                                 random_state=args.seed).fit_transform(Xs[vsub])
                proj = "UMAP"
            else:
                from sklearn.decomposition import PCA

                emb2 = PCA(n_components=2, random_state=args.seed).fit_transform(Xs[vsub])
                proj = "PCA"
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            ci = {c: i for i, c in enumerate(classes)}
            col = np.array([ci[v] for v in y[vsub]])
            ax[0].scatter(emb2[:, 0], emb2[:, 1], c=col, cmap="tab10", s=4, alpha=0.5)
            ax[0].set_title(f"{name} — {proj} by CLASS")
            handles = [plt.Line2D([0], [0], marker="o", ls="",
                                  color=plt.cm.tab10(ci[c] / 10), label=c)
                       for c in classes]
            ax[0].legend(handles=handles, fontsize=8, markerscale=1.5)
            s1 = ax[1].scatter(emb2[:, 0], emb2[:, 1], c=df["gc"].to_numpy()[vsub],
                               cmap="viridis", s=4, alpha=0.5)
            ax[1].set_title(f"{name} — {proj} by GC")
            fig.colorbar(s1, ax=ax[1], label="GC")
            fig.tight_layout()
            figpath = os.path.join(args.outdir, f"proj_{name}.png")
            fig.savefig(figpath, dpi=130)
            plt.close(fig)
            print(f"  figure -> {figpath}")
        except Exception as e:
            print(f"  [figure skipped: {e}]")

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(args.outdir, "clustering_results.csv"), index=False)
    with open(os.path.join(args.outdir, "clustering_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.outdir}/clustering_results.{{csv,json}}")


if __name__ == "__main__":
    main()
