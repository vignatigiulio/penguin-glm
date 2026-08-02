#!/usr/bin/env python3
"""
analyze_embeddings.py — variability and structure of the embedding space.

Complements cluster_embeddings.py, which already produces the 2D projections
coloured by class and by GC. This script answers different questions:

  1. SCREE / VARIANCE     how much variance do the leading components capture?
                          Do the embeddings live in a few dimensions or are they
                          spread out? -> effective dimensionality (components
                          needed for 80% / 95% of the variance).

  2. PER-DIMENSION SIGNAL how strongly is each of the D dimensions tied to GC?
                          And to function? -> is the GC attractor CONCENTRATED
                          in a few dimensions or DIFFUSE across all of them?
                          This is a second, independent reading of the NMI drop
                          produced by the conv stem: a drop caused by
                          concentration means something different from a drop
                          caused by genuine suppression.

  3. PCA vs LDA           the unsupervised axes against the supervised ones,
                          evaluated on held-out data. This is the crux of the
                          thesis in a single figure: the directions of maximum
                          variance are not the directions that separate the
                          classes, which is exactly why clustering fails while a
                          probe succeeds.

  4. SCALE COMPARISON     the same metrics side by side for baseline / S=4 /
                          S=8 / ...

USAGE (accepts several models, like cluster_embeddings.py):
  python analyze_embeddings.py \\
    --windows out/windows.parquet \\
    --emb scratch_clean=out/emb_janusdna_144_scratch_clean.npz \\
    --emb stem_s4=out/embeddings_janusdna144_stem_s4/embeddings_144_stem_s4.npz \\
    --outdir out/analysis_sweep

Output in --outdir: scree.png, pc_profile_<name>.png, pca_vs_lda_<name>.png,
dim_signal_<name>.png, summary.csv, summary.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--emb", action="append", required=True,
                    help="name=path.npz (repeatable)")
    ap.add_argument("--outdir", default="out/analysis")
    ap.add_argument("--gc-col", default="gc")
    ap.add_argument("--label-col", default=None,
                    help="class column; auto-detected if omitted")
    ap.add_argument("--top-k", type=int, default=15, help="dimensions to list")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    ids = d["ids"] if "ids" in d else None
    return X, ids


def autodetect_label_col(df):
    for c in ("label", "class", "category", "annotation"):
        if c in df.columns:
            return c
    raise SystemExit(f"class column not found; use --label-col. Columns: {list(df.columns)}")


def align(X, ids, df):
    """Align embeddings to windows through the ids, when available."""
    if ids is None:
        if len(X) != len(df):
            raise SystemExit(f"npz without 'ids' and lengths differ ({len(X)} vs {len(df)})")
        return X, df
    if "id" in df.columns:
        key = "id"
    elif "window_id" in df.columns:
        key = "window_id"
    else:
        if len(X) != len(df):
            raise SystemExit("ids present in the npz but no id column in the parquet")
        return X, df
    pos = {str(v): i for i, v in enumerate(df[key].to_numpy())}
    keep_x, keep_df = [], []
    for i, wid in enumerate(ids):
        j = pos.get(str(wid))
        if j is not None:
            keep_x.append(i)
            keep_df.append(j)
    if not keep_x:
        raise SystemExit("no id in common between the npz and the parquet")
    return X[keep_x], df.iloc[keep_df].reset_index(drop=True)


def variance_profile(X):
    """PCA: explained variance plus effective dimensionality."""
    from sklearn.decomposition import PCA

    Xc = X - X.mean(0, keepdims=True)
    n_comp = min(X.shape)
    p = PCA(n_components=n_comp, random_state=0).fit(Xc)
    ev = p.explained_variance_ratio_
    cum = np.cumsum(ev)
    return {
        "explained": ev,
        "cumulative": cum,
        "pc1": float(ev[0]),
        "pc1_pc2": float(cum[1]) if len(cum) > 1 else float(cum[-1]),
        "n_for_80": int(np.searchsorted(cum, 0.80) + 1),
        "n_for_95": int(np.searchsorted(cum, 0.95) + 1),
    }


def lda_projection(X, y, gc, seed=0):
    """SUPERVISED projection (LDA): finds the directions that separate the
    classes rather than the directions of maximum variance.

    Evaluated on held-out data, so that overfitting is not mistaken for real
    separation — an in-sample LDA will always look impressive.

    Returns: the 2D test projection, eta^2 on the classes, |corr| with GC for
    the LDA axes, and LDA accuracy as a linear reference point.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
    n_ax = min(2, len(np.unique(y)) - 1)
    lda = LDA(n_components=n_ax).fit(X[itr], y[itr])
    Zte = lda.transform(X[ite])
    acc = float(lda.score(X[ite], y[ite]))

    Zs = (Zte - Zte.mean(0, keepdims=True)) / (Zte.std(0, keepdims=True) + 1e-9)
    yte = y[ite]
    classes = np.unique(yte)
    grand = Zs.mean(0)
    ssb = np.zeros(Zs.shape[1])
    for c in classes:
        m = (yte == c)
        ssb += m.sum() * (Zs[m].mean(0) - grand) ** 2
    sst = ((Zs - grand) ** 2).sum(0) + 1e-9
    eta2 = ssb / sst

    g = gc[ite]
    g = (g - g.mean()) / (g.std() + 1e-9)
    corr_gc = np.abs(Zs.T @ g / len(g))
    return {"Z": Zte, "y": yte, "eta2": eta2, "corr_gc": corr_gc, "acc": acc}


def per_component_signal(X, gc, y, n_comp=20):
    """For each PRINCIPAL COMPONENT: explained variance, |corr| with GC, and
    eta^2 on the classes.

    Answers: are PC1/PC2 GC axes? Are there components more informative about
    function than the first two? On these embeddings the answer to both is yes,
    which is why a 2D PCA scatter looks so unpromising.
    """
    from sklearn.decomposition import PCA

    k = min(n_comp, min(X.shape))
    p = PCA(n_components=k, random_state=0)
    Z = p.fit_transform(X - X.mean(0, keepdims=True))  # (n, k) scores
    Zs = (Z - Z.mean(0, keepdims=True)) / (Z.std(0, keepdims=True) + 1e-9)
    g = (gc - gc.mean()) / (gc.std() + 1e-9)
    corr_gc = np.abs(Zs.T @ g / len(g))

    classes = np.unique(y)
    grand = Zs.mean(0)
    ssb = np.zeros(k)
    for c in classes:
        m = (y == c)
        ssb += m.sum() * (Zs[m].mean(0) - grand) ** 2
    sst = ((Zs - grand) ** 2).sum(0) + 1e-9
    eta2 = ssb / sst
    return {"var": p.explained_variance_ratio_, "corr_gc": corr_gc,
            "eta2": eta2, "Z": Z}


def per_dimension_signal(X, gc, y):
    """Per dimension: |corr| with GC, and discriminative power on the classes
    (eta^2 = variance explained by class, one-way). Both in [0,1]."""
    Xs = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-9)
    g = (gc - gc.mean()) / (gc.std() + 1e-9)
    corr_gc = np.abs(Xs.T @ g / len(g))  # |Pearson| per dimension

    classes = np.unique(y)
    grand = Xs.mean(0)
    ss_between = np.zeros(Xs.shape[1])
    for c in classes:
        m = (y == c)
        ss_between += m.sum() * (Xs[m].mean(0) - grand) ** 2
    ss_total = ((Xs - grand) ** 2).sum(0) + 1e-9
    eta2 = ss_between / ss_total
    return corr_gc, eta2


def concentration(v):
    """How concentrated a profile is on a few dimensions.
    Gini in [0,1]: 0 = uniform (diffuse), 1 = everything in one dimension."""
    x = np.sort(np.abs(v))
    n = len(x)
    s = x.sum() + 1e-12
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * s))


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df_all = (pd.read_parquet(args.windows) if args.windows.endswith(".parquet")
              else pd.read_csv(args.windows))
    label_col = args.label_col or autodetect_label_col(df_all)
    if args.gc_col not in df_all.columns:
        raise SystemExit(f"GC column '{args.gc_col}' absent. Columns: {list(df_all.columns)}")

    models, rows = {}, []
    for spec in args.emb:
        if "=" not in spec:
            raise SystemExit(f"--emb expects name=path.npz (received: {spec})")
        name, path = spec.split("=", 1)
        X, ids = load_npz(path)
        X, df = align(X, ids, df_all)
        gc = df[args.gc_col].to_numpy().astype(float)
        y = df[label_col].to_numpy()

        var = variance_profile(X)
        corr_gc, eta2 = per_dimension_signal(X, gc, y)
        pcs = per_component_signal(X, gc, y, n_comp=20)
        lda = lda_projection(X, y, gc, seed=args.seed)
        models[name] = dict(X=X, var=var, corr_gc=corr_gc, eta2=eta2,
                            pcs=pcs, lda=lda)

        # the components best for FUNCTION need not be PC1/PC2 — usually aren't
        best_fn = np.argsort(-pcs["eta2"])[:2]
        row = {
            "model": name, "n": len(X), "dim": X.shape[1],
            "var_PC1": round(var["pc1"], 4),
            "var_PC1_2": round(var["pc1_pc2"], 4),
            "eff_dim_80%": var["n_for_80"],
            "eff_dim_95%": var["n_for_95"],
            "PC1_corr_GC": round(float(pcs["corr_gc"][0]), 4),
            "PC2_corr_GC": round(float(pcs["corr_gc"][1]), 4),
            "PC1_eta2_fn": round(float(pcs["eta2"][0]), 4),
            "PC2_eta2_fn": round(float(pcs["eta2"][1]), 4),
            "best_PCs_fn": f"PC{best_fn[0]+1},PC{best_fn[1]+1}",
            "eta2_best_fn": round(float(pcs["eta2"][best_fn].sum()), 4),
            "LDA_eta2_fn": round(float(lda["eta2"].sum()), 4),
            "LDA_acc": round(lda["acc"], 4),
            "LDA1_corr_GC": round(float(lda["corr_gc"][0]), 4),
            "GC_corr_mean": round(float(corr_gc.mean()), 4),
            "GC_corr_max": round(float(corr_gc.max()), 4),
            "GC_concentration": round(concentration(corr_gc), 4),
            "fn_eta2_mean": round(float(eta2.mean()), 4),
            "fn_eta2_max": round(float(eta2.max()), 4),
            "fn_concentration": round(concentration(eta2), 4),
        }
        rows.append(row)
        print(f"\n[{name}]  n={len(X)} dim={X.shape[1]}")
        print(f"  variance: PC1={var['pc1']:.3f}  PC1+2={var['pc1_pc2']:.3f}  "
              f"eff.dim(80%)={var['n_for_80']}  eff.dim(95%)={var['n_for_95']}")
        print(f"  are PC1/PC2 GC axes? |corr GC|: PC1={pcs['corr_gc'][0]:.3f} "
              f"PC2={pcs['corr_gc'][1]:.3f}")
        print(f"  PC1/PC2 for function: eta^2 PC1={pcs['eta2'][0]:.3f} "
              f"PC2={pcs['eta2'][1]:.3f}"
              f"   |  best for function: PC{best_fn[0]+1} "
              f"({pcs['eta2'][best_fn[0]]:.3f}), "
              f"PC{best_fn[1]+1} ({pcs['eta2'][best_fn[1]]:.3f})")
        print(f"  GC      : |corr| mean={corr_gc.mean():.3f} max={corr_gc.max():.3f}  "
              f"concentration={concentration(corr_gc):.3f}")
        print(f"  function: eta^2 mean={eta2.mean():.3f} max={eta2.max():.3f}  "
              f"concentration={concentration(eta2):.3f}")
        print(f"  LDA (supervised, held-out): eta^2={lda['eta2'].sum():.3f} "
              f"acc={lda['acc']:.3f}  |corr GC| axis1={lda['corr_gc'][0]:.3f}")
        top = np.argsort(-corr_gc)[:args.top_k]
        print(f"  top-{args.top_k} dimensions by GC: {list(map(int, top))}")

        # --- figure A: per-COMPONENT profile (variance / GC / function) ---
        k = len(pcs["var"])
        xs = np.arange(1, k + 1)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].bar(xs - 0.2, pcs["corr_gc"], width=0.4, label="|corr| with GC",
                  color="#4878CF")
        ax[0].bar(xs + 0.2, pcs["eta2"], width=0.4, label="$\\eta^2$ function",
                  color="#D65F5F")
        ax[0].set_title(f"{name} — what each component encodes")
        ax[0].set_xlabel("principal component")
        ax[0].legend()
        ax[1].bar(xs, pcs["var"], color="grey")
        ax[1].set_title(f"{name} — explained variance per component")
        ax[1].set_xlabel("principal component")
        ax[1].set_ylabel("share of variance")
        fig.tight_layout()
        f = os.path.join(args.outdir, f"pc_profile_{name}.png")
        fig.savefig(f, dpi=130)
        plt.close(fig)
        print(f"  figure -> {f}")

        # --- figure B: PC1-PC2 against the best components for function ---
        Z = pcs["Z"]
        sub = np.random.default_rng(args.seed).choice(
            len(Z), size=min(6000, len(Z)), replace=False)
        classes = np.unique(y)
        cidx = {c: i for i, c in enumerate(classes)}
        cols = np.array([cidx[v] for v in y])[sub]
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].scatter(Z[sub, 0], Z[sub, 1], c=cols, cmap="tab10", s=4, alpha=0.5)
        ax[0].set_title(f"{name} — PC1 vs PC2 (max variance)\n"
                        f"$\\eta^2$ fn = {pcs['eta2'][0]+pcs['eta2'][1]:.3f}")
        ax[0].set_xlabel("PC1")
        ax[0].set_ylabel("PC2")
        a, b = best_fn
        ax[1].scatter(Z[sub, a], Z[sub, b], c=cols, cmap="tab10", s=4, alpha=0.5)
        ax[1].set_title(f"{name} — PC{a+1} vs PC{b+1} (max functional signal)\n"
                        f"$\\eta^2$ fn = {pcs['eta2'][a]+pcs['eta2'][b]:.3f}")
        ax[1].set_xlabel(f"PC{a+1}")
        ax[1].set_ylabel(f"PC{b+1}")
        handles = [plt.Line2D([0], [0], marker="o", ls="",
                              color=plt.cm.tab10(cidx[c] / 10), label=str(c))
                   for c in classes]
        ax[1].legend(handles=handles, fontsize=8, markerscale=1.2)
        fig.tight_layout()
        f = os.path.join(args.outdir, f"pc_axis_choice_{name}.png")
        fig.savefig(f, dpi=130)
        plt.close(fig)
        print(f"  figure -> {f}")

        # --- figure B2: PCA (unsupervised) against LDA (supervised) ---
        # The single most useful figure in this script: the same points, once
        # under the axes of maximum variance and once under the axes that
        # separate the classes.
        Zl, yl = lda["Z"], lda["y"]
        subl = np.random.default_rng(args.seed).choice(
            len(Zl), size=min(6000, len(Zl)), replace=False)
        colsl = np.array([cidx[v] for v in yl])[subl]
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].scatter(Z[sub, 0], Z[sub, 1], c=cols, cmap="tab10", s=4, alpha=0.5)
        ax[0].set_title(f"{name} — PCA (unsupervised)\n"
                        f"$\\eta^2$ fn = {pcs['eta2'][0]+pcs['eta2'][1]:.3f}")
        ax[0].set_xlabel("PC1")
        ax[0].set_ylabel("PC2")
        if Zl.shape[1] >= 2:
            ax[1].scatter(Zl[subl, 0], Zl[subl, 1], c=colsl, cmap="tab10",
                          s=4, alpha=0.5)
            ax[1].set_ylabel("LD2")
        else:
            ax[1].scatter(Zl[subl, 0],
                          np.random.default_rng(0).normal(0, .05, len(subl)),
                          c=colsl, cmap="tab10", s=4, alpha=0.5)
        ax[1].set_title(f"{name} — LDA (supervised, held-out)\n"
                        f"$\\eta^2$ fn = {lda['eta2'].sum():.3f}  "
                        f"acc = {lda['acc']:.3f}")
        ax[1].set_xlabel("LD1")
        ax[1].legend(handles=handles, fontsize=8, markerscale=1.2)
        fig.tight_layout()
        f = os.path.join(args.outdir, f"pca_vs_lda_{name}.png")
        fig.savefig(f, dpi=130)
        plt.close(fig)
        print(f"  figure -> {f}")

        # --- figure C: signal per ORIGINAL dimension ---
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        order = np.argsort(-corr_gc)
        ax[0].bar(range(len(corr_gc)), corr_gc[order], color="#4878CF")
        ax[0].set_title(f"{name} — |corr| with GC per dimension (sorted)")
        ax[0].set_xlabel("dimension (reordered)")
        ax[0].set_ylabel("|corr| GC")
        order2 = np.argsort(-eta2)
        ax[1].bar(range(len(eta2)), eta2[order2], color="#D65F5F")
        ax[1].set_title(f"{name} — $\\eta^2$ function per dimension (sorted)")
        ax[1].set_xlabel("dimension (reordered)")
        ax[1].set_ylabel("$\\eta^2$ class")
        fig.tight_layout()
        f = os.path.join(args.outdir, f"dim_signal_{name}.png")
        fig.savefig(f, dpi=130)
        plt.close(fig)
        print(f"  figure -> {f}")

    # comparative scree
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for name, m in models.items():
        ev, cum = m["var"]["explained"], m["var"]["cumulative"]
        k = min(40, len(ev))
        ax[0].plot(range(1, k + 1), ev[:k], marker="o", ms=3, label=name)
        ax[1].plot(range(1, len(cum) + 1), cum, label=name)
    ax[0].set_title("Scree: explained variance per component")
    ax[0].set_xlabel("component")
    ax[0].set_ylabel("share of variance")
    ax[0].legend()
    ax[1].axhline(0.8, ls="--", lw=0.8, c="grey")
    ax[1].axhline(0.95, ls="--", lw=0.8, c="grey")
    ax[1].set_title("Cumulative variance")
    ax[1].set_xlabel("number of components")
    ax[1].set_ylabel("cumulative")
    ax[1].legend()
    fig.tight_layout()
    f = os.path.join(args.outdir, "scree.png")
    fig.savefig(f, dpi=130)
    plt.close(fig)
    print(f"\ncomparative scree -> {f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\n=== SUMMARY ===")
    print(out.to_string(index=False))
    print(f"\nWrote: {args.outdir}/summary.{{csv,json}}")


if __name__ == "__main__":
    main()
