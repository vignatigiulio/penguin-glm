#!/usr/bin/env python
"""
classify_baselines.py — supervised classification of functional elements.

Trains on TRAIN, evaluates on TEST. The splits are chromosome-disjoint, so there
is no leakage between them.

This is the verdict step. Clustering can fail to separate the classes while the
information is still present and linearly recoverable, so the question that
matters is not "do the embeddings cluster?" but "do they beat the trivial
baselines?" Two baselines are provided for that comparison:

  --do-kmer     raw k-mer composition. A learned representation that cannot beat
                counting 4-mers is not contributing anything.
  --do-trivial  GC + repeat fraction + N fraction. The confounders, naked. If an
                embedding scores near this, it has encoded composition and
                little else.

Metrics on TEST: accuracy, macro-F1, MCC, plus the confusion matrix per feature
set and classifier.

Usage:
  python classify_baselines.py --windows out/windows.parquet \\
    --emb crossdna81=out/emb_crossdna_81.npz \\
    --do-kmer --do-trivial --classifier both --outdir out/clf
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from common import load_embeddings, align_to_ids, kmer_features


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--emb", action="append", default=[], help="name=file.npz")
    ap.add_argument("--do-kmer", action="store_true")
    ap.add_argument("--kmer-k", type=int, default=4)
    ap.add_argument("--do-trivial", action="store_true")
    ap.add_argument("--classifier", default="both", choices=["logreg", "rf", "both"])
    ap.add_argument("--outdir", default="out/clf")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, f1_score,
                                 matthews_corrcoef, confusion_matrix)

    cols = ["id", "label", "split", "gc", "repeat_frac", "n_frac"]
    if args.do_kmer:
        cols.append("seq")
    df = pd.read_parquet(args.windows, columns=cols)

    ids = df["id"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())
    is_tr = (df["split"] == "train").to_numpy()
    is_te = (df["split"] == "test").to_numpy()
    print(f"windows={len(ids)}  classes={list(le.classes_)}  "
          f"train={is_tr.sum()}  test={is_te.sum()}")

    # ---- assemble the feature sets ----
    feats = {}
    for spec in args.emb:
        name, path = spec.split("=", 1)
        eids, X, meta = load_embeddings(path)
        feats[name] = align_to_ids(eids, X, ids).astype(np.float32)
    if args.do_trivial:
        feats["trivial(GC+rep+N)"] = np.column_stack([
            df["gc"].to_numpy(), df["repeat_frac"].to_numpy(), df["n_frac"].to_numpy()
        ]).astype(np.float32)
    if args.do_kmer:
        print(f"computing k-mer features (k={args.kmer_k}) ...")
        feats[f"kmer{args.kmer_k}"] = kmer_features(df["seq"].tolist(), args.kmer_k)

    def make_clf(kind):
        if kind == "logreg":
            return LogisticRegression(max_iter=5000, C=1.0, n_jobs=-1)
        return RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                      random_state=args.seed)

    kinds = ["logreg", "rf"] if args.classifier == "both" else [args.classifier]

    results = []
    for fname, X in feats.items():
        Xtr, Xte = X[is_tr], X[is_te]
        ytr, yte = y[is_tr], y[is_te]
        # fitted on TRAIN only — the scaler is part of the model, not of the data
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        for kind in kinds:
            clf = make_clf(kind)
            Xf_tr = Xtr_s if kind == "logreg" else Xtr  # RF does not need scaling
            Xf_te = Xte_s if kind == "logreg" else Xte
            clf.fit(Xf_tr, ytr)
            pred = clf.predict(Xf_te)
            row = dict(
                feature=fname, clf=kind, dim=int(X.shape[1]),
                accuracy=float(accuracy_score(yte, pred)),
                macro_f1=float(f1_score(yte, pred, average="macro")),
                mcc=float(matthews_corrcoef(yte, pred)),
            )
            results.append(row)
            print(f"[{fname:22s} | {kind:6s}]  acc={row['accuracy']:.4f}  "
                  f"macroF1={row['macro_f1']:.4f}  MCC={row['mcc']:.4f}")
            cm = confusion_matrix(yte, pred)
            np.savetxt(os.path.join(args.outdir, f"cm_{fname}_{kind}.csv"),
                       cm, fmt="%d", delimiter=",",
                       header=",".join(le.classes_), comments="")

    res = pd.DataFrame(results).sort_values("mcc", ascending=False)
    res.to_csv(os.path.join(args.outdir, "classification_results.csv"), index=False)
    with open(os.path.join(args.outdir, "classification_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== RANKING (by decreasing MCC) ===")
    print(res.to_string(index=False))

    # Does the best embedding beat the best baseline?
    emb_names = [s.split("=", 1)[0] for s in args.emb]
    base = res[~res["feature"].isin(emb_names)]
    embr = res[res["feature"].isin(emb_names)]
    if len(base) and len(embr):
        gap = embr["mcc"].max() - base["mcc"].max()
        print(f"\nbest embedding MCC={embr['mcc'].max():.4f}  vs  "
              f"best baseline MCC={base['mcc'].max():.4f}  -> gap={gap:+.4f}")
        print("VERDICT:", "the embeddings ADD signal" if gap > 0.03
              else "the embeddings do NOT beat the trivial baselines")
    print(f"\nConfusion matrices and results in {args.outdir}/")


if __name__ == "__main__":
    main()
