#!/usr/bin/env python
"""
decompose_trivial.py — break the `trivial` baseline into its components.

WHY THIS EXISTS
In classify_baselines.py the feature set `trivial(GC+rep+N)` bundles three
columns. But `repeat_frac` is derived from the FASTA soft-masking, while the
`seq` column is stored uppercased: the model — and the k-mer baseline — CANNOT
see repetitiveness at all. Comparing embeddings against `trivial` is therefore
unfair in a specific direction: the baseline is using information that was
deliberately withheld from the model.

This script evaluates the components separately:

  gc          1 feature,  VISIBLE to the model
  repeat      1 feature,  NOT visible to the model
  gc+n        2 features, the honest compositional baseline
  gc+rep+n    3 features, an exact replica of `trivial` (GATE)

The `gc+rep+n` set exists as a control: it must reproduce, digit for digit, the
`trivial(GC+rep+N)` values printed by classify_baselines.py on the same parquet.
If it does not, the protocol here has drifted and none of these numbers are
comparable with the series already produced.

PROTOCOL (identical to classify_baselines.py)
  train on split=train, evaluate on split=test (disjoint scaffolds)
  StandardScaler applied to logreg only; RF on raw features
  LogisticRegression(max_iter=5000, C=1.0)
  RandomForestClassifier(n_estimators=300, random_state=seed)

Usage
  python decompose_trivial.py --windows out_L0050/windows.parquet \\
      --outdir out_L0050/clf_decomp
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

FEATURE_SETS = {
    "gc":       ["gc"],
    "repeat":   ["repeat_frac"],
    "gc+n":     ["gc", "n_frac"],
    "gc+rep+n": ["gc", "repeat_frac", "n_frac"],
}

MODEL_VISIBLE = {"gc": True, "repeat": False, "gc+n": True, "gc+rep+n": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--outdir", default="out/clf_decomp")
    ap.add_argument("--classifier", default="both",
                    choices=["logreg", "rf", "both"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, f1_score,
                                 matthews_corrcoef, confusion_matrix)

    cols = ["label", "split", "gc", "repeat_frac", "n_frac", "start", "end"]
    df = pd.read_parquet(args.windows, columns=cols)

    # effective L: the window is [c - L//2, c + L//2), so end - start = 2 * (L//2)
    L = int(np.median((df["end"] - df["start"]).to_numpy()))

    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())
    is_tr = (df["split"] == "train").to_numpy()
    is_te = (df["split"] == "test").to_numpy()
    print(f"L={L}  windows={len(df)}  classes={list(le.classes_)}  "
          f"train={is_tr.sum()}  test={is_te.sum()}")

    # Zero-variance columns: at small L, n_frac is identically 0 because
    # --max-n-frac 0.10 over a handful of bases rejects any window containing
    # even one N. Worth flagging so a flat result is not read as a finding.
    degenerate = [c for c in ("gc", "repeat_frac", "n_frac")
                  if float(df[c].to_numpy().std()) == 0.0]
    if degenerate:
        print(f"  WARNING: zero-variance (inert) columns: {degenerate}")

    kinds = ["logreg", "rf"] if args.classifier == "both" else [args.classifier]

    results = []
    for fname, fcols in FEATURE_SETS.items():
        X = df[fcols].to_numpy().astype(np.float32)
        Xtr, Xte = X[is_tr], X[is_te]
        ytr, yte = y[is_tr], y[is_te]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        for kind in kinds:
            if kind == "logreg":
                clf = LogisticRegression(max_iter=5000, C=1.0, n_jobs=-1)
                Xf_tr, Xf_te = Xtr_s, Xte_s
            else:
                clf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                             random_state=args.seed)
                Xf_tr, Xf_te = Xtr, Xte

            clf.fit(Xf_tr, ytr)
            pred = clf.predict(Xf_te)
            row = dict(
                L=L, feature=fname, clf=kind, dim=int(X.shape[1]),
                model_visible=MODEL_VISIBLE[fname],
                accuracy=float(accuracy_score(yte, pred)),
                macro_f1=float(f1_score(yte, pred, average="macro")),
                mcc=float(matthews_corrcoef(yte, pred)),
            )
            results.append(row)
            vis = "vis" if MODEL_VISIBLE[fname] else "NOT-vis"
            print(f"[{fname:10s} | {kind:6s} | {vis:7s}]  "
                  f"acc={row['accuracy']:.4f}  macroF1={row['macro_f1']:.4f}  "
                  f"MCC={row['mcc']:.4f}")

            cm = confusion_matrix(yte, pred)
            np.savetxt(os.path.join(args.outdir, f"cm_{fname}_{kind}.csv"),
                       cm, fmt="%d", delimiter=",",
                       header=",".join(le.classes_), comments="")

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(args.outdir, "decompose_results.csv"), index=False)
    with open(os.path.join(args.outdir, "decompose_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # How much of the `trivial` baseline rests on information the model cannot see
    def best(n):
        return max(r["mcc"] for r in results if r["feature"] == n)

    gc_b, rep_b, full_b = best("gc"), best("repeat"), best("gc+rep+n")
    print(f"\nL={L}  GC-only={gc_b:.4f}  repeat-only={rep_b:.4f}  "
          f"full={full_b:.4f}")
    if full_b > 0:
        print(f"  share NOT visible to the model: "
              f"repeat-only / full = {rep_b / full_b:.1%}")
    print(f"\nResults in {args.outdir}/")


if __name__ == "__main__":
    main()
