#!/usr/bin/env python
"""
compare_confusion.py — side-by-side per-class comparison of several models,
reading confusion matrices from raw CSVs or from *_results.json files.

Aggregate metrics hide where a model is actually winning. Two runs with the same
macro-F1 can differ substantially class by class, and on this task the classes
are not equally hard: intergenic and CDS separate reasonably, UTR and intron do
not. This script puts recall, precision and F1 for every model next to each
other so the difference is visible rather than averaged away.

Usage:
  python compare_confusion.py \\
    --json human=out/lora_janus144/lora_results.json \\
    --json scratch=out/lora_janus144_scratch_clean/lora_results.json \\
    --outdir out/confusion_compare
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def load_cm_from_csv(path):
    # header = class names, rows = counts
    df = pd.read_csv(path)
    classes = list(df.columns)
    cm = df.to_numpy(dtype=float)
    return cm, classes


def load_cm_from_json(path):
    with open(path) as f:
        d = json.load(f)
    cm = np.array(d["confusion_matrix"], dtype=float)
    classes = d.get("classes", [f"c{i}" for i in range(cm.shape[0])])
    return cm, classes


def per_class_metrics(cm, classes):
    """Return a DataFrame with recall/precision/F1/support per class."""
    rows = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        support = cm[i, :].sum()
        pred_pos = cm[:, i].sum()
        recall = tp / support if support > 0 else 0.0
        precision = tp / pred_pos if pred_pos > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        rows.append(dict(cls=c, recall=recall, precision=precision,
                         f1=f1, support=int(support)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cm", action="append", default=[],
                    help="name=file.csv (raw confusion matrix)")
    ap.add_argument("--json", action="append", default=[],
                    help="name=file_results.json (containing confusion_matrix)")
    ap.add_argument("--outdir", default="out/confusion_compare")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    models = {}
    ref_classes = None
    for spec in args.cm:
        name, path = spec.split("=", 1)
        cm, classes = load_cm_from_csv(path)
        models[name] = (cm, classes)
    for spec in args.json:
        name, path = spec.split("=", 1)
        cm, classes = load_cm_from_json(path)
        models[name] = (cm, classes)

    if not models:
        raise SystemExit("No model given. Use --cm or --json.")

    # class consistency. A silent ordering mismatch here would compare CDS
    # against intron and produce a table that looks entirely normal.
    for name, (cm, classes) in models.items():
        if ref_classes is None:
            ref_classes = classes
        elif classes != ref_classes:
            print(f"[WARNING] {name} has classes {classes} != {ref_classes}. "
                  "The per-class comparison assumes the same ordering.")

    # --- per-class recall, models side by side ---
    print("=== per-class RECALL (diagonal normalised by row) ===")
    recall_table = {}
    for name, (cm, classes) in models.items():
        m = per_class_metrics(cm, classes)
        recall_table[name] = m.set_index("cls")["recall"]
    recall_df = pd.DataFrame(recall_table)
    recall_df.loc["MEAN"] = recall_df.mean()
    print(recall_df.round(4).to_string())
    recall_df.to_csv(os.path.join(args.outdir, "recall_per_class.csv"))

    print("\n=== per-class F1, models side by side ===")
    f1_table = {}
    for name, (cm, classes) in models.items():
        m = per_class_metrics(cm, classes)
        f1_table[name] = m.set_index("cls")["f1"]
    f1_df = pd.DataFrame(f1_table)
    f1_df.loc["MEAN(macro)"] = f1_df.mean()
    print(f1_df.round(4).to_string())
    f1_df.to_csv(os.path.join(args.outdir, "f1_per_class.csv"))

    # --- detail per model ---
    print("\n=== detail per model (recall/precision/F1/support) ===")
    all_detail = []
    for name, (cm, classes) in models.items():
        m = per_class_metrics(cm, classes)
        m.insert(0, "model", name)
        all_detail.append(m)
        print(f"\n[{name}]")
        print(m.round(4).to_string(index=False))
    pd.concat(all_detail).to_csv(
        os.path.join(args.outdir, "detail_per_class.csv"), index=False)

    # --- pairwise comparison when exactly two models: who wins where ---
    names = list(models.keys())
    if len(names) == 2:
        a, b = names
        ma = per_class_metrics(*models[a]).set_index("cls")["f1"]
        mb = per_class_metrics(*models[b]).set_index("cls")["f1"]
        diff = (ma - mb).round(4)
        print(f"\n=== {a} vs {b}: per-class F1 difference (positive = {a} better) ===")
        print(diff.to_string())
        print(f"\n{a} wins on: {list(diff[diff > 0].index)}")
        print(f"{b} wins on: {list(diff[diff < 0].index)}")

    print(f"\nTables saved in {args.outdir}/")


if __name__ == "__main__":
    main()
