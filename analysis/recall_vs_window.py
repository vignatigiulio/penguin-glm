#!/usr/bin/env python
"""
recall_vs_window.py — one figure: per-class recall as a function of window length.

X axis: the window lengths of the sweep (log scale).
Y axis: recall, as a percentage.
Four lines, one per functional class (CDS, UTR, intergenic, intron). By default
the data are the post-LoRA model's.

Recall of a class = the share of true instances of that class the model
recognises = diagonal divided by the row sum of the confusion matrix (sklearn
convention: rows = TRUE class, columns = PREDICTED class). With four balanced
classes, chance is 25%.

This is the per-class view of the sweep. The aggregate MCC roughly doubles from
L=64 to L=2000, but it does not do so uniformly: the classes defined by
long-range structure gain far more than the ones separable on composition alone,
which is the mechanism behind the aggregate number.

WHAT IT READS
  out_L*/lora_s*/lora_results.json      LoRA confusion matrix (summed over seeds)
  out_L*/clf*/cm_{feature}_{clf}.csv    matrices saved by classify_baselines.py

USAGE
  python recall_vs_window.py                        # LoRA
  python recall_vs_window.py --model janus_clean --clf logreg
  python recall_vs_window.py --metric f1
  python recall_vs_window.py --list                 # what was found
  python recall_vs_window.py --table                # also print the numbers
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

# one colour per class, kept stable across every figure in the thesis
COLOURS = {
    "CDS":        "#b4413c",
    "UTR":        "#c9a227",
    "intron":     "#2a78d6",
    "intergenic": "#1baf7a",
}
MARKER = {"CDS": "s", "UTR": "D", "intron": "o", "intergenic": "^"}
FALLBACK = ["#b4413c", "#c9a227", "#2a78d6", "#1baf7a", "#8a6fbf"]


def _L_of(path):
    if re.search(r"out_L\d+_s\d+", path):  # skip alternate-seed replicas
        return None
    m = re.search(r"out_L(\d+)", path)
    return int(m.group(1)) if m else None


def read_cm_csv(path):
    try:
        with open(path) as f:
            rows = list(csv.reader(f))
    except Exception:
        return None, None
    if len(rows) < 2:
        return None, None
    classes = [c.strip() for c in rows[0]]
    try:
        M = np.array([[float(x) for x in r] for r in rows[1:]], dtype=float)
    except ValueError:
        return None, None
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] != len(classes):
        return None, None
    return classes, M


def collect():
    """-> {(model, clf): {L: (classes, M)}}, with seeds summed for LoRA."""
    out = {}

    # LoRA: the matrix lives inside the json, summed across seeds
    keys = ("confusion_matrix", "confusion", "cm", "test_confusion_matrix")
    for path in sorted(glob.glob("out_L*/lora_s*/lora_results.json")):
        L = _L_of(path)
        if L is None:
            continue
        try:
            d = json.load(open(path))
        except Exception:
            continue
        for k in keys:
            if k in d and isinstance(d[k], list):
                M = np.array(d[k], dtype=float)
                if M.ndim == 2 and M.shape[0] == M.shape[1]:
                    classes = d.get("classes") or [f"c{i}" for i in range(len(M))]
                    key = ("LoRA", "finetune")
                    prev = out.setdefault(key, {}).get(L)
                    out[key][L] = (classes, M if prev is None else prev[1] + M)
                break

    # baselines and frozen embeddings
    for path in sorted(glob.glob("out_L*/clf*/cm_*.csv")):
        L = _L_of(path)
        if L is None:
            continue
        name = os.path.basename(path)[3:-4]
        if "_" not in name:
            continue
        model, clf = name.rsplit("_", 1)
        classes, M = read_cm_csv(path)
        if M is None:
            continue
        out.setdefault((model, clf), {})[L] = (classes, M)

    return out


def metrics(M):
    true, pred, diag = M.sum(1), M.sum(0), np.diag(M)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(true > 0, diag / true, np.nan)
        pre = np.where(pred > 0, diag / pred, np.nan)
        f1 = np.where((rec + pre) > 0, 2 * rec * pre / (rec + pre), 0.0)
    return {"recall": rec, "precision": pre, "f1": f1}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="LoRA", help="which model to plot")
    ap.add_argument("--clf", default=None,
                    help="classifier, when the model has more than one")
    ap.add_argument("--metric", default="recall",
                    choices=["recall", "precision", "f1"])
    ap.add_argument("--out", default="figures/recall_vs_window.png")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    data = collect()
    if not data:
        print("No confusion matrix found. Run this from the directory "
              "containing the out_L* folders.")
        return

    if args.list:
        for (m, c), per_L in sorted(data.items()):
            print(f"{m:24s} {c:10s} -> L = {sorted(per_L)}")
        return

    cand = [k for k in data if args.model.lower() in k[0].lower()
            and (not args.clf or k[1] == args.clf)]
    if not cand:
        print(f"Model '{args.model}' not found.")
        print("Available:")
        for (m, c) in sorted(data):
            print(f"    --model {m} --clf {c}")
        if args.model.lower() == "lora":
            print("\nIf LoRA is missing, the matrix is not inside lora_results.json.")
        return

    key = cand[0]
    per_L = data[key]
    Ls = sorted(per_L)
    classes = per_L[Ls[0]][0]

    V = []
    for L in Ls:
        cls_L, M = per_L[L]
        # A class-order mismatch between lengths would silently plot CDS against
        # intron, so those points become NaN rather than wrong.
        V.append(metrics(M)[args.metric] if cls_L == classes
                 else [np.nan] * len(classes))
    V = np.array(V, dtype=float) * 100.0  # as a percentage

    print(f"\n[values for metric: {args.metric}]")
    print(f"L in order (positions 0..{len(Ls)-1}): {Ls}")
    for ci, cl in enumerate(classes):
        vec = "[" + ", ".join(f"{x:.2f}%" for x in V[:, ci]) + "]"
        print(f"class {cl:11s}: {vec}")
    print("-" * 50)

    if args.table:
        print(f"\n{key[0]} ({key[1]})   [{args.metric} %, per class]")
        print(f"{'L':>7} | " + " ".join(f"{c:>11}" for c in classes))
        for i, L in enumerate(Ls):
            print(f"{L:7d} | " + " ".join(f"{x:10.1f}%" for x in V[i]))
        if len(Ls) > 1:
            d = V[-1] - V[0]
            print(f"{'delta':>7} | " + " ".join(f"{x:+10.1f}" for x in d))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for ci, cl in enumerate(classes):
        ax.plot(Ls, V[:, ci], marker=MARKER.get(cl, "o"), ms=6, lw=2.2,
                color=COLOURS.get(cl, FALLBACK[ci % len(FALLBACK)]), label=cl)

    ax.axhline(100.0 / len(classes), color="#999999", lw=1.2, ls=":",
               zorder=0, label=f"chance ({100//len(classes)}%)")
    ax.set_xscale("log")
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(x) for x in Ls])
    ax.minorticks_off()
    ax.set_xlabel("L — window length (bp, log scale)")
    ax.set_ylabel(f"{args.metric} (%)")
    ax.set_title(f"Per-class {args.metric} by window length — {key[0]}", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"[figure] {key[0]} ({key[1]}), {len(classes)} classes, "
          f"L = {Ls} -> {args.out}")


if __name__ == "__main__":
    main()
