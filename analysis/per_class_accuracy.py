#!/usr/bin/env python3
"""
per_class_accuracy.py — compare per-class accuracy (recall) between two models,
reading the confusion matrix from each one's lora_results.json.

recall of class i = CM[i,i] / sum(CM[i, :])   (how many true-i are predicted i)

Useful because two models can reach nearly the same MCC by very different
routes. In this project the from-scratch and human-pretrained JanusDNA runs end
within 0.002 MCC of each other after LoRA at saturation, and the per-class
breakdown is where any actual difference between them would show up.

Usage:
  python per_class_accuracy.py penguin=out/lora_baseline_ep25 \\
                               human=out/lora_human_ep25
"""
import json
import sys
from pathlib import Path

import numpy as np

DEFAULTS = {
    "penguin": "out/lora_baseline_ep25",
    "human": "out/lora_human_ep25",
}


def load(folder):
    d = json.loads((Path(folder) / "lora_results.json").read_text())
    cm = np.array(d["confusion_matrix"], dtype=float)
    classes = d["classes"]
    return cm, classes, d.get("test_mcc")


def per_class_acc(cm):
    return cm.diagonal() / cm.sum(axis=1)


def main():
    models = {}
    args = [a for a in sys.argv[1:] if "=" in a]
    if args:
        for a in args:
            name, path = a.split("=", 1)
            models[name] = path
    else:
        models = DEFAULTS

    data = {name: load(path) for name, path in models.items()}
    # use the class order of the first model (verified identical across runs)
    ref_classes = next(iter(data.values()))[1]

    names = list(data.keys())
    print(f"\n{'class':<12}", end="")
    for n in names:
        print(f"{n:>12}", end="")
    if len(names) == 2:
        print(f"{'delta':>10}", end="")
    print()
    print("-" * (12 + 12 * len(names) + (10 if len(names) == 2 else 0)))

    accs = {n: per_class_acc(data[n][0]) for n in names}
    for i, cls in enumerate(ref_classes):
        print(f"{cls:<12}", end="")
        vals = []
        for n in names:
            v = accs[n][i]
            vals.append(v)
            print(f"{v:>12.4f}", end="")
        if len(names) == 2:
            d = vals[0] - vals[1]
            print(f"{d:>+10.4f}", end="")
        print()

    # overall accuracy and MCC, for reference
    print("-" * (12 + 12 * len(names) + (10 if len(names) == 2 else 0)))
    print(f"{'MCC':<12}", end="")
    for n in names:
        print(f"{data[n][2]:>12.4f}", end="")
    print()
    print(f"{'overall acc':<12}", end="")
    for n in names:
        cm = data[n][0]
        print(f"{cm.diagonal().sum()/cm.sum():>12.4f}", end="")
    print("\n")

    # precision per class: of everything predicted i, how much really is i
    print("per-class precision (of those predicted i, how many really are i):")
    print(f"{'class':<12}", end="")
    for n in names:
        print(f"{n:>12}", end="")
    print()
    for i, cls in enumerate(ref_classes):
        print(f"{cls:<12}", end="")
        for n in names:
            cm = data[n][0]
            prec = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
            print(f"{prec:>12.4f}", end="")
        print()


if __name__ == "__main__":
    main()
