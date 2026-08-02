#!/usr/bin/env python
"""
eval_lora.py — evaluate a saved LoRA checkpoint (best_lora.pt) without retraining.

Rebuilds the model, re-injects the adapters, loads the saved weights and the
head, and reports accuracy / macro-F1 / MCC plus the confusion matrix.

FIXED RELATIVE TO THE ORIGINAL SCRIPT. The version used during the thesis work
omitted the fp32 attention patch that finetune_lora_crossdna.py applies before
training. Adapters trained under fp32 attention were therefore evaluated on the
native bf16 autocast path — the very path the 519M checkpoint overflows on — and
the mismatch cost about 0.03 MCC (0.4474 at training time against 0.4150 here).
The patch is now applied by default; `--no-fp32-attention` exists only to
reproduce the old, mismatched behaviour deliberately.

The rule this encodes: an evaluation must reconstruct the training-time
numerical environment, not merely the training-time weights.

Usage:
    python eval_lora.py --weights $PENGUIN_ROOT/weights/519M \\
       --windows out/windows.parquet --ckpt out/lora519/best_lora.pt \\
       --split test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")

import numpy as np
import pandas as pd

# runtime patches live in env/patches/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "env" / "patches"))

BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
LORA_TARGET_SUFFIXES = (
    "comba.q_proj", "comba.k_proj", "comba.v_proj", "comba.o_proj",
    "mixer.q_proj", "mixer.k_proj", "mixer.v_proj", "mixer.o_proj",
)


def encode(seqs, device):
    import torch

    arr = np.empty((len(seqs), len(seqs[0])), dtype=np.int64)
    for r, s in enumerate(seqs):
        arr[r] = [BASE.get(c, 4) for c in s]
    return torch.from_numpy(arr).to(device)


def build_lora_linear(base, r, alpha, dropout):
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, base, r, alpha, dropout):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.r = r
            self.scaling = alpha / r
            self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            out = self.base(x)
            lora = (self.drop(x) @ self.lora_A.t()) @ self.lora_B.t()
            return out + self.scaling * lora.to(out.dtype)

    return LoRALinear(base, r, alpha, dropout)


def inject_lora(model, r, alpha, dropout):
    import torch.nn as nn

    injected = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and full.endswith(LORA_TARGET_SUFFIXES):
                setattr(mod, child_name, build_lora_linear(child, r, alpha, dropout))
                injected += 1
    return injected


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--pooling", default="mean", choices=["mean", "max"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--no-fp32-attention", action="store_true",
                    help="skip the fp32 attention patch — reproduces the old "
                         "train/eval mismatch, do not use for reported numbers")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    torch.compile = (lambda fn=None, *a, **k:
                     (fn if fn is not None else (lambda g: g)))
    import torch.nn as nn
    from transformers import AutoModelForMaskedLM
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 confusion_matrix)

    dev = args.device if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet(args.windows, columns=["id", "label", "split", "seq"])
    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())
    n_cls = len(le.classes_)
    seqs = df["seq"].tolist()
    idx = np.where((df["split"] == args.split).to_numpy())[0]
    print(f"evaluating split='{args.split}'  n={len(idx)}  "
          f"classes={list(le.classes_)}")

    model = AutoModelForMaskedLM.from_pretrained(
        args.weights, trust_remote_code=True).float().to(dev)

    # Reconstruct the training-time numerical environment, not just the weights.
    if not args.no_fp32_attention:
        _mod = sys.modules.get(type(model).__module__)
        if _mod is not None and hasattr(_mod, "preferred_amp_dtype"):
            _mod.preferred_amp_dtype = lambda: torch.float32
            import sdpa_fp32
            sdpa_fp32.install()
            print("  [patch] attention forced to fp32 — matches training")
        else:
            print("  [patch] WARNING: preferred_amp_dtype not found!")
    else:
        print("  [patch] SKIPPED (--no-fp32-attention): results will NOT match "
              "the training-time protocol")

    model.backbone.pretrain = False
    model.backbone.for_representation = True
    for p in model.parameters():
        p.requires_grad_(False)
    n_inj = inject_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    model.to(dev)

    ckpt = torch.load(args.ckpt, map_location=dev)
    sd = model.state_dict()
    miss = 0
    for n, p in ckpt["lora"].items():
        if n in sd:
            sd[n].copy_(p.to(dev))
        else:
            miss += 1
    print(f"LoRA injected into {n_inj} projections, {len(ckpt['lora'])} adapter "
          f"tensors loaded (missing: {miss})")
    if miss:
        # A silent mismatch here evaluates a partly un-adapted model and still
        # produces a plausible number, so it is worth shouting about.
        print(f"  [WARNING] {miss} adapter tensors had no matching module. "
              "The evaluated model is NOT the trained one.")

    with torch.no_grad():
        hdim = model.backbone(encode(seqs[:2], dev))[0].shape[-1]
    head = nn.Linear(hdim, n_cls).to(dev)
    head.load_state_dict({k: v.to(dev) for k, v in ckpt["head"].items()})

    model.eval()
    head.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(idx), args.batch_size):
            b = idx[i:i + args.batch_size]
            base_ids = encode([seqs[j] for j in b], dev)
            emb, _ = model.backbone(base_ids)
            pooled = (emb.mean(dim=1) if args.pooling == "mean"
                      else emb.max(dim=1).values)
            preds.append(head(pooled.float()).argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    yy = y[idx]

    acc = accuracy_score(yy, p)
    f1 = f1_score(yy, p, average="macro")
    mcc = matthews_corrcoef(yy, p)
    cm = confusion_matrix(yy, p)
    print(f"\n=== {args.split.upper()} (checkpoint {os.path.basename(args.ckpt)}) ===")
    print(f"accuracy={acc:.4f}  macro_F1={f1:.4f}  MCC={mcc:.4f}")
    print("classes:", list(le.classes_))
    print("confusion matrix:\n", cm)

    out = os.path.join(os.path.dirname(args.ckpt), f"eval_{args.split}.json")
    with open(out, "w") as f:
        json.dump(dict(split=args.split, accuracy=float(acc), macro_f1=float(f1),
                       mcc=float(mcc), classes=list(le.classes_),
                       # recorded so audit_results.py can tell the two protocols
                       # apart instead of averaging them together
                       attention_precision=("bf16" if args.no_fp32_attention
                                            else "fp32"),
                       ckpt=args.ckpt,
                       confusion_matrix=cm.tolist()), f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
