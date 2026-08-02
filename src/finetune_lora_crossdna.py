#!/usr/bin/env python
"""
finetune_lora_crossdna.py — parameter-efficient (LoRA) fine-tuning of CrossDNA
for 4-way functional classification (CDS / UTR / intron / intergenic).

STRATEGY
  * Backbone FROZEN, base weights untouched in fp32. The numerical avalanche
    that killed full fine-tuning when the backbone was unfrozen cannot start
    here, because the backbone receives no gradients on its own weights.
  * LoRA adapters (r=8, alpha=16, dropout=0.05) on the q/k/v/o projections of
    both the Comba mixer and the sliding-window attention transformer, in BOTH
    branches (A and B) — 384 projections, 0.609% of parameters.
  * A linear classification head on the mean-pooled embedding.
  * Only the adapters and the head are trained. Gradient clipping throughout.

PRECISION
Attention is forced to fp32 for every run: `preferred_amp_dtype` is overridden
and flash_attn_func is replaced by the SDPA equivalent (flash-attn does not
accept fp32). This is applied UNCONDITIONALLY, not only where it is strictly
required, because protocol parity across checkpoints matters more than saving
time on the ones that happen to be stable in bf16. The from-scratch checkpoint
produces non-finite values without it.

    Anything that later re-evaluates these adapters MUST apply the same patch.
    Evaluating fp32-trained adapters on the native bf16 path cost 0.03 MCC on
    the 519M checkpoint — see the note in the README.

Reference points on the same windows: k-mer baseline MCC ~0.243; best zero-shot
probe 0.289.

Usage — smoke test first, then the full run:

    # smoke: 400 windows, 2 epochs -> the loss must be FINITE and decreasing
    python finetune_lora_crossdna.py --weights $PENGUIN_ROOT/weights/519M \\
       --windows $PENGUIN_ROOT/out/windows.parquet --outdir out/lora519_smoke \\
       --limit 400 --epochs 2 --batch-size 16

    # full run
    python finetune_lora_crossdna.py --weights $PENGUIN_ROOT/weights/519M \\
       --windows $PENGUIN_ROOT/out/windows.parquet --outdir out/lora519 \\
       --epochs 10 --batch-size 16 --lr-lora 1e-4 --lr-head 1e-3
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

# target projections: q/k/v/o of Comba and of the SWA transformer, both branches
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
            # lora_B stays at zero: the initial delta is exactly nought, so
            # training starts from the base model rather than from a random
            # perturbation of it.
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
    ap.add_argument("--outdir", default="out/lora519")
    ap.add_argument("--pooling", default="mean", choices=["mean", "max"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = everything; >0 smoke test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import torch
    import torch.nn as nn
    from transformers import AutoModelForMaskedLM
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 confusion_matrix)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"

    # ---- data ----
    df = pd.read_parquet(args.windows, columns=["id", "label", "split", "seq"])
    if args.limit:
        # class-balanced subsample for the smoke test
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.limit // df["label"].nunique()),
                               random_state=args.seed)
        ).reset_index(drop=True)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())
    n_cls = len(le.classes_)
    seqs = df["seq"].tolist()
    tr = np.where((df["split"] == "train").to_numpy())[0]
    va = np.where((df["split"] == "val").to_numpy())[0]
    te = np.where((df["split"] == "test").to_numpy())[0]
    print(f"windows={len(df)}  classes={list(le.classes_)}  "
          f"train={len(tr)} val={len(va)} test={len(te)}")

    # ---- model + LoRA ----
    print(f"Loading weights from: {args.weights}")
    model = AutoModelForMaskedLM.from_pretrained(
        args.weights, trust_remote_code=True).float().to(dev)

    # Force fp32 attention. Applied to EVERY run so that all checkpoints share
    # one protocol, not only the ones that need it.
    _mod = sys.modules.get(type(model).__module__)
    if _mod is not None and hasattr(_mod, "preferred_amp_dtype"):
        _mod.preferred_amp_dtype = lambda: torch.float32
        import sdpa_fp32
        sdpa_fp32.install()  # flash-attn cannot do fp32; SDPA replaces it
        print("  [patch] attention forced to fp32 (preferred_amp_dtype + sdpa_fp32)")
    else:
        print("  [patch] WARNING: preferred_amp_dtype not found!")

    model.backbone.pretrain = False
    model.backbone.for_representation = True
    for p in model.parameters():
        p.requires_grad_(False)
    n_inj = inject_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    model.to(dev)  # the freshly created adapters need moving to the GPU
    print(f"LoRA injected into {n_inj} projections (r={args.lora_r}, "
          f"alpha={args.lora_alpha}, dropout={args.lora_dropout})")

    # embedding width, read from a first forward pass
    with torch.no_grad():
        probe = model.backbone(encode(seqs[:2], dev))[0]
    hdim = probe.shape[-1]
    head = nn.Linear(hdim, n_cls).to(dev)
    print(f"hidden dim={hdim}  head Linear({hdim}->{n_cls})")

    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n]
    n_train = (sum(p.numel() for p in lora_params)
               + sum(p.numel() for p in head.parameters()))
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {n_train:,} of {n_tot:,} "
          f"({100*n_train/n_tot:.3f}%)")

    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": list(head.parameters()), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    lossf = nn.CrossEntropyLoss()

    def forward_logits(idx_batch):
        base_ids = encode([seqs[i] for i in idx_batch], dev)
        emb, _ = model.backbone(base_ids)  # [B, L, H]
        pooled = emb.mean(dim=1) if args.pooling == "mean" else emb.max(dim=1).values
        return head(pooled.float())

    def set_train_mode():
        # The backbone stays in eval mode so its own dropout and any
        # normalisation statistics remain deterministic; only the LoRA dropout
        # is switched back on. Calling model.train() here would perturb the
        # frozen model and make the adaptation non-reproducible.
        model.eval()
        for m in model.modules():
            if m.__class__.__name__ == "LoRALinear":
                m.train()
        head.train()

    def evaluate(idx):
        model.eval()
        head.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(idx), args.batch_size):
                b = idx[i:i + args.batch_size]
                preds.append(forward_logits(b).argmax(1).cpu().numpy())
        p = np.concatenate(preds)
        yy = y[idx]
        return (accuracy_score(yy, p), f1_score(yy, p, average="macro"),
                matthews_corrcoef(yy, p), p, yy)

    # ---- training, early stopping on validation MCC ----
    best_mcc, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(1, args.epochs + 1):
        set_train_mode()
        perm = np.random.permutation(tr)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            opt.zero_grad()
            logits = forward_logits(b)
            loss = lossf(logits, torch.from_numpy(y[b]).to(dev))
            if not torch.isfinite(loss):
                # Abort rather than continue: a non-finite loss means the
                # precision guard has failed, and anything produced afterwards
                # would be meaningless.
                print(f"  [WARN] non-finite loss at batch {nb}, epoch {ep} -> STOP")
                return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], args.grad_clip)
            opt.step()
            tot += float(loss)
            nb += 1
            if nb % 100 == 0:
                print(f"  ep{ep} batch {nb}/{len(perm)//args.batch_size}  "
                      f"loss={tot/nb:.4f}", flush=True)
        tr_loss = tot / max(nb, 1)
        v_acc, v_f1, v_mcc, _, _ = evaluate(va)
        hist.append(dict(epoch=ep, train_loss=tr_loss, val_acc=v_acc,
                         val_f1=v_f1, val_mcc=v_mcc))
        print(f"epoch {ep:2d}  train_loss={tr_loss:.4f}  val_MCC={v_mcc:.4f}  "
              f"val_F1={v_f1:.4f}  val_acc={v_acc:.4f}"
              + ("  *best" if v_mcc > best_mcc else ""), flush=True)
        if v_mcc > best_mcc:
            best_mcc = v_mcc
            # Only the adapters and the head are saved: the backbone is frozen,
            # so storing it would duplicate gigabytes of unchanged weights.
            best_state = {
                "lora": {n: p.detach().cpu().clone()
                         for n, p in model.named_parameters() if "lora_" in n},
                "head": {k: v.detach().cpu().clone()
                         for k, v in head.state_dict().items()},
            }
            torch.save(best_state, os.path.join(args.outdir, "best_lora.pt"))
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stopping (val_MCC flat for {args.patience} epochs)")
                break

    # ---- restore the best state and evaluate on TEST ----
    if best_state is not None:
        sd = model.state_dict()
        for n, p in best_state["lora"].items():
            sd[n].copy_(p.to(dev))
        head.load_state_dict({k: v.to(dev) for k, v in best_state["head"].items()})

    t_acc, t_f1, t_mcc, t_pred, t_true = evaluate(te)
    cm = confusion_matrix(t_true, t_pred)
    print("\n=== VERDICT (test set, best on val_MCC) ===")
    print(f"accuracy={t_acc:.4f}  macro_F1={t_f1:.4f}  MCC={t_mcc:.4f}")
    print("classes:", list(le.classes_))
    print("confusion matrix:\n", cm)

    with open(os.path.join(args.outdir, "lora_results.json"), "w") as f:
        json.dump(dict(
            weights=args.weights, pooling=args.pooling,
            attention_precision="fp32",  # recorded so eval can match it
            lora=dict(r=args.lora_r, alpha=args.lora_alpha,
                      dropout=args.lora_dropout, injected=n_inj),
            lr_lora=args.lr_lora, lr_head=args.lr_head,
            trainable_params=int(n_train), total_params=int(n_tot),
            best_val_mcc=float(best_mcc),
            test_accuracy=float(t_acc), test_macro_f1=float(t_f1),
            test_mcc=float(t_mcc), classes=list(le.classes_),
            confusion_matrix=cm.tolist(), history=hist,
        ), f, indent=2)
    print(f"\nWrote {args.outdir}/lora_results.json")


if __name__ == "__main__":
    main()
