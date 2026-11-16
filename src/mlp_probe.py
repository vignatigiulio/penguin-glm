#!/usr/bin/env python
"""
mlp_probe.py — supervised MLP probe on frozen embeddings.

The backbone is never touched: this measures how much functional information is
linearly-plus-a-little recoverable from the zero-shot representation. Its
hyperparameters are part of the result and must be held fixed across models.

This is probe CONFIGURATION 2: two hidden layers of 256 with ReLU and dropout
0.3, AdamW, up to 60 epochs, early stopping on validation MCC (patience 8),
used for the GC-attractor analyses (stem sweep and step-1 run). The zero-shot
comparisons between models use mlp_probe_compare.py (CONFIGURATION 1: dropout
0.1, 30 epochs, patience 5). The two configurations give different numbers on
the same embeddings and must not be mixed.

MCC rather than accuracy is the headline metric because the four classes are
balanced by construction here but the confusion structure is not: MCC penalises
a model that wins on one easy class and collapses the other three.

The scaler is fitted on TRAIN only, and the checkpoint restored for the final
TEST evaluation is the one that maximised VAL MCC — the test split is touched
exactly once, at the end.

Usage:
    python mlp_probe.py --windows windows.parquet --emb emb.npz --outdir out/probe
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from common import load_embeddings, align_to_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--emb", required=True, help=".npz embedding file (ids/X/meta)")
    ap.add_argument("--outdir", default="out/mlp")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 confusion_matrix)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- data and splits (the same chromosome-disjoint scheme used elsewhere) ----
    df = pd.read_parquet(args.windows, columns=["id", "label", "split"])
    ids = df["id"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())
    n_cls = len(le.classes_)
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "val").to_numpy()
    te = (df["split"] == "test").to_numpy()

    # ---- embeddings aligned by id, standardised on TRAIN ONLY ----
    eids, X, meta = load_embeddings(args.emb)
    X = align_to_ids(eids, X, ids).astype(np.float32)
    scaler = StandardScaler().fit(X[tr])
    X = scaler.transform(X).astype(np.float32)
    print(f"emb dim={X.shape[1]}  classes={list(le.classes_)}  "
          f"train={tr.sum()} val={va.sum()} test={te.sum()}")

    dev = args.device if torch.cuda.is_available() else "cpu"
    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).long().to(dev)
    tr_i = torch.from_numpy(np.where(tr)[0]).to(dev)
    va_i = torch.from_numpy(np.where(va)[0]).to(dev)
    te_i = torch.from_numpy(np.where(te)[0]).to(dev)

    # ---- probe head: two hidden layers, ReLU + dropout ----
    net = nn.Sequential(
        nn.Linear(X.shape[1], args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(args.hidden, args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(args.hidden, n_cls),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    lossf = nn.CrossEntropyLoss()

    def evaluate(idx):
        net.eval()
        with torch.inference_mode():
            pred = net(Xt[idx]).argmax(1).cpu().numpy()
        yy = yt[idx].cpu().numpy()
        return (accuracy_score(yy, pred), f1_score(yy, pred, average="macro"),
                matthews_corrcoef(yy, pred), pred, yy)

    # ---- training loop, early stopping on validation MCC ----
    best_mcc, best_state, bad = -1.0, None, 0
    hist = []
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = tr_i[torch.randperm(len(tr_i), device=dev)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = lossf(net(Xt[b]), yt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        tr_loss = tot / max(nb, 1)

        v_acc, v_f1, v_mcc, _, _ = evaluate(va_i)
        hist.append(dict(epoch=ep, train_loss=tr_loss, val_acc=v_acc,
                         val_f1=v_f1, val_mcc=v_mcc))
        print(f"epoch {ep:2d}  train_loss={tr_loss:.4f}  val_MCC={v_mcc:.4f}  "
              f"val_F1={v_f1:.4f}" + ("  *best" if v_mcc > best_mcc else ""))
        if v_mcc > best_mcc:
            best_mcc = v_mcc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
            torch.save(best_state, os.path.join(args.outdir, "best_model.pt"))
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stopping (val_MCC flat for {args.patience} epochs)")
                break

    # ---- final TEST evaluation with the best checkpoint ----
    if best_state is not None:
        net.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    t_acc, t_f1, t_mcc, t_pred, t_true = evaluate(te_i)
    cm = confusion_matrix(t_true, t_pred)

    print("\n=== VERDICT (test set, checkpoint selected on val_MCC) ===")
    print(f"accuracy={t_acc:.4f}  macro_F1={t_f1:.4f}  MCC={t_mcc:.4f}")
    print("classes:", list(le.classes_))
    print("confusion matrix:\n", cm)

    results = dict(
        emb_file=args.emb, emb_meta=str(meta), dim=int(X.shape[1]),
        hidden=args.hidden, dropout=args.dropout, epochs_run=len(hist),
        best_val_mcc=float(best_mcc),
        test_accuracy=float(t_acc), test_macro_f1=float(t_f1), test_mcc=float(t_mcc),
        classes=list(le.classes_), confusion_matrix=cm.tolist(),
    )
    with open(os.path.join(args.outdir, "mlp_results.json"), "w") as f:
        json.dump(dict(results=results, history=hist), f, indent=2)

    pd.DataFrame(hist).to_csv(os.path.join(args.outdir, "training_history.csv"),
                              index=False)
    print(f"\nWrote {args.outdir}/mlp_results.json and training_history.csv")


if __name__ == "__main__":
    main()
