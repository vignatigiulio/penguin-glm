#!/usr/bin/env python
"""
mlp_probe_compare.py — the same MLP probe across several embedding sets at once.

Model-agnostic: it accepts any .npz in the canonical format (ids + X + meta), so
CrossDNA and JanusDNA go through a single code path on identical windows and an
identical split. That symmetry is the point. Comparing a number obtained here
against one obtained from a separately written script invites exactly the kind
of protocol mismatch that `analysis/audit_results.py` exists to detect.

Note the difference from mlp_probe.py: this variant (probe CONFIGURATION 1,
used for the zero-shot comparisons between models) uses dropout 0.1, 30 epochs and
patience 5, against 0.3 / 60 / 8 of configuration 2. The two are NOT
interchangeable and their outputs must not be pooled: use one or the other for a
given comparison.

Usage:
    python mlp_probe_compare.py --windows windows.parquet \\
        --emb janus144=emb_janus.npz --emb crossdna81=emb_crossdna.npz \\
        --outdir out/probe_compare
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
    ap.add_argument("--emb", action="append", required=True,
                    help="name=file.npz (repeatable)")
    ap.add_argument("--outdir", default="out/mlp_probe_compare")
    ap.add_argument("--hidden", type=int, default=256, help="MLP hidden width")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5,
                    help="early stopping on validation MCC")
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
    dev = args.device if torch.cuda.is_available() else "cpu"

    # Read the whole file and subset afterwards: pyarrow column projection can
    # fail on this parquet with "Repetition level histogram size mismatch".
    df = pd.read_parquet(args.windows)
    df = df[["id", "label", "split"]]
    ids = df["id"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())
    n_cls = len(le.classes_)
    is_tr = (df["split"] == "train").to_numpy()
    is_va = (df["split"] == "val").to_numpy()
    is_te = (df["split"] == "test").to_numpy()
    print(f"windows={len(ids)}  classes={list(le.classes_)}  "
          f"train={is_tr.sum()} val={is_va.sum()} test={is_te.sum()}")

    class MLP(nn.Module):
        def __init__(self, d_in, d_hid, d_out, p):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(p),
                nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Dropout(p),
                nn.Linear(d_hid, d_out),
            )

        def forward(self, x):
            return self.net(x)

    def train_probe(X, name):
        # scaler fitted on TRAIN only — no leakage
        scaler = StandardScaler().fit(X[is_tr])
        Xs = scaler.transform(X).astype(np.float32)
        Xtr = torch.from_numpy(Xs[is_tr]).to(dev)
        Xva = torch.from_numpy(Xs[is_va]).to(dev)
        Xte = torch.from_numpy(Xs[is_te]).to(dev)
        ytr = torch.from_numpy(y[is_tr]).to(dev)
        yva_np, yte_np = y[is_va], y[is_te]

        net = MLP(X.shape[1], args.hidden, n_cls, args.dropout).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        lossf = nn.CrossEntropyLoss()

        def eval_split(Xsplit, ynp):
            net.eval()
            with torch.no_grad():
                pred = net(Xsplit).argmax(1).cpu().numpy()
            return (matthews_corrcoef(ynp, pred), pred)

        best_mcc, best_state, bad = -1.0, None, 0
        n = Xtr.shape[0]
        for ep in range(1, args.epochs + 1):
            net.train()
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, args.batch_size):
                b = perm[i:i + args.batch_size]
                opt.zero_grad()
                loss = lossf(net(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
            v_mcc, _ = eval_split(Xva, yva_np)
            if v_mcc > best_mcc:
                best_mcc = v_mcc
                best_state = {k: v.detach().clone()
                              for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    break

        net.load_state_dict(best_state)
        t_mcc, t_pred = eval_split(Xte, yte_np)
        acc = accuracy_score(yte_np, t_pred)
        f1 = f1_score(yte_np, t_pred, average="macro")
        cm = confusion_matrix(yte_np, t_pred)
        np.savetxt(os.path.join(args.outdir, f"cm_{name}_mlpprobe.csv"),
                   cm, fmt="%d", delimiter=",",
                   header=",".join(le.classes_), comments="")
        return dict(feature=name, dim=int(X.shape[1]), best_val_mcc=float(best_mcc),
                    accuracy=float(acc), macro_f1=float(f1), mcc=float(t_mcc)), cm

    results = []
    for spec in args.emb:
        name, path = spec.split("=", 1)
        eids, X, meta = load_embeddings(path)
        X = align_to_ids(eids, X, ids).astype(np.float32)
        row, cm = train_probe(X, name)
        results.append(row)
        print(f"[{name:16s}] dim={row['dim']:4d}  val_MCC={row['best_val_mcc']:.4f}  "
              f"test: acc={row['accuracy']:.4f} F1={row['macro_f1']:.4f} "
              f"MCC={row['mcc']:.4f}")

    res = pd.DataFrame(results).sort_values("mcc", ascending=False)
    res.to_csv(os.path.join(args.outdir, "mlp_probe_results.csv"), index=False)
    with open(os.path.join(args.outdir, "mlp_probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== ZERO-SHOT MLP PROBE (descending MCC) ===")
    print(res.to_string(index=False))
    print("\nreference points on the same windows: k-mer RF 0.243 | "
          "k-mer MLP (this script, 4 seeds) 0.301 | "
          "CrossDNA-519M 0.288 | CrossDNA-8.1M 0.274")
    print(f"\nResults in {args.outdir}/")


if __name__ == "__main__":
    main()
