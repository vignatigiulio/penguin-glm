#!/usr/bin/env python
"""
finetune_lora_janusdna.py — LoRA fine-tuning of JanusDNA for 4-way functional
classification (CDS / UTR / intron / intergenic).

The CrossDNA script adapted to the JanusDNA architecture. Every difference below
was verified against the upstream repository rather than assumed:

  1. MODEL LOADING. JanusDNA is not an AutoModelForMaskedLM. The backbone is a
     JanusDNAModel loaded from a Lightning .ckpt with the "model." prefix
     stripped TWICE, exactly as in extract_janusdna.py. The forward returns
     (hidden_states, moe_loss) under return_dict=False.

  2. TOKENIZER. CaduceusTokenizer maps A/C/G/T/N to 7/8/9/10/11. Reusing
     CrossDNA's 0-4 dictionary would feed the model wrong token ids while
     failing silently.

  3. LoRA TARGETS. The 144 is almost entirely Mamba — every block is a mamba
     block, and the only attention is final_attention plus the mid-attention.
     Targeting q/k/v/o alone, as on CrossDNA, would barely touch the model.
     So the targets are:
       - Mamba projections: mamba_fwd.{in_proj, out_proj, x_proj}
         FORWARD ONLY: mamba_rev shares weights through
         bidirectional_weight_tie=True, so adapting fwd adapts both directions.
       - Attention projections: self_attn.{q,k,v,o}_proj

     dt_proj is DELIBERATELY EXCLUDED. Empirically (tools/verify_lora_targets.py),
     in_proj/out_proj/x_proj are honoured by Mamba's CUDA fast path — a LoRA
     delta changes the output — but dt_proj is manipulated by the kernel
     directly: cuda_kernels_forward sets `self.dt_proj.bias = None` and hands
     the weights to the fused kernel, which makes an adapter on it both
     incompatible and opaque in effect. Better to adapt only the projections
     verified to work than to ship one that silently does nothing.

  4. POOLING uses the real attention mask from the tokenizer.

STEM-AWARE. When the checkpoint carries a causal convolutional stem, hidden
states come out at super-token resolution M = L/S while the mask is still at
nucleotide resolution. The mask is compressed whenever the two differ; when
there is no stem the code path is unchanged. No flag is needed.

Numerical stability follows the CrossDNA script: frozen backbone, fp32 adapters,
gradient clipping, a finiteness guard on the loss, and a train-minus-val MCC gap
reported every epoch as an overfitting warning.

Usage:
    # SMOKE TEST first (400 windows, 2 epochs -> finite, decreasing loss)
    python finetune_lora_janusdna.py \\
      --repo-path $JANUSDNA_REPO \\
      --model-config <path>/model_config.json \\
      --checkpoint  <path>/checkpoints/last.ckpt \\
      --windows out/windows.parquet --outdir out/lora_janus144_smoke \\
      --limit 400 --epochs 2 --batch-size 16

    # full run
    python finetune_lora_janusdna.py \\
      --repo-path $JANUSDNA_REPO \\
      --model-config <path>/model_config.json \\
      --checkpoint  <path>/checkpoints/last.ckpt \\
      --windows out/windows.parquet --outdir out/lora_janus144 \\
      --epochs 10 --batch-size 16 --lr-lora 1e-4 --lr-head 1e-3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

# LoRA target suffixes for JanusDNA. Selection is by .endswith(), so
# "mamba_fwd.in_proj" matches the forward direction only, never mamba_rev.
LORA_TARGET_SUFFIXES = (
    # Mamba projections verified as effective (fwd only: rev is weight-tied)
    "mamba_fwd.in_proj", "mamba_fwd.out_proj", "mamba_fwd.x_proj",
    # attention projections (final_attention + mid-attention)
    "self_attn.q_proj", "self_attn.k_proj",
    "self_attn.v_proj", "self_attn.o_proj",
)


def load_backbone_and_tokenizer(repo_path, model_config_path, checkpoint_path, device):
    """Load JanusDNAModel + CaduceusTokenizer. Identical to extract_janusdna.py."""
    import torch
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

    sys.path.insert(0, str(repo_path))
    from janusdna.configuration_janusdna import JanusDNAConfig
    from janusdna.modeling_janusdna import JanusDNAModel
    from caduceus.tokenization_caduceus import CaduceusTokenizer

    with open(model_config_path) as f:
        model_cfg_dict = json.load(f)
    if "config" not in model_cfg_dict:
        raise ValueError(f"'{model_config_path}' has no 'config' key.")
    config = JanusDNAConfig(**model_cfg_dict["config"])
    print(f"[config] hidden_size={config.hidden_size} "
          f"num_hidden_layers={config.num_hidden_layers} "
          f"num_experts={config.num_experts}")

    model = JanusDNAModel(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    consume_prefix_in_state_dict_if_present(state_dict, "model.")
    consume_prefix_in_state_dict_if_present(state_dict, "model.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if "head" not in k and "decoder" not in k]
    if real_missing:
        print(f"[WARNING] {len(real_missing)} missing keys not explained by head/decoder:")
        for k in real_missing[:20]:
            print("   ", k)
    print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device)
    # Disable gradient checkpointing: the config enables it for pre-training,
    # but under LoRA with a frozen backbone it triggers "None of the inputs have
    # requires_grad" and complicates gradient flow to the adapters. Memory is
    # not the binding constraint here.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.gradient_checkpointing = False
    tokenizer = CaduceusTokenizer(
        model_max_length=config.max_position_embeddings,
        characters=("A", "C", "G", "T", "N"),
    )
    return model, tokenizer, config


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
            # lora_B at zero: initial delta is nought, so training starts from
            # the base model rather than a random perturbation of it
            self.drop = nn.Dropout(dropout)

        # --- attribute transparency ---
        # JanusDNA's Mamba mixer reaches into proj.weight / proj.bias directly
        # (e.g. x_proj.weight.device in the fast path). These properties forward
        # to the wrapped module so the backbone keeps working unchanged; without
        # them the injection breaks the model rather than adapting it.
        @property
        def weight(self):
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        @property
        def in_features(self):
            return self.base.in_features

        @property
        def out_features(self):
            return self.base.out_features

        def forward(self, x):
            out = self.base(x)
            lora = (self.drop(x) @ self.lora_A.t()) @ self.lora_B.t()
            return out + self.scaling * lora.to(out.dtype)

    return LoRALinear(base, r, alpha, dropout)


def inject_lora(model, r, alpha, dropout):
    import torch.nn as nn

    injected, names = 0, []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and full.endswith(LORA_TARGET_SUFFIXES):
                setattr(mod, child_name, build_lora_linear(child, r, alpha, dropout))
                injected += 1
                names.append(full)
    return injected, names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-path", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--outdir", default="out/lora_janus144")
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
    ap.add_argument("--resume", default="",
                    help="best_lora.pt checkpoint to restart from")
    ap.add_argument("--stem-stride", type=int, default=None,
                    help="stem stride S. Defaults to $JANUS_STEM_STRIDE when "
                         "$JANUS_STEM=1, otherwise 1 (no stem).")
    ap.add_argument("--pad-to", type=int, default=0,
                    help="force the padded length instead of computing it")
    ap.add_argument("--overfit-gap", type=float, default=0.08,
                    help="warn when train_MCC - val_MCC exceeds this")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import torch
    import torch.nn as nn
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 confusion_matrix)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"

    if args.stem_stride is None:
        args.stem_stride = (int(os.environ.get("JANUS_STEM_STRIDE", "1"))
                            if os.environ.get("JANUS_STEM") == "1" else 1)
    if args.stem_stride > 1:
        print(f"[stem] stride S={args.stem_stride}")

    # ---- data ----
    # Read the whole parquet and select columns in pandas. Projection at the
    # pyarrow level (read_parquet(columns=[...])) raises a spurious KeyError on
    # this particular file, while reading it whole works.
    df = pd.read_parquet(args.windows)
    df = df[["id", "label", "split", "seq"]]
    if args.limit:
        # Class-balanced subsample for the smoke test.
        # groupby(...).sample(...) directly, NOT groupby(...).apply(lambda g:
        # g.sample(...)): in pandas 3.x the latter moves the grouping column
        # ('label') into the index, so it vanishes from the columns and raises a
        # KeyError further down.
        per_class = max(1, args.limit // df["label"].nunique())
        df = (df.groupby("label", group_keys=False)
                .sample(n=per_class, random_state=args.seed, replace=False)
                .reset_index(drop=True))
    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())
    n_cls = len(le.classes_)
    seqs = df["seq"].tolist()
    tr = np.where((df["split"] == "train").to_numpy())[0]
    va = np.where((df["split"] == "val").to_numpy())[0]
    te = np.where((df["split"] == "test").to_numpy())[0]
    print(f"windows={len(df)}  classes={list(le.classes_)}  "
          f"train={len(tr)} val={len(va)} test={len(te)}")

    # ---- model + tokenizer + LoRA ----
    model, tokenizer, config = load_backbone_and_tokenizer(
        args.repo_path, args.model_config, args.checkpoint, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    n_inj, inj_names = inject_lora(model, args.lora_r, args.lora_alpha,
                                   args.lora_dropout)
    model.to(dev)  # move the newly created LoRA parameters onto the device
    print(f"LoRA injected into {n_inj} projections (r={args.lora_r}, "
          f"alpha={args.lora_alpha}, dropout={args.lora_dropout})")
    if n_inj == 0:
        raise RuntimeError(
            "No LoRA module was injected: the target suffixes do not match any "
            "real module name. Check LORA_TARGET_SUFFIXES."
        )
    print("[LoRA targets — first 12]")
    for nm in inj_names[:12]:
        print("   ", nm)

    def stem_pad_target(L, S, forced=0):
        """max(64*S, ceil(L/S)*S): the stem needs L divisible by S, and
        FlexAttention needs M = L/S >= 64 because q_len = 2M against a default
        BLOCK_SIZE of 128."""
        return forced if forced > 0 else max(64 * S, ((L + S - 1) // S) * S)

    def encode(batch_seqs):
        # The padding lives HERE, not in forward_logits, because encode() is
        # also used by the probe that determines hdim: putting it downstream
        # leaves that first call unpadded and it fails on S=16.
        enc = tokenizer(batch_seqs, add_special_tokens=False, padding=True,
                        return_tensors="pt", return_attention_mask=True)
        ids = enc["input_ids"].to(dev)
        am = enc["attention_mask"].to(dev)
        if args.stem_stride > 1:
            L = ids.size(1)
            target = stem_pad_target(L, args.stem_stride, args.pad_to)
            if target > L:
                pad_id = getattr(tokenizer, "pad_token_id", None)
                if pad_id is None:
                    pad_id = 0
                # right padding, marked invalid so it never enters the pooling
                ids = torch.nn.functional.pad(ids, (0, target - L), value=pad_id)
                am = torch.nn.functional.pad(am, (0, target - L), value=0)
        return ids, am

    # embedding width from a first forward pass
    with torch.no_grad():
        ids0, am0 = encode(seqs[:2])
        h0, _ = model(ids0, return_dict=False)
    hdim = h0.shape[-1]
    head = nn.Linear(hdim, n_cls).to(dev)
    print(f"hidden dim={hdim}  head Linear({hdim}->{n_cls})")

    # ---- optional resume ----
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev)
        sd = model.state_dict()
        for n, p in ck["lora"].items():
            if n in sd:
                sd[n].copy_(p.to(dev))
        head.load_state_dict({k: v.to(dev) for k, v in ck["head"].items()})
        print(f"RESUME: adapters + head loaded from {args.resume}")

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
        ids, am = encode([seqs[i] for i in idx_batch])
        emb, _ = model(ids, return_dict=False)  # [B, L or M, H]

        # STEM-AWARE: when a stem is present emb is at super-token resolution
        # M = L/S, so the nucleotide-resolution mask has to be compressed. A
        # super-token counts as valid if its block contains any real position.
        L_in = am.shape[1]
        L_h = emb.shape[1]
        if L_h != L_in:
            assert L_in % L_h == 0, (
                f"input length ({L_in}) is not a multiple of embedding length "
                f"({L_h}): stem stride is inconsistent"
            )
            S = L_in // L_h
            am_c = (am.reshape(am.shape[0], L_h, S).sum(dim=2) > 0).to(emb.dtype)
            m = am_c.unsqueeze(-1)
        else:
            m = am.unsqueeze(-1).to(emb.dtype)

        if args.pooling == "mean":
            pooled = (emb * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            neg = torch.finfo(emb.dtype).min
            pooled = emb.masked_fill(m == 0, neg).max(dim=1).values
        return head(pooled.float())

    def set_train_mode():
        model.eval()  # backbone deterministic
        for mod in model.modules():
            if mod.__class__.__name__ == "LoRALinear":
                mod.train()  # re-enable the LoRA dropout only
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
        # train MCC on a subsample the same size as val, so the gap between the
        # two is a like-for-like overfitting signal rather than a size artefact
        tr_sample = np.random.choice(tr, size=min(len(va), len(tr)), replace=False)
        _, _, tr_mcc, _, _ = evaluate(tr_sample)
        gap = tr_mcc - v_mcc
        flag = "  <-- OVERFIT?" if gap > args.overfit_gap else ""
        hist.append(dict(epoch=ep, train_loss=tr_loss, train_mcc=tr_mcc,
                         val_acc=v_acc, val_f1=v_f1, val_mcc=v_mcc, gap=gap))
        print(f"epoch {ep:2d}  train_loss={tr_loss:.4f}  train_MCC={tr_mcc:.4f}  "
              f"val_MCC={v_mcc:.4f}  gap={gap:+.4f}{flag}"
              + ("  *best" if v_mcc > best_mcc else ""), flush=True)
        if v_mcc > best_mcc:
            best_mcc = v_mcc
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
            model="janusdna", checkpoint=args.checkpoint,
            pooling=args.pooling,
            stem_stride=int(args.stem_stride),
            lora=dict(r=args.lora_r, alpha=args.lora_alpha,
                      dropout=args.lora_dropout, injected=n_inj,
                      target_suffixes=list(LORA_TARGET_SUFFIXES)),
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
