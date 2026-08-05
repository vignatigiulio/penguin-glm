#!/usr/bin/env python
"""
extract_janusdna.py — extract frozen JanusDNA embeddings for every window.

Output is the canonical .npz (ids + X + meta) consumed by everything downstream,
identical in format to the CrossDNA extractor so the two models stay comparable.

STEM-AWARE BY CONSTRUCTION. When the model carries a causal convolutional stem
(see stem/), the hidden states come out at super-token resolution M = L / S while
the tokenizer's attention mask is still at nucleotide resolution L. Pooling one
against the other would silently misalign every window. The mask is therefore
compressed to match whenever the two lengths differ — a super-token counts as
valid if its block of S nucleotides contains at least one non-padding position.
When there is no stem the two lengths are equal and the code path is bit-for-bit
the one it always was, so no flag is needed to switch between the two cases.

Default is `add_special_tokens=False`: the sequence is fed bare. The pooled mean
would otherwise include a [SEP] position that carries no genomic content.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from tqdm import tqdm

from common import save_embeddings

# runtime patches live in env/patches/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "env" / "patches"))


def load_model_and_tokenizer(repo_path: Path, model_config_path: Path,
                             checkpoint_path: Path, device: str):
    # The JanusDNA repo is not an installed package: put its root on the path so
    # janusdna.* and caduceus.* import the same way train.py does.
    sys.path.insert(0, str(repo_path))

    from janusdna.configuration_janusdna import JanusDNAConfig
    from janusdna.modeling_janusdna import JanusDNAModel
    from caduceus.tokenization_caduceus import CaduceusTokenizer

    import flexattn_blocksize as _flexattn
    _flexattn.apply()

    with open(model_config_path) as f:
        model_cfg_dict = json.load(f)

    # model_config.json has the shape {"_name_": ..., "config": {...}, "config_path": ...}
    # (see src/utils/train.py: save_cfg -> model_config.json)
    if "config" not in model_cfg_dict:
        raise ValueError(
            f"'{model_config_path}' has no 'config' key. Check that this really is "
            "the model_config.json written by the pre-training run "
            "(inside outputs/pretrain/<...>/<run_name>/)."
        )
    config = JanusDNAConfig(**model_cfg_dict["config"])

    print(f"[config] hidden_size={config.hidden_size} "
          f"num_hidden_layers={config.num_hidden_layers} "
          f"num_experts={config.num_experts} vocab_size={config.vocab_size}")

    model = JanusDNAModel(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Lightning checkpoints carry keys like "model.model.layers.0...."
    # (SequenceLightningModule.model = DNAEmbeddingModelJanusDNA; .model.model =
    # JanusDNAModel), so the prefix is stripped twice to reach the bare keys.
    consume_prefix_in_state_dict_if_present(state_dict, "model.")
    consume_prefix_in_state_dict_if_present(state_dict, "model.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if "head" not in k and "decoder" not in k]
    if real_missing:
        print(f"[WARNING] {len(real_missing)} missing keys not explained by head/decoder:")
        for k in real_missing[:20]:
            print("   ", k)
        if unexpected:
            print(f"[debug] {len(unexpected)} unexpected keys in the checkpoint "
                  "(possible name mismatch):")
            for k in unexpected[:20]:
                print("   ", k)
    print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)} "
          "(missing 'head'/'decoder' is expected: those are task heads absent "
          "from JanusDNAModel)")

    model.to(device)
    model.eval()

    tokenizer = CaduceusTokenizer(
        model_max_length=config.max_position_embeddings,
        characters=("A", "C", "G", "T", "N"),
    )

    return model, tokenizer, config


def stem_pad_target(L: int, S: int, forced: int = 0) -> int:
    """Length the input must be padded to when a stem of stride S is active.

    Two constraints, not one:
      * L must be a multiple of S, or the stem's assertion fires (1000 is not a
        multiple of 16 or 32);
      * the compressed grid M = L/S must satisfy M >= 64, because JanusDNA's
        final_attention sets q_len = 2M and FlexAttention's default BLOCK_SIZE
        is 128.

    Hence max(64*S, ceil(L/S)*S). The padding is NOT free: it was validated by
    measuring its effect on the downstream metrics, not assumed harmless.
    At 2.4% (S=16, 1000 -> 1024) NMI vs GC is unchanged to four decimals; at 51%
    (S=32, 1000 -> 2048) it moves by 5.1%, comparable to the effect under study,
    which is why S=32 is excluded rather than padded.
    """
    return forced if forced > 0 else max(64 * S, ((L + S - 1) // S) * S)


@torch.no_grad()
def embed_batch(model, tokenizer, seqs, device, add_special_tokens=False,
                stem_stride=1, pad_to=0):
    enc = tokenizer(
        seqs,
        add_special_tokens=add_special_tokens,
        padding=True,
        return_tensors="pt",
        # CaduceusTokenizer declares model_input_names=["input_ids"]; without
        # this explicit flag no attention_mask comes back at all.
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"].to(device)
    attn_in = enc["attention_mask"].to(device)

    if stem_stride > 1:
        L = input_ids.size(1)
        target = stem_pad_target(L, stem_stride, pad_to)
        if target > L:
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = 0
            # Padded on the RIGHT, and marked invalid in the mask, so the added
            # positions never enter the mean pooling.
            input_ids = torch.nn.functional.pad(input_ids, (0, target - L),
                                                value=pad_id)
            attn_in = torch.nn.functional.pad(attn_in, (0, target - L), value=0)
    enc = {"input_ids": input_ids, "attention_mask": attn_in}

    # hidden_states: (B, L, hidden_size) — already fused bidirectionally by
    # JanusDNAModel, so no separate forward/backward handling is needed here.
    hidden_states, moe_loss = model(input_ids, return_dict=False)

    attn_mask_in = enc["attention_mask"]  # (B, L) at INPUT resolution

    L_in = attn_mask_in.shape[1]
    L_h = hidden_states.shape[1]
    if L_h != L_in:
        # Stem present: compress the mask from nucleotide to super-token
        # resolution. A super-token is valid if its block of S nucleotides
        # contains at least one non-padding position.
        assert L_in % L_h == 0, (
            f"input length ({L_in}) is not a multiple of hidden length ({L_h}): "
            "stem stride is inconsistent with the sequence length"
        )
        S = L_in // L_h
        m = attn_mask_in.reshape(attn_mask_in.shape[0], L_h, S)
        attn_mask = (m.sum(dim=2) > 0).to(hidden_states.dtype).unsqueeze(-1)  # (B, M, 1)
    else:
        attn_mask = attn_mask_in.unsqueeze(-1)  # (B, L, 1) — no stem

    summed = (hidden_states * attn_mask).sum(dim=1)
    counts = attn_mask.sum(dim=1).clamp(min=1)
    mean_pooled = summed / counts  # (B, hidden_size)

    return mean_pooled.float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-path", type=Path, required=True,
                        help="JanusDNA repository root (or $JANUSDNA_REPO)")
    parser.add_argument("--model-config", type=Path, required=True,
                        help="path to model_config.json")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to the Lightning checkpoint (last.ckpt)")
    parser.add_argument("--windows-parquet", type=Path, required=True,
                        help="parquet/csv with at least the columns id, seq")
    parser.add_argument("--seq-col", type=str, default="seq")
    parser.add_argument("--output", type=Path, required=True, help="output .npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--add-special-tokens", action="store_true",
                        help="append the trailing [SEP] (default: bare sequence)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N windows (smoke test)")
    parser.add_argument("--stem-stride", type=int, default=None,
                        help="stem stride S. Defaults to $JANUS_STEM_STRIDE when "
                             "$JANUS_STEM=1, otherwise 1 (no stem).")
    parser.add_argument("--pad-to", type=int, default=0,
                        help="force the padded length instead of computing it")
    args = parser.parse_args()

    if args.stem_stride is None:
        args.stem_stride = (int(os.environ.get("JANUS_STEM_STRIDE", "1"))
                            if os.environ.get("JANUS_STEM") == "1" else 1)
    if args.stem_stride > 1:
        print(f"[stem] stride S={args.stem_stride}, inputs padded to "
              f"{stem_pad_target(1000, args.stem_stride, args.pad_to)} for L=1000")

    model, tokenizer, config = load_model_and_tokenizer(
        args.repo_path, args.model_config, args.checkpoint, args.device
    )

    if args.windows_parquet.suffix == ".parquet":
        df = pd.read_parquet(args.windows_parquet)
    else:
        df = pd.read_csv(args.windows_parquet)

    if args.seq_col not in df.columns:
        raise ValueError(f"Column '{args.seq_col}' not found. Available: {list(df.columns)}")
    if "id" not in df.columns:
        raise ValueError(
            "Column 'id' missing from the parquet: it is required for downstream "
            f"alignment (align_to_ids in common.py). Available: {list(df.columns)}"
        )

    if args.limit is not None:
        df = df.iloc[:args.limit].reset_index(drop=True)
        print(f"[smoke test] limited to {len(df)} windows")

    seqs = df[args.seq_col].tolist()
    all_embeddings = []

    for i in tqdm(range(0, len(seqs), args.batch_size), desc="JanusDNA embedding"):
        batch_seqs = seqs[i:i + args.batch_size]
        emb = embed_batch(model, tokenizer, batch_seqs, args.device,
                          args.add_special_tokens,
                          stem_stride=args.stem_stride, pad_to=args.pad_to)
        all_embeddings.append(emb)

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"[output] embeddings shape={embeddings.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": "janusdna",
        "checkpoint": str(args.checkpoint),
        "model_config": str(args.model_config),
        "hidden_size": int(embeddings.shape[1]),
        "n_windows": int(embeddings.shape[0]),
        "pooling": "mean",
        "add_special_tokens": bool(args.add_special_tokens),
        "stem_stride": int(args.stem_stride),
    }
    save_embeddings(str(args.output), df["id"].to_numpy().astype(str),
                    embeddings.astype(np.float32), meta)

    var = float(np.mean(np.var(embeddings, axis=0)))
    print(f"[output] saved to {args.output} (canonical format: ids + X + meta)")
    print(f"[check] mean embedding variance: {var:.4e} "
          f"({'ok' if var > 1e-6 else 'SUSPICIOUS, near zero'})")


if __name__ == "__main__":
    main()
