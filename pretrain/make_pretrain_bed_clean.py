#!/usr/bin/env python3
"""
make_pretrain_bed_clean.py — JanusDNA pre-training BED with the downstream test
regions removed.

THE PROBLEM THIS SOLVES
An early version of the pre-training corpus tiled the whole genome. That corpus
included the scaffolds on which the downstream task is evaluated: 98.3% of test
windows sat on scaffolds the model had already seen during pre-training. The
model never saw the LABELS, but it had modelled the sequence, and any
"from-scratch beats human-pretrained" claim built on that comparison would be
measuring memorisation rather than domain adaptation.

Difference from make_pretrain_bed.py: this reads windows.parquet, finds the
scaffolds used by the downstream 'val' and 'test' splits, and EXCLUDES them
entirely. The pre-training corpus is drawn only from scaffolds belonging to the
downstream train split.

The pre-training's own valid/test partitions — used purely to monitor perplexity
— are then carved out of what remains.

Both corpora were kept and carried through the full evaluation, so the artifact
could be quantified rather than merely fixed: contamination is worth +0.020 MCC,
of which about +0.012 is genuine in-domain signal.

Usage:
    python make_pretrain_bed_clean.py \\
        --fasta genome.fna --windows out/windows.parquet \\
        --out data/penguin/penguin_pretrain_clean.bed --seqlen 131072
"""
import argparse

import pandas as pd
from pyfaidx import Fasta

MAX_ALLOWED_LENGTH = 2 ** 20  # hard-coded in the dataloader


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--windows", required=True,
                    help="windows.parquet from the downstream task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqlen", type=int, default=131072)
    ap.add_argument("--tile-step", type=int, default=None,
                    help="tiling step. JanusDNA: 1048576 (2^20, default). "
                         "CrossDNA/HyenaDNA: set equal to seqlen, so that one "
                         "BED row is one training example.")
    ap.add_argument("--scaffold-col", default="scaffold")
    ap.add_argument("--exclude-splits", default="val,test",
                    help="downstream splits whose scaffolds are excluded")
    ap.add_argument("--global-batch", type=int, default=8,
                    help="global batch size (JanusDNA: 8, CrossDNA: 60)")
    ap.add_argument("--valid-frac", type=float, default=0.01)
    ap.add_argument("--test-frac", type=float, default=0.01)
    args = ap.parse_args()

    TILE = args.tile_step if args.tile_step is not None else MAX_ALLOWED_LENGTH
    if TILE == MAX_ALLOWED_LENGTH:
        assert MAX_ALLOWED_LENGTH % args.seqlen == 0, "seqlen must divide 2^20"
    print(f"tiling step: {TILE:,} bp | seqlen: {args.seqlen:,} bp")

    # --- scaffolds to exclude: those in the downstream val/test splits ---
    excl_splits = [s.strip() for s in args.exclude_splits.split(",")]
    w = pd.read_parquet(args.windows, columns=[args.scaffold_col, "split"])
    excluded = set(w[w["split"].isin(excl_splits)][args.scaffold_col].unique())
    kept_down = set(w[~w["split"].isin(excl_splits)][args.scaffold_col].unique())
    print(f"scaffolds in downstream splits {excl_splits}: {len(excluded)} "
          "-> EXCLUDED from pre-training")
    print(f"scaffolds in the downstream train split: {len(kept_down)}")

    # --- genome scaffolds, filtered ---
    fa = Fasta(args.fasta)
    all_seqs = [(n, len(fa[n])) for n in fa.keys()]
    genome_bp = sum(L for _, L in all_seqs)

    usable, drop_short, drop_excl = [], [], []
    for n, L in all_seqs:
        if n in excluded:
            drop_excl.append((n, L))
        elif L < args.seqlen:
            drop_short.append((n, L))
        else:
            usable.append((n, L))

    total = sum(L for _, L in usable)
    excl_bp = sum(L for _, L in drop_excl)

    # --- PRE-training splits (monitoring only) over what remains ---
    asc = sorted(usable, key=lambda x: x[1])
    split_of, test_bp, valid_bp = {}, 0, 0
    # The largest scaffold ALWAYS stays in train: valid and test are drawn from
    # the smaller ones. Otherwise a single large scaffold can swallow the whole
    # valid fraction and leave the training corpus badly depleted.
    for n, L in asc[:-1]:
        if test_bp < args.test_frac * total or test_bp == 0:
            split_of[n] = "test"
            test_bp += L
        elif valid_bp < args.valid_frac * total or valid_bp == 0:
            split_of[n] = "valid"
            valid_bp += L
        else:
            split_of[n] = "train"
    split_of[asc[-1][0]] = "train"

    counts = {"train": 0, "valid": 0, "test": 0}
    bp = {"train": 0, "valid": 0, "test": 0}
    with open(args.out, "w") as out:
        for n, L in sorted(usable, key=lambda x: x[1], reverse=True):
            sp = split_of[n]
            bp[sp] += L
            for start in range(0, L, TILE):
                out.write(f"{n}\t{start}\t{min(start + TILE, L)}\t{sp}\n")
                counts[sp] += 1

    # --- report ---
    print(f"\ngenome total          : {genome_bp:>14,} bp")
    print(f"excluded (val/test)   : {excl_bp:>14,} bp  "
          f"({100*excl_bp/genome_bp:5.1f}%)  [{len(drop_excl)} scaffolds]")
    print(f"excluded (<{args.seqlen}bp): {sum(L for _, L in drop_short):>14,} bp  "
          f"[{len(drop_short)} scaffolds]")
    print(f"PRE-TRAINING CORPUS   : {total:>14,} bp  "
          f"({100*total/genome_bp:5.1f}% of the genome)  [{len(usable)} scaffolds]")
    for sp in ("train", "valid", "test"):
        print(f"   {sp:5s}: {counts[sp]:6d} intervals | {bp[sp]:>14,} bp")

    steps_epoch = counts["train"] * max(1, TILE // args.seqlen) / args.global_batch
    print(f"\ntokens per epoch (train): {bp['train']:,}")
    print(f"optimisation steps per epoch (global batch {args.global_batch}): "
          f"{steps_epoch:.0f}")


if __name__ == "__main__":
    main()
