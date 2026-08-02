#!/usr/bin/env python3
"""
make_pretrain_bed.py — generate the pre-training BED for JanusDNA
(HG38Dataset format) from a genome FASTA.

The HG38Dataset dataloader:
  - reads a TSV with columns chr_name, start, end, split
  - REWRITES end = start + 2^20 and splits each interval into
    (2^20 / seqlen) sub-windows
  - clamps, at read time, any window running past the end of the scaffold

So all this script has to do is tile every sequence in steps of 2^20
(MAX_ALLOWED_LENGTH). Because the dataloader ignores the `end` column entirely,
generating the BED with any other interval silently produces windows of the
wrong length — the file will look correct and the training will not be.

Scaffolds shorter than --min-len (default = seqlen) are EXCLUDED: below seqlen
the dataloader's internal assertion fails.

Splits are assigned PER SCAFFOLD, so there is no leakage between train, valid
and test: the smallest scaffolds fill test and valid up to the requested
fractions, everything else goes to train.
"""
import argparse

from pyfaidx import Fasta

MAX_ALLOWED_LENGTH = 2 ** 20  # 1,048,576 — hard-coded in the dataloader, DO NOT CHANGE


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True, help="genome FASTA (pyfaidx-indexable)")
    ap.add_argument("--out", required=True, help="output BED path")
    ap.add_argument("--seqlen", type=int, default=131072,
                    help="pre-training max_length; must divide 2^20 (a power of 2)")
    ap.add_argument("--min-len", type=int, default=None,
                    help="scaffolds shorter than this are excluded (default = seqlen)")
    ap.add_argument("--valid-frac", type=float, default=0.01)
    ap.add_argument("--test-frac", type=float, default=0.01)
    args = ap.parse_args()

    assert MAX_ALLOWED_LENGTH % args.seqlen == 0, (
        f"seqlen {args.seqlen} must divide 2^20 (a power of 2 <= 1048576)"
    )
    min_len = args.min_len if args.min_len is not None else args.seqlen

    fa = Fasta(args.fasta)
    all_seqs = [(name, len(fa[name])) for name in fa.keys()]
    kept = [(n, L) for n, L in all_seqs if L >= min_len]
    dropped = [(n, L) for n, L in all_seqs if L < min_len]
    total = sum(L for _, L in kept)

    # --- per-scaffold split assignment: the smallest scaffolds seed and fill
    #     test, then valid, up to the requested fractions. At least one scaffold
    #     per split is guaranteed even when the fractions round down to zero.
    kept_asc = sorted(kept, key=lambda x: x[1])  # ascending
    test_target = args.test_frac * total
    valid_target = args.valid_frac * total
    split_of = {}
    test_bp = valid_bp = 0
    for n, L in kept_asc:
        if test_bp < test_target or test_bp == 0:
            split_of[n] = "test"
            test_bp += L
        elif valid_bp < valid_target or valid_bp == 0:
            split_of[n] = "valid"
            valid_bp += L
        else:
            split_of[n] = "train"

    # --- write the BED rows (start in steps of 2^20) ---
    counts = {"train": 0, "valid": 0, "test": 0}
    bp = {"train": 0, "valid": 0, "test": 0}
    with open(args.out, "w") as out:
        for n, L in sorted(kept, key=lambda x: x[1], reverse=True):
            sp = split_of[n]
            bp[sp] += L
            for start in range(0, L, MAX_ALLOWED_LENGTH):
                end = start + MAX_ALLOWED_LENGTH
                out.write(f"{n}\t{start}\t{end}\t{sp}\n")
                counts[sp] += 1

    # --- report ---
    print(f"Scaffolds total: {len(all_seqs)} | included: {len(kept)} | "
          f"excluded (<{min_len} bp): {len(dropped)}")
    if dropped:
        ex = ", ".join(f"{n}({L})" for n, L in dropped[:8])
        print(f"  excluded: {ex}{' ...' if len(dropped) > 8 else ''}")
    print(f"bp included: {total:,}")
    for sp in ("train", "valid", "test"):
        pct = 100 * bp[sp] / total if total else 0
        print(f"  {sp:5s}: {counts[sp]:6d} rows | {bp[sp]:>14,} bp ({pct:5.2f}%)")
    print(f"Total rows: {sum(counts.values())}")
    print(f"Samples per epoch (seqlen={args.seqlen}): "
          f"{sum(counts.values()) * (MAX_ALLOWED_LENGTH // args.seqlen):,}")


if __name__ == "__main__":
    main()
