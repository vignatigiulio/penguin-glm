#!/usr/bin/env python3
"""
make_downstream_bed.py — build a HyenaDNA/Caduceus-format BED
(chrom, start, end, split as TSV) whose split assignment is IDENTICAL to the
downstream task's.

The split is read from windows.parquet rather than recomputed, so the pre-training
partition and the evaluation partition cannot drift apart. The script refuses to
run if any scaffold appears in more than one downstream split: that would mean
the chromosome-disjoint guarantee had already been broken upstream, and silently
picking one split would bury the problem.
"""
import argparse

import pandas as pd
import polars as pl


def load_split(path):
    # polars rather than pandas here: pyarrow raises a repetition-level error on
    # this parquet when projecting columns
    df = pl.read_parquet(path, columns=["scaffold", "split"]).to_pandas()
    multi = df.groupby("scaffold")["split"].nunique()
    bad = multi[multi > 1]
    if len(bad):
        raise SystemExit(f"[STOP] splits are NOT disjoint: {list(bad.index)}")
    norm = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
    m = df.drop_duplicates("scaffold").set_index("scaffold")["split"].to_dict()
    return {k: norm[v] for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fai", required=True, help="FASTA index (.fai)")
    ap.add_argument("--parquet", required=True, help="windows.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=int, default=131072)
    ap.add_argument("--fasta", default=None, help="required only with --n-max")
    ap.add_argument("--n-max", type=float, default=None,
                    help="drop intervals whose N fraction exceeds this")
    a = ap.parse_args()

    smap = load_split(a.parquet)
    fa = None
    if a.n_max is not None:
        if not a.fasta:
            raise SystemExit("--n-max requires --fasta")
        from pyfaidx import Fasta
        fa = Fasta(a.fasta, sequence_always_upper=True)

    fai = pd.read_csv(a.fai, sep="\t", header=None, usecols=[0, 1],
                      names=["name", "length"])
    rows, skip_n = [], 0
    for _, r in fai.iterrows():
        # .iloc / named access, not r[0]: pandas >= 2 changed positional
        # indexing on Series and r[0] silently returns the wrong thing
        name, Ln = r["name"], int(r["length"])
        if name not in smap:
            continue
        split, s = smap[name], 0
        while s + a.interval <= Ln:
            e = s + a.interval
            if fa is not None:
                seq = fa[name][s:e].seq
                if seq.count("N") / len(seq) > a.n_max:
                    skip_n += 1
                    s = e
                    continue
            rows.append((name, s, e, split))
            s = e

    bed = pd.DataFrame(rows, columns=["chrom", "start", "end", "split"])
    bed.to_csv(a.out, sep="\t", header=False, index=False)
    print(f"[ok] {len(bed)} intervals -> {a.out}")
    print(bed["split"].value_counts().to_string())
    if skip_n:
        print(f"[info] {skip_n} intervals dropped for N > {a.n_max}")


if __name__ == "__main__":
    main()
