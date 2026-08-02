#!/usr/bin/env python
"""
prepare_windows.py — build the labelled window dataset from a genome assembly.

Expected inputs (downloadable with NCBI Datasets, see README):
    --fasta  GCF_965638725.1_*_genomic.fna   (soft-masked — required)
    --gff    genomic.gff

Outputs:
    <outdir>/windows.parquet   (id, scaffold, start, end, label, group,
                                gc, n_frac, repeat_frac, split, seq)
    <outdir>/summary.json      (counts and confounder statistics)

Two methodological choices are worth flagging up front, because both are
defensible rather than unique and both are isolated so they can be replaced:

  * CENTRE-BASED LABELLING — a window is labelled by the class of its centre,
    not by majority overlap. Majority overlap starves the classes made of short
    features: CDS and UTR are rarely contiguous for 1000 bp, so a majority rule
    would silently produce a wildly imbalanced dataset.
  * UTR DERIVATION — NCBI GFFs frequently omit explicit UTR features. When they
    are absent, UTRs are derived as (mRNA exons) minus (CDS).
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from common import gc_content, n_fraction, repeat_fraction, set_seed


# --------------------------------------------------------------------------- #
# Interval arithmetic on 0-based half-open intervals [s, e)
# --------------------------------------------------------------------------- #
def merge(ivals):
    if not ivals:
        return []
    ivals = sorted(ivals)
    out = [list(ivals[0])]
    for s, e in ivals[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def complement(merged, length):
    """Complement of a set of disjoint intervals within [0, length)."""
    out = []
    prev = 0
    for s, e in merged:
        if s > prev:
            out.append((prev, s))
        prev = max(prev, e)
    if prev < length:
        out.append((prev, length))
    return out


def subtract(a, b):
    """a minus b, where a and b are disjoint, SORTED interval lists [s, e).

    Two-pointer, O(len(a) + len(b)). Returns sorted disjoint intervals.
    """
    out = []
    j, nb = 0, len(b)
    for s, e in a:
        cur = s
        while j < nb and b[j][1] <= s:  # discard b entirely before this a
            j += 1
        k = j
        while k < nb and b[k][0] < e:  # b intervals starting inside [cur, e)
            bs, be = b[k]
            if bs > cur:
                out.append((cur, min(bs, e)))
            cur = max(cur, be)
            if cur >= e:
                break
            k += 1
        if cur < e:
            out.append((cur, e))
    return out


class IvalIndex:
    """Fast overlap test over disjoint, sorted intervals."""

    def __init__(self, merged):
        self.starts = [s for s, _ in merged]
        self.ends = [e for _, e in merged]

    def overlaps(self, s, e):
        st = self.starts
        if not st:
            return False
        j = bisect.bisect_right(st, s)           # intervals with start <= s
        if j - 1 >= 0 and self.ends[j - 1] > s:  # an interval containing s
            return True
        if j < len(st) and st[j] < e:            # an interval starting in [s, e)
            return True
        return False


# --------------------------------------------------------------------------- #
# GFF3 parsing -> per-category, per-scaffold intervals
# --------------------------------------------------------------------------- #
def parse_attrs(field: str) -> dict:
    d = {}
    for kv in field.strip().split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k] = v
    return d


def build_categories(gff_path: str, include_ncrna: bool):
    """Return {category: {scaffold: [(s, e), ...] merged}}.

    Default categories: CDS, UTR (5' + 3'), intron, intergenic (+ optional ncRNA).
    """
    gene = defaultdict(list)        # for intergenic (complement of genes)
    cds = defaultdict(list)
    utr = defaultdict(list)         # explicit UTRs, when the GFF has them
    ncrna = defaultdict(list)
    exon_by_tx = defaultdict(list)  # used to derive introns
    tx_scaffold = {}                # transcript_id -> scaffold
    mrna_ids = set()                # mRNA transcript IDs (used to derive UTRs)
    scaf_len_hint = {}              # from ##sequence-region, when present

    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("##sequence-region"):
                parts = line.split()
                if len(parts) >= 4:
                    scaf_len_hint[parts[1]] = int(parts[3])
                continue
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            seqid, _src, ftype, start, end = f[0], f[1], f[2], f[3], f[4]
            s0, e0 = int(start) - 1, int(end)  # -> 0-based half-open
            attrs = parse_attrs(f[8])

            if ftype == "gene":
                gene[seqid].append((s0, e0))
            elif ftype == "CDS":
                cds[seqid].append((s0, e0))
            elif ftype == "mRNA":
                mid = attrs.get("ID")
                if mid:
                    mrna_ids.add(mid)
            elif ftype in ("five_prime_UTR", "three_prime_UTR"):
                utr[seqid].append((s0, e0))
            elif ftype == "exon":
                parent = attrs.get("Parent", attrs.get("transcript_id", "NA"))
                exon_by_tx[parent].append((s0, e0))
                tx_scaffold[parent] = seqid
            elif include_ncrna and ftype in (
                "lnc_RNA", "ncRNA", "antisense_RNA", "snoRNA", "snRNA", "miRNA"
            ):
                ncrna[seqid].append((s0, e0))

    # introns = the gaps between consecutive exons of the same transcript
    intron = defaultdict(list)
    for tx, exons in exon_by_tx.items():
        if len(exons) < 2:
            continue
        scaf = tx_scaffold.get(tx)
        if scaf is None:
            continue
        ex = sorted(exons)
        for (a, b), (c, d) in zip(ex[:-1], ex[1:]):
            if c > b:
                intron[scaf].append((b, c))

    # UTRs: when the GFF does not annotate them explicitly (common in NCBI
    # releases), DERIVE them as (mRNA exons) minus (CDS), per scaffold.
    utr_derived = False
    if not any(utr.values()):
        utr_derived = True
        mrna_exon = defaultdict(list)
        for tx, exons in exon_by_tx.items():
            if tx in mrna_ids:
                scaf = tx_scaffold.get(tx)
                if scaf is not None:
                    mrna_exon[scaf].extend(exons)
        for scaf in set(list(mrna_exon.keys()) + list(cds.keys())):
            ex_m = merge(mrna_exon.get(scaf, []))
            cds_m = merge(cds.get(scaf, []))
            u = subtract(ex_m, cds_m)  # non-coding exonic = 5'/3' UTR
            if u:
                utr[scaf] = u
    print(f"      UTR: {'derived (mRNA exons - CDS)' if utr_derived else 'explicit in GFF'}")

    cats = {
        "CDS": {s: merge(v) for s, v in cds.items()},
        "UTR": {s: merge(v) for s, v in utr.items()},
        "intron": {s: merge(v) for s, v in intron.items()},
    }
    if include_ncrna:
        cats["ncRNA"] = {s: merge(v) for s, v in ncrna.items()}

    # intergenic = complement of genes (needs scaffold lengths -> from the FASTA)
    cats["_gene_merged"] = {s: merge(v) for s, v in gene.items()}
    return cats, scaf_len_hint


# Priority hierarchy used to disambiguate overlapping annotations.
PRIORITY = ["CDS", "UTR", "ncRNA", "intron", "intergenic"]


# --------------------------------------------------------------------------- #
# Scaffold -> split assignment (whole scaffolds, greedy, length-balanced)
#
# This is the anti-leakage guarantee: a scaffold belongs to exactly one split,
# so no genomic region is ever shared between train and test.
# --------------------------------------------------------------------------- #
def assign_splits(scaf_lengths: dict, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    total = sum(scaf_lengths.values())
    targets = {
        "train": (1 - val_frac - test_frac) * total,
        "val": val_frac * total,
        "test": test_frac * total,
    }
    acc = {"train": 0.0, "val": 0.0, "test": 0.0}
    assign = {}
    # longest first, with a little noise so the result is not fully deterministic
    order = sorted(scaf_lengths, key=lambda s: (-scaf_lengths[s], rng.random()))
    for scaf in order:
        # pick the split with the largest relative deficit against its target
        deficit = {k: (targets[k] - acc[k]) / max(targets[k], 1.0) for k in targets}
        pick = max(deficit, key=deficit.get)
        assign[scaf] = pick
        acc[pick] += scaf_lengths[scaf]
    return assign


# --------------------------------------------------------------------------- #
# Window sampling per (split, class), with priority and N filters
# --------------------------------------------------------------------------- #
def sample_windows(category_intervals, higher_indexes, scaffolds_in_split,
                   scaf_len, fasta, L, cap, max_n_frac, seed,
                   dedup_frac=0.5, max_attempts_mult=80):
    """Sample up to `cap` windows of length L labelled with the current class.

    CENTRE-BASED LABELLING: the window is [center - L//2, center + L//2) and its
    class is the class of the CENTRE. A centre falling inside a higher-priority
    class (e.g. an antisense CDS inside an intron) is DISCARDED. This keeps the
    priority-based disambiguation while avoiding the starvation problem a
    whole-window rule would cause for short features.

    category_intervals: {scaffold: [(s, e)]} for the current class
    higher_indexes:     {scaffold: [IvalIndex, ...]} for higher-priority classes
    scaf_len:           {scaffold: length}, used to keep the window in bounds
    """
    rng = np.random.default_rng(seed)
    half = L // 2
    tail = L - half

    # intervals eligible for the CENTRE, clamped so the whole window fits
    flat = []
    for scaf in scaffolds_in_split:
        Ln = scaf_len.get(scaf)
        if Ln is None or Ln < L:
            continue
        for (s, e) in category_intervals.get(scaf, []):
            cs = max(s, half)
            ce = min(e, Ln - tail)
            if ce > cs:
                flat.append((scaf, cs, ce))
    if not flat:
        return []

    widths = np.array([ce - cs for _, cs, ce in flat], dtype=np.float64)
    cum = np.cumsum(widths)
    total = cum[-1]

    dedup_step = max(1, int(L * dedup_frac))
    seen = set()
    out = []
    attempts = 0
    max_attempts = int(cap * max_attempts_mult)

    while len(out) < cap and attempts < max_attempts:
        attempts += 1
        r = rng.random() * total
        k = int(np.searchsorted(cum, r, side="right"))
        scaf, cs, ce = flat[k]
        center = int(rng.integers(cs, ce))
        start = center - half
        key = (scaf, start // dedup_step)
        if key in seen:
            continue

        # priority: discard if the CENTRE falls in a higher-priority class
        if any(idx.overlaps(center, center + 1)
               for idx in higher_indexes.get(scaf, [])):
            continue

        raw = str(fasta[scaf][start : start + L].seq)
        if len(raw) < L or n_fraction(raw) > max_n_frac:
            continue

        seen.add(key)
        out.append(
            dict(
                scaffold=scaf,
                start=start,
                end=start + L,
                seq=raw.upper(),  # the models expect uppercase ACGTN
                gc=round(gc_content(raw), 4),
                n_frac=round(n_fraction(raw), 4),
                # computed from the soft-masked sequence, BEFORE uppercasing
                repeat_frac=round(repeat_fraction(raw), 4),
            )
        )
    return out


def coarse_group(label: str) -> str:
    """Coarse grouping for NMI(group) vs. NMI(subtype), as in CrossDNA ED Fig. 6."""
    if label == "CDS":
        return "coding"
    if label == "intergenic":
        return "intergenic"
    return "genic_noncoding"  # UTR, intron, ncRNA


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True, help="soft-masked genomic.fna")
    ap.add_argument("--gff", required=True, help="genomic.gff")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--per-class-train", type=int, default=8000)
    ap.add_argument("--per-class-eval", type=int, default=2000)
    ap.add_argument("--max-n-frac", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--include-ncrna", action="store_true")
    ap.add_argument("--min-scaffold-len", type=int, default=50_000,
                    help="ignore scaffolds shorter than this (noise / short contigs)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    try:
        from pyfaidx import Fasta
    except ImportError:
        raise SystemExit("pyfaidx is missing: pip install pyfaidx")

    print("[1/5] Indexing FASTA ...")
    # sequence_always_upper=False is REQUIRED: repeat_frac reads the soft-masking
    fasta = Fasta(args.fasta, sequence_always_upper=False, rebuild=False)
    scaf_len = {name: len(fasta[name]) for name in fasta.keys()
                if len(fasta[name]) >= args.min_scaffold_len}
    print(f"      scaffolds used: {len(scaf_len)}  "
          f"(total bp {sum(scaf_len.values()):,})")

    print("[2/5] Parsing GFF3 and deriving categories ...")
    cats, _ = build_categories(args.gff, args.include_ncrna)
    # intergenic = complement of genes within each scaffold
    inter = {}
    for scaf, length in scaf_len.items():
        gmerged = cats["_gene_merged"].get(scaf, [])
        inter[scaf] = complement(gmerged, length)
    cats["intergenic"] = inter
    del cats["_gene_merged"]

    labels = [c for c in PRIORITY if c in cats]  # order == priority
    for c in labels:
        n_iv = sum(len(v) for v in cats[c].values())
        print(f"      {c:11s}: {n_iv:,} intervals")

    print("[3/5] Assigning scaffolds -> splits ...")
    assign = assign_splits(scaf_len, args.val_frac, args.test_frac, args.seed)
    split_scaffolds = defaultdict(list)
    for scaf, sp in assign.items():
        split_scaffolds[sp].append(scaf)
    for sp in ("train", "val", "test"):
        bp = sum(scaf_len[s] for s in split_scaffolds[sp])
        print(f"      {sp:5s}: {len(split_scaffolds[sp])} scaffolds, {bp:,} bp")

    print("[4/5] Sampling windows (priority and N filters applied) ...")
    rows = []
    for sp in ("train", "val", "test"):
        cap = args.per_class_train if sp == "train" else args.per_class_eval
        for ci, label in enumerate(labels):
            higher_labels = labels[:ci]  # higher-priority classes
            higher_idx = defaultdict(list)
            for hl in higher_labels:
                for scaf in split_scaffolds[sp]:
                    ivs = cats[hl].get(scaf, [])
                    if ivs:
                        higher_idx[scaf].append(IvalIndex(ivs))
            sub_seed = int(
                hashlib.md5(f"{args.seed}-{sp}-{label}".encode()).hexdigest()[:8],
                16,
            )
            ws = sample_windows(
                cats[label], higher_idx, split_scaffolds[sp], scaf_len, fasta,
                args.window, cap, args.max_n_frac, seed=sub_seed,
            )
            for w in ws:
                w["label"] = label
                w["group"] = coarse_group(label)
                w["split"] = sp
                key = f"{w['scaffold']}:{w['start']}-{w['end']}:{label}"
                w["id"] = hashlib.md5(key.encode()).hexdigest()[:16]
                rows.append(w)
            print(f"      {sp:5s} {label:11s}: {len(ws):>6d} windows (cap {cap})")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No windows sampled: check the FASTA/GFF and the thresholds.")
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_parquet = os.path.join(args.outdir, "windows.parquet")
    df.to_parquet(out_parquet, index=False)

    # ------------------------------------------------------------------ #
    # Confounder report. If a class has markedly different GC or repeat
    # content, a classifier may separate it for entirely trivial reasons —
    # which is precisely what the baselines downstream are there to measure.
    # ------------------------------------------------------------------ #
    print("[5/5] Confounder report (per class):")
    summary = {"window": int(args.window), "n_total": int(len(df)), "per_class": {}}
    for label, g in df.groupby("label"):
        stats = dict(
            n=int(len(g)),
            gc_mean=round(float(g.gc.mean()), 4),
            gc_std=round(float(g.gc.std()), 4),
            repeat_mean=round(float(g.repeat_frac.mean()), 4),
            n_frac_mean=round(float(g.n_frac.mean()), 4),
        )
        summary["per_class"][label] = stats
        print(f"      {label:11s} n={stats['n']:>6d}  "
              f"GC={stats['gc_mean']:.3f}±{stats['gc_std']:.3f}  "
              f"rep={stats['repeat_mean']:.3f}")
    summary["split_counts"] = {
        sp: df[df.split == sp].label.value_counts().to_dict()
        for sp in ("train", "val", "test")
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nWrote {out_parquet}  ({len(df)} windows)")
    print("NOTE: compare the GC and repeat columns across classes. If they differ "
          "substantially, the baselines in classify_baselines.py (k-mer, GC-only) "
          "will tell you how much of the signal is trivial rather than learned.")


if __name__ == "__main__":
    main()
