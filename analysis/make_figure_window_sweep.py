#!/usr/bin/env python
"""
make_figure_window_sweep.py — aggregate the window-length sweep, print summary
tables and produce the comparison figure.

WHAT IT READS FROM DISK
  out_L*/lora_s*/lora_results.json            -> LoRA ('test_mcc')
  out_L*/probe_s*/*.json                      -> zero-shot probe (best effort)
  out_L*/clf/classification_results.json      -> kmer4 baseline
  out_L*/clf_decomp/decompose_results.json    -> gc / repeat baselines
  out_L*/cluster/clustering_results.csv       -> NMI, silhouette

Everything is read from the files: no value is hardcoded, so the figure stays
aligned with the data even when a run is redone.

Table 3 is the analytical point of the script. It asks whether the LoRA gain is
CONSTANT across window lengths or PROPORTIONAL to the zero-shot score, by
projecting both scenarios from a single reference length and comparing them
against what was actually observed. Neither fits: the gain is nil at L=64 and
grows steadily to +0.10 at L=2000, which is what makes this a regime transition
rather than a scaling effect.

USAGE
  python make_figure_window_sweep.py                # tables + figure
  python make_figure_window_sweep.py --no-figure    # tables only
  python make_figure_window_sweep.py --out fig.png
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def _L_of(path):
    if re.search(r"out_L\d+_s\d+", path):
        return None  # skip alternate-seed replicas
    m = re.search(r"out_L(\d+)", path)
    return int(m.group(1)) if m else None


def collect_lora():
    """L -> list of test_mcc (one per seed)."""
    out = {}
    for f in sorted(glob.glob("out_L*/lora_s*/lora_results.json")):
        L = _L_of(f)
        if L is None:
            continue
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue
        if "test_mcc" in d:
            out.setdefault(L, []).append(float(d["test_mcc"]))
    return out


def collect_probe():
    """L -> list of zero-shot MCC.

    The probe output schema is not fixed across runs, so the first plausible key
    is taken from either a dict or a list of dicts.
    """
    keys = ("mcc", "test_mcc")
    out = {}
    for f in sorted(glob.glob("out_L*/probe_s*/*.json")):
        L = _L_of(f)
        if L is None:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        recs = d if isinstance(d, list) else [d]
        for r in recs:
            if not isinstance(r, dict):
                continue
            for k in keys:
                if k in r:
                    out.setdefault(L, []).append(float(r[k]))
                    break
    return out


def collect_baseline_kmer():
    """L -> best kmer4 MCC across logreg and RF."""
    out = {}
    for f in sorted(glob.glob("out_L*/clf/classification_results.json")):
        L = _L_of(f)
        if L is None:
            continue
        try:
            rows = json.load(open(f))
        except Exception:
            continue
        v = [float(r["mcc"]) for r in rows
             if str(r.get("feature", "")).startswith("kmer")]
        if v:
            out[L] = max(v)
    return out


def collect_baseline_decomp():
    """L -> {'gc': best MCC, 'repeat': best MCC}."""
    out = {}
    for f in sorted(glob.glob("out_L*/clf_decomp/decompose_results.json")):
        L = _L_of(f)
        if L is None:
            continue
        try:
            rows = json.load(open(f))
        except Exception:
            continue
        d = {}
        for name in ("gc", "repeat"):
            v = [float(r["mcc"]) for r in rows if r.get("feature") == name]
            if v:
                d[name] = max(v)
        if d:
            out[L] = d
    return out


def collect_cluster():
    """L -> {'nmi_class':..., 'nmi_gc':..., 'silhouette':...}."""
    out = {}
    for f in sorted(glob.glob("out_L*/cluster/clustering_results.csv")):
        L = _L_of(f)
        if L is None:
            continue
        try:
            rows = list(csv.DictReader(open(f)))
        except Exception:
            continue
        if not rows:
            continue
        r = rows[0]
        try:
            out[L] = {
                "nmi_class": float(r["NMI_class"]),
                "nmi_gc": float(r["NMI_vs_GC"]),
                "silhouette": float(r["silhouette"]),
            }
        except (KeyError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #
def _ms(vals):
    """mean, sample SD, n."""
    if not vals:
        return float("nan"), float("nan"), 0
    a = np.asarray(vals, dtype=float)
    sd = a.std(ddof=1) if len(a) > 1 else float("nan")
    return float(a.mean()), float(sd), len(a)


def _f(x, nd=4):
    return "  --  " if x != x else f"{x:.{nd}f}"


def print_tables(lora, probe, kmer, decomp, clust):
    Ls = sorted(set(lora) | set(probe) | set(kmer) | set(decomp) | set(clust))

    print("\n" + "=" * 84)
    print("TABLE 1 - test MCC: model and baselines")
    print("=" * 84)
    print(f"{'L':>5} | {'zero-shot':>9} {'SD':>7} {'n':>2} | "
          f"{'LoRA':>8} {'SD':>7} {'n':>2} | {'gain':>7} | "
          f"{'kmer4':>7} {'gc':>7}")
    print("-" * 84)
    for L in Ls:
        zm, zs, zn = _ms(probe.get(L, []))
        lm, ls, ln = _ms(lora.get(L, []))
        gain = lm - zm if (lm == lm and zm == zm) else float("nan")
        k = kmer.get(L, float("nan"))
        g = decomp.get(L, {}).get("gc", float("nan"))
        gain_s = f"{gain:+7.4f}" if gain == gain else "   --  "
        print(f"{L:5d} | {_f(zm)} {_f(zs)} {zn:2d} | "
              f"{_f(lm)} {_f(ls)} {ln:2d} | {gain_s} | "
              f"{_f(k)} {_f(g)}")

    print("\n" + "=" * 84)
    print("TABLE 2 - unsupervised clustering (k-means, k=4)")
    print("=" * 84)
    print(f"{'L':>5} | {'NMI class':>10} {'NMI vs GC':>10} {'silhouette':>11} "
          f"| {'ratio GC/class':>18}")
    print("-" * 84)
    for L in Ls:
        c = clust.get(L)
        if not c:
            continue
        ratio = c["nmi_gc"] / c["nmi_class"] if c["nmi_class"] else float("nan")
        print(f"{L:5d} | {_f(c['nmi_class']):>10} {_f(c['nmi_gc']):>10} "
              f"{_f(c['silhouette']):>11} | {ratio:18.1f}x")

    # Is the LoRA gain constant, or proportional to the zero-shot score?
    print("\n" + "=" * 84)
    print("TABLE 3 - is the LoRA gain constant or proportional?")
    print("=" * 84)
    ref = max((L for L in Ls if L in lora and L in probe), default=None)
    if ref is None:
        print("  (needs at least one length with both probe and LoRA)")
        return
    zr, _, _ = _ms(probe[ref])
    lr, _, _ = _ms(lora[ref])
    add, mul = lr - zr, (lr / zr if zr else float("nan"))
    print(f"  reference L={ref}:  zero-shot={zr:.4f}  LoRA={lr:.4f}  "
          f"-> additive {add:+.4f}   multiplicative x{mul:.3f}")
    print(f"\n{'L':>5} | {'observed':>9} | {'scen. A (add.)':>14} "
          f"{'scen. B (mult.)':>15} | {'closer to':>13}")
    print("-" * 84)
    for L in Ls:
        if L not in lora or L not in probe:
            continue
        zm, _, _ = _ms(probe[L])
        lm, _, _ = _ms(lora[L])
        a, b = zm + add, zm * mul
        closer = "A (constant)" if abs(lm - a) < abs(lm - b) else "B (proportional)"
        if L == ref:
            closer = "(reference)"
        print(f"{L:5d} | {lm:9.4f} | {a:14.4f} {b:15.4f} | {closer:>13}")


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(lora, probe, kmer, decomp, clust, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[figure] matplotlib not available: pip install matplotlib")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 9), sharex=True)

    def series(d, Ls):
        m = [_ms(d.get(L, []))[0] for L in Ls]
        s = [_ms(d.get(L, []))[1] for L in Ls]
        return np.array(m), np.array(s)

    # --- top panel: MCC ---
    Lz = sorted(probe)
    Ll = sorted(lora)
    if Ll:
        m, s = series(lora, Ll)
        ax1.errorbar(Ll, m, yerr=np.nan_to_num(s), marker="s", lw=2, capsize=3,
                     color="#b4413c", label="LoRA (fine-tuned)")
    if Lz:
        m, s = series(probe, Lz)
        ax1.errorbar(Lz, m, yerr=np.nan_to_num(s), marker="o", lw=2, capsize=3,
                     color="#2a78d6", label="zero-shot probe (frozen)")
    if kmer:
        Lk = sorted(kmer)
        ax1.plot(Lk, [kmer[L] for L in Lk], marker="^", ls="--", lw=1.8,
                 color="#1baf7a", label="kmer4 baseline (visible to the model)")
    if decomp:
        Ld = sorted(L for L in decomp if "gc" in decomp[L])
        ax1.plot(Ld, [decomp[L]["gc"] for L in Ld], marker="v", ls=":", lw=1.8,
                 color="#eb6834", label="GC-only baseline (visible)")
        # The repeat baseline is drawn because it is NOT visible to the model:
        # repeat fraction comes from soft-masking, and the sequences are stored
        # uppercased. It marks how much of the 'trivial' baseline is unfair.
        Lr = sorted(L for L in decomp if "repeat" in decomp[L])
        ax1.plot(Lr, [decomp[L]["repeat"] for L in Lr], marker="x", ls=":",
                 lw=1.4, color="#999999",
                 label="repeat-only baseline (NOT visible)")

    ax1.axvline(64, color="#cccccc", lw=1, zorder=0)
    ax1.text(64, ax1.get_ylim()[1], "  architectural\n  floor",
             va="top", fontsize=8, color="#777777")
    ax1.set_ylabel("test MCC")
    ax1.set_title("Performance as a function of window length L\n"
                  "JanusDNA-144 from scratch, A. patagonicus genome",
                  fontsize=11)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, loc="upper left")

    # --- bottom panel: clustering ---
    # Deliberately on the same x axis as the top panel: the flat NMI-vs-GC line
    # under a rising MCC curve IS the dissociation the thesis argues for.
    if clust:
        Lc = sorted(clust)
        ax2.plot(Lc, [clust[L]["nmi_gc"] for L in Lc], marker="s", lw=2,
                 color="#eb6834", label="NMI against GC")
        ax2.plot(Lc, [clust[L]["silhouette"] for L in Lc], marker="d", lw=2,
                 color="#8a6fbf", label="silhouette")
        ax2.plot(Lc, [clust[L]["nmi_class"] for L in Lc], marker="^", lw=2,
                 color="#1baf7a", label="NMI against the classes")
        ax2.set_ylabel("unsupervised clustering")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=8, loc="center left")
    ax2.axvline(64, color="#cccccc", lw=1, zorder=0)

    ax2.set_xscale("log")
    ax2.set_xlabel("L (bp, log scale)")
    ticks = sorted(set(list(probe) + list(kmer) + list(decomp)))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([str(t) for t in ticks], fontsize=8)
    ax2.minorticks_off()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"\n[figure] saved to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/window_sweep.png")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    lora = collect_lora()
    probe = collect_probe()
    kmer = collect_baseline_kmer()
    decomp = collect_baseline_decomp()
    clust = collect_cluster()

    print(f"[collected] LoRA: {len(lora)} points  probe: {len(probe)}  "
          f"kmer4: {len(kmer)}  decomp: {len(decomp)}  cluster: {len(clust)}")

    print_tables(lora, probe, kmer, decomp, clust)

    if not args.no_figure:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        make_figure(lora, probe, kmer, decomp, clust, args.out)


if __name__ == "__main__":
    main()
