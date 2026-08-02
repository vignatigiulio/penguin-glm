#!/usr/bin/env python3
"""
make_figures.py — generate the thesis figures from the REAL result files.

No number is hardcoded: every value is read from the result JSONs, and if a
result is missing the corresponding figure is skipped with a warning rather than
drawn from stale or invented data. That constraint is the reason this script
exists at all — a figure regenerated from the files is a figure that cannot
silently disagree with the tables.

Figures:
  1. zero_shot_vs_lora   the headline result (2 models x 2 regimes)
  2. nmi_gc_vs_function   the central result (embeddings organise by GC)
  3. contamination        decomposition of the JanusDNA in-domain advantage
  4. pretraining_curves   training dynamics (CrossDNA saturation)
  5. confusion_matrices   post-LoRA confusion, row-normalised

Usage:
  python make_figures.py
  python make_figures.py --png              # PNG in addition to PDF
  python make_figures.py --out-dir figures  # where to write
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PENGUIN_ROOT = Path(os.environ.get("PENGUIN_ROOT", Path.home() / "penguin_glm"))
OUT = PENGUIN_ROOT / "out"
FIG = Path("figures")

# restrained palette, suitable for print
C_SCRATCH = "#2c6fbb"   # blue:   from-scratch (in-domain)
C_HUMAN = "#c0392b"     # red:    human (transfer)
C_DIRTY = "#7f8c8d"     # grey:   contaminated
C_GC = "#e67e22"        # orange: GC
C_FUNC = "#27ae60"      # green:  function
C_REP = "#95a5a6"       # light grey: repeats

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 120, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "sans-serif",
})

SAVE_PNG = False


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    if SAVE_PNG:
        fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  [ok] {name}.pdf")


# ---------- readers for the real data ----------
def probe_seeds(prefix, feature):
    vals = []
    for s in [1, 2, 3, 4]:
        p = OUT / f"{prefix}{s}" / "mlp_probe_results.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            if r.get("feature") == feature:
                vals.append(r["mcc"])
    return np.array(vals)


def lora_mcc(folder):
    p = OUT / folder / "lora_results.json"
    return json.loads(p.read_text())["test_mcc"] if p.exists() else None


def cluster_row(folder, emb):
    p = OUT / folder / "clustering_results.json"
    if not p.exists():
        return None
    for r in json.loads(p.read_text()):
        if r.get("emb") == emb:
            return r
    return None


# ---------- FIGURE 1: zero-shot vs LoRA ----------
def fig_zero_vs_lora():
    jz_s = probe_seeds("mlp_janus_clean_seed", "janus144_scratch_clean")
    jz_h = probe_seeds("mlp_janus_clean_seed", "janus144_human")
    cz_s = probe_seeds("mlp_crossdna_seed", "crossdna81_scratch")
    cz_h = probe_seeds("mlp_crossdna_seed", "crossdna81_human")
    jl_s = lora_mcc("lora_janus144_scratch_clean")
    jl_h = lora_mcc("lora_janus144_resume")
    cl_s = lora_mcc("lora_crossdna81_scratch_clean")
    cl_h = lora_mcc("lora_crossdna81_human_fp32")

    if any(len(a) == 0 for a in [jz_s, jz_h, cz_s, cz_h]) or None in [jl_s, jl_h, cl_s, cl_h]:
        print("  [skip] zero_shot_vs_lora: missing data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, title, s_zs, h_zs, s_lo, h_lo in [
        (axes[0], "JanusDNA-144", jz_s, jz_h, jl_s, jl_h),
        (axes[1], "CrossDNA-8.1M", cz_s, cz_h, cl_s, cl_h),
    ]:
        x = np.arange(2)  # zero-shot, LoRA
        w = 0.36
        s_vals = [s_zs.mean(), s_lo]
        h_vals = [h_zs.mean(), h_lo]
        s_err = [s_zs.std(ddof=1) if len(s_zs) > 1 else 0, 0]
        h_err = [h_zs.std(ddof=1) if len(h_zs) > 1 else 0, 0]
        ax.bar(x - w / 2, s_vals, w, yerr=s_err, capsize=3, color=C_SCRATCH,
               label="from-scratch (penguin)")
        ax.bar(x + w / 2, h_vals, w, yerr=h_err, capsize=3, color=C_HUMAN,
               label="human (hg38)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["zero-shot", "LoRA"])
        ax.set_ylim(0, 0.5)
        # the signed gap is the quantity of interest, so it is annotated rather
        # than left for the reader to eyeball from the bar heights
        for xi, (sv, hv) in enumerate(zip(s_vals, h_vals)):
            d = sv - hv
            ax.annotate(f"{d:+.3f}", (xi, max(sv, hv) + 0.03), ha="center",
                        fontsize=9, fontweight="bold",
                        color=C_SCRATCH if d > 0 else C_HUMAN)
    axes[0].set_ylabel("MCC (test)")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("In-domain pre-training vs transfer: zero-shot and after LoRA", y=1.02)
    save(fig, "1_zero_shot_vs_lora")


# ---------- FIGURE 2: NMI, GC against function ----------
def fig_nmi():
    rows = [
        ("JanusDNA-144", cluster_row("cluster_janus144", "janus144")),
        ("CrossDNA-8.1M", cluster_row("cluster_81", "crossdna81_mean")),
    ]
    rows = [(n, r) for n, r in rows if r]
    if not rows:
        print("  [skip] nmi: missing data")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(rows))
    w = 0.34
    func = [r["NMI_class"] for _, r in rows]
    gc = [r["NMI_vs_GC"] for _, r in rows]
    ax.bar(x - w / 2, func, w, color=C_FUNC,
           label="function (CDS/UTR/intron/intergenic)")
    ax.bar(x + w / 2, gc, w, color=C_GC, label="GC content")
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in rows])
    ax.set_ylabel("NMI (embedding clusters vs variable)")
    ax.set_title("Zero-shot embeddings organise by composition, not function")
    ax.legend(fontsize=9, frameon=False)
    for xi, (f, g) in enumerate(zip(func, gc)):
        ax.annotate(f"{g/f:.0f}x", (xi + w / 2, g + 0.015), ha="center",
                    fontsize=9, color=C_GC)
    save(fig, "2_nmi_gc_vs_function")


# ---------- FIGURE 3: contamination decomposition (JanusDNA) ----------
def fig_contamination():
    dirty = probe_seeds("mlp_janus_clean_seed", "janus144_scratch_dirty")
    clean = probe_seeds("mlp_janus_clean_seed", "janus144_scratch_clean")
    human = probe_seeds("mlp_janus_clean_seed", "janus144_human")
    if any(len(a) == 0 for a in [dirty, clean, human]):
        print("  [skip] contamination: missing data")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = ["contaminated", "clean", "human"]
    vals = [dirty.mean(), clean.mean(), human.mean()]
    errs = [a.std(ddof=1) if len(a) > 1 else 0 for a in [dirty, clean, human]]
    cols = [C_DIRTY, C_SCRATCH, C_HUMAN]
    ax.bar(labels, vals, yerr=errs, capsize=4, color=cols, width=0.6)
    ax.set_ylabel("zero-shot MCC (test)")
    lo = min(human.mean(), *[v for v in vals]) - 0.02
    hi = max(vals) + 0.03
    ax.set_ylim(lo, hi)
    ax.set_title("JanusDNA: decomposing the in-domain advantage")
    # the two gaps are the point of the figure: how much of the apparent
    # advantage was contamination, and how much was real
    ax.annotate(f"contamination {dirty.mean()-clean.mean():+.3f}",
                (0.5, hi - 0.008), ha="center", fontsize=8.5, color=C_DIRTY)
    ax.annotate(f"in-domain signal {clean.mean()-human.mean():+.3f}",
                (1.5, hi - 0.008), ha="center", fontsize=8.5, color=C_SCRATCH)
    for xi, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (xi, v + 0.003), ha="center", fontsize=9)
    save(fig, "3_contamination_janusdna")


# ---------- FIGURE 4: pre-training curves ----------
def fig_curves():
    jlog = PENGUIN_ROOT / "janusdna/checkpoints_final/train_janus_scratch_clean.log"
    clog = PENGUIN_ROOT / "crossdna/checkpoints_final/train_crossdna_scratch_60k.log"
    CROSS_STEPS_PER_EPOCH = 7591  # from the clean CrossDNA BED

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharex=True)
    ok = False

    # --- JanusDNA: val/main_loss, once per epoch ---
    if jlog.exists():
        v = [float(x) for x in
             re.findall(r"val/main_loss_epoch=([0-9.]+)", jlog.read_text())]
        v = list(dict.fromkeys(v))  # drop consecutive duplicates
        if v:
            ep = np.arange(1, len(v) + 1)
            axes[0].plot(ep, v, "-o", color=C_SCRATCH, ms=4)
            axes[0].set_title("JanusDNA (causal objective)")
            axes[0].set_xlabel("epoch")
            axes[0].set_ylabel("val loss (main)")
            axes[0].grid(alpha=0.3)
            ok = True

    # --- CrossDNA: per-step CE, averaged per epoch ---
    if clog.exists():
        ce = [float(x) for x in re.findall(r"CE ([0-9.]+)", clog.read_text())]
        if ce:
            ce = np.array(ce)
            n_ep = int(np.ceil(len(ce) / CROSS_STEPS_PER_EPOCH))
            ce_per_epoch = [ce[i * CROSS_STEPS_PER_EPOCH:(i + 1) * CROSS_STEPS_PER_EPOCH].mean()
                            for i in range(n_ep)]
            ep = np.arange(1, len(ce_per_epoch) + 1)
            axes[1].plot(ep, ce_per_epoch, "-o", color=C_GC, ms=4)
            axes[1].set_title("CrossDNA (masked objective)")
            axes[1].set_xlabel("epoch")
            axes[1].set_ylabel("CE (mean per epoch)")
            axes[1].grid(alpha=0.3)
            ok = True

    if not ok:
        print("  [skip] curves: logs not found")
        plt.close(fig)
        return

    fig.suptitle("Pre-training dynamics (x axis: epochs)", y=1.03)
    # The caption is not decoration. Two different objectives produce two loss
    # scales, and a reader comparing the absolute values would conclude
    # something false about which model trained better.
    fig.text(0.5, -0.04,
             "Note: absolute values are not comparable (different objectives and "
             "losses); what is compared is the dynamics. CrossDNA saturates "
             "within the first epoch.",
             ha="center", fontsize=8, style="italic")
    save(fig, "4_pretraining_curves")


# ---------- FIGURE 5: confusion matrices ----------
def fig_confusion():
    panels = [
        ("JanusDNA from-scratch", "lora_janus144_scratch_clean"),
        ("JanusDNA human", "lora_janus144_resume"),
        ("CrossDNA from-scratch", "lora_crossdna81_scratch_clean"),
        ("CrossDNA human", "lora_crossdna81_human_fp32"),
    ]
    data = []
    for title, folder in panels:
        p = OUT / folder / "lora_results.json"
        if not p.exists():
            print(f"  [skip] confusion: {folder} missing")
            return
        d = json.loads(p.read_text())
        cm = np.array(d["confusion_matrix"], dtype=float)
        classes = d.get("classes", ["CDS", "UTR", "intergenic", "intron"])
        data.append((title, cm, classes))

    fig, axes = plt.subplots(2, 2, figsize=(9, 8.4))
    for ax, (title, cm, classes) in zip(axes.flat, data):
        cmn = cm / cm.sum(axis=1, keepdims=True)  # row-normalised = recall
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("predicted", fontsize=9)
        ax.set_ylabel("true", fontsize=9)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
    fig.suptitle("Confusion matrices after LoRA (row-normalised, %)", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "5_confusion_matrices")


def main():
    global SAVE_PNG, FIG
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    SAVE_PNG = args.png
    FIG = Path(args.out_dir)
    FIG.mkdir(parents=True, exist_ok=True)
    print(f"=== figures in {FIG} (reading results from {OUT}) ===")
    fig_zero_vs_lora()
    fig_nmi()
    fig_contamination()
    fig_curves()
    fig_confusion()
    print("done.")


if __name__ == "__main__":
    main()
