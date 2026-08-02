#!/usr/bin/env python
"""
audit_results.py — scan the workspace for every saved result file and produce a
single consolidated report.

The point is not just to list numbers. It is to catch the failure mode that
actually bites when writing up: the SAME embedding evaluated under two DIFFERENT
probe protocols, producing two different means that both look authoritative.
Results are therefore grouped by protocol signature (embedding file, hidden
size, dropout), not merely by feature name, and any feature evaluated under more
than one signature is flagged.

Usage:
    python audit_results.py                          # scans $PENGUIN_ROOT
    python audit_results.py --root /path/to/workspace
    python audit_results.py --out audit.md           # also write a markdown file
    python audit_results.py --filter janus           # only matching features
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

# Files we know how to read, and the kind of result they hold.
KNOWN = {
    "mlp_probe_results.json": "probe",
    "mlp_results.json": "probe",
    "lora_results.json": "lora",
    "finetune_results.json": "lora",
    "eval_test.json": "lora_eval",
    "clustering_results.json": "cluster",
    "classification_results.json": "classify",
}

# Directories that never contain results worth auditing.
SKIP_DIRS = {
    ".git", "__pycache__", "wandb", "lightning_logs", "node_modules",
    "site-packages", "env_crossdna", ".venv", "venv",
    # archived / superseded runs: scanning them mixes abandoned attempts into
    # the aggregates. Override with --include-archived if you need to look.
    "Da_eliminare", "_archivio", "archive", "old", "trash",
}


def run_family(path: str) -> str:
    """The run directory with any seed suffix stripped: mlp_probe_seed3 -> mlp_probe."""
    d = os.path.basename(os.path.dirname(path))
    # The separator is required: without it, "clf_janus144" would be truncated
    # to "clf_janu" because "s144" looks like a seed suffix.
    return re.sub(r"[_-](?:seed\d+|s\d+)$", "", d) or d


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def extract(path: Path, kind: str):
    """Normalise one result file into a list of records."""
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]

    items = data if isinstance(data, list) else [data]
    recs = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # some files nest everything under "results"
        body = item.get("results") if isinstance(item.get("results"), dict) else item

        # Some result files (the single-run probe and LoRA outputs) record no
        # feature name at all. Falling back to a literal "?" would collapse
        # every such file into one bucket and manufacture a fake conflict, so
        # the run directory is used as the identity instead.
        feature = (body.get("feature") or body.get("emb") or body.get("model")
                   or item.get("feature") or item.get("emb") or item.get("model")
                   or f"[{run_family(str(path))}]")
        clf = body.get("clf") or body.get("classifier") or ""

        rec = {
            "kind": kind,
            "feature": str(feature),
            "clf": str(clf),
            "mcc": _num(body.get("mcc")) or _num(body.get("test_mcc")),
            "acc": _num(body.get("accuracy")) or _num(body.get("test_accuracy")),
            "f1": _num(body.get("macro_f1")) or _num(body.get("test_macro_f1")),
            "nmi_class": _num(body.get("NMI_class")),
            "nmi_gc": _num(body.get("NMI_vs_GC")),
            "dim": body.get("dim"),
            # protocol signature: what makes two runs comparable or not
            "emb_file": str(body.get("emb_file") or ""),
            "hidden": body.get("hidden"),
            "dropout": body.get("dropout"),
            "lora_r": body.get("lora_r"),
            "path": str(path),
        }
        if rec["mcc"] is None and rec["nmi_class"] is None:
            continue
        recs.append(rec)
    return recs


def signature(r):
    """What distinguishes one evaluation protocol from another.

    Window length is read from the path (out_L0500/...) because it is the single
    biggest driver of the numbers and is not recorded inside the result files.
    """
    parts = []
    mL = re.search(r"out_L0*(\d+)", r["path"])
    if mL:
        parts.append(f"L={int(mL.group(1))}")
    if r["emb_file"]:
        parts.append(os.path.basename(r["emb_file"]))
    if r["hidden"]:
        parts.append(f"hidden={r['hidden']}")
    if r["dropout"] is not None:
        parts.append(f"drop={r['dropout']}")
    if r["lora_r"]:
        parts.append(f"r={r['lora_r']}")
    parts.append(run_family(r["path"]))
    return " | ".join(parts)


def fmt(m, s, n):
    if n <= 1:
        return f"{m:.4f}"
    return f"{m:.4f} ± {s:.4f}"


def default_root() -> str:
    """PENGUIN_ROOT if set, otherwise the conventional path, otherwise the
    current directory.

    The fallback matters inside containers: HOME is /root there, so the
    conventional default points at a directory that does not exist, while the
    working directory is almost always the workspace itself.
    """
    env = os.environ.get("PENGUIN_ROOT")
    if env:
        return env
    conventional = Path.home() / "penguin_glm"
    if conventional.is_dir():
        return str(conventional)
    return os.getcwd()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=default_root(),
                    help="workspace to scan (default: $PENGUIN_ROOT, else the "
                         "current directory)")
    ap.add_argument("--out", help="also write the report to this markdown file")
    ap.add_argument("--filter", default="", help="only features matching this substring")
    ap.add_argument("--include-archived", action="store_true",
                    help="also scan archived directories (Da_eliminare, _archivio, ...)")
    args = ap.parse_args()

    skip = set(SKIP_DIRS)
    if args.include_archived:
        skip -= {"Da_eliminare", "_archivio", "archive", "old", "trash"}

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(
            f"Not a directory: {root}\n"
            f"Pass --root <path>, or set PENGUIN_ROOT to your workspace."
        )
    print(f"[scanning] {root}\n")

    records, errors = [], []
    seen_hashes = {}
    duplicates = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            kind = KNOWN.get(fn)
            if not kind:
                continue
            full = Path(dirpath) / fn
            # Consolidated result directories are usually copies of the working
            # ones. Counting a file twice would silently halve the apparent
            # spread, so identical content is read once.
            try:
                h = hashlib.md5(full.read_bytes()).hexdigest()
            except Exception:
                h = None
            if h and h in seen_hashes:
                duplicates.append((str(full), seen_hashes[h]))
                continue
            if h:
                seen_hashes[h] = str(full)
            for r in extract(full, kind):
                if "error" in r:
                    errors.append((str(Path(dirpath) / fn), r["error"]))
                else:
                    records.append(r)

    if args.filter:
        records = [r for r in records if args.filter.lower() in r["feature"].lower()]

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# Results audit — {root}")
    emit()
    emit(f"{len(records)} records from {len({r['path'] for r in records})} files.")
    if duplicates:
        emit(f"{len(duplicates)} byte-identical copies skipped "
             "(consolidated directories mirroring the working ones).")
    emit()

    # ---------------- aggregate by kind / feature / protocol ---------------- #
    for kind in ("probe", "lora", "lora_eval", "cluster", "classify"):
        sub = [r for r in records if r["kind"] == kind]
        if not sub:
            continue
        emit(f"## {kind}")
        emit()
        if kind == "cluster":
            emit("| feature | n | NMI function | NMI GC | ratio |")
            emit("|---|---:|---:|---:|---:|")
            g = defaultdict(list)
            for r in sub:
                g[r["feature"]].append(r)
            for feat, rs in sorted(g.items()):
                fn = [r["nmi_class"] for r in rs if r["nmi_class"] is not None]
                gc = [r["nmi_gc"] for r in rs if r["nmi_gc"] is not None]
                if not fn or not gc:
                    continue
                mf, mg = st.mean(fn), st.mean(gc)
                emit(f"| {feat} | {len(rs)} | {mf:.4f} | {mg:.4f} | {mg/max(mf,1e-9):.1f}x |")
            emit()
            continue

        emit("| feature | protocol signature | n | MCC | accuracy |")
        emit("|---|---|---:|---:|---:|")
        g = defaultdict(list)
        for r in sub:
            g[(r["feature"], r["clf"], signature(r))].append(r)
        for (feat, clf, sig), rs in sorted(g.items()):
            mccs = [r["mcc"] for r in rs if r["mcc"] is not None]
            accs = [r["acc"] for r in rs if r["acc"] is not None]
            if not mccs:
                continue
            m, s = st.mean(mccs), (st.stdev(mccs) if len(mccs) > 1 else 0.0)
            a = f"{st.mean(accs):.4f}" if accs else "-"
            label = f"{feat} ({clf})" if clf else feat
            emit(f"| {label} | {sig} | {len(mccs)} | {fmt(m, s, len(mccs))} | {a} |")
        emit()

    # ---------------- the flag that matters --------------------------------- #
    emit("## Features evaluated under more than one protocol")
    emit()
    emit("If a feature appears below, the two means are NOT interchangeable and "
         "the thesis must say which protocol each number comes from.")
    emit()
    bysig = defaultdict(set)
    for r in records:
        if r["kind"] in ("probe", "lora") and r["mcc"] is not None:
            bysig[r["feature"]].add(signature(r))
    flagged = {f: s for f, s in bysig.items() if len(s) > 1}
    if not flagged:
        emit("None. Every feature was evaluated under a single protocol.")
    else:
        for feat, sigs in sorted(flagged.items()):
            emit(f"**{feat}**")
            for sig in sorted(sigs):
                rs = [r for r in records if r["feature"] == feat and signature(r) == sig
                      and r["mcc"] is not None]
                mccs = [r["mcc"] for r in rs]
                m = st.mean(mccs)
                s = st.stdev(mccs) if len(mccs) > 1 else 0.0
                paths = sorted({os.path.dirname(r["path"]).replace(str(root), ".") for r in rs})
                emit(f"- `{sig}` → n={len(mccs)}, MCC {fmt(m, s, len(mccs))}")
                emit(f"  - {', '.join(paths[:4])}{' ...' if len(paths) > 4 else ''}")
            emit()

    # ---------------- per-seed detail for the probe runs -------------------- #
    emit("## Per-seed detail (probe)")
    emit()
    seeded = defaultdict(list)
    for r in records:
        if r["kind"] == "probe" and r["mcc"] is not None:
            seed = re.search(r"seed(\d+)|_s(\d+)", r["path"])
            tag = seed.group(0) if seed else os.path.basename(os.path.dirname(r["path"]))
            seeded[(r["feature"], signature(r))].append((tag, r["mcc"]))
    for (feat, sig), vals in sorted(seeded.items()):
        if len(vals) < 2:
            continue
        vals.sort()
        joined = ", ".join(f"{t}={v:.4f}" for t, v in vals)
        emit(f"- **{feat}** `{sig}` → {joined}")
    emit()

    if errors:
        emit("## Unreadable files")
        emit()
        for p, e in errors:
            emit(f"- `{p}` — {e}")

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
