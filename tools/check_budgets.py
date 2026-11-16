#!/usr/bin/env python3
"""
check_budgets.py -- which checkpoint is the "baseline (no stem)" of the stem sweep,
and how many optimisation steps was each run trained for?

Reads global_step directly from the checkpoints (more reliable than the logs) and
compares file identity: if the sweep baseline IS the from-scratch reference model,
the question of budget parity is settled at once.

USAGE
    python check_budgets.py                          # auto-discovery under $PENGUIN_ROOT
    python check_budgets.py --root $PENGUIN_ROOT
    python check_budgets.py a.ckpt b.ckpt --fingerprint
    python check_budgets.py --seqlen 131072 --batch 8
NOTES
NOTE
    - Works in either environment (only torch is needed).
    - Avoids loading the weights when it can; large Lightning checkpoints can still
      take a few seconds and ~1 GB of RAM.
"""

import argparse
import glob
import os
import sys

# ---------------------------------------------------------------- discovery ---

# typical patterns: add more with --extra-glob
DEFAULT_PATTERNS = [
    "**/last.ckpt",
    "**/*.ckpt",
]

# heuristic used to label the runs in the output
LABEL_HINTS = [
    ("stem_s4", "stem S=4"),
    ("stem_s8", "stem S=8"),
    ("stem_s16", "stem S=16"),
    ("stem_s32", "stem S=32"),
    ("_s4", "stem S=4"),
    ("_s8", "stem S=8"),
    ("_s16", "stem S=16"),
    ("_s32", "stem S=32"),
    ("baseline", "baseline (no stem)"),
    ("no_stem", "baseline (no stem)"),
    ("nostem", "baseline (no stem)"),
    ("clean", "from-scratch clean"),
    ("scratch", "from-scratch"),
    ("crossdna", "CrossDNA"),
    ("janus", "JanusDNA"),
]


def label_for(path):
    low = path.lower()
    hits = []
    for key, lab in LABEL_HINTS:
        if key in low and lab not in hits:
            hits.append(lab)
    return " / ".join(hits) if hits else "?"


def discover(root, extra_globs, limit):
    found = []
    for pat in DEFAULT_PATTERNS + list(extra_globs):
        full = os.path.join(root, pat)
        for p in glob.glob(full, recursive=True):
            if os.path.isfile(p) and p not in found:
                found.append(p)
    # most recent first: usually the ones of interest
    found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if limit and len(found) > limit:
        print(f"[i] found {len(found)} checkpoints, inspecting the {limit} most recent "
              f"(use --limit 0 for all)\n")
        found = found[:limit]
    return found


# ------------------------------------------------------------------ reading ---

def load_ckpt(path):
    """Load a checkpoint, trying the light path first."""
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True), "weights_only"
    except Exception:
        pass
    try:
        return torch.load(path, map_location="cpu", weights_only=False), "full"
    except Exception as e:
        return None, f"ERROR: {type(e).__name__}: {e}"


def dig(d, *keys):
    """Look for a key at any reasonable level of the dict."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            return d[k]
    for sub in ("hyper_parameters", "hparams", "config", "cfg", "callbacks"):
        v = d.get(sub)
        if isinstance(v, dict):
            r = dig(v, *keys)
            if r is not None:
                return r
    return None


def hydra_config_near(ckpt_path):
    """Look for the Hydra config saved next to the run."""
    d = os.path.dirname(os.path.abspath(ckpt_path))
    for _ in range(4):  # climb a few levels
        cand = os.path.join(d, ".hydra", "config.yaml")
        if os.path.isfile(cand):
            return cand
        d = os.path.dirname(d)
    return None


def grep_config(cfg_path, keys=("max_steps", "global_batch", "max_length",
                                "batch_size", "seqlen", "lr")):
    out = {}
    try:
        with open(cfg_path, "r", errors="replace") as f:
            for line in f:
                s = line.strip()
                for k in keys:
                    if s.startswith(k + ":") or f" {k}:" in s:
                        out.setdefault(k, s)
    except Exception:
        pass
    return out


def fingerprint(ckpt, n_tensors=5):
    """Cheap fingerprint: spots two different files that are the same model."""
    import torch
    sd = None
    for key in ("state_dict", "model", "module"):
        if isinstance(ckpt, dict) and isinstance(ckpt.get(key), dict):
            sd = ckpt[key]
            break
    if sd is None and isinstance(ckpt, dict):
        sd = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
    if not sd:
        return None
    vals = []
    for k in sorted(sd.keys()):
        v = sd[k]
        if torch.is_tensor(v) and v.numel() > 0 and v.is_floating_point():
            vals.append(f"{k}={float(v.detach().float().abs().sum()):.6e}")
        if len(vals) >= n_tensors:
            break
    return " | ".join(vals) if vals else None


# --------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="*", help="explicit checkpoints (otherwise auto-discovery)")
    ap.add_argument("--root", default=os.environ.get("PENGUIN_ROOT",
                                                     os.path.expanduser("~/penguin_glm")))
    ap.add_argument("--extra-glob", action="append", default=[],
                    help="additional pattern, repeatable")
    ap.add_argument("--limit", type=int, default=12, help="0 = no limit")
    ap.add_argument("--seqlen", type=int, default=131072,
                    help="sequence length for the token count")
    ap.add_argument("--batch", type=int, default=8, help="global batch")
    ap.add_argument("--fingerprint", action="store_true",
                    help="weight fingerprint: detects copies under a different name")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("torch not available: activate the environment (conda activate janusdna) and retry.")

    paths = args.ckpts or discover(args.root, args.extra_glob, args.limit)
    if not paths:
        sys.exit(f"no checkpoint found under {args.root}\n"
                 f"pass the paths explicitly or use --root / --extra-glob")

    rows = []
    print(f"{'='*100}\nCHECKPOINT ISPEZIONATI\n{'='*100}\n")

    for p in paths:
        st = os.stat(p)
        ck, mode = load_ckpt(p)
        if ck is None:
            print(f"  {p}\n    {mode}\n")
            continue

        step = dig(ck, "global_step")
        epoch = dig(ck, "epoch")
        tok = (step * args.batch * args.seqlen / 1e9) if isinstance(step, int) else None

        print(f"  {p}")
        print(f"    label       : {label_for(p)}")
        print(f"    global_step : {step}    epoch: {epoch}    [read via: {mode}]")
        if tok is not None:
            print(f"    est. tokens : {tok:.1f} B   (= step x {args.batch} x {args.seqlen})")
        print(f"    identity    : realpath={os.path.realpath(p)}")
        print(f"                  dev={st.st_dev} inode={st.st_ino} size={st.st_size/1e6:.0f} MB")

        cfg = hydra_config_near(p)
        if cfg:
            g = grep_config(cfg)
            if g:
                print(f"    hydra       : {cfg}")
                for k, v in g.items():
                    print(f"                  {v}")

        fp = fingerprint(ck) if args.fingerprint else None
        if fp:
            print(f"    fingerprint : {fp}")

        rows.append(dict(path=p, label=label_for(p), step=step, epoch=epoch, tok=tok,
                         ino=(st.st_dev, st.st_ino), fp=fp,
                         real=os.path.realpath(p)))
        print()
        del ck

    # --------------------------------------------------------------- summary ---
    print(f"{'='*100}\nRIEPILOGO\n{'='*100}")
    print(f"{'global_step':>12}  {'tokens(B)':>11}  {'label':<26}  path")
    for r in sorted(rows, key=lambda r: (r["step"] is None, r["step"] or 0)):
        s = r["step"] if r["step"] is not None else "?"
        t = f"{r['tok']:.1f}" if r["tok"] is not None else "?"
        print(f"{str(s):>12}  {t:>11}  {r['label']:<26}  {r['path']}")

    # ------------------------------------------------------------- verdict ---
    print(f"\n{'='*100}\nVERDICT ON BUDGET PARITY\n{'='*100}")

    by_real, by_fp = {}, {}
    for r in rows:
        by_real.setdefault(r["real"], []).append(r["path"])
        if r["fp"]:
            by_fp.setdefault(r["fp"], []).append(r["path"])

    dup = False
    for key, ps in list(by_real.items()) + list(by_fp.items()):
        if len(ps) > 1:
            dup = True
            print("  SAME MODEL under several paths:")
            for x in ps:
                print(f"    - {x}")
    if not dup and args.fingerprint:
        print("  No duplicate checkpoint among those inspected.")
    elif not args.fingerprint:
        print("  (run again with --fingerprint to find copies under a different name)")

    steps = sorted({r["step"] for r in rows if isinstance(r["step"], int)})
    if len(steps) > 1:
        print(f"\n  Distinct step counts observed: {steps}")
        print("  If the sweep baseline and the from-scratch reference have DIFFERENT step counts,")
        print("  budget parity does not hold and must be stated as a confound.")
        print("  If they have the SAME step count (or are the same file), parity is confirmed.")
    elif len(steps) == 1:
        print(f"\n  All inspected checkpoints have global_step = {steps[0]}: budgets match.")


if __name__ == "__main__":
    main()
