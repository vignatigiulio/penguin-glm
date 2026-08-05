#!/usr/bin/env python
"""
diagnose_sdpa_patch.py — find out exactly where the SDPA replacement diverges
from flash-attn, and on what inputs.

Rather than swapping the implementation and seeing whether the model survives,
this intercepts every flash_attn_func call, runs BOTH implementations on the
same inputs, and reports them side by side. The forward continues with the
ORIGINAL result, so the model stays healthy and every layer gets measured
instead of only the first one that breaks.

What it reveals, in order of usefulness:

  1. the kwargs fla actually passes (causal, window_size, softmax_scale). If the
     replacement mishandles one of them, that is the bug, and no amount of
     precision work will fix it.
  2. whether the INPUTS are already non-finite. If they are, the attention is a
     victim rather than the culprit.
  3. which layer first diverges, and by how much.

By default the forward proceeds with the ORIGINAL result, so every layer is
measured on healthy inputs. --use-replacement instead lets the replacement drive
the forward, which is what certify_fp32_patch.py does: divergences then compound
from layer to layer, and a discrepancy invisible on healthy inputs can grow into
a non-finite value. Run both. Agreement in the first mode with failure in the
second means the replacement is correct per call but not stable under
composition.

--repeat-frac reproduces certify's stimulus: tandem repeats drive attention into
its most degenerate regime, where many keys are near-identical and the softmax
is at its flattest.

Usage:
  python diagnose_sdpa_patch.py --weights $CROSSDNA_REPO/crossdna_inference
  python diagnose_sdpa_patch.py --weights .../519M --amp-fp32
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "env" / "patches"))

BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--length", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--repeat-frac", type=float, default=0.0,
                    help="fraction of tandem-repeat sequences (certify uses 0.3)")
    ap.add_argument("--use-replacement", action="store_true",
                    help="let the replacement drive the forward, as certify does")
    ap.add_argument("--amp-fp32", action="store_true",
                    help="run the model in fp32 (as the extraction pipeline does)")
    ap.add_argument("--max-calls", type=int, default=12,
                    help="how many attention calls to report")
    ap.add_argument("--force-dtype", default="fp32", choices=["fp32", "same"],
                    help="working precision of the replacement, as install() "
                         "would set it. 'same' reproduces the diagnostic mode.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if hasattr(torch, "compile"):
        torch.compile = (lambda fn=None, *a, **k:
                         (fn if fn is not None else (lambda g: g)))

    import fla.layers.attn as A
    import sdpa_fp32
    from transformers import AutoModelForMaskedLM

    original = A.flash_attn_func
    if original is None:
        raise SystemExit("fla.layers.attn.flash_attn_func not found")

    # Go through install() rather than calling flash_attn_func_fp32 by hand:
    # the working precision is module state that install() owns, and reaching
    # past it measures a configuration nobody actually runs.
    sdpa_fp32.install(force_dtype=(None if args.force_dtype == "same" else "fp32"))
    replacement = A.flash_attn_func
    print(f"  replacement working precision: {args.force_dtype}")

    rows = []

    def probe(q, k, v, *pargs, **kw):
        ref = original(q, k, v, *pargs, **kw)
        try:
            new = replacement(q, k, v, *pargs, **kw)
            err = None
        except Exception as e:
            new, err = None, f"{type(e).__name__}: {e}"

        def stat(t):
            if t is None:
                return dict(finite=False, absmax=float("nan"))
            f = torch.isfinite(t)
            return dict(finite=bool(f.all()),
                        absmax=float(t[f].abs().max()) if f.any() else float("inf"),
                        nonfinite=int((~f).sum()))

        rec = dict(
            n=len(rows),
            kwargs={kk: vv for kk, vv in kw.items() if kk != "return_attn_probs"},
            pargs=len(pargs),
            dtype=str(q.dtype).replace("torch.", ""),
            shape=tuple(q.shape),
            q=stat(q), k=stat(k), v=stat(v),
            ref=stat(ref), new=stat(new), err=err,
        )
        if err is None and rec["ref"]["finite"] and rec["new"]["finite"]:
            a = ref.float().flatten()
            b = new.float().flatten()
            rec["cos"] = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))
            rec["rel"] = float((a - b).norm() / (a.norm() + 1e-12))
        rec["refmax"] = rec["ref"]["absmax"]
        rec["newmax"] = rec["new"]["absmax"]
        rows.append(rec)
        if args.use_replacement and new is not None:
            return new
        return ref

    A.flash_attn_func = probe

    print(f"Loading {args.weights} (amp_fp32={args.amp_fp32}) ...")
    m = AutoModelForMaskedLM.from_pretrained(
        args.weights, trust_remote_code=True, local_files_only=True
    ).float().cuda().eval()
    if args.amp_fp32:
        mod = sys.modules.get(type(m).__module__)
        if mod is not None and hasattr(mod, "preferred_amp_dtype"):
            mod.preferred_amp_dtype = lambda: torch.float32
            print("  [amp-fp32] preferred_amp_dtype -> float32")
    m.backbone.pretrain = False
    m.backbone.for_representation = True

    random.seed(args.seed)
    seqs = []
    for _ in range(args.batch):
        if random.random() < args.repeat_frac:
            unit = "".join(random.choice("ACGT")
                           for _ in range(random.choice([2, 3, 4])))
            seqs.append((unit * (args.length // len(unit) + 1))[:args.length])
        else:
            seqs.append("".join(random.choice("ACGT")
                                for _ in range(args.length)))
    print(f"stimulus: {args.batch} sequences of {args.length} bp, "
          f"repeat_frac={args.repeat_frac}, "
          f"driving forward with {'REPLACEMENT' if args.use_replacement else 'original'}")
    x = torch.tensor([[BASE[c] for c in s] for s in seqs]).cuda()

    driver = "REPLACEMENT" if args.use_replacement else "original"
    try:
        with torch.inference_mode():
            m.backbone(x)
        print(f"forward completed (driven by the {driver} implementation)\n")
    except Exception as e:
        print(f"forward INTERRUPTED (driven by the {driver} implementation): "
              f"{type(e).__name__}: {e}")
        print("The table below still reports every call made before the stop.\n")

    if not rows:
        raise SystemExit("flash_attn_func was never called: this model does not "
                         "route through fla.layers.attn, so the patch is irrelevant "
                         "to it.")

    print(f"=== {len(rows)} attention calls intercepted "
          f"(showing the first {min(args.max_calls, len(rows))}) ===\n")
    print("kwargs passed by fla on the first call:")
    print(f"  positional args: {rows[0]['pargs']}")
    for kk, vv in rows[0]["kwargs"].items():
        print(f"  {kk} = {vv!r}")
    print()

    print(f"attention input shape [B, T, H, D]: {rows[0]['shape']}"
          f"  -> {int(np.prod(rows[0]['shape']))} values per output tensor\n")

    hdr = (f"{'#':>3} {'dtype':>8} {'|q|max':>10} {'|ref|max':>10} "
           f"{'|new|max':>10} {'ref fin':>8} {'new fin':>8} "
           f"{'cosine':>10} {'rel':>10}  note")
    print(hdr)
    print("-" * len(hdr))
    for r in rows[:args.max_calls]:
        cos = f"{r['cos']:10.6f}" if "cos" in r else " " * 10
        rel = f"{r['rel']:10.3e}" if "rel" in r else " " * 10
        note = r["err"] or ""
        if not r["new"]["finite"] and r["err"] is None:
            nf = r["new"].get("nonfinite")
            tot = int(np.prod(r["shape"]))
            note = (f"replacement non-finite ({nf}/{tot} = "
                    f"{100*nf/max(tot,1):.0f}% of the output)")
        if not r["ref"]["finite"]:
            note = "REFERENCE already non-finite " + note
        print(f"{r['n']:3d} {r['dtype']:>8} {r['q']['absmax']:10.3e} "
              f"{r['refmax']:10.3e} {r['newmax']:10.3e} "
              f"{str(r['ref']['finite']):>8} "
              f"{str(r['new']['finite']):>8} {cos} {rel}  {note}")

    bad = [r for r in rows if r["err"] or not r["new"]["finite"]]
    div = [r for r in rows if "rel" in r and r["rel"] > 1e-2]
    print()
    if bad:
        first = bad[0]
        print(f"FIRST FAILURE at call {first['n']}: "
              f"{first['err'] or 'non-finite output'}")
        print(f"  inputs finite: q={first['q']['finite']} k={first['k']['finite']} "
              f"v={first['v']['finite']}  |q|max={first['q']['absmax']:.3e}")
        if first["ref"]["finite"]:
            print("  flash-attn produced a FINITE result on the same inputs, so "
                  "this is a defect of the replacement, not of the data.")
    elif div:
        print(f"No failure, but {len(div)} calls diverge by more than 1e-2 "
              "relative. Check mask, scale and window handling.")
    else:
        print("The replacement matches flash-attn on every intercepted call.")

    A.flash_attn_func = original


if __name__ == "__main__":
    main()
