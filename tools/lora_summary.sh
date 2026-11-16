#!/bin/bash
# lora_summary.sh: one table with every LoRA result under $PENGUIN_ROOT/out, and
# a check that the runs share the same LoRA hyperparameters.
#
# The reference signature is the JanusDNA-144 one (r=8, 27 adapted modules,
# 102,276 trainable parameters). Runs with a different signature are listed as
# not directly comparable. "best/ep" is the epoch of the validation maximum over
# the number of epochs run: a maximum on the last epochs means not saturated.
#
# Usage:  bash tools/lora_summary.sh            (E defaults to $PENGUIN_ROOT/out)
: "${PENGUIN_ROOT:=$HOME/penguin_glm}"
: "${E:=$PENGUIN_ROOT/out}"
python3 - "$E" << 'PY'
import json, sys, glob, os
E = sys.argv[1]
rows = []
for p in sorted(glob.glob(f"{E}/lora*/lora_results.json")):
    name = os.path.basename(os.path.dirname(p))
    if name.endswith("_smoke"):
        continue
    try:
        d = json.load(open(p))
        l = d.get("lora", {})
        h = d.get("history", [])
        best = max(range(len(h)), key=lambda i: h[i]["val_mcc"]) + 1 if h else None
        rows.append(dict(name=name, mcc=d.get("test_mcc"), acc=d.get("test_accuracy"),
                         f1=d.get("test_macro_f1"), r=l.get("r"), alpha=l.get("alpha"),
                         inj=l.get("injected"), par=d.get("trainable_params"),
                         best=best, nep=len(h), ckpt=d.get("checkpoint", "")))
    except Exception as e:
        print(f"  [skip] {name}: {e}")

print(f"{'run':34}{'MCC':>8}{'acc':>8}{'F1':>8}{'r':>4}{'inj':>5}{'par':>9}{'best/ep':>9}")
print("-" * 85)
for x in sorted(rows, key=lambda z: -(z["mcc"] or 0)):
    m = f"{x['mcc']:.4f}" if x["mcc"] is not None else "   -  "
    a = f"{x['acc']:.4f}" if x["acc"] is not None else "   -  "
    f = f"{x['f1']:.4f}" if x["f1"] is not None else "   -  "
    be = f"{x['best']}/{x['nep']}" if x["best"] else "  -  "
    print(f"{x['name']:34}{m:>8}{a:>8}{f:>8}{str(x['r']):>4}{str(x['inj']):>5}"
          f"{str(x['par']):>9}{be:>9}")

ref = [x for x in rows if x["par"] == 102276 and x["r"] == 8 and x["inj"] == 27]
print(f"\n{len(ref)}/{len(rows)} runs with identical hyperparameters (r=8, inj=27, par=102276)")
odd = [x["name"] for x in rows if x not in ref]
if odd:
    print("NOT directly comparable:", ", ".join(odd))

s4o = next((x for x in rows if x["name"] == "lora_janus144_stem_s4"), None)
s4n = next((x for x in rows if x["name"] == "lora_janus144_stem_s4_b4"), None)
if s4o and s4n:
    d = s4n["mcc"] - s4o["mcc"]
    print(f"\npatch v1 -> v2 reproducibility on S=4: {s4o['mcc']:.4f} -> {s4n['mcc']:.4f} "
          f"({d/s4o['mcc']*100:+.2f}%)")
PY
