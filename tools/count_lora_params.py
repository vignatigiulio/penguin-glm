#!/usr/bin/env python3
"""
count_lora_params.py: count, from the checkpoint files alone, the parameters
that LoRA adapts in CrossDNA, separating the student branches from the teacher
EMA copy. No GPU and, for .safetensors, no torch: only the file headers are read.

Why it exists: src/finetune_lora_crossdna.py injects adapters by matching
projection names, so the teacher EMA copy of the two branches (used only during
pre-training, never executed during adaptation) gets adapters too. The raw count
in the run logs therefore doubles the adapted projections. This script gives
both views:

  * all tensors, teacher EMA tensors ("_ema" in the name), and the rest;
  * LoRA targets (q/k/v/o of the Comba mixer and of the sliding-window
    attention, both branches), per side, and the adapter parameters for rank r;
  * the effective trainable share: (student adapters + classification head)
    over (non-teacher parameters + student adapters).

On the released CrossDNA-519M it gives the counts reported in the thesis: 192
adapted projections and 3,149,828 trainable parameters, head included, out of
522,172,165. On CrossDNA-8.1M, 96 projections and 344,580 parameters, while the
run files count 192 modules and 688,644 parameters because they include the
teacher copy. The runtime count printed by finetune_lora_crossdna.py can differ
from the header count by a few tensors stored in the file but not registered as
parameters.

Usage:
  python tools/count_lora_params.py $PENGUIN_ROOT/weights_519M
  python tools/count_lora_params.py $PENGUIN_ROOT --rank 8 --n-classes 4
"""
import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

TARGETS = tuple(f"{m}.{p}_proj.weight" for m in ("comba", "mixer") for p in "qkvo")


def tensor_shapes(f):
    """{name: shape} from a .safetensors header, or from a pytorch_model.bin."""
    if f.suffix == ".safetensors":
        with open(f, "rb") as fh:
            hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        return {k: v["shape"] for k, v in hdr.items() if k != "__metadata__"}
    if f.stat().st_size > 1.5e9:
        print(f"[skipped, too large for torch.load] {f}")
        return {}
    import torch
    sd = torch.load(f, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    return {k: list(v.shape) for k, v in sd.items() if hasattr(v, "shape")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory searched recursively for checkpoints")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--n-classes", type=int, default=4)
    args = ap.parse_args()

    folders = defaultdict(list)
    for f in Path(args.root).rglob("*"):
        if f.is_file() and (f.suffix == ".safetensors" or f.name == "pytorch_model.bin"):
            folders[f.parent].append(f)
    if not folders:
        sys.exit(f"no .safetensors or pytorch_model.bin under {args.root}")

    for d, files in sorted(folders.items()):
        shapes = {}
        for f in files:
            try:
                shapes.update(tensor_shapes(f))
            except Exception as e:
                print(f"[unreadable] {f}: {e}")
        if not any(k.endswith(TARGETS) for k in shapes):
            continue
        total = teacher = 0
        targets = {"student": 0, "teacher": 0}
        lora = {"student": 0, "teacher": 0}
        hidden = None
        for k, s in shapes.items():
            n = 1
            for x in s:
                n *= x
            total += n
            side = "teacher" if "_ema" in k else "student"
            if side == "teacher":
                teacher += n
            if k.endswith(TARGETS):
                targets[side] += 1
                lora[side] += args.rank * sum(s)   # A: r x in, B: out x r
                if hidden is None and len(s) == 2:
                    hidden = s[1]
        head = hidden * args.n_classes + args.n_classes if hidden else 0
        eff_num = lora["student"] + head
        eff_den = (total - teacher) + lora["student"]
        raw_num = lora["student"] + lora["teacher"] + head
        raw_den = total + lora["student"] + lora["teacher"]
        print(f"{d}  [{', '.join(f.name for f in files)}]")
        print(f"  tensors in total {total:,} | teacher EMA {teacher:,} | without teacher {total - teacher:,}")
        print(f"  LoRA targets {targets} | adapter parameters at r={args.rank} {lora}")
        print(f"  head Linear({hidden}->{args.n_classes}) = {head:,} parameters")
        print(f"  effective (teacher excluded): {eff_num:,} / {eff_den:,} = {100 * eff_num / eff_den:.3f}%")
        print(f"  raw (teacher included):       {raw_num:,} / {raw_den:,} = {100 * raw_num / raw_den:.3f}%")


if __name__ == "__main__":
    main()
