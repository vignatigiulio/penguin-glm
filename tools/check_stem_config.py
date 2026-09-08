#!/usr/bin/env python3
"""
check_stem_config.py — derive the stem configuration from a checkpoint and
refuse to proceed if the environment disagrees with it.

WHY THIS EXISTS
---------------
Downstream scripts (extraction, LoRA fine-tuning) rebuild the model from
`model_config.json` plus a set of environment variables, then load the
checkpoint with `strict=False`. Two kinds of mismatch are dangerous because
they do not raise:

  * A wrong number of stem stages leaves only four unexpected keys
    (`stem.blocks.<n>.conv.weight/bias`, `.norm.weight/bias`), which is below
    the tolerance of a count-based gate.
  * A wrong BLOCK changes the exclusion band `BAND = BLOCK / STRIDE`, which
    governs the final fusion offsets. The band is not a parameter, so a
    mismatch produces no missing or unexpected key at all: the embeddings are
    simply computed with a fusion the model was never trained with.

This script closes both holes by reading the structure out of the weights.

WHAT CAN AND CANNOT BE DERIVED
------------------------------
  n_stages, kernel_size, block, vocab, d_model   -> read from tensor shapes
  stride                                          -> NOT derivable: a 3-stage
      stride-1 stem and a 3-stage stride-2 stem have identical weight shapes.
      It must be supplied, and is cross-checked for consistency (for stride > 1
      the builder requires n_stages == log2(stride)).

USAGE
  python check_stem_config.py --checkpoint <run>/checkpoints/last.ckpt        # report
  python check_stem_config.py --checkpoint ... --stride 1 --block 4 --nstages 3
  eval "$(python check_stem_config.py --checkpoint ... --stride 1 --export)"  # emit env
"""

import argparse
import re
import sys

import torch

STAGE_RE = re.compile(r"stem\.blocks\.(\d+)\.")


def load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"] if "state_dict" in ckpt else ckpt


def find(sd, needle):
    for k, v in sd.items():
        if k.endswith(needle) or needle in k:
            return k, v
    return None, None


def derive(sd):
    cfg = {}

    stages = {int(m.group(1)) for k in sd for m in [STAGE_RE.search(k)] if m}
    cfg["n_stages"] = len(stages) if stages else 0
    if stages and sorted(stages) != list(range(len(stages))):
        raise SystemExit(f"[ABORT] non-contiguous stem stage indices: {sorted(stages)}")

    _, w = find(sd, "stem.blocks.0.conv.conv.weight")
    if w is None:
        _, w = find(sd, "stem.blocks.0.conv.weight")
    cfg["kernel"] = int(w.shape[-1]) if w is not None else None
    cfg["d_model"] = int(w.shape[0]) if w is not None else None

    _, emb = find(sd, "embed_tokens.weight")
    cfg["vocab"] = int(emb.shape[0]) if emb is not None else None
    if cfg["d_model"] is None and emb is not None:
        cfg["d_model"] = int(emb.shape[1])

    _, head = find(sd, "lm_head.proj.weight")
    if head is not None and cfg["vocab"]:
        rows = int(head.shape[0])
        if rows % cfg["vocab"]:
            raise SystemExit(
                f"[ABORT] lm_head rows ({rows}) not divisible by vocab "
                f"({cfg['vocab']}): cannot derive BLOCK.")
        cfg["block"] = rows // cfg["vocab"]
    else:
        _, plain = find(sd, "lm_head.weight")
        cfg["block"] = 1 if plain is not None else None
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stride", type=int, help="not derivable from weights; supply it")
    ap.add_argument("--block", type=int, help="if given, must match the checkpoint")
    ap.add_argument("--nstages", type=int, help="if given, must match the checkpoint")
    ap.add_argument("--kernel", type=int, help="if given, must match the checkpoint")
    ap.add_argument("--export", action="store_true",
                    help="print shell export lines instead of a report")
    a = ap.parse_args()

    cfg = derive(load_state_dict(a.checkpoint))
    has_stem = cfg["n_stages"] > 0

    if a.export:
        if not has_stem:
            print("export JANUS_STEM=0")
            return 0
        if a.stride is None:
            sys.exit("[ABORT] --stride is required with --export (not derivable).")
        band = cfg["block"] // a.stride if cfg["block"] % a.stride == 0 else None
        if band is None:
            sys.exit(f"[ABORT] BLOCK={cfg['block']} is not a multiple of STRIDE={a.stride}.")
        print("export JANUS_STEM=1")
        print(f"export JANUS_STEM_STRIDE={a.stride}")
        print(f"export JANUS_STEM_BLOCK={cfg['block']}")
        print(f"export JANUS_STEM_KERNEL={cfg['kernel']}")
        print(f"export JANUS_STEM_NSTAGES={cfg['n_stages']}")
        return 0

    print("=" * 62)
    print("STEM CONFIGURATION DERIVED FROM THE CHECKPOINT")
    print("=" * 62)
    if not has_stem:
        print("  no stem in this checkpoint (baseline model)")
        return 0

    rf = 1 + cfg["n_stages"] * (cfg["kernel"] - 1) if a.stride == 1 else None
    print(f"  n_stages : {cfg['n_stages']}")
    print(f"  kernel   : {cfg['kernel']}")
    print(f"  block    : {cfg['block']}   (from lm_head rows / vocab)")
    print(f"  vocab    : {cfg['vocab']}")
    print(f"  d_model  : {cfg['d_model']}")
    print(f"  stride   : {a.stride if a.stride else 'NOT DERIVABLE — supply --stride'}")
    if a.stride:
        if cfg["block"] % a.stride:
            sys.exit(f"\n[ABORT] BLOCK={cfg['block']} is not a multiple of STRIDE={a.stride}.")
        print(f"  BAND     : {cfg['block'] // a.stride}   (= block / stride, in grid cells)")
        if rf:
            print(f"  receptive field: {rf} bp")

    fail = []
    for name, given in (("block", a.block), ("n_stages", a.nstages), ("kernel", a.kernel)):
        if given is not None and given != cfg[name]:
            fail.append(f"{name}: requested {given}, checkpoint has {cfg[name]}")

    # for stride > 1 the builder composes stride-2 stages, so n_stages is fixed
    if a.stride and a.stride > 1:
        expected = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}.get(a.stride)
        if expected is not None and cfg["n_stages"] != expected:
            fail.append(f"stride={a.stride} implies n_stages={expected}, "
                        f"checkpoint has {cfg['n_stages']}")

    print("=" * 62)
    if fail:
        print("MISMATCH — the model would be rebuilt differently from how it was trained:")
        for f in fail:
            print("  -", f)
        print("\nA mismatch of this kind does not raise: load_state_dict runs with")
        print("strict=False, and BAND is not a parameter at all. The embeddings")
        print("would be silently wrong. Fix the environment and rerun.")
        return 1
    print("OK — the requested configuration matches the checkpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
