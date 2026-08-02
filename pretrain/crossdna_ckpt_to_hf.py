#!/usr/bin/env python3
"""
crossdna_ckpt_to_hf.py — convert a CrossDNA Lightning checkpoint (from-scratch
pre-training) into a HuggingFace-format directory, structurally identical to the
weights released by the authors (`crossdna_inference/`).

WHY CONVERT RATHER THAN ADAPT THE SCRIPTS
The downstream scripts load weights with
`AutoModelForMaskedLM.from_pretrained(<dir>)` and cannot read a Lightning .ckpt.
Converting the checkpoint instead of teaching the scripts to read .ckpt keeps the
evaluation protocol IDENTICAL to the one used for the human model: not one line
of downstream code changes between the two, so a difference in results cannot be
an artefact of a difference in loading.

Key mapping (verified against the real files):
    ckpt "model.<X>"  ->  HF "backbone.<X>"
Dropped: *_torchmetrics.* (metric state, not weights) and
         mlm_mask_embed / special_token_embed (pre-training only).

Usage:
  python crossdna_ckpt_to_hf.py \\
      --ckpt    .../checkpoints/last.ckpt \\
      --ref-dir $CROSSDNA_REPO/crossdna_inference \\
      --out-dir $PENGUIN_ROOT/crossdna/hf_scratch_penguin_clean
"""
import argparse
import json
import shutil
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="last.ckpt from pre-training")
    ap.add_argument("--ref-dir", required=True,
                    help="reference HF directory (the authors' weights): supplies "
                         "config.json, tokenizer and the model .py files, and is "
                         "used to VERIFY the key set")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-missing", action="store_true",
                    help="write anyway if keys are missing (default: abort)")
    args = ap.parse_args()

    ref = Path(args.ref_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. reference keys (the target to replicate) ----
    from safetensors.torch import load_file
    ref_sd = load_file(str(ref / "model.safetensors"))
    ref_keys = set(ref_sd.keys())
    print(f"[ref] {len(ref_keys)} tensors in {ref/'model.safetensors'}")

    # ---- 2. Lightning checkpoint ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    print(f"[ckpt] {len(sd)} tensors | epoch={ckpt.get('epoch')} "
          f"step={ckpt.get('global_step')}")

    # ---- 3. remap ----
    new_sd, dropped_metrics, extra = {}, [], []
    for k, v in sd.items():
        if "_torchmetrics." in k:
            dropped_metrics.append(k)
            continue
        if not k.startswith("model."):
            extra.append(k)
            continue
        nk = "backbone." + k[len("model."):]
        if nk in ref_keys:
            new_sd[nk] = v
        else:
            extra.append(k)  # in the checkpoint but not in the inference model

    missing = sorted(ref_keys - set(new_sd.keys()))

    print(f"\n[mapping] converted                  : {len(new_sd)}")
    print(f"[mapping] metric state dropped       : {len(dropped_metrics)}")
    print(f"[mapping] extra (pre-training only)  : {len(extra)}")
    for k in extra:
        print(f"      + {k}")
    print(f"[mapping] MISSING against reference  : {len(missing)}")
    for k in missing[:20]:
        print(f"      - {k}")

    # Abort by default. A partially converted model loads without error and
    # produces embeddings that look entirely reasonable, so this has to be a
    # hard stop rather than a warning.
    if missing and not args.allow_missing:
        raise SystemExit(
            "\nERROR: keys are missing relative to the reference model. The "
            "conversion would produce an incomplete model. Use --allow-missing "
            "only if you know why."
        )

    # Shape check: a mismatch here is a serious and otherwise silent error.
    bad_shapes = [k for k in new_sd if tuple(new_sd[k].shape) != tuple(ref_sd[k].shape)]
    if bad_shapes:
        print("\nERROR: incompatible shapes:")
        for k in bad_shapes[:10]:
            print(f"      {k}: ckpt {tuple(new_sd[k].shape)} vs "
                  f"ref {tuple(ref_sd[k].shape)}")
        raise SystemExit(1)
    print("[check] every shape matches the reference.")

    # ---- 4. write the HF directory ----
    torch.save(new_sd, out / "pytorch_model.bin")

    # Config and support files are COPIED from the reference so the inference
    # setup — and the model geometry — is exactly the one used for the human
    # model.
    for f in ["config.json", "configuration_crossdna.py", "modeling_crossdna.py",
              "tokenization_crossdna.py", "tokenizer_config.json",
              "special_tokens_map.json"]:
        if (ref / f).exists():
            shutil.copy2(ref / f, out / f)

    cfg = json.loads((out / "config.json").read_text())
    cfg["_name_or_path"] = "CrossDNA-scratch-penguin-clean"
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    print(f"\n[output] HF directory written to: {out}")
    print("         (config/tokenizer/code copied from the reference: identical "
          "setup to the human model)")

    # ---- 5. load test: the check that actually matters ----
    try:
        from transformers import AutoModelForMaskedLM
        m = AutoModelForMaskedLM.from_pretrained(
            str(out), trust_remote_code=True, local_files_only=True)
        n = sum(p.numel() for p in m.parameters())
        print(f"[test]   AutoModelForMaskedLM load: OK ({n:,} parameters)")
    except Exception as e:
        print(f"[test]   WARNING: load failed -> {e}")


if __name__ == "__main__":
    main()
