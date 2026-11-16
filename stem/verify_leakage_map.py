#!/usr/bin/env python3
"""
verify_leakage_map.py — measure, on the REAL MODEL, which bases each position
is able to see. This is the strongest available check of the band argument.

It does not assert: it builds the dependency map by Jacobian and prints it.
For every offset j = 0..BLOCK-1 it reports the maximum gradient on the base
that the position MUST predict. The theory says it has to be EXACTLY zero for
all BLOCK offsets; with the wrong band it stays zero only for j = 0.

  correct band (BAND = BLOCK/STRIDE) :  j=0 zero, j=1 zero, j=2 zero, j=3 zero
  old band     (BAND = 1)            :  j=0 zero, j=1..3 NON-zero  <- leakage

NO CHECKPOINT IS NEEDED: self-exclusion is a structural property of the mask
and the fusion, independent of the weights. The model is built with random
initialisation from model_config.json.

WARNING — REQUIRES A GPU. The Mamba and FlexAttention kernels do not run on
CPU. If a training job is using the same GPU this script may cause an OOM: run
it on a free GPU, or after training. Keep T small (default 256).

USAGE:
  python verify_leakage_map.py \\
      --repo-path $PENGUIN_ROOT/JanusDNA_stem_s1_b4 \\
      --model-config <run_dir>/model_config.json \\
      --stride 1 --block 4

  # then the counter-proof, which MUST show leakage:
  JANUS_STEM_BAND_OVERRIDE=1 python verify_leakage_map.py ... --stride 1 --block 4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def build_model(repo_path, model_config, stride, block, kernel, nstages, device):
    os.environ["JANUS_STEM"] = "1"
    os.environ["JANUS_STEM_STRIDE"] = str(stride)
    os.environ["JANUS_STEM_BLOCK"] = str(block)
    os.environ["JANUS_STEM_KERNEL"] = str(kernel)
    if nstages:
        os.environ["JANUS_STEM_NSTAGES"] = str(nstages)

    sys.path.insert(0, str(repo_path))
    from janusdna.configuration_janusdna import JanusDNAConfig
    from janusdna.modeling_janusdna import JanusDNAForCausalLM
    import janusdna.modeling_janusdna as M
    try:
        import flexattn_blocksize_patch
        flexattn_blocksize_patch.apply()
    except Exception:
        pass

    cfg_dict = json.load(open(model_config))
    cfg = JanusDNAConfig(**cfg_dict.get("config", cfg_dict))
    print(f"[config] hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"fusion={getattr(cfg, 'layer_fusion_strategy', '?')}")
    print(f"[env]    stride={M._STEM_S} block={M._STEM_BLOCK} BAND={M._STEM_BAND}"
          + ("   <-- OVERRIDE ACTIVE (counter-proof)"
             if os.environ.get("JANUS_STEM_BAND_OVERRIDE") else ""))

    model = JanusDNAForCausalLM(cfg).to(device).eval()
    return model, M._STEM_BAND


def dependency_map(model, stride, block, T, device, probes=3):
    """For each probed position, return max|grad| over the BLOCK target bases."""
    ids = torch.randint(0, 4, (1, T), device=device)
    emb = model.get_input_embeddings()(ids).clone().detach().requires_grad_(True)
    out = model(inputs_embeds=emb, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out[1]

    M_grid = T // stride
    band = block // stride
    pts = [p for p in (band, M_grid // 2, M_grid - band - 1) if 0 < p < M_grid - band]
    pts = pts[:probes]

    rows = []
    for k, p in enumerate(pts):
        if emb.grad is not None:
            emb.grad = None
        if logits.dim() == 3:
            sel = logits[:, p * block:(p + 1) * block, :]
        else:
            sel = logits[:, p, :, :]
        sel.float().sum().backward(retain_graph=(k != len(pts) - 1))
        g = emb.grad[0].abs().sum(-1)                       # (T,)
        lo = p * stride
        target = [float(g[lo + j]) for j in range(block) if lo + j < T]
        ctx = float(torch.cat([g[:lo], g[lo + block:]]).max())
        rows.append((p, lo, target, ctx))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-path", type=Path, required=True)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--kernel", type=int, default=7)
    ap.add_argument("--nstages", type=int, default=3)
    ap.add_argument("-T", type=int, default=256)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("[ABORT] a GPU is required: Mamba/FlexAttention kernels do not run on CPU.")
    dev = "cuda"

    print("=" * 74)
    print("DEPENDENCY MAP ON THE REAL MODEL")
    print("=" * 74)
    model, band = build_model(a.repo_path, a.model_config, a.stride, a.block,
                              a.kernel, a.nstages, dev)
    stem = getattr(model.model, "stem", None)
    if stem is not None:
        print(f"[stem]   receptive field from the module = {stem.receptive_field} bp")

    rows = dependency_map(model, a.stride, a.block, a.T, dev)

    print(f"\nEach position p predicts bases [p*{a.stride}, p*{a.stride}+{a.block}).")
    print("Theory: the gradient must be EXACTLY zero on all of them, context non-zero.\n")
    hdr = "  pos   target bases     " + "".join(f"  j={j:<10}" for j in range(a.block)) + "  context"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    leaks = 0
    for p, lo, target, ctx in rows:
        cells = "".join(f"  {t:<11.3e}" for t in target)
        flag = ""
        bad = [j for j, t in enumerate(target) if t > 0.0]
        if bad:
            leaks += 1
            flag = f"   <-- LEAKAGE su j={bad}"
        print(f"  {p:<5} [{lo},{lo+a.block})".ljust(24) + cells + f"  {ctx:.2e}{flag}")

    print("\n" + "=" * 74)
    if leaks == 0:
        print(f"RESULT: no leakage. BAND={band} covers the whole block")
        print("        of {} bases. Self-exclusion verified on the real model.".format(a.block))
        print("\n  Now run the COUNTER-PROOF, which must instead show leakage:")
        print("    JANUS_STEM_BAND_OVERRIDE=1 python verify_leakage_map.py ...")
        print("  If that one also comes out clean, this measurement measures nothing.")
    else:
        print(f"RESULT: LEAKAGE on {leaks} probed positions, with BAND={band}.")
        if os.environ.get("JANUS_STEM_BAND_OVERRIDE"):
            print("        Expected: this is the counter-proof with the band forced to its old value.")
            print("        It confirms the parametrised band is NOT cosmetic.")
        else:
            print("        NOT expected with the correct band: do not start training.")
    print("=" * 74)
    return 0 if (leaks == 0) != bool(os.environ.get("JANUS_STEM_BAND_OVERRIDE")) else 1


if __name__ == "__main__":
    sys.exit(main())
