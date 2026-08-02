#!/usr/bin/env python
"""
certify_fp32_patch.py — certify the fp32 attention patch against a reference model.

THE ARGUMENT
CrossDNA-8.1M runs natively and cleanly under flash-attn, so it serves as ground
truth. If the fp32 SDPA replacement faithfully reproduces flash-attn, then the
8.1M embeddings computed with and without the patch must COINCIDE — cosine ~1,
relative error ~0. Once that holds, the patch can be applied to 71.6M and 519M,
where there is no clean reference to compare against because those checkpoints
overflow in bf16 in the first place.

Without this step, the fp32 numbers for the large models would rest on an
assumption. With it, they rest on a measurement made where verification is
possible.

Usage:
    python certify_fp32_patch.py --weights $CROSSDNA_REPO/crossdna_inference
"""
import argparse
import os
import random
import sys

os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")

import numpy as np

# runtime patches live in env/patches/
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env", "patches"))


def load_model(weights, device):
    from transformers import AutoModelForMaskedLM

    m = AutoModelForMaskedLM.from_pretrained(
        weights, trust_remote_code=True, local_files_only=True
    ).float().to(device).eval()
    m.backbone.pretrain = False
    m.backbone.for_representation = True
    return m


def embed(model, seqs, device):
    import torch

    BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    x = torch.tensor([[BASE.get(c, 4) for c in s] for s in seqs],
                     dtype=torch.long, device=device)
    with torch.inference_mode():
        emb, _ = model.backbone(x)  # [B, L, H]
    return emb.float().mean(dim=1).cpu().numpy()  # [B, H]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="reference model (8.1M)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--length", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    random.seed(args.seed)
    # Realistic-ish sequences: mostly random ACGT with some tandem-repeat
    # stretches, because repeats are where attention behaves least like noise
    # and where a faithless approximation would show up first.
    seqs = []
    for _ in range(args.n):
        if random.random() < 0.3:
            unit = "".join(random.choice("ACGT")
                           for _ in range(random.choice([2, 3, 4])))
            s = (unit * (args.length // len(unit) + 1))[: args.length]
        else:
            s = "".join(random.choice("ACGT") for _ in range(args.length))
        seqs.append(s)

    # 1) ORIGINAL (native flash-attn) — no patch, no fp32 override
    print("[1] embedding with ORIGINAL flash-attn (internal bf16, healthy 8.1M)...")
    m1 = load_model(args.weights, args.device)
    X_orig = embed(m1, seqs, args.device)
    del m1
    torch.cuda.empty_cache()

    # 2) fp32 PATCH
    print("[2] embedding with the fp32 patch (SDPA)...")
    import sdpa_fp32
    sdpa_fp32.install()
    m2 = load_model(args.weights, args.device)
    mod = sys.modules.get(type(m2).__module__)
    if hasattr(mod, "preferred_amp_dtype"):
        mod.preferred_amp_dtype = lambda: torch.float32
    X_patch = embed(m2, seqs, args.device)

    # 3) comparison
    a = X_orig / (np.linalg.norm(X_orig, axis=1, keepdims=True) + 1e-9)
    b = X_patch / (np.linalg.norm(X_patch, axis=1, keepdims=True) + 1e-9)
    cos = float(np.mean(np.sum(a * b, axis=1)))
    rel = float(np.mean(np.linalg.norm(X_orig - X_patch, axis=1)
                        / (np.linalg.norm(X_orig, axis=1) + 1e-9)))
    maxabs = float(np.max(np.abs(X_orig - X_patch)))
    print("\n=== CERTIFICATION ===")
    print(f"mean cosine (orig vs patch) : {cos:.6f}")
    print(f"mean relative L2 error      : {rel:.3e}")
    print(f"max absolute difference     : {maxabs:.3e}")
    finite = bool(np.isfinite(X_patch).all())
    print(f"patched output all finite   : {finite}")
    ok = cos > 0.999 and rel < 1e-2 and finite
    print(f"\nRESULT: {'PATCH FAITHFUL (proceed with 71.6M/519M)' if ok else 'PATCH NOT FAITHFUL (investigate)'}")


if __name__ == "__main__":
    main()
