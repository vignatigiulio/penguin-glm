#!/usr/bin/env python
"""
verify_lora_targets.py — empirical check of which JanusDNA Mamba mixer
projections are ACTUALLY affected by a LoRA adapter.

WHY THIS EXISTS
JanusDNAMambaMixer has a CUDA fast path (mamba_ssm) that can use
x_proj.weight / dt_proj.weight directly inside the kernel, bypassing the
module's forward() entirely. Where that happens, a LoRALinear wrapper is simply
never called: the adapter trains, its parameters change, the loss goes down for
unrelated reasons, and it has no effect on the projection it was supposed to
adapt. Ghost adapters are invisible in every log.

This script settles the question BEFORE training, one module type at a time.

METHOD
For each candidate suffix (in_proj, out_proj, x_proj, dt_proj), inject LoRA on
THAT type only, force lora_B to a large non-zero value so the delta cannot be
mistaken for numerical noise, and compare the forward output with and without
it. If the output changes, the real forward respects the module. If it stays
bit-identical, the module is bypassed and must be EXCLUDED from the LoRA targets.

This is how dt_proj came to be excluded in src/finetune_lora_janusdna.py.

Usage:
  python verify_lora_targets.py \\
    --repo-path $JANUSDNA_REPO \\
    --model-config <path>/model_config.json \\
    --checkpoint  <path>/checkpoints/last.ckpt
"""
import argparse
import json
import sys

import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

CANDIDATE_SUFFIXES = [
    "mamba_fwd.in_proj",
    "mamba_fwd.out_proj",
    "mamba_fwd.x_proj",
    "mamba_fwd.dt_proj",
    # attention, as a control: these should always be respected
    "self_attn.q_proj",
    "self_attn.o_proj",
]


def load_model_and_tokenizer(repo_path, model_config_path, checkpoint_path, device):
    sys.path.insert(0, str(repo_path))
    from janusdna.configuration_janusdna import JanusDNAConfig
    from janusdna.modeling_janusdna import JanusDNAModel
    from caduceus.tokenization_caduceus import CaduceusTokenizer

    with open(model_config_path) as f:
        cfg_dict = json.load(f)
    config = JanusDNAConfig(**cfg_dict["config"])
    model = JanusDNAModel(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    consume_prefix_in_state_dict_if_present(sd, "model.")
    consume_prefix_in_state_dict_if_present(sd, "model.")
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    tok = CaduceusTokenizer(model_max_length=config.max_position_embeddings,
                            characters=("A", "C", "G", "T", "N"))
    return model, tok


class LoRAProbe(nn.Module):
    """LoRALinear whose lora_B can be forced non-zero, for the test."""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features) * 0.1)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))  # zero = inactive
        self.active = False

    # The fast path reaches into .weight / .bias directly, so these have to be
    # forwarded or the probe breaks the model instead of measuring it.
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def activate(self):
        # fill lora_B with large values -> an unmistakably non-zero delta
        with torch.no_grad():
            self.lora_B.fill_(1.0)
        self.active = True

    def forward(self, x):
        out = self.base(x)
        if self.active:
            lora = (x @ self.lora_A.t()) @ self.lora_B.t()
            out = out + self.scaling * lora.to(out.dtype)
        return out


def inject_single(model, suffix):
    """Inject LoRAProbe only into modules ending with `suffix`. Returns the probes."""
    probes = []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and full.endswith(suffix):
                probe = LoRAProbe(child).to(child.weight.device)
                setattr(mod, cname, probe)
                probes.append(probe)
    return probes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-path", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    seq = "ACGT" * 250  # 1000 bp, matching the real windows

    print("Checking which Mamba projections the real forward respects.")
    print("(LoRA delta active -> if the output does NOT change, the module is bypassed)\n")

    for suffix in CANDIDATE_SUFFIXES:
        # reload a clean model each time: injection is destructive
        model, tok = load_model_and_tokenizer(
            args.repo_path, args.model_config, args.checkpoint, dev)
        enc = tok([seq], add_special_tokens=False, padding=True,
                  return_tensors="pt", return_attention_mask=True)
        ids = enc["input_ids"].to(dev)

        probes = inject_single(model, suffix)
        if not probes:
            print(f"  {suffix:24s} -> NO module found with this suffix")
            continue

        with torch.no_grad():
            out_off, _ = model(ids, return_dict=False)
            for p in probes:
                p.activate()
            out_on, _ = model(ids, return_dict=False)

        delta = (out_on - out_off).abs().max().item()
        respected = delta > 1e-6
        status = ("RESPECTED (adapter effective)" if respected
                  else "BYPASSED (adapter IGNORED!)")
        print(f"  {suffix:24s} -> n_modules={len(probes):2d}  "
              f"max|delta|={delta:.3e}  {status}")

    print("\nConclusion: use only the 'RESPECTED' suffixes as LoRA targets.")
    print("The 'BYPASSED' ones would train ghost adapters with no effect.")


if __name__ == "__main__":
    main()
