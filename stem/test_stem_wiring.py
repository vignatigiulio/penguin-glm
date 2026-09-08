"""
test_stem_wiring.py — full-model gate for JanusDNA with the conv stem. VERSION 2.

Checks the Janus invariant at WHOLE-MODEL level:
  the prediction of base b must NOT depend on the input of base b.
  If it does, the model can copy the target -> falsely low loss, inflated
  numbers, and no error on screen.

NOTE: this is NOT a forward-causality test ("zero grad on the future").
JanusDNA is bidirectional and MUST see the future. The only forbidden thing is
that a predicted base sees itself.

WHAT CHANGED FROM v1
--------------------
In v1 stride and block coincided, so it was enough to check that the logits of
block p did not depend on the input of block p. With stride=1 and block=4 the
windows overlap: grid position m predicts bases
    m*STRIDE + j   for j in [0, BLOCK)
and ALL of them must be invisible to both the forward and the backward branch.
Checking only one would let leakage on the other three go through.

Expected logit shape: (B, M*BLOCK, vocab) from BlockHead, with the prediction of
base m*STRIDE + j at flat position m*BLOCK + j.

MANDATORY COUNTER-PROOF
-----------------------
Run once with the exclusion band forced back to its old value while the model
is configured for the wide block:
    JANUS_STEM_BAND_OVERRIDE=1
The gate MUST fail. If it passes, it is measuring nothing — the same logic as
the "same"-convolution counter-proof used for v1.

USAGE (wired into the harness by run_stem_experiment.sh):
  runs automatically in SequenceLightningModule.on_fit_start() when
  JANUS_STEM_GATE=1, on the model already placed on the GPU. on_fit_start is
  used rather than on_train_start because Lightning runs the sanity check
  between the two, and the gate must come first.
"""

import torch


@torch.no_grad()
def _forward_shapes(model, stride, block, T=512, vocab_hi=4):
    device = next(model.parameters()).device
    ids = torch.randint(0, vocab_hi, (1, T), device=device)   # A/C/G/T only
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out[1]
    M = T // stride
    assert logits.dim() == 3, (
        f"expected shape (B, M*BLOCK, V), got {tuple(logits.shape)} "
        f"({logits.dim()}D): the head is not wired in."
    )
    assert logits.shape[1] == M * block, (
        f"expected M*BLOCK={M*block} predictions (T={T}, stride={stride}, "
        f"block={block}), got {logits.shape[1]}: stride or block are inconsistent."
    )
    print(f"[OK] shapes: input T={T} -> logits {tuple(logits.shape)} "
          f"(M={M} positions x block={block}, band={block//stride})")


def test_self_exclusion(model, stride, block, T=512, vocab_hi=4, tol=0.0):
    """Backpropagate the logits of position m to the input embeddings. The
    gradient on the BLOCK bases that m predicts must be EXACTLY zero; the
    gradient on the context must not be."""
    device = next(model.parameters()).device
    ids = torch.randint(0, vocab_hi, (1, T), device=device)

    emb_layer = model.get_input_embeddings()
    emb = emb_layer(ids).clone().detach().requires_grad_(True)
    out = model(inputs_embeds=emb, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out[1]   # (1, M*BLOCK, V)

    M = T // stride
    band = block // stride
    # interior positions: avoid head and tail, where the fusion uses one branch only
    probes = [p for p in (band, M // 2, M - band - 1) if 0 < p < M - band]
    assert probes, f"M={M} is too small for band={band}: raise T in the gate"

    for p in probes:
        if emb.grad is not None:
            emb.grad = None
        lo = p * stride
        hi = min(lo + block, T)                    # window of bases predicted by p
        logits[:, p * block:(p + 1) * block, :].float().sum().backward(
            retain_graph=(p != probes[-1]))
        g = emb.grad[0]                            # (T, D)
        self_dep = g[lo:hi].abs().max().item()
        ctx = torch.cat([g[:lo], g[hi:]], dim=0).abs().max().item()
        assert self_dep <= tol, (
            f"JANUS LEAKAGE: position {p} predicts bases [{lo},{hi}) but depends "
            f"on at least one of them (max|grad|={self_dep:.3e}). The exclusion "
            f"band was not widened to BAND={band}: create_janus_mask and/or "
            f"final_fusion_repos_formasked are unpatched -> falsified loss, "
            f"inflated results."
        )
        assert ctx > 0.0, (
            f"DEGENERATE: position {p} depends on no context at all "
            f"(forward/backward do not flow)."
        )
    print(f"[OK] self-exclusion: the {block}-base window is invisible to both "
          f"branches, forward+backward context active ({len(probes)} positions checked)")


def run_gate(model, stride, block=None, T=512, device=None, logfile=None):
    """Safe on a model in training: restores train/eval state and clears grads.
    Also writes the outcome to a file (default: $JANUS_STEM_GATE_LOG or
    ./GATE_RESULT.txt) so it is not lost in the console scrollback."""
    import os as _os
    logfile = logfile or _os.environ.get("JANUS_STEM_GATE_LOG", "GATE_RESULT.txt")
    if block is None:
        block = int(_os.environ.get("JANUS_STEM_BLOCK", str(stride)))
    assert block % stride == 0, (
        f"BLOCK={block} must be a multiple of STRIDE={stride} "
        f"(the band is measured in grid cells)."
    )
    band = block // stride

    banner = "=" * 66
    lines = [banner,
             f"FULL-MODEL GATE JanusDNA+stem  stride={stride} block={block} "
             f"band={band}  T={T}",
             banner]
    print("\n" + "\n".join(lines))

    was_training = model.training
    model.eval()
    ok = False
    try:
        _forward_shapes(model, stride, block, T=T)
        test_self_exclusion(model, stride, block, T=T)
        ok = True
        lines += [
            f"[OK] shapes: T={T} -> M={T//stride} positions, {block} logits each",
            f"[OK] self-exclusion: the {block}-base window is invisible to fwd and bwd",
            "",
            "RESULT: PASSED. The exclusion band matches the predicted block.",
        ]
        print("Gate passed: the band covers the whole predicted block.")
    except Exception as e:
        lines += ["", f"RESULT: FAILED -> {e}"]
        raise
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        try:
            with open(logfile, "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"\n>>> Gate result also written to: {_os.path.abspath(logfile)}")
        except Exception:
            pass
    return ok


if __name__ == "__main__":
    print("This gate runs against the harness model (self.model).")
    print("Enable it with JANUS_STEM_GATE=1 during the smoke run:")
    print("  JANUS_STEM_GATE=1 ./run_stem_experiment.sh")
    print()
    print("COUNTER-PROOF: run once with the old band forced")
    print("  (JANUS_STEM_BAND_OVERRIDE=1). The gate MUST fail.")
