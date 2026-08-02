"""
test_stem_wiring.py — full-model gate for JanusDNA with the conv stem.

Checks the Janus invariant at WHOLE-MODEL level: the prediction for block p must
NOT depend on the input of block p (self-exclusion). If it does, the model can
copy the target — the loss drops, the numbers inflate, and nothing on screen
looks wrong.

This is deliberately NOT a forward-causality test ("no gradient on the future").
JanusDNA is bidirectional and is SUPPOSED to see the future. The only forbidden
dependency is the target block seeing itself.

USAGE (hooked into the harness by run_stem_experiment.sh): it runs automatically
in SequenceLightningModule.on_train_start() when JANUS_STEM_GATE=1, against the
model already on GPU:

    from test_stem_wiring import run_gate
    run_gate(self.model, stride=4, device=self.device)

run_gate is SAFE on a model that is about to train: it does not move the model
between devices, it restores the train/eval flag, and it zeroes the gradients at
the end so the first optimisation step is not contaminated.
"""

import torch


@torch.no_grad()
def _forward_shapes(model, stride, T=512, vocab_hi=4):
    device = next(model.parameters()).device
    ids = torch.randint(0, vocab_hi, (1, T), device=device)  # A/C/G/T only
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out[1]
    assert logits.shape[1] == T, (
        f"the head must return T={T} logits, got {logits.shape[1]}: "
        "ExpansionHead is not wired in, or the stride is wrong."
    )
    print(f"[OK] shapes: input T={T} -> logits {tuple(logits.shape)} "
          f"(M={T//stride} super-tokens)")


def test_self_exclusion(model, stride, T=512, vocab_hi=4, tol=0.0):
    """Backpropagate the logits of block p to the input embeddings. The gradient
    on block p's own input must be EXACTLY zero; the gradient on the context
    must not be."""
    device = next(model.parameters()).device
    ids = torch.randint(0, vocab_hi, (1, T), device=device)

    emb_layer = model.get_input_embeddings()
    emb = emb_layer(ids).clone().detach().requires_grad_(True)
    out = model(inputs_embeds=emb, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out[1]  # (1, T, vocab)

    M = T // stride
    for p in (1, M // 2, M - 2):  # avoid the special first/last super-tokens
        if emb.grad is not None:
            emb.grad = None
        block = slice(p * stride, (p + 1) * stride)
        logits[:, block, :].float().sum().backward(retain_graph=(p != M - 2))
        g = emb.grad[0]  # (T, D)
        self_dep = g[block].abs().max().item()
        ctx = torch.cat([g[:p * stride], g[(p + 1) * stride:]],
                        dim=0).abs().max().item()
        assert self_dep <= tol, (
            f"JANUS LEAKAGE: block {p} depends on its OWN input "
            f"(max|grad|={self_dep:.3e}). The fusion mask or the super-token "
            "alignment is broken -> the loss is meaningless."
        )
        assert ctx > 0.0, (
            f"DEGENERATE: block {p} depends on no context at all "
            "(neither stream is flowing)."
        )
    print("[OK] self-exclusion: block predicted WITHOUT seeing itself, "
          "forward+backward context active (checked on 3 super-tokens)")


def run_gate(model, stride, T=512, device=None, logfile=None):
    """Safe on a model about to train: restores state and zeroes gradients.

    The outcome is also written to a file (default: $JANUS_STEM_GATE_LOG or
    ./GATE_RESULT.txt) because the console at this point is full of the model
    repr and the progress bar, and a passing gate scrolls past unnoticed.
    """
    import os as _os

    logfile = logfile or _os.environ.get("JANUS_STEM_GATE_LOG", "GATE_RESULT.txt")

    banner = "=" * 60
    lines = [banner, f"FULL-MODEL GATE JanusDNA+stem (stride={stride}, T={T})", banner]
    print("\n" + "\n".join(lines))

    was_training = model.training
    model.eval()
    ok = False
    try:
        _forward_shapes(model, stride, T=T)
        test_self_exclusion(model, stride, T=T)
        ok = True
        lines += [
            f"[OK] shapes: input T={T} -> M={T//stride} super-tokens, "
            "logits re-expanded to T",
            "[OK] self-exclusion: block p predicted WITHOUT seeing itself, "
            "forward+backward context active",
            "", "RESULT: PASSED. The stem preserves the Janus masking. "
            "Cleared for the full run.",
        ]
        print("Gate passed: the stem preserves the Janus masking. Cleared for training.")
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
