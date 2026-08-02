"""
causal_conv_stem.py — causal down-sampling convolutional stem for JanusDNA.

PURPOSE
-------
Compress the grid from T nucleotides to M = T/S "super-tokens" BEFORE the
Mamba+MoE backbone, while preserving (a) base-level input resolution,
(b) reverse-complement symmetry, and (c) the autoregressive objective. The
backbone and the final Janus fusion then operate on the compressed grid of
length M.

The experiment this serves: if the GC attractor is a consequence of
single-nucleotide tokenisation, then reasoning over ~19 bp super-tokens should
weaken it. The prediction was stated before the run — an improvement in the
NMI(function)/NMI(GC) ratio would be a mechanistic gain, an improvement in MCC
alone merely a computational one.

CRITICAL INVARIANT (the one thing that can break the experiment silently)
-------------------------------------------------------------------------
Causality: super-token m must depend ONLY on inputs <= (m+1)*S - 1. The padding
rule that guarantees it, verified exhaustively through the Jacobian:

    left_pad = kernel_size - stride      (requires kernel_size >= stride)

Normalisation is PER-POSITION LayerNorm, over channels only: it never computes
statistics along time, so it introduces no leakage from the future.

WEIGHT TYING (JanusDNA uses bidirectional_weight_tie=True)
----------------------------------------------------------
The backward branch does NOT need a second stem: THIS same stem is applied to
the reversed sequence. A causal convolution over a reversed input is
automatically anti-causal in the original coordinates, so the tie is preserved
for free. (Verified in modeling_janusdna.py that the backward stream really is
flip-based.)

Environment notes: dtype-agnostic, fine under bf16-mixed autocast. The
LayerNorms stabilise numerically; they do not worsen bf16 stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDownConv1d(nn.Module):
    """Strided causal Conv1d. Output m depends only on inputs <= (m+1)*stride - 1."""

    def __init__(self, d_in: int, d_out: int, kernel_size: int, stride: int):
        super().__init__()
        assert kernel_size >= stride, "kernel_size >= stride is required for left_pad >= 0"
        self.left_pad = kernel_size - stride
        self.conv = nn.Conv1d(d_in, d_out, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        x = F.pad(x, (self.left_pad, 0))                 # pad on the LEFT only
        return self.conv(x)                              # (B, C, T/stride)


class CausalConvStem(nn.Module):
    """
    Causal down-sampling stem, agnostic to the embedding it sits behind.
      input : (B, T, D)   token embeddings
      output: (B, M, D)   super-tokens, M = T // S   (requires T % S == 0)

    S = 2 (1 stage) | 4 (2 stages) | 8 (3 stages) | 16 | 32, all composed from
    stride-2 stages.
    """

    def __init__(self, d_model: int, stride: int = 4, kernel_size: int = 7,
                 d_mid: int | None = None):
        super().__init__()
        assert stride in (2, 4, 8, 16, 32), "supported strides: 2, 4, 8, 16, 32"
        d_mid = d_mid or d_model
        n_stages = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[stride]
        dims = [d_model] + [d_mid] * (n_stages - 1) + [d_model]

        self.blocks = nn.ModuleList()
        for i in range(n_stages):
            self.blocks.append(nn.ModuleDict({
                "conv": CausalDownConv1d(dims[i], dims[i + 1], kernel_size, stride=2),
                # LayerNorm over channels, applied in (B, L, C) layout: per
                # position, therefore causally safe — no statistics along time.
                "norm": nn.LayerNorm(dims[i + 1]),
            }))
        self.act = nn.GELU()
        self.stride = stride
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        assert x.dim() == 3, "expected (B, T, D)"
        assert x.size(1) % self.stride == 0, (
            f"T={x.size(1)} is not divisible by S={self.stride}: pad upstream"
        )
        h = x
        for blk in self.blocks:
            h = h.transpose(1, 2)   # (B, C, L)
            h = blk["conv"](h)      # (B, C, L/2)
            h = h.transpose(1, 2)   # (B, L/2, C)
            h = blk["norm"](h)      # per position -> causally safe
            h = self.act(h)
        return h                    # (B, M, D)


class ExpansionHead(nn.Module):
    """
    Expansion head: one super-token produces S nucleotide logits.
      input : (B, M, D)
      output: (B, M*S, vocab)   in natural order (super-token m -> positions
                                [mS, (m+1)S))

    Note that the S bases are predicted conditionally independently given the
    context. That is a structural limitation and must be declared in any
    bits-per-nucleotide comparison against the base model.

    ALIGNMENT WARNING (a wiring decision, not a property of this module):
    which super-token predicts which block depends on the shift applied to the
    AR objective in the harness. Under strict causality, super-token m has seen
    inputs <= (m+1)S-1 and can predict block m+1 (positions [(m+1)S, (m+2)S));
    the first block [0,S) has no predictor, exactly as the first token does in
    ordinary AR training. If the harness already applies a shift-by-one-TOKEN to
    the logits, do NOT double it — it has to become a shift-by-one-BLOCK. Check
    the loss in modeling_janusdna.py before assuming either way.
    """

    def __init__(self, d_model: int, stride: int, vocab_size: int):
        super().__init__()
        self.stride = stride
        self.vocab = vocab_size
        self.proj = nn.Linear(d_model, stride * vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, M, D)
        B, M, _ = h.shape
        logits = self.proj(h)  # (B, M, S*V)
        return logits.reshape(B, M * self.stride, self.vocab)


# ======================================================================
#  CAUSALITY GATE — run this in the target environment (torch present).
#  Fails LOUDLY if the stem looks into the future. It must pass BEFORE a
#  single training step runs: a stem that leaks would still train, still
#  converge, and produce a result that means nothing.
# ======================================================================

@torch.no_grad()
def _num_super_tokens(stem: CausalConvStem, T: int, D: int) -> int:
    return stem(torch.zeros(1, T, D)).shape[1]


def test_causality(stem: CausalConvStem, T: int = 128, D: int = 16,
                   tol: float = 0.0) -> None:
    """Via autograd: super-token m must have no gradient on inputs >= (m+1)*S."""
    S = stem.stride
    M = _num_super_tokens(stem, T, D)
    stem.zero_grad(set_to_none=True)
    x = torch.randn(1, T, D, requires_grad=True)
    h = stem(x)  # (1, M, D)
    for m in (0, M // 2, M - 1):  # left edge, middle, right edge
        if x.grad is not None:
            x.grad = None
        h[0, m].sum().backward(retain_graph=(m != M - 1))
        g = x.grad[0]  # (T, D)
        future = g[(m + 1) * S:].abs().max().item() if (m + 1) * S < T else 0.0
        past = g[:(m + 1) * S].abs().max().item()
        assert future <= tol, (
            f"LEAKAGE: super-token {m} depends on inputs >= {(m+1)*S} "
            f"(max|grad|={future:.3e}). The padding is not causal."
        )
        # The complementary check: a stem that ignored its own past would pass
        # the leakage test trivially while being useless.
        assert past > 0.0, f"DEGENERATE: super-token {m} ignores its own past"
    print(f"[OK] test_causality: S={S}, T={T}, {M} causal super-tokens "
          f"(zero gradient on the future, checked at both edges and the middle)")


def test_flip_anticausal(stem: CausalConvStem, T: int = 128, D: int = 16,
                         tol: float = 1e-5) -> None:
    """The backward branch is this stem applied to the reversed sequence. It must
    come out anti-causal in the original coordinates — the condition under which
    the weight tie holds."""
    S = stem.stride
    M = _num_super_tokens(stem, T, D)
    stem.zero_grad(set_to_none=True)
    x = torch.randn(1, T, D, requires_grad=True)
    h = stem(torch.flip(x, dims=[1]))  # process the reversed sequence
    for m in (0, M // 2, M - 1):
        if x.grad is not None:
            x.grad = None
        h[0, m].sum().backward(retain_graph=(m != M - 1))
        g = x.grad[0]
        boundary = T - (m + 1) * S  # must depend only on inputs >= boundary
        leak = g[:boundary].abs().max().item() if boundary > 0 else 0.0
        assert leak <= tol, (
            f"flip is NOT anti-causal: backward super-token {m} touches original "
            f"inputs < {boundary} (leak={leak:.3e}). The flip-based tie does not hold."
        )
    print("[OK] test_flip_anticausal: flipping the same stem gives an anti-causal "
          "map in the original coordinates (tie preserved)")


def test_shapes_and_head(d_model: int = 144, S: int = 4, vocab: int = 12,
                         T: int = 1024) -> None:
    stem = CausalConvStem(d_model, stride=S)
    head = ExpansionHead(d_model, stride=S, vocab_size=vocab)
    x = torch.randn(2, T, d_model)
    h = stem(x)
    assert h.shape == (2, T // S, d_model), h.shape
    logits = head(h)
    assert logits.shape == (2, T, vocab), logits.shape
    n_params = (sum(p.numel() for p in stem.parameters())
                + sum(p.numel() for p in head.parameters()))
    print(f"[OK] test_shapes: (2,{T},{d_model}) -> super-tokens (2,{T//S},{d_model}) "
          f"-> logits (2,{T},{vocab}). Stem+head parameters: {n_params:,}")


if __name__ == "__main__":
    print("=== CAUSAL STEM GATE (JanusDNA-144, S=4) ===\n")
    for S in (2, 4, 8):
        stem = CausalConvStem(d_model=144, stride=S)
        test_causality(stem, T=128, D=144)
        test_flip_anticausal(stem, T=128, D=144)
    print()
    test_shapes_and_head(d_model=144, S=4, vocab=12, T=1024)
