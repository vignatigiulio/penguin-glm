"""
causal_conv_stem.py — causal convolutional stem for JanusDNA-144.

VERSION 2: STRIDE and BLOCK are DECOUPLED.
==========================================
In version 1 a single parameter S did two different jobs:
  - how far the grid is compressed (T -> M = T/S)
  - how many bases the head predicts from each position
They coincided because one grid cell was worth exactly S bases.

With overlapping-window tokenisation (window 4, step 1) the two diverge, so
two parameters are needed:

  STRIDE (S) : grid compression.  S=1 -> no compression, M = T
  BLOCK  (B) : bases predicted by each fused position.  B=4 in both setups

  BAND = B // S : width of the self-exclusion, IN GRID CELLS.
                  This is the constant hardcoded to 1 in modeling_janusdna.py.

  previous run :  S=4, B=4  -> BAND=1   (reproduces v1 exactly)
  new run      :  S=1, B=4  -> BAND=4   (overlapping windows)

CRITICAL INVARIANT (the only thing that can break the experiment silently)
-------------------------------------------------------------------------
Causality: grid position m must depend ONLY on inputs <= (m+1)*S - 1.
The padding rule that guarantees it:
    left_pad = kernel_size - stride      (requires kernel_size >= stride)
It holds unchanged for stride=1 (left_pad = k-1, output length T).
Normalisation is a PER-POSITION LayerNorm (over channels only): it never
computes statistics along time, so it introduces no leakage from the future.

RECEPTIVE FIELD — a choice to declare, not a default to inherit
--------------------------------------------------------------
    RF = 1 + sum_i (k_i - 1) * prod_{j<i} s_j
  S=4, 2 stages k=7 stride 2 :  RF = 19 bp   <- the stem of the completed run
  S=1, 3 stages k=7 stride 1 :  RF = 19 bp   <- matches it at step 1
  S=1, 2 stages k=7 stride 1 :  RF = 13 bp
  S=1, 1 stage   k=4 stride 1:  RF =  4 bp   <- literal reading of "window 4"

Only the 19 bp configuration keeps the receptive field equal to the previous
run, so that the comparison isolates grid compression alone.

WEIGHT TYING (JanusDNA uses bidirectional_weight_tie=True)
----------------------------------------------------------
The backward branch needs no second stem: THIS stem is applied to the reversed
sequence. A causal convolution over reversed input is automatically
anti-causal in the original coordinates, so the tie is preserved for free.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal Conv1d, any stride. Output m depends only on inputs <= (m+1)*stride - 1."""

    def __init__(self, d_in: int, d_out: int, kernel_size: int, stride: int):
        super().__init__()
        assert kernel_size >= stride, (
            f"kernel_size >= stride is required for left_pad >= 0 "
            f"(k={kernel_size}, s={stride})"
        )
        self.left_pad = kernel_size - stride
        self.stride = stride
        self.conv = nn.Conv1d(d_in, d_out, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        x = F.pad(x, (self.left_pad, 0))                 # pad on the LEFT only
        return self.conv(x)                              # (B, C, T/stride)


class CausalConvStem(nn.Module):
    """
    Causal stem, agnostic to the embedding it is fed.
      input : (B, T, D)   token embeddings
      output: (B, M, D)   M = T // stride   (requires T % stride == 0)

    stride > 1 : composable stride-2 stages, n_stages = log2(stride).
    stride == 1: n_stages stride-1 stages (default 2), no compression.
    """

    def __init__(self, d_model: int, stride: int = 4, kernel_size: int = 7,
                 n_stages: int | None = None, d_mid: int | None = None):
        super().__init__()
        assert stride in (1, 2, 4, 8, 16, 32), "supported strides: 1, 2, 4, 8, 16, 32"

        if stride == 1:
            n_stages = 2 if n_stages is None else int(n_stages)
            assert n_stages >= 1, "at least one stage is required"
            stage_strides = [1] * n_stages
        else:
            expected = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[stride]
            if n_stages is not None and int(n_stages) != expected:
                raise ValueError(
                    f"stride={stride} requires n_stages={expected} (stride-2 stages), "
                    f"got {n_stages}. For a free stage count use stride=1."
                )
            n_stages = expected
            stage_strides = [2] * n_stages

        d_mid = d_mid or d_model
        dims = [d_model] + [d_mid] * (n_stages - 1) + [d_model]

        self.blocks = nn.ModuleList()
        for i, s in enumerate(stage_strides):
            self.blocks.append(nn.ModuleDict({
                "conv": CausalConv1d(dims[i], dims[i + 1], kernel_size, stride=s),
                # LayerNorm over channels, applied in (B, L, C) layout: per-position,
                # therefore causality-safe (no statistics along time).
                "norm": nn.LayerNorm(dims[i + 1]),
            }))
        self.act = nn.GELU()
        self.stride = stride
        self.kernel_size = kernel_size
        self.n_stages = n_stages
        self.d_model = d_model

    @property
    def receptive_field(self) -> int:
        """RF = 1 + sum_i (k_i - 1) * prod_{j<i} s_j."""
        rf, mult = 1, 1
        for blk in self.blocks:
            conv = blk["conv"]
            rf += (conv.conv.kernel_size[0] - 1) * mult
            mult *= conv.stride
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        assert x.dim() == 3, "expected (B, T, D)"
        assert x.size(1) % self.stride == 0, (
            f"T={x.size(1)} is not divisible by stride={self.stride}: pad upstream"
        )
        h = x
        for blk in self.blocks:
            h = h.transpose(1, 2)          # (B, C, L)
            h = blk["conv"](h)             # (B, C, L/s)
            h = h.transpose(1, 2)          # (B, L/s, C)
            h = blk["norm"](h)             # per-position -> causality-safe
            h = self.act(h)
        return h                           # (B, M, D)


class BlockHead(nn.Module):
    """
    Block head. Produces BLOCK nucleotide logits from each grid position.

      input : (B, M, D)
      output: (B, M * BLOCK, vocab)   -- already flattened, order m*BLOCK + j

    ALIGNMENT (this is the definition; the patched loss follows it):
        logits[:, m*BLOCK + j]  predicts base  m*STRIDE + j     for j in [0, BLOCK)

    The output is flattened here rather than left 4D so that the ordering is
    fixed by this module, not by a reshape further downstream whose semantics
    would have to be guessed.

    With STRIDE == BLOCK every base is covered exactly once and this is
    identical to the v1 ExpansionHead, so the resulting loss matches numerically.

    With STRIDE=1, BLOCK=4 the windows overlap and every base is predicted four
    times, from four positions at horizons 1..4. The loss is then a mean over
    M*BLOCK terms rather than T: it is NOT bits-per-nucleotide comparable with
    the baseline without renormalisation. Declare this in any comparison.

    Note: the BLOCK bases are predicted conditionally independently given the
    context — a structural limitation, as in v1.
    """

    def __init__(self, d_model: int, block: int, vocab_size: int):
        super().__init__()
        self.block = block
        self.vocab = vocab_size
        self.proj = nn.Linear(d_model, block * vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, M, D)
        B, M, _ = h.shape
        logits = self.proj(h)                            # (B, M, BLOCK*V)
        return logits.reshape(B, M * self.block, self.vocab)


# Backwards-compatible alias for v1 call sites: ExpansionHead(d_model, stride, vocab).
# Returns the new flattened shape, so it must be used with the patched loss.
ExpansionHead = BlockHead


# ======================================================================
#  CAUSALITY GATE — run it in your environment (torch required).
#  Fails LOUDLY if the stem looks into the future. It must pass before a
#  single training step is taken.
# ======================================================================

@torch.no_grad()
def _num_positions(stem: CausalConvStem, T: int, D: int) -> int:
    return stem(torch.zeros(1, T, D)).shape[1]


def test_causality(stem: CausalConvStem, T: int = 128, D: int = 16,
                   tol: float = 0.0) -> None:
    """Via autograd: position m must have no gradient on inputs >= (m+1)*stride."""
    S = stem.stride
    M = _num_positions(stem, T, D)
    stem.zero_grad(set_to_none=True)
    x = torch.randn(1, T, D, requires_grad=True)
    h = stem(x)                                  # (1, M, D)
    for m in (0, M // 2, M - 1):                 # left edge, centre, right edge
        if x.grad is not None:
            x.grad = None
        h[0, m].sum().backward(retain_graph=(m != M - 1))
        g = x.grad[0]                            # (T, D)
        future = g[(m + 1) * S:].abs().max().item() if (m + 1) * S < T else 0.0
        past = g[:(m + 1) * S].abs().max().item()
        assert future <= tol, (
            f"LEAKAGE: position {m} depends on inputs >= {(m+1)*S} "
            f"(max|grad|={future:.3e}). Padding is not causal."
        )
        assert past > 0.0, f"DEGENERATE: position {m} ignores its own and past inputs"
    print(f"[OK] test_causality: stride={S}, T={T}, {M} causal positions, "
          f"RF={stem.receptive_field} bp (zero grad on the future, edges + centre)")


def test_flip_anticausal(stem: CausalConvStem, T: int = 128, D: int = 16,
                         tol: float = 1e-5) -> None:
    """Backward branch = this stem on the reversed sequence. It must come out
    anti-causal in the original coordinates (the condition for weight tying)."""
    S = stem.stride
    M = _num_positions(stem, T, D)
    stem.zero_grad(set_to_none=True)
    x = torch.randn(1, T, D, requires_grad=True)
    h = stem(torch.flip(x, dims=[1]))            # process the reversed sequence
    for m in (0, M // 2, M - 1):
        if x.grad is not None:
            x.grad = None
        h[0, m].sum().backward(retain_graph=(m != M - 1))
        g = x.grad[0]
        boundary = T - (m + 1) * S               # must depend only on inputs >= boundary
        leak = g[:boundary].abs().max().item() if boundary > 0 else 0.0
        assert leak <= tol, (
            f"flip NOT anti-causal: backward pos {m} touches original inputs "
            f"< {boundary} (leak={leak:.3e}). The flip-based tie does not hold."
        )
    print(f"[OK] test_flip_anticausal: stride={S} -> anti-causal in the original "
          f"coordinates (tie preserved)")


def test_shapes_and_head(d_model: int = 144, stride: int = 4, block: int = 4,
                         vocab: int = 12, T: int = 1024, n_stages=None) -> None:
    stem = CausalConvStem(d_model, stride=stride, n_stages=n_stages)
    head = BlockHead(d_model, block=block, vocab_size=vocab)
    x = torch.randn(2, T, d_model)
    h = stem(x)
    M = T // stride
    assert h.shape == (2, M, d_model), h.shape
    logits = head(h)
    assert logits.shape == (2, M * block, vocab), logits.shape
    assert block % stride == 0, (
        f"BLOCK={block} must be a multiple of STRIDE={stride}: the exclusion band "
        f"is measured in grid cells (BAND = BLOCK // STRIDE)."
    )
    n_params = sum(p.numel() for p in stem.parameters()) + \
        sum(p.numel() for p in head.parameters())
    print(f"[OK] test_shapes: stride={stride} block={block} band={block//stride} | "
          f"(2,{T},{d_model}) -> grid (2,{M},{d_model}) -> logits {tuple(logits.shape)} | "
          f"RF={stem.receptive_field} bp | stem+head parameters: {n_params:,}")


def test_v1_equivalence(d_model: int = 32, vocab: int = 12, T: int = 64) -> None:
    """With stride=block=4 the logit ordering must match the v1 ExpansionHead,
    which flattened to (B, M*S, V) with base m*S+j at position m*S+j."""
    head = BlockHead(d_model, block=4, vocab_size=vocab)
    h = torch.randn(2, T // 4, d_model)
    out = head(h)                                     # (2, M*4, V)
    assert out.shape == (2, T, vocab), out.shape
    proj = head.proj(h).reshape(2, T // 4, 4, vocab)
    for m in (0, 3, T // 4 - 1):
        for j in range(4):
            assert torch.equal(out[:, m * 4 + j], proj[:, m, j]), (m, j)
    print("[OK] test_v1_equivalence: logit ordering identical to the v1 ExpansionHead")


if __name__ == "__main__":
    import os as _os
    print("=== CAUSAL STEM GATE (JanusDNA-144) ===\n")

    # FIRST: the configuration THIS launch will actually use, read from the
    # environment. Without this the gate would exercise a fixed list and leave
    # the configuration about to run unverified.
    if _os.environ.get("JANUS_STEM", "0") == "1":
        _S = int(_os.environ.get("JANUS_STEM_STRIDE", "4"))
        _BL = int(_os.environ.get("JANUS_STEM_BLOCK", str(_S)))
        _K = int(_os.environ.get("JANUS_STEM_KERNEL", "7"))
        _NS = _os.environ.get("JANUS_STEM_NSTAGES", "") or None
        _NS = int(_NS) if _NS else None
        print(f"-- CONFIGURATION OF THIS LAUNCH: stride={_S} block={_BL} "
              f"kernel={_K} n_stages={_NS} --")
        _stem = CausalConvStem(d_model=144, stride=_S, kernel_size=_K, n_stages=_NS)
        print(f"   receptive field = {_stem.receptive_field} bp | "
              f"stem parameters = {sum(p.numel() for p in _stem.parameters()):,}")
        test_causality(_stem, T=128, D=144)
        test_flip_anticausal(_stem, T=128, D=144)
        test_shapes_and_head(d_model=144, stride=_S, block=_BL, vocab=12,
                             T=1024, n_stages=_NS)
        print()

    print("-- v1 setups (grid compression) --")
    for S in (2, 4, 8):
        stem = CausalConvStem(d_model=144, stride=S)
        test_causality(stem, T=128, D=144)
        test_flip_anticausal(stem, T=128, D=144)

    print("\n-- new setup: step 1, overlapping windows --")
    for ns, k in ((3, 7), (2, 7), (1, 4)):
        stem = CausalConvStem(d_model=144, stride=1, kernel_size=k, n_stages=ns)
        test_causality(stem, T=128, D=144)
        test_flip_anticausal(stem, T=128, D=144)

    print()
    test_shapes_and_head(d_model=144, stride=4, block=4, vocab=12, T=1024)
    test_shapes_and_head(d_model=144, stride=1, block=4, vocab=12, T=1024, n_stages=3)
    test_v1_equivalence()
    print("\nIf every line reads [OK], causality is DEMONSTRATED, not assumed.")
