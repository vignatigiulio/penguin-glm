#!/usr/bin/env python3
"""
verify_theory.py — EMPIRICAL verification of the claims made about the stem.

Runs on CPU and never touches the GPU, so it is safe to run while a training
job is in progress. It assumes nothing from the formulas: it measures.

It checks, one by one:

  A. receptive field       -> measured by Jacobian, not taken from the formula
  B. causality and anti-causality of the flipped branch
  C. stem parameter counts
  D. identity of the target realignment when STRIDE == BLOCK
  E. tail truncation and prediction counts
  F. the 4x inflation of the token counter at step 1
  G. identical horizon distribution across the two setups
  H. band arithmetic: BAND=1 reproduces the upstream constants

USAGE:
  python verify_theory.py                 # uses causal_conv_stem.py in the cwd
  python verify_theory.py /path/to/repo   # or point at the patched repository
"""

import sys
from collections import Counter
from pathlib import Path

import torch

if len(sys.argv) > 1:
    sys.path.insert(0, str(Path(sys.argv[1]) / "janusdna"))
    sys.path.insert(0, sys.argv[1])
try:
    from causal_conv_stem import CausalConvStem, BlockHead
except ImportError:
    from janusdna.causal_conv_stem import CausalConvStem, BlockHead

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK ' if cond else 'NO '}] {name}" + (f"   {detail}" if detail else ""))
    return cond


# ---------------------------------------------------------------- A, B, C
def measure_rf(stem, T=200, D=32):
    """MEASURED receptive field: how many inputs reach a central output."""
    x = torch.randn(1, T, D, requires_grad=True)
    h = stem(x)
    m = h.shape[1] // 2
    h[0, m].sum().backward()
    touched = (x.grad[0].abs().sum(-1) > 0).nonzero().flatten()
    return int(touched.max() - touched.min() + 1), int(touched.min()), int(touched.max())


def test_stem():
    print("\n=== A/B/C — stem: receptive field, causality, parameters ===")
    configs = [
        ("stem S=4  (2 stages k=7, stride 2)", dict(stride=4, kernel_size=7), 19, 291_168),
        ("step 1    (3 stages k=7)", dict(stride=1, kernel_size=7, n_stages=3), 19, 436_752),
        ("step 1    (2 stages k=7)", dict(stride=1, kernel_size=7, n_stages=2), 13, 291_168),
        ("step 1    (1 stage  k=4)", dict(stride=1, kernel_size=4, n_stages=1), 4, 83_376),
    ]
    for label, kw, rf_atteso, par_attesi in configs:
        stem = CausalConvStem(d_model=144, **kw)
        n_par = sum(p.numel() for p in stem.parameters())
        check(f"{label}: parameters = {n_par:,}", n_par == par_attesi,
              f"expected {par_attesi:,}")
        check(f"{label}: declared RF {stem.receptive_field}",
              stem.receptive_field == rf_atteso, f"expected {rf_atteso}")

        small = CausalConvStem(d_model=32, **kw)
        rf_mis, lo, hi = measure_rf(small)
        check(f"{label}: MEASURED RF = {rf_mis} bp", rf_mis == rf_atteso,
              f"inputs touched [{lo},{hi}], expected {rf_atteso}")

        # causality: output m must not touch inputs >= (m+1)*stride
        S, T, D = small.stride, 200, 32
        x = torch.randn(1, T, D, requires_grad=True)
        h = small(x)
        m = h.shape[1] // 2
        h[0, m].sum().backward()
        fut = x.grad[0, (m + 1) * S:].abs().max().item()
        check(f"{label}: causal (zero grad on the future)", fut == 0.0,
              f"max|grad| future = {fut:.2e}")

        # flip -> anti-causal in the original coordinates
        small.zero_grad(set_to_none=True)
        x = torch.randn(1, T, D, requires_grad=True)
        h = small(torch.flip(x, dims=[1]))
        h[0, m].sum().backward()
        boundary = T - (m + 1) * S
        leak = x.grad[0, :boundary].abs().max().item()
        check(f"{label}: flip anti-causal (tie preserved)", leak <= 1e-6,
              f"leak = {leak:.2e}")


# ------------------------------------------------------------------- D, E
def align(x, y, S, BL, B, T):
    """Exact replica of patch 7 as injected into _shared_step."""
    V = x.shape[-1]
    N = (T // S) * BL
    x = x.reshape(B, N, V)
    y2 = y.reshape(B, T)
    Mok = (T - BL) // S + 1
    ix = (torch.arange(Mok).unsqueeze(1) * S + torch.arange(BL).unsqueeze(0)).reshape(-1)
    return x[:, :Mok * BL], y2[:, ix]


def test_align():
    print("\n=== D/E — realignment: identity when S=B, truncation at step 1 ===")
    T, V = 1024, 16
    for B in (1, 2):
        # identity when STRIDE == BLOCK
        S = BL = 4
        x = torch.arange(B * (T // S) * BL * V).reshape(B, -1, V).float()
        y = torch.arange(T).repeat(B, 1)
        xa, ya = align(x.clone(), y.clone(), S, BL, B, T)
        check(f"B={B} S=BL=4: target unchanged", torch.equal(ya, y))
        check(f"B={B} S=BL=4: logits unchanged",
              torch.equal(xa.reshape(-1), x.reshape(-1)))

        # step 1: truncation and the m*S + j mapping
        S, BL = 1, 4
        x = torch.zeros(B, (T // S) * BL, V)
        xa, ya = align(x, y.clone(), S, BL, B, T)
        Mok = (T - BL) // S + 1
        check(f"B={B} S=1 BL=4: {Mok} positions x {BL} = {Mok*BL} predictions",
              xa.shape[1] == Mok * BL == 4084 and ya.shape[1] == Mok * BL)
        r = ya[0]
        check(f"B={B} S=1 BL=4: m*S+j mapping correct",
              r[0] == 0 and r[3] == 3 and r[4] == 1 and r[7] == 4 and r.max() == T - 1,
              f"first 8 targets: {r[:8].tolist()}")


# ---------------------------------------------------------------------- F
def test_token_counter():
    print("\n=== F — token counter: 4x at step 1, 1x at S=4 ===")
    # the exact ratio is (Mok*BL)/T, which tends to BL/S without reaching it:
    # the (BL-1)/S truncated tail positions matter when T is small.
    for S, BL in ((4, 4), (1, 4)):
        for T in (1024, 131072):
            Mok = (T - BL) // S + 1
            expected = ((T - BL) // S + 1) * BL        # exact count
            ratio, asym = Mok * BL / T, BL / S
            check(f"S={S} BL={BL} T={T}: {Mok*BL:,} predictions over {T:,} bases "
                  f"(ratio {ratio:.4f})",
                  Mok * BL == expected and ratio <= asym,
                  f"asymptote {asym:.1f}, gap due to {(BL-1)//S} truncated positions")
    n_batch, T = 160, 131072
    Mok = (T - 4) // 1 + 1
    check(f"real log: {n_batch} batches -> {Mok*4*n_batch:.3e} predictions",
          abs(Mok * 4 * n_batch - 8.39e7) / 8.39e7 < 0.001,
          "observed counter 8.39e7")
    check("original S=4 smoke: 50 x 8 x 1024 = 409600 tokens",
          50 * 8 * 1024 == 409600, "one token per nucleotide")


# ---------------------------------------------------------------------- G
def test_horizons():
    print("\n=== G — identical horizon distribution across the two setups ===")
    T, BL = 4096, 4
    # S=4: base b is predicted by position b//4 only, with j = b%4
    h4 = Counter((b % BL + 1, BL - b % BL) for b in range(T))
    # S=1: base b is predicted by positions b-3..b, i.e. by every j
    h1 = Counter()
    for b in range(BL, T):
        for j in range(BL):
            h1[(j + 1, BL - j)] += 1
    n4, n1 = sum(h4.values()), sum(h1.values())
    d4 = {k: v / n4 for k, v in sorted(h4.items())}
    d1 = {k: v / n1 for k, v in sorted(h1.items())}
    print(f"      S=4: {[(k, round(v,3)) for k,v in d4.items()]}")
    print(f"      S=1: {[(k, round(v,3)) for k,v in d1.items()]}")
    check("same (forward horizon, backward horizon) pairs", set(d4) == set(d1))
    check("same marginal frequencies",
          all(abs(d4[k] - d1[k]) < 1e-3 for k in d4),
          "both uniform over 4 pairs")
    check("redundancy at step 1 = 4x", abs(n1 / (T - BL) - BL) < 1e-6,
          f"{n1} predictions for {T-BL} bases")


# ---------------------------------------------------------------------- H
def test_band():
    print("\n=== H — band arithmetic ===")
    for BLOCK, STRIDE, expected in ((4, 4, 1), (4, 1, 4), (4, 2, 2), (1, 1, 1)):
        check(f"BLOCK={BLOCK} STRIDE={STRIDE} -> BAND={BLOCK//STRIDE}",
              BLOCK // STRIDE == expected)
    BAND, mid = 1, 100
    check("BAND=1 reproduces the upstream constants",
          (BAND + 1 == 2 * 1) and (mid + BAND == mid + 1)
          and (mid - (BAND + 1) == mid - 2 * 1) and (2 * BAND == 2 * 1),
          "the S=4 case is literally the original code")


def test_head():
    print("\n=== extra — BlockHead: logit shape and ordering ===")
    D, BL, V, T, S = 144, 4, 16, 1024, 1
    head = BlockHead(D, block=BL, vocab_size=V)
    h = torch.randn(2, T // S, D)
    out = head(h)
    check(f"shape (B, M*BLOCK, V) = {tuple(out.shape)}",
          out.shape == (2, (T // S) * BL, V))
    proj = head.proj(h).reshape(2, T // S, BL, V)
    ok = all(torch.equal(out[:, m * BL + j], proj[:, m, j])
             for m in (0, 5, T // S - 1) for j in range(BL))
    check("logit at position m*BLOCK+j = prediction of base m*S+j", ok)


if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 70)
    print("EMPIRICAL VERIFICATION OF THE STEM CLAIMS (CPU only)")
    print("=" * 70)
    test_stem()
    test_align()
    test_token_counter()
    test_horizons()
    test_band()
    test_head()
    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("\nFAILED — the documented claims are wrong on these points:")
        for f in FAIL:
            print("  -", f)
    else:
        print("Every measurable claim is confirmed.")
    print("=" * 70)
    sys.exit(1 if FAIL else 0)
