# Step-1 tokenisation experiment

Reference for the `stem/` scripts in their v2 form. Written for someone
reproducing the work, not as a narrative of how it was developed.

## What the previous comparison could not separate

Going from the baseline to the `S = 4` conv stem changed three things at once:

| | grid (per 1000 bp window) | window seen per position | bases predicted |
|---|---|---|---|
| baseline | 1000 | 1 | 1 |
| stem S = 4 | 250 | 19 bp | 4 |

The measured attenuation of the GC attractor (NMI against GC content, 0.404 →
0.259) is therefore attributable to any of the three, or to a combination.

Note in particular that the stem at `S = 4` did **not** see disjoint 4-base
blocks: two stride-2 convolutions with kernel 7 give a receptive field of 19 bp,
so adjacent super-tokens overlap by 15 bases. Only the prediction *targets* were
disjoint.

## What the step-1 run isolates

Running at step 1 returns the grid to the baseline density while keeping the
objective and the window unchanged, which isolates grid compression alone.

For the isolation to hold, the receptive field must stay at 19 bp. Since
`RF = 1 + n_stages * (kernel - 1)` for stride-1 stages, three stages with
kernel 7 give `1 + 3*6 = 19`.

| configuration | grid | window | stem parameters |
|---|---|---|---|
| stem S = 4 (2 stages k=7, stride 2) | 250 | 19 bp | 291,168 |
| **step 1 (3 stages k=7, stride 1)** | **1000** | **19 bp** | **436,752** |
| step 1 (2 stages k=7) | 1000 | 13 bp | 291,168 |
| step 1 (1 stage k=4) | 1000 | 4 bp | 83,376 |

The two 19 bp configurations differ in depth and parameter count (+0.4% of the
35.9M total). This is unavoidable — no two-stage stride-1 stem reaches 19 bp —
and should be stated in any comparison rather than glossed over.

## The exclusion band

JanusDNA's self-exclusion comes from the offsets in the final fusion
(`final_fusion_repos_formasked`, active when `layer_fusion_strategy = "pool"`):

```
fused[p] = hidden[p - 1] + hidden[mid_len + p + BAND]
```

The forward branch at index `p - 1` summarises everything before `p`; the
backward branch at `p + BAND` summarises everything from `p + BAND` on. The loss
is a plain cross-entropy with no shift, so the prediction is encoded entirely in
these offsets.

`BAND` is hardcoded to 1 upstream, in **grid cells**. At `S = 4` one grid cell
is worth four bases, so the band happened to cover the whole predicted block. At
step 1 a cell is worth one base while the block is still four, so three of the
four target bases would be visible to the backward branch. The model would copy
rather than predict: lower loss, better-looking curves, no error on screen.

The parametrisation is `BAND = BLOCK / STRIDE`, applied in two places — the
FlexAttention mask and the fusion offsets. Patching the mask alone has no effect,
because the fusion keeps picking slot `p + 1` regardless.

Measured on the real model (`verify_leakage_map.py`, T = 256, random weights,
since self-exclusion is structural):

| position | target bases | j = 0 | j = 1 | j = 2 | j = 3 | context |
|---|---|---|---|---|---|---|
| **BAND = 4 (correct)** ||||||
| 4 | [4, 8) | 0 | 0 | 0 | 0 | 6.00e+02 |
| 128 | [128, 132) | 0 | 0 | 0 | 0 | 2.18e+02 |
| 251 | [251, 255) | 0 | 0 | 0 | 0 | 6.31e+02 |
| **BAND = 1 (upstream value)** ||||||
| 4 | [4, 8) | 0 | 3.61e+01 | 6.25e+01 | 8.02e+01 | 6.96e+02 |
| 128 | [128, 132) | 0 | 3.12e+01 | 5.87e+01 | 8.21e+01 | 1.95e+02 |
| 251 | [251, 255) | 0 | 6.34e+01 | 1.68e+02 | 3.22e+02 | 5.43e+02 |

Exactly one base out of four stays protected with the old band, and the leakage
grows with `j`: the base furthest from the forward branch is the one the
backward branch summarises best.

## Interface

Three parameters replace the single `S` of v1:

```
STEM_STRIDE   grid compression      (1 = none)
STEM_BLOCK    bases predicted per fused position
STEM_NSTAGES  number of stem stages (stride=1 only; default 2)
STEM_KERNEL   convolution kernel    (default 7)
```

`STEM_STRIDE = STEM_BLOCK` reproduces v1 exactly: `verify_patches.py` checks
that with `BAND = 1` every parametrised expression evaluates to the original
constant, and the `S = 4` smoke run reproduces the pre-change loss to within the
run-to-run spread.

```bash
# reproduction of the v1 S=4 run
JANUS_STEM_GATE=1 STEM_STRIDE=4 STEM_BLOCK=4 ./run_stem_experiment.sh

# counter-proof: MUST fail with a leakage assertion
JANUS_STEM_GATE=1 STEM_STRIDE=1 STEM_BLOCK=4 \
  JANUS_STEM_BAND_OVERRIDE=1 ./run_stem_experiment.sh

# the step-1 run
JANUS_STEM_GATE=1 STEM_STRIDE=1 STEM_BLOCK=4 STEM_NSTAGES=3 \
  ./run_stem_experiment.sh full
```

## Reading the metrics

**The token counter is inflated 4x at step 1.** It counts predictions, not
nucleotides: each base is predicted four times. Measured on the logs, 160
batches report 8.39e7 against 2.10e7 bases actually seen. At `S = 4` the two
coincide. Comparing two runs on `num_tokens` would stop the step-1 run at a
quarter of the data. Parity is on steps, batches and bases, which
`MAX_STEPS = 50000` guarantees.

**`main_loss` is a mean over 4x predictions** with fourfold redundancy. The
marginal distribution of prediction horizons is identical between the two setups
(each base gets forward horizon `j+1` and backward horizon `4-j`, uniform over
`j = 0..3`), but the quantity is not bits-per-nucleotide in the baseline sense.

**The logged perplexity is `exp(main_loss + moe_loss)`**, which includes the
router's auxiliary loss and is unrelated to nucleotide prediction. On the smoke
run: logged 8.788, `exp(total)` 8.788, `exp(main_loss)` 4.081. The correct
quantity is `exp(main_loss)`; the logged value is inflated by
`exp(moe_loss) ≈ 2.15`.

**Validation and test loss come out well below training loss** in both runs
(S = 4: train 1.140, val 0.588, test 0.293). The pattern is identical across
runs, so it reflects the scaffold split — the held-out scaffolds are easier,
presumably richer in repeats — and not the tokenisation. Per-window variance is
large in training too, with single-step losses ranging from 0.14 to 1.3 within
one epoch. `test/loss` is therefore not an absolute yardstick across splits,
though it remains valid across runs.

## Measured cost

| run | throughput | 400,000 batches |
|---|---|---|
| stem S = 4 | 1.61 it/s | 69 h |
| step 1 (3 stages) | 0.77 it/s | 144 h |

The real ratio is **2.1x**, not the 4x-16x an asymptotic argument suggests
(linear in the backbone, quadratic in the fusion layer). At `d_model = 144` the
model already leaves the GPU underused at 32,768 positions, so quadrupling the
grid fills the hardware rather than multiplying wall-clock time.

## Verification

| script | needs | what it establishes |
|---|---|---|
| `verify_theory.py` | CPU | 44 checks: receptive field measured by Jacobian rather than taken from the formula, causality, flip anti-causality, parameter counts, realignment identity at `STRIDE = BLOCK`, tail truncation, prediction/base ratio, horizon distribution, band arithmetic |
| `verify_patches.py` | none | the 8 patches are present, no hunk falls outside the intended regions, and `BAND = 1` reproduces the upstream constants |
| `verify_leakage_map.py` | GPU | the dependency map above, with and without the correct band |
| `causal_conv_stem.py` | CPU | standalone stem gate, including the configuration read from the environment |
| `test_stem_wiring.py` | GPU | full-model self-exclusion, wired into `on_fit_start` |
| `tools/check_stem_config.py` | none | derives stage count, kernel and block from the checkpoint and refuses to proceed if the environment disagrees |

`check_stem_config.py` exists because two mismatches are silent: a wrong stage
count leaves only four unexpected keys, below any reasonable count threshold,
and a wrong `BLOCK` changes `BAND` without touching any parameter at all.
