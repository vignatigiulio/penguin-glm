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

The two 19 bp configurations differ in depth and parameter count (a further
0.4% of the total capacity). This is unavoidable, since no two-stage stride-1
stem reaches 19 bp, and should be stated in any comparison.

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

The parametrisation is `BAND = BLOCK / STRIDE`, applied in two places: the
FlexAttention mask and the fusion offsets. Patching the mask alone has no effect,
because the fusion keeps picking slot `p + 1` regardless.

Measured on the real model (`verify_leakage_map.py`, T = 256, random weights,
since self-exclusion is structural; `run_leakage_check.sh` runs both cases):

| position | target bases | j = 0 | j = 1 | j = 2 | j = 3 | context |
|---|---|---|---|---|---|---|
| **BAND = 4 (correct)** ||||||
| 4 | 4–7 | 0 | 0 | 0 | 0 | 600 |
| 128 | 128–131 | 0 | 0 | 0 | 0 | 218 |
| 251 | 251–254 | 0 | 0 | 0 | 0 | 631 |
| **BAND = 1 (upstream value)** ||||||
| 4 | 4–7 | 0 | 36.1 | 62.5 | 80.2 | 696 |
| 128 | 128–131 | 0 | 31.2 | 58.7 | 82.1 | 195 |
| 251 | 251–254 | 0 | 63.4 | 168.0 | 322.0 | 543 |

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
constant.

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
nucleotides: each base is predicted four times. Comparing two runs on
`num_tokens` would stop the step-1 run at a quarter of the data. Parity is on
steps, batches and bases, which `MAX_STEPS = 50000` guarantees.

**`main_loss` is a mean over 4x predictions** with fourfold redundancy, so it is
not directly comparable between the step-1 run and `S = 4`.

**The logged perplexity is `exp(main_loss + moe_loss)`**, which includes the
auxiliary load-balancing term of the experts. With `moe_loss` ≈ 0.8 it would
overestimate the perplexity by a factor e^0.8 ≈ 2.2; the correct quantity is
`exp(main_loss)`.

**Validation and test losses are computed on the minor scaffolds reserved to
those splits**, largely repetitive and easier to predict than the training
corpus. `test/loss` is therefore not an absolute yardstick across splits, though
it remains valid across runs.

## Results

The run completed its 50,000 steps. The prediction recorded above held in its
second branch: the NMI against GC stays at 0.2759, against 0.2591 for `S = 4` and
0.4038 for the baseline, so 88% of the attenuation survives the return to the
full grid and grid compression is not its cause. Window and objective remain
confounded; the fourth cell (`STEM_BLOCK=1`) was not run.

| | NMI vs GC | LDA η² | probe MCC (configuration 2) | LoRA MCC (saturated) |
|---|---:|---:|---:|---:|
| baseline | 0.4038 | 0.4035 | 0.3232 | 0.4096 |
| stem S = 4 | 0.2591 | 0.4064 | 0.3238 | 0.3890 |
| step 1 | 0.2759 | 0.5070 | 0.3687 | 0.4136 |

On frozen embeddings step 1 is the most informative model (LDA η² +25.7%, probe
MCC +14.1% over the baseline); after LoRA at saturation the margin is +1.0%,
within the test uncertainty. Extracting from the test-loss minimum (epoch 50)
instead of the last checkpoint (epoch 55) moves the NMI by 1.2%. The README
reports the full set of measurements, the human-checkpoint counterexample and the
per-dimension GC analysis.

## Measured cost

About 144 hours of compute for the step-1 run against 69 for `S = 4` on one
H100, a ratio of **2.1x**, much lower than the asymptotic complexity suggests
(linear in the backbone, quadratic in the fusion layer).

## Verification

| script | needs | what it establishes |
|---|---|---|
| `verify_theory.py` | CPU | 44 checks: receptive field measured by Jacobian rather than taken from the formula, causality, flip anti-causality, parameter counts, realignment identity at `STRIDE = BLOCK`, tail truncation, prediction/base ratio, horizon distribution, band arithmetic |
| `verify_patches.py` | none | the 8 patches are present, no hunk falls outside the intended regions, and `BAND = 1` reproduces the upstream constants |
| `verify_leakage_map.py` | GPU | the dependency map above, with and without the correct band (`run_leakage_check.sh` runs both) |
| `causal_conv_stem.py` | CPU | standalone stem gate, including the configuration read from the environment |
| `test_stem_wiring.py` | GPU | full-model self-exclusion, wired into `on_fit_start` |
| `tools/check_stem_config.py` | none | derives stage count, kernel and block from the checkpoint and refuses to proceed if the environment disagrees |

`check_stem_config.py` exists because two mismatches are silent: a wrong stage
count leaves only four unexpected keys, below any reasonable count threshold,
and a wrong `BLOCK` changes `BAND` without touching any parameter at all.
