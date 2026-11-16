# penguin-glm

**Genomic language models on a non-model species: functional element classification in the king penguin genome.**

This repository contains the full experimental pipeline for a Master's thesis in
Computer Engineering at Università Politecnica delle Marche. It evaluates whether
genomic language models (gLMs) pre-trained on the human genome transfer to
*Aptenodytes patagonicus* (king penguin, RefSeq `GCF_965638725.1`), and whether
pre-training from scratch on the target genome closes the gap.

Two architectures are studied side by side under an identical downstream protocol:

| Model | Architecture | Sizes used |
|---|---|---|
| **CrossDNA** | Comba + sliding-window attention, dual-strand (branches A/B) | 8.1M, 519M |
| **JanusDNA** | Mamba + Mixture-of-Experts, bidirectional weight tying | 72, 144 |

The thesis also contains an exploratory study with Evo 2 (per-base
log-likelihood across functional categories). It ran on a different cluster, and
neither its code nor its outputs are part of this repository.

---

## The main result

**Zero-shot gLM embeddings organise by GC content, not by biological function.**
Clustering (k-means, k = 4) the frozen embeddings of 120,000 genomic windows
recovers nucleotide composition far more strongly than the functional
annotation:

| Embedding | NMI vs. function | NMI vs. GC | NMI vs. repeats | silhouette |
|---|---:|---:|---:|---:|
| CrossDNA-519M (human) | 0.0175 | 0.518 | 0.044 | 0.227 |
| JanusDNA-144 (human) | 0.0192 | 0.182 | 0.027 | 0.187 |

The functional signal is nonetheless *present*: it is simply not the dominant
axis of the latent geometry. A supervised probe extracts it, and LoRA
domain adaptation extracts substantially more. The dissociation between
unsupervised geometry and extractable information is the central claim of the
thesis. The window-length sweep and the step-1 run below are its clearest
demonstrations.

---

## Task and data

**Task.** 4-way classification of functional elements (CDS, UTR, intron,
intergenic) derived from the RefSeq GFF annotation.

**Windows.** 120,000 fixed-length windows, balanced across the four classes
(30,000 each). Labels are assigned by the *centre* of the window, and UTRs are
derived as mRNA exons minus CDS rather than read directly from the GFF.

**Split.** Chromosome-disjoint 80k / 20k / 20k (train / val / test). Every
scaffold belongs to exactly one split, so no genomic region is shared across
splits.

**Confounder audit.** GC content, soft-masked repeat fraction and N fraction are
recorded per window and used both as an explicit baseline and as a colouring
variable in the projections. This is what makes the GC result above measurable
rather than anecdotal.

---

## Results

All figures below are on the held-out test set (20,000 windows), window length
L = 1000 bp unless stated otherwise. MCC = Matthews correlation coefficient.

**Two probe configurations.** Frozen embeddings are scored with a two-layer MLP
(hidden 256) in one of two configurations. Values are compared only within the
same configuration, and with the compositional reference computed in it:

| | script | dropout | max epochs | patience | used for |
|---|---|---:|---:|---:|---|
| configuration 1 | `src/mlp_probe_compare.py` | 0.1 | 30 | 5 | zero-shot comparisons between models (single run and 4 seeds) |
| configuration 2 | `src/mlp_probe.py` | 0.3 | 60 | 8 | the GC-attractor analyses (stem sweep and step-1 run) |

The test MCC has a sampling uncertainty of about ±0.009 (95% interval, bootstrap
on the confusion matrices): differences below about 0.015 between single runs are
not read as significant.

### Baselines

| Baseline | Classifier | MCC |
|---|---|---:|
| Trivial: GC + N fraction | | 0.061 |
| Trivial: GC + repeat fraction + N fraction | | 0.124 |
| 4-mer composition | random forest | 0.243 |
| 4-mer composition | MLP probe, configuration 1 (4 seeds) | **0.301 ± 0.002** |
| 4-mer composition | MLP probe, configuration 2 (4 seeds) | **0.297 ± 0.002** |

The repeat fraction comes from the soft-masking of the FASTA, which the
tokenizer removes, so the trivial baseline is read between its two values. The
fair bar for an embedding is composition scored with the same classifier: 0.301
(configuration 1) or 0.297 (configuration 2). The random forest figure
understates it. `src/kmer_mlp_baseline.py` produces both MLP rows.

### Zero-shot (frozen embeddings, probe configuration 1)

Single run, same probe and same windows for every model:

| Representation | dim | MCC |
|---|---:|---:|
| JanusDNA-144 (human) | 144 | 0.3129 |
| CrossDNA-519M (human) | 1024 | 0.2884 |
| CrossDNA-8.1M (human) | 128 | 0.2741 |
| JanusDNA-72 (human) | 72 | 0.2691 |
| 4-mer composition, MLP probe (4 seeds) | | 0.301 |
| 4-mer composition, random forest | | 0.243 |
| Trivial (GC + repeat fraction + N fraction) | | 0.124 |

Pre-training on the penguin genome against the human checkpoint, 4 seeds:

| Model | from scratch (penguin) | human | difference |
|---|---:|---:|---:|
| JanusDNA-144 | 0.322 ± 0.002 | 0.309 ± 0.004 | +0.013 |
| CrossDNA-8.1M | 0.2881 ± 0.0005 | 0.2691 ± 0.0015 | +0.0190 ± 0.0020 |

Against the 4-mer probe at 0.301, only JanusDNA-144 pre-trained on the penguin
genome is clearly above composition; the models pre-trained on the human genome
are at its level or below. In-domain pre-training helps, but modestly.
`analysis/audit_results.py` flags any feature that has been evaluated under more
than one protocol.

### After LoRA domain adaptation

| Model | Pre-training corpus | LoRA budget | MCC |
|---|---|---|---:|
| JanusDNA-144 | penguin, from scratch (clean) | to saturation | 0.4096 |
| JanusDNA-144 | human | to saturation | 0.4080 |
| CrossDNA-8.1M | penguin, from scratch (clean) | 10 epochs | 0.4216 |
| CrossDNA-8.1M | human (fp32) | 10 epochs | 0.3856 |
| CrossDNA-519M | human | | 0.447 |

"To saturation" means an extended budget (25 epochs) with early stopping on
validation MCC (`src/run_lora_saturated.sh`); both JanusDNA runs reached their
validation maximum before the end of the budget. The CrossDNA runs were not
repeated with the larger budget.

After adaptation the in-domain advantage behaves differently in the two
architectures: for JanusDNA-144 it is reabsorbed (+0.013 zero-shot, +0.002 after
LoRA with the extended budget), for CrossDNA-8.1M it is amplified (+0.019
zero-shot, +0.036 after LoRA). The two LoRA values come from different budgets:
at 10 epochs, the budget used for CrossDNA, the JanusDNA advantage was +0.018
(0.4028 against 0.3848), and it vanished only when adaptation was carried to
saturation.

### Window-length sweep

The same protocol run at eight window lengths, on JanusDNA-144 pre-trained
from scratch on the clean penguin corpus:

| L (bp) | zero-shot MCC | LoRA MCC | LoRA gain | 4-mer | NMI vs. function | NMI vs. GC |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.2155 ± 0.0073 | 0.2146 ± 0.0015 | **−0.0009** | 0.1623 | 0.0549 | 0.4038 |
| 100 | 0.2334 ± 0.0031 | 0.2492 ± 0.0032 | +0.0158 | 0.1812 | 0.0601 | 0.4038 |
| 150 | 0.2617 ± 0.0033 | 0.2827 ± 0.0013 | +0.0210 | 0.1955 | 0.0619 | 0.4096 |
| 200 | 0.2774 ± 0.0025 | 0.2979 ± 0.0099 | +0.0205 | 0.2063 | 0.0593 | 0.4130 |
| 300 | 0.2946 ± 0.0021 | 0.3239 ± 0.0036 | +0.0293 | 0.2121 | 0.0564 | 0.4190 |
| 500 | 0.3078 ± 0.0013 | 0.3604 ± 0.0016 | +0.0526 | 0.2099 | 0.0510 | 0.4150 |
| 1000 | 0.3192 ± 0.0031 | 0.4048 ± 0.0055 | +0.0856 | 0.2430 | 0.0463 | 0.4034 |
| 2000 | 0.3411 ± 0.0064 | 0.4424 ± 0.0141 | **+0.1013** | 0.2770 | 0.0448 | 0.4002 |

Zero-shot on three seeds at every length; LoRA for 10 epochs, on three seeds at
L = 64 and two elsewhere. At 10 epochs JanusDNA-144 is not yet saturated, so the
values after LoRA are lower bounds. The 4-mer column is the compositional
baseline at the same length. `analysis/make_figure_window_sweep.py` aggregates
the `out_L*/` directories and draws the figure.

1. **At L = 64, LoRA adds nothing at all** (−0.0009). At L = 2000 it adds
   +0.1013.
2. **NMI against GC is flat** (0.4002 to 0.4190) across the entire sweep, while
   supervised MCC nearly doubles. The latent geometry does not reorganise; the
   information becomes *extractable* without becoming *dominant*.
3. After LoRA the model stays between 1.32 and 1.72 times the 4-mer baseline at
   the same length.

---

## The causal convolutional stem experiment

**Question.** Is the GC attractor a consequence of single-nucleotide
tokenisation? If the model reasons over super-tokens instead of single bases,
does the latent space stop being organised by composition?

**Design.** A causal down-sampling stem (stride-2 stages of kernel 7) inserted
before the JanusDNA Mamba+MoE backbone, giving compression S with a receptive
field of 19 bp at S = 4, 43 bp at S = 8 and 91 bp at S = 16 (measured with the
Jacobian of the stem output rather than taken from the formula:
`stem/verify_theory.py`). Three architectural invariants were
verified before any run: the backward stream uses flip-based weight tying (one
stem, not two), the Janus loss is non-shifted, and LayerNorm is per-position and
therefore causally safe. A standalone gate proves causality through the
Jacobian, and a full-model gate (`stem/test_stem_wiring.py`) confirms the stem
preserves Janus self-exclusion.

**Falsifiable prediction, stated in advance.** An improvement in the
NMI(function)/NMI(GC) *ratio* would indicate a mechanistic gain; an improvement
in MCC alone would indicate a merely computational one.

**Budget.** Identical for every scale: about 52 billion bases, `max_length=131072`,
50,000 steps, `global_batch=8`, `lr=8e-3`, bf16-mixed, one H100. The budget is
identical *across the scales*, not with the baseline: the baseline runs 15,000
steps and ends near its own optimum, while the stem runs end past theirs (see
[Known issues](#known-issues); `tools/check_budgets.py` reads the step count from
each checkpoint). Note also that the logged `num_tokens` counts *predictions*,
which equals nucleotides only when the grid stride matches the predicted block;
see [Separating grid compression from the objective](#separating-grid-compression-from-the-objective).

### Sweep over the compression scale

| Scale | NMI vs GC | NMI vs function | ratio f/GC | silhouette | probe MCC (config. 2) | LoRA MCC | effective dim |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (no stem) | 0.4038 | 0.0462 | 0.114 | 0.199 | 0.3232 | **0.4096** | 3 |
| **S = 4** (19 bp) | 0.2591 | **0.0581** | **0.224** | **0.462** | **0.3238** | 0.3890 | 2 |
| S = 8 (43 bp) | 0.2886 | 0.0526 | 0.182 | 0.387 | 0.2920 | 0.3666 | 1 |
| S = 16 (91 bp) | **0.2187** | 0.0401 | 0.183 | 0.437 | 0.2594 | 0.3309* | 1 |

LoRA at saturation, except *S = 16 (10 epochs: its saturated run was not
completed). S = 32 was pre-trained but is excluded from the downstream comparison;
see [The S = 32 exclusion](#the-s--32-exclusion).

**The scales are not equivalent, and S = 4 is an optimum.** Three readings:

1. **Extractability degrades monotonically with compression**, in both regimes:
   frozen probe 0.324 → 0.292 → 0.259, after LoRA 0.389 → 0.367 at saturation
   and 0.331 for S = 16. Only S = 4 matches the baseline on frozen embeddings.
2. **After adaptation the gap from the baseline persists and grows with S**:
   −5.0% at S = 4 and −10.5% at S = 8 with LoRA at saturation. Compression
   costs information that in-domain adaptation does not recover.
3. **Reducing GC is not the same as improving.** S = 16 weakens the GC attractor
   more than any other scale (0.2187, −46% from baseline) and loses on
   everything else, including the lowest NMI vs function of the whole sweep. It
   is flattening the representation, not purifying it.

The ratio declared in advance as the discriminating metric picks out S = 4
(0.224) clearly over S = 8 and S = 16 (both ~0.18).

### Why S = 16 wins on GC and loses on everything else

| Scale | var. PC1 | eff. dim (80%) | PC2 corr. GC | mean corr. GC | LDA η² | silhouette |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.659 | 3 | 0.880 | 0.398 | 0.4035 | 0.199 |
| **S = 4** | 0.672 | 2 | **0.739** | 0.355 | 0.4064 | 0.462 |
| S = 8 | 0.807 | **1** | 0.908 | 0.286 | 0.3654 | 0.387 |
| S = 16 | 0.809 | **1** | 0.928 | 0.240 | 0.3039 | 0.437 |

Two facts contradict any extrapolation from NMI alone. **PC2's correlation with
GC rises** for S = 8 and S = 16 (0.908, 0.928 against the baseline's 0.880): the
dominant axis becomes *more* GC-centric, not less. And **the representation
collapses**: PC1 alone explains 81% of the variance and the effective
dimensionality falls to one.

At high compression GC becomes less *diffuse* (mean correlation 0.286 and 0.240)
but more *concentrated* in the dominant axis, together with everything else. The
falling NMI vs GC at S = 16 is therefore a symptom of an impoverished
representation, not of a purified one.

**S = 4 is the only scale that reorganises without collapsing**: it keeps
dimensionality, reduces the GC axis, and preserves the baseline's linear
extractability (LDA η² 0.4064 against 0.4035).

### Verdict against the prediction

At S = 4, **mechanistic gain without computational gain**. The stem measurably
weakens the GC attractor (the function/GC ratio nearly doubles, the clusters
separate far better) and none of it converts into downstream accuracy, while
under LoRA it costs about 0.02 MCC. That is a negative result on the metric that
matters practically, and a positive one on the metric the experiment was
designed to test. Beyond S = 4 both gains disappear together.

**Consequence for the multi-scale idea.** The premise of an Inception-style
multi-scale stem holds (the scales do behave differently), but the data show no
*complementarity*: no scale captures function better than the others, and all
degrade as S grows. A module fusing S ∈ {4, 8, 16} would combine one good
extractor with two that degrade extractability.

### Separating grid compression from the objective

The sweep above leaves a confound. Going from the baseline to the `S = 4` stem
changes three things at once:

| | grid (per 1000 bp window) | window seen per position | bases predicted |
|---|---:|---:|---:|
| baseline | 1000 | 1 | 1 |
| stem S = 4 | 250 | 19 bp | 4 |

The measured attenuation of the GC attractor is therefore attributable to any of
the three.

Note also that the `S = 4` stem did **not** see disjoint 4-base blocks. Two
stride-2 convolutions with kernel 7 give a 19 bp receptive field, so adjacent
super-tokens overlap by 15 bases. Only the prediction *targets* were disjoint.

**The step-1 run.** Running the same stem at stride 1 restores the baseline grid
density while holding the receptive field and the objective fixed, which isolates
grid compression alone. The receptive field is kept at 19 bp with three stride-1
stages of kernel 7 (`RF = 1 + n_stages·(kernel − 1)`).

| configuration | grid | window | stem parameters |
|---|---:|---:|---:|
| stem S = 4 (2 stages k=7, stride 2) | 250 | 19 bp | 291,168 |
| **step 1 (3 stages k=7, stride 1)** | **1000** | **19 bp** | **436,752** |

The two 19 bp configurations differ in depth and parameter count (a further
0.4% of the total capacity). No two-stage stride-1 stem reaches 19 bp, so this is unavoidable
and is stated rather than glossed over.

**The exclusion band had to be parametrised.** JanusDNA's self-exclusion comes
from the offsets in the final fusion (`final_fusion_repos_formasked`, active when
`layer_fusion_strategy = "pool"`):

```
fused[p] = hidden[p - 1] + hidden[mid_len + p + BAND]
```

The forward branch at `p - 1` summarises everything before `p`, the backward
branch at `p + BAND` everything from `p + BAND` on, and the loss is a plain
unshifted cross-entropy: the prediction is encoded entirely in these offsets.
`BAND` is hardcoded to 1 upstream, **in grid cells**. At `S = 4` one cell is
worth four bases, so the band happened to cover the whole predicted block. At
stride 1 a cell is worth one base while the block is still four, so three of the
four target bases would be visible to the backward branch: the model would copy
rather than predict, with a lower loss, better-looking curves and no error on
screen.

The parametrisation is `BAND = BLOCK / STRIDE`, applied in the FlexAttention mask
*and* in the fusion offsets. Patching the mask alone has no effect: the fusion
keeps picking slot `p + 1` regardless.

Measured on the real model with `stem/verify_leakage_map.py` (T = 256, random
weights, since self-exclusion is structural and independent of training;
`stem/run_leakage_check.sh` runs both cases with a fixed seed):

| position | target bases | j = 0 | j = 1 | j = 2 | j = 3 | context |
|---|---|---:|---:|---:|---:|---:|
| **BAND = 4 (correct)** | | | | | | |
| 4 | 4–7 | 0 | 0 | 0 | 0 | 600 |
| 128 | 128–131 | 0 | 0 | 0 | 0 | 218 |
| 251 | 251–254 | 0 | 0 | 0 | 0 | 631 |
| **BAND = 1 (upstream value)** | | | | | | |
| 4 | 4–7 | 0 | 36.1 | 62.5 | 80.2 | 696 |
| 128 | 128–131 | 0 | 31.2 | 58.7 | 82.1 | 195 |
| 251 | 251–254 | 0 | 63.4 | 168.0 | 322.0 | 543 |

Exactly one base out of four stays protected with the old band, and the leakage
grows with `j`: the base furthest from the forward branch is the one the backward
branch summarises best.

**Falsifiable prediction, stated in advance.** If the NMI against GC returns
towards the baseline 0.404, the attenuation was grid compression and the
attribution to the stem must be rewritten. If it stays near 0.259, compression is
irrelevant, but window and objective remain confounded with each other, since
both separate the baseline from either stem run.

**Result.** The attenuation is preserved at **88%**: restoring the baseline grid
density moves the NMI against GC from 0.2591 to 0.2759, against 0.4038 for the
baseline. Grid compression is not the cause. The second branch of the prediction
holds, and the attribution to the block objective remains undemonstrated, since
window and objective are still confounded with each other.

| | NMI vs GC | LDA η² | LDA1 corr. GC | probe MCC (config. 2) | LoRA MCC (saturated) |
|---|---:|---:|---:|---:|---:|
| human checkpoint (JanusDNA-144) | **0.1824** | 0.3411 | 0.340 | 0.3148 | 0.4080 |
| baseline (from scratch) | 0.4038 | 0.4035 | 0.287 | 0.3232 | 0.4096 |
| stem S = 4 | 0.2591 | 0.4064 | 0.292 | 0.3238 | 0.3890 |
| stem S = 8 | 0.2886 | 0.3654 | | 0.2920 | 0.3666 |
| stem S = 16 | 0.2187 | 0.3039 | 0.334 | 0.2594 | |
| **step 1** | 0.2759 | **0.5070** | **0.258** | **0.3687** | **0.4136** |
| 4-mer composition, MLP probe | | | | 0.297 ± 0.002 | |

**On frozen embeddings step 1 is the most informative model.** It is the only
run that attenuates GC *and* makes the functional signal more extractable: LDA η²
+25.7% over the baseline, probe MCC +14.1%, and the LDA axis least correlated
with GC (0.258 against 0.287 for the baseline and 0.292 for `S = 4`). The
`S = 4` stem achieved only the first (LDA η² +0.7%, MCC +0.2%). A linear
discriminant and a non-linear MLP probe agree in direction and magnitude.
Against the 4-mer probe of the same configuration (0.297 ± 0.002) the baseline
and `S = 4` are about 0.026 above, `S = 8` is at its level and `S = 16` falls
below it (−0.038).

**After adaptation the differences close.** With LoRA run to saturation, step 1
leads the baseline by +1.0% (0.4136 against 0.4096), within the bootstrap
uncertainty of the test MCC (about ±0.009). The same gradient appears in every
comparison: largest on the linear analysis, intermediate on the probe, smallest
after LoRA. Step 1 against the baseline: +25.7%, +14.1%, +1.0%. The human
checkpoint against the baseline: −15.5%, −2.6%, −0.4%. The more freedom the
adaptation has, the more it compensates for the starting representation, so the
effect of a tokenisation is best measured on frozen embeddings.

**A low NMI against GC is not a goal in itself.** The released human checkpoint
has the least GC-dominated embeddings of anything measured (0.1824) and less
functional signal on frozen embeddings than the from-scratch baseline (LDA η²
−15.5%). What makes step 1 meaningful is not that it has the least GC (it does
not) but that it is the only model that lowers GC *and* raises extractability.

**GC is redistributed, never removed.** A ridge regression recovers GC from the
embeddings with held-out R² between 0.996 and 0.998 in every model. The share of
variance along the GC direction does not follow the NMI: 1.3% for the baseline,
2.1% for `S = 4`, 3.5% for the human checkpoint and 0.7% for step 1, the share of an arbitrary direction (1/144).
A redistribution of variance away from GC is therefore documented for step 1
only; the other models lower the NMI through different geometric routes
(`analysis/gc_per_dimension.py`). The step-1 geometry differs from `S = 4`: PC2
stays at 0.902 correlation with GC (0.739 at `S = 4`), PC1 rises to 0.707 of the
variance and the effective dimensionality falls from 3 to 2.

**The choice of checkpoint does not matter.** The test-loss minimum is saved as
`checkpoints/test/loss.ckpt` (epoch 50). Extracting from it instead of the final
checkpoint (epoch 55), the test loss worsens by 11% while the NMI against GC
moves by 1.2% (0.2726 against 0.2759), a little over 2% of the gap between
baseline and `S = 4`.

**Compression damages extractability on both metrics.**

| compression | probe MCC (config. 2) | LoRA MCC | LoRA vs baseline |
|---|---:|---:|---:|
| S = 1 (step 1) | 0.3687 | 0.4136 | +1.0% |
| S = 4 | 0.3238 | 0.3890 | −5.0% |
| S = 8 | 0.2920 | 0.3666 | −10.5% |
| S = 16 | 0.2594 | 0.3309 (10 epochs) | |

**Still open.** Window and objective remain confounded. The fourth cell of the
design (stride 1, 19 bp, a single-base head: `STEM_BLOCK=1`) would separate them
and is supported by the code unchanged, but it was not run in this work. The
comparison of step 1 with `S = 4` is budget-matched (50,000 steps each); the
comparison with the baseline is not (15,000 steps).

**Measured cost.** About 144 hours of compute against 69 for `S = 4` on one H100,
a ratio of **2.1x**, much lower than the asymptotic complexity suggests (linear in
the backbone, quadratic in the fusion layer).

**Reading the metrics at stride 1.** Three quantities change meaning and are easy
to misread:

- **The token counter is inflated 4x.** It counts predictions, not nucleotides:
  each base is predicted four times, once per offset. Comparing runs on
  `num_tokens` would stop the step-1 run at a quarter
  of the data. Parity is on steps, batches and bases, which `MAX_STEPS = 50000`
  guarantees.
- **`main_loss` is a mean over 4x predictions.** The marginal distribution of
  prediction horizons is identical between the two setups (each base gets
  forward horizon `j+1` and backward horizon `4−j`, uniform over `j = 0..3`), but
  the quantity is not bits-per-nucleotide in the baseline sense.
- **The logged perplexity is `exp(main_loss + moe_loss)`**, which includes the
  auxiliary load-balancing term of the experts. With `moe_loss` ≈ 0.8 it would
  overestimate the perplexity by a factor e^0.8 ≈ 2.2; the meaningful quantity is
  `exp(main_loss)`. This applies to every run in this repository, not only the
  stride-1 one.

**Validation and test losses are computed on the minor scaffolds reserved to
those splits**, largely repetitive and easier to predict than the training
corpus. `test/loss` is therefore not an absolute yardstick across splits, though
it remains valid across runs.

### The S = 32 exclusion

With 1000 bp windows the compressed grid falls to M = 32, below FlexAttention's
floor of M ≥ 64 (JanusDNA's `final_attention` sets `q_len = 2M` against a default
`BLOCK_SIZE` of 128). Two routes around it were tested, and both were subjected
to a neutrality gate rather than assumed:

- **Lowering `BLOCK_SIZE`** (`stem/gates/gate_flexblock_neutrality.sh`):
  **rejected**. Minimum per-row cosine 0.937, 21% relative difference. It is not
  a re-tiling of the same computation in this build.
- **Padding to 64·S** (`stem/gates/test_impatto_padding.sh`): padding the
  window from 1000 to 1024 bp (2.4%, what S = 16 needs) leaves the NMI vs GC
  unchanged to the fourth decimal, so it is **neutral**; padding to 2048 bp (51%,
  what S = 32 needs) moves it by 5.1%, comparable to the effect under study, so it
  is **not neutral**.

So S = 16 is included with validated padding and S = 32 is excluded. Its
pre-training is valid and its losses are reported, but no downstream number from
it would be comparable with the rest.

### Applying the patch

Two versions coexist, because the sweep results above were produced by the first
and should stay reproducible.

`stem/janusdna_stem_v1.patch` applies to JanusDNA at commit `3f44954`
(2025-08-29) and reproduces the S ∈ {4, 8, 16} runs. It touches four files:
`train.py`, `janusdna/modeling_janusdna.py`, and two new files
(`janusdna/causal_conv_stem.py`, `test_stem_wiring.py`).

`stem/run_stem_experiment.sh` is the v2 patcher, used for anything new. It
applies six patch-or-die modifications to `modeling_janusdna.py` and two to
`train.py`, and takes three parameters instead of one:

```
STEM_STRIDE   grid compression      (1 = none)
STEM_BLOCK    bases predicted per fused position
STEM_NSTAGES  number of stem stages (stride=1 only; default 2)
STEM_KERNEL   convolution kernel    (default 7)
```

`STEM_STRIDE = STEM_BLOCK` reproduces v1 exactly, and this is checked rather than
asserted: `stem/verify_patches.py` confirms that with `BAND = 1` every
parametrised expression evaluates to the original constant.

```bash
# reproduce the v1 S = 4 run
JANUS_STEM_GATE=1 STEM_STRIDE=4 STEM_BLOCK=4 ./stem/run_stem_experiment.sh

# counter-proof: MUST fail with a leakage assertion
JANUS_STEM_GATE=1 STEM_STRIDE=1 STEM_BLOCK=4 \
  JANUS_STEM_BAND_OVERRIDE=1 ./stem/run_stem_experiment.sh

# the step-1 run
JANUS_STEM_GATE=1 STEM_STRIDE=1 STEM_BLOCK=4 STEM_NSTAGES=3 \
  ./stem/run_stem_experiment.sh full
```

The downstream scripts in `src/` are stem-aware without a flag: they detect the
resolution mismatch, compress the attention mask accordingly, and pad the input
to `max(64·S, ceil(L/S)·S)`. With no stem the code path is unchanged.

### Verification

| script | needs | what it establishes |
|---|---|---|
| `stem/verify_theory.py` | CPU | 44 checks: receptive field measured by Jacobian rather than taken from the formula, causality, flip anti-causality, parameter counts, realignment identity at `STRIDE = BLOCK`, tail truncation, prediction/base ratio, horizon distribution, band arithmetic |
| `stem/verify_patches.py` | none | the eight patches are present, no diff hunk falls outside the intended regions, and `BAND = 1` reproduces the upstream constants |
| `stem/verify_leakage_map.py` | GPU | the dependency map above, with and without the correct band (`stem/run_leakage_check.sh` runs both) |
| `stem/causal_conv_stem.py` | CPU | standalone stem gate, including the configuration read from the environment |
| `stem/test_stem_wiring.py` | GPU | full-model self-exclusion, wired into `on_fit_start` so it runs before Lightning's sanity check |
| `tools/check_stem_config.py` | none | derives stage count, kernel and block from a checkpoint and refuses to proceed if the environment disagrees |

`check_stem_config.py` exists because two mismatches are silent under
`load_state_dict(strict=False)`: a wrong stage count leaves only four unexpected
keys, below any reasonable count threshold, and a wrong `BLOCK` changes `BAND`
without touching a parameter at all.

## Repository layout

```
env/            environment specs, container bootstrap, runtime patches
src/            the downstream pipeline (data prep → embeddings → probes → LoRA)
pretrain/       from-scratch pre-training on the penguin genome
stem/           the causal convolutional stem experiment
analysis/       figures, aggregation, per-class breakdowns, GC analyses
tools/          smoke tests and verification utilities
docs/           file map (old thesis-era names → current names), experiment notes
```

Everything is driven by a single environment variable:

```bash
export PENGUIN_ROOT=/path/to/your/workspace     # defaults to $HOME/penguin_glm
export JANUSDNA_REPO=/path/to/JanusDNA          # only for JanusDNA steps
export CROSSDNA_REPO=/path/to/CrossDNA          # only for CrossDNA steps
```

No absolute paths are hard-coded. The original scripts were written against a
specific server layout; `docs/FILE_MAP.md` records the correspondence for anyone
cross-referencing the thesis.

---

## Environments

CrossDNA and JanusDNA have **mutually incompatible dependency stacks**:
JanusDNA pins `torch 2.5` through the `mamba_ssm` wheel, CrossDNA needs
`flash-attn 2` plus Triton for the Comba kernels. They are therefore run in
separate containers, and the pipeline is decoupled through two neutral
interchange formats: `windows.parquet` (labels + covariates) and `.npz`
embedding files. `src/common.py` imports cleanly in all three environments
because every torch import inside it is lazy.

### CrossDNA

```bash
conda env create -f env/crossdna.yml
conda activate crossdna
```

Installation order matters; see [Known issues](#known-issues).

### JanusDNA

```bash
conda create -n janusdna python=3.11    # exactly 3.11
conda activate janusdna
pip install mamba-ssm==2.2.4
```

Then install the remaining dependencies from the upstream JanusDNA repository's
own instructions. The Python version is not negotiable: the `mamba_ssm` 2.2.4
wheel is built for cp311 and torch 2.5.

### Analysis only

Clustering, probing and figures need no GPU stack:

```bash
pip install numpy pandas polars scikit-learn matplotlib umap-learn pyarrow
```

---

## Quickstart

### 1. Get the genome

```bash
datasets download genome accession GCF_965638725.1 --include genome,gff3
```

The soft-masked FASTA is required: repeat fraction is computed from lowercase
characters and must be read *before* uppercasing.

### 2. Build the windows

```bash
python src/prepare_windows.py \
  --fasta  $PENGUIN_ROOT/data/GCF_965638725.1_genomic.fna \
  --gff    $PENGUIN_ROOT/data/genomic.gff \
  --outdir $PENGUIN_ROOT/out \
  --window 1000 --per-class-train 20000 --per-class-eval 5000
```

Produces `windows.parquet` (120,000 rows: sequence, label, split, scaffold, GC,
repeat fraction, N fraction) and a `summary.json` with per-class composition.

### 3. Extract frozen embeddings

```bash
# CrossDNA: note --sdp default; the math implementation produces non-finite values
python src/extract_crossdna.py \
  --weights  $PENGUIN_ROOT/weights/8.1M \
  --windows  $PENGUIN_ROOT/out/windows.parquet \
  --out      $PENGUIN_ROOT/out/emb_crossdna_81.npz \
  --pooling mean --sdp default

# JanusDNA: stem-aware automatically, no flag needed
python src/extract_janusdna.py \
  --repo-path       $JANUSDNA_REPO \
  --model-config    $JANUSDNA_REPO/imported_checkpoint/144_without_midattn_model_config.json \
  --checkpoint      $PENGUIN_ROOT/checkpoints/janus144.ckpt \
  --windows-parquet $PENGUIN_ROOT/out/windows.parquet \
  --output          $PENGUIN_ROOT/out/emb_janusdna_144.npz
```

The larger checkpoints need help. Two mechanisms are available and they are not
equivalent:

- `--amp-fp32 --sdpa-patch` forces the attention into fp32. Use this where it
  suffices: it changes precision, not values.
- `--clamp` zeroes non-finite values and COUNTS them. It is a containment
  measure, not a fix: the count is written into the `.npz` metadata
  (`clamp_fraction`) and must be reported alongside any result derived from
  that file.

**The CrossDNA-519M embeddings behind the results were extracted with `--sdp
default --clamp`, and the clamp never intervened**: its counter stayed at zero
over about 9.46 × 10¹² intermediate activations checked during the full
extraction, so no value was zeroed. `tools/inspect_embeddings.py` separates
non-finite values from clamped ones in an existing archive.

Sanity-check the result before spending GPU time downstream:

```bash
python tools/check_embeddings.py --npz $PENGUIN_ROOT/out/emb_janusdna_144.npz --expected-dim 144
```

### 4. Cluster, probe, classify

```bash
# cluster_embeddings.py and classify_baselines.py take --emb as name=path and
# accept the flag repeatedly, so several models go into one comparison.
python src/cluster_embeddings.py \
  --windows $PENGUIN_ROOT/out/windows.parquet \
  --emb janus144=$PENGUIN_ROOT/out/emb_janusdna_144.npz \
  --emb crossdna81=$PENGUIN_ROOT/out/emb_crossdna_81.npz \
  --outdir $PENGUIN_ROOT/out/cluster --umap

python src/classify_baselines.py \
  --windows $PENGUIN_ROOT/out/windows.parquet \
  --emb janus144=$PENGUIN_ROOT/out/emb_janusdna_144.npz \
  --outdir $PENGUIN_ROOT/out/clf --do-kmer --do-trivial

# probe configuration 1 (the 4-seed zero-shot table): name=path, repeatable
for s in 1 2 3 4; do
  python src/mlp_probe_compare.py \
    --windows $PENGUIN_ROOT/out/windows.parquet \
    --emb janus144=$PENGUIN_ROOT/out/emb_janusdna_144.npz \
    --seed $s --outdir $PENGUIN_ROOT/out/mlp_probe_seed$s
done

# probe configuration 2 (the stem and step-1 comparison): one bare path per run
python src/mlp_probe.py \
  --windows $PENGUIN_ROOT/out/windows.parquet \
  --emb $PENGUIN_ROOT/out/emb_janusdna_144.npz \
  --outdir $PENGUIN_ROOT/out/probe

# the 4-mer baseline through the same probe (configuration 1 by default)
python src/kmer_mlp_baseline.py
```

### 5. LoRA fine-tuning

```bash
python src/finetune_lora_janusdna.py \
  --repo-path    $JANUSDNA_REPO \
  --model-config .../144_without_midattn_model_config.json \
  --checkpoint   .../janus144.ckpt \
  --windows      .../windows.parquet \
  --outdir       .../lora_janus144 \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
```

LoRA configuration: r = 8, α = 16, dropout = 0.05, backbone frozen, `lr_lora`
1e-4 and `lr_head` 1e-3, early stopping on validation MCC. The default budget is
10 epochs, which is not enough for JanusDNA to saturate:
`bash src/run_lora_saturated.sh` re-runs the baseline, the human checkpoint and
the stems with a 25-epoch budget and reports, for each run, whether the
validation maximum fell before the end. `tools/lora_summary.sh` tabulates every
LoRA result and flags runs whose hyperparameters differ.

For CrossDNA the adapters target q/k/v/o of both the Comba and the SWA mixer in
branches A and B: 192 projections and 3,149,828 trainable parameters out of
522,172,165 for the CrossDNA-519M, 96 projections and 344,580 parameters for the
CrossDNA-8.1M. Name matching also reaches the teacher EMA copy of the model,
which is never executed during adaptation and contributes nothing, so the result
files count those adapters too (192 modules and 688,644 parameters for the 8.1M).
`src/finetune_lora_crossdna.py` saves both counts, and `tools/count_lora_params.py`
derives them from the checkpoint headers without a GPU. For JanusDNA (27
projections, 102,276 trainable parameters, 0.286%), `dt_proj` is **excluded**: it
is structurally incompatible with LoRA, and `tools/verify_lora_targets.py`
demonstrates empirically which projections an adapter actually reaches.

---

## Pre-training from scratch

```bash
# --windows is required: it is what tells the script which scaffolds to exclude
python pretrain/make_pretrain_bed_clean.py \
  --fasta   $PENGUIN_ROOT/data/penguin/GCF_965638725.1.fna \
  --windows $PENGUIN_ROOT/out/windows.parquet \
  --out     $PENGUIN_ROOT/data/penguin/penguin_pretrain_clean.bed \
  --seqlen  131072

bash  pretrain/pretrain_janusdna.sh
bash  pretrain/pretrain_crossdna.sh

python pretrain/crossdna_ckpt_to_hf.py \
  --ckpt    .../checkpoints/last.ckpt \
  --ref-dir $CROSSDNA_REPO/crossdna_inference \
  --out-dir $PENGUIN_ROOT/crossdna/hf_scratch_penguin_clean
```

**The tiling step is not free.** JanusDNA's `HG38Dataset` ignores the BED file's
`end` column and recomputes it as `start + MAX_ALLOWED_LENGTH`, where
`MAX_ALLOWED_LENGTH = 2^20 = 1048576`. `--seqlen` is the training sequence
length and must DIVIDE 2^20; `--tile-step` is the BED interval and defaults to
2^20, which is what JanusDNA needs. For CrossDNA and HyenaDNA, set
`--tile-step` equal to `--seqlen` so one BED row is one training example.
Getting this wrong silently produces windows of the wrong length.

CrossDNA checkpoints must be converted to HuggingFace layout before extraction
or LoRA; `crossdna_ckpt_to_hf.py` produces a directory structurally identical to
the released weights.

### Corpus construction

The pre-training corpus is built from the downstream TRAIN scaffolds only.
`make_pretrain_bed_clean.py` reads `windows.parquet`, collects every scaffold
used by the downstream val and test splits, and excludes those scaffolds
entirely before tiling.

This is not optional. Pre-training never sees the labels, but it does model the
sequence, and a model that has already fitted the nucleotide statistics of a
test scaffold will score better there for reasons that have nothing to do with
domain adaptation. Any comparison between an in-domain and a transferred model
is only meaningful if the in-domain corpus and the evaluation set are disjoint
at the scaffold level, the same guarantee the downstream split itself provides.

On this assembly the exclusion removes a substantial fraction of the genome,
which is why the from-scratch budgets are as small as they are.

---

## Known issues

These are environment-specific traps, each of which cost real debugging time.

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError: Autotuner.__init__() got an unexpected keyword argument 'do_bench'` | `flash-attn` silently downgrades Triton to 3.1.0 | reinstall `triton==3.2.0` **after** flash-attn, with `--no-deps --force-reinstall` |
| Crash on `import fla` | `fla/ops/attn/parallel.py` applies `@torch.compile` at import time; `DISABLE_TORCH_COMPILE=1` is not enough | monkey-patch `torch.compile` to identity *before* any fla/transformers import (see `env/patches/`) |
| `ModuleNotFoundError: fla` despite install | `fla` is a namespace package split across two PyPI distributions | install both `flash-linear-attention==0.4.0` and `fla-core==0.4.0` |
| NumPy ABI errors | NumPy 2 breaks the pinned torch build | pin `numpy<2` |
| `repetition level histogram size mismatch` reading `windows.parquet` | pyarrow version mismatch | read scalar columns with `polars` instead |
| `row[0]` returns the wrong thing | pandas ≥ 2 changed positional indexing | use `row.iloc[0]` |
| Non-finite values in CrossDNA embeddings | overflow with the `math` SDP implementation | pass `--sdp default`; the fp32 attention replacement lives in `env/patches/sdpa_fp32.py` |
| CrossDNA-519M dtype mismatch `bf16 vs fp32` | autocast bf16 is incompatible with the `fla` kernels | extract in fp32, no autocast |
| CrossDNA returns non-finite values in 16-bit autocast | numerical fragility of the model in bf16: the from-scratch 8.1M checkpoint fails already at the first forward pass. `tools/certify_fp32_patch.py` reproduces the failure as its control run. | run in fp32 (`--amp-fp32 --sdpa-patch`), which is what this pipeline does throughout |
| `device-side assert` in the Triton kernel for S=16/32 | the compressed grid falls below FlexAttention's floor: `q_len = 2M` against `BLOCK_SIZE = 128` needs M ≥ 64 | pad the input to 64·S, and validate the padding on the METRICS (`stem/gates/`) rather than assuming it |
| `AssertionError: T=1000 not divisible by S` | 1000 is not a multiple of 16 or 32 | right-padding inside `encode()`, marked invalid in the mask. It must live in `encode()`, not in the forward: the probe that determines the hidden width calls `encode()` too, and padding downstream leaves that first call unpadded |
| A shell script continues after a Python crash | `set -e` sees the exit code of `tee`, not of the command feeding it | add `set -o pipefail` |
| `RuntimeError: size of tensor a (250) must match tensor b (1000)` | pooling a super-token-resolution hidden state against a nucleotide-resolution mask | compress the mask: a super-token is valid if its block holds at least one real position |
| A stem checkpoint silently loads into a model built without the stem | `load_state_dict(strict=False)` ignores the `stem.*` weights without error | gate on `missing`/`unexpected` before every extraction and fine-tune |
| Two scales running in parallel destroy each other's patched repo | a shared mount and a `rm -rf` at startup | one patched copy per scale, `JanusDNA_stem_s<N>` |
| `rsync: command not found` in the container | minimal image | fall back to `cp -a` plus pruning of `.git`/`wandb`/`*.ckpt` |
| The exclusion band silently covers only part of the predicted block | `BAND` is hardcoded to 1 **in grid cells**; it matched the block only because one cell was worth one block | parametrise `BAND = BLOCK / STRIDE` in both `create_janus_mask` and `final_fusion_repos_formasked`; patching the mask alone has no effect |
| A wrong stem stage count or a wrong `BLOCK` passes the weight-loading gate | a missing stage leaves only four unexpected keys, below the threshold; `BAND` is not a parameter at all, so a wrong `BLOCK` produces no mismatch | derive the configuration from the checkpoint with `tools/check_stem_config.py` before rebuilding the model |
| Logged perplexity looks implausibly high | the harness logs `exp(main_loss + moe_loss)`, including the MoE router's auxiliary term | report `exp(main_loss)`; with `moe_loss` ≈ 0.8 the logged value is inflated by e^0.8 ≈ 2.2 |
| Held-out loss degrades over the last epochs of every 50k-step run | genuine overfitting of the stem runs, which end past their optimum; the baseline at 15k steps ends near it | use `checkpoints/test/loss.ckpt` (the test-loss minimum, which *is* saved) when the training point matters; for the NMI against GC it does not (1.2%) |
| LoRA has not converged at 10 epochs | the validation maximum falls on the last epochs (nine more epochs gave the human JanusDNA +0.018 MCC) | run `src/run_lora_saturated.sh` before reading small LoRA margins |
| Two LoRA results look comparable but are not | one of them was resumed from an earlier run and trained for more epochs | compare runs with the same budget; `tools/lora_summary.sh` shows the epoch of the maximum and the epochs run |
| The full-model gate never runs before a shape crash | `on_train_start` fires *after* Lightning's sanity check | wire the gate into `on_fit_start` instead |
| Lightning resumes an already-finished run | a stale `last.ckpt` in the output directory | wipe the smoke-test directory before the real run |
| `fsspec_exists()` crashes on `train.ckpt=null` | Hydra passes the literal `null` | pass a non-existent path string instead |

`env/container_setup.sh` applies the recurring ones in the right order; source it
on entering a fresh container.

---

## Limitations

- **Single species, single annotation.** All conclusions are about one avian
  genome with one RefSeq annotation. The GC-attractor result is consistent with
  what has been reported for other gLMs, but this work does not establish it
  generally.
- **The label scheme is a modelling choice.** Centre-based labelling and
  UTR-as-exon-minus-CDS are defensible but not unique; both are isolated in a
  single function in `src/prepare_windows.py` so they can be replaced without
  touching anything else.
- **The test set is concentrated.** 13,715 of the 20,000 test windows come from
  one chromosome (`NC_134951.1`). Bootstrap intervals on the test MCC (about
  ±0.009) reflect window sampling only; variability between chromosomes is not
  estimated.
- **One seed per pre-training run.** The run-to-run variability of pre-training
  is not measured; seeds vary only the probes.
- **Compute budgets are not uniform.** The JanusDNA baseline runs 15,000 steps,
  the stem runs 50,000; LoRA was run to saturation for JanusDNA, not for
  CrossDNA. Step 1 against `S = 4` is the only fully budget-matched comparison
  of the stem experiment.
- **Window and objective remain confounded** in the stem experiment; the fourth
  cell that would separate them was not run.
- **Pre-training budgets are small.** 15K steps (JanusDNA) and 60K steps
  (CrossDNA) on a single H100 are far from the compute used for the released
  human checkpoints, so "from scratch" here means "from scratch on a student
  budget".

---

## Citing

If you use this code, please cite this repository; machine-readable metadata is
in `CITATION.cff`. The reference to the thesis (Master's thesis, Università
Politecnica delle Marche, 2026) will be added here once its title is final.

<!-- TODO after the degree: add the thesis title and supervisor here and in
     CITATION.cff (references section). -->

## Acknowledgements and upstream code

This work builds on two external codebases, neither of which is redistributed
here:

- **JanusDNA**: the conv-stem experiment is distributed as a patch
  (`stem/janusdna_stem_v1.patch`, plus the v2 patcher in
  `stem/run_stem_experiment.sh`) to be applied to an upstream checkout at commit
  `3f44954`, not as a fork. JanusDNA itself derives from Caduceus.
- **CrossDNA**: used unmodified; only the runtime patches in `env/patches/`
  are ours.

Please respect the upstream licences when applying the patch.

## Licence

MIT, see `LICENSE`. The licence covers the code in this repository only.
