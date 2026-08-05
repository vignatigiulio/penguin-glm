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
| **CrossDNA** | Comba + sliding-window attention, dual-strand (branches A/B) | 8.1M, 28.6M, 71.6M, 519M |
| **JanusDNA** | Mamba + Mixture-of-Experts, bidirectional weight tying | 72d, 144d |

---

## The main result

**Zero-shot gLM embeddings organise by GC content, not by biological function.**
Clustering the frozen embeddings of 120,000 genomic windows recovers nucleotide
composition an order of magnitude more strongly than it recovers the functional
annotation:

| Embedding | dim | NMI vs. function | NMI vs. GC | ratio |
|---|---:|---:|---:|---:|
| CrossDNA-8.1M (human) | 128 | 0.018 | 0.509 | 27.7× |
| CrossDNA-28.6M (human) | 256 | 0.013 | 0.379 | 29.7× |
| CrossDNA-519M (human) | 1024 | 0.018 | 0.518 | 29.5× |
| JanusDNA-72 (human) | 72 | 0.020 | 0.352 | 17.9× |
| JanusDNA-144 (human) | 144 | 0.019 | 0.182 | 9.5× |

The functional signal is nonetheless *present* — it is simply not the dominant
axis of the latent geometry. A supervised probe extracts it, and LoRA
domain adaptation extracts substantially more. The dissociation between
unsupervised geometry and extractable information is the central claim of the
thesis, and the window-length sweep below is its cleanest demonstration.

---

## Task and data

**Task.** 4-way classification of functional elements — CDS / UTR / intron /
intergenic — derived from the RefSeq GFF annotation.

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

### Baselines

| Baseline | Classifier | MCC |
|---|---|---:|
| Trivial (GC + repeat + N) | logistic regression | 0.124 |
| Trivial (GC + repeat + N) | random forest | 0.081 |
| 4-mer composition | logistic regression | 0.223 |
| 4-mer composition | random forest | **0.243** |

The 4-mer random forest is the bar every embedding has to clear: it is pure
composition with no learned representation at all.

### Zero-shot (frozen embeddings + MLP probe, 4 seeds)

| Model | Pre-training corpus | MCC |
|---|---|---:|
| JanusDNA-144 | penguin, from scratch (clean) | 0.3218 ± 0.0016 |
| JanusDNA-144 | human | 0.3088 ± 0.0043 |
| CrossDNA-8.1M | penguin, from scratch (clean) | 0.2881 ± 0.0005 |
| CrossDNA-8.1M | human | 0.2691 ± 0.0015 |

All five rows come from the same probe protocol (2-layer MLP on frozen
embeddings, hidden 256, dropout 0.3, four seeds). Earlier single-seed probe runs
on the same human embeddings gave 0.272 for CrossDNA-8.1M and 0.289 for
CrossDNA-519M; those numbers are not comparable with the table above and are not
used here. `analysis/audit_results.py` flags any feature that has been evaluated
under more than one protocol.

In-domain pre-training helps, but modestly: **+0.013 MCC** for JanusDNA and
**+0.019** for CrossDNA — a small return for a full pre-training run on the
target genome.

### After LoRA domain adaptation

| Model | Pre-training corpus | MCC |
|---|---|---:|
| JanusDNA-144 | penguin, from scratch (clean) | 0.4028 |
| JanusDNA-144 | human | 0.4027 |
| CrossDNA-8.1M | penguin, from scratch (clean) | 0.4216 |
| CrossDNA-8.1M | human (fp32) | 0.3856 |
| CrossDNA-519M | human | 0.4474 |

Note the asymmetry. For JanusDNA the in-domain advantage is **reabsorbed** after
adaptation (0.4028 vs. 0.4027 — identical). For CrossDNA it is **amplified**
(+0.019 zero-shot → +0.036 after LoRA). This divergence is reported as an open
question rather than explained; the most likely candidates are model scale and
the quality of the human checkpoint.

### Window-length sweep

The same protocol run at eight window lengths, on JanusDNA-144 pre-trained
from scratch on the clean penguin corpus:

| L (bp) | zero-shot MCC | LoRA MCC | LoRA gain | NMI vs. function | NMI vs. GC |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.2155 | 0.2146 | **−0.0009** | 0.055 | 0.404 |
| 100 | 0.2334 | 0.2492 | +0.0158 | 0.060 | 0.404 |
| 150 | 0.2617 | 0.2827 | +0.0210 | 0.062 | 0.410 |
| 200 | 0.2774 | 0.2979 | +0.0205 | 0.059 | 0.413 |
| 300 | 0.2946 | 0.3239 | +0.0293 | 0.056 | 0.419 |
| 500 | 0.3078 | 0.3604 | +0.0526 | 0.051 | 0.415 |
| 1000 | 0.3192 | 0.4048 | +0.0856 | 0.046 | 0.403 |
| 2000 | 0.3411 | 0.4424 | **+0.1013** | 0.045 | 0.400 |

Three things are worth reading off this table.

1. **At L = 64, LoRA adds nothing at all** (−0.0009). At L = 2000 it adds
   +0.1013. This is a regime transition, not a scaling curve: below a certain
   context length there is simply no long-range structure for adaptation to
   exploit, and the model falls back on composition — which the frozen
   representation already encodes.
2. **NMI against GC is flat** (0.400–0.419) across the entire sweep, while
   supervised MCC nearly doubles. The latent geometry does not reorganise; the
   information becomes *extractable* without becoming *dominant*.
3. Zero-shot MCC at L = 64 (0.2155) is below the 4-mer baseline at L = 1000
   (0.243), which is the expected sanity check.

---

## The causal convolutional stem experiment

**Question.** Is the GC attractor a consequence of single-nucleotide
tokenisation? If the model reasons over super-tokens instead of single bases,
does the latent space stop being organised by composition?

**Design.** A causal down-sampling stem — stride-2 stages of kernel 7 — inserted
before the JanusDNA Mamba+MoE backbone, giving compression S with a receptive
field of roughly 5S bp. Three architectural invariants were verified before any
run: the backward stream uses flip-based weight tying (one stem, not two), the
Janus loss is non-shifted, and LayerNorm is per-position and therefore causally
safe. A standalone gate proves causality through the Jacobian, and a full-model
gate (`stem/test_stem_wiring.py`) confirms the stem preserves Janus
self-exclusion.

**Falsifiable prediction, stated in advance.** An improvement in the
NMI(function)/NMI(GC) *ratio* would indicate a mechanistic gain; an improvement
in MCC alone would indicate a merely computational one.

**Budget.** Identical for every scale: ~52B tokens, `max_length=131072`,
50,000 steps, `global_batch=8`, `lr=8e-3`, bf16-mixed, one H100.

### Sweep over the compression scale

| Scale | NMI vs GC | NMI vs function | ratio f/GC | silhouette | zero-shot MCC | LoRA MCC | effective dim |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (no stem) | 0.4038 | 0.0462 | 0.114 | 0.199 | 0.3232 | **0.403** | 3 |
| **S = 4** (~19 bp) | 0.2591 | **0.0581** | **0.224** | **0.462** | **0.3238** | 0.383 | 2 |
| S = 8 (~40 bp) | 0.2886 | 0.0526 | 0.182 | 0.387 | 0.2920 | 0.360 | 1 |
| S = 16 (~80 bp) | **0.2187** | 0.0401 | 0.183 | 0.437 | 0.2594 | 0.331 | 1 |

S = 32 was pre-trained but is excluded from the downstream comparison; see
[The S = 32 exclusion](#the-s--32-exclusion).

**The scales are not equivalent, and S = 4 is an optimum.** Three readings:

1. **Extractability degrades monotonically with compression**, in both regimes:
   zero-shot 0.324 → 0.292 → 0.259, post-LoRA 0.383 → 0.360 → 0.331. Only S = 4
   matches the baseline zero-shot.
2. **The gap from the baseline widens with S even after adaptation.** At S = 4
   fine-tuning reabsorbs the difference (0.383 against 0.403, within variance);
   at S = 8 and S = 16 it no longer does. The distinction is substantive: at
   S = 4 the information is present but reorganised, beyond it the information
   is lost in the compression and in-domain adaptation does not recover it.
3. **Reducing GC is not the same as improving.** S = 16 weakens the GC attractor
   more than any other scale (0.2187, −46% from baseline) and loses on
   everything else, including the lowest NMI vs function of the whole sweep. It
   is flattening the representation, not purifying it.

The ratio declared in advance as the discriminating metric picks out S = 4
(0.224) clearly over S = 8 and S = 16 (both ~0.18).

### Why S = 16 wins on GC and loses on everything else

| Scale | var. PC1 | eff. dim (80%) | PC2 corr. GC | mean corr. GC | LDA η² | LDA acc. |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.659 | 3 | 0.880 | 0.399 | 0.4035 | 0.465 |
| **S = 4** | 0.672 | 2 | **0.739** | 0.355 | **0.4064** | 0.462 |
| S = 8 | 0.807 | **1** | 0.908 | 0.286 | 0.3654 | 0.445 |
| S = 16 | 0.809 | **1** | 0.928 | 0.240 | 0.3039 | 0.420 |

Two facts contradict any extrapolation from NMI alone. **PC2's correlation with
GC rises** for S = 8 and S = 16 (0.908, 0.928 against the baseline's 0.880): the
dominant axis becomes *more* GC-centric, not less. And **the representation
collapses**: PC1 alone explains 81% of the variance and the effective
dimensionality falls to one.

The unifying explanation is that at high compression GC becomes less *diffuse*
(mean correlation 0.286 and 0.240) but more *concentrated* in the dominant axis.
It is not attenuated, it is compressed into a single direction along with
everything else. The falling NMI vs GC at S = 16 is therefore a symptom of an
impoverished representation, not of a purified one.

This also resolves the otherwise odd combination of high silhouette with falling
NMI against every measured variable: if the representation flattens onto one
axis, the points line up and k-means separates them by construction, with no
biological correspondence. Better-looking clusters, less information.

**S = 4 is the only scale that reorganises without collapsing**: it keeps
dimensionality, reduces the GC axis, and preserves the baseline's linear
extractability (LDA η² 0.4064 against 0.4035).

### Verdict against the prediction

At S = 4, **mechanistic gain without computational gain**. The stem measurably
weakens the GC attractor — the function/GC ratio nearly doubles, the clusters
separate far better — and none of it converts into downstream accuracy, while
under LoRA it costs about 0.02 MCC. That is a negative result on the metric that
matters practically, and a positive one on the metric the experiment was
designed to test. Beyond S = 4 both gains disappear together.

**Consequence for the multi-scale idea.** The premise of an Inception-style
multi-scale stem holds — the scales do behave differently — but the data show no
*complementarity*: no scale captures function better than the others, and all
degrade as S grows. A module fusing S ∈ {4, 8, 16} would combine one good
extractor with two that degrade extractability. The remaining argument would be
that learned weights could select the best scale, and in that case the sweep has
already given the answer at far lower cost.

### The S = 32 exclusion

With 1000 bp windows the compressed grid falls to M = 32, below FlexAttention's
floor of M ≥ 64 (JanusDNA's `final_attention` sets `q_len = 2M` against a default
`BLOCK_SIZE` of 128). Two routes around it were tested, and both were subjected
to a neutrality gate rather than assumed:

- **Lowering `BLOCK_SIZE`** (`stem/gates/gate_flexblock_neutrality.sh`) —
  **rejected**. Minimum per-row cosine 0.937, 21% relative difference. It is not
  a re-tiling of the same computation in this build.
- **Padding to 64·S** (`stem/gates/test_impatto_padding.sh`) — **neutral at
  2.4%** (S = 16 → 1024: NMI vs GC unchanged to four decimals), **not neutral at
  51%** (S = 32 → 2048: NMI vs GC moves 5.1%, comparable to the effect under
  study).

So S = 16 is included with validated padding and S = 32 is excluded. Its
pre-training is valid and its losses are reported, but no downstream number from
it would be comparable with the rest.

Note the division of labour between the two padding gates. The vector-level gate
gives 0.9956 at 2.4% — below its own 0.9999 threshold, so it does not clear
S = 16 on its own. What cleared S = 16 is the metric-level test: the vectors move
slightly, the conclusions do not. Testing the quantity the argument rests on,
rather than a proxy for it, is what made the difference.

### Applying the patch

`stem/janusdna_stem.patch` applies to JanusDNA at commit `3f44954` (2025-08-29).
It touches four files: `train.py`, `janusdna/modeling_janusdna.py`, and two new
files (`janusdna/causal_conv_stem.py`, `test_stem_wiring.py`).

The downstream scripts in `src/` are stem-aware without a flag: they detect the
resolution mismatch, compress the attention mask accordingly, and pad the input
to `max(64·S, ceil(L/S)·S)`. With no stem the code path is unchanged.

## Repository layout

```
env/            environment specs, container bootstrap, runtime patches
src/            the downstream pipeline (data prep → embeddings → probes → LoRA)
pretrain/       from-scratch pre-training on the penguin genome
stem/           the causal convolutional stem experiment
analysis/       figures, aggregation, per-class breakdowns
tools/          smoke tests and verification utilities
docs/           file map (old thesis-era names → current names)
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

CrossDNA and JanusDNA have **mutually incompatible dependency stacks** —
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

Installation order matters — see [Known issues](#known-issues).

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
# CrossDNA — note --sdp default; the math implementation yields ~15% non-finites
python src/extract_crossdna.py \
  --weights  $PENGUIN_ROOT/weights/8.1M \
  --windows  $PENGUIN_ROOT/out/windows.parquet \
  --out      $PENGUIN_ROOT/out/emb_crossdna_81.npz \
  --pooling mean --sdp default

# JanusDNA — stem-aware automatically, no flag needed
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
  suffices — it changes precision, not values.
- `--clamp` zeroes non-finite values and COUNTS them. It is a containment
  measure, not a fix: the count is written into the `.npz` metadata
  (`clamp_fraction`) and must be reported alongside any result derived from
  that file.

**The 519M results in this repository were produced with `--sdp default
--clamp`, in bf16.** On a short check the same checkpoint clamps on the order of
2-3% of the values it inspects, so a non-trivial fraction of the intermediate
activations behind those embeddings were zeroed rather than computed. The
numbers are reported as measured; the caveat travels with them.

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

# mlp_probe.py takes a single bare path instead — one model per run.
python src/mlp_probe.py \
  --windows $PENGUIN_ROOT/out/windows.parquet \
  --emb $PENGUIN_ROOT/out/emb_janusdna_144.npz \
  --outdir $PENGUIN_ROOT/out/probe
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
1e-4 and `lr_head` 1e-3. For CrossDNA the adapters target q/k/v/o of both the
Comba and the SWA mixer in branches A and B — 384 projections, 0.609% of
parameters. For JanusDNA, `dt_proj` is **excluded**: it is structurally
incompatible with LoRA, and `tools/verify_lora_targets.py` demonstrates
empirically which projections an adapter actually reaches.

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
at the scaffold level — the same guarantee the downstream split itself provides.

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
| ~15% non-finite values in CrossDNA embeddings | fp32 overflow with the `math` SDP implementation | pass `--sdp default`; the fp32 attention replacement lives in `env/patches/sdpa_fp32.py` |
| CrossDNA-519M dtype mismatch `bf16 vs fp32` | autocast bf16 is incompatible with the `fla` kernels | extract in fp32, no autocast |
| CrossDNA-519M zeroes 2-3% of inspected values under `--clamp`, even with fp32 attention | residual instability of the checkpoint | report `clamp_fraction` from the `.npz` metadata with any 519M result |
| CrossDNA returns NaN in bf16 for some batch sizes, with the ORIGINAL flash-attn and no patches | numerical fragility of the upstream model in bf16: on the 8.1M checkpoint, splitting a batch of 64 into two of 32 is enough to produce non-finite values upstream of the attention. `tools/certify_fp32_patch.py` reproduces it as its control run. | run in fp32 (`--amp-fp32 --sdpa-patch`), which is what this pipeline does throughout |
| `device-side assert` in the Triton kernel for S=16/32 | the compressed grid falls below FlexAttention's floor: `q_len = 2M` against `BLOCK_SIZE = 128` needs M ≥ 64 | pad the input to 64·S, and validate the padding on the METRICS (`stem/gates/`) rather than assuming it |
| `AssertionError: T=1000 not divisible by S` | 1000 is not a multiple of 16 or 32 | right-padding inside `encode()`, marked invalid in the mask. It must live in `encode()`, not in the forward: the probe that determines the hidden width calls `encode()` too, and padding downstream leaves that first call unpadded |
| A shell script continues after a Python crash | `set -e` sees the exit code of `tee`, not of the command feeding it | add `set -o pipefail` |
| `RuntimeError: size of tensor a (250) must match tensor b (1000)` | pooling a super-token-resolution hidden state against a nucleotide-resolution mask | compress the mask: a super-token is valid if its block holds at least one real position |
| A stem checkpoint silently loads into a model built without the stem | `load_state_dict(strict=False)` ignores the `stem.*` weights without error | gate on `missing`/`unexpected` before every extraction and fine-tune |
| Two scales running in parallel destroy each other's patched repo | a shared mount and a `rm -rf` at startup | one patched copy per scale, `JanusDNA_stem_s<N>` |
| `rsync: command not found` in the container | minimal image | fall back to `cp -a` plus pruning of `.git`/`wandb`/`*.ckpt` |
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
- **CrossDNA-28M is excluded from the headline comparisons.** It produced an
  anomalous MCC of 0.121 traced to numerical precision, not to a genuine
  capability gap.
- **CrossDNA-519M LoRA: two evaluation files, one correct value.** The training
  script reports 0.4474 and a separate re-evaluation from the saved checkpoint
  reports 0.4150. The gap is a train/eval precision mismatch, not model
  variance: training forces fp32 attention (`preferred_amp_dtype` override plus
  the SDPA replacement in `env/patches/sdpa_fp32.py`), while the standalone
  evaluation script omitted both and therefore ran the native bf16 autocast path
  — the very path the 519M is known to overflow on. **0.4474 is the value
  consistent with the training protocol**; `src/eval_lora.py` now installs the
  same patch, so the two agree.
- **Pre-training budgets are small.** 15K steps (JanusDNA) and 60K steps
  (CrossDNA) on a single H100 are far from the compute used for the released
  human checkpoints, so "from scratch" here means "from scratch on a student
  budget".

---

## Citing

If you use this code, please cite the thesis:

> G. Vignati. *[THESIS TITLE]*. Master's thesis, Università Politecnica delle
> Marche, 2026. Supervisor: [SUPERVISOR NAME].

<!-- TODO before publishing: replace [THESIS TITLE] and [SUPERVISOR NAME] here
     and in CITATION.cff. -->

Machine-readable metadata is in `CITATION.cff`.

## Acknowledgements and upstream code

This work builds on two external codebases, neither of which is redistributed
here:

- **JanusDNA** — the conv-stem experiment is distributed as a patch
  (`stem/janusdna_stem.patch`) to be applied to an upstream checkout at commit
  `3f44954`, not as a fork. JanusDNA itself derives from Caduceus.
- **CrossDNA** — used unmodified; only the runtime patches in `env/patches/`
  are ours.

Please respect the upstream licences when applying the patch.

## Licence

MIT — see `LICENSE`. The licence covers the code in this repository only.
