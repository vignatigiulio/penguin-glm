#!/bin/bash
# ============================================================================
# Conv-stem experiment on JanusDNA-144 — FULLY ISOLATED from the baseline.
#
# What it does, in order:
#   1. copies the JanusDNA repository to a NEW location (never touches the
#      baseline checkout)
#   2. applies stem/janusdna_stem.patch to the COPY
#   3. compiles the patched files, runs the standalone causality gate, and
#      verifies that the imported `janusdna` really is the patched copy
#   4. launches training in the copy, writing to a SEPARATE output directory
#
# Nothing belonging to the baseline is modified. That isolation is the point:
# the experiment has to be comparable against a baseline that has not moved.
#
# USAGE:
#   ./run_stem_experiment.sh              # smoke (seqlen 1024, 50 steps) — DEFAULT
#   ./run_stem_experiment.sh full         # real run, bp-matched to the baseline
#   STEM_STRIDE=8 ./run_stem_experiment.sh    # override S (default 4)
#
# Paths come from the environment:
#   PENGUIN_ROOT    workspace root      (default: $HOME/penguin_glm)
#   JANUSDNA_REPO   baseline checkout   (default: $PENGUIN_ROOT/../JanusDNA)
# ============================================================================
set -euo pipefail

: "${PENGUIN_ROOT:=$HOME/penguin_glm}"
: "${JANUSDNA_REPO:=$(dirname "$PENGUIN_ROOT")/JanusDNA}"

STEM_STRIDE="${STEM_STRIDE:-4}"
STEM_KERNEL="${STEM_KERNEL:-7}"

SRC_REPO="${JANUSDNA_REPO}"                                       # baseline, untouched
STEM_REPO="${PENGUIN_ROOT}/JanusDNA_stem_s${STEM_STRIDE}"         # patched copy, per stride
PATCH_FILE="${PATCH_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/janusdna_stem.patch}"
OUT_ROOT="${OUT_ROOT:-$PENGUIN_ROOT/janusdna_stem/pretrain_scratch}"
FASTA="${FASTA:-$PENGUIN_ROOT/data/penguin/GCF_965638725.1.fna}"
BED="${BED:-$PENGUIN_ROOT/data/penguin/penguin_pretrain_clean.bed}"
CONDA_ENV="${CONDA_ENV:-janusdna}"

MODE="${1:-smoke}"

# ---- safety guards: never confuse the copy with the baseline ----
[ "$STEM_REPO" != "$SRC_REPO" ] || { echo "ERROR: STEM_REPO == SRC_REPO"; exit 1; }
case "$STEM_REPO" in
  *_stem*|*JanusDNA_stem*) : ;;
  *) echo "ERROR: STEM_REPO does not contain '_stem', refusing for safety"; exit 1;;
esac
[ -d "$SRC_REPO/janusdna" ] || { echo "ERROR: $SRC_REPO/janusdna does not exist"; exit 1; }
[ -f "$PATCH_FILE" ] || { echo "ERROR: patch not found: $PATCH_FILE"; exit 1; }
[ -f "$FASTA" ] || { echo "ERROR: FASTA not found: $FASTA"; exit 1; }
[ -f "$BED" ]   || { echo "ERROR: BED not found: $BED"; exit 1; }

echo ">>> [1/4] Copying the repository (baseline untouched): $SRC_REPO -> $STEM_REPO"
# Always a fresh copy: guarantees a deterministic patched state and makes double
# application impossible.
if [ -d "$STEM_REPO" ]; then
  echo "    (removing the previous copy $STEM_REPO)"
  rm -rf "$STEM_REPO"
fi
mkdir -p "$(dirname "$STEM_REPO")"
# Portable copy: rsync when available, otherwise cp -a plus pruning. rsync is
# absent from some minimal container images, which is why the fallback exists.
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude='.git' --exclude='*.ckpt' --exclude='wandb/' "$SRC_REPO/" "$STEM_REPO/"
else
  mkdir -p "$STEM_REPO"
  cp -a "$SRC_REPO/." "$STEM_REPO/"
  rm -rf "$STEM_REPO/.git" "$STEM_REPO/wandb" 2>/dev/null || true
  find "$STEM_REPO" -name '*.ckpt' -delete 2>/dev/null || true
fi

echo ">>> [2/4] Applying $PATCH_FILE"
cd "$STEM_REPO"
if git apply --check "$PATCH_FILE" 2>/dev/null; then
  git apply "$PATCH_FILE"
elif patch -p1 --dry-run < "$PATCH_FILE" >/dev/null 2>&1; then
  patch -p1 < "$PATCH_FILE"
else
  echo "[ABORT] the patch does not apply cleanly to $SRC_REPO."
  echo "        The patch targets JanusDNA at commit 3f44954. Check out that"
  echo "        commit, or rebase the patch onto the version you have."
  exit 1
fi
echo "[OK] patch applied"

echo ">>> [3/4] Pre-launch checks"
python3 -m py_compile "$STEM_REPO/janusdna/modeling_janusdna.py" \
  && echo "[OK] patched modeling_janusdna.py compiles"
python3 -m py_compile "$STEM_REPO/train.py" \
  && echo "[OK] patched train.py compiles"

if ! python -c "import torch" >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    source /etc/profile.d/conda.sh 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || true
  fi
fi

# Standalone gate: the causality mathematics, before any GPU is involved.
echo "    -- standalone stem gate --"
python "$STEM_REPO/janusdna/causal_conv_stem.py" \
  || { echo "[ABORT] standalone gate failed"; exit 1; }

# CRITICAL anti-contamination check: confirm that the `janusdna` actually
# imported is the PATCHED COPY and not an editable install pointing at the
# baseline. If it were the baseline, the patch would be silently ignored and the
# "experiment" would be a second run of the control.
export PYTHONPATH="$STEM_REPO:${PYTHONPATH:-}"
JANUS_STEM=1 python - "$STEM_REPO" << 'PY'
import sys, os, janusdna, janusdna.modeling_janusdna as m
want = sys.argv[1]
got = os.path.dirname(janusdna.__file__)
if not got.startswith(want):
    sys.exit(f"[ABORT] 'janusdna' imported from {got}, NOT from {want}. "
             "Probably an editable install of the baseline: the patch would be ignored.")
assert hasattr(m, "_USE_STEM"), "[ABORT] patch not loaded: _USE_STEM absent from the imported module"
print(f"[OK] import verified: janusdna from {got} | _USE_STEM present")
PY

echo ">>> [4/4] Launching training (MODE=$MODE, S=$STEM_STRIDE)"

# ---- enable the stem through the environment (read at model import time) ----
export JANUS_STEM=1
export JANUS_STEM_STRIDE="${STEM_STRIDE}"
export JANUS_STEM_KERNEL="${STEM_KERNEL}"
export JANUS_STEM_GATE="${JANUS_STEM_GATE:-1}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
# gate result on a persistent path: it must survive the rm -rf of the copy
mkdir -p "$OUT_ROOT"
export JANUS_STEM_GATE_LOG="$OUT_ROOT/GATE_RESULT.txt"

# ---- run knobs (identical to the baseline; only the paths differ) ----
if [ "$MODE" = "smoke" ]; then
  SEQLEN=1024;   MAX_STEPS=50;    GLOBAL_BATCH=8; PER_GPU_BATCH=1
  echo "    SMOKE: seqlen=$SEQLEN steps=$MAX_STEPS"
else
  SEQLEN="${SEQLEN:-131072}"; MAX_STEPS="${MAX_STEPS:-50000}"
  GLOBAL_BATCH="${GLOBAL_BATCH:-8}"; PER_GPU_BATCH="${PER_GPU_BATCH:-1}"
  echo "    FULL: seqlen=$SEQLEN steps=$MAX_STEPS (bp-matched to the baseline)"
fi
LR="${LR:-8e-3}"; NUM_WORKERS="${NUM_WORKERS:-4}"

# ---- 144-dim wo_midattn architecture (identical to the baseline) ----
D_MODEL=144; FLEX_ATTN_MODEL=256; INTER_FFN_MODEL=576; N_LAYER=8
ROUTER_AUX_LOSS_COEF=0.2; OUTPUT_ROUTER_LOGITS="true"
BIDIRECTIONAL_WEIGHT_TIE="true"; BIDIRECTIONAL_ATTN_TIE="false"; RC_AUG="false"

RUN_NAME="janusdna_stem_s${STEM_STRIDE}_penguin_len-$((SEQLEN/1024))k_lr-${LR}_step-${MAX_STEPS}"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"

# ---- EXPLICIT resume control, to avoid accidental resumes between runs ----
# train.py calls fsspec_exists(config.train.ckpt) BEFORE the is-not-None check,
# and fsspec_exists(None) crashes. So "no resume" is expressed as a NON-EXISTENT
# PATH string, not null: fsspec_exists -> False -> the train-from-scratch branch.
NO_CKPT="${RUN_DIR}/__no_resume__.ckpt"   # deliberately does not exist
if [ "$MODE" = "smoke" ]; then
  # Lightning happily "resumes" an already-finished smoke run and exits
  # immediately, which looks like success. Wipe the directory instead.
  [ -d "$RUN_DIR" ] && { echo "    (clearing the previous smoke directory)"; rm -rf "$RUN_DIR"; }
  CKPT_ARG="train.ckpt=${NO_CKPT}"
  echo "    resume: DISABLED (smoke always runs from scratch)"
else
  LAST="${RUN_DIR}/checkpoints/last.ckpt"
  if [ -f "$LAST" ]; then
    CKPT_ARG="train.ckpt=${LAST}"
    echo "    resume: from ${LAST} (crash recovery)"
  else
    CKPT_ARG="train.ckpt=${NO_CKPT}"
    echo "    resume: no previous checkpoint, starting from scratch"
  fi
fi
mkdir -p "${RUN_DIR}"

cd "${STEM_REPO}"
python -m train \
  experiment=hg38/hg38 \
  model="janusdna" \
  dataset.fasta_file="${FASTA}" \
  dataset.bed_file="${BED}" \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=${PER_GPU_BATCH} \
  dataset.batch_size_eval=${PER_GPU_BATCH} \
  dataset.mlm=False \
  dataset.rc_aug="${RC_AUG}" \
  dataset.add_eos=false \
  loader.num_workers=${NUM_WORKERS} \
  +model.config.output_router_logits="${OUTPUT_ROUTER_LOGITS}" \
  +model.config.router_aux_loss_coef="${ROUTER_AUX_LOSS_COEF}" \
  model.config.bidirectional_weight_tie="${BIDIRECTIONAL_WEIGHT_TIE}" \
  model.config.bidirectional_attn_tie="${BIDIRECTIONAL_ATTN_TIE}" \
  model.config.num_hidden_layers=${N_LAYER} \
  model.config.hidden_size=${D_MODEL} \
  model.config.flex_attn_n_embd=${FLEX_ATTN_MODEL} \
  +model.config.intermediate_size=${INTER_FFN_MODEL} \
  model.config.expert_layer_period=2 \
  model.config.expert_layer_offset=1 \
  model.config.intermediate_factor=4 \
  model.config.num_attention_heads=4 \
  model.config.attn_implementation="flash_attention_2" \
  model.config.attn_layer_period=8 \
  model.config.attn_layer_offset=100 \
  optimizer.lr="${LR}" \
  train.global_batch_size=${GLOBAL_BATCH} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  +trainer.val_check_interval=1.0 \
  wandb.mode=offline \
  wandb.name="${RUN_NAME}" \
  hydra.run.dir="${RUN_DIR}" \
  ${CKPT_ARG} \
  2>&1 | tee -a "${RUN_DIR}/train.log"

echo ""
echo ">>> DONE. Isolated output in: ${RUN_DIR}"
echo ">>> Gate result: ${JANUS_STEM_GATE_LOG}"
