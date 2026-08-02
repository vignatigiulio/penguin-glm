#!/bin/bash
# pretrain_janusdna.sh — pre-train JanusDNA-144 FROM SCRATCH (random init) on the
# penguin genome, single GPU.
#
# Faithful to the upstream scripts/pre_train/slurm_JanusDNA_wo_midattn_144dim.sh:
# same architecture (8 layers, all Mamba, MoE, bidirectional weight tie) and the
# same objective (character-level causal LM, NOT masked LM). Differences: one GPU
# with gradient accumulation instead of SLURM/DDP, checkpoints on a persistent
# path, wandb offline. No checkpoint is ever loaded, so this really is scratch.
#
# Usage:
#   ./pretrain_janusdna.sh smoke   # quick end-to-end check (seqlen 1024, 50 steps)
#   ./pretrain_janusdna.sh         # real run
#
# Paths come from the environment; set them or accept the defaults:
#   PENGUIN_ROOT    workspace root          (default: $HOME/penguin_glm)
#   JANUSDNA_REPO   JanusDNA repository     (default: $PENGUIN_ROOT/../JanusDNA)
set -euo pipefail

: "${PENGUIN_ROOT:=$HOME/penguin_glm}"
: "${JANUSDNA_REPO:=$(dirname "$PENGUIN_ROOT")/JanusDNA}"

FASTA="${FASTA:-$PENGUIN_ROOT/data/penguin/GCF_965638725.1.fna}"
BED="${BED:-$PENGUIN_ROOT/data/penguin/penguin_pretrain_clean.bed}"
OUT_ROOT="${OUT_ROOT:-$PENGUIN_ROOT/janusdna/pretrain_scratch}"
CONDA_ENV="${CONDA_ENV:-janusdna}"

# ---- run knobs (override from the environment: SEQLEN=8192 MAX_STEPS=20000 ./script)
SEQLEN="${SEQLEN:-131072}"        # must divide 2^20; 131072 matches the human model
MAX_STEPS="${MAX_STEPS:-50000}"   # human budget = 50000 (~52B tokens)
LR="${LR:-8e-3}"                  # needs tuning; 8e-3 is the human/8-GPU value
GLOBAL_BATCH="${GLOBAL_BATCH:-8}" # effective batch, as upstream
PER_GPU_BATCH="${PER_GPU_BATCH:-1}"  # keep at 1 for seqlen 131072
NUM_WORKERS="${NUM_WORKERS:-4}"
CKPT="${CKPT:-}"                  # to RESUME: CKPT=<RUN_DIR>/checkpoints/last.ckpt
TAG="${TAG:-}"                    # label in the run name, e.g. TAG=clean
ACCUM=$(( GLOBAL_BATCH / PER_GPU_BATCH ))

# ---- smoke mode: end-to-end verification in a few minutes ----
if [ "${1:-}" = "smoke" ]; then
  SEQLEN=1024; MAX_STEPS=50; GLOBAL_BATCH=8; PER_GPU_BATCH=1
  echo ">> SMOKE TEST: seqlen=$SEQLEN, max_steps=$MAX_STEPS"
fi

# ---- 144-dim wo_midattn architecture (identical to upstream) ----
D_MODEL=144
FLEX_ATTN_MODEL=256
INTER_FFN_MODEL=576
N_LAYER=8
ROUTER_AUX_LOSS_COEF=0.2
OUTPUT_ROUTER_LOGITS="true"
BIDIRECTIONAL_WEIGHT_TIE="true"
BIDIRECTIONAL_ATTN_TIE="false"
RC_AUG="false"

RUN_NAME="janusdna_scratch_penguin${TAG:+_${TAG}}_len-$((SEQLEN/1024))k_lr-${LR}_step-${MAX_STEPS}"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

# Optional resume. Note that train.ckpt must NOT be passed as the literal `null`
# when there is nothing to resume from: Hydra forwards it and fsspec_exists()
# crashes on it. Omitting the argument entirely is the correct no-op.
RESUME_ARG=""
if [ -n "${CKPT}" ]; then
  if [ -f "${CKPT}" ]; then
    RESUME_ARG="train.ckpt=${CKPT}"
    echo ">> RESUMING from: ${CKPT}"
  else
    echo ">> WARNING: CKPT=${CKPT} not found, starting from scratch"
  fi
fi

# ---- environment ----
# The image usually has the env active already. Activate conda ONLY if torch is
# not importable, and never abort if activation fails (non-conda-init shells).
if ! python -c "import torch" >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    source /etc/profile.d/conda.sh 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || true
  fi
fi
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
cd "${JANUSDNA_REPO}"

# ---- launch (accumulate_grad_batches is derived by the config from
#      GLOBAL_BATCH / (devices * batch)) ----
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
  ${RESUME_ARG} \
  2>&1 | tee -a "${RUN_DIR}/train.log"
