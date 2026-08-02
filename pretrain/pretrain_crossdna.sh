#!/bin/bash
# pretrain_crossdna.sh — pre-train CrossDNA-8.1M FROM SCRATCH on the penguin
# genome (CLEAN corpus), single GPU.
#
# Adapted from upstream scripts/pre_train/CrossDNA_2k.sh: same architecture
# (d_model=128, depth=6 — the 8.1M configuration), same objective, same fp32
# precision. Differences: penguin paths, a BED with the downstream val/test
# scaffolds removed, a budget expressed in STEPS rather than epochs, wandb
# offline.
#
# Usage:
#   ./pretrain_crossdna.sh smoke          # quick check
#   MAX_STEPS=20000 ./pretrain_crossdna.sh
#
# Paths come from the environment:
#   PENGUIN_ROOT    workspace root       (default: $HOME/penguin_glm)
#   CROSSDNA_REPO   CrossDNA repository  (default: $PENGUIN_ROOT/../CrossDNA)
set -euo pipefail

: "${PENGUIN_ROOT:=$HOME/penguin_glm}"
: "${CROSSDNA_REPO:=$(dirname "$PENGUIN_ROOT")/CrossDNA}"

FASTA="${FASTA:-$PENGUIN_ROOT/data/penguin/GCF_965638725.1_bAptPat1.pri.cur_genomic.fna}"
BED="${BED:-$PENGUIN_ROOT/data/penguin/penguin_pretrain_clean_2k.bed}"
OUT_ROOT="${OUT_ROOT:-$PENGUIN_ROOT/crossdna/pretrain_scratch_clean}"

# ---- 8.1M recipe (identical to CrossDNA_2k.sh) ----
SEQLEN="${SEQLEN:-2048}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
D_MODEL="${D_MODEL:-128}"
DEPTH="${DEPTH:-6}"
BATCH_SIZE="${BATCH_SIZE:-60}"     # per GPU; global batch = 60 on one GPU
LR="${LR:-3e-4}"                   # the official script's value (the config says 1e-4)
RC_AUG="false"

MAX_STEPS="${MAX_STEPS:-20000}"    # budget in steps (1 epoch = 7590 steps on the clean corpus)
NUM_WORKERS="${NUM_WORKERS:-8}"
TAG="${TAG:-clean}"
CKPT="${CKPT:-}"

if [ "${1:-}" = "smoke" ]; then
  MAX_STEPS=50; BATCH_SIZE=8
  echo ">> SMOKE TEST: max_steps=$MAX_STEPS batch=$BATCH_SIZE"
fi

RUN_NAME="crossdna81_scratch_penguin_${TAG}_len-${SEQLEN}_lr-${LR}_step-${MAX_STEPS}"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

RESUME_ARG=""
if [ -n "${CKPT}" ] && [ -f "${CKPT}" ]; then
  RESUME_ARG="train.ckpt=${CKPT}"
  echo ">> RESUMING from: ${CKPT}"
fi

# ---- environment (the image already has the env active) ----
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
# nvjitlink is required by the Mamba/Comba kernels, as in the original script
NVJIT=$(python -c "import nvidia.nvjitlink, os; print(os.path.join(os.path.dirname(nvidia.nvjitlink.__file__),'lib'))" 2>/dev/null || true)
[ -n "${NVJIT}" ] && export LD_LIBRARY_PATH="${NVJIT}:${LD_LIBRARY_PATH:-}"

cd "${CROSSDNA_REPO}"

# trainer.precision=32 is deliberate: CrossDNA overflows in bf16 on this
# configuration, and the training is the one place where the damage would be
# irreversible rather than merely mis-measured.
python -m train \
  experiment=hg38-pretrain/crossdna \
  dataset.fasta_file="${FASTA}" \
  dataset.bed_file="${BED}" \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=${BATCH_SIZE} \
  dataset.batch_size_eval=${BATCH_SIZE} \
  dataset.rc_aug="${RC_AUG}" \
  dataset.add_eos=False \
  loader.num_workers=${NUM_WORKERS} \
  model.config.d_model=${D_MODEL} \
  model.config.depth=${DEPTH} \
  model.config.block_size=${BLOCK_SIZE} \
  optimizer.lr="${LR}" \
  +trainer.max_steps=${MAX_STEPS} \
  ~trainer.max_epochs \
  trainer.devices=1 \
  trainer.precision=32 \
  +trainer.strategy="auto" \
  +trainer.val_check_interval=1.0 \
  scheduler.t_in_epochs=false \
  scheduler.t_initial=${MAX_STEPS} \
  scheduler.warmup_t=$(( MAX_STEPS / 10 > 0 ? MAX_STEPS / 10 : 1 )) \
  wandb.mode=offline \
  wandb.name="${RUN_NAME}" \
  hydra.run.dir="${RUN_DIR}" \
  ${RESUME_ARG} \
  2>&1 | tee -a "${RUN_DIR}/train.log"
