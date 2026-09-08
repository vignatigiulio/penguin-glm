#!/bin/bash
# extract_stem.sh — embedding extraction for JanusDNA-144 + conv stem, isolated
# and protocol-matched to the baseline.
#
#   STEP 0  CONFIGURATION GATE: the stem structure is derived from the
#           checkpoint and compared with the environment. See below for why a
#           count-based gate is not enough on its own.
#   STEP 1  smoke run (200 windows) with the stem active
#   STEP 2  WEIGHT-LOADING GATE (missing/unexpected ~ 0)
#   STEP 3  embedding sanity check (dimension, no NaN)
#   STEP 4  full extraction
#
# src/extract_janusdna.py is stem-aware: it detects the resolution mismatch
# itself and compresses the attention mask from nucleotide to super-token
# resolution, so one extractor serves both the baseline and the stem.
#
# WHY STEP 0 EXISTS
# -----------------
# The model is rebuilt from model_config.json plus environment variables, then
# the checkpoint is loaded with strict=False. Two mismatches do not raise:
#   * A wrong stage count leaves only four unexpected keys
#     (stem.blocks.<n>.conv.weight/bias, .norm.weight/bias), below the
#     tolerance of STEP 2.
#   * A wrong BLOCK changes BAND = BLOCK / STRIDE, which drives the final
#     fusion offsets. BAND is not a parameter, so there is no missing or
#     unexpected key at all: the embeddings are simply computed with a fusion
#     the model was never trained with.
#
# Usage:
#   bash extract_stem.sh                                        # S=4, block=4
#   STEM_STRIDE=1 STEM_BLOCK=4 STEM_NSTAGES=3 bash extract_stem.sh
set -e

: "${PENGUIN_ROOT:=$HOME/penguin_glm}"
STEM_STRIDE="${STEM_STRIDE:-4}"
STEM_BLOCK="${STEM_BLOCK:-$STEM_STRIDE}"
STEM_KERNEL="${STEM_KERNEL:-7}"
STEM_NSTAGES="${STEM_NSTAGES:-}"          # stride=1 only; empty = builder default
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ $(( STEM_BLOCK % STEM_STRIDE )) -ne 0 ]; then
  echo "ERROR: STEM_BLOCK=$STEM_BLOCK is not a multiple of STEM_STRIDE=$STEM_STRIDE."
  echo "       The band is measured in grid cells: BAND = BLOCK / STRIDE."
  exit 1
fi
STEM_BAND=$(( STEM_BLOCK / STEM_STRIDE ))
TAG="s${STEM_STRIDE}_b${STEM_BLOCK}"

# Patched copy. Naming has evolved across runs, so several layouts are accepted,
# most specific first: s<S>_b<B> (v2), s<S> (v1 sweep), no suffix (first S=4 run).
for cand in "JanusDNA_stem_${TAG}" "JanusDNA_stem_s${STEM_STRIDE}" "JanusDNA_stem"; do
  if [ -d "${PENGUIN_ROOT}/${cand}/janusdna" ]; then STEM_REPO="${PENGUIN_ROOT}/${cand}"; break; fi
done
[ -n "${STEM_REPO:-}" ] || { echo "ERROR: no patched repository found under ${PENGUIN_ROOT}"; exit 1; }

# Run directory: same naming evolution.
if [ -z "${RUN_DIR:-}" ]; then
  for cand in "janusdna_stem_${TAG}_penguin_len-128k_lr-8e-3_step-50000" \
              "janusdna_stem_s${STEM_STRIDE}_penguin_len-128k_lr-8e-3_step-50000"; do
    if [ -d "${PENGUIN_ROOT}/janusdna_stem/pretrain_scratch/${cand}" ]; then
      RUN_DIR="${PENGUIN_ROOT}/janusdna_stem/pretrain_scratch/${cand}"; break
    fi
  done
fi
[ -n "${RUN_DIR:-}" ] || { echo "ERROR: no run directory found; set RUN_DIR explicitly"; exit 1; }

MODEL_CONFIG="${RUN_DIR}/model_config.json"
CHECKPOINT="${RUN_DIR}/checkpoints/last.ckpt"
WINDOWS_PARQUET="${WINDOWS_PARQUET:-${PENGUIN_ROOT}/out/windows.parquet}"
SEQ_COL="seq"
OUTPUT_DIR="${PENGUIN_ROOT}/out/embeddings_janusdna144_stem_${TAG}"
EXPECTED_DIM=144
BATCH_SIZE=32
SMOKE_TEST_N=200
EXTRACTOR="${REPO_DIR}/src/extract_janusdna.py"

export JANUS_STEM=1
export JANUS_STEM_STRIDE="${STEM_STRIDE}"
export JANUS_STEM_BLOCK="${STEM_BLOCK}"
export JANUS_STEM_KERNEL="${STEM_KERNEL}"
[ -n "$STEM_NSTAGES" ] && export JANUS_STEM_NSTAGES="${STEM_NSTAGES}"
export PYTHONPATH="${STEM_REPO}:${PYTHONPATH:-}"
echo ">>> stride=${STEM_STRIDE} block=${STEM_BLOCK} band=${STEM_BAND} | repo=${STEM_REPO}"
echo ">>> run=${RUN_DIR}"

# ---- guards ----
[ -f "$MODEL_CONFIG" ]    || { echo "ERROR: model_config not found: $MODEL_CONFIG"; exit 1; }
[ -f "$CHECKPOINT" ]      || { echo "ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }
[ -f "$WINDOWS_PARQUET" ] || { echo "ERROR: parquet not found: $WINDOWS_PARQUET"; exit 1; }
[ -f "$EXTRACTOR" ]       || { echo "ERROR: extractor not found: $EXTRACTOR"; exit 1; }
mkdir -p "${OUTPUT_DIR}"

SMOKE_NPZ="${OUTPUT_DIR}/smoke_${SMOKE_TEST_N}.npz"
SMOKE_LOG="${OUTPUT_DIR}/smoke_${SMOKE_TEST_N}.log"

echo "=========================================="
echo "STEP 0/4: configuration gate (checkpoint vs environment)"
echo "=========================================="
python "${REPO_DIR}/tools/check_stem_config.py" \
    --checkpoint "${CHECKPOINT}" \
    --stride "${STEM_STRIDE}" \
    --block "${STEM_BLOCK}" \
    --kernel "${STEM_KERNEL}" \
    ${STEM_NSTAGES:+--nstages "${STEM_NSTAGES}"}

echo ""
echo "=========================================="
echo "STEP 1/4: smoke test on ${SMOKE_TEST_N} windows (stem active)"
echo "=========================================="
python "$EXTRACTOR" \
    --repo-path "${STEM_REPO}" \
    --model-config "${MODEL_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --windows-parquet "${WINDOWS_PARQUET}" \
    --seq-col "${SEQ_COL}" \
    --output "${SMOKE_NPZ}" \
    --batch-size "${BATCH_SIZE}" \
    --limit "${SMOKE_TEST_N}" 2>&1 | tee "${SMOKE_LOG}"

echo ""
echo "=========================================="
echo "STEP 2/4: weight-loading gate (missing/unexpected ~ 0)"
echo "=========================================="
python - "${SMOKE_LOG}" << 'PY'
import re, sys
log = open(sys.argv[1]).read()
m = re.search(r"missing=(\d+)\s+unexpected=(\d+)", log)
if not m:
    sys.exit("[GATE] could not find 'missing=.. unexpected=..' in the log. "
             "Check the extractor's load_state_dict report.")
missing, unexpected = int(m.group(1)), int(m.group(2))
print(f"[GATE] missing={missing} unexpected={unexpected}")
if missing > 8 or unexpected > 8:
    sys.exit(f"[GATE FAILED] mismatch too large (missing={missing}, unexpected={unexpected}). "
             "The stem most likely did NOT activate: the model was built without it, the "
             "stem.*/lm_head weights were ignored, and the embeddings are NOT valid. "
             "Check JANUS_STEM=1 and --repo-path.")
print("[GATE OK] checkpoint weights (stem included) loaded into the right model.")
print("          NOTE: this threshold cannot catch a wrong stage count or a wrong")
print("          BLOCK. STEP 0 is what covers those.")
PY

echo ""
echo "=========================================="
echo "STEP 3/4: embedding sanity (dim ${EXPECTED_DIM}, no NaN)"
echo "=========================================="
python "${REPO_DIR}/tools/check_embeddings.py" \
    --npz "${SMOKE_NPZ}" --expected-dim "${EXPECTED_DIM}"

echo ""
echo "=========================================="
echo "STEP 4/4: gates passed -> FULL extraction"
echo "=========================================="
read -p "Proceed on the whole windows.parquet? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Stopped. The smoke output ${SMOKE_NPZ} is left for inspection."
    exit 0
fi

FULL_NPZ="${OUTPUT_DIR}/embeddings_144_stem_${TAG}.npz"
python "$EXTRACTOR" \
    --repo-path "${STEM_REPO}" \
    --model-config "${MODEL_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --windows-parquet "${WINDOWS_PARQUET}" \
    --seq-col "${SEQ_COL}" \
    --output "${FULL_NPZ}" \
    --batch-size "${BATCH_SIZE}"

echo ""
echo "=========================================="
echo "COMPLETE: ${FULL_NPZ}"
echo "Next: src/cluster_embeddings.py on this .npz -> NMI function vs GC,"
echo "compared against the baseline."
echo "=========================================="
