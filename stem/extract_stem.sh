#!/bin/bash
# extract_stem.sh — embedding extraction for JanusDNA-144 + conv stem, isolated
# and protocol-matched to the baseline.
#
#   STEP 1  smoke run (200 windows) with the stem active
#   STEP 2  WEIGHT-LOADING GATE (missing/unexpected ~ 0). load_state_dict runs
#           with strict=False, so a mismatch does not raise; this step catches it.
#   STEP 3  embedding sanity check (dimension, no NaN)
#   STEP 4  full extraction
#
# The original version of this script generated a stem-aware copy of the
# extractor on the fly. That is no longer necessary: src/extract_janusdna.py
# detects the resolution mismatch itself and compresses the attention mask
# accordingly, so the same extractor serves both the baseline and the stem.
#
# Usage:
#   bash extract_stem.sh
#   STEM_STRIDE=8 bash extract_stem.sh
set -e

: "${PENGUIN_ROOT:=$HOME/penguin_glm}"
STEM_STRIDE="${STEM_STRIDE:-4}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Patched copy: newer runs use JanusDNA_stem_s<N>, the first S=4 run used
# JanusDNA_stem with no suffix. Both are accepted, the suffixed one preferred.
if   [ -d "${PENGUIN_ROOT}/JanusDNA_stem_s${STEM_STRIDE}/janusdna" ]; then
  STEM_REPO="${PENGUIN_ROOT}/JanusDNA_stem_s${STEM_STRIDE}"
elif [ -d "${PENGUIN_ROOT}/JanusDNA_stem/janusdna" ]; then
  STEM_REPO="${PENGUIN_ROOT}/JanusDNA_stem"
else
  echo "ERROR: no patched repository found (neither JanusDNA_stem_s${STEM_STRIDE} nor JanusDNA_stem)"
  exit 1
fi

RUN_DIR="${RUN_DIR:-${PENGUIN_ROOT}/janusdna_stem/pretrain_scratch/janusdna_stem_s${STEM_STRIDE}_penguin_len-128k_lr-8e-3_step-50000}"
MODEL_CONFIG="${RUN_DIR}/model_config.json"
CHECKPOINT="${RUN_DIR}/checkpoints/last.ckpt"
WINDOWS_PARQUET="${WINDOWS_PARQUET:-${PENGUIN_ROOT}/out/windows.parquet}"
SEQ_COL="seq"
OUTPUT_DIR="${PENGUIN_ROOT}/out/embeddings_janusdna144_stem_s${STEM_STRIDE}"
EXPECTED_DIM=144
BATCH_SIZE=32
SMOKE_TEST_N=200
EXTRACTOR="${REPO_DIR}/src/extract_janusdna.py"

export JANUS_STEM=1
export JANUS_STEM_STRIDE="${STEM_STRIDE}"
export JANUS_STEM_KERNEL=7
export PYTHONPATH="${STEM_REPO}:${PYTHONPATH:-}"
echo ">>> scale S=${STEM_STRIDE} | repo=${STEM_REPO}"

# ---- guards ----
[ -f "$MODEL_CONFIG" ]    || { echo "ERROR: model_config not found: $MODEL_CONFIG"; exit 1; }
[ -f "$CHECKPOINT" ]      || { echo "ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }
[ -f "$WINDOWS_PARQUET" ] || { echo "ERROR: parquet not found: $WINDOWS_PARQUET"; exit 1; }
[ -f "$EXTRACTOR" ]       || { echo "ERROR: extractor not found: $EXTRACTOR"; exit 1; }
[ -d "$STEM_REPO/janusdna" ] || { echo "ERROR: patched repo not found: $STEM_REPO"; exit 1; }
mkdir -p "${OUTPUT_DIR}"

SMOKE_NPZ="${OUTPUT_DIR}/smoke_${SMOKE_TEST_N}.npz"
SMOKE_LOG="${OUTPUT_DIR}/smoke_${SMOKE_TEST_N}.log"

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
PY

echo ""
echo "=========================================="
echo "STEP 3/4: embedding sanity (dim ${EXPECTED_DIM}, no NaN)"
echo "=========================================="
python "${REPO_DIR}/tools/check_embeddings.py" \
    --npz "${SMOKE_NPZ}" --expected-dim "${EXPECTED_DIM}"

echo ""
echo "=========================================="
echo "STEP 4/4: smoke + gate + sanity passed -> FULL extraction"
echo "=========================================="
read -p "Proceed on the whole windows.parquet? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Stopped. The smoke output ${SMOKE_NPZ} is left for inspection."
    exit 0
fi

FULL_NPZ="${OUTPUT_DIR}/embeddings_144_stem_s${STEM_STRIDE}.npz"
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
