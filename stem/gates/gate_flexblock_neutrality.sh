#!/bin/bash
# gate_flexblock_neutrality.sh — is lowering FlexAttention's BLOCK_SIZE neutral?
#
# env/patches/flexattn_blocksize.py prescribes this check in its own docstring:
# before trusting it at short grids, re-extract a length that already works with
# and without the patch and confirm the embeddings agree to numerical tolerance.
# If they diverge non-trivially, the patch is NOT neutral in this environment and
# must be abandoned rather than worked around.
#
# WHY IT CAN BE TESTED AT ALL. S=8 runs WITHOUT the patch (M=125, so q_len=250 is
# above the 128 block), so the same windows can be extracted both ways and
# compared. If they agree, the patch could be used for S=16 and S=32, where the
# Triton kernel otherwise fails with a device-side assert.
#
# RECORDED OUTCOME ON THIS PROJECT: **REJECTED**. Minimum per-row cosine 0.937,
# mean relative difference 21%. Lowering BLOCK_SIZE is not a re-tiling of the
# same computation in this build; it changes the representations. The short
# grids were therefore reached by padding instead — see
# gate_padding_neutrality.sh — and S=32 was excluded entirely.
#
# This script is kept so the rejection is reproducible rather than asserted.
#
# Usage:
#   bash gate_flexblock_neutrality.sh
set -e
set -o pipefail

: "${PENGUIN_ROOT:=$HOME/penguin_glm}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S="${S:-8}"
N="${N:-200}"                       # trial windows
OUT="${PENGUIN_ROOT}/out/gate_flexblock"
mkdir -p "${OUT}"

STEM_REPO="${PENGUIN_ROOT}/JanusDNA_stem_s${S}"
[ -d "${STEM_REPO}/janusdna" ] || STEM_REPO="${PENGUIN_ROOT}/JanusDNA_stem"
RUN_DIR="${RUN_DIR:-${PENGUIN_ROOT}/janusdna_stem/pretrain_scratch/janusdna_stem_s${S}_penguin_len-128k_lr-8e-3_step-50000}"
EXTRACTOR="${REPO_DIR}/src/extract_janusdna.py"

[ -f "${EXTRACTOR}" ] || { echo "ERROR: extractor not found: ${EXTRACTOR}"; exit 1; }
[ -f "${RUN_DIR}/checkpoints/last.ckpt" ] || { echo "ERROR: no S=${S} checkpoint at ${RUN_DIR}"; exit 1; }
[ -d "${STEM_REPO}/janusdna" ] || { echo "ERROR: patched repo not found"; exit 1; }

export JANUS_STEM=1
export JANUS_STEM_STRIDE=${S}
export JANUS_STEM_KERNEL=7
export PYTHONPATH="${STEM_REPO}:${PYTHONPATH:-}"

run_extract () {   # $1 = label, $2 = JANUS_FLEX_BLOCK value ("" = no patch)
  echo "--- extraction ${1} (JANUS_FLEX_BLOCK='${2}') ---"
  if [ -n "$2" ]; then export JANUS_FLEX_BLOCK="$2"; else unset JANUS_FLEX_BLOCK; fi
  python "${EXTRACTOR}" \
      --repo-path "${STEM_REPO}" \
      --model-config "${RUN_DIR}/model_config.json" \
      --checkpoint "${RUN_DIR}/checkpoints/last.ckpt" \
      --windows-parquet "${PENGUIN_ROOT}/out/windows.parquet" \
      --seq-col seq \
      --output "${OUT}/emb_${1}.npz" \
      --batch-size 32 --limit ${N} > "${OUT}/log_${1}.txt" 2>&1
  echo "    -> ${OUT}/emb_${1}.npz"
}

echo "=========================================="
echo "BLOCK_SIZE NEUTRALITY GATE  (S=${S}, ${N} windows)"
echo "=========================================="
run_extract "no_patch" ""
run_extract "patch32" "32"

echo ""
echo "--- numerical comparison ---"
python - "${OUT}/emb_no_patch.npz" "${OUT}/emb_patch32.npz" << 'PY'
import sys
import numpy as np

a = np.load(sys.argv[1], allow_pickle=True)
b = np.load(sys.argv[2], allow_pickle=True)
Xa, Xb = a["X"].astype(np.float64), b["X"].astype(np.float64)
assert Xa.shape == Xb.shape, f"different shapes: {Xa.shape} vs {Xb.shape}"
if "ids" in a and "ids" in b:
    assert list(map(str, a["ids"])) == list(map(str, b["ids"])), "ids misaligned"

diff = np.abs(Xa - Xb)
scale = np.abs(Xa).mean() + 1e-12
rel = diff.mean() / scale
# Per-ROW cosine, not a global norm: these embeddings are consumed by clustering
# and by a probe, both of which see one window at a time. A global average would
# hide a subset of badly displaced rows.
cos = (Xa * Xb).sum(1) / (np.linalg.norm(Xa, axis=1) * np.linalg.norm(Xb, axis=1) + 1e-12)

print(f"  shape           : {Xa.shape}")
print(f"  |diff| max      : {diff.max():.3e}")
print(f"  |diff| mean     : {diff.mean():.3e}   (relative: {rel:.3e})")
print(f"  per-row cosine  : min={cos.min():.8f}  mean={cos.mean():.8f}")

# bf16 carries ~3 decimal digits, so relative differences around 1e-3 are
# expected and harmless. Anything larger is not rounding.
if cos.min() > 0.9999 and rel < 1e-2:
    print("\n[GATE PASSED] the patch is NEUTRAL within numerical tolerance.")
    print("It may be used for S=16 and S=32.")
elif cos.min() > 0.99:
    print("\n[WARNING] small but non-negligible differences (cosine < 0.9999).")
    print("Decide whether that is acceptable, and report the deviation with any result.")
    sys.exit(2)
else:
    print("\n[GATE FAILED] the patch is NOT neutral in this environment.")
    print("Do not use it: S=16 and S=32 must be excluded or handled another way.")
    sys.exit(1)
PY
