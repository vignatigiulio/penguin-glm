#!/bin/bash
# gate_padding_neutrality.sh — is heavy masked padding neutral on the embeddings?
#
# CONTEXT. With a stem, the length the final attention sees is M = L/S, and
# FlexAttention requires q_len = 2M >= 128, i.e. M >= 64. On 1000 bp windows:
#     S=4  -> M=250  ok         S=8  -> M=125  ok
#     S=16 -> M=63   FAILS      S=32 -> M=32   FAILS
# Lowering BLOCK_SIZE was rejected by gate_flexblock_neutrality.sh (minimum
# cosine 0.937, 21% relative — not numerical noise). That leaves padding: extend
# L to 64*S (1024 for S=16, 2048 for S=32) so that M = 64. The padding is
# excluded from the mean pooling by the mask, but for S=32 it would be over half
# the sequence, so before relying on it that has to be DEMONSTRATED rather than
# assumed.
#
# METHOD. S=8 works both natively (1000 bp, M=125) and padded, so the same
# windows can be extracted both ways and compared directly.
#
# RECORDED OUTCOMES:
#   1024 (+2.4%) -> min cosine 0.9956   the vectors move slightly
#   2048  (+51%) -> min cosine 0.631    the vectors move a great deal
#
# Note that 0.9956 is below the 0.9999 threshold, so this gate alone does not
# clear S=16. It answers "do the vectors change?", which is not the question the
# conclusions rest on. test_impatto_padding.sh asks the one that matters — do
# the METRICS change — and it is that test which cleared S=16.
#
# Usage:
#   bash gate_padding_neutrality.sh [PAD_TO]     # default 1024
set -e
set -o pipefail

: "${PENGUIN_ROOT:=$HOME/penguin_glm}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S="${S:-8}"
N="${N:-200}"
# 1024 -> the fraction S=16 needs to reach M=64
# 2048 -> the fraction S=32 needs   [ALREADY TESTED: REJECTED, 51% padding]
PAD_TO="${1:-1024}"
OUT="${PENGUIN_ROOT}/out/gate_padding_${PAD_TO}"
mkdir -p "${OUT}"

STEM_REPO="${PENGUIN_ROOT}/JanusDNA_stem_s${S}"
[ -d "${STEM_REPO}/janusdna" ] || STEM_REPO="${PENGUIN_ROOT}/JanusDNA_stem"
RUN_DIR="${RUN_DIR:-${PENGUIN_ROOT}/janusdna_stem/pretrain_scratch/janusdna_stem_s${S}_penguin_len-128k_lr-8e-3_step-50000}"
EXTRACTOR="${REPO_DIR}/src/extract_janusdna.py"

[ -f "${EXTRACTOR}" ] || { echo "ERROR: extractor not found: ${EXTRACTOR}"; exit 1; }
[ -f "${RUN_DIR}/checkpoints/last.ckpt" ] || { echo "ERROR: no S=${S} checkpoint"; exit 1; }

export JANUS_STEM=1
export JANUS_STEM_STRIDE=${S}
export JANUS_STEM_KERNEL=7
export PYTHONPATH="${STEM_REPO}:${PYTHONPATH:-}"
unset JANUS_FLEX_BLOCK   # the BLOCK_SIZE patch was rejected; not used here

run_extract () {   # $1 = label, $2 = --pad-to value (0 = natural length)
  echo "--- extraction ${1} (pad-to=${2}) ---"
  python "${EXTRACTOR}" \
      --repo-path "${STEM_REPO}" \
      --model-config "${RUN_DIR}/model_config.json" \
      --checkpoint "${RUN_DIR}/checkpoints/last.ckpt" \
      --windows-parquet "${PENGUIN_ROOT}/out/windows.parquet" \
      --seq-col seq \
      --output "${OUT}/emb_${1}.npz" \
      --stem-stride "${S}" --pad-to "${2}" \
      --batch-size 32 --limit ${N} > "${OUT}/log_${1}.txt" 2>&1
  echo "    -> ${OUT}/emb_${1}.npz"
}

echo "=========================================="
echo "PADDING NEUTRALITY GATE (S=${S}, ${N} windows)"
echo "        native 1000 bp  vs  padded to ${PAD_TO} bp"
echo "=========================================="
run_extract "native_1000" 0
run_extract "padded_${PAD_TO}" "${PAD_TO}"

echo ""
echo "--- numerical comparison ---"
python - "${OUT}/emb_native_1000.npz" "${OUT}/emb_padded_${PAD_TO}.npz" "${PAD_TO}" << 'PY'
import sys
import numpy as np

a = np.load(sys.argv[1], allow_pickle=True)
b = np.load(sys.argv[2], allow_pickle=True)
pad_to = int(sys.argv[3])
frac = (pad_to - 1000) * 100.0 / pad_to
Xa, Xb = a["X"].astype(np.float64), b["X"].astype(np.float64)
assert Xa.shape == Xb.shape, f"different shapes: {Xa.shape} vs {Xb.shape}"
if "ids" in a and "ids" in b:
    assert list(map(str, a["ids"])) == list(map(str, b["ids"])), "ids misaligned"

diff = np.abs(Xa - Xb)
rel = diff.mean() / (np.abs(Xa).mean() + 1e-12)
cos = (Xa * Xb).sum(1) / (np.linalg.norm(Xa, axis=1) * np.linalg.norm(Xb, axis=1) + 1e-12)
print(f"  shape           : {Xa.shape}")
print(f"  |diff| max      : {diff.max():.3e}")
print(f"  |diff| mean     : {diff.mean():.3e}  (relative: {rel:.3e})")
print(f"  per-row cosine  : min={cos.min():.8f}  mean={cos.mean():.8f}")

if cos.min() > 0.9999 and rel < 1e-2:
    print(f"\n[GATE PASSED] padding to {pad_to} bp ({frac:.1f}% of the sequence) "
          "leaves the vectors unchanged.")
elif cos.min() > 0.99:
    print(f"\n[PARTIAL] padding to {pad_to} bp ({frac:.1f}%) moves the vectors "
          "slightly (cosine < 0.9999).")
    print("Vector-level neutrality is NOT established. Run "
          "test_impatto_padding.sh to find out whether the METRICS move, which "
          "is what the conclusions actually depend on.")
    sys.exit(2)
else:
    print(f"\n[GATE FAILED] padding to {pad_to} bp ({frac:.1f}%) alters the "
          "representations substantially.")
    print("The corresponding scale cannot be evaluated on equal terms and must "
          "be excluded and documented.")
    sys.exit(1)
PY
