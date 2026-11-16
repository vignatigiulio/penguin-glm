#!/bin/bash
# test_impatto_padding.sh — does the padding move the METRICS, not just the vectors?
#
# THE QUESTION. gate_padding_neutrality.sh checks whether the vectors move with
# the padding. But "do the vectors change?" is not what the conclusions rest on.
# The question that matters is:
#
#     does the padding change NMI vs GC, NMI vs function, and silhouette —
#     the quantities the argument is built from?
#
# A vector-level gate can fail while every conclusion stands, and it can pass
# while a metric shifts. Testing the metric directly removes the inference.
#
# METHOD. S=8 runs both natively (1000 bp, M=125) and padded to 1024 (M=128), so
# the comparison is available on one scale. Extract the padded embeddings over
# ALL windows, cluster both ways, and compare the metrics.
#
# RECORDED OUTCOMES:
#   1024 (+2.4%) -> NMI vs GC unchanged to four decimals  -> S=16 INCLUDED
#   2048  (+51%) -> NMI vs GC moves by 5.1%               -> S=32 EXCLUDED
#
# 5.1% is comparable to the effect under study, so at that padding level the
# measurement could not be distinguished from its own artefact.
#
# Usage:
#   bash test_impatto_padding.sh [PAD_TO]     # default 1024
set -e
set -o pipefail

: "${PENGUIN_ROOT:=$HOME/penguin_glm}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S="${S:-8}"
PAD_TO="${1:-1024}"
OUT="${PENGUIN_ROOT}/out/test_impatto_padding_${PAD_TO}"
mkdir -p "${OUT}"

STEM_REPO="${PENGUIN_ROOT}/JanusDNA_stem_s${S}"
[ -d "${STEM_REPO}/janusdna" ] || STEM_REPO="${PENGUIN_ROOT}/JanusDNA_stem"
RUN_DIR="${RUN_DIR:-${PENGUIN_ROOT}/janusdna_stem/pretrain_scratch/janusdna_stem_s${S}_penguin_len-128k_lr-8e-3_step-50000}"
EXTRACTOR="${REPO_DIR}/src/extract_janusdna.py"
NATIVE_NPZ="${PENGUIN_ROOT}/out/embeddings_janusdna144_stem_s${S}/embeddings_144_stem_s${S}.npz"
PADDED_NPZ="${OUT}/embeddings_144_stem_s${S}_pad${PAD_TO}.npz"

[ -f "${EXTRACTOR}" ]  || { echo "ERROR: extractor not found: ${EXTRACTOR}"; exit 1; }
[ -f "${NATIVE_NPZ}" ] || { echo "ERROR: native S=${S} extraction missing: ${NATIVE_NPZ}"; exit 1; }
[ -f "${RUN_DIR}/checkpoints/last.ckpt" ] || { echo "ERROR: no S=${S} checkpoint"; exit 1; }

export JANUS_STEM=1
export JANUS_STEM_STRIDE=${S}
export JANUS_STEM_KERNEL=7
export PYTHONPATH="${STEM_REPO}:${PYTHONPATH:-}"
unset JANUS_FLEX_BLOCK

echo "=========================================="
echo "STEP 1: full padded extraction (S=${S}, pad-to=${PAD_TO})"
echo "=========================================="
if [ -f "${PADDED_NPZ}" ]; then
  echo "  already present, reusing: ${PADDED_NPZ}"
else
  python "${EXTRACTOR}" \
      --repo-path "${STEM_REPO}" \
      --model-config "${RUN_DIR}/model_config.json" \
      --checkpoint "${RUN_DIR}/checkpoints/last.ckpt" \
      --windows-parquet "${PENGUIN_ROOT}/out/windows.parquet" \
      --seq-col seq \
      --output "${PADDED_NPZ}" \
      --stem-stride "${S}" --pad-to "${PAD_TO}" \
      --batch-size 32 2>&1 | tee "${OUT}/log_extract.txt"
fi

echo ""
echo "=========================================="
echo "STEP 2: clustering both ways, same protocol"
echo "=========================================="
python "${REPO_DIR}/src/cluster_embeddings.py" \
    --windows "${PENGUIN_ROOT}/out/windows.parquet" \
    --emb "native=${NATIVE_NPZ}" \
    --emb "padded=${PADDED_NPZ}" \
    --outdir "${OUT}/cluster" 2>&1 | tee "${OUT}/log_cluster.txt"

echo ""
echo "=========================================="
echo "STEP 3: metric comparison"
echo "=========================================="
python - "${OUT}/cluster/clustering_results.json" "${PAD_TO}" << 'PY'
import json
import sys

rows = {r["emb"]: r for r in json.load(open(sys.argv[1]))}
pad_to = int(sys.argv[2])
frac = (pad_to - 1000) * 100.0 / pad_to
a, b = rows.get("native"), rows.get("padded")
if not a or not b:
    sys.exit(f"missing entries in the clustering output: {list(rows)}")

keys = [("NMI_vs_GC", "NMI vs GC"), ("NMI_class", "NMI vs function"),
        ("silhouette", "silhouette"), ("ARI_class", "ARI"),
        ("NMI_vs_repeat", "NMI vs repeats")]
print(f"{'metric':20} {'native':>10} {'padded':>10} {'delta':>10} {'rel':>9}")
worst = 0.0
for k, label in keys:
    va, vb = a.get(k), b.get(k)
    if va is None or vb is None:
        continue
    d = vb - va
    rel = abs(d) / max(abs(va), 1e-12)
    if k in ("NMI_vs_GC", "NMI_class"):
        worst = max(worst, rel)
    print(f"{label:20} {va:10.4f} {vb:10.4f} {d:+10.4f} {100*rel:8.1f}%")

print(f"\npadding: {frac:.1f}% of the sequence")
print(f"largest relative shift on the two decisive metrics: {100*worst:.1f}%")
# The threshold is not arbitrary: the effect being measured (the stem's action
# on NMI vs GC) is tens of percent. An artefact of a few percent is tolerable;
# one of the same order as the effect is not.
if worst < 0.01:
    print("\n[PASSED] the metrics are unchanged. The scale requiring this "
          "padding can be included in the comparison.")
elif worst < 0.03:
    print("\n[ACCEPTABLE] the metrics move by a few percent, well below the "
          "effect under study. Include the scale and state the caveat.")
else:
    print("\n[FAILED] the metrics move by an amount comparable to the effect "
          "being measured.")
    print("The scale requiring this padding cannot be compared on equal terms "
          "and must be excluded.")
    sys.exit(1)
PY
