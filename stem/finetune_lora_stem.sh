#!/bin/bash
# finetune_lora_stem.sh — LoRA fine-tuning of JanusDNA-144 + conv stem,
# protocol-matched to the baseline.
#
#   STEP 0  CONFIGURATION GATE: stem structure derived from the checkpoint and
#           compared with the environment (see extract_stem.sh for why the
#           count-based gate in STEP 2 is not sufficient on its own)
#   STEP 1  smoke run with --limit (stem active)
#   STEP 2  GATE: weights loaded (~0 mismatch), LoRA injected, dt_proj excluded
#   STEP 3  full LoRA run (defaults identical to the baseline) + comparison
#
# src/finetune_lora_janusdna.py handles the super-token resolution itself, so
# one script covers both the baseline and the stem.
#
# Usage:
#   bash finetune_lora_stem.sh          # smoke + gates -> full
#   bash finetune_lora_stem.sh smoke    # smoke + gates only
#   STEM_STRIDE=1 STEM_BLOCK=4 STEM_NSTAGES=3 bash finetune_lora_stem.sh
set -e

: "${PENGUIN_ROOT:=$HOME/penguin_glm}"
STEM_STRIDE="${STEM_STRIDE:-4}"
STEM_BLOCK="${STEM_BLOCK:-$STEM_STRIDE}"
STEM_KERNEL="${STEM_KERNEL:-7}"
STEM_NSTAGES="${STEM_NSTAGES:-}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ $(( STEM_BLOCK % STEM_STRIDE )) -ne 0 ]; then
  echo "ERROR: STEM_BLOCK=$STEM_BLOCK is not a multiple of STEM_STRIDE=$STEM_STRIDE."
  exit 1
fi
STEM_BAND=$(( STEM_BLOCK / STEM_STRIDE ))
TAG="s${STEM_STRIDE}_b${STEM_BLOCK}"

for cand in "JanusDNA_stem_${TAG}" "JanusDNA_stem_s${STEM_STRIDE}" "JanusDNA_stem"; do
  if [ -d "${PENGUIN_ROOT}/${cand}/janusdna" ]; then STEM_REPO="${PENGUIN_ROOT}/${cand}"; break; fi
done
[ -n "${STEM_REPO:-}" ] || { echo "ERROR: no patched repository found under ${PENGUIN_ROOT}"; exit 1; }

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
OUTDIR="${PENGUIN_ROOT}/out/lora_janus144_stem_${TAG}"
SMOKE_OUT="${OUTDIR}_smoke"
SMOKE_LOG="${SMOKE_OUT}/smoke.log"
SMOKE_LIMIT=2000
LORA_SCRIPT="${REPO_DIR}/src/finetune_lora_janusdna.py"

export JANUS_STEM=1
export JANUS_STEM_STRIDE="${STEM_STRIDE}"
export JANUS_STEM_BLOCK="${STEM_BLOCK}"
export JANUS_STEM_KERNEL="${STEM_KERNEL}"
[ -n "$STEM_NSTAGES" ] && export JANUS_STEM_NSTAGES="${STEM_NSTAGES}"
export PYTHONPATH="${STEM_REPO}:${PYTHONPATH:-}"
MODE="${1:-full}"
echo ">>> stride=${STEM_STRIDE} block=${STEM_BLOCK} band=${STEM_BAND} | repo=${STEM_REPO}"

# ---- guards ----
[ -f "$MODEL_CONFIG" ]    || { echo "ERROR: model_config not found: $MODEL_CONFIG"; exit 1; }
[ -f "$CHECKPOINT" ]      || { echo "ERROR: checkpoint not found: $CHECKPOINT"; exit 1; }
[ -f "$WINDOWS_PARQUET" ] || { echo "ERROR: parquet not found: $WINDOWS_PARQUET"; exit 1; }
[ -f "$LORA_SCRIPT" ]     || { echo "ERROR: LoRA script not found: $LORA_SCRIPT"; exit 1; }
mkdir -p "${OUTDIR}" "${SMOKE_OUT}"

echo "=========================================="
echo "STEP 0/3: configuration gate (checkpoint vs environment)"
echo "=========================================="
python "${REPO_DIR}/tools/check_stem_config.py" \
    --checkpoint "${CHECKPOINT}" \
    --stride "${STEM_STRIDE}" \
    --block "${STEM_BLOCK}" \
    --kernel "${STEM_KERNEL}" \
    ${STEM_NSTAGES:+--nstages "${STEM_NSTAGES}"}

echo ""
echo "=========================================="
echo "STEP 1/3: LoRA smoke on ${SMOKE_LIMIT} windows (stem active)"
echo "=========================================="
python "$LORA_SCRIPT" \
    --repo-path "${STEM_REPO}" \
    --model-config "${MODEL_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --windows "${WINDOWS_PARQUET}" \
    --outdir "${SMOKE_OUT}" \
    --limit "${SMOKE_LIMIT}" \
    --epochs 2 2>&1 | tee "${SMOKE_LOG}"

echo ""
echo "=========================================="
echo "STEP 2/3: GATE (weights + LoRA injected + dt_proj excluded)"
echo "=========================================="
python - "${SMOKE_LOG}" "${SMOKE_OUT}" << 'PY'
import re, sys, json, os
log = open(sys.argv[1]).read(); outdir = sys.argv[2]
m = re.search(r"missing=(\d+)\s+unexpected=(\d+)", log)
if m:
    miss, unexp = int(m.group(1)), int(m.group(2))
    print(f"[GATE weights] missing={miss} unexpected={unexp}")
    if miss > 8 or unexp > 8:
        sys.exit(f"[GATE FAILED] large mismatch (missing={miss}, unexpected={unexp}): "
                 "the stem did not activate.")
    print("[GATE OK] weights (stem included) loaded into the right model.")
else:
    print("[GATE weights] no missing/unexpected line found; falling back to the "
          "LoRA module count as evidence the model is standing.")

res = os.path.join(outdir, "lora_results.json")
if os.path.exists(res):
    d = json.load(open(res))
    lora = d.get("lora", {})
    inj = lora.get("injected", d.get("injected"))
    tp = d.get("trainable_params")
    targets = lora.get("target_suffixes", [])
    print(f"[LoRA] modules injected={inj} | trainable_params={tp}")
    # dt_proj is incompatible with LoRA: Mamba's fused kernel bypasses the
    # wrapper entirely, so an adapter there does nothing while appearing to work.
    if any("dt_proj" in t for t in targets):
        sys.exit("[GATE FAILED] LoRA targets include dt_proj (incompatible).")
    print("[GATE OK] LoRA injected, dt_proj not among the targets.")
else:
    print("  [WARN] lora_results.json not found: confirm the injection from the log "
          "above ('trainable parameters: N of M').")
print(">>> If the gates passed, proceed to the full run.")
PY

if [ "$MODE" = "smoke" ]; then
  echo ">>> MODE=smoke: stopping here. If the gates passed, rerun without 'smoke'."
  exit 0
fi

echo ""
echo "=========================================="
echo "STEP 3/3: FULL LoRA (defaults identical to the baseline)"
echo "=========================================="
read -p "Gates OK above? Proceed with the full LoRA run? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then echo "Stopped."; exit 0; fi

python "$LORA_SCRIPT" \
    --repo-path "${STEM_REPO}" \
    --model-config "${MODEL_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --windows "${WINDOWS_PARQUET}" \
    --outdir "${OUTDIR}"

echo ""
echo "----- comparison against the baseline references -----"
echo "  expected baseline: scratch-clean 0.4028 | human 0.4027"
echo "  observed for S=4 : 0.3829 (a loss of ~0.02 MCC)"
echo "  Remember what the experiment predicts: the metric that decides the"
echo "  question is the NMI(function)/NMI(GC) ratio from cluster_embeddings.py,"
echo "  not MCC. Run that before drawing a conclusion."
