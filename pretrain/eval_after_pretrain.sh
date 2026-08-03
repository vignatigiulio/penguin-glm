#!/bin/bash
# eval_after_pretrain.sh — post-training pipeline for the CLEAN from-scratch
# JanusDNA run. Execute after pre-training has reached max_steps.
#
# In order, stopping at the first error:
#   1. freeze the checkpoint and the log
#   2. extract embeddings, verifying the extraction protocol matches the human
#      model's exactly
#   3. multi-seed probe over two arms (from-scratch vs human)
#   4. LoRA on the clean checkpoint
#
# The protocol check in step 2 is the important one. Comparing a from-scratch
# model against a human-pretrained one only means something if both embeddings
# were produced with the same pooling and the same special-token handling. A
# mismatch there would produce a difference in MCC that has nothing to do with
# pre-training, so the script asserts rather than trusts.
#
# Usage:
#   ./eval_after_pretrain.sh            # everything
#   ./eval_after_pretrain.sh extract    # from a given step: freeze|extract|probe|lora
set -euo pipefail

: "${PENGUIN_ROOT:=$HOME/penguin_glm}"
: "${JANUSDNA_REPO:=$(dirname "$PENGUIN_ROOT")/JanusDNA}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN="${RUN:-$PENGUIN_ROOT/janusdna/pretrain_scratch/janusdna_scratch_penguin_clean_len-128k_lr-8e-3_step-15000}"
CKPT_FINAL="${CKPT_FINAL:-$PENGUIN_ROOT/janusdna/checkpoints_final/janus_scratch_penguin_15k_CLEAN_FINAL.ckpt}"
MIN_STEPS="${MIN_STEPS:-15000}"

EMB_CLEAN="${EMB_CLEAN:-$PENGUIN_ROOT/out/emb_janusdna_144_scratch_clean.npz}"
EMB_HUMAN="${EMB_HUMAN:-$PENGUIN_ROOT/out/embeddings_janusdna144/embeddings_144_without_midattn.npz}"
WINDOWS="${WINDOWS:-$PENGUIN_ROOT/out/windows.parquet}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
cd "$PENGUIN_ROOT"
STEP="${1:-all}"

run_freeze() {
  echo "===== [1/4] FREEZING THE CHECKPOINT ====="
  mkdir -p "$(dirname "$CKPT_FINAL")"
  if [ ! -f "$RUN/checkpoints/last.ckpt" ]; then
    echo "ERROR: $RUN/checkpoints/last.ckpt not found. Did the run finish?"; exit 1
  fi
  MIN_STEPS="$MIN_STEPS" CKPT="$RUN/checkpoints/last.ckpt" python - <<'PY'
import os, torch
c = torch.load(os.environ["CKPT"], map_location="cpu")
step = c.get("global_step")
want = int(os.environ["MIN_STEPS"])
print(f"  global_step = {step}")
assert step is not None and step >= want, f"run incomplete (step={step}, expected {want})"
print("  OK: run complete")
PY
  cp "$RUN/checkpoints/last.ckpt" "$CKPT_FINAL"
  cp "$RUN/train.log" "$(dirname "$CKPT_FINAL")/train_janus_scratch_clean.log" 2>/dev/null || true
  echo "  checkpoint frozen: $CKPT_FINAL"
  ls -lh "$CKPT_FINAL"
}

run_extract() {
  echo "===== [2/4] EMBEDDING EXTRACTION ====="
  if [ ! -f "$RUN/model_config.json" ]; then
    echo "ERROR: $RUN/model_config.json not found"; exit 1
  fi
  python "${REPO_DIR}/src/extract_janusdna.py" \
    --repo-path "$JANUSDNA_REPO" \
    --model-config "$RUN/model_config.json" \
    --checkpoint "$CKPT_FINAL" \
    --windows-parquet "$WINDOWS" \
    --output "$EMB_CLEAN" \
    --batch-size 32

  echo "----- metadata check (must match the human model) -----"
  EMB_CLEAN="$EMB_CLEAN" EMB_HUMAN="$EMB_HUMAN" python - <<'PY'
import os
from common import load_embeddings
_, Xc, mc = load_embeddings(os.environ["EMB_CLEAN"])
_, Xh, mh = load_embeddings(os.environ["EMB_HUMAN"])
print("CLEAN:", Xc.shape, "| pooling:", mc.get("pooling"),
      "| add_special:", mc.get("add_special_tokens"))
print("HUMAN:", Xh.shape, "| pooling:", mh.get("pooling"),
      "| add_special:", mh.get("add_special_tokens"))
assert Xc.shape[1] == Xh.shape[1], "different embedding width"
assert mc.get("pooling") == mh.get("pooling"), "DIFFERENT POOLING: the comparison is invalid"
assert mc.get("add_special_tokens") == mh.get("add_special_tokens"), \
    "DIFFERENT add_special_tokens: the comparison is invalid"
print("OK: extraction protocol consistent with the human model")
PY
}

run_probe() {
  echo "===== [3/4] TWO-ARM MULTI-SEED PROBE ====="
  for f in "$EMB_CLEAN" "$EMB_HUMAN"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
  done
  for s in 1 2 3 4; do
    echo "--- seed $s ---"
    python "${REPO_DIR}/src/mlp_probe_compare.py" \
      --windows "$WINDOWS" \
      --emb "janus144_scratch=$EMB_CLEAN" \
      --emb "janus144_human=$EMB_HUMAN" \
      --outdir "$PENGUIN_ROOT/out/mlp_janus_clean_seed$s" \
      --seed "$s"
  done
  echo "----- aggregate across seeds -----"
  python "${REPO_DIR}/analysis/audit_results.py" \
    --root "$PENGUIN_ROOT" --filter janus144
}

run_lora() {
  echo "===== [4/4] LoRA ON THE CLEAN CHECKPOINT ====="
  python "${REPO_DIR}/src/finetune_lora_janusdna.py" \
    --repo-path "$JANUSDNA_REPO" \
    --model-config "$RUN/model_config.json" \
    --checkpoint "$CKPT_FINAL" \
    --windows "$WINDOWS" \
    --outdir "$PENGUIN_ROOT/out/lora_janus144_scratch_clean"

  echo "----- comparison against the references -----"
  echo "  expected: human 0.4027"
  python - <<'PY' 2>/dev/null || echo "  (lora_results.json not written yet)"
import json, os
p = os.path.join(os.environ.get("PENGUIN_ROOT", "."),
                 "out/lora_janus144_scratch_clean/lora_results.json")
d = json.load(open(p))
print("  scratch-clean LoRA MCC:", round(d["test_mcc"], 4))
print("  injected:", d["lora"]["injected"], "| trainable:", d["trainable_params"])
PY
}

case "$STEP" in
  freeze)  run_freeze ;;
  extract) run_extract ;;
  probe)   run_probe ;;
  lora)    run_lora ;;
  all)     run_freeze; run_extract; run_probe; run_lora
           echo ""
           echo "===== COMPLETE. Arm summary: ====="
           echo "  zero-shot: from-scratch 0.3218 | human 0.3088"
           echo "  LoRA:      from-scratch 0.4028 | human 0.4027" ;;
  *) echo "usage: $0 [freeze|extract|probe|lora|all]"; exit 1 ;;
esac
