#!/bin/bash
# =============================================================================
# run_lora_saturated.sh: re-run the JanusDNA LoRA adaptations with an extended
# epoch budget, so that every model is compared at saturation rather than at a
# fixed budget.
#
# WHY
#   The first LoRA runs stopped at 10 epochs, and no model was saturated there:
#     - the human-pretrained model, continued for 9 more epochs, gained +0.018
#       MCC (0.3848 -> 0.4027). That 19-epoch value is not comparable with the
#       10-epoch values of the other models;
#     - the from-scratch baseline reaches 0.4096 with a 25-epoch budget against
#       0.4028 at 10, and at 10 epochs its validation maximum fell on the last
#       epoch available.
#   Margins between models that have not converged are not defensible.
#
# WHAT IT DOES
#   Re-runs the baseline, the stems and the human checkpoint with --epochs 25
#   and lets early stopping on validation MCC decide when to stop. Everything
#   else is unchanged: same LoRA hyperparameters, same windows, same split.
#   Outputs go to $PENGUIN_ROOT/out/lora_<name>_ep<EPOCHS>/.
#
# USAGE (from anywhere; the script finds the repository from its own path)
#   bash src/run_lora_saturated.sh             # every configuration
#   bash src/run_lora_saturated.sh step1       # one or a few, by name
#   EPOCHS=40 bash src/run_lora_saturated.sh   # different budget
#   DRY=1 bash src/run_lora_saturated.sh       # print the commands, run nothing
#
# PATHS
#   PENGUIN_ROOT    workspace (default $HOME/penguin_glm)
#   JANUSDNA_REPO   unpatched upstream JanusDNA, used for baseline and human
#   STEM_REPO_V2    a v2-patched copy (stem/run_stem_experiment.sh), used when the
#                   per-scale copy of a stem configuration does not exist. With
#                   STRIDE = BLOCK the v2 patch reproduces v1 (stem/verify_patches.py)
#   HUMAN_RUN       run directory of the released human checkpoint, holding
#                   model_config.json and checkpoints/last.ckpt
# =============================================================================
set -u

: "${PENGUIN_ROOT:=$HOME/penguin_glm}"
: "${JANUSDNA_REPO:=$PENGUIN_ROOT/JanusDNA}"
: "${STEM_REPO_V2:=$PENGUIN_ROOT/JanusDNA_stem_s4_b4}"
: "${HUMAN_RUN:=$JANUSDNA_REPO/imported_checkpoint/extracted_finalmlp_144/janusdna_len-131k_d_model-144_inter_dim-576_n_layer-8_lr-8e-3_step-50K_moeloss-true_1head_onlymoe_finalmlp}"
: "${EPOCHS:=25}"
: "${DRY:=0}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
E="${PENGUIN_ROOT}/out"
W="${WINDOWS:-$E/windows.parquet}"
LORA="${REPO_DIR}/src/finetune_lora_janusdna.py"
PS="${PENGUIN_ROOT}/janusdna_stem/pretrain_scratch"

[ -f "$LORA" ] || { echo "ERROR: $LORA not found"; exit 1; }
[ -f "$W" ]    || { echo "ERROR: $W not found (set WINDOWS=...)"; exit 1; }

# name | repository to import | run directory | stem variables ("-" = none)
CONFIGS=(
"baseline|${JANUSDNA_REPO}|${PENGUIN_ROOT}/janusdna/pretrain_scratch/janusdna_scratch_penguin_clean_len-128k_lr-8e-3_step-15000|-"
"human|${JANUSDNA_REPO}|${HUMAN_RUN}|-"
"s4|${PENGUIN_ROOT}/JanusDNA_stem_s4_b4|${PS}/janusdna_stem_s4_penguin_len-128k_lr-8e-3_step-50000|STRIDE=4 BLOCK=4 NSTAGES="
"s8|${PENGUIN_ROOT}/JanusDNA_stem_s8|${PS}/janusdna_stem_s8_penguin_len-128k_lr-8e-3_step-50000|STRIDE=8 BLOCK=8 NSTAGES="
"s16|${PENGUIN_ROOT}/JanusDNA_stem_s16|${PS}/janusdna_stem_s16_penguin_len-128k_lr-8e-3_step-50000|STRIDE=16 BLOCK=16 NSTAGES="
"step1|${PENGUIN_ROOT}/JanusDNA_stem_s1_b4|${PS}/janusdna_stem_s1_b4_penguin_len-128k_lr-8e-3_step-50000|STRIDE=1 BLOCK=4 NSTAGES=3"
)
# The fourth cell of the step-1 design (stride 1, 19 bp, single-base head) was
# not run in this work. If it is, add:
# "s1b1|${PENGUIN_ROOT}/JanusDNA_stem_s1_b1|${PS}/janusdna_stem_s1_b1_penguin_len-128k_lr-8e-3_step-50000|STRIDE=1 BLOCK=1 NSTAGES=3"

WANT=("$@")
run_wanted() { [ ${#WANT[@]} -eq 0 ] && return 0; for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

echo "=================================================================="
echo "LoRA to saturation: budget $EPOCHS epochs, early stopping on val MCC"
echo "=================================================================="

FAILED=()
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME REPO RUN STEM <<< "$cfg"
  run_wanted "$NAME" || continue

  OUT="$E/lora_${NAME}_ep${EPOCHS}"
  MC="$RUN/model_config.json"
  CK="$RUN/checkpoints/last.ckpt"

  echo ""
  echo "=== $NAME  ->  $OUT"
  if [ ! -f "$MC" ] || [ ! -f "$CK" ]; then
    echo "  SKIPPED: model_config.json or last.ckpt missing in $RUN"
    FAILED+=("$NAME (missing files)"); continue
  fi
  if [ "$STEM" != "-" ] && [ ! -d "$REPO/janusdna" ] && [ -d "$STEM_REPO_V2/janusdna" ]; then
    echo "  note: $REPO not found, using the v2 copy $STEM_REPO_V2"
    REPO="$STEM_REPO_V2"
  fi
  if [ ! -d "$REPO/janusdna" ]; then
    echo "  SKIPPED: repository not found: $REPO"
    FAILED+=("$NAME (missing repository)"); continue
  fi

  # Stem variables are set only where they apply, and explicitly removed
  # everywhere else so that nothing is inherited from the calling shell.
  ENVV=(env -u JANUS_STEM -u JANUS_STEM_STRIDE -u JANUS_STEM_BLOCK
        -u JANUS_STEM_KERNEL -u JANUS_STEM_NSTAGES
        "PYTHONPATH=${REPO}:${REPO_DIR}/src")
  if [ "$STEM" != "-" ]; then
    eval "$STEM"
    ENVV+=(JANUS_STEM=1 JANUS_STEM_STRIDE="$STRIDE" JANUS_STEM_BLOCK="$BLOCK" JANUS_STEM_KERNEL=7)
    [ -n "$NSTAGES" ] && ENVV+=(JANUS_STEM_NSTAGES="$NSTAGES")
    echo "  stem: stride=$STRIDE block=$BLOCK band=$((BLOCK/STRIDE)) nstages=${NSTAGES:-default}"

    # gate: the environment must match what the checkpoint was trained with
    if [ -f "$REPO_DIR/tools/check_stem_config.py" ]; then
      python "$REPO_DIR/tools/check_stem_config.py" --checkpoint "$CK" \
        --stride "$STRIDE" --block "$BLOCK" --kernel 7 \
        ${NSTAGES:+--nstages "$NSTAGES"} || {
          echo "  ABORTED: environment does not match the checkpoint"
          FAILED+=("$NAME (config gate)"); continue; }
    fi
  else
    echo "  no stem"
  fi

  CMD=("${ENVV[@]}" python "$LORA" --repo-path "$REPO" --model-config "$MC"
       --checkpoint "$CK" --windows "$W" --outdir "$OUT" --epochs "$EPOCHS")

  if [ "$DRY" = "1" ]; then printf '  %q' "${CMD[@]}"; echo; continue; fi
  mkdir -p "$OUT"
  set -o pipefail
  "${CMD[@]}" 2>&1 | tee "$OUT/run.log" || FAILED+=("$NAME (run failed)")
  set +o pipefail
done

[ "$DRY" = "1" ] && exit 0

echo ""
echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
python3 - "$E" "$EPOCHS" << 'PY'
import json, sys, os
E, ep = sys.argv[1], sys.argv[2]
print(f"{'model':12}{'MCC':>9}{'acc':>9}{'best/ep':>10}   saturated?")
print("-" * 56)
for n in ("baseline", "human", "s4", "s8", "s16", "step1", "s1b1"):
    f = f"{E}/lora_{n}_ep{ep}/lora_results.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); h = d.get("history", [])
    best = max(range(len(h)), key=lambda i: h[i]["val_mcc"]) + 1 if h else 0
    sat = "yes" if best < len(h) - 1 else "NO, still rising"
    print(f"{n:12}{d['test_mcc']:9.4f}{d['test_accuracy']:9.4f}{f'{best}/{len(h)}':>10}   {sat}")
print("\nA run is saturated when its validation maximum does NOT fall on the last epochs.")
print("If one is still rising, run it again with a larger EPOCHS.")
PY

if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""; echo "NOT COMPLETED:"; printf '  - %s\n' "${FAILED[@]}"
fi
