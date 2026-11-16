#!/usr/bin/env bash
# Runs verify_leakage_map.py twice with a fixed seed:
#   1) correct band (BAND = BLOCK/STRIDE = 4)  -> must show NO leakage
#   2) counter-proof (JANUS_STEM_BAND_OVERRIDE=1) -> MUST show leakage
# Needs a free GPU. The repository defaults to the v2 copy used for the step-1
# run, $PENGUIN_ROOT/JanusDNA_stem_s1_b4. It re-runs, with a fixed seed, the
# measurement behind the dependency-map table in the README (T = 256, random weights).
#
# Optional overrides:
#   REPO_PATH=/path/to/JanusDNA_stem_s1_b4  MODEL_CONFIG=/path/model_config.json  SEED=0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="${REPO_PATH:-${PENGUIN_ROOT:-$HOME/penguin_glm}/JanusDNA_stem_s1_b4}"
SEED="${SEED:-0}"
OUT="${OUT:-$SCRIPT_DIR/leakage_check_$(date +%Y%m%d_%H%M%S).log}"

[ -f "$SCRIPT_DIR/verify_leakage_map.py" ] || { echo "ERROR: verify_leakage_map.py not found in $SCRIPT_DIR"; exit 1; }
[ -d "$REPO_PATH/janusdna" ] || { echo "ERROR: $REPO_PATH/janusdna not found. Set REPO_PATH=..."; exit 1; }

if [ -z "${MODEL_CONFIG:-}" ]; then
  mapfile -t CANDIDATES < <(find "$REPO_PATH" "$(dirname "$REPO_PATH")" -maxdepth 6 \
                              -name model_config.json 2>/dev/null | sort -u)
  if [ "${#CANDIDATES[@]}" -eq 0 ]; then
    echo "ERROR: no model_config.json found. Set MODEL_CONFIG=..."; exit 1
  elif [ "${#CANDIDATES[@]}" -gt 1 ]; then
    echo "ERROR: several model_config.json found, pick one with MODEL_CONFIG=...:"
    printf '  %s\n' "${CANDIDATES[@]}"; exit 1
  fi
  MODEL_CONFIG="${CANDIDATES[0]}"
fi

export PYTHONPATH="$REPO_PATH:$SCRIPT_DIR:$SCRIPT_DIR/../env/patches:${PYTHONPATH:-}"

run() {  # $1 = label
  python3 - "$SEED" "$REPO_PATH" "$MODEL_CONFIG" <<'PY'
import sys, torch
seed, repo, cfg = int(sys.argv[1]), sys.argv[2], sys.argv[3]
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
import verify_leakage_map as v
sys.argv = ["verify_leakage_map.py", "--repo-path", repo, "--model-config", cfg,
            "--stride", "1", "--block", "4", "--kernel", "7", "--nstages", "3", "-T", "256"]
sys.exit(v.main())
PY
}

{
  echo "repo:   $REPO_PATH"
  echo "config: $MODEL_CONFIG"
  echo "seed:   $SEED"
  echo; echo "################ 1/2  CORRECT BAND ################"
  unset JANUS_STEM_BAND_OVERRIDE
  if run; then R1=ok; else R1=FAIL; fi
  echo; echo "################ 2/2  COUNTER-PROOF (BAND=1) ################"
  if JANUS_STEM_BAND_OVERRIDE=1 run; then R2=ok; else R2=FAIL; fi
  echo; echo "SUMMARY: correct band -> $R1 | counter-proof -> $R2"
  [ "$R1" = ok ] && [ "$R2" = ok ] && echo "Both as expected: the table is reproduced." \
                                   || echo "Something is off: check the output above."
} 2>&1 | tee "$OUT"

echo "Log saved to $OUT"
