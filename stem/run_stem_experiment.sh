#!/bin/bash
# ============================================================================
# Conv-stem experiment on JanusDNA-144 — FULLY ISOLATED from the baseline.
#
# VERSION 2: STRIDE and BLOCK are DECOUPLED.
#   STEM_STRIDE : grid compression   (1 = no compression)
#   STEM_BLOCK  : bases predicted by each fused position
#   BAND = BLOCK / STRIDE  -> width of the self-exclusion in grid cells,
#                             i.e. the constant hardcoded to 1 upstream.
#
#   previous run (reproduction):  STEM_STRIDE=4 STEM_BLOCK=4  -> BAND=1
#   new run (step 1)          :  STEM_STRIDE=1 STEM_BLOCK=4  -> BAND=4
#
# WHY THREE MORE PATCHES THAN v1
#   The band constant "1" does not live in the mask alone. It sits in two
#   places, and patching the mask by itself has no effect:
#     - create_janus_mask            -> which backward keys are visible
#     - final_fusion_repos_formasked -> WHICH backward slot is summed:
#         fused[p] = hidden[p-1] + hidden[mid_len + p + BAND]
#       Leave this one unpatched and the fusion keeps picking slot p+1, so
#       bases p+1..p+BLOCK-1 stay visible regardless of the mask.
#   On top of that the upstream loss is a plain unshifted CE: with the block
#   head the target must be realigned to the flattened predictions.
#
# USAGE:
#   ./run_stem_experiment.sh                                   # smoke, default
#   ./run_stem_experiment.sh full                              # real run
#   STEM_STRIDE=1 STEM_BLOCK=4 ./run_stem_experiment.sh full   # step-1 run
#   STEM_STRIDE=4 STEM_BLOCK=4 ./run_stem_experiment.sh full   # NON-REGRESSION
# ============================================================================
set -euo pipefail

# ======================= PATHS (CHECK THESE) =================================
SRC_REPO="${SRC_REPO:-/workspace/JanusDNA}"            # baseline, NEVER touched
STAGING="${STAGING:-/workspace/penguin_glm}"                # where the two .py files live
OUT_ROOT="${OUT_ROOT:-/workspace/penguin_glm/janusdna_stem/pretrain_scratch}"
FASTA="${FASTA:-/workspace/penguin_glm/data/penguin/GCF_965638725.1.fna}"
BED="${BED:-/workspace/penguin_glm/data/penguin/penguin_pretrain_clean.bed}"
CONDA_ENV="${CONDA_ENV:-janusdna}"
# =============================================================================

# ---- stem knobs ----
STEM_STRIDE="${STEM_STRIDE:-4}"
STEM_BLOCK="${STEM_BLOCK:-4}"
STEM_KERNEL="${STEM_KERNEL:-7}"
STEM_NSTAGES="${STEM_NSTAGES:-}"          # stride=1 only; empty = default (2)

# one copy per configuration: parallel runs do not collide
STEM_REPO="${STEM_REPO:-${STAGING}/JanusDNA_stem_s${STEM_STRIDE}_b${STEM_BLOCK}}"

MODE="${1:-smoke}"

if [ $(( STEM_BLOCK % STEM_STRIDE )) -ne 0 ]; then
  echo "ERROR: STEM_BLOCK=$STEM_BLOCK is not a multiple of STEM_STRIDE=$STEM_STRIDE."
  echo "       The band is measured in grid cells: BAND = BLOCK / STRIDE."
  exit 1
fi
STEM_BAND=$(( STEM_BLOCK / STEM_STRIDE ))

# ---- safety guards: never confuse the copy with the baseline ----
[ "$STEM_REPO" != "$SRC_REPO" ] || { echo "ERROR: STEM_REPO == SRC_REPO"; exit 1; }
case "$STEM_REPO" in *_stem*|*JanusDNA_stem*) : ;; *) echo "ERROR: STEM_REPO does not contain '_stem', refusing for safety"; exit 1;; esac
[ -d "$SRC_REPO/janusdna" ] || { echo "ERROR: $SRC_REPO/janusdna does not exist"; exit 1; }
[ -f "$STAGING/causal_conv_stem.py" ] || { echo "ERROR: missing $STAGING/causal_conv_stem.py"; exit 1; }
[ -f "$STAGING/test_stem_wiring.py" ] || { echo "ERROR: missing $STAGING/test_stem_wiring.py"; exit 1; }
[ -f "$FASTA" ] || { echo "ERROR: FASTA not found: $FASTA"; exit 1; }
[ -f "$BED" ]   || { echo "ERROR: BED not found: $BED"; exit 1; }

echo ">>> [1/5] Copying repo (baseline untouched): $SRC_REPO -> $STEM_REPO"
if [ -d "$STEM_REPO" ]; then echo "    (removing previous copy $STEM_REPO)"; rm -rf "$STEM_REPO"; fi
mkdir -p "$(dirname "$STEM_REPO")"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude='.git' --exclude='*.ckpt' --exclude='wandb/' "$SRC_REPO/" "$STEM_REPO/"
else
  mkdir -p "$STEM_REPO"
  cp -a "$SRC_REPO/." "$STEM_REPO/"
  rm -rf "$STEM_REPO/.git" "$STEM_REPO/wandb" 2>/dev/null || true
  find "$STEM_REPO" -name '*.ckpt' -delete 2>/dev/null || true
fi

echo ">>> [2/5] Placing the .py files into the copy"
cp "$STAGING/causal_conv_stem.py" "$STEM_REPO/janusdna/causal_conv_stem.py"
cp "$STAGING/test_stem_wiring.py" "$STEM_REPO/test_stem_wiring.py"

echo ">>> [3/5] Patching the model (patch-or-die, 6 patches + 2 on train.py)"
python3 - "$STEM_REPO" << 'PY'
import sys, os
repo = sys.argv[1]
mod = os.path.join(repo, "janusdna", "modeling_janusdna.py")
with open(mod) as f: src = f.read()
orig = src

def _find_unique(c, a):
    n = c.count(a)
    if n == 0: sys.exit(f"[PATCH FAILED] anchor NOT found: {a!r}")
    if n > 1:  sys.exit(f"[PATCH FAILED] anchor not unique ({n}x): {a!r}")
    return c.index(a)

def insert_after(c, a, lines):
    i = _find_unique(c, a); ls = c.rfind("\n",0,i)+1; ind = c[ls:i]; eol = c.find("\n",i)
    return c[:eol] + "".join("\n"+(ind+l if l else "") for l in lines) + c[eol:]

def replace_line(c, a, lines):
    i = _find_unique(c, a); ls = c.rfind("\n",0,i)+1; ind = c[ls:i]; eol = c.find("\n",i)
    return c[:ls] + "\n".join((ind+l if l else "") for l in lines) + c[eol:]

def replace_text(c, a, b):
    _find_unique(c, a)
    return c.replace(a, b)

# ---------------------------------------------------------------- 1) constants
src = insert_after(src, "from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss", [
 "", "# === conv-stem: injected for the isolated experiment (enabled via env) ===",
 "import os as _stem_os",
 '_USE_STEM = _stem_os.environ.get("JANUS_STEM", "0") == "1"',
 '_STEM_S = int(_stem_os.environ.get("JANUS_STEM_STRIDE", "4"))',
 '_STEM_BLOCK = int(_stem_os.environ.get("JANUS_STEM_BLOCK", str(_STEM_S)))',
 '_STEM_K = int(_stem_os.environ.get("JANUS_STEM_KERNEL", "7"))',
 '_STEM_NSTAGES = _stem_os.environ.get("JANUS_STEM_NSTAGES", "") or None',
 "if _STEM_NSTAGES is not None: _STEM_NSTAGES = int(_STEM_NSTAGES)",
 "if _USE_STEM:",
 '    assert _STEM_BLOCK % _STEM_S == 0, ("JANUS_STEM_BLOCK must be a multiple of "',
 '        "JANUS_STEM_STRIDE: the band is measured in grid cells.")',
 "# Exclusion band IN GRID CELLS. Without the stem it stays 1 = upstream behaviour.",
 "_STEM_BAND = (_STEM_BLOCK // _STEM_S) if _USE_STEM else 1",
 '_BAND_OVR = _stem_os.environ.get("JANUS_STEM_BAND_OVERRIDE", "")',
 "if _BAND_OVR:",
 "    # Counter-proof ONLY: force the wrong band. The gate MUST fail.",
 "    _STEM_BAND = int(_BAND_OVR)",
 "# ========================================================================="])

# ---------------------------------------------- 2) stem in JanusDNAModel.__init__
src = insert_after(src, "self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)", [
 "# conv-stem (isolated experiment)",
 "self.use_conv_stem = _USE_STEM",
 "if self.use_conv_stem:",
 '    assert not config.layer_fusion, "stem implemented for layer_fusion=False only"',
 '    assert getattr(config, "layer_fusion_strategy", None) == "pool", (',
 '        "the stem assumes final_fusion_repos_formasked (layer_fusion_strategy=\'pool\'). "',
 '        "With another strategy the band does not go through the fusion and "',
 '        "patches 5/6 have no effect: refusing to start.")',
 "    from .causal_conv_stem import CausalConvStem",
 "    self.stem = CausalConvStem(config.hidden_size, stride=_STEM_S,",
 "                               kernel_size=_STEM_K, n_stages=_STEM_NSTAGES)",
 "    print(f'[stem] stride={_STEM_S} block={_STEM_BLOCK} band={_STEM_BAND} '",
 "          f'kernel={_STEM_K} RF={self.stem.receptive_field}bp')"])

# ------------------------------- 3) forward: stem on both branches, tie preserved
src = replace_line(src, "hidden_states = inputs_embeds if self.layer_fusion else torch.concat([inputs_embeds, inputs_embeds.flip(dims=(1,))], dim=1)", [
 "if getattr(self, 'use_conv_stem', False):",
 "    _fwd = self.stem(inputs_embeds)",
 "    _bwd = self.stem(inputs_embeds.flip(dims=(1,)))",
 "    hidden_states = torch.concat([_fwd, _bwd], dim=1)",
 "else:",
 "    hidden_states = inputs_embeds if self.layer_fusion else torch.concat([inputs_embeds, inputs_embeds.flip(dims=(1,))], dim=1)"])

# --------------------------------------------------- 4) BlockHead replaces lm_head
src = replace_line(src, "self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)", [
 "if _USE_STEM:",
 "    from .causal_conv_stem import BlockHead",
 "    self.lm_head = BlockHead(config.hidden_size, _STEM_BLOCK, config.vocab_size)",
 "    self._tied_weights_keys = []",
 "    config.tie_word_embeddings = False",
 "else:",
 "    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)"])

# ------------------------------------- 5) FlexAttention mask: parametric band
#   fused[p] = fwd[p-1] + bwd[p+BAND]  ->  gap in original coordinates = BAND+1
#   (with BAND=1 this is 2, i.e. the upstream "2 * 1": exact non-regression)
src = replace_text(src,
 "(kv_idx >= orig_seq_len) & (q_idx < orig_seq_len) & (kv_idx >= orig_seq_len + q_idx + 2 * 1)",
 "(kv_idx >= orig_seq_len) & (q_idx < orig_seq_len) & (kv_idx >= orig_seq_len + q_idx + (_STEM_BAND + 1))")
src = replace_text(src,
 "(kv_idx < orig_seq_len) & (q_idx >= orig_seq_len) & (q_idx >= kv_idx + orig_seq_len + 2 * 1)",
 "(kv_idx < orig_seq_len) & (q_idx >= orig_seq_len) & (q_idx >= kv_idx + orig_seq_len + (_STEM_BAND + 1))")

# ------------------------------------------- 6) final fusion: parametric band
#   THIS is where what the fused position sees is actually decided.
src = replace_text(src,
 'assert mid_len > 2 * 1, "sequence length must be larger than 2 * 1 for JanusDNA final fusion."',
 'assert mid_len > 2 * _STEM_BAND, "sequence length must be larger than 2 * band for JanusDNA final fusion."')
src = replace_text(src,
 "fused_hidden_states[:, head_indices, :] = hidden_states[:, mid_len + 1 + head_indices, :]",
 "fused_hidden_states[:, head_indices, :] = hidden_states[:, mid_len + _STEM_BAND + head_indices, :]")
src = replace_text(src,
 "middle_indices = torch.arange(mid_len - (2 * 1), device=hidden_states.device)",
 "middle_indices = torch.arange(mid_len - (_STEM_BAND + 1), device=hidden_states.device)")
src = replace_text(src,
 "fused_hidden_states[:, 1 + middle_indices, :] = hidden_states[:, middle_indices, :] + hidden_states[:, mid_len + 1 + middle_indices + 1, :]",
 "fused_hidden_states[:, 1 + middle_indices, :] = hidden_states[:, middle_indices, :] + hidden_states[:, mid_len + 1 + middle_indices + _STEM_BAND, :]")
src = replace_text(src,
 "tail_indices = torch.arange(1, device=hidden_states.device)",
 "tail_indices = torch.arange(_STEM_BAND, device=hidden_states.device)")

if src == orig: sys.exit("[PATCH FAILED] no change applied")
with open(mod,"w") as f: f.write(src)
print("[OK] 6 patches applied to", mod)

# ---------------------------------------- full-model gate in train.py (env-gated)
tr = os.path.join(repo, "train.py")
with open(tr) as f: tsrc = f.read()
torig = tsrc
if "def on_fit_start" in tsrc:
    sys.exit("[PATCH FAILED] on_fit_start already present in train.py: handle manually")
# 7) TARGET realignment, at the ONE point where x and y meet.
#    BlockHead returns M*BLOCK predictions per window, with base
#    m*STRIDE + j at flat position m*BLOCK + j: here we build the matching
#    target and truncate positions whose block would run past the end.
#    y is NOT shifted (seq[:-1]/seq[1:] are commented out in hg38_dataset.py).
#    B and T are read from `batch` (input_ids), NOT inferred from x/y shapes:
#    the harness flattens the batch on both logits and target, and V is not
#    config.vocab_size but the embedding's actual vocabulary.
#    With STRIDE == BLOCK this is the exact identity.
tsrc = insert_after(tsrc, "x, y, w, moe_loss = self.forward(batch)", [
 "# === conv-stem: block target realignment ===",
 "import os as _stem_os",
 'if _stem_os.environ.get("JANUS_STEM", "0") == "1":',
 '    _s = int(_stem_os.environ.get("JANUS_STEM_STRIDE", "4"))',
 '    _bl = int(_stem_os.environ.get("JANUS_STEM_BLOCK", str(_s)))',
 "    _b0 = batch[0] if isinstance(batch, (tuple, list)) and len(batch) else batch",
 "    if not getattr(self, '_stem_align_logged', False):",
 "        self._stem_align_logged = True",
 "        self._stem_align_show = True",
 "        _dx = tuple(x.shape) if torch.is_tensor(x) else type(x).__name__",
 "        _db = tuple(_b0.shape) if torch.is_tensor(_b0) else type(_b0).__name__",
 "        print(f'[stem-align] IN  batch0={_db} x={_dx} y={tuple(y.shape)} "
 "s={_s} bl={_bl}', flush=True)",
 "    if not (torch.is_tensor(_b0) and _b0.dim() == 2):",
 "        raise RuntimeError(f'[stem-align] unexpected batch[0]: cannot "
 "derive B and T. See the IN line above.')",
 "    _B, _T = int(_b0.size(0)), int(_b0.size(1))",
 "    _N = (_T // _s) * _bl",
 "    _Mok = (_T - _bl) // _s + 1",
 "    _K = _Mok * _bl",
 "    def _align(_x, _y):",
 "        _xd, _yd, _V = _x.dim(), _y.dim(), int(_x.shape[-1])",
 "        if _x.numel() != _B * _N * _V:",
 "            raise RuntimeError(f'[stem-align] x={tuple(_x.shape)} has {_x.numel()} '",
 "                               f'elements, expected B*M*BLOCK*V={_B*_N*_V} '",
 "                               f'(B={_B} T={_T} M={_T//_s} BLOCK={_bl} V={_V})')",
 "        _x = _x.reshape(_B, _N, _V)",
 "        _y2 = _y.reshape(_B, _T)",
 "        _ix = (torch.arange(_Mok, device=_y.device).unsqueeze(1) * _s",
 "               + torch.arange(_bl, device=_y.device).unsqueeze(0)).reshape(-1)",
 "        _x, _y2 = _x[:, :_K], _y2[:, _ix]",
 "        if _xd == 2:",
 "            _x = _x.reshape(_B * _K, _V)",
 "        if _yd == 1:",
 "            _y2 = _y2.reshape(-1)",
 "        return _x, _y2",
 "    _hit, _yy = 0, y",
 "    if torch.is_tensor(x):",
 "        x, _yy = _align(x, y)",
 "        _hit = 1",
 "    elif isinstance(x, (tuple, list)):",
 "        _new = []",
 "        for _xi in x:",
 "            if torch.is_tensor(_xi):",
 "                _xi, _yy = _align(_xi, y)",
 "                _hit += 1",
 "            _new.append(_xi)",
 "        if _hit:",
 "            x = _new",
 "    y = _yy",
 "    if getattr(self, '_stem_align_show', False):",
 "        self._stem_align_show = False",
 "        _dx = tuple(x.shape) if torch.is_tensor(x) else type(x).__name__",
 "        print(f'[stem-align] OUT x={_dx} y={tuple(y.shape)} hit={_hit}', flush=True)",
 "    if _hit == 0:",
 "        raise RuntimeError(f'[stem-align] no tensor in x (type={type(x).__name__})')",
 "# ================================================",
])

tsrc = insert_after(tsrc, "class SequenceLightningModule(pl.LightningModule):", [
 "",
 "    def on_fit_start(self):",
 "        import os as _stem_os",
 '        if _stem_os.environ.get("JANUS_STEM_GATE", "0") == "1":',
 "            from test_stem_wiring import run_gate",
 '            _s = int(_stem_os.environ.get("JANUS_STEM_STRIDE", "4"))',
 '            _b = int(_stem_os.environ.get("JANUS_STEM_BLOCK", str(_s)))',
 "            run_gate(self.model, stride=_s, block=_b, device=self.device)",
])
if tsrc == torig: sys.exit("[PATCH FAILED] train.py not modified")
with open(tr,"w") as f: f.write(tsrc)
print("[OK] block realignment + full-model gate injected into", tr)
PY

echo ">>> [4/5] Pre-launch checks"
python3 -m py_compile "$STEM_REPO/janusdna/modeling_janusdna.py" && echo "[OK] patched modeling_janusdna.py compiles"
python3 -m py_compile "$STEM_REPO/train.py" && echo "[OK] patched train.py compiles"

if ! python -c "import torch" >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    source /etc/profile.d/conda.sh 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || true
  fi
fi

echo "    -- standalone stem gate --"
python "$STEM_REPO/janusdna/causal_conv_stem.py" || { echo "[ABORT] standalone gate failed"; exit 1; }

export PYTHONPATH="$STEM_REPO:${PYTHONPATH:-}"
JANUS_STEM=1 JANUS_STEM_STRIDE="$STEM_STRIDE" JANUS_STEM_BLOCK="$STEM_BLOCK" \
python - "$STEM_REPO" << 'PY'
import sys, os, janusdna, janusdna.modeling_janusdna as m
want = sys.argv[1]
got = os.path.dirname(janusdna.__file__)
if not got.startswith(want):
    sys.exit(f"[ABORT] 'janusdna' importato da {got}, NON da {want}. "
             f"Likely an editable pip install of the baseline: the patch would be ignored.")
assert hasattr(m, "_USE_STEM"), "[ABORT] patch not loaded: _USE_STEM missing"
assert hasattr(m, "_STEM_BAND"), "[ABORT] v2 patch not loaded: _STEM_BAND missing"
print(f"[OK] import verified: janusdna from {got} | band={m._STEM_BAND}")
PY

echo ">>> [5/5] Launching training (MODE=$MODE, stride=$STEM_STRIDE block=$STEM_BLOCK band=$STEM_BAND)"

export JANUS_STEM=1
export JANUS_STEM_STRIDE="${STEM_STRIDE}"
export JANUS_STEM_BLOCK="${STEM_BLOCK}"
export JANUS_STEM_KERNEL="${STEM_KERNEL}"
[ -n "$STEM_NSTAGES" ] && export JANUS_STEM_NSTAGES="${STEM_NSTAGES}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$OUT_ROOT"
export JANUS_STEM_GATE_LOG="$OUT_ROOT/GATE_RESULT_s${STEM_STRIDE}_b${STEM_BLOCK}.txt"

if [ "$MODE" = "smoke" ]; then
  SEQLEN=1024;   MAX_STEPS=50;    GLOBAL_BATCH=8; PER_GPU_BATCH=1
  echo "    SMOKE: seqlen=$SEQLEN steps=$MAX_STEPS"
else
  SEQLEN="${SEQLEN:-131072}"; MAX_STEPS="${MAX_STEPS:-50000}"; GLOBAL_BATCH="${GLOBAL_BATCH:-8}"; PER_GPU_BATCH="${PER_GPU_BATCH:-1}"
  echo "    FULL: seqlen=$SEQLEN steps=$MAX_STEPS"
fi
LR="${LR:-8e-3}"; NUM_WORKERS="${NUM_WORKERS:-4}"

# MEMORY NOTE: with STEM_STRIDE=1 the backbone works on SEQLEN positions rather
# than SEQLEN/4 -> ~4x activations. On OOM, reduce PER_GPU_BATCH and compensate
# with accumulation to keep GLOBAL_BATCH IDENTICAL to previous runs, otherwise
# the iso-budget comparison breaks.

D_MODEL=144; FLEX_ATTN_MODEL=256; INTER_FFN_MODEL=576; N_LAYER=8
ROUTER_AUX_LOSS_COEF=0.2; OUTPUT_ROUTER_LOGITS="true"
BIDIRECTIONAL_WEIGHT_TIE="true"; BIDIRECTIONAL_ATTN_TIE="false"; RC_AUG="false"

RUN_NAME="janusdna_stem_s${STEM_STRIDE}_b${STEM_BLOCK}_penguin_len-$((SEQLEN/1024))k_lr-${LR}_step-${MAX_STEPS}"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"

NO_CKPT="${RUN_DIR}/__no_resume__.ckpt"
if [ "$MODE" = "smoke" ]; then
  [ -d "$RUN_DIR" ] && { echo "    (cleaning previous smoke dir)"; rm -rf "$RUN_DIR"; }
  CKPT_ARG="train.ckpt=${NO_CKPT}"
  echo "    resume: DISABLED (smoke starts from scratch)"
else
  LAST="${RUN_DIR}/checkpoints/last.ckpt"
  if [ -f "$LAST" ]; then
    CKPT_ARG="train.ckpt=${LAST}"; echo "    resume: from ${LAST} (crash recovery)"
  else
    CKPT_ARG="train.ckpt=${NO_CKPT}"; echo "    resume: no checkpoint found, starting from scratch"
  fi
fi
mkdir -p "${RUN_DIR}"

cd "${STEM_REPO}"
python -m train \
  experiment=hg38/hg38 \
  model="janusdna" \
  dataset.fasta_file="${FASTA}" \
  dataset.bed_file="${BED}" \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=${PER_GPU_BATCH} \
  dataset.batch_size_eval=${PER_GPU_BATCH} \
  dataset.mlm=False \
  dataset.rc_aug="${RC_AUG}" \
  dataset.add_eos=false \
  loader.num_workers=${NUM_WORKERS} \
  +model.config.output_router_logits="${OUTPUT_ROUTER_LOGITS}" \
  +model.config.router_aux_loss_coef="${ROUTER_AUX_LOSS_COEF}" \
  model.config.bidirectional_weight_tie="${BIDIRECTIONAL_WEIGHT_TIE}" \
  model.config.bidirectional_attn_tie="${BIDIRECTIONAL_ATTN_TIE}" \
  model.config.num_hidden_layers=${N_LAYER} \
  model.config.hidden_size=${D_MODEL} \
  model.config.flex_attn_n_embd=${FLEX_ATTN_MODEL} \
  +model.config.intermediate_size=${INTER_FFN_MODEL} \
  model.config.expert_layer_period=2 \
  model.config.expert_layer_offset=1 \
  model.config.intermediate_factor=4 \
  model.config.num_attention_heads=4 \
  model.config.attn_implementation="flash_attention_2" \
  model.config.attn_layer_period=8 \
  model.config.attn_layer_offset=100 \
  optimizer.lr="${LR}" \
  train.global_batch_size=${GLOBAL_BATCH} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  +trainer.val_check_interval=1.0 \
  wandb.mode=offline \
  wandb.name="${RUN_NAME}" \
  hydra.run.dir="${RUN_DIR}" \
  ${CKPT_ARG} \
  2>&1 | tee -a "${RUN_DIR}/train.log"

echo ""
echo ">>> DONE. Isolated output in: ${RUN_DIR}"
echo ">>> DOWNSTREAM: export the SAME variables before extraction and LoRA, and point"
echo "    --repo-path at ${STEM_REPO} (NON al baseline), otherwise you extract from a"
echo "    model with no stem and band 1."
