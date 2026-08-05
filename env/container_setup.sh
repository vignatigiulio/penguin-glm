#!/bin/bash
# container_setup.sh — re-apply the recurring environment fixes on entering a
# fresh container.
#
#   source env/container_setup.sh
#
# Must be sourced, not executed: conda activation has to affect the current
# shell. Containers here are run with --rm, so nothing outside the mounted
# workspace survives; every one of the fixes below has to be redone each time.
#
# Order matters. Each block exists because of a specific failure documented in
# the README's "Known issues" table.

set -u

: "${PENGUIN_ROOT:=$HOME/penguin_glm}"
# Environment names are case-sensitive and differ between machines: try the
# configured name first, then the common case variants, before giving up.
: "${CONDA_ENV:=CrossDNA}"
: "${CONDA_PREFIX_ROOT:=/opt/conda}"

export PENGUIN_ROOT
export HYDRA_FULL_ERROR=1

# --------------------------------------------------------------------------- #
# 1. conda
# --------------------------------------------------------------------------- #
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" != "base" ]; then
  echo ">> conda env already active: ${CONDA_DEFAULT_ENV} (leaving it alone)"
elif [ -f "$CONDA_PREFIX_ROOT/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_PREFIX_ROOT/etc/profile.d/conda.sh"
  activated=""
  for name in "$CONDA_ENV" "$(echo "$CONDA_ENV" | tr 'A-Z' 'a-z')" \
              "$(echo "$CONDA_ENV" | tr 'a-z' 'A-Z')"; do
    if conda activate "$name" 2>/dev/null; then
      echo ">> activated conda env: $name"
      activated="$name"
      break
    fi
  done
  if [ -z "$activated" ]; then
    echo "   !! could not activate '$CONDA_ENV' or its case variants."
    echo "      available: $(conda env list 2>/dev/null | awk 'NR>2 && NF {print $1}' | tr '\n' ' ')"
  fi
fi

# --------------------------------------------------------------------------- #
# 2. NumPy must stay on 1.x
#
# The pinned torch build is compiled against the NumPy 1 ABI. NumPy 2 produces
# import-time ABI errors that look unrelated to NumPy.
# --------------------------------------------------------------------------- #
python - <<'PY' || pip install "numpy<2"
import numpy, sys
sys.exit(0 if numpy.__version__.startswith("1.") else 1)
PY

# --------------------------------------------------------------------------- #
# 3. pyarrow
#
# The base image ships a pyarrow too old to read windows.parquet
# ("OSError: Repetition level histogram size mismatch"). Upgrading it can pull
# NumPy 2 back in, so the pin is re-checked immediately afterwards.
# --------------------------------------------------------------------------- #
pip install -U pyarrow >/dev/null 2>&1
python - <<'PY' || pip install "numpy<2"
import numpy, sys
sys.exit(0 if numpy.__version__.startswith("1.") else 1)
PY
python -c "import numpy, pyarrow; print('>> numpy', numpy.__version__, '| pyarrow', pyarrow.__version__)"

# --------------------------------------------------------------------------- #
# 4. fla + triton (CrossDNA only)
#
# `fla` is a namespace package assembled from TWO PyPI distributions:
# flash-linear-attention provides the layers, fla-core provides fla/__init__.py,
# fla/ops/ and fla/modules/. Installing only one gives a ModuleNotFoundError
# that looks like a broken install.
#
# Triton must be 3.2.0 — fla calls Autotuner(do_bench=...), which 3.1.0 does not
# accept. flash-attn silently downgrades triton during its own install, so the
# reinstall has to come AFTER flash-attn, and with --no-deps so it does not drag
# anything else along.
# --------------------------------------------------------------------------- #
echo ">> realigning ephemeral packages (fla 0.4.0 + triton 3.2.0) ..."
pip install --no-deps --no-cache-dir \
  "flash-linear-attention==0.4.0" "fla-core==0.4.0" "triton==3.2.0" \
  >/tmp/env_fix.log 2>&1 \
  && echo "   install OK" \
  || { echo "   ERROR (see /tmp/env_fix.log)"; tail -5 /tmp/env_fix.log; }

if python -c "import fla" >/dev/null 2>&1; then
  python -c "import triton,sys; sys.exit(0 if triton.__version__.startswith('3.2') else 1)" \
    || pip install --no-deps --force-reinstall triton==3.2.0 >/dev/null 2>&1
  python -c "import triton; print('>> triton', triton.__version__)"
fi

# --------------------------------------------------------------------------- #
# 5. Verification
# --------------------------------------------------------------------------- #
echo ">> environment check:"
python - <<'PY'
import torch
print("   torch", torch.__version__, "| cuda", torch.version.cuda)

# torch.compile MUST be neutralised before importing fla, exactly as every
# script in src/ and tools/ does. fla/ops/attn/parallel.py applies the decorator
# at module scope and the inductor here is incompatible with Triton 3.2.0.
# Importing fla without this raises "must be called with a dataclass type or
# instance" — a failure of the check, not of the environment. Reporting it as
# an error would send someone debugging a healthy install.
if hasattr(torch, "compile"):
    torch.compile = (lambda fn=None, *a, **k:
                     (fn if fn is not None else (lambda g: g)))
try:
    import fla, fla.modules, fla.layers.comba  # noqa: F401
    print("   fla + comba: OK (with torch.compile neutralised, as the scripts do)")
except Exception as e:
    print("   fla: ERROR ->", e)
try:
    import pytorch_lightning  # noqa: F401
    print("   lightning: OK")
except Exception as e:
    print("   lightning: ERROR ->", e)
print("   GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY

echo ">> ready. Workspace: $PENGUIN_ROOT"
