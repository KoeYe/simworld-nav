#!/usr/bin/env bash
# One-time setup for a machine that will TRAIN (runs verl/vagen + vLLM).
#
# Deliberately does not use the system python: the cluster's boxes carry
# 3.10 with no pip, and the stack needs 3.12. uv brings its own interpreter,
# which is why it is here rather than a requirements.txt and good wishes.
set -euo pipefail

ROOT="${NAV_ROOT:-/data/koe}"
VENV="$ROOT/rl-venv"

command -v uv >/dev/null 2>&1 || {
  echo "== installing uv (no system pip needed)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

echo "== python 3.12 venv at $VENV"
uv venv --python 3.12 "$VENV"

echo "== torch / vllm / flash-attn"
# numpy>=2 on purpose: transformers 4.57 needs it, and on numpy 1.x scipy
# explodes while importing PreTrainedModel with an error that names neither.
uv pip install --python "$VENV/bin/python" \
  "torch==2.8.*" "vllm==0.11.*" "numpy>=2.2" "transformers==4.57.*" \
  "flash-attn==2.8.*" --no-build-isolation

echo
echo "ready. start a run with:"
echo "  EB_UE_ENDPOINTS=http://<render-host>:18800 $ROOT/simworld-nav/ops/run_trainer.sh"
