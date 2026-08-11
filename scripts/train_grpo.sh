#!/usr/bin/env bash
# Launch a GRPO run on whatever GPUs are free right now.
#
# Everything machine-specific is an environment variable with a default, so
# this is the file another machine runs unchanged. The one at /data/murray was
# the same script with the paths welded in.
#
# Required:
#   MODEL_PATH        the model directory (a HF snapshot path)
# Usually set:
#   HF_HOME           where the weights live
#   EXPERIMENT_DIR    where checkpoints, rollouts and validation dumps go
# See docs/RUNNING_ELSEWHERE.md for the rest.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

: "${MODEL_PATH:?set MODEL_PATH to the model directory}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$REPO/exps/grpo_courier}"
mkdir -p "$EXPERIMENT_DIR"

# flashinfer JIT-compiles against CUDA_HOME. Where the system nvcc is older
# than the CUDA torch was built for, compilation fails inside math.h with an
# error that names neither. Pointing CUDA_HOME at the conda environment that
# holds the matching toolkit is the fix; harmless when they already agree.
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
  export CUDA_HOME="${CONDA_PREFIX}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

# Chosen fresh, from what is free at this second. A snapshot taken when the
# config was edited is already wrong by the time vLLM starts on a shared
# machine: fifteen launches died on "Free memory on device ... less than
# desired" before this existed.
read -r _DEV _MEM _N _SP <<< "$(bash "$HERE/pick_gpus.sh")"
if [ "$_DEV" = "NONE" ]; then
  echo "No GPU has ${MIN_FREE_MB:-18000} MB free right now." >&2
  nvidia-smi --query-gpu=index,memory.free --format=csv >&2
  exit 1
fi
# Honoured if the caller named the cards, for the same reason ROLLOUT_MEM is:
# the picker ranks by free memory at this instant, and on a shared host the
# instant is not the run. It chose two cards with another user's 3 GB still on
# them over four that were completely empty.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$_DEV}
# Computed from live free memory, unless the caller has already said what it
# wants. It used to overwrite an explicit setting without a word, which on a
# shared host is exactly when someone is overriding it: the computed fraction
# asked for more than another user had left free, the rollout engine refused
# to start, and the trainer stayed alive and produced no steps -- a run that
# reads as slow rather than as broken.
export ROLLOUT_MEM=${ROLLOUT_MEM:-$_MEM}
export N_GPUS=$_N
export SP=$_SP
echo "PICKED devices=$_DEV vllm_mem=$_MEM gpus=$_N sp=$_SP"

# Sizes that have actually run. See the table in docs/RUNNING_ELSEWHERE.md
# before changing them; the learning rate and the reward scale multiply.
export TRAIN_BS="${TRAIN_BS:-6}"
export PROMPT_LEN="${PROMPT_LEN:-4096}"
export RESP_LEN="${RESP_LEN:-36864}"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export KL_COEF="${KL_COEF:-0.005}"
export TEST_FREQ="${TEST_FREQ:-10}"

exec bash "$REPO/embodiedbench/training/vagen/train_grpo_courier.sh"
