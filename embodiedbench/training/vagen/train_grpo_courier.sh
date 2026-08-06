#!/bin/bash
# GRPO on the courier benchmark, using VAGEN's verl stack.
#
# Adapted from vendor/vagen/examples/deliverybench/train_grpo_qwen25vl3b.sh.
# Two deliberate differences from that script:
#
#   * the environment is registered from the command line
#     (+env_registry.Courier=...), so nothing inside the gitignored vendor/
#     checkout has to be edited for this to run;
#   * no API key is baked in. The original has a live WANDB_API_KEY committed
#     into it; export your own or leave the logger on console.
#
# Prerequisites that are NOT satisfied by the default env -- see WORK_SUMMARY.md:
#   ray, and a rollout backend (sglang, or vllm on a torch build that works).
set -x

PROJECT_NAME="verl_vagen"
EXPERIMENT_NAME="grpo_courier_qwen3vl4b"

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
VAGEN=${REPO}/vendor/vagen
EXPERIMENT_DIR=${EXPERIMENT_DIR:-~/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}}
DATASET_TRAIN=${REPO}/embodiedbench/training/vagen/train_courier.yaml
DATASET_VAL=${REPO}/embodiedbench/training/vagen/val_courier.yaml
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to a local checkpoint}

export HYDRA_FULL_ERROR=1
# NOTE: do not set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
# It looks like the right answer to the backward pass OOMing on a 1.92 GiB
# allocation with 889 MiB free, but vLLM refuses to start with it:
#   AssertionError: Expandable segments are not compatible with memory pool.
# The failure surfaces as "Engine core initialization failed" during
# load_model, which reads as not enough memory for the weights and sends you
# tuning gpu_memory_utilization instead. Shorten the sequence instead.
export WANDB_MODE=${WANDB_MODE:-offline}
# The adapter lives in this repo, not in VAGEN, so both must be importable.
export PYTHONPATH=${REPO}:${VAGEN}:${PYTHONPATH}
mkdir -p "${EXPERIMENT_DIR}"

# PREFLIGHT: torch must actually see the GPUs before ray is told there are
# N of them.
#
# A GPU released moments earlier still lists free memory in nvidia-smi while
# CUDA cannot open it yet, and torch then reports a smaller device_count than
# CUDA_VISIBLE_DEVICES names. verl asks for the rank it was configured for and
# dies with "device >= 0 && device < num_gpus ... device=2, num_gpus=2", which
# reads like a framework bug about device indexing. It is not: it is launching
# too soon after killing whatever held the card. Four separate GPU counts were
# blamed for this before the cards were simply checked.
# Counting is not enough: device_count has reported four while a later import
# in the same run saw two. Every card is opened and written to here, which is
# the only claim that matters.
SEEN=$("${PYTHON:-python3}" - <<'PYEOF'
import torch
ok = 0
for i in range(torch.cuda.device_count()):
    try:
        torch.zeros(8, device=f"cuda:{i}")
        ok += 1
    except Exception as error:
        print(f"cuda:{i} unusable: {error}", flush=True)
print(ok)
PYEOF
)
SEEN=$(echo "${SEEN}" | tail -1)
WANT=${N_GPUS:-1}
if [ "${SEEN}" != "${WANT}" ]; then
  echo "PREFLIGHT FAILED: torch sees ${SEEN} GPU(s), N_GPUS=${WANT}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "the cards named are not all open yet -- wait for whatever held them to exit, then retry"
  exit 1
fi
echo "PREFLIGHT: torch sees ${SEEN} GPU(s), as configured"

cd "${VAGEN}" || exit 1

PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path="${VAGEN}/vagen/configs" \
    --config-name='vagen_multiturn' \
    +env_registry.Courier=embodiedbench.training.vagen_courier_env.CourierGymEnv \
    data.train_files="${DATASET_TRAIN}" \
    data.val_files="${DATASET_VAL}" \
    data.train_batch_size=${TRAIN_BS:-2} \
    data.max_prompt_length=${PROMPT_LEN:-4096} \
    data.max_response_length=${RESP_LEN:-8192} \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BS:-2} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP:-1} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_BACKEND:-vllm} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_MEM:-0.4} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${VAGEN}/vagen/configs/agent.yaml" \
    critic.strategy=fsdp2 \
    critic.model.path="${MODEL_PATH}" \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.optimizer_offload=True \
    reward_model.strategy=fsdp2 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${N_GPUS:-1} \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir="${EXPERIMENT_DIR}/verl_checkpoints" \
    trainer.rollout_data_dir="${EXPERIMENT_DIR}/rollout_data" \
    2>&1 | tee "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}.log"
