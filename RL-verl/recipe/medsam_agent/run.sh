#!/bin/bash

set -x

MODEL=${MODEL:-"medsam2"}  # Options: "medsam2" or "imisnet"; can be overridden by env
SAVE_CHECKPOINT_DIR=/your_verl_savepath/verl_qwen3_checkpoints
DATASET_TRAIN='your/train/dataset/path'
DATASET_VAL='your/val/dataset/path'

PROJECT_NAME="medsam_agent"
EXPERIMENT_NAME="medsam-test-1e-5"

REF_MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"

ACTOR_LR=1e-5
SAVE_FREQ=200

# Determine directories
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# RL-verl root (two levels up from this script: recipe/medsam_agent -> recipe -> RL-verl)
RL_VERL_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
LOG_DIR="${RL_VERL_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "Using model: $MODEL_TYPE"

if [ "$MODEL" == "medsam2" ]; then
    export MODEL_TYPE="medsam2"
    TOOL_CONFIG="recipe/medsam_agent/configs/medsam2_point_tools_config.yaml"
elif [ "$MODEL" == "imisnet" ]; then
    export MODEL_TYPE="imisnet"
    TOOL_CONFIG="recipe/medsam_agent/configs/imisnet_point_tools_config.yaml"
else
    echo "Invalid MODEL: $MODEL. Use 'medsam2' or 'imisnet'."
    exit 1
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=$(pwd)/recipe/medsam_agent/configs \
    --config-name='medsam_multiturn' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=[${DATASET_VAL}] \
    data.train_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','tensorboard'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=1000 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=10 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}.log"
