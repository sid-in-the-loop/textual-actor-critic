#!/bin/bash
set -euo pipefail

# 1-GPU Shared Actor Optimization Script
# This configuration minimizes memory by sharing weights between HL and LL.

SHARED_MODEL=${1:-meta-llama/Llama-3.2-1B-Instruct}
SAVE_DIR=${2:-/home/ssmurali/mlmt/checkpoints/math/1gpu_shared}
VALUE_MODEL=${3:-roberta-base}

mkdir -p "$SAVE_DIR"

# Environment Variables
export HF_TOKEN=${HF_TOKEN}
export OPENAI_API_KEY=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
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=online
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export RUN_NAME="1GPU_Shared_Actor_${TIMESTAMP}"
export LOG_DIR="logs/mlmt/math/${RUN_NAME}"
mkdir -p "$LOG_DIR"

# Note: Reduced batch size and memory utilization for 1-GPU stability
python -m verl.trainer.main_ppo \
    data.train_files=/home/ssmurali/mlmt/data/mlmt/math/train.parquet \
    data.val_files=/home/ssmurali/mlmt/data/mlmt/math/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.truncation=right \
    data.return_raw_chat=true \
    trainer.project_name=mlmt_math \
    trainer.experiment_name=${RUN_NAME} \
    trainer.default_local_dir=${SAVE_DIR}/${RUN_NAME} \
    trainer.total_epochs=3 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    actor_rollout_ref.model.path=$SHARED_MODEL \
    actor_rollout_ref.model.use_lora=true \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.lora_dropout=0.05 \
    actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.temperature=0.1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.model.path=$SHARED_MODEL \
    critic.model.use_lora=true \
    critic.model.lora_rank=16 \
    critic.model.lora_alpha=32 \
    critic.model.lora_dropout=0.05 \
    critic.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.adv_estimator=reinforce \
    env.env_name=math \
    env.rollout.n=1 \
    mlmt_rl.enable=true \
    mlmt_rl.shared_actor=true \
    mlmt_rl.high_level.algorithm=reinforce \
    mlmt_rl.low_level.algorithm=reinforce \
    mlmt_rl.value_fn.model_path=$VALUE_MODEL \
    mlmt_rl.use_llm_success_eval=true \
    2>&1 | tee ${LOG_DIR}/train.log

