#!/bin/bash
set -euo pipefail

# Args: SHARED_MODEL SAVE_DIR VALUE_MODEL
SHARED_MODEL=${1:-meta-llama/Llama-3.2-1B-Instruct}
SAVE_DIR=${2:-/home/ssmurali/mlmt/checkpoints/code/HLT_LLT_RR_mbpp}
VALUE_MODEL=${3:-roberta-base}
mkdir -p "$SAVE_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="HLT_LLT_RR_mbpp_${TIMESTAMP}"
LOG_DIR="/home/ssmurali/mlmt/logs/mlmt/code/${RUN_NAME}"
mkdir -p "$LOG_DIR"

python -m verl.trainer.main_ppo \
    data.train_files=/home/ssmurali/mlmt/data/mlmt/code/train.parquet \
    data.val_files=null \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.truncation=right \
    data.return_raw_chat=true \
    +data.dataloader_num_workers=4 \
    trainer.project_name=mlmt_code \
    trainer.experiment_name=${RUN_NAME} \
    trainer.default_local_dir=${SAVE_DIR}/${RUN_NAME} \
    trainer.total_epochs=500 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.total_training_steps=500 \
    actor_rollout_ref.model.path=$SHARED_MODEL \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.total_training_steps=500 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.model.path=$SHARED_MODEL \
    algorithm.adv_estimator=reinforce \
    env.env_name=code \
    env.rollout.n=1 \
    mlmt_rl.enable=true \
    mlmt_rl.env_type=code \
    mlmt_rl.shared_actor=false \
    mlmt_rl.high_level.model_path=$SHARED_MODEL \
    mlmt_rl.high_level.algorithm=reinforce \
    mlmt_rl.high_level.freeze=false \
    mlmt_rl.low_level.model_path=$SHARED_MODEL \
    mlmt_rl.low_level.algorithm=reinforce \
    mlmt_rl.low_level.freeze=false \
    +mlmt_rl.high_level.update_frequency=1 \
    +mlmt_rl.low_level.update_frequency=5 \
    +mlmt_rl.high_level.max_tokens=512 \
    mlmt_rl.value_fn.model_path=$VALUE_MODEL \
    mlmt_rl.use_llm_success_eval=true \
    mlmt_rl.score_mode.enable=true \
    mlmt_rl.score_mode.beta_turn1_stage1=0.05 \
    mlmt_rl.score_mode.stages="[{name:stage1,steps:250},{name:stage2,steps:500}]" \
    2>&1 | tee ${LOG_DIR}/train.log

