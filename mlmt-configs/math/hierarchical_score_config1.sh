#!/bin/bash
set -euo pipefail

SHARED_MODEL=${1:-meta-llama/Llama-3.2-1B-Instruct}
SAVE_DIR=${2:-/home/ssmurali/mlmt/checkpoints/math/hierarchical_score_config1}
VALUE_MODEL=${3:-roberta-base}
mkdir -p "$SAVE_DIR"

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=online
export RAY_INCLUDE_DASHBOARD=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="Hierarchical_Config1_${TIMESTAMP}"
LOG_DIR="logs/mlmt/math/${RUN_NAME}"
mkdir -p "$LOG_DIR"

python -m verl.trainer.main_ppo \
    data.train_files=/home/ssmurali/mlmt/data/mlmt/math/train.parquet \
    data.val_files=/home/ssmurali/mlmt/data/mlmt/math/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.truncation=right \
    data.return_raw_chat=true \
    +data.dataloader_num_workers=4 \
    trainer.project_name=mlmt_math \
    trainer.experiment_name=${RUN_NAME} \
    trainer.default_local_dir=${SAVE_DIR}/${RUN_NAME} \
    trainer.total_epochs=999 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.total_training_steps=1250 \
    actor_rollout_ref.model.path=$SHARED_MODEL \
    actor_rollout_ref.model.use_lora=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.total_training_steps=1250 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.model.path=$SHARED_MODEL \
    algorithm.adv_estimator=reinforce \
    env.env_name=math \
    env.rollout.n=1 \
    mlmt_rl.enable=true \
    mlmt_rl.shared_actor=false \
    mlmt_rl.score_mode.enable=false \
    mlmt_rl.low_level.model_path=$SHARED_MODEL \
    mlmt_rl.high_level.model_path=$SHARED_MODEL \
    mlmt_rl.use_llm_success_eval=true \
    mlmt_rl.value_fn.model_path=$VALUE_MODEL \
    +mlmt_rl.stage_control.stage_id=1 \
    +mlmt_rl.stage_control.beta2=0.1 \
    +mlmt_rl.stage_control.beta1=0.0 \
    +mlmt_rl.stage_control.beta_L=0.01 \
    +mlmt_rl.stage_control.beta_H=0.0 \
    +mlmt_rl.stage_control.alpha=10.0 \
    +mlmt_rl.stage_control.schedule="[{step:250,stage_id:2,beta2:0.0,beta1:0.01,beta_L:0.01,beta_H:0.0}]" \
    2>&1 | tee ${LOG_DIR}/train.log
