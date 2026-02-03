#!/bin/bash
set -euo pipefail

# HLT_LLT_RR 1:10:5 - Bi-Level Contrastive Regularization
# HL update: 1, LL update: 10, Reg Gap: 5
SHARED_MODEL=${1:-meta-llama/Llama-3.2-1B-Instruct}
SAVE_DIR=${2:-/home/ssmurali/mlmt/checkpoints/math/HLT_LLT_RR_1_10_5}
mkdir -p "$SAVE_DIR"

export OPENAI_API_KEY=${OPENAI_API_KEY}
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=online

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="HLT_LLT_RR_1_10_5_${TIMESTAMP}"

python -m verl.trainer.main_ppo \
    data.train_files=/home/ssmurali/mlmt/data/mlmt/math/train_1020.parquet \
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
    trainer.save_freq=50 \
    trainer.total_training_steps=400 \
    +trainer.lora_only_save=false \
    actor_rollout_ref.model.path=$SHARED_MODEL \
    actor_rollout_ref.model.use_lora=false \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.total_training_steps=400 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.model.path=$SHARED_MODEL \
    critic.model.use_lora=false \
    critic.model.lora_rank=0 \
    algorithm.adv_estimator=reinforce \
    env.env_name=math \
    env.rollout.n=1 \
    mlmt_rl.enable=true \
    mlmt_rl.shared_actor=false \
    mlmt_rl.high_level.model_path=$SHARED_MODEL \
    +mlmt_rl.high_level.use_lora=false \
    +mlmt_rl.high_level.lora_rank=0 \
    mlmt_rl.high_level.algorithm=reinforce \
    mlmt_rl.high_level.freeze=false \
    mlmt_rl.low_level.model_path=$SHARED_MODEL \
    +mlmt_rl.low_level.use_lora=false \
    +mlmt_rl.low_level.lora_rank=0 \
    mlmt_rl.low_level.algorithm=reinforce \
    mlmt_rl.low_level.freeze=false \
    +mlmt_rl.high_level.update_frequency=1 \
    +mlmt_rl.low_level.update_frequency=10 \
    +mlmt_rl.high_level.max_tokens=512 \
    mlmt_rl.use_llm_success_eval=true \
    +mlmt_rl.reg_enabled=true \
    +mlmt_rl.reg_lambda=1.0 \
    +mlmt_rl.reg_gap=5 \
    +mlmt_rl.stage_control.stage_id=1 \
    +mlmt_rl.stage_control.beta2=0.1 \
    +mlmt_rl.stage_control.beta1=0.0 \
    +mlmt_rl.stage_control.beta_L=0.01 \
    +mlmt_rl.stage_control.beta_H=0.0 \
    +mlmt_rl.stage_control.alpha=10.0 \
    +mlmt_rl.stage_control.schedule="[{step:200,stage_id:2,beta2:0.0,beta1:0.01,beta_L:0.01,beta_H:0.0}]"
