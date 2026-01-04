#!/bin/bash
set -euo pipefail

SHARED_MODEL=${1:-meta-llama/Llama-3.2-1B-Instruct}
SAVE_DIR=${2:-/home/ssmurali/mlmt/checkpoints/math/HLT_LLT_RR_1_1}
VALUE_MODEL=${3:-roberta-base}
mkdir -p "$SAVE_DIR"

export OPENAI_API_KEY=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
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="HLT_LLT_RR_1_1_${TIMESTAMP}"
LOG_DIR="logs/mlmt/math/${RUN_NAME}"
mkdir -p "$LOG_DIR"

python -m verl.trainer.main_ppo \
    data.train_files=/home/ssmurali/mlmt/data/mlmt/math/train.parquet \
    data.val_files=/home/ssmurali/mlmt/data/mlmt/math/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.truncation=right \
    data.return_raw_chat=true \
    +data.dataloader_num_workers=4 \
    trainer.project_name=mlmt_math \
    trainer.experiment_name=${RUN_NAME} \
    trainer.default_local_dir=${SAVE_DIR}/${RUN_NAME} \
    trainer.total_epochs=3 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    actor_rollout_ref.model.path=$SHARED_MODEL \
    actor_rollout_ref.model.use_lora=true \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.lora_dropout=0.05 \
    actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.total_training_steps=315 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.model.path=$SHARED_MODEL \
    critic.model.use_lora=true \
    critic.model.lora_rank=16 \
    critic.model.lora_alpha=32 \
    critic.model.lora_dropout=0.05 \
    critic.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    algorithm.adv_estimator=reinforce \
    env.env_name=math \
    env.rollout.n=1 \
    mlmt_rl.enable=true \
    mlmt_rl.shared_actor=false \
    mlmt_rl.high_level.model_path=$SHARED_MODEL \
    mlmt_rl.high_level.algorithm=reinforce \
    mlmt_rl.high_level.freeze=false \
    mlmt_rl.low_level.model_path=$SHARED_MODEL \
    mlmt_rl.low_level.algorithm=reinforce \
    mlmt_rl.low_level.freeze=false \
    +mlmt_rl.high_level.update_frequency=5 \
    +mlmt_rl.low_level.update_frequency=15 \
    +mlmt_rl.high_level.max_tokens=512 \
    mlmt_rl.value_fn.model_path=$VALUE_MODEL \
    mlmt_rl.use_llm_success_eval=true \
    2>&1 | tee ${LOG_DIR}/train.log

