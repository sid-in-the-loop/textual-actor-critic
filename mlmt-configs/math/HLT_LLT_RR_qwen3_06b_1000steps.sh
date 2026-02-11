#!/bin/bash
set -euo pipefail

# Combined HLT_LLT_RR 1:50 - Stage 1 (300 steps) + Stage 2 (700 steps)
# LoRA trained, Qwen3-0.6B
# HL update: 1, LL update: 50, No Reg Gap
SHARED_MODEL=${1:-Qwen/Qwen3-0.6B}
SAVE_DIR=${2:-/home/ssmurali/mlmt/checkpoints/math}
mkdir -p "$SAVE_DIR"

export OPENAI_API_KEY=${OPENAI_API_KEY}
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=online

# Fixed RUN_NAME for auto-resume support (no timestamp)
RUN_NAME=${RUN_NAME:-"HLT_LLT_RR_qwen3_06b_1000steps"}
echo "Using experiment name: $RUN_NAME"

LOG_DIR="logs/mlmt/math/${RUN_NAME}"
mkdir -p "$LOG_DIR"

python -m verl.trainer.main_ppo \
    data.train_files=/home/ssmurali/mlmt/data/math_datasets/dapo/train-00000-of-00001.parquet \
    data.val_files=/home/ssmurali/mlmt/data/math_datasets/dapo/train-00000-of-00001.parquet \
    data.prompt_key=source_prompt \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
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
    trainer.resume_mode=auto \
    trainer.total_training_steps=1000 \
    +trainer.lora_only_save=true \
    reward_model.reward_manager=dapo \
    actor_rollout_ref.model.path=$SHARED_MODEL \
    actor_rollout_ref.model.use_lora=true \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.lora_dropout=0.05 \
    actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.total_training_steps=1000 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    algorithm.adv_estimator=reinforce \
    env.env_name=math \
    env.rollout.n=1 \
    mlmt_rl.enable=true \
    mlmt_rl.shared_actor=false \
    mlmt_rl.high_level.model_path=$SHARED_MODEL \
    +mlmt_rl.high_level.use_lora=true \
    +mlmt_rl.high_level.lora_rank=16 \
    +mlmt_rl.high_level.lora_alpha=32 \
    +mlmt_rl.high_level.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    mlmt_rl.high_level.algorithm=reinforce \
    mlmt_rl.high_level.freeze=false \
    mlmt_rl.low_level.model_path=$SHARED_MODEL \
    +mlmt_rl.low_level.use_lora=true \
    +mlmt_rl.low_level.lora_rank=16 \
    +mlmt_rl.low_level.lora_alpha=32 \
    +mlmt_rl.low_level.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    mlmt_rl.low_level.algorithm=reinforce \
    mlmt_rl.low_level.freeze=false \
    +mlmt_rl.high_level.update_frequency=1 \
    +mlmt_rl.low_level.update_frequency=50 \
    +mlmt_rl.high_level.max_tokens=512 \
    +mlmt_rl.reg_enabled=false \
    mlmt_rl.use_llm_success_eval=true \
    +mlmt_rl.stage_control.stage_id=1 \
    +mlmt_rl.stage_control.beta2=0.1 \
    +mlmt_rl.stage_control.beta1=0.0 \
    +mlmt_rl.stage_control.beta_L=0.01 \
    +mlmt_rl.stage_control.beta_H=0.0 \
    +mlmt_rl.stage_control.alpha=10.0 \
    +mlmt_rl.stage_control.schedule="[{step:300,stage_id:2,beta2:0.0,beta1:0.01,beta_L:0.01,beta_H:0.0}]" \
    2>&1 | tee ${LOG_DIR}/train.log
