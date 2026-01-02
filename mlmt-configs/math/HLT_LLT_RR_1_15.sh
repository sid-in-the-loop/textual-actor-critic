#!/bin/bash
set -euo pipefail

SHARED_MODEL=${1:-meta-llama/Llama-3.2-1B-Instruct}
SAVE_DIR=${2:-/home/ssmurali/mlmt/checkpoints/math/HLT_LLT_RR_1_15}
VALUE_MODEL=${3:-roberta-base}
mkdir -p "$SAVE_DIR"

export OPENAI_API_KEY=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
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=online

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="HLT_LLT_RR_1_15_LoRA_${TIMESTAMP}"
LOG_DIR="logs/mlmt/math/${RUN_NAME}"
mkdir -p "$LOG_DIR"

# Start Proxy in background (Automated workaround for internet on compute nodes)
echo "Setting up SOCKS proxy via login1..."
ssh -N -f -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -D 127.0.0.1:1080 login1

export ALL_PROXY=socks5h://127.0.0.1:1080
export HTTP_PROXY=$ALL_PROXY
export HTTPS_PROXY=$ALL_PROXY
export http_proxy=$ALL_PROXY
export https_proxy=$ALL_PROXY

# Ensure dependencies for proxy are installed
pip install "httpx[socks]"

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
    trainer.total_epochs=5 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.resume_mode=disable \
    actor_rollout_ref.model.path=$SHARED_MODEL \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules='all-linear' \
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
    +mlmt_rl.high_level.update_frequency=1 \
    +mlmt_rl.low_level.update_frequency=15 \
    +mlmt_rl.high_level.max_tokens=512 \
    mlmt_rl.value_fn.model_path=$VALUE_MODEL \
    mlmt_rl.use_llm_success_eval=true \
    2>&1 | tee ${LOG_DIR}/train.log
