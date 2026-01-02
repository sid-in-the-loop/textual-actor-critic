#!/bin/bash
set -euo pipefail

# Checkpoint paths (Pointing directly to the 'actor' folder where config.json resides)
CKPT_1_1="/home/ssmurali/mlmt/checkpoints/math/HLT_LLT_RR_1_1/HLT_LLT_RR_1_1_20260101_165640/global_step_100/actor"
CKPT_1_15="/home/ssmurali/mlmt/checkpoints/math/HLT_LLT_RR_1_15/HLT_LLT_RR_1_15_20260101_165638/global_step_100/actor"

# Dataset
DATA_PATH="/home/ssmurali/mlmt/data/math_datasets/math500_test.parquet"

# Start Proxy in background (Automated workaround for internet on compute nodes)
echo "Setting up SOCKS proxy via login1..."
ssh -N -f -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -D 127.0.0.1:1080 login1

export ALL_PROXY=socks5h://127.0.0.1:1080
export HTTP_PROXY=$ALL_PROXY
export HTTPS_PROXY=$ALL_PROXY
export http_proxy=$ALL_PROXY
export https_proxy=$ALL_PROXY

# Eval for 1:1 case
echo "Evaluating 1:1 checkpoint: $CKPT_1_1"
python -m verl.trainer.main_ppo \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    trainer.project_name=mlmt_eval \
    trainer.experiment_name=eval_math500_1_1 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    env.env_name=math \
    trainer.total_epochs=0 \
    trainer.val_before_train=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    critic.ppo_mini_batch_size=128 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.path=meta-llama/Llama-3.2-1B-Instruct \
    actor_rollout_ref.model.path=$CKPT_1_1 \
    actor_rollout_ref.rollout.response_length=1024 \
    mlmt_rl.enable=true \
    +mlmt_rl.low_level.max_tokens_turn1=1024 \
    +mlmt_rl.low_level.max_tokens_turn3=2048 \
    mlmt_rl.use_llm_success_eval=true \
    2>&1 | tee logs/mlmt/eval/math500/eval_1_1.log

# Eval for 1:15 case
echo "Evaluating 1:15 checkpoint: $CKPT_1_15"
python -m verl.trainer.main_ppo \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    trainer.project_name=mlmt_eval \
    trainer.experiment_name=eval_math500_1_15 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    env.env_name=math \
    trainer.total_epochs=0 \
    trainer.val_before_train=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    critic.ppo_mini_batch_size=128 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.path=meta-llama/Llama-3.2-1B-Instruct \
    actor_rollout_ref.model.path=$CKPT_1_15 \
    actor_rollout_ref.rollout.response_length=1024 \
    mlmt_rl.enable=true \
    +mlmt_rl.low_level.max_tokens_turn1=1024 \
    +mlmt_rl.low_level.max_tokens_turn3=2048 \
    mlmt_rl.use_llm_success_eval=true \
    2>&1 | tee logs/mlmt/eval/math500/eval_1_15.log
