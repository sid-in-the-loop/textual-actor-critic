#!/bin/bash
set -x

MACHINE_SPECIFIC_RAY_DIR="/tmp/ray_$(hostname)_$(whoami)_$$"
mkdir -p $MACHINE_SPECIFIC_RAY_DIR
export RAY_TMPDIR=$MACHINE_SPECIFIC_RAY_DIR

GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

if [[ "$GPU_MODEL" == *"A6000"* || "$GPU_MODEL" == *"L40S"* ]]; then
    echo "Detected $GPU_MODEL, disabling NCCL P2P"
    export NCCL_P2P_DISABLE=1
else
    echo "Detected $GPU_MODEL, keeping NCCL P2P enabled"
fi

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export RAY_DEBUG=0
export WANDB_API_KEY=1e255990efc627595f0c805e0546cc7f0ff08b17

# Using meta-llama/Llama-3.2-1B-Instruct as the base model
MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH="/home/ssmurali/mlmt/data/math_datasets/curriculum/train_early.parquet"
VAL_PATH="/home/ssmurali/mlmt/data/math_datasets/test.parquet"

python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_PATH \
    data.val_files=$VAL_PATH \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_lora=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.total_training_steps=100 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.model.path=$MODEL_PATH \
    algorithm.adv_estimator=reinforce \
    env.env_name=math \
    env.rollout.n=1 \
    mlmt_rl.enable=true \
    mlmt_rl.shared_actor=true \
    mlmt_rl.use_llm_success_eval=true \
    mlmt_rl.value_fn.model_path=roberta-base \
    mlmt_rl.score_mode.enable=true \
    mlmt_rl.score_mode.batch_size=128 \
    mlmt_rl.score_mode.stages="[{name:stage1,steps:50},{name:stage2,steps:50}]" \
    mlmt_rl.score_mode.alpha=10.0 \
    mlmt_rl.score_mode.beta_turn1_stage1=0.1 \
    mlmt_rl.score_mode.beta_turn1_stage2=0.01 \
    mlmt_rl.score_mode.beta_turn2=0.01 \
    mlmt_rl.score_mode.reward_turn1_stage2=true \
    mlmt_rl.score_mode.prompts.turn2_instruction="" \
    trainer.logger=['console','wandb'] \
    trainer.project_name='math_score' \
    trainer.experiment_name='llama_3.2_1b_score_early_step_prompt' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.total_epochs=999 \
    trainer.total_training_steps=100 \
    trainer.val_before_train=False \
    trainer.default_local_dir="./checkpoints/math/ScoRe/early_step_prompt"
