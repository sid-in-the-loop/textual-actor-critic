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

MODEL_PATH="Qwen/Qwen3-1.7B"
DATA_PATH="/home/ssmurali/mlmt/data/math_datasets/dapo/train-00000-of-00001.parquet"
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
    actor_rollout_ref.actor.optim.total_training_steps=1000 \
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
    mlmt_rl.null_feedback_experiment=true \
    mlmt_rl.num_turns=2 \
    mlmt_rl.stage_control.stage_id=1 \
    mlmt_rl.stage_control.beta2=0.1 \
    mlmt_rl.stage_control.beta1=0.01 \
    mlmt_rl.stage_control.beta_L=0.01 \
    mlmt_rl.stage_control.beta_H=0.01 \
    mlmt_rl.stage_control.alpha=10.0 \
    mlmt_rl.stage_control.schedule="[{step:200,stage_id:2}]" \
    trainer.logger=['console','wandb'] \
    trainer.project_name='math_null_feedback' \
    trainer.experiment_name='qwen3_1.7b_null_feedback_2turn' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=-1 \
    trainer.total_epochs=999 \
    trainer.total_training_steps=1000 \
    trainer.val_before_train=False \
    trainer.default_local_dir="./checkpoints/math/null_feedback/2turn"
