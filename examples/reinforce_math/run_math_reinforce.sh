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
DATA_PATH="/home/ssmurali/mlmt/data/math_datasets/curriculum/train_mid.parquet"
VAL_PATH="/home/ssmurali/mlmt/data/math_datasets/test.parquet"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=$DATA_PATH \
    data.val_files=$VAL_PATH \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    env.env_name=math \
    trainer.logger=['console','wandb'] \
    trainer.project_name='math_reinforce' \
    trainer.experiment_name='llama_3.2_1b_reinforce_mid' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=100 \
    trainer.val_before_train=False \
    trainer.default_local_dir="./checkpoints/llama_3.2_1b_reinforce_mid"
