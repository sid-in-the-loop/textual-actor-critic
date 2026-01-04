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
export HF_TOKEN=${HF_TOKEN}
export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
export OPENAI_API_KEY=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
export CMU_GATEWAY_API_KEY=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

MODEL_DIR=/data/group_data/cx_group/verl_agent_shared


train_data_size=32
val_data_size=16
group_size=8


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.compute_mean_std_cross_all_data=False \
    data.train_files=dummy_data/text/train.parquet \
    data.val_files=dummy_data/text/val.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=20000  \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.model.path=/data/group_data/cx_group/behavior_priming/checkpoint/qwen3_1.7b/web_qwen_sft_behavior/checkpoint-924 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=22000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=22000 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.max_model_len=4096 \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    env.rule_reward_coef=0 \
    env.env_name=deepresearch \
    env.dataset=afm \
    env.seed=0 \
    env.rollout.n=$group_size \
    env.rollout.k=1 \
    env.max_steps=3 \
    env.use_explicit_thinking=False \
    env.use_critique=False \
    env.replace_input=False \
    env.use_rule_reward=False \
    env.rule_number=5 \
    env.use_dense_reward=False \
    env.belief_shaped_grpo.enable=True \
    env.belief_shaped_grpo.alpha=1.0 \
    env.belief_shaped_grpo.max_candidates=5 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='DeepResearch_RL' \
    trainer.experiment_name='deepresearch_1.7b_sft_grpo_180steps_4gpu' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=25 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=180 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$MODEL_DIR/checkpoint/afm_apm_sft_1.7b_all_positive_base_reward_grpo \
    trainer.val_before_train=False $@

