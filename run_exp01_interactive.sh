#!/bin/bash
# Interactive Exp 1 Run (1 GPU) - Copy and paste this entire script

cd /home/ssmurali/verl-agent

# export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export RAY_DEBUG=0
export WANDB_API_KEY=1e255990efc627595f0c805e0546cc7f0ff08b17
export HF_TOKEN=${HF_TOKEN}
export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
export CMU_GATEWAY_API_KEY=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

# Local Ray config (1 GPU)
export RAY_TMPDIR="/tmp/ray_local_$$"

MODEL_DIR=/data/group_data/cx_group/verl_agent_shared

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    env.env_name=math \
    env.max_steps=1 \
    env.rollout.n=8 \
    data.train_files=data/math_datasets/train.parquet \
    data.val_files=data/math_datasets/math500_test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.do_sample=False \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
    actor_rollout_ref.rollout.max_model_len=3072 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    hierarchical.enabled=False \
    hierarchical.logging_enabled=True \
    hierarchical.experiment_id=exp01_local \
    hierarchical.ll_trainable=True \
    hierarchical.max_prompt_tokens=4096 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='HICRL_MATH' \
    trainer.experiment_name='exp01_grpo_baseline_qwen3_1.7b_local' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=999999 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.resume_mode=auto \
    trainer.default_local_dir=$MODEL_DIR/new_checkpoint/exp01_grpo_baseline_local \
    trainer.val_before_train=False \
    trainer.log_val_generations=1
