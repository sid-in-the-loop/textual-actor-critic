# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
from copy import deepcopy

import hydra
import ray

from omegaconf import open_dict
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"
        }
        for key in ["HF_TOKEN", "OPENAI_API_KEY"]:
            if key in os.environ:
                env_vars[key] = os.environ[key]

        ray.init(
            include_dashboard=False,
            runtime_env={"env_vars": env_vars},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        mlmt_enabled = config.get('mlmt_rl', {}).get('enable', False)
        shared_actor = config.get('mlmt_rl', {}).get('shared_actor', True)

        # Apply SCoRe overrides BEFORE env creation to ensure batch size consistency
        score_mode_cfg = config.get("mlmt_rl", {}).get("score_mode", {})
        if bool(score_mode_cfg.get("enable", False)):
            with open_dict(config):
                batch_size = int(score_mode_cfg.get("batch_size", config.data.train_batch_size))
                config.data.train_batch_size = batch_size
                config.data.gen_batch_size = batch_size
                config.data.random_sample_each_get = True
                print(f"[SCoRe] Early override: train_batch_size set to {batch_size} for environment consistency.")

        with open_dict(config):
            if mlmt_enabled:
                mlmt_cfg = config.mlmt_rl
                stage_cfg = mlmt_cfg.get("stage_control", {})
                if not stage_cfg:
                    stage_cfg = {"stage_id": 1}
                stage_cfg.setdefault("stage_id", 1)
                stage_cfg.setdefault("beta2", 0.0)
                stage_cfg.setdefault("beta1", 0.0)
                stage_cfg.setdefault("beta_L", config.algorithm.kl_ctrl.get("kl_coef", 0.0))
                stage_cfg.setdefault("beta_H", 0.0)
                stage_cfg.setdefault("alpha", 0.0)
                mlmt_cfg["stage_control"] = stage_cfg
                low_model_path = mlmt_cfg.low_level.get("model_path") or config.actor_rollout_ref.model.path
                config.actor_rollout_ref.model.path = low_model_path

                # Explicitly override low-level model params if provided
                if "use_lora" in mlmt_cfg.low_level:
                    config.actor_rollout_ref.model.use_lora = mlmt_cfg.low_level.use_lora
                if "lora_rank" in mlmt_cfg.low_level:
                    config.actor_rollout_ref.model.lora_rank = mlmt_cfg.low_level.lora_rank
                if "lora_alpha" in mlmt_cfg.low_level:
                    config.actor_rollout_ref.model.lora_alpha = mlmt_cfg.low_level.lora_alpha
                if "target_modules" in mlmt_cfg.low_level:
                    config.actor_rollout_ref.model.target_modules = mlmt_cfg.low_level.target_modules

                if not shared_actor:
                    high_actor_cfg = deepcopy(config.actor_rollout_ref)
                    # Explicitly override high-level model params if provided
                    high_actor_cfg.model.path = mlmt_cfg.high_level.get("model_path", high_actor_cfg.model.path)
                    if "use_lora" in mlmt_cfg.high_level:
                        high_actor_cfg.model.use_lora = mlmt_cfg.high_level.use_lora
                    if "lora_rank" in mlmt_cfg.high_level:
                        high_actor_cfg.model.lora_rank = mlmt_cfg.high_level.lora_rank
                    if "lora_alpha" in mlmt_cfg.high_level:
                        high_actor_cfg.model.lora_alpha = mlmt_cfg.high_level.lora_alpha
                    if "target_modules" in mlmt_cfg.high_level:
                        high_actor_cfg.model.target_modules = mlmt_cfg.high_level.target_modules
                    config.high_actor_rollout_ref = high_actor_cfg
                else:
                    config.high_actor_rollout_ref = config.actor_rollout_ref
            else:
                config.high_actor_rollout_ref = config.actor_rollout_ref

        # download the checkpoint from hdfs
        low_model_local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        from agent_system.environments import make_envs

        critique_envs = None
        if config.env.use_critique:
            res = make_envs(config)
            envs = res[0]
            val_envs = res[1]
            critique_envs = res[2]
        else:
            envs, val_envs = make_envs(config)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer_path = config.actor_rollout_ref.model.get("tokenizer_path", None)
        if tokenizer_path is not None:
            # Check if it's a HuggingFace Hub model ID (doesn't start with / or hdfs://)
            # For Hub IDs, use directly; for local/HDFS paths, use copy_to_local
            if not tokenizer_path.startswith("/") and not tokenizer_path.startswith("hdfs://"):
                # HuggingFace Hub model ID - use directly
                tokenizer_local_path = tokenizer_path
            else:
                # Local or HDFS path - use copy_to_local
                tokenizer_local_path = copy_to_local(tokenizer_path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
        else:
            tokenizer_local_path = low_model_local_path
        tokenizer = hf_tokenizer(tokenizer_local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(tokenizer_local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        high_tokenizer = tokenizer
        if mlmt_enabled and not shared_actor:
            high_model_path = config.high_actor_rollout_ref.model.path
            high_tokenizer_path = config.high_actor_rollout_ref.model.get("tokenizer_path", None)
            if high_tokenizer_path is not None:
                if not high_tokenizer_path.startswith("/") and not high_tokenizer_path.startswith("hdfs://"):
                    high_tokenizer_local = high_tokenizer_path
                else:
                    high_tokenizer_local = copy_to_local(high_tokenizer_path, use_shm=config.high_actor_rollout_ref.model.get("use_shm", False))
            else:
                high_model_local_path = copy_to_local(high_model_path, use_shm=config.high_actor_rollout_ref.model.get("use_shm", False))
                high_tokenizer_local = high_model_local_path
            high_tokenizer = hf_tokenizer(high_tokenizer_local, trust_remote_code=trust_remote_code)
        # vllm early verify

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        if mlmt_enabled and not shared_actor:
            # Partition GPUs between the two actors to avoid vLLM process conflict
            # Assuming an even number of GPUs.
            n_gpus = config.trainer.n_gpus_per_node
            assert n_gpus >= 2, "Need at least 2 GPUs to separate high and low level actors"

            low_gpus = n_gpus // 2
            high_gpus = n_gpus - low_gpus

            high_pool_id = "high_pool"
            resource_pool_spec = {
                global_pool_id: [low_gpus] * config.trainer.nnodes,
                high_pool_id: [high_gpus] * config.trainer.nnodes,
            }
            mapping = {
                Role.ActorRollout: global_pool_id,
                Role.Critic: global_pool_id,
                Role.RefPolicy: global_pool_id,
                Role.HighActorRollout: high_pool_id,
            }
        else:
            resource_pool_spec = {
                global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }
            mapping = {
                Role.ActorRollout: global_pool_id,
                Role.Critic: global_pool_id,
                Role.RefPolicy: global_pool_id,
            }

        if mlmt_enabled:
            with open_dict(config):
                # Safety check: ppo_mini_batch_size cannot exceed train_batch_size
                # because the HL actor only has 1x samples (no Turn 3 repeat).
                train_bs = config.data.train_batch_size
                if config.actor_rollout_ref.actor.ppo_mini_batch_size > train_bs:
                    print(f"[Warning] Overriding ppo_mini_batch_size ({config.actor_rollout_ref.actor.ppo_mini_batch_size}) to {train_bs} for HL actor safety.")
                    config.actor_rollout_ref.actor.ppo_mini_batch_size = train_bs

                # Force rollout.n = 1. Multi-turn branching is handled in mlmt_multi_turn_loop.
                if config.actor_rollout_ref.rollout.n != 1:
                    print(f"[Warning] Overriding actor_rollout_ref.rollout.n ({config.actor_rollout_ref.rollout.n}) to 1 for MLMT.")
                    config.actor_rollout_ref.rollout.n = 1

                if not shared_actor:
                    config.high_actor_rollout_ref.actor.ppo_mini_batch_size = min(config.high_actor_rollout_ref.actor.ppo_mini_batch_size, train_bs)
                    config.high_actor_rollout_ref.rollout.n = 1

        if mlmt_enabled and not shared_actor:
            role_worker_mapping[Role.HighActorRollout] = ray.remote(actor_rollout_cls)
            # mapping already set above

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            # mapping already set in resource pool setup block above

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == 'episode':
            from agent_system.reward_manager.episode import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)

        # Note that we always use function-based RM for validation
        val_reward_fn = None
        if config.data.get('val_files'):
            val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        print(f"DEBUG: mlmt_enabled={mlmt_enabled}, shared_actor={shared_actor}")
        print(f"DEBUG: resource_pool_spec={resource_pool_spec}")
        print(f"DEBUG: mapping={mapping}")
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.env.env_name != "none":
            # If MLMT is enabled, we allow n > 1 because our TrajectoryCollector handles
            # the multi-turn group sampling using verl's rollout workers.
            # Also allow n > 1 if the algorithm explicitly requires it (GRPO, REINFORCE).
            is_multi_sample_algo = config.algorithm.adv_estimator in ['grpo', 'reinforce', 'reinforce_plus_plus', 'rloo']
            if not mlmt_enabled and not is_multi_sample_algo:
                assert config.actor_rollout_ref.rollout.n == 1, "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            high_level_tokenizer=high_tokenizer,
            shared_actor=shared_actor,
        )

        # Initialize OpenAI agent if configured
        openai_agent = None
        if config.actor_rollout_ref.get("use_openai_agent", False):
            from agent_system.multi_turn_rollout.openai_agent import OpenAIAgentWorker
            openai_config = config.actor_rollout_ref.get("openai_config", {})
            openai_agent = OpenAIAgentWorker(tokenizer=tokenizer, config=openai_config)
            print("Using OpenAI agent (GPT-5-mini via CMU Gateway) for trajectory collection")

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)

        # --- MODIFICATION: Stage I Downsampling for SCoRe regime ---
        if mlmt_enabled and config.mlmt_rl.stage_control.get("stage_id") == 1:
            downsample_size = 240
            if len(train_dataset) > downsample_size:
                import random
                from torch.utils.data import Subset
                subset_seed = config.data.get("seed", 42)
                rng = random.Random(subset_seed)
                # Sample unique indices for the Stage I pool
                indices = rng.sample(range(len(train_dataset)), downsample_size)
                train_dataset = Subset(train_dataset, indices)
                print(f"[SCoRe] Stage I: Downsampled training pool to {downsample_size} unique samples (seed={subset_seed}).")

        val_dataset = None
        if config.data.get('val_files'):
            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        else:
            val_dataset = None
        train_sampler = create_rl_sampler(config.data, train_dataset)
        mlmt_stage_cfg = None
        if config.get("mlmt_rl", {}).get("enable", False):
            mlmt_stage_cfg = config.mlmt_rl.get("stage_control")

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            critique_envs=critique_envs,
            openai_agent=openai_agent,
            high_level_tokenizer=high_tokenizer,
            shared_actor=shared_actor,
            mlmt_stage_cfg=mlmt_stage_cfg,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    print(f"Loaded data number: {len(dataset)} from {data_paths}")

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
