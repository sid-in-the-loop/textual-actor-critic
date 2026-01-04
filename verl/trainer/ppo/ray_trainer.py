# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import random
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from gigpo import core_gigpo

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch

from verl.trainer.hierarchical.context_manager import HierarchicalContextManager, HierarchicalLogger

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    HighActorRollout = 7


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE = "reinforce"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(data: DataProto, reward_coef=float, rule_number: int = 5):

    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']

    ## for debug
    scores = data.batch['token_level_scores'].sum(dim=-1)
    print(f"first 10 rewards before add dense penalty: {scores[:10]}")

    # Initialize metrics
    metrics = {}

    # Calculate valid ratios for each rule
    rule_valid_ratios = {}
    for rule_idx in range(1, rule_number + 1):
        rule_field = f'is_rule{rule_idx}_valid'
        if rule_field in data.non_tensor_batch:
            rule_valid_data = data.non_tensor_batch[rule_field].astype(np.float32)
            rule_valid_ratios[rule_field] = np.mean(rule_valid_data).item()
            metrics[f'{rule_field}_ratio'] = rule_valid_ratios[rule_field]

    # Calculate valid action ratio
    if 'is_action_valid' in data.non_tensor_batch:
        valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
        metrics['valid_action_ratio'] = valid_action_ratio

    # Initialize variables for average violations calculation
    total_violations_sum = 0.0
    total_passed_sum = 0.0
    total_penalty_sum = 0.0

    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        # Calculate total rule violations for this step
        total_rule_violations = 0.0
        total_rule_passed = 0.0
        
        # Debug: print all keys in non_tensor_batch for this step

        # Count violations for each rule
        for rule_idx in range(1, rule_number + 1):
            rule_field = f'is_rule{rule_idx}_valid'
            if rule_field in data_item.non_tensor_batch:
                # Rule is invalid if it's False (violation)
                rule_val = data_item.non_tensor_batch[rule_field]
                # Robust cast for scalar or ndarray-like
                try:
                    rule_valid = rule_val.astype(np.float32)
                except AttributeError:
                    rule_valid = np.float32(rule_val)
                rule_invalid = 1.0 - rule_valid
                total_rule_passed += rule_valid
                total_rule_violations += rule_invalid

        # Add action invalid penalty
        if 'is_action_valid' in data_item.non_tensor_batch:
            action_val = data_item.non_tensor_batch['is_action_valid']
            try:
                action_valids = action_val.astype(np.float32)
            except AttributeError:
                action_valids = np.float32(action_val)
            action_invalids = 1 - action_valids
            total_rule_violations += action_invalids

        # Apply penalty: total violations * reward_coef
        penalty = total_rule_violations * reward_coef
        total_penalty_sum += penalty

        # Apply penalty to reward tensor
        if valid_response_length > 0:
            reward_tensor[i, valid_response_length - 1] -= penalty

        # Apply penalty to step rewards if available
        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= penalty

        # Accumulate total violations for average calculation
        total_violations_sum += total_rule_violations
        total_passed_sum += total_rule_passed

    # Calculate average violations per step
    if len(data) > 0:
        avg_violations = total_violations_sum / len(data)
        metrics['avg_violations_per_step'] = avg_violations
        avg_passed = total_passed_sum / len(data)
        metrics['avg_passed_per_step'] = avg_passed
        avg_penalty = total_penalty_sum / len(data)
        metrics['avg_penalty_per_step'] = avg_penalty

    ## for debug
    scores = data.batch['token_level_scores'].sum(dim=-1)
    print(f"first 10 rewards after add dense penalty: {scores[:10]}")

    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, compute_mean_std_cross_all_data=True, step_advantage_w=1.0, gigpo_mode="mean_std_norm", **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        compute_mean_std_cross_all_data (bool, optional): Whether to compute_mean_std across all data in the batch. Defaults to True.
    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            compute_mean_std_cross_all_data=compute_mean_std_cross_all_data,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE:
        advantages, returns = core_algos.compute_reinforce_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
        critique_envs=None,
        openai_agent=None,
        high_level_tokenizer=None,
        shared_actor=True,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.envs = envs
        self.val_envs = val_envs
        self.critique_envs = critique_envs
        self.traj_collector = traj_collector
        self.openai_agent = openai_agent
        self.high_level_tokenizer = high_level_tokenizer or tokenizer
        self.shared_actor = shared_actor
        self.mlmt_cfg = config.get("mlmt_rl", {})
        self.mlmt_enabled = bool(self.mlmt_cfg.get("enable", False))
        self.value_worker = None
        self.value_cfg = None
        value_cfg_node = OmegaConf.select(self.config, "mlmt_rl.value_fn")
        if self.mlmt_enabled and value_cfg_node is not None:
            from agent_system.value_function.roberta_worker import RobertaValueWorker
            value_cfg = OmegaConf.to_container(value_cfg_node, resolve=True)
            self.value_cfg = value_cfg
            self.value_worker = RobertaValueWorker(value_cfg)
            self.value_worker.init_model()

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.high_actor_rollout_wg = None

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        from pathlib import Path
        # Hierarchical HICRL Integration
        self.hierarchical_manager = HierarchicalContextManager(config.get('hierarchical'), self.tokenizer)
        self.hierarchical_logger = HierarchicalLogger(
            config.get('hierarchical'),
            model_name=Path(config.actor_rollout_ref.model.path).name,
            experiment_name=config.trainer.experiment_name,
            tokenizer=self.tokenizer,
            repeat_n=config.actor_rollout_ref.rollout.n
        )

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        self.val_dataloader = None
        if self.val_dataset is not None:
            val_batch_size = self.config.data.get("val_batch_size", None)  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=self.config.data.get("dataloader_num_workers", 8),
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )
            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
            print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")
        else:
            print(f"Size of train dataloader: {len(self.train_dataloader)}. Validation is disabled.")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        if self.val_dataloader is None:
            return {}
        print(f"--- begin validaion ---")

        reward_tensor_lst = []
        data_source_lst = []
        success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            # Hierarchical HICRL Integration: Inject context and manage tokens
            if self.hierarchical_manager.enabled:
                test_gen_batch = self.hierarchical_manager.prepare_batch(test_gen_batch, split="val")

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            if self.mlmt_enabled and not self.shared_actor:
                print("[MLMT] Validation: using dedicated high-level actor worker.")
            test_output_gen_batch, _ = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    openai_agent=self.openai_agent,
                                                    high_actor_rollout_wg=self.high_actor_rollout_wg,
                                                    )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            reward_extra_infos_dict = result.get("extra_info", {})
            
            # Display rewards
            if reward_extra_infos_dict and "reward" in reward_extra_infos_dict:
                rewards_list = reward_extra_infos_dict["reward"]
                if rewards_list:
                    rewards_array = np.array(rewards_list)
                    mean_reward = rewards_array.mean()
                    min_reward = rewards_array.min()
                    max_reward = rewards_array.max()
                    std_reward = rewards_array.std()
                    print(f"\n🔍 Validation Batch Rewards:")
                    print(f"   Individual: {[f'{r:.2f}' for r in rewards_list[:5]]}...")
                    print(f"   Mean: {mean_reward:.2f}")
                    print(f"   Range: [{min_reward:.2f}, {max_reward:.2f}]")
                    print(f"   Std: {std_reward:.3f}")

            # Hierarchical HICRL Integration: Log detailed samples
            if self.hierarchical_logger.enabled:
                self.hierarchical_logger.log_batch(split="val", batch=test_batch, step=self.global_steps)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        # Hierarchical HICRL Integration: Log validation metrics
        self.hierarchical_logger.log_metrics(metric_dict)

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
            if self.mlmt_enabled and not self.shared_actor:
                high_actor_pool = self.resource_pool_manager.get_resource_pool(Role.HighActorRollout)
                high_actor_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.HighActorRollout],
                    config=self.config.high_actor_rollout_ref,
                    role="high_actor_rollout",
                )
                self.resource_pool_to_cls[high_actor_pool]["high_actor_rollout"] = high_actor_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        if self.mlmt_enabled and not self.shared_actor:
            self.high_actor_rollout_wg = all_wg["high_actor_rollout"]
            self.high_actor_rollout_wg.init_model()
        else:
            self.high_actor_rollout_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        low_actor_local_path = os.path.join(local_global_step_folder, "low_actor")
        high_actor_local_path = os.path.join(local_global_step_folder, "high_actor")

        actor_remote_root = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}")
        low_actor_remote_path = None if actor_remote_root is None else os.path.join(actor_remote_root, "low_actor")
        high_actor_remote_path = None if actor_remote_root is None else os.path.join(actor_remote_root, "high_actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        
        lora_only = self.config.trainer.get("lora_only_save", False)

        self.actor_rollout_wg.save_checkpoint(
            low_actor_local_path,
            low_actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
            lora_only=lora_only,
        )

        if self.mlmt_enabled and not self.shared_actor:
            # Dual-actor setup: persist the dedicated high-level policy.
            self.high_actor_rollout_wg.save_checkpoint(
                high_actor_local_path,
                high_actor_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_actor_ckpt_to_keep,
                lora_only=lora_only,
            )

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        low_actor_path = os.path.join(global_step_folder, "low_actor")
        legacy_actor_path = os.path.join(global_step_folder, "actor")
        if not os.path.exists(low_actor_path):
            if os.path.exists(legacy_actor_path):
                print("Low-level checkpoint not found, falling back to legacy 'actor' directory.")
                low_actor_path = legacy_actor_path
            else:
                raise FileNotFoundError(f"Missing low-level actor checkpoint under {global_step_folder}")
        self.actor_rollout_wg.load_checkpoint(low_actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        if self.mlmt_enabled and not self.shared_actor:
            high_actor_path = os.path.join(global_step_folder, "high_actor")
            if not os.path.exists(high_actor_path):
                if os.path.exists(legacy_actor_path):
                    print("High-level checkpoint not found, falling back to legacy 'actor' directory.")
                    high_actor_path = legacy_actor_path
                else:
                    print("High-level checkpoint missing; initializing from low-level weights.")
                    high_actor_path = low_actor_path
            self.high_actor_rollout_wg.load_checkpoint(
                high_actor_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _split_batches_by_actor(self, batch: DataProto):
        if not self.mlmt_enabled or self.shared_actor:
            return [("actor", None, batch)]

        turns = batch.non_tensor_batch.get("turn", None)
        if turns is None:
            return [("actor", None, batch)]

        turns = np.array(turns)
        actor_batches = []

        low_indices = np.where(turns != 2)[0]
        if low_indices.size > 0:
            actor_batches.append(("low", low_indices.tolist(), batch.select_idxs(low_indices.tolist())))

        high_indices = np.where(turns == 2)[0]
        if high_indices.size > 0:
            actor_batches.append(("high", high_indices.tolist(), batch.select_idxs(high_indices.tolist())))

        if not actor_batches:
            actor_batches.append(("actor", None, batch))
        return actor_batches

    def _assert_turn_metadata(self, batch: DataProto):
        if self.mlmt_enabled and not self.shared_actor:
            if "turn" not in batch.non_tensor_batch:
                raise ValueError("MLMT dual-actor mode requires 'turn' metadata in non_tensor_batch.")

    @staticmethod
    def _prefix_metrics(metrics_dict, prefix):
        if not prefix:
            return metrics_dict
        return {f"{prefix}/{k}": v for k, v in metrics_dict.items()}

    def _train_single_actor(self, batch: DataProto, actor_wg, prefix: str, timing_raw: Dict[str, float], is_last_step: bool):
        local_metrics = {}
        value_training_samples = []
        value_aug_tensor = None
        llm_success_tensor = None
        turn_tensor = None
        reward_tensor = None
        reward_extra_infos_dict = {}
        rewards_list = []
        rewards_array = np.array([])
        llm_success_weight = float(self.mlmt_cfg.get("llm_success_weight", 0.0))

        # Resolve algorithm and freeze status for MLMT
        adv_estimator = self.config.algorithm.adv_estimator
        freeze = False
        if self.mlmt_enabled:
            if prefix == "high":
                adv_estimator = self.mlmt_cfg.high_level.get("algorithm", adv_estimator)
                freeze = bool(self.mlmt_cfg.high_level.get("freeze", False))
            elif prefix == "low":
                adv_estimator = self.mlmt_cfg.low_level.get("algorithm", adv_estimator)
                freeze = bool(self.mlmt_cfg.low_level.get("freeze", False))
        batch = adjust_batch(self.config, batch)
        batch.batch["response_mask"] = compute_response_mask(batch)

        if self.config.trainer.balance_batch:
            balance_metrics = {}
            self._balance_batch(batch, metrics=balance_metrics, logging_prefix="global_seqlen")
            local_metrics.update(self._prefix_metrics(balance_metrics, prefix))

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        turn_indices = batch.non_tensor_batch.get("turn", None)
        if turn_indices is not None:
            turn_tensor = torch.tensor(np.array(turn_indices).astype(np.int64), device=batch.batch["responses"].device)
        value_text_nt = batch.non_tensor_batch.get("value_text", None)
        episode_rewards_nt = batch.non_tensor_batch.get("episode_rewards", None)
        value_aug_tensor = None

        reward_extra_infos_dict = {}
        if self.use_rm:
            reward_tensor = self.rm_wg.compute_rm_score(batch)
            batch = batch.union(reward_tensor)

        if self.config.reward_model.launch_reward_fn_async:
            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
        else:
            reward_start = time.time()
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
            batch.batch["token_level_scores"] = reward_tensor
            print(f"⏱️ Reward Computation (including Judge) took {time.time() - reward_start:.2f}s")

        with _timer("old_log_prob", timing_raw):
            old_log_prob = actor_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
            local_metrics.update(self._prefix_metrics(old_log_prob_metrics, prefix))
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

            if "rollout_log_probs" in batch.batch:
                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                actor_old_log_probs = batch.batch["old_log_probs"]
                attention_mask = batch.batch["attention_mask"]
                responses = batch.batch["responses"]
                response_length = responses.size(1)
                response_mask = attention_mask[:, -response_length:]

                rollout_probs = torch.exp(rollout_old_log_probs)
                actor_probs = torch.exp(actor_old_log_probs)
                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                rollout_diff_metrics = {
                    "training/rollout_probs_diff_max": float(torch.max(rollout_probs_diff).detach().item()),
                    "training/rollout_probs_diff_mean": float(torch.mean(rollout_probs_diff).detach().item()),
                    "training/rollout_probs_diff_std": float(torch.std(rollout_probs_diff).detach().item()),
                }
                local_metrics.update(self._prefix_metrics(rollout_diff_metrics, prefix))

                if self.use_reference_policy:
                    with _timer("ref", timing_raw):
                        ref_worker = self.ref_policy_wg if not self.ref_in_actor else actor_wg
                        ref_log_prob = ref_worker.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                if self.use_critic:
                    with _timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor

                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

        llm_scores = batch.non_tensor_batch.get("llm_success", None)
        if llm_success_weight > 0 and llm_scores is not None:
            llm_success_tensor = torch.tensor(llm_scores, dtype=batch.batch["token_level_scores"].dtype, device=batch.batch["token_level_scores"].device)
            local_metrics.update(self._prefix_metrics({"llm_success/mean": float(llm_success_tensor.mean().item())}, prefix))

        if llm_success_weight > 0 and llm_success_tensor is not None and turn_tensor is not None:
            response_mask = batch.batch["response_mask"]
            for i in range(len(batch.batch)):
                if turn_tensor[i] == 3:
                    response_len = response_mask[i].sum().long()
                    if response_len > 0:
                        batch.batch["token_level_scores"][i, response_len - 1] += llm_success_weight * llm_success_tensor[i]

        if self.mlmt_enabled and turn_tensor is not None:
            from agent_system.multi_turn_rollout.mlmt_utils import apply_symmetric_reaping
            lambda_coef = self.config.get("mlmt_lambda", 0.1)
            use_symmetric = self.config.get("mlmt_use_symmetric_rewards", True)
            response_mask = batch.batch["response_mask"]
            values = batch.batch.get("values", None)

            if use_symmetric:
                pass

            print(f"MLMT-RL Bi-Level Training (Final Reward Only)")

        if reward_extra_infos_dict and "reward" in reward_extra_infos_dict:
            rewards_list = reward_extra_infos_dict["reward"]
            if rewards_list:
                rewards_array = np.array(rewards_list) 
                
                # Check for NaNs/Infs in rewards
                if not np.all(np.isfinite(rewards_array)):
                    print(f"\n\033[1;31m🚨 CRITICAL: Non-finite rewards detected in {prefix or 'actor'}!\033[0m")
                    print(f"   NaNs: {np.isnan(rewards_array).sum()}, Infs: {np.isinf(rewards_array).sum()}")
                    print(f"   Values: {rewards_array}")
                
                prefix_str = prefix or "actor"
                print(f"\n{prefix_str} Rewards:")
                print(f"   Individual: {[f'{r:.2f}' for r in rewards_list[:5]]}...")
                print(f"   Mean: {rewards_array.mean():.2f}")
                print(f"   Range: [{rewards_array.min():.2f}, {rewards_array.max():.2f}]")
                print(f"   Std: {rewards_array.std():.3f}")
                # --- MODIFICATION: BOLD PPRWINT REWARD ---
                print(f"\033[1m🚀 [{prefix_str.upper()}] STEP MEAN REWARD: {rewards_array.mean():.4f}\033[0m")

        belief_scores = None
        mode = None
        if hasattr(self.config.env, "belief_shaped_grpo") and self.config.env.belief_shaped_grpo.get("enable", False):
            belief_scores = batch.non_tensor_batch.get("belief_scores_accumulated", None)
            mode = "shaped"
        elif hasattr(self.config.env, "pure_belief_grpo") and self.config.env.pure_belief_grpo.get("enable", False):
            belief_scores = batch.non_tensor_batch.get("belief_scores_accumulated", None)
            mode = "pure"

        if belief_scores is not None:
            belief_scores = np.array(belief_scores)
            response_mask = batch.batch["response_mask"]
            response_lengths = response_mask.sum(dim=-1)
            for i, resp_len in enumerate(response_lengths):
                if resp_len > 0 and i < len(belief_scores):
                    if mode == "pure":
                        batch.batch["token_level_scores"][i, resp_len - 1] = belief_scores[i]
                    else:
                        batch.batch["token_level_scores"][i, resp_len - 1] += belief_scores[i]
            print(f"📊 Applied belief scores to token_level_scores (mode={mode}, total={belief_scores.sum():.4f})")

        reward_coef = self.config.env.rule_reward_coef
        if not self.config.env.use_dense_reward:
            reward_coef = 0.0
        batch, invalid_metrics = apply_invalid_action_penalty(
            batch,
            reward_coef=reward_coef,
            rule_number=self.config.env.rule_number,
        )
        local_metrics.update(self._prefix_metrics(invalid_metrics, prefix))

        if self.hierarchical_manager.enabled and self.config.hierarchical.kl_max.enabled:
            kl_cfg = self.config.hierarchical.kl_max
            if "old_log_probs" in batch.batch and "ref_log_prob" in batch.batch:
                response_mask = batch.batch["response_mask"]
                logp_ctx = (batch.batch["old_log_probs"] * response_mask).sum(-1)
                logp_noctx = (batch.batch["ref_log_prob"] * response_mask).sum(-1)
                delta_logp = logp_ctx - logp_noctx
                
                tau_prime = kl_cfg.get("clip_tau", 10.0)
                lambda_coef = kl_cfg.get("lambda_coef", 0.0)
                clipped_delta = torch.clamp(delta_logp, max=tau_prime)
                
                response_lengths = response_mask.sum(-1).long()
                for i in range(len(batch.batch)):
                    last_idx = response_lengths[i] - 1
                    if last_idx >= 0:
                        batch.batch["token_level_scores"][i, last_idx] += lambda_coef * clipped_delta[i]

        if self.hierarchical_logger.enabled:
            self.hierarchical_logger.log_batch(split="train", batch=batch, step=self.global_steps)

        if self.config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
            local_metrics.update(self._prefix_metrics(kl_metrics, prefix))
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        if freeze:
            print(f"❄️ {prefix or 'actor'} is frozen. Skipping update.")
            return local_metrics, reward_extra_infos_dict, batch, value_training_samples

        # --- MODIFICATION #2: Alternating Updates ---
        tL = self.config.get("mlmt_rl", {}).get("low_level", {}).get("update_frequency", 1)
        tH = self.config.get("mlmt_rl", {}).get("high_level", {}).get("update_frequency", 1)
        
        is_high_level = (prefix == "high")

        # If 1:1, both are updated at the same time
        if tL == 1 and tH == 1:
            should_update = True
        else:
            cycle_len = tL + tH
            current_step_in_cycle = (self.global_steps) % cycle_len
            
            if is_high_level:
                should_update = (current_step_in_cycle >= tL or tL == 0)
            else:
                should_update = (current_step_in_cycle < tL or tH == 0)
            
        if not should_update:
            print(f"⏳ {prefix or 'actor'} skipping update this step (Freq: tL={tL}, tH={tH}, Step={self.global_steps}, CycleIdx={current_step_in_cycle})")
            return local_metrics, reward_extra_infos_dict, batch, value_training_samples

        #Single HL Update per Step (Average) ---
        # If LL is gRPO (n > 1) and we are updating the High Level, 
        # force one update per batch by setting mini_batch_size to total samples.
        n = self.config.actor_rollout_ref.rollout.n
        if is_high_level and n > 1:
            total_samples = len(batch)
            print(f"⚖️ HL Averaging: LL is gRPO (n={n}), forcing HL to perform 1 update on {total_samples} samples.")
            # We temporarily inject this into the worker's update call if possible, 
            # or just rely on the fact that if we use the whole batch, it's one update.
            batch.meta_info["force_single_update"] = True

        print(f"🚀 Training {prefix or 'actor'} with {adv_estimator} algorithm")
        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
        batch = compute_advantage(
            batch,
            adv_estimator=adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
            compute_mean_std_cross_all_data=self.config.algorithm.compute_mean_std_cross_all_data,
            use_pf_ppo=self.config.algorithm.use_pf_ppo,
            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
            gigpo_mode=self.config.algorithm.gigpo.mode,
                        )

        if self.use_critic and (not self.hierarchical_manager.enabled or self.hierarchical_manager.ll_trainable):
            with _timer("update_critic", timing_raw):
                update_start = time.time()
                critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                local_metrics.update(self._prefix_metrics(critic_output_metrics, prefix))
                print(f"⏱️ Critic Update took {time.time() - update_start:.2f}s")

        if self.config.trainer.critic_warmup <= self.global_steps:
            if not self.hierarchical_manager.enabled or self.hierarchical_manager.ll_trainable:
                with _timer("update_actor", timing_raw):
                    update_start = time.time()
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = actor_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    print(f"⏱️ Actor Update took {time.time() - update_start:.2f}s")
                local_metrics.update(self._prefix_metrics(actor_output_metrics, prefix))

        return local_metrics, reward_extra_infos_dict, batch, value_training_samples

    def _update_value_worker(self, samples):
        if self.value_worker is None or not samples:
            return {}
        update_steps = self.value_cfg.get("update_steps_per_iter", 1) if self.value_cfg else 1
        batch_size = min(len(samples), 32)
        metrics_accum = {}
        for _ in range(update_steps):
            if not samples:
                break
            if len(samples) > batch_size:
                batch_samples = random.sample(samples, batch_size)
            else:
                batch_samples = samples
            texts, targets = zip(*batch_samples)
            update_metrics = self.value_worker.update_value_fn(list(texts), list(targets))
            for k, v in update_metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v
        if metrics_accum:
            for k in metrics_accum:
                metrics_accum[k] /= update_steps
        return metrics_accum

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Hierarchical HICRL Integration: Log run metadata
        self.hierarchical_logger.log_run_meta(self.config)

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # Hierarchical HICRL Integration: Inject context and manage tokens
                if self.hierarchical_manager.enabled:
                    gen_batch = self.hierarchical_manager.prepare_batch(gen_batch, split="train")

                is_last_step = self.global_steps >= self.total_training_steps
                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        ################ agent-environment loop ###############
                        if self.mlmt_enabled and not self.shared_actor:
                            print("[MLMT] Training: using dedicated high-level actor worker.")
                        gen_batch_output, belief_trajectories = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            critique_envs=self.critique_envs,
                            is_train=True,
                            openai_agent=self.openai_agent,
                            high_actor_rollout_wg=self.high_actor_rollout_wg,
                            global_steps=self.global_steps,
                        )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    batch = gen_batch_output

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor

                    self._assert_turn_metadata(batch)
                    actor_batches = self._split_batches_by_actor(batch)
                    reward_extra_infos_dict = {}
                    value_training_buffer = []

                    original_batch = batch
                    # Keys that are typically added or updated during training sub-steps
                    keys_to_update = ['token_level_scores', 'token_level_rewards', 'advantages', 'returns', 'values', 'old_log_probs', 'ref_log_prob']

                    for prefix, indices, actor_batch in actor_batches:
                        actor_wg = self.high_actor_rollout_wg if prefix == "high" else self.actor_rollout_wg
                        prefix_name = prefix if self.mlmt_enabled and not self.shared_actor else None
                        sub_metrics, reward_info, trained_batch, value_samples = self._train_single_actor(actor_batch, actor_wg, prefix_name, timing_raw, is_last_step)
                        metrics.update(sub_metrics)
                        value_training_buffer.extend(value_samples)

                        if indices is not None:
                            # Update original batch with computed tensors for global logging
                            for k in keys_to_update:
                                if k in trained_batch.batch:
                                    val = trained_batch.batch[k]
                                    if k not in original_batch.batch:
                                        shape = (len(original_batch), *val.shape[1:])
                                        original_batch.batch[k] = torch.zeros(shape, dtype=val.dtype, device=val.device)
                                    original_batch.batch[k][indices] = val

                            # Sync non-tensor metadata (rewards, info, etc.)
                            for k, v in trained_batch.non_tensor_batch.items():
                                if k not in original_batch.non_tensor_batch:
                                    original_batch.non_tensor_batch[k] = np.empty(len(original_batch), dtype=object)
                                # Ensure we don't overwrite with None if already present
                                original_batch.non_tensor_batch[k][indices] = v

                            # Aggregate reward extra info for dumping
                            if reward_info:
                                for k, v in reward_info.items():
                                    if k not in reward_extra_infos_dict:
                                        reward_extra_infos_dict[k] = [None] * len(original_batch)
                                    for i, idx in enumerate(indices):
                                        reward_extra_infos_dict[k][idx] = v[i]
                        else:
                            # Single actor case
                            original_batch = trained_batch
                            reward_extra_infos_dict = reward_info

                    batch = original_batch
                    logging_batch = batch

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(logging_batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(logging_batch.batch["responses"], skip_special_tokens=True)
                            scores = logging_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    value_metrics = self._update_value_worker(value_training_buffer)
                    metrics.update(value_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # Log belief curves to wandb if available
                if hasattr(self, 'traj_collector') and belief_trajectories:
                    try:
                        plot_figure = self.traj_collector.plot_belief_curves(belief_trajectories)
                        if plot_figure is not None and "wandb" in self.logger.logger:
                            import wandb
                            # Convert matplotlib figure to wandb Image
                            import io
                            buf = io.BytesIO()
                            plot_figure.savefig(buf, format='png', dpi=1200, bbox_inches='tight')
                            buf.seek(0)
                            plot_figure.close()

                            # Log to wandb
                            self.logger.logger["wandb"]._get_logger().log({
                                "belief_curves": wandb.Image(buf, caption=f"Belief scores over turns (step {self.global_steps})")
                            }, step=self.global_steps)
                            buf.close()
                    except Exception as e:
                        print(f"Warning: Failed to log belief curves to wandb: {e}")

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
