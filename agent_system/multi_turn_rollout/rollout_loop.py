# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import torch
import numpy as np
from verl import DataProto
import matplotlib.pyplot as plt
import os
import wandb
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.torch_functional import get_response_mask
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.multi_turn_rollout.llm_success_evaluator import LLMSuccessEvaluator
from agent_system.environments import EnvironmentManagerBase
from agent_system.critique.critique import *
from agent_system.critique.rule_reward_new import *
from agent_system.belief_calculator import BeliefCalculator
from typing import List, Dict
from tensordict import TensorDict
import time
import sys
import asyncio
import re
from verl.utils.reward_score.math import compute_score_async

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None, high_level_tokenizer: PreTrainedTokenizer = None, shared_actor: bool = True):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.low_level_tokenizer = self.tokenizer
        self.high_level_tokenizer = high_level_tokenizer or tokenizer
        self.shared_actor = shared_actor
        self.processor = processor
        self.mlmt_cfg = self.config.get("mlmt_rl", {})
        self.use_llm_success_eval = bool(self.mlmt_cfg.get("use_llm_success_eval", False))
        self.llm_success_evaluator = None
        if self.use_llm_success_eval:
            eval_cfg = self.mlmt_cfg.get("llm_eval", {})
            self.llm_success_evaluator = LLMSuccessEvaluator(eval_cfg)

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        tokenizer: PreTrainedTokenizer = None,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        tokenizer = tokenizer or self.tokenizer

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        # else:
        #     print(f"Warning: No text observation found!")

        
        chat = np.array([{"content": obs_content, "role": "user"}])
        
        # Apply chat template
        prompt_with_chat_template = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                             tokenizer=tokenizer,
                                                                             max_length=self.config.data.max_prompt_length,
                                                                             pad_token_id=tokenizer.pad_token_id,
                                                                             left_pad=True,
                                                                             truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
        tokenizer: PreTrainedTokenizer = None,
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        tokenizer = tokenizer or self.tokenizer

        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
                tokenizer=tokenizer,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def _build_hybrid_batch_output(self, batch_input_for_training: DataProto, batch_output_for_generation: DataProto, actor_rollout_wg=None) -> DataProto:
        """
        Build complete tensors for training data by properly combining training inputs with generation outputs.
        
        Args:
            batch_input_for_training (DataProto): Batch input for training (without critique)
            batch_output_for_generation (DataProto): Batch containing generated responses
            actor_rollout_wg: Actor rollout worker group to get generation_config from
            
        Returns:
            DataProto: Updated training batch with correct complete sequence tensors
        """
        # Get base tensors from batch_input_for_training
        prompts = batch_input_for_training.batch['input_ids']  # (batch_size, prompt_length) - renamed to match vLLM rollout
        training_attention_mask = batch_input_for_training.batch['attention_mask']  # (batch_size, prompt_length)  
        training_position_ids = batch_input_for_training.batch['position_ids']  # (batch_size, prompt_length) or (batch_size, 3, prompt_length)
        
        # Get generated responses
        responses = batch_output_for_generation.batch['responses']  # (batch_size, response_length)
        rollout_log_probs = batch_output_for_generation.batch['rollout_log_probs']  # (batch_size, response_length)
        batch_size, response_length = responses.shape
        
        # Validate batch_size consistency
        if prompts.size(0) != batch_size:
            raise RuntimeError(f"Batch size mismatch: training batch has {prompts.size(0)}, generation output has {batch_size}")
        
        # 1. Build complete input_ids: [prompts + responses] (consistent with vLLM rollout line 322)
        input_ids = torch.cat([prompts, responses], dim=-1)
        
        # 2. Build complete attention_mask: [prompt_mask + response_mask]
        # Get eos_token_id from actor_rollout_wg, exactly like generate_sequences does
        generation_config = actor_rollout_wg.get_generation_config()[0]
        eos_token_id = generation_config.eos_token_id
        assert eos_token_id is not None, "eos_token_id could not be determined from any source"
        
        response_attention_mask = get_response_mask(
            response_id=responses, 
            eos_token=eos_token_id, 
            dtype=training_attention_mask.dtype
        )
        attention_mask = torch.cat([training_attention_mask, response_attention_mask], dim=-1)
        
        # 3. Build complete position_ids
        # Use exactly the same computation as vLLM rollout (lines 324-335)
        delta_position_id = torch.arange(1, response_length + 1, device=training_position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        
        if training_position_ids.dim() == 3:  # qwen2vl mrope case (lines 327-328)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        
        # Consistent with vLLM rollout line 334
        response_position_ids = training_position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([training_position_ids, response_position_ids], dim=-1)
        
        # 4. Build batch structure consistent with vLLM rollout (lines 340-350)
        # Use TensorDict to ensure consistency with original implementation
        from tensordict import TensorDict
        
        batch = TensorDict(
            {
                "prompts": prompts,                    # Preserve original prompts field
                "responses": responses,                # Preserve responses field  
                "input_ids": input_ids,                     # Complete sequence [prompts + responses]
                "attention_mask": attention_mask,      # Complete attention_mask
                "position_ids": position_ids,         # Complete position_ids
            },
            batch_size=batch_size,
        )
        
        # 5. Add other tensors from generation output (like rollout_log_probs)
        for key, value in batch_output_for_generation.batch.items():
            if key not in ['input_ids', 'attention_mask', 'position_ids', 'responses', 'prompts']:
                batch[key] = value

        # pop "raw_prompt_ids" from batch_input_for_training
        batch_input_for_training.non_tensor_batch.pop("raw_prompt_ids")

        non_tensor_batch = batch_input_for_training.non_tensor_batch.copy()
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def _create_hybrid_tensor(self, key, generated_input_id, generated_attention_mask, generated_position_id, generated_response, prompt_for_generation, prompt_input_id_for_training, prompt_attention_mask_for_training, prompt_position_ids_for_training):
        """
        function to create hybrid tensor for input_ids, attention_mask, position_ids
        """
        import torch
        
        assert generated_input_id.shape == generated_attention_mask.shape == generated_position_id.shape, "generated_input_id, generated_attention_mask, generated_position_id must have the same shape"

        # Check padding token
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        assert pad_token_id is not None, "pad_token_id could not be determined from any source"

        # Calculate prompt length (real length)
        real_old_prompt_length = (prompt_for_generation != pad_token_id).sum().item()
        real_new_prompt_length = (prompt_input_id_for_training != pad_token_id).sum().item()
        new_prompt_mask_length = (prompt_attention_mask_for_training == 1).sum().item()
        assert new_prompt_mask_length == real_new_prompt_length, "New prompt attention_mask effective length inconsistent"

        # Calculate prompt length (including padding)
        old_prompt_length = prompt_for_generation.shape[0]
        new_prompt_length = prompt_input_id_for_training.shape[0]
        assert old_prompt_length == new_prompt_length, "New and old prompt lengths inconsistent"
        prompt_length = old_prompt_length

      
        if key == 'input_ids':
            # input_ids: fully replace prompt part, keep response part unchanged
            # Structure: [new_prompt_part | response_part]
            hybrid_tensor = generated_input_id.clone()

            # Fully replace prompt part
            hybrid_tensor[:prompt_length] = prompt_input_id_for_training[:prompt_length]

            # Verify hybrid results
            hybrid_prompt = hybrid_tensor[:prompt_length]
            original_prompt = prompt_input_id_for_training[:prompt_length]
            assert torch.equal(hybrid_prompt, original_prompt), "Prompt part inconsistent"
            hybrid_response = hybrid_tensor[prompt_length:]
            original_response = generated_input_id[prompt_length:]
            assert torch.equal(hybrid_response, original_response), "Response part inconsistent"
            
            
        elif key == 'attention_mask':
            # attention_mask: use batch for prompt part, rubric_batch for response part
            hybrid_tensor = generated_attention_mask.clone()

            # Verify original data
            prompt_mask_generated = hybrid_tensor[:prompt_length].sum().item()
            # print(f"  Number of valid tokens in prompt part before mixing: {prompt_mask_generated}")

            hybrid_tensor[:prompt_length] = prompt_attention_mask_for_training[:prompt_length]

            # Verify hybrid results
            prompt_mask_hybrid = hybrid_tensor[:prompt_length].sum().item()
            # print(f"  Number of valid tokens in prompt part after mixing: {prompt_mask_hybrid}")
            
        elif key == 'position_ids':
            # position_ids: need to recalculate to maintain continuity
            hybrid_tensor = generated_position_id.clone()

            # 1. Replace prompt part
            hybrid_tensor[:prompt_length] = prompt_position_ids_for_training[:prompt_length]

            # Find valid positions in response part (not pad_token_id positions)
            response_start = prompt_length

            # Double verify response length consistency
            response = generated_input_id[response_start:]
            response_length = (response != pad_token_id).sum().item()
            # print(f"  Valid length of response part: {response_length}")

            attention_mask_for_response = generated_attention_mask[response_start:]
            response_attention_length = (attention_mask_for_response != 0).sum().item()
            # print(f"  Valid attention_mask length of response part: {response_attention_length}")

            assert response_length == response_attention_length, "Response part inconsistent"

            # 2. Renumber response part starting from prompt length, need to add position_id to entire response
            # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
            # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
            full_response_length = generated_response.shape[0]
            hybrid_tensor[response_start:response_start + full_response_length] = torch.arange(
                real_new_prompt_length, real_new_prompt_length + full_response_length,
                dtype=hybrid_tensor.dtype, device=hybrid_tensor.device
            )
                     
        else:
            raise ValueError(f"Unknown field '{key}'")

        # print(f"  Hybrid completed, output tensor shape: {hybrid_tensor.shape}")
        return hybrid_tensor


    def _build_hybrid_batch_output_new(self, batch_input_for_training: DataProto, batch_output_for_generation: DataProto, actor_rollout_wg=None) -> DataProto:
        """
        Build complete tensors for training data by properly combining training inputs with generation outputs.
        
        Args:
            batch_input_for_training (DataProto): Batch input for training (without critique)
            batch_output_for_generation (DataProto): Batch containing generated responses
            actor_rollout_wg: Actor rollout worker group to get generation_config from
            
        Returns:
            DataProto: Updated training batch with correct complete sequence tensors
        """
        # First initialize with a copy of the actual generated batch_output
        import copy
        hybrid_batch_output = DataProto(
            batch=batch_output_for_generation.batch.clone() if batch_output_for_generation.batch is not None else None,
            non_tensor_batch=copy.deepcopy(batch_output_for_generation.non_tensor_batch),
            meta_info=copy.deepcopy(getattr(batch_output_for_generation, 'meta_info', {}))
        )

        # Fully replace: directly use values from batch_input_for_training
        fully_replace_fields = {'prompts', 'raw_prompt'}

        # Keep unchanged: use values from batch_output_for_generation (response-related)
        keep_unchanged_fields = {'responses', 'rollout_log_probs'}

        # Partial replace: fields requiring hybrid processing (prompt part uses batch_input_for_training, response part uses batch_output_for_generation)
        partial_replace_fields = {'input_ids', 'attention_mask', 'position_ids'}

        # Process all fields in hybrid_batch_output.batch, note prompt alignment
        for key, tensor in hybrid_batch_output.batch.items():
            if key in fully_replace_fields:
                # Fully replace
                if key == 'prompts': # prompts in batch_input_for_training is input_ids
                    hybrid_batch_output.batch[key] = batch_input_for_training.batch['input_ids']
                else:
                    hybrid_batch_output.batch[key] = batch_input_for_training.batch[key]
            elif key in keep_unchanged_fields:
                # Keep unchanged
                pass
            elif key in partial_replace_fields:
                hybrid_tensor_list = []
                for i in range(len(tensor)):
                    hybrid_tensor = self._create_hybrid_tensor(
                        key, 
                        generated_input_id=batch_output_for_generation.batch['input_ids'][i],
                        generated_attention_mask=batch_output_for_generation.batch['attention_mask'][i],
                        generated_position_id=batch_output_for_generation.batch['position_ids'][i],
                        generated_response=batch_output_for_generation.batch['responses'][i],
                        prompt_for_generation=batch_output_for_generation.batch['prompts'][i],
                        prompt_input_id_for_training=batch_input_for_training.batch['input_ids'][i],
                        prompt_attention_mask_for_training=batch_input_for_training.batch['attention_mask'][i],
                        prompt_position_ids_for_training=batch_input_for_training.batch['position_ids'][i],
                    )
                    hybrid_tensor_list.append(hybrid_tensor)
                hybrid_batch_output.batch[key] = torch.stack(hybrid_tensor_list, dim=0)
            else:
                raise ValueError(f"Unknown field '{key}'")
        
        # Process hybrid_batch_output.non_tensor_batch
        for key, value in hybrid_batch_output.non_tensor_batch.items():
            if key in fully_replace_fields:
                hybrid_batch_output.non_tensor_batch[key] = batch_input_for_training.non_tensor_batch[key]
            elif key in keep_unchanged_fields:
                pass
            else:
                raise ValueError(f"Unknown field '{key}'")

        return hybrid_batch_output
        

    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def plot_belief_curves(self, completed_belief_trajectories):
        """Plot belief score curves over turns for completed trajectories and return plot data."""
        if not completed_belief_trajectories:
            return None

        # Group trajectories by episode length for plotting
        trajectories_by_length = {}
        for traj in completed_belief_trajectories:
            length = traj['episode_length']
            if length not in trajectories_by_length:
                trajectories_by_length[length] = []
            trajectories_by_length[length].append(traj)

        # Plot curves for different episode lengths
        plt.figure(figsize=(12, 8))

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        plotted_lengths = []

        for i, (length, trajectories) in enumerate(sorted(trajectories_by_length.items())):
            if length in [10, 20, 30, 40, 50]:  # Only plot specified lengths
                color = colors[i % len(colors)]
                plotted_lengths.append(length)

                # Plot multiple trajectories for this length
                for j, traj in enumerate(trajectories[:5]):  # Plot up to 5 trajectories per length
                    belief_scores = traj['trajectory']
                    turns = list(range(1, len(belief_scores) + 1))

                    label = f'Length {length}' if j == 0 else None
                    alpha = 0.7 if j == 0 else 0.3  # Make first trajectory more prominent

                    plt.plot(turns, belief_scores, color=color, alpha=alpha, linewidth=2 if j == 0 else 1,
                           label=label if j == 0 else None)

        if plotted_lengths:
            plt.xlabel('Turn Number')
            plt.ylabel('Belief Score')
            plt.title('Belief Scores Over Turns by Episode Length')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)  # Belief scores are typically 0-1

            # Return the plot figure instead of saving
            return plt.gcf()
        else:
            plt.close()
            return None

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            openai_agent=None,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[List[Dict]]): Complete trajectory data for all environments.
                - Outer List: Length = batch_size, each element represents one environment's trajectory
                - Inner List: Length = number of steps taken, each element represents one timestep
                - Dict: Contains all data for one environment at one timestep, including:
                    * 'input_ids': Input token IDs (torch.Tensor)
                    * 'responses': Generated response token IDs (torch.Tensor) 
                    * 'rewards': Step reward value (float)
                    * 'active_masks': Whether this step is active (bool)
                    * 'uid': Question identifier (str) - multiple trajectories for same question share this
                    * 'traj_uid': Individual trajectory identifier (str) - unique for each trajectory
                    * 'anchor_obs': Anchor observation data (Any)
                    * 'environment_feedback': Feedback from environment (str, if available)
                    * 'question': Question text from environment info (str, if available)
                    * 'ground_truth': Ground truth answer from environment info (str, if available)
                    * 'question_id': Real dataset ID from environment info (str, if available)
                    * Other model inputs/outputs and metadata
            episode_rewards (np.ndarray): Total accumulated rewards for each environment.
                - Shape: (batch_size,), dtype: float32
                - Each element is the sum of all step rewards for that environment's trajectory
            episode_lengths (np.ndarray): Total number of steps taken by each environment.
                - Shape: (batch_size,), dtype: int32  
                - Each element is the count of active steps before termination
            success (Dict[str, np.ndarray]): Success evaluation metrics for each environment.
                - Keys: Metric names (e.g., 'task_success', 'goal_achieved')
                - Values: Boolean arrays of shape (batch_size,) indicating success/failure
            traj_uid (np.ndarray): Unique identifiers for each individual trajectory.
                - Shape: (batch_size,), dtype: object (UUID strings)
                - Each element uniquely identifies one environment's trajectory (different from uid which groups trajectories by question)
        """
        # Initial observations from the environment
        obs, infos = envs.reset()

        # Initialize trajectory collection
        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        if len(gen_batch.batch) != lenght_obs:
            if self.config.env.rollout.n > 0 and envs.is_train: # train mode, rollout n trajectories for each question
                gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            else: # evaluation mode, truncate the gen_batch to the length of obs
                gen_batch = gen_batch.truncate(truncate_length=lenght_obs)
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        batch_size = len(gen_batch.batch['input_ids'])
        batch_output = None
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)

        # Belief-shaped GRPO initialization
        belief_calculator = None
        belief_scores_accumulated = np.zeros(batch_size, dtype=np.float32)
        previous_beliefs = [0.0] * batch_size  # Initialize with 0.0 instead of None
        belief_scores_over_time = [[] for _ in range(batch_size)]  # Track belief scores per step per rollout
        completed_belief_trajectories = []  # Store completed belief trajectories for plotting
        use_pure_belief_grpo = False

        if hasattr(self.config.env, 'pure_belief_grpo') and self.config.env.pure_belief_grpo.get('enable', False):
            alpha = self.config.env.pure_belief_grpo.get('alpha', 1.0)
            max_candidates = self.config.env.pure_belief_grpo.get('max_candidates', 5)
            print(f"🔬 INITIALIZING Pure Belief GRPO with alpha={alpha}, max_candidates={max_candidates}")
            belief_calculator = BeliefCalculator(alpha=alpha, max_candidates=max_candidates)
            # Set up persistent event loop for async operations
            belief_calculator.setup_event_loop()
            use_pure_belief_grpo = True
            print(f"✅ Pure Belief GRPO enabled - USING ONLY BELIEF SCORES AS REWARD")
        elif hasattr(self.config.env, 'belief_shaped_grpo') and self.config.env.belief_shaped_grpo.get('enable', False):
            alpha = self.config.env.belief_shaped_grpo.get('alpha', 1.0)
            max_candidates = self.config.env.belief_shaped_grpo.get('max_candidates', 5)
            print(f"🔬 INITIALIZING Belief-shaped GRPO with alpha={alpha}, max_candidates={max_candidates}")
            belief_calculator = BeliefCalculator(alpha=alpha, max_candidates=max_candidates)
            # Set up persistent event loop for async operations
            belief_calculator.setup_event_loop()
            print(f"✅ Belief-shaped GRPO enabled - ADDING BELIEF SCORES TO TERMINAL REWARDS")

        print(f"🎯 Belief calculator initialized: {belief_calculator is not None}")
        
        # Trajectory collection loop
        # Initialize timing tracking
        step_timings = []
        total_preprocessing_time = 0.0
        total_generation_time = 0.0
        total_environment_time = 0.0
        total_belief_time = 0.0
        total_reward_processing_time = 0.0

        for _step in range(self.config.env.max_steps):
            step_start_time = time.time()

            active_masks = np.logical_not(is_done)
            completed_count = is_done.sum()
            active_count = batch_size - completed_count
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [Rollout Loop] step {_step + 1}: {completed_count}/{batch_size} completed, {active_count} active")

            # Measure preprocessing time
            preprocessing_start = time.time()
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            preprocessing_time = time.time() - preprocessing_start
            total_preprocessing_time += preprocessing_time
            print(f"⏱️ Preprocessing took {preprocessing_time:.3f}s")

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # Measure generation time
            generation_start = time.time()
            # Use OpenAI agent if provided, otherwise use model worker
            if openai_agent is not None:
                print(f"[Rollout Loop] Using OpenAI agent for generation (step {_step + 1})")
                batch_output = openai_agent.generate_sequences(batch_input)
            else:
                print(f"[Rollout Loop] Using local model worker for generation (step {_step + 1})")
                batch_output = actor_rollout_wg.generate_sequences(batch_input)
            generation_time = time.time() - generation_start
            total_generation_time += generation_time
            print(f"⏱️ Generation took {generation_time:.3f}s")

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)

            responses = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

            # Measure environment step time
            env_step_start = time.time()
            next_input, rewards, dones, infos = envs.step(responses)
            env_step_time = time.time() - env_step_start
            total_environment_time += env_step_time
            print(f"⏱️ Environment step took {env_step_time:.3f}s")

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            # Extract environment feedback from infos
            if 'environment_feedback' in infos[0]:
                batch.non_tensor_batch['environment_feedback'] = np.array([info['environment_feedback'] for info in infos], dtype=object)
            else:
                batch.non_tensor_batch['environment_feedback'] = np.array(['' for _ in range(batch_size)], dtype=object)

            # Extract question, ground_truth, and question_id from infos
            if 'question' in infos[0]:
                batch.non_tensor_batch['question'] = np.array([info['question'] for info in infos], dtype=object)
            if 'ground_truth' in infos[0]:
                batch.non_tensor_batch['ground_truth'] = np.array([info['ground_truth'] for info in infos], dtype=object)
            if 'question_id' in infos[0]:
                batch.non_tensor_batch['question_id'] = np.array([info['question_id'] for info in infos], dtype=object)

            # Belief-shaped GRPO: Compute and accumulate belief scores
            if belief_calculator is not None:
                belief_computation_start = time.time()
                print(f"🧠 BELIEF COMPUTATION: Step {_step + 1}, active_envs={active_count}")
                print(f"🎯 Belief mode: {'PURE BELIEF GRPO' if self.config.env.pure_belief_grpo.enable else 'BELIEF-SHAPED GRPO'}")
                belief_scores = np.zeros(batch_size, dtype=np.float32)

                # Debug: Check what data is available
                sample_info = infos[0] if infos and len(infos) > 0 else {}
                print(f"📋 Sample info keys: {list(sample_info.keys()) if sample_info else 'NO_INFOS'}")

                # Collect batch data for belief computation
                batch_belief_data = []
                env_indices = []

                for i in range(batch_size):
                    if active_masks[i]:  # Only compute for active environments
                        question = ""
                        k_t_combined = ""

                        # Extract question and evidence from current step info
                        if 'question' in infos[i]:
                            question = infos[i]['question']
                        if 'K_t_combined' in infos[i]:
                            k_t_combined = infos[i]['K_t_combined']

                        if question and k_t_combined and k_t_combined.strip():
                            batch_belief_data.append((k_t_combined, question, previous_beliefs[i]))
                            env_indices.append(i)
                        else:
                            print(f"  ⏭️ Skipping env {i}: missing/invalid data (question={bool(question)}, k_t='{k_t_combined[:30] if k_t_combined else 'None'}...')")
                            belief_scores[i] = 0.0  # Explicitly set to 0 for missing data

                # Batch compute belief scores
                if batch_belief_data:
                    print(f"🔍 Batch computing belief for {len(batch_belief_data)} environments (step {_step + 1})...")
                    try:
                        # ✅ CORRECT: Use the persistent loop
                        batch_results = belief_calculator._event_loop.run_until_complete(
                            belief_calculator.compute_belief_scores_batch(batch_belief_data)
                        )

                        # Assign results back to belief_scores array
                        for idx, (belief_score, metadata) in enumerate(batch_results):
                            env_i = env_indices[idx]
                            belief_scores[env_i] = belief_score
                            previous_beliefs[env_i] = belief_score
                            top_hyp = metadata.get("top_hypothesis", "unknown")
                            confidence = metadata.get("concentration", 0.0)
                            print(f"  ✅ Belief computed for env {env_i}: score={belief_score:.4f}, top_hyp='{top_hyp}', confidence={confidence:.4f}")

                    except Exception as e:
                        print(f"  ❌ Error in batch belief computation: {e}")
                        # Set all active environments to 0 on error
                        for env_i in env_indices:
                            belief_scores[env_i] = 0.0

                # Accumulate belief scores per rollout (per environment)
                belief_scores_accumulated += belief_scores * torch_to_numpy(active_masks)
                active_envs = active_masks.sum().item()
                print(f"📊 Belief accumulation: {active_envs} active rollouts, per-rollout accumulated: {belief_scores_accumulated}")

                # Track belief scores over time for plotting
                for i in range(batch_size):
                    if active_masks[i]:
                        belief_scores_over_time[i].append(float(belief_scores[i]))

                # Add belief scores to the batch for logging
                batch.non_tensor_batch['belief_scores'] = belief_scores
                belief_time = time.time() - belief_computation_start
                total_belief_time += belief_time
                print(f"⏱️ Belief computation took {belief_time:.3f}s")
            else:
                print(f"🤔 No belief calculator for step {_step + 1}")

            # Measure reward processing time
            reward_processing_start = time.time()

            # Create reward tensor, only assign rewards for active environments
            episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)

            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            # Attach belief accumulation per-env so downstream reward path can see it
            for i in range(batch_size):
                batch_list[i]['belief_scores_accumulated'] = float(belief_scores_accumulated[i])
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            reward_processing_time = time.time() - reward_processing_start
            total_reward_processing_time += reward_processing_time
            print(f"⏱️ Reward processing took {reward_processing_time:.3f}s")

            # Track total step time
            step_total_time = time.time() - step_start_time
            step_timings.append(step_total_time)
            print(f"⏱️ Step {len(step_timings)} total time: {step_total_time:.3f}s")

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Reset previous_beliefs for newly completed environments and store belief trajectories
            if belief_calculator is not None:
                for i in range(batch_size):
                    if dones[i]:  # This environment just completed
                        previous_beliefs[i] = 0.0  # Reset for potential reuse
                        # Store completed belief trajectory for plotting
                        if belief_scores_over_time[i]:  # Only if we have belief data
                            completed_belief_trajectories.append({
                                'trajectory': belief_scores_over_time[i].copy(),
                                'final_belief': belief_scores_accumulated[i],
                                'episode_length': len(belief_scores_over_time[i])
                            })
                        belief_scores_over_time[i] = []  # Reset for potential reuse
                
            # Update observations for next step
            obs = next_input

            # Break if all environments are done
            if is_done.all():
                break

        # Print timing summary
        total_trajectory_time = sum(step_timings)
        avg_step_time = total_trajectory_time / len(step_timings) if step_timings else 0.0
        print("\n🔍 TIMING ANALYSIS SUMMARY:")
        print(f"Total trajectory time: {total_trajectory_time:.3f}s")
        print(f"Average step time: {avg_step_time:.3f}s")
        print(f"Preprocessing: {total_preprocessing_time:.3f}s")
        print(f"Generation: {total_generation_time:.3f}s")
        print(f"Environment: {total_environment_time:.3f}s")
        print(f"Belief: {total_belief_time:.3f}s")
        print(f"Reward processing: {total_reward_processing_time:.3f}s")
        # Calculate percentages
        if total_trajectory_time > 0:
            preprocessing_pct = (total_preprocessing_time / total_trajectory_time) * 100
            generation_pct = (total_generation_time / total_trajectory_time) * 100
            env_step_pct = (total_environment_time / total_trajectory_time) * 100
            belief_pct = (total_belief_time / total_trajectory_time) * 100
            reward_pct = (total_reward_processing_time / total_trajectory_time) * 100
            print(".1f")
            print(".1f")
            print(".1f")
            print(".1f")
            print(".1f")
        # Belief GRPO: Handle reward computation
        if belief_calculator is not None:
            if use_pure_belief_grpo:
                # Pure Belief GRPO: Replace rewards entirely with belief scores
                original_rewards = episode_rewards.copy()
                episode_rewards = belief_scores_accumulated.copy()
                print(f"Pure Belief GRPO: REPLACING all rewards with accumulated belief scores!")
                print(f"  Original rewards: {original_rewards}")
                print(f"  Per-rollout belief scores: {belief_scores_accumulated}")
                print(f"  Final rewards (pure belief): {episode_rewards}")
            else:
                # Belief-shaped GRPO: Add/subtract belief scores based on correctness
                # episode_rewards contains task rewards (0/1 for wrong/correct)
                # Store original task rewards for logging
                original_task_rewards = episode_rewards.copy()

                # Add/subtract belief scores based on correctness
                belief_modifiers = np.zeros_like(belief_scores_accumulated)
                for i in range(len(episode_rewards)):
                    if original_task_rewards[i] > 0:  # Correct answer
                        belief_modifiers[i] = belief_scores_accumulated[i]
                        episode_rewards[i] += belief_scores_accumulated[i]
                    else:  # Wrong answer
                        belief_modifiers[i] = -belief_scores_accumulated[i]
                        episode_rewards[i] -= belief_scores_accumulated[i]

                # Store pre-normalization rewards for logging
                pre_norm_rewards = episode_rewards.copy()

                # Normalize to [-1, 1] range to control variance
                max_expected_range = 10.0  # Expected max belief score accumulation
                episode_rewards = np.clip(episode_rewards, -max_expected_range, max_expected_range)
                episode_rewards = episode_rewards / max_expected_range

                print(f"Belief-shaped GRPO: Add/subtract belief scores based on correctness")
                print(f"  Original task rewards: {original_task_rewards}")
                print(f"  Belief modifiers: {belief_modifiers}")
                print(f"  Pre-normalization rewards: {pre_norm_rewards}")
                print(f"  Final rewards (normalized to [-1,1]): {episode_rewards}")

                # Compute and log GRPO advantages for sample groups
                group_size = 8  # GRPO group size
                num_groups = len(episode_rewards) // group_size

                if num_groups > 0:
                    # Show advantages for first group
                    group_rewards = episode_rewards[:group_size]
                    group_mean = np.mean(group_rewards)
                    group_std = np.std(group_rewards) + 1e-6  # Add epsilon for stability
                    group_advantages = (group_rewards - group_mean) / group_std

                    print(f"🎯 Belief-Shaped GRPO Advantages (Group 1, {group_size} trajectories):")
                    print(f"  Group rewards: {group_rewards}")
                    print(f"  Group mean: {group_mean:.3f}, std: {group_std:.3f}")
                    print(f"  GRPO advantages: {group_advantages}")

                    # Compare with what vanilla GRPO would give (using original task rewards)
                    original_group_rewards = original_task_rewards[:group_size]
                    orig_mean = np.mean(original_group_rewards)
                    orig_std = np.std(original_group_rewards) + 1e-6
                    orig_advantages = (original_group_rewards - orig_mean) / orig_std

                    print(f"🔄 Vanilla GRPO Advantages (same group, original rewards only):")
                    print(f"  Original rewards: {original_group_rewards}")
                    print(f"  GRPO advantages: {orig_advantages}")

            # Store accumulated belief scores for debugging/logging purposes
            # Note: Belief scores are already incorporated into episode_rewards above
            # Also propagated per-timestep in total_batch_list entries
            if 'belief_scores_accumulated' not in batch.non_tensor_batch:
                batch.non_tensor_batch['belief_scores_accumulated'] = belief_scores_accumulated
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        # Log success rate to wandb
        if success:
            success_rate = np.mean(success.get('success', np.array([])))
            print(f"📊 Success Rate: {success_rate:.3f}")
            try:
                wandb.log({"rollout/success_rate": success_rate})
            except:
                pass  # wandb might not be initialized

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, completed_belief_trajectories
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, _ = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid = filter_group_data(batch_list=batch_list,
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid

    def mlmt_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            high_actor_rollout_wg=None,
            openai_agent=None,
            global_steps=0,
            ) -> DataProto:
        """
        MLMT-RL three-turn loop:
        1. Turn 1 (Solver): sample z ~ π_L(·|x)
        2. Turn 2 (Feedback): sample g ~ π_H(·|x, z)
        3. Turn 3 (Refine): sample ŷ ~ π_L(·|x, z, g)
        """
        loop_start = time.time()
        from agent_system.multi_turn_rollout.mlmt_utils import (
            prepare_mlmt_turn1_prompt,
            prepare_mlmt_feedback_prompt,
            prepare_mlmt_refinement_prompt,
            prepare_mlmt_code_turn1_prompt,
            prepare_mlmt_code_feedback_prompt,
            prepare_mlmt_code_refinement_prompt
        )

        # Choose actor workers
        if high_actor_rollout_wg is None:
            high_actor_rollout_wg = actor_rollout_wg

        low_tokenizer = self.low_level_tokenizer
        high_tokenizer = self.high_level_tokenizer

        # Initial observations from the environment
        obs, infos = envs.reset()
        base_batch_size = len(gen_batch.batch['input_ids'])
        
        # Optimized Sampling Logic based on Case 1 / Case 2:
        # We branch (n > 1) ONLY if the lower level is trainable AND uses gRPO.
        is_low_trainable = self.mlmt_cfg.get('low_level', {}).get('algorithm', 'none') != 'none'
        low_algo = self.mlmt_cfg.get('low_level', {}).get('algorithm', 'none')
        
        if is_low_trainable and low_algo == 'grpo':
            n_to_use = self.config.env.rollout.n if self.config.env.rollout.n > 0 else 8
        else:
            # Case 1 (Frozen) or LL-REINFORCE: No branching.
            n_to_use = 1
            
        n = n_to_use
        total_samples = base_batch_size * n
        
        # Determine environment type
        data_sources = gen_batch.non_tensor_batch.get('data_source', ['math'] * base_batch_size)
        is_code = 'mbpp' in data_sources[0].lower() or 'code' in data_sources[0].lower()
        # Turn 1: Solver initial attempt z
        # Prepare prompts for Turn 1
        questions = []
        ground_truths = []
        for i, info in enumerate(infos):
            # Try to get question from info, then fallback to gen_batch
            q = None
            gt = None
            if isinstance(info, dict):
                q = info.get('question')
                gt = info.get('ground_truth')
            
            if q is None:
                # In verl, non_tensor_batch items are usually lists
                prompts = gen_batch.non_tensor_batch.get('prompt')
                if prompts and i < len(prompts):
                    q = prompts[i]
            
            if q is None:
                # Final fallback to raw_prompt if available
                raw_prompts = gen_batch.non_tensor_batch.get('raw_prompt')
                if raw_prompts and i < len(raw_prompts):
                    q = raw_prompts[i]
            
            if q is None:
                q = "" # Extreme fallback
            
            if gt is None:
                # Fallback for ground truth from non_tensor_batch
                gts = gen_batch.non_tensor_batch.get('answer') or gen_batch.non_tensor_batch.get('ground_truth')
                if gts and i < len(gts):
                    gt = gts[i]

            questions.append(q)
            ground_truths.append(gt)

        if is_code:
            turn1_prompts = [prepare_mlmt_code_turn1_prompt(q) for q in questions]
        else:
            turn1_prompts = [prepare_mlmt_turn1_prompt(q) for q in questions]
        
        # For Turn 1
        turn1_obs = {'text': turn1_prompts, 'image': None, 'anchor': None}
        turn1_batch = self.preprocess_batch(gen_batch=gen_batch, obs=turn1_obs, tokenizer=low_tokenizer)
        
        # Generate Turn 1 response z
        batch_input_t1 = turn1_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids"]
        )
        batch_input_t1.meta_info = gen_batch.meta_info
        # Set branching multiplicity for Turn 1
        batch_input_t1.meta_info['n'] = n
        
        env_name = 'Code' if is_code else 'Math'
        print(f"[MLMT Loop] Step {global_steps} ({env_name}): Generating initial solutions z (n={n})...")
        
        # Timing instrumentation
        t1_start = time.time()
        
        t1_start = time.time()
        batch_output_t1 = actor_rollout_wg.generate_sequences(batch_input_t1)
        print(f"⏱️ Turn 1 Generation took {time.time() - t1_start:.2f}s")
        
        # Expansion: if n > 1, repeat input questions and metadata to match output
        if n > 1:
            expanded_questions = []
            expanded_gts = []
            for q, gt in zip(questions, ground_truths):
                expanded_questions.extend([q] * n)
                expanded_gts.extend([gt] * n)
            questions = expanded_questions
            ground_truths = expanded_gts
            turn1_batch = turn1_batch.repeat(n)
            
        z_responses = low_tokenizer.batch_decode(batch_output_t1.batch['responses'], skip_special_tokens=True)
        
        # Turn 2: Feedback policy g
        if is_code:
            turn2_prompts = [prepare_mlmt_code_feedback_prompt(q, z) for q, z in zip(questions, z_responses)]
        else:
            turn2_prompts = [prepare_mlmt_feedback_prompt(q, z) for q, z in zip(questions, z_responses)]
        turn2_obs = {'text': turn2_prompts, 'image': None, 'anchor': None}
        
        # Note: we need a DataProto of size total_samples for the next stages
        # We use a dummy gen_batch of the right size
        dummy_gen_batch = gen_batch.repeat(n) if n > 1 else gen_batch
        turn2_batch = self.preprocess_batch(gen_batch=dummy_gen_batch, obs=turn2_obs, tokenizer=high_tokenizer)
        
        batch_input_t2 = turn2_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids"]
        )
        batch_input_t2.meta_info = dummy_gen_batch.meta_info
        # Set max_tokens for feedback generation to limit response length
        batch_input_t2.meta_info['max_tokens'] = 512
        # Turn 2 is always REINFORCE (1-to-1)
        batch_input_t2.meta_info['n'] = 1

        print(f"[MLMT Loop] Step 2: Generating feedback g...")
        
        # For multi-turn feedback, we must remove the n=1 override as verl dispatcher only accepts DataProto
        t2_start = time.time()
        batch_output_t2 = high_actor_rollout_wg.generate_sequences(batch_input_t2)
        print(f"⏱️ Turn 2 Generation took {time.time() - t2_start:.2f}s")

        # If n > 1, the worker generated n feedbacks for each input solution.
        # We only need 1-to-1 mapping, so we take the first feedback of each group.
        if n > 1:
            batch_output_t2 = batch_output_t2[0:len(batch_output_t2):n]
            
        g_feedbacks = high_tokenizer.batch_decode(batch_output_t2.batch['responses'], skip_special_tokens=True)
        
        # Turn 3: Refinement policy y_hat
        if is_code:
            turn3_prompts = [prepare_mlmt_code_refinement_prompt(q, z, g) for q, z, g in zip(questions, z_responses, g_feedbacks)]
        else:
            turn3_prompts = [prepare_mlmt_refinement_prompt(q, g) for q, g in zip(questions, g_feedbacks)]
        turn3_obs = {'text': turn3_prompts, 'image': None, 'anchor': None}
        turn3_batch = self.preprocess_batch(gen_batch=dummy_gen_batch, obs=turn3_obs, tokenizer=low_tokenizer)
        
        batch_input_t3 = turn3_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids"]
        )
        batch_input_t3.meta_info = dummy_gen_batch.meta_info
        # Set max_tokens for Turn 3 to limit response length and prevent hallucinations
        batch_input_t3.meta_info['max_tokens'] = 1024
        # Turn 3 is always 1-to-1 mapping
        batch_input_t3.meta_info['n'] = 1
        
        print(f"[MLMT Loop] Step 3: Generating refined solutions y_hat...")
        
        t3_start = time.time()
        batch_output_t3 = actor_rollout_wg.generate_sequences(batch_input_t3)
        print(f"⏱️ Turn 3 Generation took {time.time() - t3_start:.2f}s")

        # Again, slice to maintain 1-to-1 mapping if n > 1
        if n > 1:
            batch_output_t3 = batch_output_t3[0:len(batch_output_t3):n]
            
        y_hat_responses = low_tokenizer.batch_decode(batch_output_t3.batch['responses'], skip_special_tokens=True)
        
        # Get rewards from environment for the final refined responses
        _, rewards, dones, infos = envs.step(y_hat_responses)

        # --- MODIFICATION: Define correctness for logging ---
        async def get_turn1_correctness():
            tasks = [compute_score_async(z, gt) for z, gt in zip(z_responses, ground_truths)]
            return await asyncio.gather(*tasks)
            
        try:
            # We use the existing asyncio loop or create a new one for this synchronous-looking call
            turn1_correctness = asyncio.run(get_turn1_correctness())
        except:
            turn1_correctness = [0.0] * total_samples
            
        turn3_correctness = [float(r) for r in rewards]

        # Logging for inspection
        try:
            import json
            with open("trajectories.jsonl", "a") as f:
                for i in range(len(questions)):
                    log_entry = {
                        "question": questions[i],
                        "turn1_z": z_responses[i],
                        "turn2_g": g_feedbacks[i],
                        "turn3_y_hat": y_hat_responses[i],
                        "reward": float(rewards[i])
                    }
                    f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Failed to log trajectories: {e}")
        
        # total_batch_list[env_idx][step_idx]
        total_batch_list = [[] for _ in range(total_samples)]
        traj_uids = [str(uuid.uuid4()) for _ in range(total_samples)]
        
        # Use a base UID for the question, but append turn for GRPO grouping if needed
        if n > 0: # env grouping
            base_uids = []
            for i in range(base_batch_size):
                uid = str(uuid.uuid4())
                base_uids.extend([uid] * n)
        else:
            uid = str(uuid.uuid4())
            base_uids = [uid for _ in range(total_samples)]
        
        value_texts = [
            f"Question: {questions[i]}\nInitial Solution: {z_responses[i]}\nFeedback: {g_feedbacks[i]}\nRefined Solution: {y_hat_responses[i]}"
            for i in range(total_samples)
        ]
        
        # Success metrics
        episode_rewards = rewards.numpy() if isinstance(rewards, torch.Tensor) else rewards
        
        # --- Consolidated Trajectory Logging ---
        try:
            import json
            import os
            trajectory_log_dir = "logs/trajectories"
            os.makedirs(trajectory_log_dir, exist_ok=True)
            log_path = os.path.join(trajectory_log_dir, "current_rollout.jsonl")
            with open(log_path, "a") as f:
                for i in range(total_samples):
                    data = {
                        "question": questions[i],
                        "ground_truth": ground_truths[i],
                        "turn1_z": z_responses[i],
                        "turn1_correct": bool(turn1_correctness[i]),
                        "turn2_g": g_feedbacks[i],
                        "turn3_y_hat": y_hat_responses[i],
                        "turn3_correct": bool(turn3_correctness[i]),
                        "reward": float(episode_rewards[i])
                    }
                    f.write(json.dumps(data) + "\n")
        except Exception as e:
            print(f"Trajectory logging failed: {e}")
        
        episode_lengths = np.array([3] * total_samples)
        traj_uid_arr = np.array(traj_uids, dtype=object)
        
        llm_success_scores = np.zeros(total_samples, dtype=np.float32)
        llm_feedback_scores = np.zeros(total_samples, dtype=np.float32)
        if self.use_llm_success_eval and self.llm_success_evaluator is not None:
            try:
                # Evaluate all 1024 refinements
                eval_results = self.llm_success_evaluator.evaluate_batch(questions, y_hat_responses)
                if eval_results:
                    llm_success_scores = np.array([res.get('success', 0.0) for res in eval_results], dtype=np.float32)
                    llm_feedback_scores = np.array([res.get('feedback_quality', 0.0) for res in eval_results], dtype=np.float32)
            except Exception as exc:
                print(f"LLM success evaluation failed: {exc}")

        # Process Turn 1
        t1_batch_full = turn1_batch.union(batch_output_t1)
        if 'prompts' not in t1_batch_full.batch:
            t1_batch_full.batch['prompts'] = turn1_batch.batch['input_ids']
            
        t1_list = to_list_of_dict(t1_batch_full)
        for i in range(total_samples):
            t1_list[i]['turn'] = 1
            t1_list[i]['active_masks'] = True
            t1_list[i]['uid'] = f"{base_uids[i]}_turn1"
            t1_list[i]['traj_uid'] = traj_uids[i]
            t1_list[i]['episode_rewards'] = float(episode_rewards[i]) 
            t1_list[i]['episode_lengths'] = 3.0 
            t1_list[i]['value_text'] = ""
            t1_list[i]['llm_success'] = float(llm_success_scores[i])
            t1_list[i]['llm_feedback_quality'] = float(llm_feedback_scores[i])
            total_batch_list[i].append(t1_list[i])
            
        # Process Turn 2
        t2_batch_full = turn2_batch.union(batch_output_t2)
        if 'prompts' not in t2_batch_full.batch:
            t2_batch_full.batch['prompts'] = turn2_batch.batch['input_ids']
            
        t2_list = to_list_of_dict(t2_batch_full)
        for i in range(total_samples):
            t2_list[i]['turn'] = 2
            t2_list[i]['active_masks'] = True
            t2_list[i]['uid'] = f"{base_uids[i]}_turn2"
            t2_list[i]['traj_uid'] = traj_uids[i]
            t2_list[i]['episode_rewards'] = float(episode_rewards[i])
            t2_list[i]['episode_lengths'] = 3.0
            t2_list[i]['value_text'] = value_texts[i]
            t2_list[i]['llm_success'] = float(llm_success_scores[i])
            t2_list[i]['llm_feedback_quality'] = float(llm_feedback_scores[i])
            total_batch_list[i].append(t2_list[i])
            
        # Process Turn 3
        t3_batch_full = turn3_batch.union(batch_output_t3)
        if 'prompts' not in t3_batch_full.batch:
            t3_batch_full.batch['prompts'] = turn3_batch.batch['input_ids']
            
        t3_list = to_list_of_dict(t3_batch_full)
        for i in range(total_samples):
            t3_list[i]['turn'] = 3
            t3_list[i]['active_masks'] = True
            t3_list[i]['uid'] = f"{base_uids[i]}_turn3"
            t3_list[i]['traj_uid'] = traj_uids[i]
            t3_list[i]['episode_rewards'] = float(episode_rewards[i])
            t3_list[i]['episode_lengths'] = 3.0
            t3_list[i]['value_text'] = ""
            t3_list[i]['llm_success'] = float(llm_success_scores[i])
            t3_list[i]['llm_feedback_quality'] = float(llm_feedback_scores[i])
            total_batch_list[i].append(t3_list[i])
            
        success = {
            'success': (episode_rewards > 0),
            'llm_success': llm_success_scores,
            'llm_feedback_quality': llm_feedback_scores
        }
        
        print(f"⏱️ Total MLMT Rollout Loop took {time.time() - loop_start:.2f}s")
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid_arr, []

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            critique_envs: EnvironmentManagerBase = None,
            is_train: bool = True,
            openai_agent=None,
            high_actor_rollout_wg=None,
            global_steps=0,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        elif self.mlmt_cfg.get('enable', False) and is_train:
            # MLMT-RL Sampling
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, belief_trajectories = \
                self.mlmt_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                high_actor_rollout_wg=high_actor_rollout_wg,
                openai_agent=openai_agent,
                global_steps=global_steps,
            )
        elif self.config.env.use_critique and is_train:
            # Critique Sampling
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.critique_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                critique_envs=critique_envs,
            )
        elif self.config.env.use_rule_reward and is_train:
            # Rule Reward Sampling
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.rule_reward_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, belief_trajectories = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                openai_agent=openai_agent,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
        )
        
        return gen_batch_output, belief_trajectories

    def critique_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            critique_envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct rollout with critique generation for each question.
        First performs normal rollout like vanilla, then calls critique function to generate 
        critique for each question based on the collected trajectories.
        
        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.
            critique_envs (EnvironmentManagerBase): Critique environment manager instance.
        Returns:
            tuple: Same as vanilla_multi_turn_loop plus critique data
        """
        # Perform first normal rollout 
        total_batch_list, episode_rewards, episode_lengths, success, traj_uid, _ = \
            self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        
        print(f"Vanilla rollout done, total_batch_list size: {len(total_batch_list)}.")
        
        # Generate critiques for each question
        critique_data = organize_trajectory_data_for_critique(
            total_batch_list=total_batch_list,
            gen_batch=gen_batch,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            success=success,
            traj_uid=traj_uid,
            tokenizer=self.tokenizer,
        )
        critique_results = critique(
            critique_data=critique_data,
            use_ground_truth=self.config.algorithm.get('use_ground_truth', True),
        )
        
        # Perform second rollout with critiques
        critique_batch_list, critique_episode_rewards, critique_episode_lengths, critique_success, critique_traj_uid = \
            self._critique_vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                critique_envs=critique_envs,
                critique_results=critique_results,
            )

        print(f"Critique rollout done, critique_batch_list size: {len(critique_batch_list)}.")
        
        # Combine rollout results: replace first k trajectories of each question with critique trajectories
        combined_batch_list, combined_episode_rewards, combined_episode_lengths, combined_success, combined_traj_uid = \
            combine_vanilla_and_critique_trajectories(
                vanilla_results=(total_batch_list, episode_rewards, episode_lengths, success, traj_uid),
                critique_results=(critique_batch_list, critique_episode_rewards, critique_episode_lengths, critique_success, critique_traj_uid),
                k=self.config.env.rollout.k,
                n=self.config.env.rollout.n
            )

        print(f"Final rollout done, combined_batch_list size: {len(combined_batch_list)}.")
        
        return combined_batch_list, combined_episode_rewards, combined_episode_lengths, combined_success, combined_traj_uid

    def rule_reward_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct rollout with rule reward for each question.
        
        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.
        Returns:
            tuple: Same as vanilla_multi_turn_loop plus rule reward data
        """
        # Perform first normal rollout 
        total_batch_list, episode_rewards, episode_lengths, success, traj_uid, _ = \
            self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        
        # Generate rule reward for each question
        trajectory_data = organize_trajectory_data_for_rule_reward(
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            success=success,
            traj_uid=traj_uid,
            tokenizer=self.tokenizer,
        )
        rule_reward_results = rule_reward(
            trajectory_data=trajectory_data,
            use_ground_truth=self.config.env.get('use_ground_truth', True),
        )
        
        # Combine rollout results: add reward information to returned data
        new_batch_list, new_episode_rewards, new_episode_lengths, new_success, new_traj_uid = \
            add_rule_reward_to_trajectories(
                vanilla_results=(total_batch_list, episode_rewards, episode_lengths, success, traj_uid),
                rule_reward_results=rule_reward_results,
                reward_coef=self.config.env.rule_reward_coef,
                dense_reward=self.config.env.use_dense_reward,
            )

        return new_batch_list, new_episode_rewards, new_episode_lengths, new_success, new_traj_uid


    def _critique_vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            critique_envs: EnvironmentManagerBase,
            critique_results: Dict[str, Dict],
    ) -> tuple:
        """
        Perform rollout with critique feedback using critique_envs.
        
        Args:
            gen_batch (DataProto): Original batch data
            actor_rollout_wg: Actor model workers
            critique_envs (EnvironmentManagerBase): Environment manager with k rollouts per question
            critiques (List[str]): Generated critiques for each question
            critique_data (List[Dict]): Organized critique data containing question info
            
        Returns:
            Same format as vanilla_multi_turn_loop: batch_list, episode_rewards, episode_lengths, success, traj_uid
        """
        
        # Reset critique environments with critique feedback
        # We need to manually reset the underlying environments with critique
        questions = []
        question_ids = []
        ground_truths = []
        critiques = []
        
        # Extract questions and corresponding critiques from critique_data (now a dictionary)
        for question_uid, critique_result in critique_results.items():
            question = critique_result['question']
            question_id = critique_result['question_id']
            ground_truth = critique_result['ground_truth']
            critique = critique_result['critique']
            
            questions.append(question)
            question_ids.append(question_id)
            ground_truths.append(ground_truth)
            critiques.append(critique)
        
        # Reset the underlying environments with critiques
        # We directly call the underlying environment's reset method with critique parameter
        obs, infos = critique_envs.envs.reset(
            questions=questions,
            question_ids=question_ids,
            ground_truths=ground_truths,
            critiques=critiques,
        )
        # Create observation dict in the expected format
        obs = {'text': obs, 'image': None, 'anchor': obs}
        
        # Create observation dict without critique to replace input for training
        obs_wo_critique = []
        for i in range(len(infos)):
            info = infos[i]
            obs_wo_critique.append(info['input_wo_critique'])
        obs_wo_critique = {'text': obs_wo_critique, 'image': None, 'anchor': obs_wo_critique}
        assert len(obs_wo_critique['text']) == len(obs['text']), f"obs_wo_critique size {len(obs_wo_critique['text'])} does not match obs size {len(obs['text'])}"
                
        # Initialize trajectory collection
        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        if len(gen_batch.batch) != lenght_obs:
            if self.config.env.rollout.k > 0 and critique_envs.is_train: # train mode, rollout k trajectories for each question
                gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.k, interleave=True)
            else: # evaulation mode, truncate the gen_batch to the length of obs
                gen_batch = gen_batch.truncate(truncate_length=lenght_obs)
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        batch_size = len(gen_batch.batch['input_ids'])
        batch_output = None
        
        # Reuse original UIDs from critique_results instead of creating new ones
        uid_batch = []
        question_uids = list(critique_results.keys())  # Get the original question UIDs
        
        assert self.config.env.rollout.k > 0, "critique rollout requires env grouping k > 0"
        # With env grouping: multiple trajectories per question
        for i in range(batch_size):
            # Map each environment to its corresponding question UID
            question_idx = i // self.config.env.rollout.k
            assert question_idx < len(question_uids), f"question_idx {question_idx} >= len(question_uids) {len(question_uids)}"
            uid_batch.append(question_uids[question_idx])

        uid_batch = np.array(uid_batch, dtype=object)

        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            
            active_masks = np.logical_not(is_done)
            completed_count = is_done.sum()
            active_count = batch_size - completed_count
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [Critique Rollout Loop] step {_step + 1}: {completed_count}/{batch_size} completed, {active_count} active")

            # Use obs (with critique) for LLM generation
            batch_for_generation = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            
            # Use obs_wo_critique for training data assembly
            # TODO: for debugging, not change input here, keep the input for training the same as the input for generation
            # batch_for_training = self.preprocess_batch(gen_batch=gen_batch, obs=obs_wo_critique)
            batch_for_training = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
                         
            # Debug logging for observation updates
            if _step in [0, 2]:
                with open(f"/home/jjiahe/code/verl-agent_new/input_w_critique.txt", "a") as f:
                    f.write(f"Step {_step + 1}:\n")
                with open(f"/home/jjiahe/code/verl-agent_new/input_wo_critique.txt", "a") as f:
                    f.write(f"Step {_step + 1}:\n")
                
                for i in range(0, min(4, len(obs['text'])), 4):
                    with open(f"/home/jjiahe/code/verl-agent_new/input_w_critique.txt", "a") as f:
                        f.write(f"Input {i}: {obs['text'][i]}\n")
                    with open(f"/home/jjiahe/code/verl-agent_new/input_wo_critique.txt", "a") as f:
                        f.write(f"Input {i}: {obs_wo_critique['text'][i]}\n")
                
                with open(f"/home/jjiahe/code/verl-agent_new/input_w_critique.txt", "a") as f:
                    f.write("-" * 60 + "\n")
                with open(f"/home/jjiahe/code/verl-agent_new/input_wo_critique.txt", "a") as f:
                    f.write("-" * 60 + "\n")

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch_for_generation.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch_for_generation.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch_for_generation.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            
            # Extract input for generation using obs (with critique)
            batch_input = batch_for_generation.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            dummy_batch_input = batch_for_training.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info
            dummy_batch_input.meta_info = gen_batch.meta_info
                        
            batch_output = actor_rollout_wg.generate_sequences(batch_input)
            
            if self.config.env.replace_input:
                batch_output = self._build_hybrid_batch_output_new(batch_input_for_training=dummy_batch_input, batch_output_for_generation=batch_output, actor_rollout_wg=actor_rollout_wg)
                        
            # Add uid and traj_uid to the batch
            batch_for_generation.non_tensor_batch['uid'] = uid_batch
            batch_for_generation.non_tensor_batch['traj_uid'] = traj_uid
    
            batch_for_training.non_tensor_batch['uid'] = uid_batch
            batch_for_training.non_tensor_batch['traj_uid'] = traj_uid
            
            if self.config.env.replace_input:
                batch = batch_for_training
            else:
                batch = batch_for_generation
                
            batch = batch.union(batch_output)
            
            responses = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_input, rewards, dones, infos = critique_envs.step(responses)

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            # Extract environment feedback from infos
            if 'environment_feedback' in infos[0]:
                batch.non_tensor_batch['environment_feedback'] = np.array([info['environment_feedback'] for info in infos], dtype=object)
            else:
                batch.non_tensor_batch['environment_feedback'] = np.array(['' for _ in range(batch_size)], dtype=object)

            # Extract question, ground_truth, and question_id from infos
            if 'question' in infos[0]:
                batch.non_tensor_batch['question'] = np.array([info['question'] for info in infos], dtype=object)
            if 'ground_truth' in infos[0]:
                batch.non_tensor_batch['ground_truth'] = np.array([info['ground_truth'] for info in infos], dtype=object)
            if 'question_id' in infos[0]:
                batch.non_tensor_batch['question_id'] = np.array([info['question_id'] for info in infos], dtype=object)


            # Create reward tensor, only assign rewards for active environments
            episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_input
            # Create observation dict without critique to replace input for training
            obs_wo_critique = []
            for i in range(len(infos)):
                info = infos[i]
                obs_wo_critique.append(info['input_wo_critique'])
            obs_wo_critique = {'text': obs_wo_critique, 'image': None, 'anchor': obs_wo_critique}
            assert len(obs_wo_critique['text']) == len(obs['text']), f"obs_wo_critique size {len(obs_wo_critique['text'])} does not match obs size {len(obs['text'])}"

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = critique_envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid
    
