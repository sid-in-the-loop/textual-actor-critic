"""
OpenAI Agent Wrapper for DeepResearch Trajectory Collection

Uses CMU Gateway API to generate responses via GPT-5-mini instead of local model.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional
from verl import DataProto
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from openai import OpenAI
import time

# CMU Gateway configuration
CMU_GATEWAY_BASE_URL = "https://ai-gateway.andrew.cmu.edu"
# Prefer the single user-provided key by default
GATEWAY_API_KEY = os.getenv("CMU_GATEWAY_API_KEY", "sk-dUplmEab2H7EFRaOISG1Ew")
MODEL_NAME = "gpt-4o-mini-2024-07-18"


class OpenAIAgentWorker:
    """
    OpenAI agent worker that replaces the actor rollout worker for generation.
    Compatible with the existing rollout loop interface.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, config: Optional[Dict] = None):
        """
        Initialize OpenAI agent.
        
        Args:
            tokenizer: Tokenizer for encoding/decoding (needed for DataProto format)
            config: Optional config dict with generation parameters
        """
        self.tokenizer = tokenizer
        self.config = config or {}
        
        # Initialize OpenAI client
        # Prefer the single user-provided CMU gateway key (set in environment)
        env_key = os.getenv("CMU_GATEWAY_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_key = GATEWAY_API_KEY
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=CMU_GATEWAY_BASE_URL
        )
        
        # Generation parameters
        self.temperature = self.config.get('temperature', 0.7)

        
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Generate sequences using OpenAI API.
        
        Args:
            prompts: DataProto containing input_ids, attention_mask, etc.
            
        Returns:
            DataProto with same structure as model rollout (responses, input_ids, etc.)
        """
        batch_size = prompts.batch['input_ids'].shape[0]
        device = prompts.batch['input_ids'].device
        
        # Log that we're using OpenAI API
        print(f"[OpenAI Agent] Generating {batch_size} sequences via {MODEL_NAME} API (CMU Gateway)")
        
        # Extract prompts and decode to text
        input_ids = prompts.batch['input_ids'].cpu()
        attention_mask = prompts.batch['attention_mask'].cpu()
        
        # Decode each prompt
        prompt_texts = []
        for i in range(batch_size):
            # Get valid tokens (non-padded)
            valid_mask = attention_mask[i].bool()
            valid_input_ids = input_ids[i][valid_mask]
            prompt_text = self.tokenizer.decode(valid_input_ids, skip_special_tokens=False)
            prompt_texts.append(prompt_text)
        
        # Generate responses via OpenAI API
        responses_text = []
        for idx, prompt_text in enumerate(prompt_texts):
            try:
                print(f"[OpenAI Agent] Calling API for sequence {idx+1}/{batch_size}...")
                print(f"[OpenAI Agent] Prompt length: {len(prompt_text)} chars")
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=self.temperature,
                )
                response_text = response.choices[0].message.content
                print(f"[OpenAI Agent] Received response (length: {len(response_text)} chars): {response_text[:100]}...")
                responses_text.append(response_text)
            except Exception as e:
                print(f"[OpenAI Agent] Error calling OpenAI API: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: return a simple response
                responses_text.append("Error: Could not generate response")
        
        # Encode responses back to token IDs
        responses_list = []
        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        prompts_list = []
        
        for i, response_text in enumerate(responses_text):
            # Encode response
            response_ids = self.tokenizer.encode(
                response_text,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)
            
            # Get original prompt
            prompt_ids = input_ids[i]
            prompt_mask = attention_mask[i].bool()
            prompt_ids_valid = prompt_ids[prompt_mask]
            
            # Concatenate prompt + response
            full_ids = torch.cat([prompt_ids_valid, response_ids])
            
            # Create attention mask (all 1s for full sequence)
            full_attention_mask = torch.ones(len(full_ids), dtype=torch.long)
            
            # Create position IDs (0-indexed sequential)
            full_position_ids = torch.arange(len(full_ids), dtype=torch.long)
            
            responses_list.append(response_ids)
            input_ids_list.append(full_ids)
            attention_mask_list.append(full_attention_mask)
            position_ids_list.append(full_position_ids)
            prompts_list.append(prompt_ids_valid)
        
        # Pad sequences to same length
        # Note: input_ids = prompt + response, so we pad everything to max_input_len
        max_input_len = max(len(seq) for seq in input_ids_list)
        max_response_len = max(len(resp) for resp in responses_list) if responses_list else 0
        
        padded_responses = []
        padded_input_ids = []
        padded_attention_mask = []
        padded_position_ids = []
        padded_prompts = []
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        for i in range(batch_size):
            # Pad responses (right pad) - pad all responses to the same max_response_len
            resp = responses_list[i]
            if len(resp) < max_response_len:
                resp_padded = torch.cat([resp, torch.full((max_response_len - len(resp),), pad_token_id, dtype=torch.long)])
            else:
                resp_padded = resp[:max_response_len]  # Truncate if too long
            padded_responses.append(resp_padded)
            
            # Pad input_ids (right pad) - this is prompt + response
            inp = input_ids_list[i]
            inp_padded = torch.cat([inp, torch.full((max_input_len - len(inp),), pad_token_id, dtype=torch.long)])
            padded_input_ids.append(inp_padded)
            
            # Pad attention mask (right pad with 0s)
            attn = attention_mask_list[i]
            attn_padded = torch.cat([attn, torch.zeros(max_input_len - len(attn), dtype=torch.long)])
            padded_attention_mask.append(attn_padded)
            
            # Pad position IDs (right pad with last position)
            pos = position_ids_list[i]
            last_pos = pos[-1] if len(pos) > 0 else 0
            pos_padded = torch.cat([pos, torch.full((max_input_len - len(pos),), last_pos, dtype=torch.long)])
            padded_position_ids.append(pos_padded)
            
            # Pad prompts (right pad) - prompts should match input_ids length for consistency
            prompt = prompts_list[i]
            prompt_padded = torch.cat([prompt, torch.full((max_input_len - len(prompt),), pad_token_id, dtype=torch.long)])
            padded_prompts.append(prompt_padded)
        
        # Stack into tensors
        responses_tensor = torch.stack(padded_responses).to(device)
        input_ids_tensor = torch.stack(padded_input_ids).to(device)
        attention_mask_tensor = torch.stack(padded_attention_mask).to(device)
        position_ids_tensor = torch.stack(padded_position_ids).to(device)
        prompts_tensor = torch.stack(padded_prompts).to(device)
        
        # Create dummy log_probs (zeros, since we don't have log probs from OpenAI)
        # Use max_input_len since that's the length of the full sequence
        log_probs = torch.zeros(batch_size, max_input_len, dtype=torch.float32).to(device)
        
        # Build DataProto
        batch = TensorDict(
            {
                "prompts": prompts_tensor,
                "responses": responses_tensor,
                "input_ids": input_ids_tensor,
                "rollout_log_probs": log_probs,
                "attention_mask": attention_mask_tensor,
                "position_ids": position_ids_tensor,
            },
            batch_size=batch_size,
        )
        
        return DataProto(batch=batch, meta_info=prompts.meta_info)

