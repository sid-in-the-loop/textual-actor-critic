import os
import torch
import torch.distributed.tensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import sys

def convert_high_actor():
    # Absolute path to the high_actor directory
    fsdp_path = "/data/group_data/cx_group/ssmurali/mlmt_checkpoints/math/HLT_LLT_RR_1_15/HLT_LLT_RR_1_15_20260103_195834/global_step_100/high_actor"
    base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    output_path = "/data/group_data/cx_group/ssmurali/mlmt_checkpoints/math/HLT_LLT_RR_1_15/HLT_LLT_RR_1_15_20260103_195834/global_step_100/high_actor_hf"
    world_size = 2

    print(f"🔄 Converting High Actor shards from {fsdp_path}")
    os.makedirs(output_path, exist_ok=True)

    state_dict = defaultdict(list)
    for rank in tqdm(range(world_size), desc="Loading shards"):
        filepath = os.path.join(fsdp_path, f"model_world_size_{world_size}_rank_{rank}.pt")
        # DTensors require weights_only=False
        shard = torch.load(filepath, map_location="cpu", weights_only=False)
        for key, value in shard.items():
            state_dict[key].append(value.to_local())

    print("Merging shards...")
    merged_state_dict = {}
    for key, tensors in state_dict.items():
        merged_state_dict[key] = torch.cat(tensors, dim=0)

    print("Saving HuggingFace model...")
    config = AutoConfig.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(merged_state_dict)
    model.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(output_path)
    print(f"✨ High Actor converted to HF at {output_path}")

if __name__ == "__main__":
    convert_high_actor()
