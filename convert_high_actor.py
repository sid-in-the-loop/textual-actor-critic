import os
import torch
import torch.distributed.tensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import sys

def convert_fsdp_to_hf(fsdp_path, base_model_path, output_path, world_size=2):
    print(f"Converting FSDP at {fsdp_path} to HF at {output_path}...")
    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists. Skipping.")
        return True

    os.makedirs(output_path, exist_ok=True)
    state_dict = defaultdict(list)

    for rank in tqdm(range(world_size), desc="Loading shards"):
        filepath = os.path.join(fsdp_path, f"model_world_size_{world_size}_rank_{rank}.pt")
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found.")
            return False
        
        # We need to use torch.load with a specific map_location if needed, 
        # but here we just need to load and Cat the sharded tensors.
        this_state_dict = torch.load(filepath, map_location="cpu", weights_only=False)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    print("Merging shards...")
    merged_state_dict = {}
    for key in state_dict:
        merged_state_dict[key] = torch.cat(state_dict[key], dim=0)

    print(f"Loading config from {base_model_path}...")
    config = AutoConfig.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_config(config)
    
    print("Loading state dict into model...")
    model.load_state_dict(merged_state_dict)
    
    print(f"Saving HF model to {output_path}...")
    model.save_pretrained(output_path, max_shard_size="10GB")
    
    print(f"Saving tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("✓ Conversion complete!")
    return True

if __name__ == "__main__":
    fsdp_path = "/data/group_data/cx_group/ssmurali/mlmt_checkpoints/math/HLT_LLT_RR_1_15/HLT_LLT_RR_1_15_20260103_195834/global_step_100/high_actor"
    base_model = "meta-llama/Llama-3.2-1B-Instruct"
    output_path = "/data/group_data/cx_group/ssmurali/mlmt_checkpoints/math/HLT_LLT_RR_1_15/HLT_LLT_RR_1_15_20260103_195834/global_step_100/high_actor_hf"
    
    success = convert_fsdp_to_hf(fsdp_path, base_model, output_path, world_size=2)
    sys.exit(0 if success else 1)


