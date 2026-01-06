import os
import torch
import torch.distributed.tensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import sys

def convert_fsdp_to_hf(fsdp_path, base_model_path, output_path, world_size=2):
    print(f"\n🔄 Converting: {fsdp_path}")
    print(f"➡️ To HF:      {output_path}")
    
    if os.path.exists(os.path.join(output_path, "config.json")):
        print(f"ℹ️ Output already exists at {output_path}. Skipping.")
        return True

    os.makedirs(output_path, exist_ok=True)
    state_dict = defaultdict(list)

    for rank in tqdm(range(world_size), desc="Loading shards"):
        filepath = os.path.join(fsdp_path, f"model_world_size_{world_size}_rank_{rank}.pt")
        if not os.path.exists(filepath):
            print(f"❌ Error: {filepath} not found.")
            return False
        
        # Load shard (DTensors require weights_only=False)
        shard = torch.load(filepath, map_location="cpu", weights_only=False)
        for key, value in shard.items():
            state_dict[key].append(value.to_local())

    print("Merging shards...")
    merged_state_dict = {}
    for key in state_dict:
        merged_state_dict[key] = torch.cat(state_dict[key], dim=0)

    print(f"Loading config/model from {base_model_path}...")
    config = AutoConfig.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_config(config)
    
    # Match architecture dtypes if necessary
    model.load_state_dict(merged_state_dict)
    
    print(f"Saving HF model...")
    model.save_pretrained(output_path, max_shard_size="5GB")
    
    print(f"Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Conversion complete for {os.path.basename(fsdp_path)}!")
    return True

if __name__ == "__main__":
    base_checkpoint = "/data/group_data/cx_group/ssmurali/mlmt_checkpoints/math/HLT_LLT_RR_1_15/HLT_LLT_RR_1_15_20260104_200425/global_step_189"
    base_model = "meta-llama/Llama-3.2-1B-Instruct"
    
    print(f"🚀 Starting conversion for Step 189 models...")
    
    # 1. Convert High Actor
    h_success = convert_fsdp_to_hf(
        fsdp_path=os.path.join(base_checkpoint, "high_actor"),
        base_model_path=base_model,
        output_path=os.path.join(base_checkpoint, "high_actor_hf")
    )
    
    # 2. Convert Low Actor
    l_success = convert_fsdp_to_hf(
        fsdp_path=os.path.join(base_checkpoint, "low_actor"),
        base_model_path=base_model,
        output_path=os.path.join(base_checkpoint, "low_actor_hf")
    )
    
    if h_success and l_success:
        print("\n✨ All models converted successfully!")
    else:
        print("\n⚠️ Some conversions failed.")
        sys.exit(1)

