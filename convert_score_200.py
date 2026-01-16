from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import os
from collections import defaultdict
from tqdm import tqdm

# Configuration for your specific run
fsdp_checkpoint_path = "/home/ssmurali/mlmt/checkpoints/math/ScoRe/ScoRe_20260108_114421/global_step_200/actor"
base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
output_path = "/home/ssmurali/mlmt/checkpoints/math/ScoRe/ScoRe_20260108_114421/global_step_200/actor_hf"

def main():
    os.makedirs(output_path, exist_ok=True)
    state_dict = defaultdict(list)

    # Based on your ls output, world_size is 4
    world_size = 4 
    
    print(f"Merging {world_size} FSDP shards from {fsdp_checkpoint_path}...")
    
    # We need to load sharded tensors and merge them
    # FSDP shards usually have DTensor metadata that requires special handling
    for rank in range(world_size):
        filepath = os.path.join(fsdp_checkpoint_path, f"model_world_size_{world_size}_rank_{rank}.pt")
        print(f"Loading rank {rank} from {filepath}")
        
        # Load shard
        shard = torch.load(filepath, map_location="cpu", weights_only=False)
        
        for key, value in shard.items():
            # If it's a DistributedTensor, we need to gather it
            if hasattr(value, "to_local"):
                state_dict[key].append(value.to_local())
            else:
                state_dict[key] = value # Not sharded

    print("Concatenating shards...")
    final_state_dict = {}
    for key, pieces in state_dict.items():
        if isinstance(pieces, list):
            # Concatenate along the first dimension (FSDP default)
            final_state_dict[key] = torch.cat(pieces, dim=0)
        else:
            final_state_dict[key] = pieces

    print(f"Loading config from {base_model_path}...")
    config = AutoConfig.from_pretrained(base_model_path)
    
    # Handle possible vocab size mismatch if extra tokens were added
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    
    print("Loading state dict into model...")
    # We might need strict=False if some buffers aren't in the state_dict
    missing, unexpected = model.load_state_dict(final_state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    print(f"Saving HF model to {output_path}...")
    model.save_pretrained(output_path)
    
    print(f"Saving tokenizer to {output_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("✅ Conversion complete!")

if __name__ == "__main__":
    main()


