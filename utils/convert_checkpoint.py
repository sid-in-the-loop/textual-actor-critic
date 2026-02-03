import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed.tensor  
from glob import glob
from collections import defaultdict
from tqdm import tqdm
import os

def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoints to Hugging Face format")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the FSDP checkpoint directory (containing model_*.pt)")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Base model for config and tokenizer")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for HF checkpoint")
    parser.add_argument("--world_size", type=int, default=2, help="World size used during training")
    args = parser.parse_args()

    state_dict = defaultdict(list)

    # Check for shards
    shard_pattern = os.path.join(args.input_path, f"model_world_size_{args.world_size}_rank_*.pt")
    shards = glob(shard_pattern)
    if not shards:
        print(f"❌ Error: No shards found matching {shard_pattern}")
        return

    for rank in tqdm(range(args.world_size), desc="Loading sharded checkpoints"):
        filepath = os.path.join(args.input_path, f"model_world_size_{args.world_size}_rank_{rank}.pt")
        if not os.path.exists(filepath):
            print(f"❌ Error: {filepath} not found.")
            continue
        
        print(f"Loading {filepath}...")
        this_state_dict = torch.load(filepath, weights_only=False, map_location="cpu")
        for key, value in this_state_dict.items():
            # Handle DTensor to local tensor conversion
            if hasattr(value, "to_local"):
                state_dict[key].append(value.to_local())
            else:
                state_dict[key].append(value)

    print("Concatenating shards...")
    for key in state_dict:
        if len(state_dict[key]) > 1:
            # For 1D tensors (biases, layer norms), they might be replicated across ranks in some FSDP setups
            # but usually they are concatenated for weights.
            # We assume concatenating along dim 0 is correct for sharded parameters.
            try:
                # Filter out potential duplicates if they are already full size (replicated)
                if state_dict[key][0].shape == state_dict[key][1].shape:
                     # Check if they are identical (replicated)
                     if torch.equal(state_dict[key][0], state_dict[key][1]):
                         state_dict[key] = state_dict[key][0]
                         continue
                
                state_dict[key] = torch.cat(state_dict[key], dim=0)
            except Exception as e:
                print(f"Warning: Failed to concatenate {key}: {e}. Taking first shard.")
                state_dict[key] = state_dict[key][0]
        else:
            state_dict[key] = state_dict[key][0]

    print(f"Loading config from {args.base_model}...")
    config = AutoConfig.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_config(config)
    
    print("Loading state dict into model...")
    # Map keys if needed (verl sometimes adds prefixes, though usually it doesn't in FSDP save)
    # If keys don't match, we might need a rename loop
    model_state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            new_state_dict[k] = v
        elif f"model.{k}" in model_state_dict:
            new_state_dict[f"model.{k}"] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)

    print(f"Saving HF model to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path, max_shard_size="10GB")

    print(f"Saving tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_path)
    print("✅ Conversion complete.")

if __name__ == "__main__":
    main()
