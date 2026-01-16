import torch
import os

fsdp_checkpoint_path = "/home/ssmurali/mlmt/checkpoints/math/ScoRe/ScoRe_20260108_114421/global_step_500/actor"
filepath = os.path.join(fsdp_checkpoint_path, "model_world_size_4_rank_0.pt")

if not os.path.exists(filepath):
    print(f"File {filepath} not found.")
else:
    print(f"Loading {filepath}...")
    shard = torch.load(filepath, map_location="cpu", weights_only=False)
    print("\nFirst 20 keys in shard:")
    for i, key in enumerate(list(shard.keys())[:20]):
        print(f"{i}: {key}")
    
    # Check for prefix
    all_keys = list(shard.keys())
    prefixes = set()
    for key in all_keys:
        if "." in key:
            prefixes.add(key.split(".")[0])
    print(f"\nTop-level prefixes found: {prefixes}")


