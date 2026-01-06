import torch
try:
    print('Loading low_actor shard...')
    d = torch.load('/data/group_data/cx_group/ssmurali/mlmt_checkpoints/math/HLT_LLT_RR_1_15/HLT_LLT_RR_1_15_20260103_195834/global_step_100/low_actor/model_world_size_2_rank_0.pt', map_location='cpu', weights_only=False)
    print('Success!')
    print(f'Keys: {list(d.keys())[:5]}')
except Exception as e:
    print(f'Failed to load low_actor shard: {e}')


