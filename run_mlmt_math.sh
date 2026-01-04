#!/bin/bash
export PYTHONUNBUFFERED=1

# GPU Model Detection (from run_deepresearcher_1.7b.sh)
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_MODEL" == *"A6000"* || "$GPU_MODEL" == *"L40S"* ]]; then
    echo "Detected $GPU_MODEL, disabling NCCL P2P"
    export NCCL_P2P_DISABLE=1
else
    echo "Detected $GPU_MODEL, keeping NCCL P2P enabled"
fi

# Model and Experiment Paths
MODEL_PATH=$1
if [ -z "$MODEL_PATH" ]; then
    echo "Usage: ./run_mlmt_math.sh <model_path_or_name>"
    exit 1
fi

EXPERIMENT_NAME="mlmt_math_$(date +%Y%m%d_%H%M%S)"

# Activate Environment
source ~/miniconda3/bin/activate mlmt

# Launch Training
python verl/trainer/main_ppo.py \
    --config-name mlmt_math \
    actor_rollout_ref.model.path=$MODEL_PATH \
    critic.model.path=$MODEL_PATH \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.project_name=mlmt_rl_math \
    trainer.n_gpus_per_node=$(nvidia-smi -L | wc -l) \
    2>&1 | tee "logs/${EXPERIMENT_NAME}.log"





