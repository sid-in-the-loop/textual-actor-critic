#!/bin/bash
#SBATCH --job-name=is_checkpoint3_analysis
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=general
#SBATCH --time=4:00:00

set -x

GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

if [[ "$GPU_MODEL" == *"A6000"* || "$GPU_MODEL" == *"L40S"* ]]; then
    echo "Detected $GPU_MODEL, disabling NCCL P2P"
    export NCCL_P2P_DISABLE=1
else
    echo "Detected $GPU_MODEL, keeping NCCL P2P enabled"
fi

time=$(date +%Y%m%d_%H%M%S)

source ~/miniconda3/bin/activate verl-agent

experiment_name=is_checkpoint3_analysis

VERL_OUTPUT_FILE="verl_logs/verl_${experiment_name}_${time}.out"

echo "=== Starting IS Checkpoint 3 analysis at ${time} ==="

# Set up environment variables
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=1e255990efc627595f0c805e0546cc7f0ff08b17
export HF_TOKEN=${HF_TOKEN}
export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
export OPENAI_API_KEY=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
export CMU_GATEWAY_API_KEY=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

# Create logs directory if it doesn't exist
mkdir -p slurm_logs
mkdir -p verl_logs

echo "Running IS Checkpoint 3 analysis on 200 sampled trajectories"
stdbuf -oL -eL python analysis/is_checkpoint3.py \
  --log_dir /home/ssmurali/verl-agent/deepresearch_logs/val/20251215_212118 \
  --output_dir /home/ssmurali/verl-agent/deepresearch_outputs/val/20251215_212118 \
  > "$VERL_OUTPUT_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== IS Checkpoint 3 analysis completed successfully at $(date +%Y%m%d_%H%M%S) ==="
else
    echo "=== IS Checkpoint 3 analysis FAILED with exit code $EXIT_CODE at $(date +%Y%m%d_%H%M%S) ==="
    exit $EXIT_CODE
fi

