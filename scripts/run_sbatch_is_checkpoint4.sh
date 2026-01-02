#!/bin/bash
#SBATCH --job-name=is_checkpoint4_analysis
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

experiment_name=is_checkpoint4_analysis

VERL_OUTPUT_FILE="verl_logs/verl_${experiment_name}_${time}.out"

echo "=== Starting IS Checkpoint 4 analysis at ${time} ==="

# Set up environment variables
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=1e255990efc627595f0c805e0546cc7f0ff08b17
export HF_TOKEN=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
export HUGGING_FACE_HUB_TOKEN=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
export OPENAI_API_KEY=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
export CMU_GATEWAY_API_KEY=sk-dUplmEab2H7EFRaOISG1Ew

# Create logs directory if it doesn't exist
mkdir -p slurm_logs
mkdir -p verl_logs

echo "Running IS Checkpoint 4 analysis on trajectories"
stdbuf -oL -eL python analysis/is_checkpoint4.py \
  --log_dir /home/ssmurali/verl-agent/deepresearch_logs/val/20251215_212118 \
  --output_dir /home/ssmurali/verl-agent/deepresearch_outputs/val/20251215_212118 \
  > "$VERL_OUTPUT_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== IS Checkpoint 4 analysis completed successfully at $(date +%Y%m%d_%H%M%S) ==="

    # Run plot script on the generated belief trace
    echo "Running plot script on belief_trace_c4.jsonl..."
    stdbuf -oL -eL python plot2.py \
      --belief_path /home/ssmurali/verl-agent/deepresearch_outputs/val/20251215_212118/belief_trace_c4.jsonl \
      --output_dir /home/ssmurali/verl-agent/deepresearch_outputs/val/20251215_212118 \
      >> "$VERL_OUTPUT_FILE" 2>&1

    PLOT_EXIT_CODE=$?
    if [ $PLOT_EXIT_CODE -eq 0 ]; then
        echo "=== Plot generation completed successfully at $(date +%Y%m%d_%H%M%S) ==="
    else
        echo "=== Plot generation FAILED with exit code $PLOT_EXIT_CODE at $(date +%Y%m%d_%H%M%S) ==="
    fi
else
    echo "=== IS Checkpoint 4 analysis FAILED with exit code $EXIT_CODE at $(date +%Y%m%d_%H%M%S) ==="
    exit $EXIT_CODE
fi
