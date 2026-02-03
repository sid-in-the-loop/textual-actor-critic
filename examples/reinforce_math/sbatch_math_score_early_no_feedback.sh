#!/bin/bash
#SBATCH --job-name=score_step_prompt
#SBATCH --output=logs/score_step_prompt_%j.out
#SBATCH --error=logs/score_step_prompt_%j.err
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=general
#SBATCH --time=24:00:00

# Load environment
source /home/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate mlmt

# Ensure unbuffered output for logging
export PYTHONUNBUFFERED=1
# Reduce Ray verbosity
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_LOG_TO_STDERR=0
# W&B configuration
export WANDB_MODE=online
export WANDB_START_METHOD=thread

# Set working directory
cd /home/ssmurali/mlmt

# Run ScoRe math training without feedback prompt
echo "Starting ScoRe training on MATH early dataset (no feedback prompt)..."
chmod +x examples/reinforce_math/run_math_score_early_no_feedback.sh
./examples/reinforce_math/run_math_score_early_no_feedback.sh

echo "Training completed."
