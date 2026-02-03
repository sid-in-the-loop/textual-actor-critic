#!/bin/bash
#SBATCH --job-name=math_reinforce
#SBATCH --output=logs/math_reinforce_%j.out
#SBATCH --error=logs/math_reinforce_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=general
#SBATCH --time=12:00:00

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

# Run REINFORCE math training
echo "Starting REINFORCE training on MATH mid dataset..."
chmod +x examples/reinforce_math/run_math_reinforce.sh
./examples/reinforce_math/run_math_reinforce.sh

echo "REINFORCE training completed."
