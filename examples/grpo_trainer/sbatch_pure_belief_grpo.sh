#!/bin/bash
#SBATCH --job-name=pure_belief_grpo
#SBATCH --output=/home/ssmurali/verl-agent/logs_new/pure_belief_grpo_%j.out
#SBATCH --error=/home/ssmurali/verl-agent/logs_new/pure_belief_grpo_%j.err
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=general
#SBATCH --time=24:00:00

# Load environment
source /home/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent

# Ensure unbuffered output for logging
export PYTHONUNBUFFERED=1
# Reduce Ray verbosity to show only warnings/errors
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_LOG_TO_STDERR=0
# W&B configuration
export WANDB_MODE=online
export WANDB_START_METHOD=thread
# Add W&B entity if needed (uncomment and set if required)
# export WANDB_ENTITY=your_username


# Set working directory
cd /home/ssmurali/verl-agent

# Run pure belief GRPO
echo "Starting Pure Belief GRPO training..."
echo "W&B API Key set: $(if [ -n "$WANDB_API_KEY" ]; then echo 'YES'; else echo 'NO'; fi)"
echo "W&B Mode: $WANDB_MODE"

# Test W&B connectivity
python3 -c "
import wandb
import os
try:
    # Test authentication
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    print('W&B authentication: SUCCESS')
except Exception as e:
    print(f'W&B authentication: FAILED - {e}')
" || echo "W&B test failed, continuing anyway"

./examples/grpo_trainer/run_deepresearch_pure_bgrpo.sh

echo "Pure Belief GRPO training completed."
