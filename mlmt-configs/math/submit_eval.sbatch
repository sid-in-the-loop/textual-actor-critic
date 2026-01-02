#!/bin/bash
#SBATCH --job-name=mlmt_eval
#SBATCH --output=/home/ssmurali/mlmt/logs/eval_%j.out
#SBATCH --error=/home/ssmurali/mlmt/logs/eval_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=92G
#SBATCH --partition=general
#SBATCH --time=04:00:00

source /home/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate mlmt

export PYTHONUNBUFFERED=1
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_LOG_TO_STDERR=0
export WANDB_MODE=offline
export HF_TOKEN=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

cd /home/ssmurali/mlmt

# Usage: sbatch mlmt-configs/math/submit_eval.sbatch
CONFIG_SCRIPT="mlmt-configs/math/eval_math500.sh"

echo "Launching evaluation with config: $CONFIG_SCRIPT"
bash "$CONFIG_SCRIPT"

