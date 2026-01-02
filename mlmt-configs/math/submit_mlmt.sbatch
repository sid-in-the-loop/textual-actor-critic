#!/bin/bash
#SBATCH --job-name=mlmt_train
#SBATCH --output=/home/ssmurali/mlmt/logs/%j.out
#SBATCH --error=/home/ssmurali/mlmt/logs/%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=184G
#SBATCH --partition=general
#SBATCH --time=24:00:00

source /home/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate mlmt

export PYTHONUNBUFFERED=1
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_LOG_TO_STDERR=0
export WANDB_MODE=offline
export HF_TOKEN=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
export OPENAI_API_KEY=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

cd /home/ssmurali/mlmt

# Usage: sbatch mlmt-configs/math/submit_mlmt.sbatch <config_script_path>
CONFIG_SCRIPT=${1:-mlmt-configs/math/HLF_LLT_RR.sh}

echo "Launching training with config: $CONFIG_SCRIPT"
bash "$CONFIG_SCRIPT"

