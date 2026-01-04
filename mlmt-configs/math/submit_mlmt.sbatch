#!/bin/bash
#SBATCH --job-name=mlmt_train
#SBATCH --output=/home/ssmurali/mlmt/logs/%j.out
#SBATCH --error=/home/ssmurali/mlmt/logs/%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=184G
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --exclude=babel-s5-24,babel-s9-24,babel-w5-32
source /home/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate mlmt

export PYTHONUNBUFFERED=1
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_LOG_TO_STDERR=0
export WANDB_MODE=offline
export HF_TOKEN=${HF_TOKEN}
export OPENAI_API_KEY=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

cd /home/ssmurali/mlmt

# Usage: sbatch mlmt-configs/math/submit_mlmt.sbatch <config_script_path>
CONFIG_SCRIPT=${1:-mlmt-configs/math/HLF_LLT_RR.sh}

echo "Launching training with config: $CONFIG_SCRIPT"
bash "$CONFIG_SCRIPT"

