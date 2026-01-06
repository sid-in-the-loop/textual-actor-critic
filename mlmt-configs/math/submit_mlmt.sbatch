#!/bin/bash
#SBATCH --job-name=mlmt_train
#SBATCH --output=/home/ssmurali/mlmt/logs/%j.out
#SBATCH --error=/home/ssmurali/mlmt/logs/%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=184G
#SBATCH --partition=preempt
#SBATCH --time=4-00:00:00
#SBATCH --exclude=babel-s5-24,babel-s9-24,babel-w5-32,babel-x5-24,babel-y9-12
source /home/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate mlmt

export PYTHONUNBUFFERED=1
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_LOG_TO_STDERR=0
export WANDB_MODE=online
export HF_TOKEN=${HF_TOKEN}
export OPENAI_API_KEY=${OPENAI_API_KEY}

cd /home/ssmurali/mlmt

# Usage: sbatch mlmt-configs/math/submit_mlmt.sbatch <config_script_path>
CONFIG_SCRIPT=${1:-mlmt-configs/math/HLF_LLT_RR.sh}

echo "Launching training with config: $CONFIG_SCRIPT"
bash "$CONFIG_SCRIPT"

