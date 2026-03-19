#!/bin/bash
#SBATCH --job-name="FlashAttn_Build"
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_attention/install_flash_attn.out"

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness

echo "Loading CUDA module..."
module load cuda/12.8.1

echo "Initiating Flash Attention PyTorch Wheel Build..."
# Limit parallel jobs to prevent node-level OOM while GCC builds the matrices
MAX_JOBS=4 pip install flash-attn --no-build-isolation

echo "Build Complete!"
