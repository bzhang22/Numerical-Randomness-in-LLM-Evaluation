#!/bin/bash
#SBATCH --job-name=llm_deep_tracing
#SBATCH --output=logs/trace_%j.out
#SBATCH --error=logs/trace_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus=1
#SBATCH --time=04:00:00

export HUGGINGFACE_HUB_CACHE="/blue/liguanpeng/bohanzhang1/hf_home"
export HF_HOME="/blue/liguanpeng/bohanzhang1/hf_home"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness

echo "Current Time: $(date)"
echo "Host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

echo "Initiating Deep Hidden-State Tracing on A100..."
python extract_traces.py --csv pairwise_compare.csv --results_dir results --output_dir trace --models Meta-Llama-3.1-8B Llama-3.2-3B

echo "Tracing Completely Finished at $(date) ✨"
