#!/bin/bash

# Tier-1 Models
MODELS=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "mistralai/Mistral-7B-v0.3"
    "meta-llama/Meta-Llama-3.1-8B"
)

# Logs directory
LOG_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs"
mkdir -p $LOG_DIR

# Ensure main SLURM script is executable
chmod +x /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/run_evaluation_slurm.sh

for model in "${MODELS[@]}"; do
    model_name=$(basename $model)
    
    # 1. Precision Sweep
    sbatch \
        --job-name="P_${model_name}" \
        --partition=hpg-b200 \
        --gpus=1 \
        --cpus-per-task=4 \
        --mem=32g \
        --time=12:00:00 \
        --output="${LOG_DIR}/P_${model_name}_%j.out" \
        --error="${LOG_DIR}/P_${model_name}_%j.err" \
        --export=MODEL="$model",MODE="precision" \
        /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/run_evaluation_slurm.sh
           
    # 2. Batch Sweep
    sbatch \
        --job-name="B_${model_name}" \
        --partition=hpg-b200 \
        --gpus=1 \
        --cpus-per-task=4 \
        --mem=32g \
        --time=12:00:00 \
        --output="${LOG_DIR}/B_${model_name}_%j.out" \
        --error="${LOG_DIR}/B_${model_name}_%j.err" \
        --export=MODEL="$model",MODE="batch" \
        /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/run_evaluation_slurm.sh
        
    echo "Submitted Precision and Batch jobs for $model_name"
done

echo "All Tier-1 Model jobs submitted successfully!"
