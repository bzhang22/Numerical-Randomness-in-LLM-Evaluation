#!/bin/bash

# Tier-1 Models + Gemma 2B
MODELS=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "mistralai/Mistral-7B-v0.3"
    "meta-llama/Meta-Llama-3.1-8B"
    "google/gemma-2-2b"
)

# Logs directory for attention experiments
LOG_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_attention"
mkdir -p $LOG_DIR

for model in "${MODELS[@]}"; do
    model_name=$(basename $model)
    
    # Submit Attention Sweep Job
    sbatch \
        --job-name="A_${model_name}" \
        --partition=hpg-b200 \
        --gpus=1 \
        --cpus-per-task=4 \
        --mem=32g \
        --time=12:00:00 \
        --output="${LOG_DIR}/A_${model_name}_%j.out" \
        --error="${LOG_DIR}/A_${model_name}_%j.err" \
        --export=MODEL="$model" \
        /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/run_attention_slurm.sh
           
    echo "Submitted Attention Sweep jobs for $model_name"
done

echo "All Attention jobs submitted successfully!"
