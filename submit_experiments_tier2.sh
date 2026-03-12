#!/bin/bash

# Tier-2 Models
MODELS_1GPU=(
    # "google/gemma-2-2b"
)

MODELS_2GPU=(
    "google/gemma-2-27b"
    "01-ai/Yi-34B"
)

MODELS_4GPU=(
    "meta-llama/Meta-Llama-3.1-70B"
)

# Logs directory
LOG_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_tier2"
mkdir -p $LOG_DIR

# Ensure main SLURM script is executable
chmod +x /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/run_evaluation_slurm.sh

# Function to submit jobs given model, gpus, cpus, mem
submit_jobs() {
    local model=$1
    local gpus=$2
    local cpus=$3
    local mem=$4
    
    model_name=$(basename $model)
    
    # 1. Precision Sweep
    sbatch \
        --job-name="P2_${model_name}" \
        --partition=hpg-b200 \
        --gpus=$gpus \
        --cpus-per-task=$cpus \
        --mem=$mem \
        --time=48:00:00 \
        --output="${LOG_DIR}/P2_${model_name}_%j.out" \
        --error="${LOG_DIR}/P2_${model_name}_%j.err" \
        --export=MODEL="$model",MODE="precision" \
        /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/run_evaluation_slurm.sh
           
    # 2. Batch Sweep
    sbatch \
        --job-name="B2_${model_name}" \
        --partition=hpg-b200 \
        --gpus=$gpus \
        --cpus-per-task=$cpus \
        --mem=$mem \
        --time=48:00:00 \
        --output="${LOG_DIR}/B2_${model_name}_%j.out" \
        --error="${LOG_DIR}/B2_${model_name}_%j.err" \
        --export=MODEL="$model",MODE="batch" \
        /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/run_evaluation_slurm.sh
        
    echo "Submitted Precision and Batch jobs for $model_name | GPUs: $gpus, CPUs: $cpus, Mem: $mem"
}

for model in "${MODELS_1GPU[@]}"; do
    submit_jobs $model 1 4 32g
done

for model in "${MODELS_2GPU[@]}"; do
    submit_jobs $model 2 8 128g
done

for model in "${MODELS_4GPU[@]}"; do
    submit_jobs $model 4 4 256g
done

echo "All Tier-2 Model jobs submitted successfully!"
