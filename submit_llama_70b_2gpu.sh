#!/bin/bash

# Script to submit Llama-3.1-70B on 2 B200 GPUs as requested by user

model="meta-llama/Meta-Llama-3.1-70B"
gpus=2
cpus=8
mem=64g

LOG_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_tier2"
mkdir -p $LOG_DIR

# Ensure main SLURM script is executable
chmod +x /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/run_evaluation_slurm.sh

model_name=$(basename $model)

# 1. Precision Sweep
sbatch \
    --job-name="P2_${model_name}_2gpu" \
    --partition=hpg-b200 \
    --gpus=$gpus \
    --cpus-per-task=$cpus \
    --mem=$mem \
    --time=48:00:00 \
    --output="${LOG_DIR}/P2_${model_name}_2gpu_%j.out" \
    --error="${LOG_DIR}/P2_${model_name}_2gpu_%j.err" \
    --export=MODEL="$model",MODE="precision" \
    /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/run_evaluation_slurm.sh
       
# 2. Batch Sweep
sbatch \
    --job-name="B2_${model_name}_2gpu" \
    --partition=hpg-b200 \
    --gpus=$gpus \
    --cpus-per-task=$cpus \
    --mem=$mem \
    --time=48:00:00 \
    --output="${LOG_DIR}/B2_${model_name}_2gpu_%j.out" \
    --error="${LOG_DIR}/B2_${model_name}_2gpu_%j.err" \
    --export=MODEL="$model",MODE="batch" \
    /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/run_evaluation_slurm.sh
    
echo "Submitted Precision and Batch jobs for $model on 2 B200 GPUs."
