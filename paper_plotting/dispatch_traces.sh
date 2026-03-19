#!/bin/bash

# Target Models mapping to the same tier configurations as before, avoiding 70B for now due to QOSGrpMemLimit
MODELS_1GPU=(
    "Llama-3.2-1B"
    "Llama-3.2-3B"
    "Mistral-7B-v0.3"
    "Meta-Llama-3.1-8B"
    "gemma-2-2b"
)

MODELS_2GPU=(
    "gemma-2-27b"
    "Yi-34B"
)

# Ensure tracing script is executable
chmod +x /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/submit_tracing.sh

mkdir -p trace
mkdir -p logs

submit_trace() {
    local model=$1
    local gpus=$2
    local cpus=$3
    local mem=$4
    
    sbatch \
        --job-name="Trace_${model}" \
        --partition=hpg-b200 \
        --gpus=$gpus \
        --cpus-per-task=$cpus \
        --mem=${mem} \
        --export=MODEL="$model" \
        /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/submit_tracing.sh
        
    echo "Submitted Tracing Job for $model | GPUs: $gpus, Mem: $mem"
}

for model in "${MODELS_1GPU[@]}"; do
    submit_trace "$model" 1 4 8g
done

for model in "${MODELS_2GPU[@]}"; do
    submit_trace "$model" 2 8 8g
done

echo "Deep Tracing Arrays completely dispatched."
