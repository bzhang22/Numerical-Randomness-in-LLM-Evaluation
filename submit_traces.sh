#!/bin/bash
# Slurm Precision Traces Dispatcher

LOG_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_traces"
mkdir -p $LOG_DIR

DATASETS=("piqa" "gsm8k" "cmmlu" "humaneval")

MODELS=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Meta-Llama-3.1-8B"
    "mistralai/Mistral-7B-v0.3"
    "google/gemma-2-27b"
    "01-ai/Yi-34B"
    "meta-llama/Meta-Llama-3.1-70B"
)

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        gpus=1
        mem="32g"
        
        if [[ "$model" == *"70B"* ]]; then
            gpus=4
            mem="256g"
        elif [[ "$model" == *"27b"* || "$model" == *"34B"* ]]; then
            gpus=2
            mem="128g"
        fi
        
        model_name=$(basename $model)
        
        sbatch \
            --job-name="T_${model_name}_${dataset}" \
            --partition=hpg-b200 \
            --gpus=$gpus \
            --cpus-per-task=4 \
            --mem=$mem \
            --time=04:00:00 \
            --output="${LOG_DIR}/${model_name}_${dataset}_%j.out" \
            --error="${LOG_DIR}/${model_name}_${dataset}_%j.err" \
            --wrap="/blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness/bin/python /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/extract_mitigation_traces.py \
                --model=$model \
                --dataset=$dataset"
                
    done
done

echo "Submitted all 28 Layer Trace Extraction jobs."
