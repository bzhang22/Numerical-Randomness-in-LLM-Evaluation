#!/bin/bash
source ~/.bashrc
export HF_HOME=/blue/liguanpeng/bohanzhang1/hf_home
export PIP_CACHE_DIR=/blue/liguanpeng/bohanzhang1/pip_cache
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness

cd /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/

LOG_FILE="/home/bohanzhang1/large_hooks_multi_datasets.log"

echo "=== STARTING MULTI-DATASET LAYER VARIANCE ANALYSIS ===" > $LOG_FILE

DATASETS=("piqa" "commonsense_qa" "cmmlu")

# Iterate over each dataset
for ds in "${DATASETS[@]}"; do
    echo "==========================================" | tee -a $LOG_FILE
    echo "=== DATASET: $ds" | tee -a $LOG_FILE
    echo "==========================================" | tee -a $LOG_FILE

    echo "=== STARTING LAYER VARIANCE ANALYSIS FOR 0.5B ($ds) ===" | tee -a $LOG_FILE
    python /home/bohanzhang1/analyze_flip_layers.py --model Qwen/Qwen2.5-0.5B --dataset $ds --limit 1000 --dtype float16 >> $LOG_FILE 2>&1

    echo "=== STARTING LAYER VARIANCE ANALYSIS FOR 3B ($ds) ===" | tee -a $LOG_FILE
    python /home/bohanzhang1/analyze_flip_layers.py --model Qwen/Qwen2.5-3B --dataset $ds --limit 1000 --dtype float16 >> $LOG_FILE 2>&1

    echo "=== STARTING LAYER VARIANCE ANALYSIS FOR 7B ($ds) ===" | tee -a $LOG_FILE
    python /home/bohanzhang1/analyze_flip_layers.py --model Qwen/Qwen2.5-7B --dataset $ds --limit 1000 --dtype bfloat16 >> $LOG_FILE 2>&1

    echo "=== STARTING LAYER VARIANCE ANALYSIS FOR 32B ($ds) ===" | tee -a $LOG_FILE
    python /home/bohanzhang1/analyze_flip_layers.py --model Qwen/Qwen2.5-32B --dataset $ds --limit 1000 --dtype bfloat16 >> $LOG_FILE 2>&1
done

echo "=== ALL MULTI-DATASET ANALYSIS COMPLETE ===" | tee -a $LOG_FILE
