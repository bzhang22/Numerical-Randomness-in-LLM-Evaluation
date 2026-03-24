#!/bin/bash
#SBATCH --job-name="Exp_All"
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
#SBATCH --output="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_traces/all_mitigations_%j.out"
#SBATCH --error="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_traces/all_mitigations_%j.err"

LOG_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_traces"
mkdir -p $LOG_DIR

PYTHON_BIN="/blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness/bin/python"
SCRIPT_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation"

DATASETS=("piqa" "cmmlu" "gsm8k" "humaneval")
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
        model_name=$(basename $model)
        echo "=========================================="
        echo "Processing $model on $dataset..."
        echo "=========================================="
        
        # 1. Gen FP32 Reference
        out="results_mitigation/${model_name}_${dataset}_fp32_reference.jsonl"
        if [ ! -f "$out" ]; then
            $PYTHON_BIN $SCRIPT_DIR/run_mitigation.py --model=$model --dataset=$dataset --variant=fp32_reference --dtype=float32 --output=$out
        fi
        
        # 2. Gen BF16 Baseline & BF16 Variants
        for variant in "bf16_baseline" "attention" "norm" "lm_head" "attention_lm_head"; do
            out="results_mitigation/${model_name}_${dataset}_${variant}.jsonl"
            if [ ! -f "$out" ]; then
                $PYTHON_BIN $SCRIPT_DIR/run_mitigation.py --model=$model --dataset=$dataset --variant=$variant --dtype=bfloat16 --output=$out
            fi
        done
        
        # 3. Gen FP16 Baseline & FP16 Variants
        for var_target in "fp16_baseline" "fp16_attention" "fp16_norm" "fp16_lm_head" "fp16_attention_lm_head"; do
            out="results_mitigation/${model_name}_${dataset}_${var_target}.jsonl"
            
            # Map fp16 filename to internal script variant name
            base_var=${var_target#fp16_}
            if [[ "$base_var" == "baseline" ]]; then base_var="fp16_baseline"; fi
            
            if [ ! -f "$out" ]; then
                $PYTHON_BIN $SCRIPT_DIR/run_mitigation.py --model=$model --dataset=$dataset --variant=$base_var --dtype=float16 --output=$out
            fi
        done

        # 4. Extract Traces for BF16 and FP16
        echo "Extracting BF16 Traces..."
        if [ ! -f "traces_mitigation/${model_name}_${dataset}_bfloat16_traces.json" ]; then
            $PYTHON_BIN $SCRIPT_DIR/extract_mitigation_traces.py --model=$model --dataset=$dataset --base_dtype=bfloat16
        fi
        
        echo "Extracting FP16 Traces..."
        if [ ! -f "traces_mitigation/${model_name}_${dataset}_float16_traces.json" ]; then
            $PYTHON_BIN $SCRIPT_DIR/extract_mitigation_traces.py --model=$model --dataset=$dataset --base_dtype=float16
        fi
    done
done

# 5. Automatically Aggregate Data once EVERYTHING is processed
$PYTHON_BIN $SCRIPT_DIR/analyze_mitigation.py
$PYTHON_BIN $SCRIPT_DIR/plot_mitigation.py

echo "MEGA LOOP DONE!"
