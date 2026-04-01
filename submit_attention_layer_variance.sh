#!/bin/bash
#SBATCH --job-name="Attn_Layers"
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1-00:00:00
#SBATCH --output="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_attention/layer_variance_%j.out"
#SBATCH --error="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_attention/layer_variance_%j.err"

source ~/.bashrc
export HF_HOME=/blue/liguanpeng/bohanzhang1/hf_home
export HF_DATASETS_TRUST_REMOTE_CODE=1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness

cd /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/
LOG_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_attention"
mkdir -p $LOG_DIR

PYTHON_BIN="/blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness/bin/python"
SCRIPT_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation"

DATASETS=("piqa" "cmmlu")

# Target models that fit well into single GPU state extraction
MODELS=(
    "meta-llama/Llama-3.2-1B:bfloat16"
    "meta-llama/Llama-3.2-3B:bfloat16"
    "meta-llama/Meta-Llama-3.1-8B:bfloat16"
    "mistralai/Mistral-7B-v0.3:bfloat16"
)

JSONL_OUT="$SCRIPT_DIR/attention_layer_variance_results.jsonl"
rm -f $JSONL_OUT

echo "=== ATTENTION LAYER VARIANCE SWEEP ==="
for dataset in "${DATASETS[@]}"; do
    for model_info in "${MODELS[@]}"; do
        model="${model_info%:*}"
        dtype="${model_info#*:}"
        
        echo "=========================================="
        echo "Testing Attention Variance: $model on $dataset ($dtype)"
        echo "=========================================="
        
        $PYTHON_BIN $SCRIPT_DIR/analyze_attention_layer_variance.py \
            --model=$model \
            --dataset=$dataset \
            --dtype=$dtype \
            --out=$JSONL_OUT
    done
done

echo "=== COMPLETED ATTENTION LAYER SWEEP ==="
