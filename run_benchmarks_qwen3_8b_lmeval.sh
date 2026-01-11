#!/bin/bash
# set -e

# Define variables
# Use absolute path to python in env
CONDA_CMD="/home/bohanzhang1/miniconda3/envs/llm_randomness/bin/python run_benchmark_lmeval.py --batch_size 1"
MODEL="Qwen/Qwen3-8B"
SEED=123
LIMIT_PPL=1000
LIMIT_QA=1000
CMMLU_SUBSET="agronomy"

# Output Directories
BASE_DIR="qwen3-8b_lmeval"
PPL_DIR="$BASE_DIR/ppl"
QA_DIR="$BASE_DIR/commonsense_qa"
CMMLU_DIR="$BASE_DIR/cmmlu"
GSM8K_DIR="$BASE_DIR/gsm8k_ppl"
PIQA_DIR="$BASE_DIR/piqa"

mkdir -p "$PPL_DIR"
mkdir -p "$QA_DIR"
mkdir -p "$CMMLU_DIR"
mkdir -p "$GSM8K_DIR"
mkdir -p "$PIQA_DIR"

echo "======================================================="
echo "Running Benchmarks for $MODEL (Seed: $SEED)"
echo "Output Directories:"
echo "  PPL:   $PPL_DIR"
echo "  QA:    $QA_DIR"
echo "  CMMLU: $CMMLU_DIR"
echo "  GSM8K: $GSM8K_DIR"
echo "  PIQA:  $PIQA_DIR"
echo "======================================================="

# Function to run variants
run_variants() {
    TASK_TYPE=$1
    DATASET=$2
    DIR=$3
    LIMIT=$4
    EXTRA_ARGS=$5

    echo "--- $DATASET ($TASK_TYPE) ---"

    echo "[1/4] HF | FP16 | TF32: Off"
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 $EXTRA_ARGS --output "$DIR/hf_fp16_notf32.json"

    echo "[2/4] HF | FP16 | TF32: On"
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --tf32 $EXTRA_ARGS --output "$DIR/hf_fp16_tf32.json"

    echo "[3/4] HF | FP32 | TF32: Off"
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float32 $EXTRA_ARGS --output "$DIR/hf_fp32_notf32.json"

    echo "[4/4] HF | FP32 | TF32: On"
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float32 --tf32 $EXTRA_ARGS --output "$DIR/hf_fp32_tf32.json"
}

# =======================================================
# PERPLEXITY BENCHMARKS
# =======================================================
run_variants "perplexity" "wikitext" "$PPL_DIR" $LIMIT_PPL ""

# =======================================================
# COMMONSENSE QA BENCHMARKS
# =======================================================
run_variants "qa" "commonsense_qa" "$QA_DIR" $LIMIT_QA ""

# =======================================================
# CMMLU BENCHMARKS
# =======================================================
echo "--- CMMLU ($CMMLU_SUBSET) ---"
echo "[1/4] HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --output "$CMMLU_DIR/hf_fp16_notf32.json"

echo "[2/4] HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --tf32 --output "$CMMLU_DIR/hf_fp16_tf32.json"

echo "[3/4] HF | FP32 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --output "$CMMLU_DIR/hf_fp32_notf32.json"

echo "[4/4] HF | FP32 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --tf32 --output "$CMMLU_DIR/hf_fp32_tf32.json"


# =======================================================
# GSM8K BENCHMARKS (PPL)
# =======================================================
run_variants "perplexity" "gsm8k" "$GSM8K_DIR" $LIMIT_PPL ""

# =======================================================
# PIQA BENCHMARKS (QA)
# =======================================================
run_variants "qa" "piqa" "$PIQA_DIR" $LIMIT_QA ""

echo "======================================================="
echo "Benchmarks completed successfully!"
echo "======================================================="
