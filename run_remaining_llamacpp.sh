#!/bin/bash
# set -e (Disabled to allow continuation after OOM)

# Define variables
CONDA_CMD="./venv/bin/python run_benchmark.py"
MODEL="Qwen/Qwen3-0.6B"
GGUF_MODEL_PATH="./models/Qwen3-0.6B.fp16.gguf"
SEED=123
LIMIT_PPL=20
LIMIT_QA=100
GPU_UTIL=0.6
CMMLU_SUBSET="agronomy"

# Output Directories
BASE_DIR="qwen3-0.6b"
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
echo "Running Remaining LlamaCpp Benchmarks for $MODEL"
echo "======================================================="

# --- LlamaCpp Backend (PPL) ---
echo "[9/60] PPL | LlamaCpp | FP16 | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --output "$PPL_DIR/llamacpp_fp16_notf32.jsonl"

echo "[10/60] PPL | LlamaCpp | FP16 | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --output "$PPL_DIR/llamacpp_fp16_tf32.jsonl"

echo "[11/60] PPL | LlamaCpp | FP16 (Simulated FP32) | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --output "$PPL_DIR/llamacpp_fp16_sim_fp32_notf32.jsonl"

echo "[12/60] PPL | LlamaCpp | FP16 (Simulated FP32) | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --output "$PPL_DIR/llamacpp_fp16_sim_fp32_tf32.jsonl"
unset NVIDIA_TF32_OVERRIDE

# --- LlamaCpp Backend (QA) ---
echo "[21/60] QA | LlamaCpp | FP16 | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --output "$QA_DIR/llamacpp_fp16_notf32.jsonl"

echo "[22/60] QA | LlamaCpp | FP16 | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --output "$QA_DIR/llamacpp_fp16_tf32.jsonl"

echo "[23/60] QA | LlamaCpp | FP16 (Simulated FP32) | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --output "$QA_DIR/llamacpp_fp16_sim_fp32_notf32.jsonl"

echo "[24/60] QA | LlamaCpp | FP16 (Simulated FP32) | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --output "$QA_DIR/llamacpp_fp16_sim_fp32_tf32.jsonl"
unset NVIDIA_TF32_OVERRIDE

# --- LlamaCpp Backend (CMMLU) ---
echo "[33/60] CMMLU | LlamaCpp | FP16 | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --output "$CMMLU_DIR/llamacpp_fp16_notf32.jsonl"

echo "[34/60] CMMLU | LlamaCpp | FP16 | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --output "$CMMLU_DIR/llamacpp_fp16_tf32.jsonl"

echo "[35/60] CMMLU | LlamaCpp | FP16 (Simulated FP32) | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --output "$CMMLU_DIR/llamacpp_fp16_sim_fp32_notf32.jsonl"

echo "[36/60] CMMLU | LlamaCpp | FP16 (Simulated FP32) | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --output "$CMMLU_DIR/llamacpp_fp16_sim_fp32_tf32.jsonl"
unset NVIDIA_TF32_OVERRIDE

# --- LlamaCpp Backend (GSM8K) ---
echo "[45/60] GSM8K | LlamaCpp | FP16 | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --output "$GSM8K_DIR/llamacpp_fp16_notf32.jsonl"

echo "[46/60] GSM8K | LlamaCpp | FP16 | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --output "$GSM8K_DIR/llamacpp_fp16_tf32.jsonl"

echo "[47/60] GSM8K | LlamaCpp | FP16 (Simulated FP32) | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --output "$GSM8K_DIR/llamacpp_fp16_sim_fp32_notf32.jsonl"

echo "[48/60] GSM8K | LlamaCpp | FP16 (Simulated FP32) | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --output "$GSM8K_DIR/llamacpp_fp16_sim_fp32_tf32.jsonl"
unset NVIDIA_TF32_OVERRIDE

# --- LlamaCpp Backend (PIQA) ---
echo "[57/60] PIQA | LlamaCpp | FP16 | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --output "$PIQA_DIR/llamacpp_fp16_notf32.jsonl"

echo "[58/60] PIQA | LlamaCpp | FP16 | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --output "$PIQA_DIR/llamacpp_fp16_tf32.jsonl"

echo "[59/60] PIQA | LlamaCpp | FP16 (Simulated FP32) | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --output "$PIQA_DIR/llamacpp_fp16_sim_fp32_notf32.jsonl"

echo "[60/60] PIQA | LlamaCpp | FP16 (Simulated FP32) | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --output "$PIQA_DIR/llamacpp_fp16_sim_fp32_tf32.jsonl"
unset NVIDIA_TF32_OVERRIDE

echo "======================================================="
echo "Remaining benchmarks completed!"
echo "======================================================="
