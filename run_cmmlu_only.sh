#!/bin/bash
set -e

# Define variables
CONDA_CMD="$HOME/miniconda/bin/conda run -n llm_randomness python run_benchmark.py"
MODEL="Qwen/Qwen3-0.6B"
GGUF_MODEL_PATH="/home/UFAD/bohanzhang1/.cache/huggingface/hub/models--MaziyarPanahi--Qwen3-0.6B-GGUF/snapshots/16d75108d73a476af91a4f6df4cd77e854b42d04/Qwen3-0.6B.fp16.gguf"
SEED=123
LIMIT_QA=100
GPU_UTIL=0.6
CMMLU_SUBSET="agronomy"

# Output Directories
BASE_DIR="qwen3-0.6b"
CMMLU_DIR="$BASE_DIR/cmmlu"

mkdir -p "$CMMLU_DIR"

echo "======================================================="
echo "Running CMMLU Benchmarks for $MODEL (Seed: $SEED)"
echo "Output Directory: $CMMLU_DIR"
echo "======================================================="

# =======================================================
# CMMLU BENCHMARKS (12 Runs)
# =======================================================

echo "--- CMMLU Benchmarks ($CMMLU_SUBSET) ---"

# --- HuggingFace Backend ---
echo "[1/12] CMMLU | HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --output "$CMMLU_DIR/hf_fp16_notf32.jsonl"

echo "[2/12] CMMLU | HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --tf32 --output "$CMMLU_DIR/hf_fp16_tf32.jsonl"

echo "[3/12] CMMLU | HF | FP32 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --output "$CMMLU_DIR/hf_fp32_notf32.jsonl"

echo "[4/12] CMMLU | HF | FP32 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --tf32 --output "$CMMLU_DIR/hf_fp32_tf32.jsonl"

# --- vLLM Backend ---
echo "[5/12] CMMLU | vLLM | FP16 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --output "$CMMLU_DIR/vllm_fp16_notf32.jsonl"

echo "[6/12] CMMLU | vLLM | FP16 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --tf32 --output "$CMMLU_DIR/vllm_fp16_tf32.jsonl"

echo "[7/12] CMMLU | vLLM | FP32 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --output "$CMMLU_DIR/vllm_fp32_notf32.jsonl"

echo "[8/12] CMMLU | vLLM | FP32 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --tf32 --output "$CMMLU_DIR/vllm_fp32_tf32.jsonl"

# --- LlamaCpp Backend ---
echo "[9/12] CMMLU | LlamaCpp | FP16 | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --output "$CMMLU_DIR/llamacpp_fp16_notf32.jsonl"

echo "[10/12] CMMLU | LlamaCpp | FP16 | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --output "$CMMLU_DIR/llamacpp_fp16_tf32.jsonl"

echo "[11/12] CMMLU | LlamaCpp | FP32 (Simulated) | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --output "$CMMLU_DIR/llamacpp_fp32_notf32.jsonl"

echo "[12/12] CMMLU | LlamaCpp | FP32 (Simulated) | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --output "$CMMLU_DIR/llamacpp_fp32_tf32.jsonl"
unset NVIDIA_TF32_OVERRIDE

echo "======================================================="
echo "All 12 CMMLU benchmarks completed successfully!"
echo "======================================================="
