#!/bin/bash
set -e

# Define variables
CONDA_CMD="$HOME/miniconda/bin/conda run -n llm_randomness python run_benchmark.py"
MODEL="Qwen/Qwen3-0.6B"
GGUF_MODEL_PATH="/home/UFAD/bohanzhang1/.cache/huggingface/hub/models--MaziyarPanahi--Qwen3-0.6B-GGUF/snapshots/16d75108d73a476af91a4f6df4cd77e854b42d04/Qwen3-0.6B.fp16.gguf"
SEED=123
LIMIT_PPL=20
LIMIT_QA=100
GPU_UTIL=0.6

# Output Directories
BASE_DIR="qwen3-0.6b"
GSM8K_DIR="$BASE_DIR/gsm8k_ppl"
PIQA_DIR="$BASE_DIR/piqa"

mkdir -p "$GSM8K_DIR"
mkdir -p "$PIQA_DIR"

echo "======================================================="
echo "Running GSM8K (PPL) and PIQA (QA) Benchmarks"
echo "Model: $MODEL (Seed: $SEED)"
echo "Output Directories:"
echo "  GSM8K: $GSM8K_DIR"
echo "  PIQA:  $PIQA_DIR"
echo "======================================================="

# =======================================================
# GSM8K BENCHMARKS (PPL) (12 Runs)
# =======================================================

echo "--- GSM8K Benchmarks (Perplexity) ---"

# --- HuggingFace Backend ---
echo "[1/24] GSM8K | HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float16 --output "$GSM8K_DIR/hf_fp16_notf32.jsonl"

echo "[2/24] GSM8K | HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float16 --tf32 --output "$GSM8K_DIR/hf_fp16_tf32.jsonl"

echo "[3/24] GSM8K | HF | FP32 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float32 --output "$GSM8K_DIR/hf_fp32_notf32.jsonl"

echo "[4/24] GSM8K | HF | FP32 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float32 --tf32 --output "$GSM8K_DIR/hf_fp32_tf32.jsonl"

# --- vLLM Backend ---
echo "[5/24] GSM8K | vLLM | FP16 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --output "$GSM8K_DIR/vllm_fp16_notf32.jsonl"

echo "[6/24] GSM8K | vLLM | FP16 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --tf32 --output "$GSM8K_DIR/vllm_fp16_tf32.jsonl"

echo "[7/24] GSM8K | vLLM | FP32 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --output "$GSM8K_DIR/vllm_fp32_notf32.jsonl"

echo "[8/24] GSM8K | vLLM | FP32 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --tf32 --output "$GSM8K_DIR/vllm_fp32_tf32.jsonl"

# --- LlamaCpp Backend ---
echo "[9/24] GSM8K | LlamaCpp | FP16 | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --output "$GSM8K_DIR/llamacpp_fp16_notf32.jsonl"

echo "[10/24] GSM8K | LlamaCpp | FP16 | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --output "$GSM8K_DIR/llamacpp_fp16_tf32.jsonl"

echo "[11/24] GSM8K | LlamaCpp | FP32 (Simulated) | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --output "$GSM8K_DIR/llamacpp_fp32_notf32.jsonl"

echo "[12/24] GSM8K | LlamaCpp | FP32 (Simulated) | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --output "$GSM8K_DIR/llamacpp_fp32_tf32.jsonl"
unset NVIDIA_TF32_OVERRIDE

# =======================================================
# PIQA BENCHMARKS (QA) (12 Runs)
# =======================================================

echo "--- PIQA Benchmarks (QA) ---"

# --- HuggingFace Backend ---
echo "[13/24] PIQA | HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --output "$PIQA_DIR/hf_fp16_notf32.jsonl"

echo "[14/24] PIQA | HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --tf32 --output "$PIQA_DIR/hf_fp16_tf32.jsonl"

echo "[15/24] PIQA | HF | FP32 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --output "$PIQA_DIR/hf_fp32_notf32.jsonl"

echo "[16/24] PIQA | HF | FP32 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --tf32 --output "$PIQA_DIR/hf_fp32_tf32.jsonl"

# --- vLLM Backend ---
echo "[17/24] PIQA | vLLM | FP16 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --output "$PIQA_DIR/vllm_fp16_notf32.jsonl"

echo "[18/24] PIQA | vLLM | FP16 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --tf32 --output "$PIQA_DIR/vllm_fp16_tf32.jsonl"

echo "[19/24] PIQA | vLLM | FP32 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --output "$PIQA_DIR/vllm_fp32_notf32.jsonl"

echo "[20/24] PIQA | vLLM | FP32 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --tf32 --output "$PIQA_DIR/vllm_fp32_tf32.jsonl"

# --- LlamaCpp Backend ---
echo "[21/24] PIQA | LlamaCpp | FP16 | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --output "$PIQA_DIR/llamacpp_fp16_notf32.jsonl"

echo "[22/24] PIQA | LlamaCpp | FP16 | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --output "$PIQA_DIR/llamacpp_fp16_tf32.jsonl"

echo "[23/24] PIQA | LlamaCpp | FP32 (Simulated) | TF32: Off"
export NVIDIA_TF32_OVERRIDE=0
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --output "$PIQA_DIR/llamacpp_fp32_notf32.jsonl"

echo "[24/24] PIQA | LlamaCpp | FP32 (Simulated) | TF32: On"
export NVIDIA_TF32_OVERRIDE=1
$CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL_PATH" --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --output "$PIQA_DIR/llamacpp_fp32_tf32.jsonl"
unset NVIDIA_TF32_OVERRIDE

echo "======================================================="
echo "All 24 GSM8K/PIQA benchmarks completed successfully!"
echo "======================================================="
