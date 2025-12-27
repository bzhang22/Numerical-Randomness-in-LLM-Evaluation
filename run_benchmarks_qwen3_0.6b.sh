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
echo "Running Benchmarks for $MODEL (Seed: $SEED)"
echo "Output Directories:"
echo "  PPL:   $PPL_DIR"
echo "  QA:    $QA_DIR"
echo "  CMMLU: $CMMLU_DIR"
echo "  GSM8K: $GSM8K_DIR"
echo "  PIQA:  $PIQA_DIR"
echo "======================================================="

# =======================================================
# PERPLEXITY BENCHMARKS (12 Runs)
# =======================================================

echo "--- Perplexity Benchmarks ---"

# --- HuggingFace Backend ---
echo "[1/60] PPL | HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float16 --output "$PPL_DIR/hf_fp16_notf32.jsonl"

echo "[2/60] PPL | HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float16 --tf32 --output "$PPL_DIR/hf_fp16_tf32.jsonl"

echo "[3/60] PPL | HF | FP32 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float32 --output "$PPL_DIR/hf_fp32_notf32.jsonl"

echo "[4/60] PPL | HF | FP32 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float32 --tf32 --output "$PPL_DIR/hf_fp32_tf32.jsonl"

# --- vLLM Backend ---
echo "[5/60] PPL | vLLM | FP16 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --output "$PPL_DIR/vllm_fp16_notf32.jsonl"

echo "[6/60] PPL | vLLM | FP16 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --tf32 --output "$PPL_DIR/vllm_fp16_tf32.jsonl"

echo "[7/60] PPL | vLLM | FP32 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --output "$PPL_DIR/vllm_fp32_notf32.jsonl"

echo "[8/60] PPL | vLLM | FP32 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --tf32 --output "$PPL_DIR/vllm_fp32_tf32.jsonl"

# --- LlamaCpp Backend ---
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

# =======================================================
# COMMONSENSE QA BENCHMARKS (12 Runs)
# =======================================================

echo "--- CommonsenseQA Benchmarks ---"

# --- HuggingFace Backend ---
echo "[13/60] QA | HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --output "$QA_DIR/hf_fp16_notf32.jsonl"

echo "[14/60] QA | HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --tf32 --output "$QA_DIR/hf_fp16_tf32.jsonl"

echo "[15/60] QA | HF | FP32 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --output "$QA_DIR/hf_fp32_notf32.jsonl"

echo "[16/60] QA | HF | FP32 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --tf32 --output "$QA_DIR/hf_fp32_tf32.jsonl"

# --- vLLM Backend ---
echo "[17/60] QA | vLLM | FP16 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --output "$QA_DIR/vllm_fp16_notf32.jsonl"

echo "[18/60] QA | vLLM | FP16 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --tf32 --output "$QA_DIR/vllm_fp16_tf32.jsonl"

echo "[19/60] QA | vLLM | FP32 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --output "$QA_DIR/vllm_fp32_notf32.jsonl"

echo "[20/60] QA | vLLM | FP32 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --tf32 --output "$QA_DIR/vllm_fp32_tf32.jsonl"

# --- LlamaCpp Backend ---
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

# =======================================================
# CMMLU BENCHMARKS (12 Runs)
# =======================================================

echo "--- CMMLU Benchmarks ($CMMLU_SUBSET) ---"

# --- HuggingFace Backend ---
echo "[25/60] CMMLU | HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --output "$CMMLU_DIR/hf_fp16_notf32.jsonl"

echo "[26/60] CMMLU | HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --tf32 --output "$CMMLU_DIR/hf_fp16_tf32.jsonl"

echo "[27/60] CMMLU | HF | FP32 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --output "$CMMLU_DIR/hf_fp32_notf32.jsonl"

echo "[28/60] CMMLU | HF | FP32 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --tf32 --output "$CMMLU_DIR/hf_fp32_tf32.jsonl"

# --- vLLM Backend ---
echo "[29/60] CMMLU | vLLM | FP16 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --output "$CMMLU_DIR/vllm_fp16_notf32.jsonl"

echo "[30/60] CMMLU | vLLM | FP16 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --tf32 --output "$CMMLU_DIR/vllm_fp16_tf32.jsonl"

echo "[31/60] CMMLU | vLLM | FP32 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --output "$CMMLU_DIR/vllm_fp32_notf32.jsonl"

echo "[32/60] CMMLU | vLLM | FP32 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --tf32 --output "$CMMLU_DIR/vllm_fp32_tf32.jsonl"

# --- LlamaCpp Backend ---
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

# =======================================================
# GSM8K BENCHMARKS (PPL) (12 Runs)
# =======================================================

echo "--- GSM8K Benchmarks (Perplexity) ---"

# --- HuggingFace Backend ---
echo "[37/60] GSM8K | HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float16 --output "$GSM8K_DIR/hf_fp16_notf32.jsonl"

echo "[38/60] GSM8K | HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float16 --tf32 --output "$GSM8K_DIR/hf_fp16_tf32.jsonl"

echo "[39/60] GSM8K | HF | FP32 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float32 --output "$GSM8K_DIR/hf_fp32_notf32.jsonl"

echo "[40/60] GSM8K | HF | FP32 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float32 --tf32 --output "$GSM8K_DIR/hf_fp32_tf32.jsonl"

# --- vLLM Backend ---
echo "[41/60] GSM8K | vLLM | FP16 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --output "$GSM8K_DIR/vllm_fp16_notf32.jsonl"

echo "[42/60] GSM8K | vLLM | FP16 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --tf32 --output "$GSM8K_DIR/vllm_fp16_tf32.jsonl"

echo "[43/60] GSM8K | vLLM | FP32 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --output "$GSM8K_DIR/vllm_fp32_notf32.jsonl"

echo "[44/60] GSM8K | vLLM | FP32 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --tf32 --output "$GSM8K_DIR/vllm_fp32_tf32.jsonl"

# --- LlamaCpp Backend ---
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

# =======================================================
# PIQA BENCHMARKS (QA) (12 Runs)
# =======================================================

echo "--- PIQA Benchmarks (QA) ---"

# --- HuggingFace Backend ---
echo "[49/60] PIQA | HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --output "$PIQA_DIR/hf_fp16_notf32.jsonl"

echo "[50/60] PIQA | HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --tf32 --output "$PIQA_DIR/hf_fp16_tf32.jsonl"

echo "[51/60] PIQA | HF | FP32 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --output "$PIQA_DIR/hf_fp32_notf32.jsonl"

echo "[52/60] PIQA | HF | FP32 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --tf32 --output "$PIQA_DIR/hf_fp32_tf32.jsonl"

# --- vLLM Backend ---
echo "[53/60] PIQA | vLLM | FP16 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --output "$PIQA_DIR/vllm_fp16_notf32.jsonl"

echo "[54/60] PIQA | vLLM | FP16 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --tf32 --output "$PIQA_DIR/vllm_fp16_tf32.jsonl"

echo "[55/60] PIQA | vLLM | FP32 | TF32: Off"
$CONDA_CMD --backend vllm --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --output "$PIQA_DIR/vllm_fp32_notf32.jsonl"

echo "[56/60] PIQA | vLLM | FP32 | TF32: On"
$CONDA_CMD --backend vllm --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --tf32 --output "$PIQA_DIR/vllm_fp32_tf32.jsonl"

# --- LlamaCpp Backend ---
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
echo "All 60 benchmarks completed successfully!"
echo "======================================================="
