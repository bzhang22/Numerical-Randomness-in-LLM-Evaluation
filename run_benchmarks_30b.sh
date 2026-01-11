#!/bin/bash
# set -e (Disabled to allow continuation after OOM)

# Define variables
# Use absolute path to python in env
CONDA_CMD="/home/bohanzhang1/miniconda3/envs/llm_randomness/bin/python run_benchmark_lmeval.py --load_in_4bit --batch_size 1"
MODEL="Qwen/Qwen2.5-32B-Instruct"
# GGUF_MODEL_PATH="./models/Qwen2.5-32B-Instruct.gguf" # specific GGUF path if you have one
SEED=123
LIMIT_PPL=1000 # Increased limit for PPL too? Assuming usually file based, but for consistency.
LIMIT_QA=1000 # Increased to 1000
GPU_UTIL=0.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CMMLU_SUBSET="agronomy"

# Output Directories
BASE_DIR="qwen2.5-32b_lmeval"
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
# PERPLEXITY BENCHMARKS
# =======================================================

echo "--- Perplexity Benchmarks (Wikitext) ---"

# --- HuggingFace Backend ---
echo "[1/60] PPL | HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float16 --output "$PPL_DIR/hf_fp16_notf32.json"

echo "[2/60] PPL | HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float16 --tf32 --output "$PPL_DIR/hf_fp16_tf32.json"

echo "[3/60] PPL | HF | FP32 | TF32: Off (May OOM)"
$CONDA_CMD --backend hf --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float32 --output "$PPL_DIR/hf_fp32_notf32.json" || echo "Failed (OOM?)"

echo "[4/60] PPL | HF | FP32 | TF32: On (May OOM)"
$CONDA_CMD --backend hf --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float32 --tf32 --output "$PPL_DIR/hf_fp32_tf32.json" || echo "Failed (OOM?)"

# --- vLLM Backend ---
# echo "[5/60] PPL | vLLM | FP16 | TF32: Off"
# $CONDA_CMD --backend vllm --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --output "$PPL_DIR/vllm_fp16_notf32.json"

# echo "[6/60] PPL | vLLM | FP16 | TF32: On"
# $CONDA_CMD --backend vllm --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --tf32 --output "$PPL_DIR/vllm_fp16_tf32.json"

# echo "[7/60] PPL | vLLM | FP32 | TF32: Off (May OOM)"
# $CONDA_CMD --backend vllm --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --output "$PPL_DIR/vllm_fp32_notf32.json" || echo "Failed (OOM?)"

# echo "[8/60] PPL | vLLM | FP32 | TF32: On (May OOM)"
# $CONDA_CMD --backend vllm --model $MODEL --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --tf32 --output "$PPL_DIR/vllm_fp32_tf32.json" || echo "Failed (OOM?)"


# =======================================================
# COMMONSENSE QA BENCHMARKS
# =======================================================

echo "--- CommonsenseQA Benchmarks ---"

# --- HuggingFace Backend ---
echo "[13/60] QA | HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --output "$QA_DIR/hf_fp16_notf32.json"

echo "[14/60] QA | HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --tf32 --output "$QA_DIR/hf_fp16_tf32.json"

echo "[15/60] QA | HF | FP32 | TF32: Off (May OOM)"
$CONDA_CMD --backend hf --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --output "$QA_DIR/hf_fp32_notf32.json" || echo "Failed (OOM?)"

echo "[16/60] QA | HF | FP32 | TF32: On (May OOM)"
$CONDA_CMD --backend hf --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --tf32 --output "$QA_DIR/hf_fp32_tf32.json" || echo "Failed (OOM?)"

# --- vLLM Backend ---
# echo "[17/60] QA | vLLM | FP16 | TF32: Off"
# $CONDA_CMD --backend vllm --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --output "$QA_DIR/vllm_fp16_notf32.json"

# echo "[18/60] QA | vLLM | FP16 | TF32: On"
# $CONDA_CMD --backend vllm --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --tf32 --output "$QA_DIR/vllm_fp16_tf32.json"

# echo "[19/60] QA | vLLM | FP32 | TF32: Off (May OOM)"
# $CONDA_CMD --backend vllm --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --output "$QA_DIR/vllm_fp32_notf32.json" || echo "Failed (OOM?)"

# echo "[20/60] QA | vLLM | FP32 | TF32: On (May OOM)"
# $CONDA_CMD --backend vllm --model $MODEL --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --tf32 --output "$QA_DIR/vllm_fp32_tf32.json" || echo "Failed (OOM?)"

# =======================================================
# CMMLU BENCHMARKS
# =======================================================

echo "--- CMMLU Benchmarks ($CMMLU_SUBSET) ---"

# --- HuggingFace Backend ---
echo "[25/60] CMMLU | HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --output "$CMMLU_DIR/hf_fp16_notf32.json"

echo "[26/60] CMMLU | HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --tf32 --output "$CMMLU_DIR/hf_fp16_tf32.json"

echo "[27/60] CMMLU | HF | FP32 | TF32: Off (May OOM)"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --output "$CMMLU_DIR/hf_fp32_notf32.json" || echo "Failed (OOM?)"

echo "[28/60] CMMLU | HF | FP32 | TF32: On (May OOM)"
$CONDA_CMD --backend hf --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --tf32 --output "$CMMLU_DIR/hf_fp32_tf32.json" || echo "Failed (OOM?)"

# --- vLLM Backend ---
# echo "[29/60] CMMLU | vLLM | FP16 | TF32: Off"
# $CONDA_CMD --backend vllm --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --output "$CMMLU_DIR/vllm_fp16_notf32.json"

# echo "[30/60] CMMLU | vLLM | FP16 | TF32: On"
# $CONDA_CMD --backend vllm --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --tf32 --output "$CMMLU_DIR/vllm_fp16_tf32.json"

# echo "[31/60] CMMLU | vLLM | FP32 | TF32: Off (May OOM)"
# $CONDA_CMD --backend vllm --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --output "$CMMLU_DIR/vllm_fp32_notf32.json" || echo "Failed (OOM?)"

# echo "[32/60] CMMLU | vLLM | FP32 | TF32: On (May OOM)"
# $CONDA_CMD --backend vllm --model $MODEL --dataset cmmlu --dataset_subset $CMMLU_SUBSET --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --tf32 --output "$CMMLU_DIR/vllm_fp32_tf32.json" || echo "Failed (OOM?)"

# =======================================================
# GSM8K BENCHMARKS (PPL)
# =======================================================

echo "--- GSM8K Benchmarks (Perplexity) ---"
# Note: lm_eval might try to run standard gsm8k which is generation-based, not just PPL.
# If we just want PPL, we need to check if lm_eval supports PPL on GSM8K natively?
# Usually GSM8K is CoT generation. 
# We'll run it and see if lm_eval defaults to generation (which is fine, just slower).
# Or we can skip if not desired. Assuming we run it.

# --- HuggingFace Backend ---
echo "[37/60] GSM8K | HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float16 --output "$GSM8K_DIR/hf_fp16_notf32.json"

echo "[38/60] GSM8K | HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float16 --tf32 --output "$GSM8K_DIR/hf_fp16_tf32.json"

echo "[39/60] GSM8K | HF | FP32 | TF32: Off (May OOM)"
$CONDA_CMD --backend hf --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float32 --output "$GSM8K_DIR/hf_fp32_notf32.json" || echo "Failed (OOM?)"

echo "[40/60] GSM8K | HF | FP32 | TF32: On (May OOM)"
$CONDA_CMD --backend hf --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --dtype float32 --tf32 --output "$GSM8K_DIR/hf_fp32_tf32.json" || echo "Failed (OOM?)"

# --- vLLM Backend ---
# echo "[41/60] GSM8K | vLLM | FP16 | TF32: Off"
# $CONDA_CMD --backend vllm --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --output "$GSM8K_DIR/vllm_fp16_notf32.json"

# echo "[42/60] GSM8K | vLLM | FP16 | TF32: On"
# $CONDA_CMD --backend vllm --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --tf32 --output "$GSM8K_DIR/vllm_fp16_tf32.json"

# echo "[43/60] GSM8K | vLLM | FP32 | TF32: Off (May OOM)"
# $CONDA_CMD --backend vllm --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --output "$GSM8K_DIR/vllm_fp32_notf32.json" || echo "Failed (OOM?)"

# echo "[44/60] GSM8K | vLLM | FP32 | TF32: On (May OOM)"
# $CONDA_CMD --backend vllm --model $MODEL --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --tf32 --output "$GSM8K_DIR/vllm_fp32_tf32.json" || echo "Failed (OOM?)"

# =======================================================
# PIQA BENCHMARKS (QA)
# =======================================================

echo "--- PIQA Benchmarks (QA) ---"

# --- HuggingFace Backend ---
echo "[49/60] PIQA | HF | FP16 | TF32: Off"
$CONDA_CMD --backend hf --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --output "$PIQA_DIR/hf_fp16_notf32.json"

echo "[50/60] PIQA | HF | FP16 | TF32: On"
$CONDA_CMD --backend hf --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --dtype float16 --tf32 --output "$PIQA_DIR/hf_fp16_tf32.json"

echo "[51/60] PIQA | HF | FP32 | TF32: Off (May OOM)"
$CONDA_CMD --backend hf --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --output "$PIQA_DIR/hf_fp32_notf32.json" || echo "Failed (OOM?)"

echo "[52/60] PIQA | HF | FP32 | TF32: On (May OOM)"
$CONDA_CMD --backend hf --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --dtype float32 --tf32 --output "$PIQA_DIR/hf_fp32_tf32.json" || echo "Failed (OOM?)"

# --- vLLM Backend ---
# echo "[53/60] PIQA | vLLM | FP16 | TF32: Off"
# $CONDA_CMD --backend vllm --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --output "$PIQA_DIR/vllm_fp16_notf32.json"

# echo "[54/60] PIQA | vLLM | FP16 | TF32: On"
# $CONDA_CMD --backend vllm --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float16 --tf32 --output "$PIQA_DIR/vllm_fp16_tf32.json"

# echo "[55/60] PIQA | vLLM | FP32 | TF32: Off (May OOM)"
# $CONDA_CMD --backend vllm --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --output "$PIQA_DIR/vllm_fp32_notf32.json" || echo "Failed (OOM?)"

# echo "[56/60] PIQA | vLLM | FP32 | TF32: On (May OOM)"
# $CONDA_CMD --backend vllm --model $MODEL --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --gpu_memory_utilization $GPU_UTIL --dtype float32 --tf32 --output "$PIQA_DIR/vllm_fp32_tf32.json" || echo "Failed (OOM?)"

echo "======================================================="
echo "Benchmarks completed successfully!"
echo "======================================================="
