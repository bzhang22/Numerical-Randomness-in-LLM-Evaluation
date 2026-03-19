#!/bin/bash

# CPU Benchmark Script
# Runs HF (FP32) and LlamaCpp (CPU) for all models.

# Setup
source ./venv/bin/activate
CONDA_CMD="$(pwd)/venv/bin/python run_benchmark.py"
LIMIT_PPL=20
LIMIT_QA=100
SEED=123

# Ensure output directories exist
mkdir -p cpu_results

# Function to run benchmarks for a specific model
# Function to run benchmarks for a specific model
run_model_benchmarks() {
    local NAME=$1
    local HF_MODEL=$2
    local GGUF_MODEL=$3
    
    echo "======================================================="
    echo "Running CPU Benchmarks for $NAME"
    echo "HF Model: $HF_MODEL"
    echo "GGUF Model: $GGUF_MODEL"
    echo "======================================================="
    
    # Structure: cpu/$NAME/{dataset_folder}
    local BASE_DIR="cpu/$NAME"
    local PPL_DIR="$BASE_DIR/ppl"
    local QA_DIR="$BASE_DIR/commonsense_qa"
    local CMMLU_DIR="$BASE_DIR/cmmlu"
    local GSM8K_DIR="$BASE_DIR/gsm8k_ppl"
    local PIQA_DIR="$BASE_DIR/piqa"
    
    mkdir -p "$PPL_DIR"
    mkdir -p "$QA_DIR"
    mkdir -p "$CMMLU_DIR"
    mkdir -p "$GSM8K_DIR"
    mkdir -p "$PIQA_DIR"
    
    # --- HuggingFace (CPU, FP32) ---
    echo "--- HF (CPU | FP32) ---"
    
    # 1. Wikitext PPL
    echo "Running Wikitext PPL..."
    $CONDA_CMD --backend hf --model "$HF_MODEL" --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --device cpu --dtype float32 --output "$PPL_DIR/hf_fp32_wikitext_ppl.jsonl"
    
    # 2. GSM8K PPL
    echo "Running GSM8K PPL..."
    $CONDA_CMD --backend hf --model "$HF_MODEL" --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --device cpu --dtype float32 --output "$GSM8K_DIR/hf_fp32_gsm8k_ppl.jsonl"
    
    # 3. CMMLU (Agronomy) QA
    echo "Running CMMLU QA..."
    $CONDA_CMD --backend hf --model "$HF_MODEL" --dataset cmmlu --dataset_subset agronomy --task qa --limit $LIMIT_QA --seed $SEED --device cpu --dtype float32 --output "$CMMLU_DIR/hf_fp32_cmmlu_qa.jsonl"
    
    # 4. CommonsenseQA QA
    echo "Running CommonsenseQA QA..."
    $CONDA_CMD --backend hf --model "$HF_MODEL" --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --device cpu --dtype float32 --output "$QA_DIR/hf_fp32_cqa_qa.jsonl"
    
    # 5. PIQA QA
    echo "Running PIQA QA..."
    $CONDA_CMD --backend hf --model "$HF_MODEL" --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --device cpu --dtype float32 --output "$PIQA_DIR/hf_fp32_piqa_qa.jsonl"

    # --- HuggingFace (CPU, FP16) ---
    echo "--- HF (CPU | FP16) ---"
    
    # 1. Wikitext PPL
    echo "Running Wikitext PPL (FP16)..."
    $CONDA_CMD --backend hf --model "$HF_MODEL" --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --device cpu --dtype float16 --output "$PPL_DIR/hf_fp16_wikitext_ppl.jsonl"
    
    # 2. GSM8K PPL
    echo "Running GSM8K PPL (FP16)..."
    $CONDA_CMD --backend hf --model "$HF_MODEL" --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --device cpu --dtype float16 --output "$GSM8K_DIR/hf_fp16_gsm8k_ppl.jsonl"
    
    # 3. CMMLU (Agronomy) QA
    echo "Running CMMLU QA (FP16)..."
    $CONDA_CMD --backend hf --model "$HF_MODEL" --dataset cmmlu --dataset_subset agronomy --task qa --limit $LIMIT_QA --seed $SEED --device cpu --dtype float16 --output "$CMMLU_DIR/hf_fp16_cmmlu_qa.jsonl"
    
    # 4. CommonsenseQA QA
    echo "Running CommonsenseQA QA (FP16)..."
    $CONDA_CMD --backend hf --model "$HF_MODEL" --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --device cpu --dtype float16 --output "$QA_DIR/hf_fp16_cqa_qa.jsonl"
    
    # 5. PIQA QA
    echo "Running PIQA QA (FP16)..."
    $CONDA_CMD --backend hf --model "$HF_MODEL" --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --device cpu --dtype float16 --output "$PIQA_DIR/hf_fp16_piqa_qa.jsonl"

    # --- LlamaCpp (CPU) ---
    echo "--- LlamaCpp (CPU) ---"
    
    # 1. Wikitext PPL
    echo "Running Wikitext PPL..."
    $CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL" --dataset wikitext --task perplexity --limit $LIMIT_PPL --seed $SEED --n_gpu_layers 0 --output "$PPL_DIR/llamacpp_wikitext_ppl.jsonl"
    
    # 2. GSM8K PPL
    echo "Running GSM8K PPL..."
    $CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL" --dataset gsm8k --task perplexity --limit $LIMIT_PPL --seed $SEED --n_gpu_layers 0 --output "$GSM8K_DIR/llamacpp_gsm8k_ppl.jsonl"
    
    # 3. CMMLU QA
    echo "Running CMMLU QA..."
    $CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL" --dataset cmmlu --dataset_subset agronomy --task qa --limit $LIMIT_QA --seed $SEED --n_gpu_layers 0 --output "$CMMLU_DIR/llamacpp_cmmlu_qa.jsonl"
    
    # 4. CommonsenseQA QA
    echo "Running CommonsenseQA QA..."
    $CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL" --dataset commonsense_qa --task qa --limit $LIMIT_QA --seed $SEED --n_gpu_layers 0 --output "$QA_DIR/llamacpp_cqa_qa.jsonl"
    
    # 5. PIQA QA
    echo "Running PIQA QA..."
    $CONDA_CMD --backend llama_cpp --model "$GGUF_MODEL" --dataset piqa --task qa --limit $LIMIT_QA --seed $SEED --n_gpu_layers 0 --output "$PIQA_DIR/llamacpp_piqa_qa.jsonl"
    
    echo "Completed $NAME"
    echo ""
}

# Run for all models
# Note: Ensure models are downloaded via download_assets.sh first!

# Qwen3-0.6B
# Mapped to Qwen/Qwen2.5-0.5B-Instruct
run_model_benchmarks "qwen3-0.6b" "Qwen/Qwen2.5-0.5B-Instruct" "./models/Qwen3-0.6B.fp16.gguf"

# Qwen3-4B
# Mapped to Qwen/Qwen2.5-3B-Instruct
run_model_benchmarks "qwen3-4b" "Qwen/Qwen2.5-3B-Instruct" "./models/Qwen3-4B.fp16.gguf"

# Qwen3-8B
# Mapped to Qwen/Qwen2.5-7B-Instruct
run_model_benchmarks "qwen3-8b" "Qwen/Qwen2.5-7B-Instruct" "./models/Qwen3-8B.fp16.gguf"

# Llama3-8B
run_model_benchmarks "llama3-8b" "NousResearch/Meta-Llama-3-8B" "./models/Meta-Llama-3-8B.Q4_K_M.gguf"

echo "All CPU benchmarks complete!"
