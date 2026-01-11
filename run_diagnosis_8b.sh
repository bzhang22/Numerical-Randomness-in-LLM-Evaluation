#!/bin/bash
# set -e

# Diagnosis Script for Qwen3 8B
# Factors: Batch Size, Attention Implementation, GEMM Determinism

CONDA_CMD="/home/bohanzhang1/miniconda3/envs/llm_randomness/bin/python run_benchmark_lmeval.py"
MODEL="Qwen/Qwen3-8B"
DATASET="wikitext"
TASK="perplexity"
LIMIT=100
SEED=123
OUT_DIR="diagnosis_8b"

mkdir -p "$OUT_DIR"

echo "======================================================="
echo "Running Diagnosis for $MODEL"
echo "Output Directory: $OUT_DIR"
echo "======================================================="

# 1. Config A (Baseline): Batch 1, Default Attn, Default GEMM
echo "[1/5] Baseline (Batch 1, Default Attn, Default GEMM)..."
$CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK --limit $LIMIT --seed $SEED --dtype float16 --batch_size 1 --output "$OUT_DIR/baseline_batch1.json"

# 2. Factor 1: Batch Size Variants
echo "[2/5] Batch Size 4..."
$CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK --limit $LIMIT --seed $SEED --dtype float16 --batch_size 4 --output "$OUT_DIR/batch4.json" || echo "Batch 4 Failed"

echo "[3/5] Batch Size 8..."
$CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK --limit $LIMIT --seed $SEED --dtype float16 --batch_size 8 --output "$OUT_DIR/batch8.json" || echo "Batch 8 Failed"

# 3. Factor 2: Attention Variants (vs Baseline)
# Math/Eager Attention (Deterministic-friendly)
echo "[4/5] Attention: Eager (Math)..."
$CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK --limit $LIMIT --seed $SEED --dtype float16 --batch_size 1 --attn_implementation eager --output "$OUT_DIR/attn_eager.json"

# 4. Factor 3: GEMM Determinism (vs Baseline)
# Deterministic Algorithms enabled
echo "[5/5] GEMM: Deterministic..."
$CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK --limit $LIMIT --seed $SEED --dtype float16 --batch_size 1 --deterministic --output "$OUT_DIR/gemm_deterministic.json"

echo "======================================================="
echo "Diagnosis Completed. Check $OUT_DIR for results."
echo "======================================================="
