#!/bin/bash
# set -e

# Diagnosis Phase 7: Variables on Large Model (Qwen3-8B)
# Focus: Check Attn Eager and Gemm Determinism robustness

MODEL="Qwen/Qwen3-8B" 
CONDA_CMD="/home/bohanzhang1/miniconda3/envs/llm_randomness/bin/python run_benchmark_lmeval.py"
LIMIT=1000
SEED=123
OUT_DIR="diagnosis_vars_large"

mkdir -p "$OUT_DIR"

echo "======================================================="
echo "Running Diagnosis Phase 7 (Variables 8B: $MODEL)"
echo "Limit: $LIMIT"
echo "Output Directory: $OUT_DIR"
echo "======================================================="

DATASETS=("commonsense_qa" "piqa")

for DATASET in "${DATASETS[@]}"; do
    TASK_TYPE="qa"
    echo "--- Dataset: $DATASET ---"

    # 1. Baseline (Batch 1) - Control
    echo "[$DATASET] [1/6] Baseline (Batch 1)..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 1 --output "$OUT_DIR/${DATASET}_baseline_b1.json"

    # 2. Attn Eager (Batch 1)
    echo "[$DATASET] [2/6] Attn Eager (Batch 1)..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 1 --attn_implementation eager --output "$OUT_DIR/${DATASET}_eager_b1.json"

    # 3. Gemm Det (Batch 1)
    echo "[$DATASET] [3/6] Gemm Deterministic (Batch 1)..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 1 --deterministic --output "$OUT_DIR/${DATASET}_det_b1.json"

    # 4. Baseline (Batch 32) - Control
    echo "[$DATASET] [4/6] Baseline (Batch 32)..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 32 --output "$OUT_DIR/${DATASET}_baseline_b32.json"

    # 5. Attn Eager (Batch 32)
    echo "[$DATASET] [5/6] Attn Eager (Batch 32)..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 32 --attn_implementation eager --output "$OUT_DIR/${DATASET}_eager_b32.json"

    # 6. Gemm Det (Batch 32)
    echo "[$DATASET] [6/6] Gemm Deterministic (Batch 32)..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 32 --deterministic --output "$OUT_DIR/${DATASET}_det_b32.json"

    echo "--- Completed $DATASET ---"
    echo ""
done

echo "======================================================="
echo "Diagnosis Phase 7 Completed."
echo "======================================================="
