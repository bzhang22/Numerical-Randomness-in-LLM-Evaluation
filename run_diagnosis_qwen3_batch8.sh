#!/bin/bash
# set -e

# Diagnosis Phase 4: Batch 8 Combinatorial Tests (Qwen3 0.6B / Qwen2.5 0.5B-Instruct)
# Focus: Interactions with Batch Size 8

MODEL="Qwen/Qwen2.5-0.5B-Instruct" 
CONDA_CMD="/home/bohanzhang1/miniconda3/envs/llm_randomness/bin/python run_benchmark_lmeval.py"
LIMIT=100
SEED=123
OUT_DIR="diagnosis_qwen3_combinatorial"

mkdir -p "$OUT_DIR"

echo "======================================================="
echo "Running Diagnosis Phase 4 (Batch 8 Combinatorial)"
echo "Output Directory: $OUT_DIR"
echo "======================================================="

DATASETS=("wikitext" "piqa" "commonsense_qa")

for DATASET in "${DATASETS[@]}"; do
    TASK_TYPE="qa"
    if [ "$DATASET" == "wikitext" ]; then
        TASK_TYPE="perplexity"
    fi

    echo "--- Dataset: $DATASET ($TASK_TYPE) ---"

    # 1. Control: Batch 8 (Default)
    echo "[$DATASET] [1/4] Batch 8 (Control)..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 8 --output "$OUT_DIR/${DATASET}_batch8_control.json"

    # 2. Batch 8 + Attn Eager
    echo "[$DATASET] [2/4] Batch 8 + Attn Eager..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 8 --attn_implementation eager --output "$OUT_DIR/${DATASET}_batch8_eager.json"

    # 3. Batch 8 + GEMM Deterministic
    echo "[$DATASET] [3/4] Batch 8 + GEMM Deterministic..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 8 --deterministic --output "$OUT_DIR/${DATASET}_batch8_gemm_det.json"

    # 4. Batch 8 + Full Det
    echo "[$DATASET] [4/4] Batch 8 + Attn Eager + GEMM Det..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 8 --attn_implementation eager --deterministic --output "$OUT_DIR/${DATASET}_batch8_full_det.json"

    echo "--- Completed $DATASET ---"
    echo ""
done

echo "======================================================="
echo "Diagnosis Phase 4 Completed."
echo "======================================================="
