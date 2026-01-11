#!/bin/bash
# set -e

# Diagnosis Phase 6: Large Model (Qwen3-8B)
# Focus: Determine behavior of 8B model

MODEL="Qwen/Qwen3-8B" 
CONDA_CMD="/home/bohanzhang1/miniconda3/envs/llm_randomness/bin/python run_benchmark_lmeval.py"
LIMIT=1000
SEED=123
OUT_DIR="diagnosis_large"

mkdir -p "$OUT_DIR"

echo "======================================================="
echo "Running Diagnosis Phase 6 (Large Model: $MODEL)"
echo "Limit: $LIMIT"
echo "Output Directory: $OUT_DIR"
echo "======================================================="

DATASETS=("commonsense_qa" "piqa")
BATCH_SIZES=(1 2 4 8 16 32)

for DATASET in "${DATASETS[@]}"; do
    TASK_TYPE="qa"
    
    echo "--- Dataset: $DATASET ---"

    for BS in "${BATCH_SIZES[@]}"; do
        echo "[$DATASET] Batch Size $BS..."
        $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size $BS --output "$OUT_DIR/${DATASET}_batch${BS}.json"
    done

    echo "--- Completed $DATASET ---"
    echo ""
done

echo "======================================================="
echo "Diagnosis Phase 6 Completed."
echo "======================================================="
