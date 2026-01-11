#!/bin/bash
# set -e

# Diagnosis Phase 4: High Sample (1000) & Batch Scale
# Focus: Determine if accuracy drop is stable across various batch sizes with N=1000

MODEL="Qwen/Qwen2.5-0.5B-Instruct" 
CONDA_CMD="/home/bohanzhang1/miniconda3/envs/llm_randomness/bin/python run_benchmark_lmeval.py"
LIMIT=1000
SEED=123
OUT_DIR="diagnosis_high_sample"

mkdir -p "$OUT_DIR"

echo "======================================================="
echo "Running Diagnosis Phase 4 (High Sample: $MODEL)"
echo "Limit: $LIMIT"
echo "Output Directory: $OUT_DIR"
echo "======================================================="

DATASETS=("commonsense_qa" "piqa")
BATCH_SIZES=(1 2 4 8 16 32)

for DATASET in "${DATASETS[@]}"; do
    TASK_TYPE="qa"
    # wikitext skipped for now as CQA is the main signal source
    
    echo "--- Dataset: $DATASET ---"

    for BS in "${BATCH_SIZES[@]}"; do
        echo "[$DATASET] Batch Size $BS..."
        $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size $BS --output "$OUT_DIR/${DATASET}_batch${BS}.json"
    done

    echo "--- Completed $DATASET ---"
    echo ""
done

echo "======================================================="
echo "Diagnosis Phase 4 Completed."
echo "======================================================="
