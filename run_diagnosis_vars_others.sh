#!/bin/bash
# set -e

# Diagnosis Phase 8: Variables on Small (0.5B) & Medium (3B) Models
# Focus: check Attn Eager and Gemm Determinism effects

CONDA_CMD="/home/bohanzhang1/miniconda3/envs/llm_randomness/bin/python run_benchmark_lmeval.py"
LIMIT=1000
SEED=123
OUT_DIR="diagnosis_vars_others"

mkdir -p "$OUT_DIR"

MODELS=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct")
MODEL_NAMES=("0.5B" "3B")
DATASETS=("commonsense_qa" "piqa")

echo "======================================================="
echo "Running Diagnosis Phase 8"
echo "Limit: $LIMIT"
echo "Output Directory: $OUT_DIR"
echo "======================================================="

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    M_NAME="${MODEL_NAMES[$i]}"
    
    echo "======================================================="
    echo "MODEL: $MODEL ($M_NAME)"
    echo "======================================================="

    for DATASET in "${DATASETS[@]}"; do
        TASK_TYPE="qa"
        echo "--- Dataset: $DATASET ---"

        # 1. Baseline (Batch 1)
        echo "[$M_NAME] [$DATASET] [1/6] Baseline (Batch 1)..."
        $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 1 --output "$OUT_DIR/${M_NAME}_${DATASET}_baseline_b1.json"

        # 2. Attn Eager (Batch 1)
        echo "[$M_NAME] [$DATASET] [2/6] Attn Eager (Batch 1)..."
        $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 1 --attn_implementation eager --output "$OUT_DIR/${M_NAME}_${DATASET}_eager_b1.json"

        # 3. Gemm Det (Batch 1)
        echo "[$M_NAME] [$DATASET] [3/6] Gemm Deterministic (Batch 1)..."
        $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 1 --deterministic --output "$OUT_DIR/${M_NAME}_${DATASET}_det_b1.json"

        # 4. Baseline (Batch 32)
        echo "[$M_NAME] [$DATASET] [4/6] Baseline (Batch 32)..."
        $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 32 --output "$OUT_DIR/${M_NAME}_${DATASET}_baseline_b32.json"

        # 5. Attn Eager (Batch 32)
        echo "[$M_NAME] [$DATASET] [5/6] Attn Eager (Batch 32)..."
        $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 32 --attn_implementation eager --output "$OUT_DIR/${M_NAME}_${DATASET}_eager_b32.json"

        # 6. Gemm Det (Batch 32)
        echo "[$M_NAME] [$DATASET] [6/6] Gemm Deterministic (Batch 32)..."
        $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 32 --deterministic --output "$OUT_DIR/${M_NAME}_${DATASET}_det_b32.json"

        echo "--- Completed $DATASET for $M_NAME ---"
        echo ""
    done
done

echo "======================================================="
echo "Diagnosis Phase 8 Completed."
echo "======================================================="
