#!/bin/bash
# set -e

# Diagnosis Phase 3: Combinatorial Tests (Qwen3 0.6B / Qwen2.5 0.5B-Instruct)
# Focus: Can Eager Attn or GEMM Det fix Batch Size 4 issues?

MODEL="Qwen/Qwen2.5-0.5B-Instruct" 
CONDA_CMD="/home/bohanzhang1/miniconda3/envs/llm_randomness/bin/python run_benchmark_lmeval.py"
LIMIT=100
SEED=123
OUT_DIR="diagnosis_qwen3_combinatorial"

mkdir -p "$OUT_DIR"

echo "======================================================="
echo "Running Diagnosis Phase 3 (Combinatorial: $MODEL)"
echo "Output Directory: $OUT_DIR"
echo "Target: Batch Size 4 Interactions"
echo "======================================================="

DATASETS=("wikitext" "piqa" "commonsense_qa")

for DATASET in "${DATASETS[@]}"; do
    TASK_TYPE="qa"
    if [ "$DATASET" == "wikitext" ]; then
        TASK_TYPE="perplexity"
    fi

    echo "--- Dataset: $DATASET ($TASK_TYPE) ---"

    # 1. Control: Batch 4 (Default Attn, Default GEMM)
    # Re-running to ensure comparable environment in this session
    echo "[$DATASET] [1/4] Batch 4 (Control)..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 4 --output "$OUT_DIR/${DATASET}_batch4_control.json"

    # 2. Batch 4 + Attn Eager
    echo "[$DATASET] [2/4] Batch 4 + Attn Eager..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 4 --attn_implementation eager --output "$OUT_DIR/${DATASET}_batch4_eager.json"

    # 3. Batch 4 + GEMM Deterministic
    echo "[$DATASET] [3/4] Batch 4 + GEMM Deterministic..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 4 --deterministic --output "$OUT_DIR/${DATASET}_batch4_gemm_det.json"

    # 4. Batch 4 + Attn Eager + GEMM Deterministic (Full Det)
    echo "[$DATASET] [4/4] Batch 4 + Attn Eager + GEMM Det..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 4 --attn_implementation eager --deterministic --output "$OUT_DIR/${DATASET}_batch4_full_det.json"

    echo "--- Completed $DATASET ---"
    echo ""
done

echo "======================================================="
echo "Diagnosis Phase 3 Completed."
echo "======================================================="
