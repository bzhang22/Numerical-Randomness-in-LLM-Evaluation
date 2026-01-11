#!/bin/bash
# set -e

# Diagnosis Phase 2: Small Model (Qwen3 0.6B / Qwen2.5 0.5B-Instruct)
# Datasets: wikitext, piqa, commonsense_qa
# Factors: Batch Size, Attn Impl, GEMM Determinism

# Use Qwen/Qwen2.5-0.5B-Instruct as standard proxy for "Qwen3 0.6B" request if 3 doesn't exist. 
# But let's try Qwen/Qwen3-0.5B first if available? No, safe bet is Qwen/Qwen2.5-0.5B-Instruct.
# Actually, I'll define MODEL variable so it's easy to change.
MODEL="Qwen/Qwen2.5-0.5B-Instruct" 
# Use absolute path
CONDA_CMD="/home/bohanzhang1/miniconda3/envs/llm_randomness/bin/python run_benchmark_lmeval.py"
LIMIT=100
SEED=123
OUT_DIR="diagnosis_qwen3_small"

mkdir -p "$OUT_DIR"

echo "======================================================="
echo "Running Diagnosis Phase 2 (Small Model: $MODEL)"
echo "Output Directory: $OUT_DIR"
echo "======================================================="

# Datasets Loop
DATASETS=("wikitext" "piqa" "commonsense_qa")

for DATASET in "${DATASETS[@]}"; do
    # Task Arg (qa or perplexity) - mostly for internal logging/logic if strict
    # wikitext is PPL, others QA
    TASK_TYPE="qa"
    if [ "$DATASET" == "wikitext" ]; then
        TASK_TYPE="perplexity"
    fi

    echo "--- Dataset: $DATASET ($TASK_TYPE) ---"

    # 1. Baseline: Batch 1, Default Attn, Default GEMM
    echo "[$DATASET] [1/5] Baseline (Batch 1)..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 1 --output "$OUT_DIR/${DATASET}_baseline.json"

    # 2. Batch 4
    echo "[$DATASET] [2/5] Batch 4..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 4 --output "$OUT_DIR/${DATASET}_batch4.json"

    # 3. Batch 8
    echo "[$DATASET] [3/5] Batch 8..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 8 --output "$OUT_DIR/${DATASET}_batch8.json"

    # 4. Attn Eager
    echo "[$DATASET] [4/5] Attn Eager (Batch 1)..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 1 --attn_implementation eager --output "$OUT_DIR/${DATASET}_attn_eager.json"

    # 5. GEMM Deterministic
    echo "[$DATASET] [5/5] GEMM Deterministic (Batch 1)..."
    $CONDA_CMD --backend hf --model $MODEL --dataset $DATASET --task $TASK_TYPE --limit $LIMIT --seed $SEED --dtype float16 --batch_size 1 --deterministic --output "$OUT_DIR/${DATASET}_gemm_det.json"

    echo "--- Completed $DATASET ---"
    echo ""
done

echo "======================================================="
echo "Diagnosis Phase 2 Completed."
echo "======================================================="
