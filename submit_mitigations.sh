#!/bin/bash
#SBATCH --job-name=mitigation_array
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-167%1
#SBATCH --output=/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_mitigation/array_%A_%a.out
#SBATCH --error=/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_mitigation/array_%A_%a.err

LOG_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_mitigation"
OUT_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/results_mitigation"
mkdir -p $LOG_DIR
mkdir -p $OUT_DIR

DATASETS=("piqa" "gsm8k" "cmmlu" "humaneval")
VARIANTS=("bf16_baseline" "fp32_reference" "attention" "norm" "lm_head" "attention_lm_head")

MODELS=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Meta-Llama-3.1-8B"
    "mistralai/Mistral-7B-v0.3"
    "google/gemma-2-27b"
    "01-ai/Yi-34B"
    "meta-llama/Meta-Llama-3.1-70B"
)

# Generate cartesian product
TASK_LIST=()
for m in "${MODELS[@]}"; do
  for d in "${DATASETS[@]}"; do
    for v in "${VARIANTS[@]}"; do
      TASK_LIST+=("$m $d $v")
    done
  done
done

# Pull specific args based on Array Index
PARAMS=(${TASK_LIST[$SLURM_ARRAY_TASK_ID]})

MODEL=${PARAMS[0]}
DATASET=${PARAMS[1]}
VARIANT=${PARAMS[2]}

MODEL_NAME=$(basename $MODEL)
OUT_FILE="${OUT_DIR}/${MODEL_NAME}_${DATASET}_${VARIANT}.jsonl"

echo "Executing Cluster Node configuration: $MODEL | $DATASET | $VARIANT"

# Skip executing if the JSONL was already successfully compiled in the aborted sweep
if [ -s "$OUT_FILE" ] && [ -s "${OUT_FILE/.jsonl/_meta.json}" ]; then
    echo "Output $OUT_FILE already exists from previous runs. Skipping."
    exit 0
fi

/blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness/bin/python /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/run_mitigation.py \
    --model=$MODEL \
    --dataset=$DATASET \
    --variant=$VARIANT \
    --dtype=bfloat16 \
    --limit=500 \
    --output=$OUT_FILE
