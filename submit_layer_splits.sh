#!/bin/bash
#SBATCH --job-name=layer_mitigation
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-167%1
#SBATCH --output=/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_mitigation/layer_split_%A_%a.out
#SBATCH --error=/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_mitigation/layer_split_%A_%a.err

LOG_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_mitigation"
OUT_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/results_mitigation"
mkdir -p $LOG_DIR
mkdir -p $OUT_DIR

DATASETS=("piqa" "cmmlu")
SPLITS=("first_half" "last_half" "first_quarter" "last_quarter" "middle" "1" "1,-1")
# We test the complete attention+norm pipeline to see how placement affects stability
VARIANTS=("attention_norm" "attention" "attention_norm_lm_head")

MODELS=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "mistralai/Mistral-7B-v0.3"
    "meta-llama/Meta-Llama-3.1-8B"
)

# Generate cartesian product
TASK_LIST=()
for m in "${MODELS[@]}"; do
  for d in "${DATASETS[@]}"; do
    for v in "${VARIANTS[@]}"; do
      for s in "${SPLITS[@]}"; do
        TASK_LIST+=("$m $d $v $s")
      done
    done
  done
done

# Current index bounds: 4 models * 2 datasets * 3 variants * 7 splits = 168 tasks

# Pull specific args based on Array Index
PARAMS=(${TASK_LIST[$SLURM_ARRAY_TASK_ID]})

MODEL=${PARAMS[0]}
DATASET=${PARAMS[1]}
VARIANT=${PARAMS[2]}
SPLIT=${PARAMS[3]}

MODEL_NAME=$(basename $MODEL)
OUT_FILE="${OUT_DIR}/${MODEL_NAME}_${DATASET}_${VARIANT}_${SPLIT}.jsonl"

echo "Executing Mitigation Split: $MODEL | $DATASET | $VARIANT | $SPLIT"

if [ -s "$OUT_FILE" ] && [ -s "${OUT_FILE/.jsonl/_meta.json}" ]; then
    echo "Output $OUT_FILE already exists from previous runs. Skipping."
    exit 0
fi

/blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness/bin/python /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/run_mitigation.py \
    --model=$MODEL \
    --dataset=$DATASET \
    --variant=$VARIANT \
    --layer_split=$SPLIT \
    --dtype=bfloat16 \
    --limit=1000 \
    --output=$OUT_FILE
