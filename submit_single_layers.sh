#!/bin/bash
#SBATCH --job-name=layer_sweep
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-111%30
#SBATCH --output=/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_mitigation/layer_sweep_%A_%a.out
#SBATCH --error=/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_mitigation/layer_sweep_%A_%a.err

LOG_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_mitigation"
OUT_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/results_mitigation"
mkdir -p $LOG_DIR
mkdir -p $OUT_DIR

DATASETS=("piqa" "cmmlu")
LAYERS=({0..27})
VARIANT="attention_norm"

MODELS=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
)

# Generate cartesian product
TASK_LIST=()
for m in "${MODELS[@]}"; do
  for d in "${DATASETS[@]}"; do
    for l in "${LAYERS[@]}"; do
      TASK_LIST+=("$m $d $l")
    done
  done
done

# Current index bounds: 2 models * 2 datasets * 28 layers = 112 tasks

# Pull specific args based on Array Index
PARAMS=(${TASK_LIST[$SLURM_ARRAY_TASK_ID]})

MODEL=${PARAMS[0]}
DATASET=${PARAMS[1]}
LAYER=${PARAMS[2]}

# Skip index bounds that exceed Llama-3.2-1B limits to save GPU hours
if [ "$MODEL" == "meta-llama/Llama-3.2-1B" ] && [ "$LAYER" -ge 16 ]; then
    echo "Skipping out of bounds layer $LAYER for 1B model."
    exit 0
fi

MODEL_NAME=$(basename $MODEL)
OUT_FILE="${OUT_DIR}/${MODEL_NAME}_${DATASET}_${VARIANT}_layer${LAYER}.jsonl"

echo "Executing Single Layer Injection: $MODEL | $DATASET | $VARIANT | Layer $LAYER"

if [ -s "$OUT_FILE" ] && [ -s "${OUT_FILE/.jsonl/_meta.json}" ]; then
    echo "Output $OUT_FILE already exists from previous runs. Skipping."
    exit 0
fi

# We use layer_split=$LAYER to target the exact int parameter logic we pushed earlier
/blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness/bin/python /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/run_mitigation.py \
    --model=$MODEL \
    --dataset=$DATASET \
    --variant=$VARIANT \
    --layer_split=$LAYER \
    --dtype=bfloat16 \
    --limit=1000 \
    --output=$OUT_FILE
