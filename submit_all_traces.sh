#!/bin/bash
# Slurm Precision Traces Sequential Dispatcher
#SBATCH --job-name="T_All"
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
#SBATCH --output="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_traces/all_traces_%j.out"
#SBATCH --error="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_traces/all_traces_%j.err"

LOG_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_traces"
mkdir -p $LOG_DIR

# Using only 1 tiny model and 1 dataset to squeeze into the remaining tiny group limit!
DATASETS=("piqa")

MODELS=(
    "meta-llama/Llama-3.2-1B"
)

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Processing $model on $dataset..."
        /blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness/bin/python /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/extract_mitigation_traces.py \
            --model=$model \
            --dataset=$dataset
    done
done

echo "All Sequential Trace Extractions Done!"
