#!/bin/bash
#SBATCH --job-name=llm_causality
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --gpus=1

source ~/.bashrc
export HF_HOME=/blue/liguanpeng/bohanzhang1/hf_home
export PIP_CACHE_DIR=/blue/liguanpeng/bohanzhang1/pip_cache

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness

cd /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/

echo "=== STAGE 1: CORE EXPERIMENT: Layer MAE Validation ==="
for model in "Qwen/Qwen2.5-3B" "NousResearch/Meta-Llama-3-8B"; do
    for data in "commonsense_qa" "piqa"; do
        for prec in "bfloat16" "float32"; do
            echo "Running $model on $data with $prec..."
            python /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/core_layer_mae.py \
                --model $model --dataset $data --limit 200 --dtype $prec \
                --out ${model: -2}_${data}_${prec}_core_mae.jsonl
        done
    done
done

echo "=== STAGE 2: INTERVENTION EXPERIMENTS ==="
for layer in 0 5 10 15 20 25 30; do
    echo "Running Clamping on Layer $layer..."
    python /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/intervention_experiments.py \
        --model Qwen/Qwen2.5-3B --dataset piqa --limit 50 --dtype bfloat16 \
        --mode clamp --layer $layer --out clamp_layer_${layer}_results.jsonl
done

for layer in 0 5 10 15 20 25 30; do
    echo "Running Noise Injection on Layer $layer..."
    python /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/intervention_experiments.py \
        --model Qwen/Qwen2.5-3B --dataset piqa --limit 50 --dtype bfloat16 \
        --mode noise --layer $layer --noise_scale 1e-4 --out noise_layer_${layer}_results.jsonl
done

echo "=== STAGE 3: MODULE BREAKDOWN ==="
python /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/attention_module_breakdown.py \
    --model Qwen/Qwen2.5-3B --limit 20 --dtype bfloat16 \
    --out module_breakdown_qwen3b.jsonl

echo "=== STAGE 4: PLOTTING DELIVERABLES ==="
python /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/plot_deliverables.py \
    --core_logs 3B_piqa_bfloat16_core_mae.jsonl 3B_piqa_float32_core_mae.jsonl \
    --core_labels "Qwen2.5-3B (BF16)" "Qwen2.5-3B (FP32)" \
    --clamp_logs clamp_layer_*.jsonl \
    --out_dir .

echo "=== PIPELINE COMPLETE ==="
