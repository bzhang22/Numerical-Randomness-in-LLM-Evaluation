#!/bin/bash
#SBATCH --job-name="Attention_Sweep"
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=2-00:00:00
#SBATCH --output="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_attention/all_attention_%j.out"
#SBATCH --error="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_attention/all_attention_%j.err"

source ~/.bashrc
export HF_HOME=/blue/liguanpeng/bohanzhang1/hf_home
export PIP_CACHE_DIR=/blue/liguanpeng/bohanzhang1/pip_cache
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_ALLOW_CODE_EVAL=1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness

cd /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/
LOG_DIR="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/logs_attention"
mkdir -p $LOG_DIR

CMD="lm_eval"
if command -v lm-eval &> /dev/null; then
    CMD="lm-eval"
fi

TASKS="piqa,cmmlu,gsm8k,humaneval"
ATTNS=("eager" "sdpa" "flash_attention_2")
DTYPE="bfloat16"

MODELS=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "Meta-Llama-3.1-8B"
    "mistralai/Mistral-7B-v0.3"
    "google/gemma-2-27b"
    "01-ai/Yi-34B"
    "meta-llama/Meta-Llama-3.1-70B"
)

echo "=== Comprehensive Attention Sweep Job Started ==="
echo "Node: $(hostname)"

for model in "${MODELS[@]}"; do
    model_name=$(basename $model)
    
    # Correct local paths if un-prefixed for Meta models directly in the group space, 
    # but lm_eval handles HF paths natively if they match or auth is configured.
    if [[ "$model" == "Meta-Llama-3.1-8B" ]]; then
        model="meta-llama/Meta-Llama-3.1-8B"
    fi

    for attn in "${ATTNS[@]}"; do
        echo "--------------------------------------------------------"
        echo "Running Attention Sweep: Model=$model, Attn=$attn, Dtype=$DTYPE"
        out_dir="results/${model_name}/attention/${attn}"
        mkdir -p $out_dir
        
        # Check if output already exists to avoid duplicate work sequentially!
        if [ -d "$out_dir" ] && [ "$(ls -A $out_dir)" ]; then
            echo "Results already exist in $out_dir. Skipping."
            continue
        fi

        $CMD --model hf \
            --model_args pretrained=${model},dtype=${DTYPE},trust_remote_code=True,parallelize=True,attn_implementation=${attn} \
            --tasks ${TASKS} \
            --trust_remote_code \
            --confirm_run_unsafe_code \
            --batch_size 1 \
            --gen_kwargs temperature=0,do_sample=False \
            --output_path ${out_dir} \
            --log_samples
            
        echo "Completed Attn=$attn for $model_name"
    done
done

echo "=== All Attention Sweeps Completed ==="
