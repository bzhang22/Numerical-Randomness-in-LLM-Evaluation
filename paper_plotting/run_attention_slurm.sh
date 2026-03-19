#!/bin/bash
source ~/.bashrc
export HF_HOME=/blue/liguanpeng/bohanzhang1/hf_home
export PIP_CACHE_DIR=/blue/liguanpeng/bohanzhang1/pip_cache
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_ALLOW_CODE_EVAL=1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness

cd /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/

TASKS="piqa,cmmlu,gsm8k,humaneval"
model_name=$(basename $MODEL)

echo "=== Attention Sweep Job Started ==="
echo "Node: $(hostname)"
echo "MODEL: $MODEL"

CMD="lm_eval"
if command -v lm-eval &> /dev/null; then
    CMD="lm-eval"
fi

ATTNS=("eager" "sdpa" "flash_attention_2")
DTYPE="bfloat16"

for attn in "${ATTNS[@]}"; do
    echo "--------------------------------------------------------"
    echo "Running Attention Sweep: Model=$MODEL, Attn=$attn, Dtype=$DTYPE"
    out_dir="results/${model_name}/attention/${attn}"
    mkdir -p $out_dir
    
    $CMD --model hf \
        --model_args pretrained=${MODEL},dtype=${DTYPE},trust_remote_code=True,parallelize=True,attn_implementation=${attn} \
        --tasks ${TASKS} \
        --trust_remote_code \
        --confirm_run_unsafe_code \
        --batch_size 1 \
        --gen_kwargs temperature=0,do_sample=False \
        --output_path ${out_dir} \
        --log_samples
        
    echo "Completed Attn=$attn"
done

echo "=== Attention Sweep Job Completed ==="
