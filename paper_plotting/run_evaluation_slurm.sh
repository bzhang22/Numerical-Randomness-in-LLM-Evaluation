#!/bin/bash
source ~/.bashrc
export HF_HOME=/blue/liguanpeng/bohanzhang1/hf_home
export PIP_CACHE_DIR=/blue/liguanpeng/bohanzhang1/pip_cache
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
# export HF_TOKEN="<REDACTED>"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_ALLOW_CODE_EVAL=1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness

cd /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/

TASKS="wikitext,lambada_openai"
model_name=$(basename $MODEL)

echo "=== Job Started ==="
echo "Node: $(hostname)"
echo "MODEL: $MODEL"
echo "MODE: $MODE"

# lm-eval run command - using what's available
CMD="lm_eval"
if command -v lm-eval &> /dev/null; then
    CMD="lm-eval"
fi

if [ "$MODE" = "precision" ]; then
    DTYPES=("float32" "float16" "bfloat16")
    for dtype in "${DTYPES[@]}"; do
        echo "--------------------------------------------------------"
        echo "Running Precision Sweep: Model=$MODEL, Dtype=$dtype"
        out_dir="results/${model_name}/precision/${dtype}"
        mkdir -p $out_dir
        
        $CMD --model hf \
            --model_args pretrained=${MODEL},dtype=${dtype},trust_remote_code=True,parallelize=True \
            --tasks ${TASKS} \
            --trust_remote_code \
            --confirm_run_unsafe_code \
            --batch_size 1 \
            --gen_kwargs temperature=0,do_sample=False \
            --output_path ${out_dir} \
            --log_samples
            
        echo "Completed Dtype=$dtype"
    done
elif [ "$MODE" = "batch" ]; then
    BATCH_SIZES=(1 2 4 8 16)
    DTYPE="bfloat16"
    for bs in "${BATCH_SIZES[@]}"; do
        echo "--------------------------------------------------------"
        echo "Running Batch Sweep: Model=$MODEL, BS=$bs, Dtype=$DTYPE"
        out_dir="results/${model_name}/batch/bs${bs}"
        mkdir -p $out_dir
        
        $CMD --model hf \
            --model_args pretrained=${MODEL},dtype=${DTYPE},trust_remote_code=True,parallelize=True \
            --tasks ${TASKS} \
            --trust_remote_code \
            --confirm_run_unsafe_code \
            --batch_size ${bs} \
            --gen_kwargs temperature=0,do_sample=False \
            --output_path ${out_dir} \
            --log_samples
            
        echo "Completed BS=$bs"
    done
fi

echo "=== Job Completed ==="
