import argparse
import json
import os
import lm_eval
from lm_eval import utils
from lm_eval.models.huggingface import HFLM
# vLLM integration in lm_eval often works via 'vllm' model type string

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "vllm"], required=True)
    parser.add_argument("--model", required=True, help="HF model ID or path")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_subset", type=str, default=None)
    parser.add_argument("--task", type=str, default="qa") # evaluated via lm_eval tasks
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--batch_size", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default=None, help="eager, sdpa, or flash_attention_2")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic algorithms")
    
    args = parser.parse_args()
    
    if args.deterministic:
        import torch
        print("Enabling deterministic mode...")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Map dataset/custom args to lm_eval task names
    # lm_eval uses specific task registry names
    task_name = args.dataset
    if args.dataset == "cmmlu":
        if args.dataset_subset:
            task_name = f"cmmlu_{args.dataset_subset}"
        else:
            task_name = "cmmlu" # or all subsets? usually cmmlu is a group
    elif args.dataset == "wikitext":
        task_name = "wikitext" # mapping to specific wikitext task
        # wikitext in lm_eval is usually perplexity based?
    
    # Prepare model args
    model_args = f"pretrained={args.model},dtype={args.dtype},trust_remote_code=True"
    if args.attn_implementation:
        model_args += f",attn_implementation={args.attn_implementation}"
    
    if args.backend == "vllm":
        model_type = "vllm"
        model_args += f",gpu_memory_utilization={args.gpu_memory_utilization}"
        # vllm in lm_eval might need tensor_parallel_size etc if multi-gpu
    else:
        model_type = "hf"
        if args.device == "cuda":
            # If 8bit/4bit, device_map=auto is usually required/good
            if args.load_in_8bit or args.load_in_4bit:
                model_args += ",device_map=auto"
            else:
                # If explicit device not set, or we want auto?
                # User script passes --device cuda.
                # If we don't have quant, we might OOM. 
                # Let's adhere to arg, but if user wants auto, they should omit device?
                # But current script passes --device cuda.
                # Let's force device_map=auto if we suspect OOM or just default to it?
                # No, standard run_benchmark behavior was explicit.
                # But for lm_eval, parallelize=True implies device_map=auto.
                # We'll stick to single device unless quant is on.
                # Or better: always use device_map=auto if backend is hf?
                # It safer for large models.
                model_args += ",device_map=auto" 
        
        if args.load_in_8bit:
            model_args += ",load_in_8bit=True"
        if args.load_in_4bit:
            model_args += ",load_in_4bit=True"
        
        # TF32? lm_eval doesn't have direct arg, handled by torch global setting
    
    if args.tf32:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print(f"Running lm_eval with model={model_type}, args={model_args}, task={task_name}, limit={args.limit}")
    
    try:
        results = lm_eval.simple_evaluate(
            model=model_type,
            model_args=model_args,
            tasks=[task_name],
            limit=args.limit,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            batch_size=args.batch_size,
            # device=args.device if args.backend == "hf" else None, 
        )
        
        # Save results
        print(f"Saving results to {args.output}")
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
            
        print(utils.make_table(results))
        
    except Exception as e:
        print(f"Error running evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
