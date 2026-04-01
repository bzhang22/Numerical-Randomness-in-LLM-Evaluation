import argparse
import random
import numpy as np
import torch
import sys
import os
import gc
import json
from copy import deepcopy

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append("/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/")
from run_benchmark import DatasetLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_metrics(baseline_tensor, target_tensor):
    diff = baseline_tensor - target_tensor
    abs_diff = torch.abs(diff)
    mae = torch.mean(abs_diff).item()
    baseline_mean_abs = torch.mean(torch.abs(baseline_tensor)).item()
    rel_mae = mae / baseline_mean_abs if baseline_mean_abs > 0 else 0.0
    max_diff = torch.max(abs_diff).item()
    return {"mae": mae, "rel_mae": rel_mae, "max_diff": max_diff}

def get_hidden_states(model_name, dtype_str, attn_type, prompts, tokenizer):
    print(f"\n--- Loading {model_name} with {attn_type} ---")
    torch_dtype = getattr(torch, dtype_str)
    
    # Use single device or auto depending on size. Assuming node GPUs fit it.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype, 
        device_map="auto", 
        trust_remote_code=True, 
        attn_implementation=attn_type
    )
    model.eval()
    
    states_list = []
    top1_list = []
    
    print(f"Extracting states for {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        top1 = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
        top1_list.append(top1)
        
        layer_dict = {}
        for l_idx, state in enumerate(outputs.hidden_states[1:]):
            layer_dict[l_idx] = state[:, -1, :].detach().cpu().float()
        states_list.append(layer_dict)
        
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return states_list, top1_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset", type=str, default="piqa")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--out", type=str, default="attention_layer_variance_results.jsonl")
    args = parser.parse_args()

    set_all_seeds(123)
    split = "validation"
    if args.dataset in ["cmmlu", "gsm8k"]: split = "test"
    loader = DatasetLoader(args.dataset, split=split)
    items = loader.load(limit=args.limit)
    if not items: return
    prompts = [item["prompt"] for item in items] if isinstance(items[0], dict) else items

    print(f"=== Attention Layer Variance: {args.model} on {args.dataset} [{args.dtype}] ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 1. Baseline: EAGER
    eager_states, eager_top1 = get_hidden_states(args.model, args.dtype, "eager", prompts, tokenizer)
    
    # 2. Target 1: SDPA
    sdpa_states, sdpa_top1 = get_hidden_states(args.model, args.dtype, "sdpa", prompts, tokenizer)
    

    print("\nComputing per-layer metrics...")
    final_results = []
    num_layers = len(eager_states[0])
    
    for i in range(len(prompts)):
        # Compare Eager vs SDPA
        layer_metrics_sdpa = []
        for l in range(num_layers):
            metrics = calculate_metrics(eager_states[i][l], sdpa_states[i][l])
            layer_metrics_sdpa.append({"layer": l, **metrics})
            
        final_results.append({
            "prompt_idx": i,
            "model": args.model.split("/")[-1],
            "dataset": args.dataset,
            "dtype": args.dtype,
            "target_attn": "sdpa",
            "flip": (eager_top1[i] != sdpa_top1[i]),
            "baseline_top1": eager_top1[i],
            "target_top1": sdpa_top1[i],
            "layer_metrics": layer_metrics_sdpa
        })
        
        
    with open(args.out, 'a') as f:
        for res in final_results:
            f.write(json.dumps(res) + "\n")
            
    print(f"Saved attention layer metrics to {args.out}")

if __name__ == "__main__":
    main()
