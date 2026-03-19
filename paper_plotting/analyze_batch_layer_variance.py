import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import gc
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append("/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/")
from run_benchmark import DatasetLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_metrics(baseline_tensor, batched_tensor):
    """
    Calculate MAE, Relative MAE, and Max Diff between baseline and batched hidden states.
    Tensors expected to be 1D vectors of shape [hidden_dim].
    """
    diff = baseline_tensor - batched_tensor
    abs_diff = torch.abs(diff)
    
    mae = torch.mean(abs_diff).item()
    baseline_mean_abs = torch.mean(torch.abs(baseline_tensor)).item()
    rel_mae = mae / baseline_mean_abs if baseline_mean_abs > 0 else 0.0
    
    max_diff = torch.max(abs_diff).item()
    
    return {
        "mae": mae,
        "rel_mae": rel_mae,
        "max_diff": max_diff
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="NousResearch/Meta-Llama-3-8B")
    parser.add_argument("--dataset", type=str, default="commonsense_qa")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out", type=str, default="batch_layer_variance.jsonl")
    args = parser.parse_args()

    set_all_seeds(123)
    split = "validation"
    if args.dataset in ["cmmlu", "gsm8k"]: split = "test"
    loader = DatasetLoader(args.dataset, split=split)
    items = loader.load(limit=args.limit)
    if not items: return
    prompts = [item["prompt"] for item in items] if isinstance(items[0], dict) else items

    print(f"--- Layer Variance: {args.model} on {args.dataset} with [{args.dtype}] (Batch: 1 vs {args.batch_size}) ---")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    # Crucial for left-padding batched inference
    tokenizer.padding_side = "left"  
    
    torch_dtype = getattr(torch, args.dtype)
    
    # We test on eager mode
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()

    # 1. Baseline: Batch Size = 1
    # Store layer states per prompt. Shape: list of dicts {layer_idx: tensor[1, hidden_dim]}
    baseline_states_list = []
    baseline_top1 = []
    print("Running baseline (batch_size=1)...")
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        top1 = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
        baseline_top1.append(top1)
        
        # Extract the hidden state of the LAST token for each layer
        layer_dict = {}
        for l_idx, state in enumerate(outputs.hidden_states[1:]):
            layer_dict[l_idx] = state[:, -1, :].detach().cpu().float()
        baseline_states_list.append(layer_dict)

    # 2. Batched: Batch Size = args.batch_size
    batched_states_list = []
    batched_top1 = []
    print(f"Running batched (batch_size={args.batch_size})...")
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i+args.batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        top1_batch = torch.argmax(outputs.logits[:, -1, :], dim=-1).tolist()
        batched_top1.extend(top1_batch)
        
        # outputs.hidden_states[1:] -> each is (batch, seq_len, hidden_dim)
        # We need to unbatch them and just take the last token
        num_layers = len(outputs.hidden_states) - 1
        for b_idx in range(len(batch_prompts)):
            layer_dict = {}
            for l_idx in range(num_layers):
                state = outputs.hidden_states[l_idx + 1]
                # Extract batch b_idx, last token -1
                layer_dict[l_idx] = state[b_idx:b_idx+1, -1, :].detach().cpu().float()
            batched_states_list.append(layer_dict)

    # 3. Compare and output JSONL
    print("Computing per-layer metrics...")
    final_results = []
    flips = 0
    num_layers = len(baseline_states_list[0])
    
    for i in range(len(prompts)):
        b1_top1 = baseline_top1[i]
        bN_top1 = batched_top1[i]
        flip = (b1_top1 != bN_top1)
        if flip: flips += 1
        
        b1_layers = baseline_states_list[i]
        bN_layers = batched_states_list[i]
        
        layer_metrics = []
        for l in range(num_layers):
            metrics = calculate_metrics(b1_layers[l], bN_layers[l])
            layer_metrics.append({
                "layer": l,
                **metrics
            })
            
        final_results.append({
            "prompt_idx": i,
            "model": args.model.split("/")[-1],
            "dataset": args.dataset,
            "dtype": args.dtype,
            "flip": flip,
            "b1_top1": b1_top1,
            "bN_top1": bN_top1,
            "layer_metrics": layer_metrics
        })
        
    print(f"Batch Size Flips (1 vs {args.batch_size}): {flips} / {len(prompts)}")
    
    with open(args.out, 'a') as f:
        for res in final_results:
            f.write(json.dumps(res) + "\n")
            
    print(f"Saved layer metrics to {args.out}")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
