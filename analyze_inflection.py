#!/usr/bin/env python3
import argparse
import random
import numpy as np
import torch
import sys
import gc
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append("/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/")
from run_benchmark import DatasetLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_tokens(model_name, prompts, attn_impl, dtype):
    print(f"Loading {model_name} with {attn_impl}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = getattr(torch, dtype)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True, attn_implementation=attn_impl
    )
    model.eval()

    tokens = []
    print(f"Generating tokens for {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, pad_token_id=tokenizer.pad_token_id, do_sample=False)
            top_idx = torch.argmax(outputs.scores[0], dim=-1)
            tokens.append(top_idx.item())
            
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return tokens, tokenizer

def get_detailed_layer_outputs(model_name, flipped_prompts, attn_impl, dtype):
    print(f"Loading {model_name} with {attn_impl} for inflection tracking...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = getattr(torch, dtype)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True, attn_implementation=attn_impl
    )
    model.eval()

    # Dictionary: {prompt_idx: {layer_num: tensor}}
    layer_outputs = {i: {} for i in range(len(flipped_prompts))}
    hooks = []
    current_prompt_idx = 0
    
    def get_hook(layer_idx):
        def hook_fn(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_outputs[current_prompt_idx][layer_idx] = hidden[:, -1, :].detach().cpu().float()
        return hook_fn

    layers = getattr(model, "model").layers if hasattr(model, "model") else model.layers
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(get_hook(i)))

    for i, prompt in enumerate(flipped_prompts):
        current_prompt_idx = i
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
             model(**inputs)
             
    for h in hooks:
        h.remove()
        
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return layer_outputs

def calculate_tensor_stats(tensor):
    return {
        "norm": torch.norm(tensor).item(),
        "max": torch.max(torch.abs(tensor)).item(),
        "var": torch.var(tensor).item(),
        "mean": torch.mean(tensor).item()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--dataset", type=str, default="commonsense_qa")
    parser.add_argument("--limit", type=int, default=100) # smaller limit to find flips quickly
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    set_all_seeds(123)
    split = "validation"
    if args.dataset in ["cmmlu", "gsm8k"]: split = "test"
    loader = DatasetLoader(args.dataset, split=split)
    items = loader.load(limit=args.limit)
    if not items: return
    prompts = [item["prompt"] for item in items] if isinstance(items[0], dict) else items

    # Find flips
    eager_tokens, tokenizer = generate_tokens(args.model, prompts, "eager", args.dtype)
    sdpa_tokens, _ = generate_tokens(args.model, prompts, "sdpa", args.dtype)

    flipped_indices = [i for i, (e, s) in enumerate(zip(eager_tokens, sdpa_tokens)) if e != s]
    print(f"\nFound {len(flipped_indices)} flipped prompts out of {len(prompts)}.")
    
    if len(flipped_indices) == 0:
        print("No flips found. Cannot analyze turning points.")
        return

    # Just analyze the first flipped prompt to trace the inflection point closely
    target_idx = flipped_indices[0]
    flipped_prompts = [prompts[target_idx]]
    print(f"Analyzing Prompt Index {target_idx}")

    eager_layer_outputs = get_detailed_layer_outputs(args.model, flipped_prompts, "eager", args.dtype)
    sdpa_layer_outputs = get_detailed_layer_outputs(args.model, flipped_prompts, "sdpa", args.dtype)

    print("\n" + "="*80)
    print(f"INFLECTION POINT ANALYSIS: {args.model} [{args.dtype}]")
    print("="*80)
    print(f"{'Layer':<6} | {'MAE':<12} | {'Max_Diff':<12} | {'Eager_Max':<12} | {'SDPA_Max':<12} | {'Eager_Var':<12} | {'SDPA_Var':<12}")
    
    num_layers = len(eager_layer_outputs[0])
    for layer_idx in range(num_layers):
        e_out = eager_layer_outputs[0][layer_idx]
        s_out = sdpa_layer_outputs[0][layer_idx]
        
        mae = torch.mean(torch.abs(e_out - s_out)).item()
        max_err = torch.max(torch.abs(e_out - s_out)).item()
        
        e_stats = calculate_tensor_stats(e_out)
        s_stats = calculate_tensor_stats(s_out)
        
        print(f"{layer_idx:<6} | {mae:<12.5e} | {max_err:<12.5e} | {e_stats['max']:<12.5e} | {s_stats['max']:<12.5e} | {e_stats['var']:<12.5e} | {s_stats['var']:<12.5e}")

if __name__ == "__main__":
    main()
