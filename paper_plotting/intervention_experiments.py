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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_benchmark import DatasetLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_layer_hooks_and_run(model, inputs, intervention_layer=-1, intervention_tensor=None, noise_scale=None, max_new_tokens=1):
    """
    Run forward pass and capture hidden states, with optional intervention (clamp or noise injection).
    - If `intervention_tensor` is provided: Replace output at `intervention_layer` with `intervention_tensor`.
    - If `noise_scale` is provided: Add noise of that magnitude to `intervention_layer`.
    """
    layer_outputs = {}
    hooks = []
    
    def get_hook(layer_idx):
        def hook_fn(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            
            # Intervention logic
            if layer_idx == intervention_layer:
                if intervention_tensor is not None:
                    # Clamp: replace the hidden state with the provided tensor
                    hidden.data.copy_(intervention_tensor.data)
                elif noise_scale is not None:
                    # Noise injection: add tiny random noise
                    noise = torch.randn_like(hidden) * noise_scale
                    hidden.data.add_(noise)
            
            layer_outputs[layer_idx] = hidden.detach().cpu().float()
            
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            
        return hook_fn

    layers = getattr(model, "model").layers if hasattr(model, "model") else model.layers
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(get_hook(i)))
        
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            return_dict_in_generate=True, 
            output_scores=True, 
            pad_token_id=model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0
        )
        
    for h in hooks:
        h.remove()
        
    scores = outputs.scores[0]
    probs = F.softmax(scores, dim=-1)
    top2_probs, top2_indices = torch.topk(probs, 2, dim=-1)
    
    top1_ids = top2_indices[:, 0].tolist()
    
    return layer_outputs, top1_ids

def process_batch_clamp(model_name, dtype, batch_prompts, intervention_layer):
    """
    Clamp Experiment: Replace SDPA's output at layer k with Eager's output at layer k.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    torch_dtype = getattr(torch, dtype)
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)

    # 1. Run Eager to get baseline targets
    model_eager = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True, attn_implementation="eager"
    )
    model_eager.eval()
    eager_inputs = {k: v.to(model_eager.device) for k, v in inputs.items()}
    eager_layers, eager_top1 = get_layer_hooks_and_run(model_eager, eager_inputs)
    
    # Extract the target tensor to inject
    intervention_tensor = eager_layers[intervention_layer].to(model_eager.device).to(torch_dtype)
    
    del model_eager
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Run SDPA with Clamp Intervention
    model_sdpa = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True, attn_implementation="sdpa"
    )
    model_sdpa.eval()
        
    sdpa_inputs = {k: v.to(model_sdpa.device) for k, v in inputs.items()}
    
    # Run Baseline SDPA
    sdpa_layers_baseline, sdpa_top1_baseline = get_layer_hooks_and_run(model_sdpa, sdpa_inputs)
    
    # Run Clamped SDPA
    sdpa_layers_clamped, sdpa_top1_clamped = get_layer_hooks_and_run(
        model_sdpa, sdpa_inputs, intervention_layer, intervention_tensor=intervention_tensor
    )

    del model_sdpa
    torch.cuda.empty_cache()
    gc.collect()
    
    results = []
    for i in range(len(batch_prompts)):
        # Calculate post-intervention Max MAE
        max_mae_post_k = 0
        for l in range(intervention_layer + 1, len(eager_layers)):
            diff = eager_layers[l][i:i+1] - sdpa_layers_clamped[l][i:i+1]
            mae = torch.mean(torch.abs(diff)).item()
            if mae > max_mae_post_k:
                max_mae_post_k = mae
                
        results.append({
            "prompt_idx": "TBD",
            "layer_k": intervention_layer,
            "eager_top1": eager_top1[i],
            "sdpa_baseline_top1": sdpa_top1_baseline[i],
            "sdpa_clamped_top1": sdpa_top1_clamped[i],
            "flip_cured": (sdpa_top1_baseline[i] != eager_top1[i]) and (sdpa_top1_clamped[i] == eager_top1[i]),
            "max_mae_post_k": max_mae_post_k
        })
    return results

def process_batch_noise(model_name, dtype, batch_prompts, intervention_layer, noise_scale):
    """
    Noise Injection Experiment: Add tiny noise to Eager's output at layer k.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    torch_dtype = getattr(torch, dtype)
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)

    model_eager = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True, attn_implementation="eager"
    )
    model_eager.eval()
    eager_inputs = {k: v.to(model_eager.device) for k, v in inputs.items()}
    
    # Run Baseline Eager
    eager_layers_baseline, eager_top1_baseline = get_layer_hooks_and_run(model_eager, eager_inputs)
    
    # Run Noisy Eager
    eager_layers_noisy, eager_top1_noisy = get_layer_hooks_and_run(
        model_eager, eager_inputs, intervention_layer, noise_scale=noise_scale
    )
    
    del model_eager
    torch.cuda.empty_cache()
    gc.collect()

    results = []
    for i in range(len(batch_prompts)):
        max_mae_post_k = 0
        for l in range(intervention_layer + 1, len(eager_layers_baseline)):
            diff = eager_layers_baseline[l][i:i+1] - eager_layers_noisy[l][i:i+1]
            mae = torch.mean(torch.abs(diff)).item()
            if mae > max_mae_post_k:
                max_mae_post_k = mae

        results.append({
            "prompt_idx": "TBD",
            "layer_k": intervention_layer,
            "noise_scale": noise_scale,
            "eager_baseline_top1": eager_top1_baseline[i],
            "eager_noisy_top1": eager_top1_noisy[i],
            "flip_induced": (eager_top1_baseline[i] != eager_top1_noisy[i]),
            "max_mae_post_k": max_mae_post_k
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--dataset", type=str, default="piqa")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mode", type=str, choices=["clamp", "noise"], required=True)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--noise_scale", type=float, default=1e-4) # used only if mode == "noise"
    parser.add_argument("--out", type=str, default="intervention_results.jsonl")
    args = parser.parse_args()

    set_all_seeds(123)
    
    split = "validation"
    if args.dataset in ["cmmlu", "gsm8k"]: split = "test"
    loader = DatasetLoader(args.dataset, split=split)
    items = loader.load(limit=args.limit)
    if not items: return
    prompts = [item["prompt"] for item in items] if isinstance(items[0], dict) else items

    print(f"--- Intervention ({args.mode}) Experiment ---")
    print(f"Model: {args.model}")
    print(f"Layer K: {args.layer}")
    
    with open(args.out, 'w') as f: pass # truncate
        
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i+args.batch_size]
        print(f"Processing prompts {i} to {i+len(batch_prompts)-1}...")
        
        if args.mode == "clamp":
            batch_res = process_batch_clamp(args.model, args.dtype, batch_prompts, args.layer)
        else:
            batch_res = process_batch_noise(args.model, args.dtype, batch_prompts, args.layer, args.noise_scale)
            
        with open(args.out, 'a') as f:
            for j, res in enumerate(batch_res):
                res["prompt_idx"] = i + j
                res["model"] = args.model
                res["dataset"] = args.dataset
                f.write(json.dumps(res) + "\n")
                
    print(f"Finished processing. Results saved to {args.out}")

if __name__ == "__main__":
    main()
