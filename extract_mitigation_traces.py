import argparse
import json
import torch
import numpy as np
import os
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = "/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/results_mitigation"
TRACE_DIR = "/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/traces_mitigation"
os.makedirs(TRACE_DIR, exist_ok=True)

from run_benchmark import DatasetLoader

def get_layer_hooks_and_run(model, inputs):
    layer_outputs = {}
    hooks = []
    
    def get_hook(layer_idx):
        def hook_fn(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Just grab the LAST sequence position (the target prediction token)
            layer_outputs[layer_idx] = hidden[:, -1, :].detach().cpu().float()
            return output
        return hook_fn

    layers = getattr(model, "model").layers if hasattr(model, "model") else model.layers
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(get_hook(i)))
        
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1, 
            return_dict_in_generate=True, 
            output_scores=True, 
            do_sample=False,
            temperature=0.0
        )
        
    for h in hooks:
        h.remove()
        
    scores = outputs.scores[0][0]
    top_probs, _ = torch.topk(torch.softmax(scores, dim=-1), 2)
    margin = (top_probs[0] - top_probs[1]).item()
        
    return layer_outputs, margin

def load_jsonl(filepath):
    if not os.path.exists(filepath):
        return None
    data = {}
    with open(filepath, "r") as f:
        for line in f:
            j = json.loads(line)
            data[j["id"]] = j
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base_dtype", default="bfloat16")
    args = parser.parse_args()
    
    model_name_short = args.model.split("/")[-1]
    
    base_variant = "bf16_baseline" if args.base_dtype == "bfloat16" else "fp16_baseline"
    
    fp32_path = os.path.join(RESULTS_DIR, f"{model_name_short}_{args.dataset}_fp32_reference.jsonl")
    base_path = os.path.join(RESULTS_DIR, f"{model_name_short}_{args.dataset}_{base_variant}.jsonl")
    
    fp32_res = load_jsonl(fp32_path)
    base_res = load_jsonl(base_path)
    
    if not fp32_res or not base_res:
        print(f"Skipping {args.model} {args.dataset} - base logs missing.")
        return
        
    # Find Divergent and Stable cases
    divergent_ids = []
    stable_ids = []
    
    for prompt_id in fp32_res:
        if prompt_id not in base_res: continue
        if fp32_res[prompt_id]["generated_tokens"] != base_res[prompt_id]["generated_tokens"]:
            divergent_ids.append(prompt_id)
        else:
            stable_ids.append(prompt_id)
            
    # Sample up to 20 divergent, 10 stable
    selected_div = divergent_ids[:20]
    selected_stable = stable_ids[:10]
    target_ids = selected_div + selected_stable
    
    if not target_ids:
        print("No paired valid IDs found.")
        return
        
    print(f"Targeting {len(selected_div)} divergent, {len(selected_stable)} stable for {model_name_short}")
    
    # Needs loader
    split = "validation"
    if args.dataset in ["cmmlu", "gsm8k"]: split = "test"
    if args.dataset == "piqa": split = "validation"
    
    loader = DatasetLoader(args.dataset, split=split)
    items = loader.load(limit=2000)
    items_dict = {item["id"]: item for item in items}
    
    # Load FP32 model
    print("Loading FP32 Reference...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    
    model_fp32 = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="auto")
    model_fp32.eval()
    
    fp32_traces = {}
    for tid in target_ids:
        inputs = tokenizer(items_dict[tid]["prompt"], return_tensors="pt").to(model_fp32.device)
        layers, margin = get_layer_hooks_and_run(model_fp32, inputs)
        fp32_traces[tid] = {"layers": layers, "margin": margin}
        
    del model_fp32
    torch.cuda.empty_cache()
    gc.collect()
    
    # We will test execution using run_mitigation.py's exact patcher
    import sys
    sys.path.append("/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation")
    from run_mitigation import apply_fp32_mitigation
    
    variants = [base_variant, "attention", "norm", "lm_head", "attention_lm_head"]
    
    trace_results = {tid: {"fp32_margin": fp32_traces[tid]["margin"], "is_divergent": tid in selected_div, "variants": {}} for tid in target_ids}
    
    dt = getattr(torch, args.base_dtype)
    for variant in variants:
        print(f"Loading {args.base_dtype} for variant: {variant}...")
        model_base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt, device_map="auto")
        model_base.eval()
        
        apply_fp32_mitigation(model_base, variant, dt)
        
        for tid in target_ids:
            inputs = tokenizer(items_dict[tid]["prompt"], return_tensors="pt").to(model_base.device)
            layers, margin = get_layer_hooks_and_run(model_base, inputs)
            
            # Compute MAE vs FP32
            maes = []
            for l_idx in layers:
                if l_idx in fp32_traces[tid]["layers"]:
                    diff = torch.abs(layers[l_idx] - fp32_traces[tid]["layers"][l_idx])
                    maes.append(diff.mean().item())
            
            trace_results[tid]["variants"][variant] = {
                "margin": margin,
                "layer_maes": maes
            }
            
        del model_base
        torch.cuda.empty_cache()
        gc.collect()
        
    # Save results
    out_path = os.path.join(TRACE_DIR, f"{model_name_short}_{args.dataset}_{args.base_dtype}_traces.json")
    with open(out_path, "w") as f:
        json.dump(trace_results, f, indent=2)
        
    print(f"Successfully dumped {out_path}")

if __name__ == "__main__":
    main()
