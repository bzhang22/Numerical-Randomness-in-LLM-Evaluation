import argparse
import random
import numpy as np
import torch
import torch.nn as nn
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

def get_module_hooks_and_run(model, inputs):
    """
    Run forward pass and capture fine-grained states within one target layer.
    """
    # Track: input_layernorm, q_proj, k_proj, v_proj, o_proj, mlp, and residual
    # Since Qwen structure: model.layers[i].self_attn.o_proj, etc.
    # We will hook these specific submodules for all layers or just target layer.
    
    module_outputs = {}
    hooks = []
    
    # We want to trace specific components
    components_to_hook = {
        "input_layernorm": lambda layer: layer.input_layernorm,
        "q_proj": lambda layer: layer.self_attn.q_proj,
        "k_proj": lambda layer: layer.self_attn.k_proj,
        "v_proj": lambda layer: layer.self_attn.v_proj,
        "attn_out": lambda layer: layer.self_attn.o_proj, # Before residual add? No this is o_proj out
        "post_attention_layernorm": lambda layer: layer.post_attention_layernorm,
        "mlp_out": lambda layer: layer.mlp,
        "layer_out": lambda layer: layer
    }
    
    def get_hook(layer_idx, comp_name):
        def hook_fn(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            
            if layer_idx not in module_outputs:
                module_outputs[layer_idx] = {}
                
            module_outputs[layer_idx][comp_name] = hidden.detach().cpu().float()
        return hook_fn

    layers = getattr(model, "model").layers if hasattr(model, "model") else model.layers
    
    for i, layer in enumerate(layers):
        for comp_name, get_comp_fn in components_to_hook.items():
            try:
                comp = get_comp_fn(layer)
                hooks.append(comp.register_forward_hook(get_hook(i, comp_name)))
            except AttributeError:
                pass # skip if component not found (architecture mismatch)
                
    with torch.no_grad():
        outputs = model(**inputs)
        
    for h in hooks:
        h.remove()
        
    return module_outputs

def calculate_metrics(eager_tensor, sdpa_tensor):
    diff = eager_tensor - sdpa_tensor
    abs_diff = torch.abs(diff)
    mae = torch.mean(abs_diff).item()
    max_diff = torch.max(abs_diff).item()
    return {"mae": mae, "max_diff": max_diff}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--limit", type=int, default=10) # Small sample
    parser.add_argument("--out", type=str, default="attention_breakdown.jsonl")
    args = parser.parse_args()

    set_all_seeds(123)
    
    # Hardcoded few prompts for deep tracing
    prompts = [
        "Question: Identify the bird. A. Sparrow B. Eagle C. Penguin D. Ostrich Answer:",
        "Question: Identify the car. A. Ford B. Toyota C. Honda D. BMW Answer:",
        "Question: Identify the fruit. A. Apple B. Banana C. Orange D. Grape Answer:"
    ]
    if args.limit < len(prompts):
        prompts = prompts[:args.limit]

    print(f"--- Attention Breakdown Experiment ---")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    torch_dtype = getattr(torch, args.dtype)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    # 1. Run Eager
    print("Running Eager...")
    model_eager = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True, attn_implementation="eager"
    )
    model_eager.eval()
    eager_inputs = {k: v.to(model_eager.device) for k, v in inputs.items()}
    eager_modules = get_module_hooks_and_run(model_eager, eager_inputs)
    del model_eager
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Run SDPA
    print("Running SDPA...")
    model_sdpa = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True, attn_implementation="sdpa"
    )
    model_sdpa.eval()
    sdpa_inputs = {k: v.to(model_sdpa.device) for k, v in inputs.items()}
    sdpa_modules = get_module_hooks_and_run(model_sdpa, sdpa_inputs)
    del model_sdpa
    torch.cuda.empty_cache()
    gc.collect()
    
    # Compare
    with open(args.out, 'w') as f:
        for p_idx in range(len(prompts)):
            res = {"prompt_idx": p_idx, "model": args.model, "layers": {}}
            for l_idx in eager_modules.keys():
                layer_res = {}
                for comp in eager_modules[l_idx].keys():
                    if comp in sdpa_modules[l_idx]:
                        e_tensor = eager_modules[l_idx][comp][p_idx:p_idx+1]
                        s_tensor = sdpa_modules[l_idx][comp][p_idx:p_idx+1]
                        layer_res[comp] = calculate_metrics(e_tensor, s_tensor)
                res["layers"][l_idx] = layer_res
            f.write(json.dumps(res) + "\n")
            
    print(f"Saved module breakdown to {args.out}")

if __name__ == "__main__":
    main()
