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

def get_layer_hooks_and_run(model, inputs, max_new_tokens=1):
    """
    Since max_new_tokens=1, a single forward pass with output_hidden_states=True 
    gives us everything we need for greedy decoding and hidden states, without the overhead of generate().
    """
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    logits = outputs.logits[:, -1, :] # [batch_size, vocab_size]
    
    # Store layer outputs
    # outputs.hidden_states is a tuple of (embedding_output, + output of each layer)
    # So outputs.hidden_states[1:] corresponds to the output of each transformer block.
    layer_outputs = {
        i: state.detach().cpu().float() 
        for i, state in enumerate(outputs.hidden_states[1:])
    }
        
    probs = F.softmax(logits, dim=-1)
    top2_probs, top2_indices = torch.topk(probs, 2, dim=-1)
    
    top1_ids = top2_indices[:, 0].tolist()
    margins = (top2_probs[:, 0] - top2_probs[:, 1]).tolist()
    
    return layer_outputs, top1_ids, margins

def calculate_metrics(eager_tensor, sdpa_tensor):
    """
    Calculate MAE, Relative MAE, Max Diff, and Cosine Sim between two tensors.
    Tensors expected to be floating point matrices of identical shape.
    """
    diff = eager_tensor - sdpa_tensor
    abs_diff = torch.abs(diff)
    
    mae = torch.mean(abs_diff).item()
    eager_mean_abs = torch.mean(torch.abs(eager_tensor)).item()
    rel_mae = mae / eager_mean_abs if eager_mean_abs > 0 else 0.0
    
    max_diff = torch.max(abs_diff).item()
    
    # Cosine Similarity along the feature dimension (last dim)
    # Average across batch and sequence
    num = torch.sum(eager_tensor * sdpa_tensor, dim=-1)
    den = torch.norm(eager_tensor, dim=-1) * torch.norm(sdpa_tensor, dim=-1)
    cos_sim = torch.mean(num / torch.clamp(den, min=1e-8)).item()
    
    return {
        "mae": mae,
        "rel_mae": rel_mae,
        "max_diff": max_diff,
        "cos_sim": cos_sim
    }

def process_all_prompts(model_name, dtype, all_prompts, batch_size=1, prefill_only=False):
    """
    Loads model in eager, processes all prompts, unloads. Loads sdpa, processes all, unloads.
    Then compares the results.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Crucial for generation
        
    torch_dtype = getattr(torch, dtype)
    max_new_toks = 0 if prefill_only else 1

    # Storage for results
    # We will just store the extracted top1, margins, and the layer hidden states on CPU to save GPU memory
    eager_results = [] # list of dicts: {'top1', 'margin', 'layers': {l: tensor}}
    sdpa_results = []
    
    # helper
    def gather_backend(attn_impl):
        print(f"\n---> Loading {model_name} with {attn_impl}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True, attn_implementation=attn_impl
        )
        model.eval()
        
        backend_res = []
        
        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i+batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
            model_inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            if prefill_only:
                with torch.no_grad():
                    out = model(**model_inputs, output_hidden_states=True)
                
                layers_dict = {
                    idx: state.detach().cpu().float() 
                    for idx, state in enumerate(out.hidden_states[1:])
                }
                top1 = torch.argmax(out.logits[:, -1, :], dim=-1).tolist()
                margins = [0]*len(batch_prompts)
            else:
                layers_dict, top1, margins = get_layer_hooks_and_run(model, model_inputs, max_new_tokens=1)
                
            # Unbatch results to store per-prompt
            for b_idx in range(len(batch_prompts)):
                prompt_layers = {l: layers_dict[l][b_idx:b_idx+1] for l in layers_dict.keys()}
                backend_res.append({
                    "top1": top1[b_idx],
                    "margin": margins[b_idx],
                    "layers": prompt_layers
                })
                
            print(f"Processed {len(backend_res)}/{len(all_prompts)} prompts...")
                
        # Only delete memory AFTER processing all batches
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return backend_res

    # 1. Run Eager
    eager_results = gather_backend("eager")
    
    # 2. Run SDPA
    sdpa_results = gather_backend("sdpa")
    
    # 3. Compare
    print("Computing metrics...")
    final_results = []
    num_layers = len(eager_results[0]["layers"])
    
    for i in range(len(all_prompts)):
        e_res = eager_results[i]
        s_res = sdpa_results[i]
        
        flip = (e_res["top1"] != s_res["top1"])
        
        layer_metrics = []
        max_mae_post_0 = 0
        
        for l in range(num_layers):
            e_tensor = e_res["layers"][l]
            s_tensor = s_res["layers"][l]
            metrics = calculate_metrics(e_tensor, s_tensor)
            layer_metrics.append({
                "layer": l,
                **metrics
            })
            if metrics["mae"] > max_mae_post_0:
                max_mae_post_0 = metrics["mae"]

        final_results.append({
            "prompt_idx": i,
            "flip": flip,
            "eager_margin": e_res["margin"],
            "sdpa_margin": s_res["margin"],
            "eager_top1": e_res["top1"],
            "sdpa_top1": s_res["top1"],
            "max_mae": max_mae_post_0,
            "layer_metrics": layer_metrics
        })
        
    return final_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--dataset", type=str, default="commonsense_qa")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=1) # highly bound to shape, 1 recommended
    parser.add_argument("--prefill_only", action="store_true", help="Only do a forward pass with no decode generation")
    parser.add_argument("--out", type=str, default="layer_mae_results.jsonl")
    args = parser.parse_args()

    set_all_seeds(123)
    
    split = "validation"
    if args.dataset in ["cmmlu", "gsm8k"]: split = "test"
    loader = DatasetLoader(args.dataset, split=split)
    items = loader.load(limit=args.limit)
    if not items: return
    prompts = [item["prompt"] for item in items] if isinstance(items[0], dict) else items

    print(f"--- Core Layer MAE Experiment ---")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} (Limit: {args.limit})")
    print(f"Dtype: {args.dtype}")
    print(f"Prefill Only: {args.prefill_only}")
    print(f"Output: {args.out}")
    print(f"---------------------------------")
    
    with open(args.out, 'w') as f:
        pass # truncate
        
    print(f"Starting batched evaluation of {len(prompts)} prompts...")
    final_res = process_all_prompts(args.model, args.dtype, prompts, batch_size=args.batch_size, prefill_only=args.prefill_only)
    
    with open(args.out, 'a') as f:
        for res in final_res:
            f.write(json.dumps(res) + "\n")
                
    print(f"Finished processing. Results saved to {args.out}")

if __name__ == "__main__":
    main()
