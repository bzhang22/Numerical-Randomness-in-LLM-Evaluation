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

def trace_outlier_token(model_name, flipped_prompts, attn_impl, dtype, target_layer=26):
    print(f"Tracing context for {model_name} at layer {target_layer}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = getattr(torch, dtype)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True, attn_implementation=attn_impl
    )
    model.eval()

    hooks = []
    layer_tensors = {}
    current_prompt_idx = 0
    
    def get_hook(layer_idx):
        def hook_fn(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden shape: (batch, seq_len, hidden_dim)
            layer_tensors[current_prompt_idx] = hidden[0].detach().cpu().float() 
        return hook_fn

    # Register only on the target layer to save memory and trace the full sequence
    layers = getattr(model, "model").layers if hasattr(model, "model") else model.layers
    if hasattr(model, 'model') and target_layer < len(model.model.layers):
        hooks.append(model.model.layers[target_layer].register_forward_hook(get_hook(target_layer)))
    else:
        hooks.append(model.layers[target_layer].register_forward_hook(get_hook(target_layer)))

    results = []

    for i, prompt in enumerate(flipped_prompts):
        current_prompt_idx = i
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0].tolist() 
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        with torch.no_grad():
             model(**inputs.to(model.device))
             
        hidden_states = layer_tensors[i] # Shape: (seq_len, hidden_dim)
        
        # 1. Find the sequence index with the massive outlier activation
        max_vals_per_token, _ = torch.max(torch.abs(hidden_states), dim=-1)
        outlier_seq_idx = torch.argmax(max_vals_per_token).item()
        max_val_global = max_vals_per_token[outlier_seq_idx].item()
        
        # 2. Extract surrounding context
        context_window = 10
        start_idx = max(0, outlier_seq_idx - context_window)
        end_idx = min(len(input_tokens), outlier_seq_idx + context_window + 1)
        
        outlier_token = input_tokens[outlier_seq_idx]
        context_tokens = input_tokens[start_idx:end_idx]
        context_string = tokenizer.convert_tokens_to_string(context_tokens)
        
        # 3. Find the specific dimension
        max_dim = torch.argmax(torch.abs(hidden_states[outlier_seq_idx])).item()

        results.append({
            "prompt_idx": i,
            "max_val": max_val_global,
            "outlier_idx": outlier_seq_idx,
            "outlier_token": outlier_token,
            "max_dim": max_dim,
            "context_string": context_string
        })
             
    for h in hooks:
        h.remove()
        
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--dataset", type=str, default="commonsense_qa")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--layer", type=int, default=26)
    args = parser.parse_args()

    set_all_seeds(123)
    split = "validation"
    if args.dataset in ["cmmlu", "gsm8k"]: split = "test"
    loader = DatasetLoader(args.dataset, split=split)
    items = loader.load(limit=args.limit)
    if not items: return
    prompts = [item["prompt"] for item in items] if isinstance(items[0], dict) else items

    # Find flips
    eager_tokens, _ = generate_tokens(args.model, prompts, "eager", args.dtype)
    sdpa_tokens, _ = generate_tokens(args.model, prompts, "sdpa", args.dtype)
    flipped_indices = [i for i, (e, s) in enumerate(zip(eager_tokens, sdpa_tokens)) if e != s]
    
    if len(flipped_indices) == 0:
        print("No flips found.")
        return

    flipped_prompts = [prompts[i] for i in flipped_indices]
    
    print("\n" + "="*80)
    print(f"OUTLIER CONTEXT EXTRACTION: {args.model} -> Layer {args.layer}")
    print("="*80)
    
    eager_results = trace_outlier_token(args.model, flipped_prompts, "eager", args.dtype, args.layer)
    
    for r in eager_results:
        print(f"\n--- Flipped Prompt {r['prompt_idx']} ---")
        print(f"Max Layer {args.layer} Activation: {r['max_val']:.4f} (at dim {r['max_dim']})")
        print(f"Token Triggering Outlier (pos {r['outlier_idx']}): '{r['outlier_token']}'")
        print(f"Context Window:\n{r['context_string']}")

if __name__ == "__main__":
    main()
