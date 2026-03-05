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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--dataset", type=str, default="commonsense_qa")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    set_all_seeds(123)
    split = "validation"
    if args.dataset in ["cmmlu", "gsm8k"]: split = "test"
    loader = DatasetLoader(args.dataset, split=split)
    items = loader.load(limit=args.limit)
    if not items: return
    prompts = [item["prompt"] for item in items] if isinstance(items[0], dict) else items

    print(f"--- Evaluating {args.model} on {args.dataset} with [{args.dtype}] (Batch: 1 vs {args.batch_size}) ---")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    # Crucial for batched causal generation
    tokenizer.padding_side = "left"  
    
    torch_dtype = getattr(torch, args.dtype)
    
    # We only test eager mode
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()

    # 1. Baseline: Batch Size = 1
    baseline_tokens = []
    print("Running baseline (batch_size=1)...")
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, pad_token_id=tokenizer.pad_token_id, do_sample=False)
            top_idx = torch.argmax(outputs.scores[0], dim=-1)
            baseline_tokens.append(top_idx.item())

    # 2. Batched: Batch Size = args.batch_size
    batched_tokens = []
    print(f"Running batched (batch_size={args.batch_size})...")
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i+args.batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, pad_token_id=tokenizer.pad_token_id, do_sample=False)
            top_idx = torch.argmax(outputs.scores[0], dim=-1)
            batched_tokens.extend(top_idx.tolist())

    # Compare
    flips = 0
    for i, (b1, bN) in enumerate(zip(baseline_tokens, batched_tokens)):
        if b1 != bN:
            flips += 1
            
    print(f"Batch Size Flips (1 vs {args.batch_size}): {flips} / {len(prompts)}")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
