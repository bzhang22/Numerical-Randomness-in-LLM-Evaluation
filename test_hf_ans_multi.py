#!/usr/bin/env python3
import argparse
import random
import numpy as np
import torch
import sys
import gc
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the directory containing the local module to sys.path so we can import DatasetLoader
sys.path.append("/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/")
from run_benchmark import DatasetLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_generation_results(model_name, prompts, attn_impl, dtype="float16"):
    print(f"\n" + "="*50)
    print(f"Phase: Hugging Face with '{attn_impl}' Attention (dtype: {dtype})")
    print("="*50)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype, 
        device_map="cuda", 
        trust_remote_code=True,
        attn_implementation=attn_impl
    )
    model.eval()

    top1_tokens = []
    top1_logprobs = []
    
    print(f"Running HF ({attn_impl}) generation (max_new_tokens=1) for {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts):
        if i > 0 and i % 100 == 0:
            print(f"  Processed {i}/{len(prompts)} prompts...")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1, 
                return_dict_in_generate=True, 
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
            
            # The last token generation logit logic
            logits = outputs.scores[0] # Shape: [batch_size, vocab_size]
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            top_lp, top_idx = torch.max(logprobs, dim=-1)
            
            top1_tokens.append(top_idx.item())
            top1_logprobs.append(top_lp.item())
            
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {"tokens": top1_tokens, "logprobs": top1_logprobs}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset", type=str, required=True, choices=["piqa", "commonsense_qa", "cmmlu", "gsm8k"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()

    set_all_seeds(123)

    # Determine split semantics mapping the run_benchmark rules
    split = "validation"
    if args.dataset in ["cmmlu", "gsm8k"]: split = "test"
    if args.dataset == "piqa": split = "validation"
    
    loader = DatasetLoader(args.dataset, split=split)
    items = loader.load(limit=args.limit)

    if not items:
        print(f"Failed to load dataset {args.dataset}. Exiting.")
        return

    # Extract raw string prompts
    if isinstance(items[0], dict):
        prompts = [item["prompt"] for item in items]
    else:
        # e.g., GSM8K texts return list of strings directly
        prompts = items

    print(f"Loaded {len(prompts)} prompts for '{args.dataset}'. Model: {args.model}")

    # ==========================
    # Phase 1: Eager (Exact Math)
    # ==========================
    eager_results = get_generation_results(args.model, prompts, "eager", args.dtype)
    if eager_results is None:
        return

    # ==========================
    # Phase 2: SDPA (Flash Attention backend)
    # ==========================
    sdpa_results = get_generation_results(args.model, prompts, "sdpa", args.dtype)
    if sdpa_results is None:
        return

    # ==========================
    # Compare
    # ==========================
    print("\n" + "="*50)
    print(f"DATASET: {args.dataset.upper()}")
    print("COMPARISON RESULTS (Final Answer: HF Eager vs HF SDPA)")
    print("="*50)
    
    eager_tokens = eager_results["tokens"]
    sdpa_tokens = sdpa_results["tokens"]
    
    eager_logprobs = np.array(eager_results["logprobs"])
    sdpa_logprobs = np.array(sdpa_results["logprobs"])

    exact_matches = sum(1 for e, s in zip(eager_tokens, sdpa_tokens) if e == s)
    flips = len(prompts) - exact_matches
    
    print(f"Total Prompts: {len(prompts)}")
    print(f"Identical Final Tokens Generated: {exact_matches}/{len(prompts)} ({(exact_matches/len(prompts))*100:.2f}%)")
    print(f"Token Flips (Different Answers): {flips}/{len(prompts)} ({(flips/len(prompts))*100:.2f}%)")
    
    logprob_diff = np.abs(eager_logprobs - sdpa_logprobs)
    print(f"\nTop-1 Logprob Absolute Difference:")
    print(f"  Max Diff:  {logprob_diff.max():.8e}")
    print(f"  Mean Diff: {logprob_diff.mean():.8e}")
    
    print("="*50)

if __name__ == "__main__":
    main()
