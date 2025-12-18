#!/usr/bin/env python3
"""
numerical_probe_vllm.py

Numerical randomness probe for LLM inference using vLLM only.
Compares vLLM Run 0 (reference) vs vLLM Run 1..N (probe runs).
"""

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import lm_eval.tasks

# Set environment variables BEFORE importing vLLM
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM not installed. Please install vllm.")
    exit(1)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_prompts(prompts_file: Optional[str], num_prompts: int, dataset_name: Optional[str] = None) -> List[str]:
    if dataset_name == "piqa":
        print("Loading prompts from PIQA via lm_eval...")
        task_dict = lm_eval.tasks.get_task_dict(["piqa"])
        t = task_dict["piqa"]
        if t.has_validation_docs():
            docs = list(t.validation_docs())
        elif t.has_test_docs():
            docs = list(t.test_docs())
        else:
             docs = list(t.training_docs())
        if not docs:
            raise ValueError("PIQA dataset seems empty via lm_eval?")
        goals = [d["goal"] for d in docs]
        out = (goals * ((num_prompts + len(goals) - 1) // len(goals)))[:num_prompts]
        return out

    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            raise ValueError(f"No prompts found in {prompts_file}")
        out = (lines * ((num_prompts + len(lines) - 1) // len(lines)))[:num_prompts]
        return out

    base = ["The capital of France is", "What is 2+2?"]
    out = []
    for i in range(num_prompts):
        out.append(f"[{i:04d}] {base[i % len(base)]}")
    return out

@dataclass
class PromptStats:
    ref_top1_id: int
    ref_top1_logprob: float
    max_logprob_abs_diff: float = 0.0
    min_ref_token_logprob: float = float("inf")
    max_ref_token_logprob: float = float("-inf")
    top1_flip_count: int = 0
    runs_seen: int = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--dataset", type=str, choices=["piqa"], default=None)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--num_prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_csv", type=str, default="vllm_probe_results.csv")
    
    args = ap.parse_args()
    set_all_seeds(args.seed)

    prompts = load_prompts(None, args.num_prompts, dataset_name=args.dataset)
    print(f"Prompts: {len(prompts)} | Model: {args.model} | Runs: {args.runs}")

    stats: List[Optional[PromptStats]] = [None] * len(prompts)

    print("\nInitializing vLLM...")
    print("(This may take 2-3 minutes on first run - compiling kernels)")
    
   
    llm = LLM(
        model=args.model, 
        dtype="float16", 
        seed=args.seed, 
        trust_remote_code=True,
        gpu_memory_utilization=0.5,  # Reduced from 0.9 to avoid OOM
        max_model_len=512,  # Reduced from 2048
        max_num_seqs=32,  # Limit concurrent sequences
        disable_log_stats=True,
        enforce_eager=True,  # Disable CUDA graphs to avoid hangs
    )
    
    print("vLLM initialized successfully!")
    
    # Sampling params
    sp = SamplingParams(
        temperature=0.0, 
        max_tokens=1, 
        logprobs=20,  # Max allowed by vLLM
        prompt_logprobs=0
    )

    # Store reference logprobs from Run 0
    ref_logprobs_dict = []  # List of dicts: [{token_id: logprob, ...}, ...]
    
    # Run 0: Reference
    print("\n[Run 0] vLLM Reference Run...")
    outputs = llm.generate(prompts, sp, use_tqdm=True)
    
    for i, out in enumerate(outputs):
        vllm_top1 = out.outputs[0].token_ids[0]
        first_token_logprobs = out.outputs[0].logprobs[0]  # Dict[int, float]
        
        # Store reference
        ref_logprobs_dict.append(first_token_logprobs)
        
        # Initialize stats
        top1_logprob = first_token_logprobs[vllm_top1].logprob  # Extract float from Logprob object
        stats[i] = PromptStats(
            ref_top1_id=vllm_top1,
            ref_top1_logprob=top1_logprob,
            min_ref_token_logprob=top1_logprob,
            max_ref_token_logprob=top1_logprob,
            runs_seen=1
        )
    
    # Runs 1..N: Probe
    for r in range(1, args.runs):
        print(f"\n[Run {r}] vLLM Probe Run...")
        outputs = llm.generate(prompts, sp, use_tqdm=True)
        
        for i, out in enumerate(outputs):
            st = stats[i]
            if st is None: 
                continue
            
            vllm_top1 = out.outputs[0].token_ids[0]
            current_logprobs = out.outputs[0].logprobs[0]
            
            # Check flip
            if vllm_top1 != st.ref_top1_id:
                st.top1_flip_count += 1
            st.runs_seen += 1
            
            # Update ref token logprob range
            ref_id = st.ref_top1_id
            if ref_id in current_logprobs:
                val = current_logprobs[ref_id].logprob  # Extract float
                st.min_ref_token_logprob = min(st.min_ref_token_logprob, val)
                st.max_ref_token_logprob = max(st.max_ref_token_logprob, val)
            
            # Compute max abs diff over intersection of tokens
            local_max_diff = 0.0
            ref_logprobs = ref_logprobs_dict[i]
            
            # Check all tokens that appear in both current and reference
            all_tokens = set(current_logprobs.keys()) | set(ref_logprobs.keys())
            for tid in all_tokens:
                curr_obj = current_logprobs.get(tid, None)
                ref_obj = ref_logprobs.get(tid, None)
                
                # Only compare if both are present
                if curr_obj is not None and ref_obj is not None:
                    curr_val = curr_obj.logprob
                    ref_val = ref_obj.logprob
                    diff = abs(curr_val - ref_val)
                    if diff > local_max_diff:
                        local_max_diff = diff
            
            st.max_logprob_abs_diff = max(st.max_logprob_abs_diff, local_max_diff)

    # Save Results
    print(f"\nSaving results to {args.out_csv}...")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "prompt_id", "prompt", "ref_top1_token_id", "ref_top1_logprob",
            "logprobs_max_abs_diff",
            "ref_token_logprob_range",
            "top1_flip_count", "runs_seen"
        ])
        for i, (p, st) in enumerate(zip(prompts, stats)):
             if st:
                w.writerow([
                    i, p, st.ref_top1_id, f"{st.ref_top1_logprob:.8f}",
                    f"{st.max_logprob_abs_diff:.8e}",
                    f"{(st.max_ref_token_logprob - st.min_ref_token_logprob):.8e}",
                    st.top1_flip_count, st.runs_seen
                ])
    
    # Print summary
    print("\n" + "="*80)
    print("VLLM NUMERICAL PROBE SUMMARY")
    print("="*80)
    
    max_diffs = [st.max_logprob_abs_diff for st in stats if st]
    flip_counts = [st.top1_flip_count for st in stats if st]
    
    print(f"Prompts: {len(prompts)}")
    print(f"Runs: {args.runs}")
    print(f"\nLogProb Max Abs Diff:")
    print(f"  Min: {min(max_diffs):.6e}")
    print(f"  Max: {max(max_diffs):.6e}")
    print(f"  Mean: {sum(max_diffs)/len(max_diffs):.6e}")
    
    num_flips = sum(1 for f in flip_counts if f > 0)
    print(f"\nTop-1 Token Flips:")
    print(f"  Prompts with flips: {num_flips}/{len(prompts)} ({100*num_flips/len(prompts):.1f}%)")
    
    print(f"\nResults saved to: {args.out_csv}")

if __name__ == "__main__":
    main()
