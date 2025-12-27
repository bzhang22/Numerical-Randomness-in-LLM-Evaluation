#!/usr/bin/env python3
"""
compare_hf_vs_vllm.py - Direct comparison: HF reference vs vLLM outputs
"""

import argparse
import csv
import os
import random
import gc
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import lm_eval.tasks

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_prompts(num_prompts: int, dataset_name: Optional[str] = None) -> List[str]:
    if dataset_name == "piqa":
        print("Loading prompts from PIQA...")
        task_dict = lm_eval.tasks.get_task_dict(["piqa"])
        t = task_dict["piqa"]
        docs = list(t.validation_docs() if t.has_validation_docs() else t.test_docs())
        goals = [d["goal"] for d in docs]
        return (goals * ((num_prompts + len(goals) - 1) // len(goals)))[:num_prompts]
    return [f"Prompt {i}" for i in range(num_prompts)]

@dataclass
class PromptStats:
    ref_top1_id: int
    ref_top1_logprob: float
    max_logprob_abs_diff: float = 0.0
    top1_flip_count: int = 0
    runs_seen: int = 0

@torch.inference_mode()
def forward_last_logprobs_hf(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits.float()
    last_index = attention_mask.long().sum(dim=1) - 1
    bsz, _, vocab = logits.shape
    idx = last_index.view(bsz, 1, 1).expand(bsz, 1, vocab)
    last_logits = logits.gather(dim=1, index=idx).squeeze(1)
    return torch.nn.functional.log_softmax(last_logits, dim=-1)

def chunked(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i : i + bs]

def run_hf_phase(args, prompts):
    # ===== PHASE 1: HuggingFace Reference (GPU) =====
    print("\n[Phase 1] Loading HF model on GPU for reference...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    hf_dtype = getattr(torch, args.dtype)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=hf_dtype,
        device_map="cuda:0",
        trust_remote_code=True
    )
    hf_model.eval()
    
    print("Computing HF reference logprobs...")
    ref_logprobs_list = []
    for batch in chunked(prompts, 4):
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda:0")
        lp = forward_last_logprobs_hf(hf_model, enc["input_ids"], enc["attention_mask"])
        ref_logprobs_list.append(lp.cpu())
    
    ref_logprobs = torch.cat(ref_logprobs_list, dim=0)  # [N, V]
    
    print("Clearing HF model from GPU...")
    del hf_model
    del tok
    torch.cuda.empty_cache()
    gc.collect()
    return ref_logprobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="piqa")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--num_prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_csv", type=str, default="hf_vs_vllm_results.csv")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--tf32", action="store_true", help="Enable TF32 (TensorFloat-32)")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM GPU memory utilization")
    ap.add_argument("--phase", type=str, default="all", choices=["all", "hf", "vllm"], help="Execution phase")
    args = ap.parse_args()
    
    set_all_seeds(args.seed)
    
    # Set TF32
    if args.tf32:
        print("Enabling TF32...")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
    prompts = load_prompts(args.num_prompts, args.dataset)
    print(f"HF vs vLLM Comparison: {len(prompts)} prompts, {args.runs} vLLM runs, dtype={args.dtype}, tf32={args.tf32}")
    
    stats: List[Optional[PromptStats]] = [None] * len(prompts)
    
    ref_logprobs_file = "ref_logprobs.pt"
    ref_logprobs = None
    
    # ===== PHASE 1 =====
    if args.phase in ["all", "hf"]:
        ref_logprobs = run_hf_phase(args, prompts)
        torch.save(ref_logprobs, ref_logprobs_file)
        print(f"Saved reference logprobs to {ref_logprobs_file}")
        
    if args.phase == "hf":
        return

    # ===== PHASE 2 =====
    if args.phase == "vllm":
        if os.path.exists(ref_logprobs_file):
            print(f"Loading reference logprobs from {ref_logprobs_file}")
            ref_logprobs = torch.load(ref_logprobs_file)
        else:
            raise FileNotFoundError(f"Could not find {ref_logprobs_file}. Run with --phase hf first.")
            
    # Initialize stats with HF reference
    for i in range(len(prompts)):
        top1_id = int(torch.argmax(ref_logprobs[i]).item())
        top1_lp = float(ref_logprobs[i, top1_id].item())
        stats[i] = PromptStats(ref_top1_id=top1_id, ref_top1_logprob=top1_lp, runs_seen=1)
        
    import time
    time.sleep(2)
    
    # ===== PHASE 2: vLLM Runs (GPU) =====
    print("\n[Phase 2] Loading vLLM on GPU...")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        seed=args.seed,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=512,
        max_num_seqs=32,
        disable_log_stats=True,
        enforce_eager=True,
        max_logprobs=100,  # Explicitly allow up to 100 logprobs
    )
    
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=100, prompt_logprobs=0)
    
    # Run vLLM multiple times and compare against HF reference
    for r in range(args.runs):
        print(f"\n[vLLM Run {r+1}/{args.runs}]...")
        outputs = llm.generate(prompts, sp, use_tqdm=True)
        
        for i, out in enumerate(outputs):
            st = stats[i]
            vllm_top1 = out.outputs[0].token_ids[0]
            vllm_logprobs = out.outputs[0].logprobs[0]
            
            # Check if vLLM prediction differs from HF
            if vllm_top1 != st.ref_top1_id:
                st.top1_flip_count += 1
            
            # Compute max logprob difference
            # Compare vLLM's top logprobs against HF reference
            for tid, lp_obj in vllm_logprobs.items():
                vllm_val = lp_obj.logprob
                hf_val = float(ref_logprobs[i, tid].item())
                diff = abs(vllm_val - hf_val)
                st.max_logprob_abs_diff = max(st.max_logprob_abs_diff, diff)
            
            st.runs_seen += 1
    
    # Save results
    print(f"\nSaving results to {args.out_csv}...")
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "prompt_id", "prompt", 
            "hf_top1_token_id", "hf_top1_logprob",
            "max_logprob_diff_hf_vs_vllm",
            "vllm_flip_count", "vllm_runs"
        ])
        for i, (p, st) in enumerate(zip(prompts, stats)):
            if st:
                w.writerow([
                    i, p, st.ref_top1_id, f"{st.ref_top1_logprob:.8f}",
                    f"{st.max_logprob_abs_diff:.8e}",
                    st.top1_flip_count, st.runs_seen - 1  # -1 because run 0 was HF
                ])
    
    # Print summary
    max_diffs = [st.max_logprob_abs_diff for st in stats if st]
    flips = sum(1 for st in stats if st and st.top1_flip_count > 0)
    
    print("\n" + "="*70)
    print("HF vs vLLM COMPARISON SUMMARY")
    print("="*70)
    print(f"Reference: HuggingFace (GPU, {args.dtype})")
    print(f"Comparison: vLLM (GPU, {args.dtype}) x {args.runs} runs")
    print(f"TF32 Enabled: {args.tf32}")
    print(f"\nLogProb Max Abs Diff (HF vs vLLM):")
    print(f"  Min:  {min(max_diffs):.6e}")
    print(f"  Max:  {max(max_diffs):.6e}")
    print(f"  Mean: {sum(max_diffs)/len(max_diffs):.6e}")
    print(f"\nToken Prediction Differences:")
    print(f"  Prompts where vLLM differs from HF: {flips}/{len(prompts)} ({100*flips/len(prompts):.1f}%)")
    print(f"\nResults saved to: {args.out_csv}")

if __name__ == "__main__":
    main()
