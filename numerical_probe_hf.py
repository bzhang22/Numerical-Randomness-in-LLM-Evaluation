#!/usr/bin/env python3
"""
numerical_probe.py

Numerical randomness probe for LLM inference using HuggingFace Transformers.

Goal (no tasks / no accuracy):
- Fix a set of prompts (e.g., 1000 short prompts)
- Run forward pass N times under the same seed
- Compare score-level outputs at the last token position:
  1) logits max_abs_diff (relative to run0)
  2) logprob max-min for a fixed token (run0 top1 token)
  3) top1 token flip rate across runs

This probes numerical non-determinism even when argmax/output is stable.

Example:
  python numerical_probe.py \
    --model EleutherAI/pythia-410m \
    --dtype float16 \
    --device cuda:0 \
    --batch_size 8 \
    --num_prompts 1000 \
    --runs 20 \
    --max_new_tokens 0 \
    --max_length 256 \
    --seed 123 \
    --out_csv probe_results.csv

You can also provide prompts from a text file (one prompt per line):
  --prompts_file prompts.txt
"""

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
# from datasets import load_dataset  # direct load is broken for PIQA on this env
import lm_eval.tasks
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Utilities
# -----------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch(
    deterministic: bool,
    tf32: bool,
    cudnn_benchmark: bool,
) -> None:
    # TF32 controls matmul precision on Ampere+ (3090 is Ampere).
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    # cudnn benchmark may choose different algorithms (can affect determinism).
    torch.backends.cudnn.benchmark = cudnn_benchmark

    if deterministic:
        # Enforce deterministic algorithms when possible (may throw if not possible).
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False


def load_prompts(prompts_file: Optional[str], num_prompts: int, dataset_name: Optional[str] = None) -> List[str]:
    if dataset_name == "piqa":
        print("Loading prompts from PIQA via lm_eval...")
        # Use lm-eval to load the task, which handles the complex loading logic/config
        task_dict = lm_eval.tasks.get_task_dict(["piqa"])
        t = task_dict["piqa"]
        
        # We need the validation set (or test if val not avail, but PIQA has val)
        if t.has_validation_docs():
            docs = list(t.validation_docs())
        elif t.has_test_docs():
            docs = list(t.test_docs())
        else:
            # fallback to training
             docs = list(t.training_docs())
             
        if not docs:
            raise ValueError("PIQA dataset seems empty via lm_eval?")
            
        # Extract 'goal' from docs
        goals = [d["goal"] for d in docs]
        
        out = (goals * ((num_prompts + len(goals) - 1) // len(goals)))[:num_prompts]
        return out

    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            raise ValueError(f"No prompts found in {prompts_file}")
        # If file has fewer lines than requested, repeat cyclically
        out = (lines * ((num_prompts + len(lines) - 1) // len(lines)))[:num_prompts]
        return out

    # Default synthetic prompts: short, diverse, stable
    # (These are just to exercise compute; replace with your own if you want.)
    base = [
        "Explain why the sky is blue in one sentence.",
        "Write a short definition of entropy.",
        "Translate 'good morning' into Chinese.",
        "What is 17 + 25? Answer with a number only.",
        "Complete: The capital of France is",
        "Give one synonym for 'happy'.",
        "Name one planet in our solar system.",
        "Finish the sentence: Machine learning is",
        "Provide a short fact about penguins.",
        "What does CPU stand for?",
    ]
    # Expand deterministically
    out = []
    for i in range(num_prompts):
        out.append(f"[{i:04d}] {base[i % len(base)]}")
    return out


def chunked(lst: List[str], bs: int):
    for i in range(0, len(lst), bs):
        yield lst[i : i + bs]


def dtype_from_str(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


@dataclass
class PromptStats:
    # reference (run0)
    ref_top1_id: int
    ref_top1_logprob: float

    # aggregated across runs (including run0)
    max_logit_abs_diff: float = 0.0
    min_ref_token_logprob: float = float("inf")
    max_ref_token_logprob: float = float("-inf")
    top1_flip_count: int = 0  # number of runs where top1 != ref_top1_id
    runs_seen: int = 0


# -----------------------------
# Core probe
# -----------------------------
@torch.inference_mode()
def forward_last_logits(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Returns last-position logits for each sample: [B, V]
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits  # [B, T, V]
    # last token position for each sample (based on attention_mask)
    # last_index = (attention_mask.sum(dim=1) - 1)
    last_index = attention_mask.long().sum(dim=1) - 1  # [B]
    # gather last logits
    bsz, _, vocab = logits.shape
    idx = last_index.view(bsz, 1, 1).expand(bsz, 1, vocab)
    last_logits = logits.gather(dim=1, index=idx).squeeze(1)  # [B, V]
    return last_logits


def main():
    ap = argparse.ArgumentParser(description="Numerical probe for score-level randomness.")
    ap.add_argument("--model", type=str, required=True, help="HF model name or local path")
    ap.add_argument("--dtype", type=str, default="float16", help="float16|bfloat16|float32")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_prompts", type=int, default=1000)
    ap.add_argument("--prompts_file", type=str, default=None, help="Optional text file, one prompt per line")
    ap.add_argument("--runs", type=int, default=20, help="Number of repeated forwards")
    ap.add_argument("--max_length", type=int, default=256, help="Tokenizer truncation length")
    ap.add_argument("--seed", type=int, default=123)

    # determinism / kernel knobs
    ap.add_argument("--deterministic", action="store_true", help="Force deterministic algorithms (may error)")
    ap.add_argument("--tf32", action="store_true", help="Enable TF32 matmul (can change numerics)")
    ap.add_argument("--cudnn_benchmark", action="store_true", help="Enable cuDNN benchmark (can change alg selection)")

    # New args for PIQA and CPU Ref
    ap.add_argument("--dataset", type=str, default=None, choices=["piqa"], help="Dataset to load prompts from (e.g. piqa)")
    ap.add_argument("--use_cpu_ref", action="store_true", help="Run the reference pass (Run 0) on CPU, then move to device for others.")

    ap.add_argument("--out_csv", type=str, default="probe_results.csv")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    configure_torch(args.deterministic, args.tf32, args.cudnn_benchmark)

    device = torch.device(args.device)
    dt = dtype_from_str(args.dtype)

    # If using CPU ref, load to CPU first. Otherwise load to target device.
    load_device = "cpu" if args.use_cpu_ref else device
    
    print(f"Loading tokenizer/model: {args.model} -> {load_device} (initial)")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        # common for causal LMs
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dt,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    prompts = load_prompts(args.prompts_file, args.num_prompts)
    print(f"Prompts: {len(prompts)} | runs={args.runs} | bs={args.batch_size} | dtype={args.dtype} | tf32={args.tf32}")

    # Storage for per-prompt stats
    stats: List[Optional[PromptStats]] = [None] * len(prompts)

    # We'll do:
    # - Run 0: compute reference logits, ref top1 token, ref token logprob
    # - Run k>0: compute current logits, compare against reference (logit abs diff),
    #           compute logprob of ref token, top1 flips.

    # To avoid storing huge reference logits [N, V], we store only:
    # - reference top1 token id
    # - reference top1 logprob
    # - reference logits for *full vocab* would be needed for exact max_abs_diff.
    #
    # However, we *can* compute max_abs_diff by storing ref logits batch-by-batch on CPU
    # and reusing them on each run. That would still be heavy if you store all N*V.
    #
    # Better compromise:
    # - measure max_abs_diff only on a small set of "probe tokens" (e.g., topK tokens of run0),
    #   OR
    # - measure logprob jitter (most meaningful) + top1 flips (discrete)
    #
    # You asked specifically for logits max_abs_diff, so here we do:
    # - store ref last_logits in float32 on CPU in chunks (still potentially big).
    #
    # For pythia-410m vocab ~50k, N=1000 => 1000*50k*4 bytes ~ 200MB (float32) for ref logits.
    # This is acceptable on most machines. If too big, reduce num_prompts or store float16.
    store_ref_dtype = torch.float32
    ref_logits_cpu: List[torch.Tensor] = []  # list of [B, V] tensors on CPU in order

    def tokenize_batch(batch_prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = tok(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # -----------------
    # Run 0 (reference)
    # -----------------
    print("\n[Run 0] Computing reference logits/top1...")
    idx0 = 0
    for batch in chunked(prompts, args.batch_size):
        input_ids, attn = tokenize_batch(batch)
        last_logits = forward_last_logits(model, input_ids, attn)  # [B, V] on GPU dtype
        last_logits_f32 = last_logits.float()

        # store ref logits on CPU (float32)
        ref_logits_cpu.append(last_logits_f32.cpu().to(store_ref_dtype))

        # compute ref top1 + ref top1 logprob
        logp = torch.log_softmax(last_logits_f32, dim=-1)  # [B, V]
        top1 = torch.argmax(last_logits_f32, dim=-1)       # [B]
        top1_lp = logp.gather(dim=-1, index=top1.view(-1, 1)).squeeze(1)  # [B]

        for j in range(len(batch)):
            pid = idx0 + j
            stats[pid] = PromptStats(
                ref_top1_id=int(top1[j].item()),
                ref_top1_logprob=float(top1_lp[j].item()),
                max_logit_abs_diff=0.0,
                min_ref_token_logprob=float(top1_lp[j].item()),
                max_ref_token_logprob=float(top1_lp[j].item()),
                top1_flip_count=0,
                runs_seen=1,
            )
        idx0 += len(batch)

    # -----------------
    # Runs 1..K
    # -----------------
    for r in range(1, args.runs):
        print(f"\n[Run {r}] Forward and compare...")
        set_all_seeds(args.seed)  # keep same seed each run
        # note: seed doesn't guarantee determinism, but keeps RNG consistent if used.

        idx = 0
        ref_chunk_i = 0
        for batch in chunked(prompts, args.batch_size):
            input_ids, attn = tokenize_batch(batch)
            last_logits = forward_last_logits(model, input_ids, attn).float()  # [B, V] float32 on GPU
            ref_logits = ref_logits_cpu[ref_chunk_i].to(device)                # [B, V] float32 on GPU
            ref_chunk_i += 1

            # logits max abs diff per sample
            diff = (last_logits - ref_logits).abs().max(dim=-1).values  # [B]

            # logprob of the reference token (run0 top1) per sample
            logp = torch.log_softmax(last_logits, dim=-1)  # [B, V]
            for j in range(len(batch)):
                pid = idx + j
                st = stats[pid]
                assert st is not None

                ref_tok = st.ref_top1_id
                lp_ref_tok = float(logp[j, ref_tok].item())

                st.max_logit_abs_diff = max(st.max_logit_abs_diff, float(diff[j].item()))
                st.min_ref_token_logprob = min(st.min_ref_token_logprob, lp_ref_tok)
                st.max_ref_token_logprob = max(st.max_ref_token_logprob, lp_ref_tok)

                # top1 flip?
                top1_tok = int(torch.argmax(last_logits[j]).item())
                if top1_tok != st.ref_top1_id:
                    st.top1_flip_count += 1

                st.runs_seen += 1

            idx += len(batch)

    # -----------------
    # Summarize + write CSV
    # -----------------
    # Compute global summaries
    max_abs_diffs = [st.max_logit_abs_diff for st in stats if st is not None]
    ref_tok_ranges = [(st.max_ref_token_logprob - st.min_ref_token_logprob) for st in stats if st is not None]
    flip_rates = [st.top1_flip_count / max(1, (st.runs_seen - 1)) for st in stats if st is not None]

    def q(arr, p):
        arr = sorted(arr)
        k = int(round((len(arr) - 1) * p))
        return arr[k]

    print("\n" + "=" * 80)
    print("NUMERICAL PROBE SUMMARY")
    print("=" * 80)
    print(f"model={args.model} dtype={args.dtype} device={args.device} bs={args.batch_size} runs={args.runs}")
    print(f"deterministic={args.deterministic} tf32={args.tf32} cudnn_benchmark={args.cudnn_benchmark}")
    print(f"num_prompts={len(prompts)} max_length={args.max_length}")

    print("\nlogits max_abs_diff (per prompt, relative to run0):")
    print(f"  min={min(max_abs_diffs):.3e}  p50={q(max_abs_diffs,0.5):.3e}  p90={q(max_abs_diffs,0.9):.3e}  max={max(max_abs_diffs):.3e}")

    print("\nlogprob range for fixed token (run0 top1 token): max-min over runs")
    print(f"  min={min(ref_tok_ranges):.3e}  p50={q(ref_tok_ranges,0.5):.3e}  p90={q(ref_tok_ranges,0.9):.3e}  max={max(ref_tok_ranges):.3e}")

    overall_flip = sum(1 for fr in flip_rates if fr > 0)
    print("\ntop1 token flip (relative to run0 top1):")
    print(f"  prompts with any flip = {overall_flip}/{len(flip_rates)} ({overall_flip/len(flip_rates):.2%})")
    print(f"  flip-rate p50={q(flip_rates,0.5):.3e}  p90={q(flip_rates,0.9):.3e}  max={max(flip_rates):.3e}")

    # Write per-prompt CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "prompt_id",
            "prompt",
            "ref_top1_token_id",
            "ref_top1_logprob",
            "logits_max_abs_diff",
            "ref_token_logprob_min",
            "ref_token_logprob_max",
            "ref_token_logprob_range",
            "top1_flip_count",
            "runs_seen",
        ])
        for i, (p, st) in enumerate(zip(prompts, stats)):
            assert st is not None
            w.writerow([
                i,
                p,
                st.ref_top1_id,
                f"{st.ref_top1_logprob:.10f}",
                f"{st.max_logit_abs_diff:.10e}",
                f"{st.min_ref_token_logprob:.10f}",
                f"{st.max_ref_token_logprob:.10f}",
                f"{(st.max_ref_token_logprob - st.min_ref_token_logprob):.10e}",
                st.top1_flip_count,
                st.runs_seen,
            ])

    print(f"\nWrote per-prompt results to: {args.out_csv}")

    # Print top unstable prompts (by logits diff)
    topk = 10
    ranked = sorted(range(len(stats)), key=lambda i: stats[i].max_logit_abs_diff, reverse=True)  # type: ignore
    print(f"\nTop-{topk} most unstable prompts by logits_max_abs_diff:")
    for i in ranked[:topk]:
        st = stats[i]
        assert st is not None
        print(f"  id={i:4d}  logits_max_abs_diff={st.max_logit_abs_diff:.3e}  "
              f"logprob_range={st.max_ref_token_logprob - st.min_ref_token_logprob:.3e}  "
              f"flip_count={st.top1_flip_count}  prompt={prompts[i][:80]}")


if __name__ == "__main__":
    main()
