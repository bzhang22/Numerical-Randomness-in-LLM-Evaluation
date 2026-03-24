import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

RESULTS_DIR = "/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/results_mitigation"

DATASETS = ["piqa", "gsm8k", "cmmlu", "humaneval"]
VARIANTS = ["bf16_baseline", "fp16_baseline", "fp32_reference", "attention", "fp16_attention", "norm", "fp16_norm", "lm_head", "fp16_lm_head", "attention_lm_head", "fp16_attention_lm_head"]

MODELS = [
    "Llama-3.2-1B",
    "Llama-3.2-3B",
    "Meta-Llama-3.1-8B",
    "Mistral-7B-v0.3",
    "gemma-2-27b",
    "Yi-34B",
    "Meta-Llama-3.1-70B"
]

def load_jsonl(filepath):
    if not os.path.exists(filepath):
        return None
    data = {}
    with open(filepath, "r") as f:
        for line in f:
            j = json.loads(line)
            data[j["id"]] = j
    return data

def calculate_mismatch(toks_a, toks_b):
    if not toks_a and not toks_b: return 0.0
    if not toks_a or not toks_b: return 1.0
    
    length = min(len(toks_a), len(toks_b))
    mismatches = 0
    for i in range(length):
        if toks_a[i] != toks_b[i]:
            mismatches += 1
            
    mismatches += abs(len(toks_a) - len(toks_b))
    return mismatches / max(len(toks_a), len(toks_b))

def calculate_first_divergence(toks_a, toks_b):
    if not toks_a and not toks_b: return -1
    if not toks_a or not toks_b: return 0
    length = min(len(toks_a), len(toks_b))
    for i in range(length):
        if toks_a[i] != toks_b[i]:
            return i
    if len(toks_a) != len(toks_b):
        return length
    return -1

def main():
    summary_data = []
    
    for model in MODELS:
        for dataset in DATASETS:
            fp32_path = os.path.join(RESULTS_DIR, f"{model}_{dataset}_fp32_reference.jsonl")
            bf16_path = os.path.join(RESULTS_DIR, f"{model}_{dataset}_bf16_baseline.jsonl")
            
            fp32_results = load_jsonl(fp32_path)
            bf16_results = load_jsonl(bf16_path)
            
            if not fp32_results or not bf16_results:
                continue
                
            fp32_acc = sum(1 for v in fp32_results.values() if v["correct"]) / len(fp32_results)
            
            for variant in VARIANTS:
                if variant in ["fp32_reference", "bf16_baseline", "fp16_baseline"]: continue
                
                var_path = os.path.join(RESULTS_DIR, f"{model}_{dataset}_{variant}.jsonl")
                var_results = load_jsonl(var_path)
                
                if not var_results: continue
                
                # Determine correct baseline
                base_var_name = "fp16_baseline" if variant.startswith("fp16_") else "bf16_baseline"
                base_path = os.path.join(RESULTS_DIR, f"{model}_{dataset}_{base_var_name}.jsonl")
                base_results = load_jsonl(base_path)
                if not base_results: continue
                
                base_acc = sum(1 for v in base_results.values() if v["correct"]) / len(base_results)
                base_latency_avg = np.median([v["latency_sec"] for v in base_results.values()])
                
                var_acc = sum(1 for v in var_results.values() if v["correct"]) / len(var_results)
                var_latency_avg = np.median([v["latency_sec"] for v in var_results.values()])
                
                exact_matches = 0
                total_mismatch_rate = 0.0
                first_divs = []
                flips_cured = 0
                
                total_compared = 0
                
                for prompt_id in fp32_results:
                    if prompt_id not in var_results or prompt_id not in base_results:
                        continue
                        
                    fp32_toks = fp32_results[prompt_id]["generated_tokens"]
                    var_toks = var_results[prompt_id]["generated_tokens"]
                    base_toks = base_results[prompt_id]["generated_tokens"]
                    
                    if fp32_toks == var_toks:
                        exact_matches += 1
                        
                    total_mismatch_rate += calculate_mismatch(fp32_toks, var_toks)
                    
                    div_pos = calculate_first_divergence(fp32_toks, var_toks)
                    if div_pos != -1:
                        first_divs.append(div_pos)
                        
                    # Check Correctness Flip
                    if not base_results[prompt_id]["correct"] and var_results[prompt_id]["correct"] and fp32_results[prompt_id]["correct"]:
                        flips_cured += 1
                        
                    total_compared += 1
                
                if total_compared > 0:
                    summary_data.append({
                        "model": model,
                        "dataset": dataset,
                        "variant": variant,
                        "exact_match_vs_fp32": exact_matches / total_compared,
                        "token_mismatch_rate": total_mismatch_rate / total_compared,
                        "flips_cured_count": flips_cured,
                        "first_divergence_median": np.median(first_divs) if first_divs else -1,
                        "benchmark_acc": var_acc,
                        "benchmark_gap_vs_fp32": var_acc - fp32_acc,
                        "benchmark_gap_vs_baseline": var_acc - base_acc,
                        "latency_overhead_percent": ((var_latency_avg / base_latency_avg) - 1.0) * 100 if base_latency_avg > 0 else 0
                    })
                    
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(RESULTS_DIR, "mitigation_analysis_summary.csv"), index=False)
        print("Analysis complete. Dumped summary to mitigation_analysis_summary.csv.")
        
        # Build Table M1 aggregated across datasets
        table_m1 = df.groupby('variant')[['exact_match_vs_fp32', 'token_mismatch_rate', 'benchmark_gap_vs_fp32', 'latency_overhead_percent']].mean().reset_index()
        table_m1.to_csv(os.path.join(RESULTS_DIR, "Table_M1.csv"), index=False)
        print("Built Table M1 aggregate!")
    else:
        print("No valid paired results found yet.")

if __name__ == "__main__":
    main()
