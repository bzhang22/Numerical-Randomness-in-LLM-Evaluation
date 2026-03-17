import os
import json
import jsonlines
import pandas as pd
import argparse
from typing import Dict, Any, Tuple
import Levenshtein

def load_samples(filepath: str) -> Dict[str, Any]:
    """Loads a samples.jsonl file into a dictionary keyed by doc_id."""
    samples = {}
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return samples
        
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            samples[obj['doc_id']] = obj
    return samples

def find_first_divergence(str1: str, str2: str) -> int:
    """Finds the character index where two strings first diverge."""
    min_len = min(len(str1), len(str2))
    for i in range(min_len):
        if str1[i] != str2[i]:
            return i
    if len(str1) != len(str2):
        return min_len
    return -1

def calculate_token_mismatch_rate(str1: str, str2: str) -> float:
    """Calculates a rough mismatch rate based on word tokens."""
    # Note: A real token-level mismatch would require the specific tokenizer.
    # We use a word-level approximation for string outputs.
    tokens1 = str1.split()
    tokens2 = str2.split()
    
    if not tokens1 and not tokens2:
        return 0.0
    
    # Calculate Levenshtein distance on token lists
    distance = Levenshtein.distance(tokens1, tokens2)
    max_len = max(len(tokens1), len(tokens2))
    
    return distance / max_len if max_len > 0 else 0.0

def extract_prediction(item: dict) -> str:
    resps = item.get('resps', [])
    if not resps: return ""
    
    try:
        # Multiple-choice logical check: resps is [[["-6.98", "False"]], [["-7.49", "False"]]]
        if isinstance(resps, list) and isinstance(resps[0], list) and isinstance(resps[0][0], list):
            import numpy as np
            loglikelihoods = [float(choice[0][0]) for choice in resps]
            return str(np.argmax(loglikelihoods))
    except Exception:
        pass
        
    # Standard generative sequence fallback
    try:
        if isinstance(resps[0], list):
            return str(resps[0][0])
        return str(resps[0])
    except Exception:
        return str(resps)

def compare_configs(model_name: str, benchmark: str, config_a_name: str, config_a_path: str, config_b_name: str, config_b_path: str) -> pd.DataFrame:
    """Compares two configurations for a specific model and benchmark."""
    
    import glob
    
    samples_a_file = None
    samples_b_file = None
    
    # Search for samples_{benchmark}_*.jsonl or samples.jsonl recursively
    search_pattern_a = os.path.join(config_a_path, "**", f"samples_{benchmark}*.jsonl")
    search_pattern_b = os.path.join(config_b_path, "**", f"samples_{benchmark}*.jsonl")
    
    matches_a = glob.glob(search_pattern_a, recursive=True)
    matches_b = glob.glob(search_pattern_b, recursive=True)
    
    if matches_a:
        samples_a_file = matches_a[0]
    else:
        # Fallback to general samples.jsonl
        fallback_a = glob.glob(os.path.join(config_a_path, "**", "samples.jsonl"), recursive=True)
        if fallback_a: samples_a_file = fallback_a[0]

    if matches_b:
        samples_b_file = matches_b[0]
    else:
        fallback_b = glob.glob(os.path.join(config_b_path, "**", "samples.jsonl"), recursive=True)
        if fallback_b: samples_b_file = fallback_b[0]
        
    if not samples_a_file or not samples_b_file:
        print(f"File not found for {benchmark} in {config_a_path} or {config_b_path}")
        return pd.DataFrame()

    samples_a = load_samples(samples_a_file)
    samples_b = load_samples(samples_b_file)
    
    records = []
    
    common_keys = set(samples_a.keys()).intersection(set(samples_b.keys()))
    
    if not common_keys:
        print(f"Warning: No common document IDs found between {config_a_name} and {config_b_name} for {benchmark}.")
        return pd.DataFrame()
        
    for doc_id in common_keys:
        item_a = samples_a[doc_id]
        item_b = samples_b[doc_id]
        
        resp_a = extract_prediction(item_a)
        resp_b = extract_prediction(item_b)
        
        # Determine correctness if available
        acc_a = item_a.get('acc', None)
        acc_b = item_b.get('acc', None)
        
        exact_match = (resp_a == resp_b)
        first_div_pos = -1 if exact_match else find_first_divergence(resp_a, resp_b)
        
        edit_dist = 0 if exact_match else Levenshtein.distance(resp_a, resp_b)
        mismatch_rate = 0.0 if exact_match else calculate_token_mismatch_rate(resp_a, resp_b)
        
        correctness_flip = False
        if acc_a is not None and acc_b is not None:
            correctness_flip = (acc_a != acc_b)
            
        records.append({
            'model': model_name,
            'benchmark': benchmark,
            'prompt_id': doc_id,
            'config_A': config_a_name,
            'config_B': config_b_name,
            'exact_match': exact_match,
            'first_divergence_pos': first_div_pos,
            'token_mismatch_rate': mismatch_rate,
            'edit_distance': edit_dist,
            'correctness_flip': correctness_flip
        })
        
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Pairwise Comparison of LM-Eval Results")
    parser.add_argument("--base_dir", type=str, default="results", help="Base directory containing results")
    parser.add_argument("--output_file", type=str, default="pairwise_compare.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    
    models = [d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))]
    benchmarks = ["piqa", "cmmlu", "gsm8k", "humaneval", "wikitext", "lambada_openai"]
    
    all_records = []
    
    for model in models:
        model_dir = os.path.join(args.base_dir, model)
        
        # 1. Compare Precision Sweep
        precision_dir = os.path.join(model_dir, "precision")
        if os.path.exists(precision_dir):
            precision_configs = ["float32", "float16", "bfloat16"]
            
            # Pairwise combinations (FP32 vs FP16, FP32 vs BF16, FP16 vs BF16)
            pairs = [("float32", "float16"), ("float32", "bfloat16"), ("float16", "bfloat16")]
            
            for benchmark in benchmarks:
                for conf_a, conf_b in pairs:
                    path_a = os.path.join(precision_dir, conf_a)
                    path_b = os.path.join(precision_dir, conf_b)
                    
                    if os.path.exists(path_a) and os.path.exists(path_b):
                        print(f"Comparing Precision: {model} - {benchmark} - {conf_a} vs {conf_b}")
                        df = compare_configs(model, benchmark, conf_a, path_a, conf_b, path_b)
                        if not df.empty:
                            all_records.append(df)
                            
        # 2. Compare Batch Sweep
        batch_dir = os.path.join(model_dir, "batch")
        if os.path.exists(batch_dir):
            batch_configs = ["bs1", "bs2", "bs4", "bs8", "bs16"]
            
            for benchmark in benchmarks:
                for conf_b in ["bs2", "bs4", "bs8", "bs16"]:
                    path_a = os.path.join(batch_dir, "bs1")
                    path_b = os.path.join(batch_dir, conf_b)
                    
                    if os.path.exists(path_a) and os.path.exists(path_b):
                        print(f"Comparing Batch: {model} - {benchmark} - bs1 vs {conf_b}")
                        df = compare_configs(model, benchmark, "bs1", path_a, conf_b, path_b)
                        if not df.empty:
                            all_records.append(df)
                            
        # 3. Compare Attention Sweep
        attention_dir = os.path.join(model_dir, "attention")
        if os.path.exists(attention_dir):
            pairs = [("eager", "sdpa"), ("eager", "flash_attention_2"), ("sdpa", "flash_attention_2")]
            
            for benchmark in benchmarks:
                for conf_a, conf_b in pairs:
                    path_a = os.path.join(attention_dir, conf_a)
                    path_b = os.path.join(attention_dir, conf_b)
                    
                    if os.path.exists(path_a) and os.path.exists(path_b):
                        print(f"Comparing Attention: {model} - {benchmark} - {conf_a} vs {conf_b}")
                        df = compare_configs(model, benchmark, conf_a, path_a, conf_b, path_b)
                        if not df.empty:
                            all_records.append(df)

    if all_records:
        final_df = pd.concat(all_records, ignore_index=True)
        final_df.to_csv(args.output_file, index=False)
        print(f"Comparison complete. Results saved to {args.output_file}")
        print(f"Total divergence cases found: {len(final_df[~final_df['exact_match']])}")
    else:
        print("No paired results found to compare.")

if __name__ == "__main__":
    main()
