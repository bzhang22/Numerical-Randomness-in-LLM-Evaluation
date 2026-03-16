import os
import json
import glob
import torch
import pandas as pd
import numpy as np
import argparse
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def locate_jsonl(base_dir, model_name, dtype, benchmark):
    search_path = os.path.join(base_dir, model_name, "precision", dtype, "**", f"samples_{benchmark}_*.jsonl")
    files = glob.glob(search_path, recursive=True)
    if not files:
        return None
    return sorted(files)[-1]

def build_tracing_dataset(csv_path, base_dir, target_models, samples_per_benchmark=5):
    """Parses the CSV and fetches exact Prompt + Output strings from the FP32 logs."""
    df = pd.read_csv(csv_path)
    dataset = []
    
    for model in target_models:
        model_df = df[df['model'] == model]
        if model_df.empty:
            continue
            
        benchmarks = ["gsm8k", "piqa", "cmmlu", "humaneval"]
        for bench in benchmarks:
            bench_df = model_df[model_df['benchmark'] == bench]
            if bench_df.empty:
                continue
                
            # Filter solely for FP32 vs BF16 combinations to find divergent targets
            div_mask = (~bench_df['exact_match']) & (bench_df['config_A'] == 'float32') & (bench_df['config_B'] == 'bfloat16')
            stable_mask = (bench_df['exact_match']) & (bench_df['config_A'] == 'float32') & (bench_df['config_B'] == 'bfloat16')
            
            div_cases = bench_df[div_mask].head(samples_per_benchmark)
            stable_cases = bench_df[stable_mask].head(samples_per_benchmark)
            
            selected_cases = pd.concat([div_cases, stable_cases])
            
            if selected_cases.empty:
                continue
                
            jsonl_path = locate_jsonl(base_dir, model, "float32", bench)
            if not jsonl_path:
                print(f"WARNING: Cannot find FP32 JSONL for {model} {bench}")
                continue
                
            # Load JSONL into memory
            log_entries = {}
            with open(jsonl_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    log_entries[entry.get("doc_id", -1)] = entry
            
            for _, row in selected_cases.iterrows():
                doc_id = int(row['prompt_id'])
                if doc_id not in log_entries:
                    continue
                entry = log_entries[doc_id]
                
                # Extract prompt and strictly the FP32 model's answer
                prompt = entry.get('arguments', {}).get('gen_args_0', {}).get('arg_0', "")
                resps = entry.get('resps', [[]])
                if not prompt or not resps or not resps[0]:
                    continue
                    
                fp32_output = resps[0][0]
                
                dataset.append({
                    "model": model,
                    "benchmark": bench,
                    "prompt_id": doc_id,
                    "is_divergent": not row['exact_match'],
                    "prompt": prompt,
                    "fp32_output": fp32_output,
                    "first_divergence_pos": row['first_divergence_pos']
                })
    return pd.DataFrame(dataset)

def extract_tensors_for_model(model_name, dtype, dataset, device="cuda"):
    """Loads a specific model precision, runs teacher-forcing over the dataset, and extracts matrices."""
    print(f"\n--- Loading {model_name} in {dtype} ---")
    hf_path = f"meta-llama/{model_name}" if "Llama" in model_name else f"mistralai/{model_name}"
    
    # Map friendly dtypes to torch kwargs
    torch_dtype = torch.float32
    if dtype == "float16": torch_dtype = torch.float16
    if dtype == "bfloat16": torch_dtype = torch.bfloat16
        
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    model = AutoModelForCausalLM.from_pretrained(
        hf_path, 
        torch_dtype=torch_dtype, 
        device_map="auto"
    )
    model.eval()
    
    extracted_data = {}
    
    with torch.no_grad():
        for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Tracing {dtype}"):
            if row['model'] != model_name:
                continue
                
            prompt = str(row['prompt'])
            output = str(row['fp32_output'])
            full_text = prompt + output
            
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
            
            prompt_len = prompt_ids.shape[1]
            total_len = full_ids.shape[1]
            
            if total_len <= prompt_len:
                continue # No actual generation
                
            # Teacher-forced single forward pass generating all hidden states!
            outputs = model(full_ids, output_hidden_states=True)
            
            # Hidden states: tuple of (batch, seq_len, hidden_dim) for each layer
            # We specifically extract the tokens AFTER the prompt
            hidden_states = []
            for layer_idx, h_state in enumerate(outputs.hidden_states):
                # Shape: (seq_len_gen, hidden_dim)
                h_gen = h_state[0, prompt_len-1:-1, :].cpu().to(torch.float32) 
                hidden_states.append(h_gen)
                
            # Stack to (num_layers, seq_len_gen, hidden_dim)
            stacked_hidden = torch.stack(hidden_states)
            
            # Extract Logits to calculate Top-1 / Top-2 bounds
            logits = outputs.logits[0, prompt_len-1:-1, :].cpu().to(torch.float32)
            
            extracted_data[row['prompt_id']] = {
                "hidden_states": stacked_hidden,
                "logits": logits
            }
            
    # Purge VRAM violently
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return extracted_data

def compute_and_save_traces(model_name, dataset, output_dir):
    """Mechanically extracts FP32, FP16, and BF16 locally to compute differentials."""
    
    fp32_data = extract_tensors_for_model(model_name, "float32", dataset)
    fp16_data = extract_tensors_for_model(model_name, "float16", dataset)
    bf16_data = extract_tensors_for_model(model_name, "bfloat16", dataset)
    
    for _, row in dataset[dataset['model'] == model_name].iterrows():
        pid = row['prompt_id']
        if pid not in fp32_data or pid not in bf16_data or pid not in fp16_data:
            continue
            
        t_fp32 = fp32_data[pid]
        t_fp16 = fp16_data[pid]
        t_bf16 = bf16_data[pid]
        
        num_layers, seq_len, _ = t_fp32["hidden_states"].shape
        
        # 1. Calculate Layerwise MAE (Mean Absolute Error) over the hidden dimension
        # Shape becomes (num_layers,) representing the average error across all generated tokens and hidden dims
        mae_fp16 = torch.nn.functional.l1_loss(t_fp32["hidden_states"], t_fp16["hidden_states"], reduction='none').mean(dim=(1,2)).numpy()
        mae_bf16 = torch.nn.functional.l1_loss(t_fp32["hidden_states"], t_bf16["hidden_states"], reduction='none').mean(dim=(1,2)).numpy()
        
        # 2. Calculate Token Flip Probability vs Margin
        # Extract top-2 margins from the baseline FP32 logits
        fp32_vals, fp32_idx = torch.topk(t_fp32["logits"], k=2, dim=-1)
        margins = (fp32_vals[:, 0] - fp32_vals[:, 1]).numpy()
        
        # Check if BF16 / FP16 actually flipped the Top-1 token selection exactly at this step!
        bf16_idx = torch.argmax(t_bf16["logits"], dim=-1)
        fp16_idx = torch.argmax(t_fp16["logits"], dim=-1)
        
        flipped_bf16 = (fp32_idx[:, 0] != bf16_idx).numpy()
        flipped_fp16 = (fp32_idx[:, 0] != fp16_idx).numpy()
        
        # Structure payload
        trace_dir = os.path.join(output_dir, model_name, row['benchmark'])
        os.makedirs(trace_dir, exist_ok=True)
        
        trace = {
            "metadata": {
                "model": model_name,
                "benchmark": row['benchmark'],
                "prompt_id": pid,
                "is_divergent": bool(row['is_divergent'])
            },
            "layerwise_mae": {
                "fp32_vs_fp16": mae_fp16.tolist(),
                "fp32_vs_bf16": mae_bf16.tolist()
            },
            "token_step_dynamics": {
                "fp32_top2_margin": margins.tolist(),
                "flipped_in_bf16": flipped_bf16.tolist(),
                "flipped_in_fp16": flipped_fp16.tolist()
            }
        }
        
        with open(os.path.join(trace_dir, f"trace_{pid}.json"), 'w') as f:
            json.dump(trace, f)
            
    print(f"[{model_name}] Extracted traces written to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="pairwise_compare.csv")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="trace")
    parser.add_argument("--models", nargs="+", default=["Meta-Llama-3.1-8B", "Llama-3.2-3B"])
    args = parser.parse_args()
    
    print("Building Teacher Forcing Dataset...")
    dataset = build_tracing_dataset(args.csv, args.results_dir, args.models, samples_per_benchmark=5)
    print(f"Assembled {len(dataset)} sequence targets for deep layer stripping.")
    
    for model in args.models:
        compute_and_save_traces(model, dataset, args.output_dir)

if __name__ == "__main__":
    main()
