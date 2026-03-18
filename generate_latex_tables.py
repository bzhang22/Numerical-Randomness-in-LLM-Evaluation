import pandas as pd
import os
import json
import glob
import numpy as np

def generate_mismatch_table(csv_path: str):
    df = pd.read_csv(csv_path)
    # Filter out generative datasets for the exact match table
    df = df[~df['benchmark'].isin(['wikitext', 'lambada_openai', 'lambada'])]
    
    # Calculate mismatch rate
    df['mismatch'] = ~df['exact_match']
    
    # Exclude gemma-2 models due to their float16 overflow architectural defect
    df = df[~df['model'].str.contains('gemma', case=False, na=False)]
    
    
    # Standardize precision pairs
    def get_pair(a, b):
        mapping = {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16"}
        m_a, m_b = mapping.get(a, a), mapping.get(b, b)
        if m_a == "BF16" and m_b == "FP16" or m_b == "BF16" and m_a == "FP16": return "BF16 vs FP16"
        if m_a == "BF16" and m_b == "FP32" or m_b == "BF16" and m_a == "FP32": return "BF16 vs FP32"
        if m_a == "FP16" and m_b == "FP32" or m_b == "FP16" and m_a == "FP32": return "FP16 vs FP32"
        return f"{m_a} vs {m_b}"
        
    df['Precision Pair'] = df.apply(lambda r: get_pair(r['config_A'], r['config_B']), axis=1)
    
    # Filter only the 3 precision pairs
    valid_pairs = ["BF16 vs FP16", "BF16 vs FP32", "FP16 vs FP32"]
    df = df[df['Precision Pair'].isin(valid_pairs)]
    
    summary = df.groupby(['model', 'benchmark', 'Precision Pair'])['mismatch'].mean().reset_index()
    summary['Mismatch Rate (%)'] = (summary['mismatch'] * 100).round(2)
    
    # Rename columns for LaTeX
    summary = summary.rename(columns={'model': 'Model', 'benchmark': 'Benchmark'})
    summary = summary[['Model', 'Benchmark', 'Precision Pair', 'Mismatch Rate (%)']]
    
    # Sort for consistent output
    summary = summary.sort_values(by=['Model', 'Benchmark', 'Precision Pair'])
    
    latex_str = summary.to_latex(index=False, float_format="%.2f", na_rep="-", column_format="lllr")
    return latex_str

def extract_generative_ppl(base_dir: str):
    scores = []
    models = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    for model in models:
        for dtype in ["float32", "float16", "bfloat16"]:
            model_dir = os.path.join(base_dir, model, "precision", dtype)
            if not os.path.exists(model_dir):
                continue
            for subdir in os.listdir(model_dir):
                sub_path = os.path.join(model_dir, subdir)
                if not os.path.isdir(sub_path): continue
                json_files = glob.glob(os.path.join(sub_path, "results_*.json"))
                if not json_files: continue
                latest_json = sorted(json_files)[-1]
                with open(latest_json, 'r') as f:
                    try:
                        data = json.load(f)
                        res = data.get("results", {})
                    except: continue
                    for benchmark in ["wikitext", "lambada_openai"]:
                        if benchmark in res:
                            ppl = res[benchmark].get("word_perplexity,none", res[benchmark].get("perplexity,none", np.nan))
                            if pd.isna(ppl) and "acc,none" in res[benchmark]:
                                continue # We only want PPL
                            scores.append({
                                "Model": model,
                                "Benchmark": benchmark,
                                "Precision": dtype,
                                "PPL": ppl
                            })
    
    if not scores:
        return "No Generative PPL data found."
        
    df = pd.DataFrame(scores)
    # Exclude gemma-2 models due to their float16 overflow architectural defect
    df = df[~df['Model'].str.contains('gemma', case=False, na=False)]
    
    # Pivot to get FP32, FP16, BF16 as columns
    pivot = df.pivot_table(index=['Model', 'Benchmark'], columns='Precision', values='PPL', aggfunc='mean').reset_index()
    
    for col in ["float32", "float16", "bfloat16"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
            
    pivot = pivot.rename(columns={
        "float32": "FP32 PPL",
        "float16": "FP16 PPL",
        "bfloat16": "BF16 PPL"
    })
    
    pivot = pivot[['Model', 'Benchmark', 'FP32 PPL', 'FP16 PPL', 'BF16 PPL']]
    pivot = pivot.sort_values(by=['Model', 'Benchmark'])
    
    latex_str = pivot.to_latex(index=False, float_format="%.2f", na_rep="-", column_format="llrrr")
    return latex_str

def main():
    print("=== Table 1: Multiple Choice Exact Match Mismatch Rate ===")
    print(generate_mismatch_table("pairwise_compare.csv"))
    print("\n")
    print("=== Table 2: Generative Tasks Perplexity ===")
    print(extract_generative_ppl("results"))

if __name__ == "__main__":
    main()
