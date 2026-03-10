import os
import json
import pandas as pd
import argparse
import random
from typing import List, Dict, Any

def get_divergent_cases(comparison_file: str, max_cases: int = 50) -> pd.DataFrame:
    """Reads the comparison CSV and returns a sample of divergent cases."""
    if not os.path.exists(comparison_file):
        print(f"Comparison file {comparison_file} not found. Run pairwise_compare.py first.")
        return pd.DataFrame()
        
    df = pd.read_csv(comparison_file)
    divergent = df[~df['exact_match']]
    
    if len(divergent) > max_cases:
        return divergent.sample(n=max_cases, random_state=42)
    return divergent

def write_dummy_trace(row: pd.Series, output_dir: str):
    """
    Creates a mock trace file for a divergent case.
    In the real implementation, this would involve loading the specific model config
    and re-running the prompt_id through the model to capture internal states using hooks.
    """
    model = row['model']
    benchmark = row['benchmark']
    config = f"{row['config_A']}_vs_{row['config_B']}"
    prompt_id = row['prompt_id']
    
    trace_dir = os.path.join(output_dir, model, benchmark, config)
    os.makedirs(trace_dir, exist_ok=True)
    
    trace_file = os.path.join(trace_dir, f"trace_{prompt_id}.json")
    
    # Structure based on user requirements for internal-level tracing
    trace_data = {
        "metadata": {
            "model": model,
            "benchmark": benchmark,
            "prompt_id": prompt_id,
            "config_A": row['config_A'],
            "config_B": row['config_B'],
            "first_divergence_pos": int(row['first_divergence_pos']),
            "correctness_flip": bool(row['correctness_flip'])
        },
        "per_step_logits": [], # Would be populated by hook output
        "top1_top2_margin": [], # Would be populated by hook output
        "selected_hidden_states": [] # Would be populated by hook output
    }
    
    with open(trace_file, 'w') as f:
        json.dump(trace_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Extract internal traces for divergent cases")
    parser.add_argument("--comparison_file", type=str, default="pairwise_compare.csv")
    parser.add_argument("--output_dir", type=str, default="trace")
    parser.add_argument("--max_cases", type=int, default=50)
    
    args = parser.parse_args()
    
    divergent_cases = get_divergent_cases(args.comparison_file, args.max_cases)
    
    if divergent_cases.empty:
        print("No divergent cases found to trace.")
        return
        
    print(f"Extracting traces for {len(divergent_cases)} divergent cases...")
    
    for _, row in divergent_cases.iterrows():
        write_dummy_trace(row, args.output_dir)
        
    print(f"Traces saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
