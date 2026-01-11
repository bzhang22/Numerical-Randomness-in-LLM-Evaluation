
import json
import os
import glob
from tabulate import tabulate

def get_metric(filepath, metric_name):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', {})
        task_key = None
        for key in results.keys():
            if key in ['wikitext', 'piqa', 'commonsense_qa', 'cqa']:
                task_key = key
                break
        
        if not task_key:
             if len(results) > 0:
                task_key = list(results.keys())[0]
             else:
                return 'No Results'

        res = results[task_key]
        
        keys_to_try = [f"{metric_name},none", metric_name, f"{metric_name},stderr"]
        for k in keys_to_try:
            if k in res:
                return res[k]
                
        return 'Metric Not Found'
    except Exception as e:
        return f"Error: {e}"

def main():
    base_dir = "diagnosis_qwen3_combinatorial"
    phase2_dir = "diagnosis_qwen3_small" # For absolute baseline
    
    datasets = {
        "wikitext": "word_perplexity",
        "piqa": "acc",
        "commonsense_qa": "acc"
    }
    
    configs = [
        ("Batch 4 (Control)", "_batch4_control.json"),
        ("Batch 4 + Eager", "_batch4_eager.json"),
        ("Batch 4 + Gemm Det", "_batch4_gemm_det.json"),
        ("Full Det (B4+E+D)", "_batch4_full_det.json")
    ]

    for dataset, metric in datasets.items():
        print(f"\n=== Dataset: {dataset} (Metric: {metric}) ===")
        
        # Try to get absolute baseline from Phase 2
        abs_baseline_val = None
        abs_base_path = os.path.join(phase2_dir, f"{dataset}_baseline.json")
        if os.path.exists(abs_base_path):
            abs_baseline_val = get_metric(abs_base_path, metric)
            if isinstance(abs_baseline_val, (int, float)):
                print(f"Phase 2 Baseline (Batch 1): {abs_baseline_val:.6f}")
            else:
                print(f"Phase 2 Baseline (Batch 1): Not Available ({abs_baseline_val})")
        else:
            print("Phase 2 Baseline: Not Found")

        table_data = []
        
        headers = ["Configuration", "Value", "Diff (Abs Base)", "Status"]
        
        for config_name, suffix in configs:
            filename = f"{dataset}{suffix}"
            filepath = os.path.join(base_dir, filename)
            
            row = [config_name]
            
            if not os.path.exists(filepath):
                row.extend(["Missing", "-", "Not Run/Failed"])
                table_data.append(row)
                continue
            
            val = get_metric(filepath, metric)
            
            if isinstance(val, (int, float)):
                status = "OK"
                diff = "-"
                
                if isinstance(abs_baseline_val, (int, float)):
                    if val == abs_baseline_val:
                        diff = "0.0 (Identical)"
                        status = "MATCH BASELINE"
                    else:
                        delta = val - abs_baseline_val
                        diff = f"{delta:+.6f}"
                        status = "MISMATCH" if abs(delta) > 1e-9 else "Match (<1e-9)"
                else:
                    diff = "N/A"
                    status = "?"
                
                row.extend([f"{val:.6f}", diff, status])
            else:
                row.extend([str(val), "-", "Error"])
            
            table_data.append(row)
            
        print(tabulate(table_data, headers=headers, floatfmt=".6f"))

if __name__ == "__main__":
    main()
