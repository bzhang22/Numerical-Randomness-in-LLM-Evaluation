
import json
import os
import glob
from tabulate import tabulate

def get_metric(filepath, metric_name):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Determine task key (wikitext, piqa, commonsense_qa)
        # Results usually keyed by task name
        results = data.get('results', {})
        
        # Find the task key in results
        task_key = None
        for key in results.keys():
            if key in ['wikitext', 'piqa', 'commonsense_qa', 'cqa']:
                task_key = key
                break
        
        if not task_key:
            # Fallback: try taking the first key
            if len(results) > 0:
                task_key = list(results.keys())[0]
            else:
                return 'No Results'

        res = results[task_key]
        
        # Try variations of metric name
        # e.g. 'acc,none', 'acc', 'word_perplexity,none'
        keys_to_try = [
            f"{metric_name},none",
            metric_name,
            f"{metric_name},stderr" # just in case, though unlikely for value
        ]
        
        for k in keys_to_try:
            if k in res:
                return res[k]
                
        return 'Metric Not Found'
    except Exception as e:
        return f"Error: {e}"

def main():
    base_dir = "diagnosis_qwen3_small"
    
    datasets = {
        "wikitext": "word_perplexity",
        "piqa": "acc",
        "commonsense_qa": "acc"
    }
    
    configs = [
        ("Baseline", "_baseline.json"),
        ("Batch 4", "_batch4.json"),
        ("Batch 8", "_batch8.json"),
        ("Attn Eager", "_attn_eager.json"),
        ("GEMM Det", "_gemm_det.json")
    ]

    for dataset, metric in datasets.items():
        print(f"\n=== Dataset: {dataset} (Metric: {metric}) ===")
        
        table_data = []
        baseline_val = None
        
        # Get Baseline First
        base_filename = f"{dataset}{configs[0][1]}"
        base_path = os.path.join(base_dir, base_filename)
        
        if os.path.exists(base_path):
            baseline_val = get_metric(base_path, metric)
        
        files_found = 0
        
        headers = ["Configuration", "Value", "Diff", "Status"]
        
        for config_name, suffix in configs:
            filename = f"{dataset}{suffix}"
            filepath = os.path.join(base_dir, filename)
            
            row = [config_name]
            
            if not os.path.exists(filepath):
                row.extend(["Missing", "-", "Not Run/Failed"])
                table_data.append(row)
                continue
            
            files_found += 1
            val = get_metric(filepath, metric)
            
            if isinstance(val, (int, float)):
                status = "OK"
                diff = "-"
                
                if config_name == "Baseline":
                    row.extend([f"{val:.6f}", "-", "Baseline"])
                else:
                    if isinstance(baseline_val, (int, float)):
                        if val == baseline_val:
                            diff = "0.0 (Identical)"
                            status = "MATCH"
                        else:
                            delta = val - baseline_val
                            diff = f"{delta:+.6f}"
                            status = "MISMATCH" if abs(delta) > 1e-9 else "Match (<1e-9)"
                    else:
                        diff = "No Base"
                        status = "?"
                    
                    row.extend([f"{val:.6f}", diff, status])
            else:
                row.extend([str(val), "-", "Error"])
            
            table_data.append(row)
            
        print(tabulate(table_data, headers=headers, floatfmt=".6f"))

if __name__ == "__main__":
    main()
