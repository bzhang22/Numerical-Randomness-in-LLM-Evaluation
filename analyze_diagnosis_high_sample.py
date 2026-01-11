
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
    base_dir = "diagnosis_high_sample"
    
    datasets = {
        "commonsense_qa": "acc",
        "piqa": "acc"
    }
    
    batch_sizes = [1, 2, 4, 8, 16, 32]

    for dataset, metric in datasets.items():
        print(f"\n=== Dataset: {dataset} (Metric: {metric}, N=1000) ===")
        
        table_data = []
        baseline_val = None
        
        # Get Batch 1 as Baseline
        base_filename = f"{dataset}_batch1.json"
        base_path = os.path.join(base_dir, base_filename)
        
        if os.path.exists(base_path):
            baseline_val = get_metric(base_path, metric)
        
        headers = ["Batch Size", "Accuracy", "Diff from Batch 1", "Status"]
        
        for bs in batch_sizes:
            filename = f"{dataset}_batch{bs}.json"
            filepath = os.path.join(base_dir, filename)
            
            row = [str(bs)]
            
            if not os.path.exists(filepath):
                row.extend(["Missing", "-", "Pending/Failed"])
                table_data.append(row)
                continue
            
            val = get_metric(filepath, metric)
            
            if isinstance(val, (int, float)):
                status = "OK"
                diff = "-"
                
                if bs == 1:
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
