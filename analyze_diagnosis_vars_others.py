
import json
import os
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
        
        if not task_key and len(results) > 0:
             task_key = list(results.keys())[0]

        if not task_key: return None

        res = results[task_key]
        
        keys_to_try = [f"{metric_name},none", metric_name]
        for k in keys_to_try:
            if k in res:
                return res[k]
        return None
    except:
        return None

def main():
    base_dir = "diagnosis_vars_others"
    datasets = {
        "commonsense_qa": "acc",
        "piqa": "acc"
    }
    
    models = [("0.5B", "0.5B (Small)"), ("3B", "3B (Medium)")]

    configs = [
        ("baseline_b1", "Baseline (Batch 1)"),
        ("eager_b1", "Eager Attn (Batch 1)"),
        ("det_b1", "Deterministic (Batch 1)"),
        ("baseline_b32", "Baseline (Batch 32)"),
        ("eager_b32", "Eager Attn (Batch 32)"),
        ("det_b32", "Deterministic (Batch 32)"),
    ]

    for model_prefix, model_label in models:
        print(f"\n==========================================")
        print(f"Model: {model_label}")
        print(f"==========================================")

        for dataset, metric in datasets.items():
            print(f"\n--- Dataset: {dataset} (Metric: {metric}) ---")
            table_data = []
            
            # Get control value (Baseline Batch 1)
            base_path = os.path.join(base_dir, f"{model_prefix}_{dataset}_baseline_b1.json")
            baseline_val = None
            if os.path.exists(base_path):
                 baseline_val = get_metric(base_path, metric)

            for suffix, label in configs:
                filename = f"{model_prefix}_{dataset}_{suffix}.json"
                filepath = os.path.join(base_dir, filename)
                
                row = [label]
                if os.path.exists(filepath):
                    val = get_metric(filepath, metric)
                    if val is not None:
                        row.append(f"{val:.6f}")
                        
                        if baseline_val is not None:
                            if val == baseline_val:
                                row.append("0.0 (Identical)")
                            else:
                                diff = val - baseline_val
                                row.append(f"{diff:+.6f}")
                        else:
                            row.append("-")
                    else:
                        row.extend(["Error", "-"])
                else:
                    row.extend(["Pending", "-"])
                
                table_data.append(row)
            
            print(tabulate(table_data, headers=["Configuration", "Accuracy", "Diff from Base"], floatfmt=".6f"))

if __name__ == "__main__":
    main()
