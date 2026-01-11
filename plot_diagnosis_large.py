
import json
import os
import matplotlib.pyplot as plt

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
    base_dir = "diagnosis_large"
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    datasets = {
        "commonsense_qa": {"metric": "acc", "label": "CommonsenseQA (acc)"},
        "piqa": {"metric": "acc", "label": "PIQA (acc)"}
    }

    plt.figure(figsize=(10, 6))
    
    for dataset_key, info in datasets.items():
        x = []
        y = []
        for bs in batch_sizes:
            filename = f"{dataset_key}_batch{bs}.json"
            filepath = os.path.join(base_dir, filename)
            
            if os.path.exists(filepath):
                val = get_metric(filepath, info["metric"])
                if val is not None:
                    x.append(bs)
                    y.append(val)
        
        if len(x) > 0:
            width = 0.35
            x_indices = range(len(x))
            offset = -width/2 if dataset_key == "commonsense_qa" else width/2
            
            bars = plt.bar([i + offset for i in x_indices], y, width=width, label=info["label"])
            plt.bar_label(bars, fmt='%.3f', padding=3, rotation=90 if len(batch_sizes) > 4 else 0)
            
    plt.xticks(range(len(batch_sizes)), labels=[str(b) for b in batch_sizes])
    plt.xlabel("Batch Size")
    plt.ylim(0.5, 0.85) # Adjust range for expected 8B accuracy
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Batch Size (RTX 5090, Qwen3-8B, N=1000)")
    plt.legend(loc='lower right')
    plt.grid(True, axis='y', ls="-", alpha=0.5)
    
    output_path = "batch_size_accuracy_plot_large.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
