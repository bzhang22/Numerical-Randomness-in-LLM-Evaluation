
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
            if key in ['commonsense_qa']:
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

def plot_model(model_name, data_source_dir, file_prefix_pattern, output_filename, ylim=None):
    dataset = "commonsense_qa"
    metric = "acc"
    
    # Configs to plot
    configs = [
        ("baseline_b1", "Base B1", "grey"),
        ("eager_b1", "Eager B1", "red" if model_name == "0.5B" else "grey"), # Highlight fragile eager
        ("det_b1", "Det B1", "grey"),
        ("baseline_b32", "Base B32", "blue"),
        ("eager_b32", "Eager B32", "blue"),
        ("det_b32", "Det B32", "blue"),
    ]
    
    labels = []
    values = []
    colors = []
    
    for suffix, label, color in configs:
        # Construct filename based on pattern
        if file_prefix_pattern:
             filename = f"{file_prefix_pattern}_{dataset}_{suffix}.json"
        else:
             filename = f"{dataset}_{suffix}.json"
             
        filepath = os.path.join(data_source_dir, filename)
        
        val = 0
        if os.path.exists(filepath):
            v = get_metric(filepath, metric)
            if v is not None:
                val = v
        
        labels.append(label)
        values.append(val)
        colors.append(color)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors, width=0.6)
    plt.bar_label(bars, fmt='%.3f', padding=3)
    
    if ylim:
        plt.ylim(ylim)
    else:
        # Dynamic zoom
        if max(values) > 0:
            plt.ylim(min(values) * 0.98, max(values) * 1.01)

    plt.title(f"Variant robustness: {model_name} (CommonsenseQA)")
    plt.ylabel("Accuracy")
    plt.grid(True, axis='y', ls="--", alpha=0.5)
    
    plt.savefig(output_filename, dpi=150)
    print(f"Saved {output_filename}")
    plt.close()

def main():
    # 0.5B
    plot_model(
        "0.5B (Small)", 
        "diagnosis_vars_others", 
        "0.5B", 
        "vars_plot_0.5B.png",
        ylim=(0.56, 0.585)
    )
    
    # 3B
    plot_model(
        "3B (Medium)", 
        "diagnosis_vars_others", 
        "3B", 
        "vars_plot_3B.png",
        ylim=(0.775, 0.79)
    )
    
    # 8B
    plot_model(
        "8B (Large)", 
        "diagnosis_vars_large", 
        None, # No prefix in this dir
        "vars_plot_8B.png",
        ylim=(0.780, 0.795)
    )

if __name__ == "__main__":
    main()
