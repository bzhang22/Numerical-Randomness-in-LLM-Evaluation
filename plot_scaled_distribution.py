import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def parse_confidence_log(filepaths):
    # data[dataset][model_dtype] = {'eager_margins': [], 'sdpa_margins': []}
    data = {}
    current_model = None
    current_dataset = None
    
    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    hdr_match = re.search(r'--- Evaluating (.*?) on (.*?) with \[(.*?)\] ---', line)
                    if hdr_match:
                        model = hdr_match.group(1).split('/')[-1]
                        dataset = hdr_match.group(2)
                        dtype = hdr_match.group(3)
                        current_dataset = dataset
                        current_model = f"{model} ({dtype})"
                        
                        if current_dataset not in data:
                            data[current_dataset] = {}
                        if current_model not in data[current_dataset]:
                            data[current_dataset][current_model] = {'eager_margins': [], 'sdpa_margins': []}
                            
                    margin_match = re.search(r'Eager Margin Between Tokens: ([0-9\.\-nan]+) \| SDPA Margin Between Tokens: ([0-9\.\-nan]+)', line)
                    if margin_match and current_model and current_dataset:
                        eager_m = margin_match.group(1)
                        sdpa_m = margin_match.group(2)
                        if eager_m.lower() != 'nan' and sdpa_m.lower() != 'nan':
                            data[current_dataset][current_model]['eager_margins'].append(abs(float(eager_m)))
                            data[current_dataset][current_model]['sdpa_margins'].append(abs(float(sdpa_m)))
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found.")
            
    return data

def plot_confidence_cdf(data, output_path):
    # Ensure we include all datasets that have at least 1 flip
    datasets_with_data = [ds for ds, ds_data in data.items() if any(len(m['eager_margins']) > 0 for m in ds_data.values())]
    num_datasets = len(datasets_with_data)
    
    if num_datasets == 0:
        print("No valid tracking data found to plot for distributions.")
        return
        
    fig, axes = plt.subplots(1, num_datasets, figsize=(8 * num_datasets, 7), squeeze=False)
    axes = axes.flatten()
    
    colors = plt.cm.tab20.colors
    color_map = {}
    color_idx = 0
    
    for i, ds in enumerate(datasets_with_data):
        ax = axes[i]
        ds_data = data[ds]
        
        for model_name, m_data in ds_data.items():
            margins = np.array(m_data['eager_margins'])
            if len(margins) == 0: continue
                
            if model_name not in color_map:
                color_map[model_name] = colors[color_idx % len(colors)]
                color_idx += 1
            c = color_map[model_name]
            
            margins = np.sort(margins)
            p = 1.0 * np.arange(len(margins)) / (len(margins) - 1) if len(margins) > 1 else np.array([1.0])
            
            # Use markers for very few points, otherwise normal line
            marker_type = 'o' if len(margins) < 10 else None
            ax.plot(margins, p, label=f"{model_name} (N={len(margins)})", color=c, linewidth=2, marker=marker_type, markersize=5)
            
        title_ds = ds.upper() if ds in ["piqa", "cmmlu"] else ds.replace('_', ' ').title()
        ax.set_title(f"Flips Confidence Margins: {title_ds}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Confidence Difference between Top 2 Answers", fontsize=12)
        if i == 0:
            ax.set_ylabel("Cumulative Proportion of Flips", fontsize=12)
        ax.set_xlim(xmin=0)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=9, loc="lower right")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved scaled distribution plot to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=str, nargs='+', default=[
        "/home/bohanzhang1/scaled_precision_confidence_small.log",
        "/home/bohanzhang1/scaled_precision_confidence_large.log"
    ])
    args = parser.parse_args()
    
    data = parse_confidence_log(args.logs)
    output_path = "/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/experiment_results/plots/scaled_precision_flip_cdf.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_confidence_cdf(data, output_path)

if __name__ == "__main__":
    main()
