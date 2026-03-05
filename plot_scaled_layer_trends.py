import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_layer_log(filepaths):
    # Dictionary structure: {dataset_name: {model_name: {prompt_idx: {layer_num: mae}}}}
    data = {}
    
    for filepath in filepaths:
        current_dataset = None
        current_model = None
        current_prompt = None
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Parse header: --- Evaluating Qwen/Qwen2.5-3B on commonsense_qa with [float16] ---
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
                            data[current_dataset][current_model] = {}
                    
                    # Match flipped prompt index
                    prompt_match = re.search(r'--- Flipped Prompt Index (\d+) ---', line)
                    if prompt_match and current_dataset and current_model:
                        current_prompt = int(prompt_match.group(1))
                        if current_prompt not in data[current_dataset][current_model]:
                            data[current_dataset][current_model][current_prompt] = {}
                        
                    # Match layer MAE
                    layer_match = re.search(r'Layer\s+(\d+)\s+\|\s+MAE:\s+([0-9\.e\+\-]+)', line)
                    if layer_match and current_dataset and current_model and current_prompt is not None:
                        layer = int(layer_match.group(1))
                        mae = float(layer_match.group(2))
                        data[current_dataset][current_model][current_prompt][layer] = mae
                        
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found.")
            
    return data

def plot_scaled_layer_variance(data, output_path):
    # Check what datasets we actually have data for
    datasets_with_data = [ds for ds, ds_data in data.items() if any(model_data for model_data in ds_data.values())]
    num_datasets = len(datasets_with_data)
    
    if num_datasets == 0:
        print("No valid tracking data found to plot for distributions.")
        return
        
    fig, axes = plt.subplots(1, num_datasets, figsize=(7 * num_datasets, 7), squeeze=False)
    axes = axes.flatten()
    
    colors = plt.cm.tab20.colors
    color_map = {}
    color_idx = 0
    
    for i, dataset_name in enumerate(datasets_with_data):
        ax = axes[i]
        ds_data = data[dataset_name]
        
        # Plot each prompt as a transparent line, and the mean as a thick line
        for model_name, prompts in ds_data.items():
            if not prompts: continue
            
            all_layers = set()
            for p_data in prompts.values():
                all_layers.update(p_data.keys())
            
            max_layer = max(all_layers) if all_layers else 0
            layers = range(max_layer + 1)
            
            # Accumulate sums to find the average
            sum_maes = np.zeros(max_layer + 1)
            counts = np.zeros(max_layer + 1)
            
            if model_name not in color_map:
                color_map[model_name] = colors[color_idx % len(colors)]
                color_idx += 1
            c = color_map[model_name]
            
            # Plot individual lines lightly
            for prompt_idx, layer_data in prompts.items():
                x = sorted(layer_data.keys())
                y = [layer_data[l] for l in x]
                ax.plot(x, y, color=c, alpha=0.15, linewidth=1)
                
                for l, val in zip(x, y):
                    sum_maes[l] += val
                    counts[l] += 1
                    
            # Calculate and plot the mean line
            mean_maes = np.divide(sum_maes, counts, out=np.zeros_like(sum_maes), where=counts!=0)
            ax.plot(layers, mean_maes, color=c, linewidth=3, label=model_name, marker='o', markersize=4)

        ax.set_yscale('log')
        ax.set_xlabel('Transformer Layer Depth', fontsize=12, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Mean Absolute Error (Log Scale)', fontsize=12, fontweight='bold')
        
        # Make dataset name presentable
        title_ds = dataset_name.upper() if dataset_name in ["piqa", "cmmlu"] else dataset_name.replace('_', ' ').title()
        ax.set_title(f'Hidden State Divergence\n{title_ds} Prompts', fontsize=14, fontweight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(fontsize=9, loc='lower right')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved scaled layer MAE variance plot to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=str, nargs='+', default=[
        "/home/bohanzhang1/scaled_precision_layer_small.log",
        "/home/bohanzhang1/scaled_precision_layer_large.log"
    ])
    args = parser.parse_args()
    
    data = parse_layer_log(args.logs)
    output_path = "/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/scaled_precision_layer_trends.png"
    plot_scaled_layer_variance(data, output_path)

if __name__ == "__main__":
    main()
