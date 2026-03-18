import sys
print("=== PYTHON EXECUTABLE STARTED ===", flush=True)

import matplotlib
matplotlib.use('Agg')

import os
print("Imported os", flush=True)
import json
import pandas as pd
print("Imported pandas", flush=True)
import seaborn as sns
print("Imported seaborn", flush=True)
import matplotlib.pyplot as plt
print("Imported matplotlib", flush=True)
import numpy as np
print("Imported numpy", flush=True)

def gather_dataset_trace_data(trace_dir: str):
    mae_records = []
    if not os.path.exists(trace_dir):
        print(f"Directory {trace_dir} not found!", flush=True)
        return pd.DataFrame()
        
    print(f"Walking directory {trace_dir}...", flush=True)
    count_files = 0
    for root, dirs, files in os.walk(trace_dir):
        for file in files:
            if not file.endswith('.json'): continue
            count_files += 1
            if count_files % 100 == 0:
                print(f"Searched {count_files} JSON files so far...", flush=True)
            path = os.path.join(root, file)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                meta = data.get('metadata', {})
                model = meta.get('model', 'Unknown')
                benchmark = meta.get('benchmark', 'Unknown')
                prompt_id = meta.get('prompt_id', 0)
                
                # ONLY plot divergence cases!
                if not meta.get('is_divergent', False):
                    continue
                
                mae = data.get('layerwise_mae', {})
                fp32_fp16 = mae.get('fp32_vs_fp16', [])
                fp32_bf16 = mae.get('fp32_vs_bf16', [])
                
                for layer_idx, error in enumerate(fp32_fp16):
                    if error > 0:
                        mae_records.append({'model': model, 'benchmark': benchmark, 'layer': layer_idx, 'precision_pair': 'FP16', 'log10_mae': float(np.log10(error)), 'prompt_id': prompt_id})
                for layer_idx, error in enumerate(fp32_bf16):
                    if error > 0:
                        mae_records.append({'model': model, 'benchmark': benchmark, 'layer': layer_idx, 'precision_pair': 'BF16', 'log10_mae': float(np.log10(error)), 'prompt_id': prompt_id})
                    
            except Exception as e:
                print(f"Error parsing {path}: {e}", flush=True)
                continue
    
    print(f"Total valid JSON traces parsed: {count_files}", flush=True)
    return pd.DataFrame(mae_records)

def main():
    print("Starting generator script...", flush=True)
    output_dir = "paper_plots/dataset_mae_trends"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Gathering STRICTLY Divergent cross-model dataset MAE traces...", flush=True)
    mae_df = gather_dataset_trace_data("trace")
    if mae_df.empty:
        print("No MAE data found.", flush=True)
        return
        
    print(f"Data gathered successfully! Total points to plot: {len(mae_df)}", flush=True)
    
    # Standardize benchmark names
    mae_df['benchmark'] = mae_df['benchmark'].str.split('_').str[0]
    
    datasets = mae_df['benchmark'].unique()
    print(f"Found datasets to plot: {datasets}", flush=True)
    
    for dataset in datasets:
        print(f"Processing dataset: {dataset}...", flush=True)
        df_ds = mae_df[mae_df['benchmark'] == dataset].copy()
        if df_ds.empty: continue
        
        plt.figure(figsize=(14, 8))
        
        df_ds['unit_id'] = df_ds['model'] + "_" + df_ds['precision_pair'] + "_" + df_ds['prompt_id'].astype(str)
        
        print(f"  Drawing lineplot for {dataset}...", flush=True)
        sns.lineplot(
            data=df_ds, 
            x='layer', 
            y='log10_mae', 
            hue='model', 
            style='precision_pair',
            units='unit_id',
            estimator=None,
            alpha=0.6,
            linewidth=1.5
        )
        
        print(f"  Decorating plot for {dataset}...", flush=True)
        plt.title(f'[{dataset.upper()}] DIVERGENT Cases: Individual Log10 MAE Trends (BF16 & FP16 vs FP32)')
        plt.xlabel('Transformer Layer Depth (Absolute)')
        plt.ylabel('Log10(Mean Absolute Error) vs FP32 Baseline')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        filtered_handles, filtered_labels = [], []
        for h, l in zip(handles, labels):
            if "prompt_id" not in l and "unit_id" not in l and not "_" in l:
                filtered_handles.append(h)
                filtered_labels.append(l)
            if l in df_ds['model'].unique() or l in df_ds['precision_pair'].unique() or l in ["model", "precision_pair"]:
                filtered_handles.append(h)
                filtered_labels.append(l)

        plt.legend(filtered_handles, filtered_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        
        out_file = os.path.join(output_dir, f"{dataset}_combined_trend.pdf")
        
        print(f"  Saving to PDF: {out_file} (this can take a moment)...", flush=True)
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_file}", flush=True)
        
    print("All done!", flush=True)

if __name__ == "__main__":
    main()
