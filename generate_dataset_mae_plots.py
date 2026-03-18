import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

def gather_dataset_trace_data(trace_dir: str):
    mae_records = []
    if not os.path.exists(trace_dir):
        return pd.DataFrame()
        
    for root, dirs, files in os.walk(trace_dir):
        for file in files:
            if not file.endswith('.json'): continue
            path = os.path.join(root, file)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                meta = data.get('metadata', {})
                model = meta.get('model', 'Unknown')
                benchmark = meta.get('benchmark', 'Unknown')
                
                # ONLY plot divergence cases!
                if not meta.get('is_divergent', False):
                    continue
                
                mae = data.get('layerwise_mae', {})
                fp32_fp16 = mae.get('fp32_vs_fp16', [])
                fp32_bf16 = mae.get('fp32_vs_bf16', [])
                
                for layer_idx, error in enumerate(fp32_fp16):
                    if error > 0: # Log scale requires positive values
                        mae_records.append({'model': model, 'benchmark': benchmark, 'layer': layer_idx, 'precision_pair': 'FP32 vs FP16', 'mae': error})
                for layer_idx, error in enumerate(fp32_bf16):
                    if error > 0:
                        mae_records.append({'model': model, 'benchmark': benchmark, 'layer': layer_idx, 'precision_pair': 'FP32 vs BF16', 'mae': error})
                    
            except Exception as e:
                print(f"Error parsing {path}: {e}")
                continue
    return pd.DataFrame(mae_records)

def main():
    output_dir = "paper_plots/dataset_mae_trends_log"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Gathering STRICTLY Divergent cross-model dataset MAE traces for Log Scale...")
    mae_df = gather_dataset_trace_data("trace")
    if mae_df.empty:
        print("No MAE data found.")
        return
        
    # Standardize benchmark names
    mae_df['benchmark'] = mae_df['benchmark'].str.split('_').str[0]
    
    datasets = mae_df['benchmark'].unique()
    pairs = mae_df['precision_pair'].unique()
    
    for dataset in datasets:
        df_ds = mae_df[mae_df['benchmark'] == dataset]
        
        for pair in pairs:
            df_pair = df_ds[df_ds['precision_pair'] == pair]
            if df_pair.empty: continue
            
            plt.figure(figsize=(12, 7))
            sns.lineplot(
                data=df_pair, 
                x='layer', 
                y='mae', 
                hue='model', 
                style='model',
                markers=True, 
                dashes=False,
                errorbar=None, 
                linewidth=2,
                alpha=0.8
            )
            
            # Apply Log Scale for Error!
            plt.yscale("log")
            
            plt.title(f'[{dataset.upper()}] DIVERGENT Only Log-Scale Layer-wise Error Trend: {pair}')
            plt.xlabel('Transformer Layer Depth (Absolute)')
            plt.ylabel('Log Mean Absolute Error (MAE) vs FP32 Baseline')
            
            # Place legend outside
            plt.legend(title='Model Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()
            
            safe_pair = pair.replace(" ", "_")
            out_file = os.path.join(output_dir, f"{dataset}_{safe_pair}_log_trend.pdf")
            plt.savefig(out_file, bbox_inches='tight')
            plt.close()
            print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
