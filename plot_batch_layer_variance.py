import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import os

def parse_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        return pd.DataFrame()
        
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                model = obj["model"]
                layer_metrics = obj["layer_metrics"]
                
                for lm in layer_metrics:
                    data.append({
                        "Model": model,
                        "Dataset": obj["dataset"],
                        "Prompt": obj["prompt_idx"],
                        "Layer": lm["layer"],
                        "MAE": lm["mae"]
                    })
            except Exception as e:
                pass
    return pd.DataFrame(data)

def plot_layer_trends(df, out_path):
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Calculate mean MAE per layer per model
    agg_df = df.groupby(["Model", "Layer"])["MAE"].mean().reset_index()
    
    sns.lineplot(data=agg_df, x="Layer", y="MAE", hue="Model", marker="o", linewidth=2.5)
    plt.title("Per-Layer Error Propagation (Batch=1 vs Batch=8)", fontsize=15, fontweight='bold')
    plt.xlabel("Transformer Layer", fontsize=12)
    plt.ylabel("Mean Absolute Error", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved layer trends plot to {out_path}")

def plot_cdf_final_layer(df, out_path):
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Extract only the final layer for each model
    final_layer_data = []
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        max_layer = model_df['Layer'].max()
        final_layer_data.append(model_df[model_df['Layer'] == max_layer])
        
    if not final_layer_data: return
    final_df = pd.concat(final_layer_data)
    
    sns.ecdfplot(data=final_df, x="MAE", hue="Model", linewidth=2.5)
    plt.title("CDF of Final Layer Token MAE Divergence", fontsize=15, fontweight='bold')
    plt.xlabel("Final Layer Mean Absolute Error (MAE)", fontsize=12)
    plt.ylabel("Cumulative Probability", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved CDF plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/batch_layer_variance_results.jsonl")
    parser.add_argument("--out_trends", type=str, default="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/batch_layer_mae_trends.png")
    parser.add_argument("--out_cdf", type=str, default="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/batch_final_layer_cdf.png")
    args = parser.parse_args()
    
    df = parse_jsonl(args.data)
    if df.empty:
        print("Data is empty. Plots not generated.")
    else:
        plot_layer_trends(df, args.out_trends)
        plot_cdf_final_layer(df, args.out_cdf)
