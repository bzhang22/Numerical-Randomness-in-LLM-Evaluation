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
                target_attn = obj["target_attn"]
                layer_metrics = obj["layer_metrics"]
                
                for lm in layer_metrics:
                    data.append({
                        "Model": model,
                        "Dataset": obj["dataset"],
                        "Target Implementation": target_attn,
                        "Layer": lm["layer"],
                        "MAE": lm["mae"]
                    })
            except Exception as e:
                pass
    return pd.DataFrame(data)

def plot_layer_trends(df, out_path):
    sns.set_theme(style="whitegrid")
    
    agg_df = df.groupby(["Model", "Dataset", "Target Implementation", "Layer"])["MAE"].mean().reset_index()
    
    g = sns.relplot(
        data=agg_df, 
        kind="line",
        x="Layer", 
        y="MAE", 
        hue="Model", 
        style="Target Implementation",
        col="Dataset",
        col_wrap=2,
        markers=True, 
        linewidth=2.5,
        height=5, aspect=1.2
    )
    g.fig.suptitle("Eager vs Optimized Attention: Layer-wise Error Propagation", fontsize=16, fontweight='bold', y=1.02)
    g.set_axis_labels("Transformer Layer", "Mean Absolute Error")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved layer trends plot to {out_path}")

def plot_cdf_final_layer(df, out_path):
    sns.set_theme(style="whitegrid")
    
    final_layer_data = []
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        max_layer = model_df['Layer'].max()
        final_layer_data.append(model_df[model_df['Layer'] == max_layer])
        
    if not final_layer_data: return
    final_df = pd.concat(final_layer_data)
    
    # Save the table
    summary_table = final_df.groupby(["Model", "Dataset", "Target Implementation"])["MAE"].mean().reset_index()
    summary_table.to_csv("attention_layer_mae_table.csv", index=False)
    print("Saved attention_layer_mae_table.csv")
    
    g = sns.displot(
        data=final_df, 
        kind="ecdf",
        x="MAE", 
        hue="Model", 
        col="Dataset",
        row="Target Implementation",
        linewidth=2.5,
        height=4, aspect=1.5
    )
    g.fig.suptitle("CDF of Final Layer MAE: Flash Attention vs SDPA", fontsize=16, fontweight='bold', y=1.02)
    g.set_axis_labels("Final Layer Mean Absolute Error (MAE)", "Cumulative Probability")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved CDF plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="attention_layer_variance_results.jsonl")
    parser.add_argument("--out_trends", type=str, default="attention_layer_mae_trends.png")
    parser.add_argument("--out_cdf", type=str, default="attention_final_layer_cdf.png")
    args = parser.parse_args()
    
    df = parse_jsonl(args.data)
    if df.empty:
        print("Data is empty. Plots not generated.")
    else:
        plot_layer_trends(df, args.out_trends)
        plot_cdf_final_layer(df, args.out_cdf)
