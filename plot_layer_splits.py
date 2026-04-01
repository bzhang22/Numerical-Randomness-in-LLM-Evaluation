import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import os
import glob

def parse_mitigation_results(results_dir):
    data = []
    # Using _meta.json handles memory constraints and skips massive jsonl processing
    files = glob.glob(os.path.join(results_dir, "*_*_*_meta.json"))
    
    for filepath in files:
        if not os.path.exists(filepath): continue
        with open(filepath, 'r') as f:
            try:
                obj = json.load(f)
                model = obj.get("model", "").split("/")[-1]
                dataset = obj.get("dataset", "")
                variant = obj.get("variant", "")
                layer_split = obj.get("layer_split", "all")
                accuracy = obj.get("accuracy", 0.0)
                
                data.append({
                    "Model": model,
                    "Dataset": dataset,
                    "Variant": variant,
                    "Layer Region": layer_split,
                    "Accuracy": accuracy
                })
            except Exception as e:
                pass
    return pd.DataFrame(data)

def plot_layer_ablation(df, out_path):
    sns.set_theme(style="whitegrid")
    
    # No need to groupby mean if data is already aggregated from meta.json
    agg_df = df.copy()
    
    # Define an order for the x-axis to make it logical
    split_order = ["first_quarter", "first_half", "middle", "last_half", "last_quarter", "1", "1,-1", "all"]
    # Filter only regions actually present
    present_orders = [o for o in split_order if o in agg_df["Layer Region"].unique()]
    # Add any unexpected regions at the end
    for r in agg_df["Layer Region"].unique():
        if r not in present_orders:
            present_orders.append(r)
            
    # Relabel 'all' as 'Full Transformer Pipeline' for clarity if preferred
    
    g = sns.catplot(
        data=agg_df, 
        kind="bar",
        x="Layer Region", 
        y="Accuracy", 
        hue="Variant", 
        col="Model", 
        row="Dataset",
        order=present_orders,
        palette="mako",
        height=4, aspect=1.5,
        sharey=False
    )
    g.fig.suptitle("Impact of Localized Layer Upgrades (FP32) on Final Accuracy", fontsize=16, fontweight='bold', y=1.02)
    g.set_axis_labels("Mitigated Transformer Region", "Accuracy")
    g.set_xticklabels(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved layer ablation plot to {out_path}")
    
    # Save accompanying table
    table_path = out_path.replace(".png", ".csv")
    agg_df.to_csv(table_path, index=False)
    print(f"Saved layer ablation table to {table_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/results_mitigation")
    parser.add_argument("--out_plot", type=str, default="layer_precision_ablation.png")
    args = parser.parse_args()
    
    df = parse_mitigation_results(args.results_dir)
    
    if df.empty:
        print(f"No jsonl data found in {args.results_dir}.")
    else:
        # Filter dataframe for ONLY our new layer_split rows if possible
        # Some older files won't have 'layer_split' in JSON, so they default to 'all'.
        plot_layer_ablation(df, args.out_plot)
