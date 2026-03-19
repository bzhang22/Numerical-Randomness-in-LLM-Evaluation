import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import argparse
import seaborn as sns

def parse_logs(filepaths):
    data = []
    current_model = None
    current_ds = None
    current_dtype = None
    
    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Example Line: --- Evaluating Qwen/Qwen2.5-3B on commonsense_qa with [bfloat16] (Batch: 1 vs 8) ---
                    hdr_match = re.search(r'--- Evaluating (.*?) on (.*?) with \[(.*?)\]', line)
                    if hdr_match:
                        current_model = hdr_match.group(1).split('/')[-1]
                        current_ds = hdr_match.group(2)
                        current_dtype = hdr_match.group(3)
                        
                    # Example Line: Batch Size Flips (1 vs 8): 20 / 1000
                    flip_match = re.search(r'Batch Size Flips \(1 vs \d+\):\s+(\d+)\s+/\s+(\d+)', line)
                    if flip_match and current_model:
                        flips = int(flip_match.group(1))
                        total = int(flip_match.group(2))
                        flip_rate = (flips / total) * 100.0
                        
                        data.append({
                            'Model': current_model,
                            'Dataset': current_ds,
                            'Precision': current_dtype,
                            'Flip Rate (%)': flip_rate,
                            'Flips': flips,
                            'Total': total
                        })
        except FileNotFoundError:
            pass
            
    return pd.DataFrame(data)

def plot_variance(df, output_path):
    if df.empty:
        print("No valid tracking data found to plot for batch distributions.")
        return
        
    df.to_csv("batch_size_table.csv", index=False)
    print("Saved batch_size_table.csv")
    
    sns.set_theme(style="whitegrid")
    
    # We plot the flip rate across datasets for each Model-Precision combo
    g = sns.catplot(
        data=df, 
        kind="bar",
        x="Model", 
        y="Flip Rate (%)", 
        hue="Precision",
        col="Dataset",
        col_wrap=2,
        errorbar=None,
        palette=["#3498db", "#e74c3c", "#f1c40f"],
        height=6, aspect=1.2
    )
    
    g.fig.suptitle("Tokens Flipped by Sequence Padding (Batch=1 vs Batch=8)", fontsize=16, fontweight='bold', y=1.02)
    g.set_axis_labels("Model Architecture", "Token Flip Rate (%)")
    g.set_xticklabels(rotation=20, ha='right')
    
    # Add numerical overlay
    for ax in g.axes.flat:
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}%', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', 
                            xytext=(0, 9), 
                            textcoords='offset points',
                            fontsize=9)
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved batch variance plot targeting padding instabilities to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    import glob
    default_logs = glob.glob("/home/bohanzhang1/batch_variance*.log")
    
    parser.add_argument("--logs", type=str, nargs='+', default=default_logs)
    parser.add_argument("--out", type=str, default="batch_size_flips.png")
    args = parser.parse_args()
    
    df = parse_logs(args.logs)
    plot_variance(df, args.out)
