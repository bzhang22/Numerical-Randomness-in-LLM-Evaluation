import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found. Please run pairwise_compare.py first.")
    return pd.read_csv(filepath)

def plot_precision_heatmap(df: pd.DataFrame, output_dir: str):
    """Generates a heatmap of exact match rates for precision comparisons."""
    precision_df = df[df['config_A'].isin(['float32', 'float16', 'bfloat16'])]
    if precision_df.empty:
        print("No precision data found for heatmap.")
        return
        
    summary = precision_df.groupby(['model', 'benchmark', 'config_A', 'config_B'])['exact_match'].mean().reset_index()
    summary['match_rate'] = summary['exact_match'] * 100
    
    for model in summary['model'].unique():
        model_data = summary[summary['model'] == model]
        pivot_data = model_data.pivot_table(
            index='config_A', 
            columns='config_B', 
            values='match_rate',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', fmt='.1f', vmin=0, vmax=100)
        plt.title(f'Precision Consistency Heatmap - {model}\n(Exact Match %)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'precision_heatmap_{model.split("/")[-1]}.png'))
        plt.close()

def plot_mismatch_rate_vs_batch(df: pd.DataFrame, output_dir: str):
    """Plots token mismatch rate against batch size."""
    # Assuming config_A is bs1 and config_B varies (bs2, bs4, bs8, bs16)
    batch_df = df[df['config_A'] == 'bs1']
    if batch_df.empty:
        print("No batch data found for mismatch rate plot.")
        return
        
    # Extract numerical batch size from config string (e.g. 'bs4' -> 4)
    batch_df['batch_size'] = batch_df['config_B'].str.extract(r'bs(\d+)').astype(float)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=batch_df, x='batch_size', y='token_mismatch_rate', hue='model', style='benchmark', markers=True)
    
    plt.xscale('log', base=2)
    plt.xticks([2, 4, 8, 16], ['BS=2', 'BS=4', 'BS=8', 'BS=16'])
    plt.ylabel('Token Mismatch Rate')
    plt.xlabel('Batch Size (vs BS=1 baseline)')
    plt.title('Token Mismatch Rate vs. Batch Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mismatch_rate_vs_batch.png'))
    plt.close()

def plot_first_divergence_dist(df: pd.DataFrame, output_dir: str):
    """Plots the distribution of the first divergence position."""
    div_df = df[df['first_divergence_pos'] >= 0]
    if div_df.empty:
        print("No divergence position data found.")
        return
        
    plt.figure(figsize=(10, 6))
    sns.histplot(data=div_df, x='first_divergence_pos', hue='model', bins=50, multiple='stack')
    plt.xlabel('First Divergence Position (Character Index)')
    plt.ylabel('Count')
    plt.title('Distribution of First Divergence Positions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'first_divergence_distribution.png'))
    plt.close()

def plot_attention_divergence(df: pd.DataFrame, output_dir: str):
    """Generates a bar plot of exact match rates for attention algorithm comparisons."""
    attention_pairs = [("eager", "sdpa"), ("eager", "flash_attention_2"), ("sdpa", "flash_attention_2")]
    
    attention_df = df[df.apply(lambda row: (row['config_A'], row['config_B']) in attention_pairs or (row['config_B'], row['config_A']) in attention_pairs, axis=1)]
    
    if attention_df.empty:
        print("No attention data found for plotting.")
        return
        
    summary = attention_df.groupby(['model', 'config_A', 'config_B'])['exact_match'].mean().reset_index()
    summary['match_rate'] = summary['exact_match'] * 100
    summary['comparison_label'] = summary['config_A'] + ' vs ' + summary['config_B']
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary, x='model', y='match_rate', hue='comparison_label')
    plt.title('Attention Implementation Consistency\n(Exact Match %)')
    plt.ylabel('Exact Match Rate (%)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_consistency_bar.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for LLM numerical randomness")
    parser.add_argument("--input_csv", type=str, default="pairwise_compare.csv")
    parser.add_argument("--output_dir", type=str, default="plots")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        df = load_data(args.input_csv)
        print(f"Loaded {len(df)} comparison records. Generating plots...")
        
        plot_precision_heatmap(df, args.output_dir)
        plot_mismatch_rate_vs_batch(df, args.output_dir)
        plot_first_divergence_dist(df, args.output_dir)
        plot_attention_divergence(df, args.output_dir)
        
        print(f"Plots saved to {args.output_dir}/")
        
    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()
