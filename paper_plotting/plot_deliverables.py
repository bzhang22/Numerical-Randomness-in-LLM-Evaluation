import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from collections import defaultdict

def parse_core_mae(filepath):
    """Parses JSONL from core_layer_mae.py"""
    records = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                records.append(json.loads(line))
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    return records

def parse_clamp(filepath):
    """Parses JSONL from intervention_experiments.py"""
    records = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                records.append(json.loads(line))
    except Exception as e:
        pass
    return records

def plot_layerwise_mae(all_records, labels, output_path):
    """
    Plots Layer-wise MAE (log scale) for different precision/models.
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    for records, label in zip(all_records, labels):
        if not records: continue
        
        # Aggregate MAE across prompts for each layer
        layer_maes = defaultdict(list)
        for r in records:
            for lm in r['layer_metrics']:
                layer_maes[lm['layer']].append(lm['mae'])
                
        layers = sorted(list(layer_maes.keys()))
        avg_maes = [sum(layer_maes[l])/len(layer_maes[l]) for l in layers]
        
        plt.plot(layers, avg_maes, marker='o', label=label, linewidth=2, markersize=4)

    plt.yscale('log')
    plt.title("Layer-wise Hidden State MAE (Eager vs SDPA)", fontsize=14, fontweight='bold')
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Mean Absolute Error (log scale)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

def plot_flip_vs_nonflip_mae(records, output_path):
    """
    Splits prompts into flip and non-flip groups and plots their MAE over layers.
    """
    if not records: return
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    flip_maes = defaultdict(list)
    non_flip_maes = defaultdict(list)
    
    num_flips = 0
    num_non_flips = 0

    for r in records:
        if r['flip']:
            num_flips += 1
            for lm in r['layer_metrics']: flip_maes[lm['layer']].append(lm['mae'])
        else:
            num_non_flips += 1
            for lm in r['layer_metrics']: non_flip_maes[lm['layer']].append(lm['mae'])

    layers = sorted(list(flip_maes.keys())) if flip_maes else sorted(list(non_flip_maes.keys()))
    
    if flip_maes:
        avg_flip = [sum(flip_maes[l])/len(flip_maes[l]) for l in layers]
        plt.plot(layers, avg_flip, marker='o', color='red', label=f'Flipped (n={num_flips})', linewidth=2)
        
    if non_flip_maes:
        avg_non = [sum(non_flip_maes[l])/len(non_flip_maes[l]) for l in layers]
        plt.plot(layers, avg_non, marker='s', color='blue', label=f'Stable (n={num_non_flips})', linewidth=2, linestyle='--')

    plt.yscale('log')
    plt.title("MAE Trajectory: Token Flip vs Stable Groups", fontsize=14, fontweight='bold')
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Mean Absolute Error (log scale)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

def plot_clamp_intervention(clamp_records, output_path):
    """
    Plots intervention layer `k` vs Max MAE post `k`. Shows how clamping early reduces downstream errors.
    """
    if not clamp_records: return
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Group by layer_k
    k_mae = defaultdict(list)
    for r in clamp_records:
        k_mae[r['layer_k']].append(r['max_mae_post_k'])
        
    layers = sorted(list(k_mae.keys()))
    avg_mae = [sum(k_mae[l])/len(k_mae[l]) for l in layers]
    
    plt.plot(layers, avg_mae, marker='D', color='green', linewidth=2, markersize=8)
    plt.yscale('log')
    plt.title("Impact of SDPA Clamping Intervention", fontsize=14, fontweight='bold')
    plt.xlabel("Intervention Layer (k: replacing SDPA output with Eager target)", fontsize=12)
    plt.ylabel("Maximum MAE in Layers > k (log scale)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core_logs", type=str, nargs='+', help="JSONLs from core_layer_mae.py")
    parser.add_argument("--core_labels", type=str, nargs='+', help="Labels for the core logs")
    parser.add_argument("--clamp_logs", type=str, nargs='+', help="JSONLs from intervention_experiments.py")
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()
    
    if args.core_logs:
        records_list = [parse_core_mae(log) for log in args.core_logs]
        plot_layerwise_mae(records_list, args.core_labels, f"{args.out_dir}/1_layer_mae_comparison.png")
        # Plot flip vs non-flip for the first log
        plot_flip_vs_nonflip_mae(records_list[0], f"{args.out_dir}/2_flip_vs_nonflip_mae.png")
        
    if args.clamp_logs:
        # Combine all clamp logs (assuming they are scans from same model/dataset)
        all_clamp = []
        for log in args.clamp_logs:
            all_clamp.extend(parse_clamp(log))
        plot_clamp_intervention(all_clamp, f"{args.out_dir}/4_clamp_intervention.png")
