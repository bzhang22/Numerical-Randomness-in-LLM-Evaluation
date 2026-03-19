import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_all_traces(base_dir="trace"):
    data = []
    files = glob.glob(os.path.join(base_dir, "**", "trace_*.json"), recursive=True)
    for f in files:
        with open(f, 'r') as file:
            data.append(json.load(file))
    return data

def plot_p4_layerwise_mae(traces, output_dir="paper_plots"):
    """Figure P4: Maps the sequence mean absoluate error dynamically across Transformer blocks."""
    # Group traces physically by model
    model_map = {}
    for t in traces:
        if 'layerwise_mae' not in t or 'metadata' not in t:
            continue
        m = t['metadata']['model']
        if m not in model_map:
            model_map[m] = {"fp32_vs_fp16": [], "fp32_vs_bf16": []}
        model_map[m]["fp32_vs_fp16"].append(t['layerwise_mae']['fp32_vs_fp16'])
        model_map[m]["fp32_vs_bf16"].append(t['layerwise_mae']['fp32_vs_bf16'])
        
    for model, matrices in model_map.items():
        # Shape: (num_traces, num_layers) -> average across traces
        if not matrices["fp32_vs_fp16"]:
            continue
            
        fp16_arr = np.array(matrices["fp32_vs_fp16"]).mean(axis=0)
        bf16_arr = np.array(matrices["fp32_vs_bf16"]).mean(axis=0)
        layers = range(len(fp16_arr))
        
        plt.figure(figsize=(10, 6))
        plt.plot(layers, fp16_arr, marker='o', label='FP32 vs FP16')
        plt.plot(layers, bf16_arr, marker='s', label='FP32 vs BF16')
        
        plt.xlabel('Transformer Layer Index')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title(f'Layer-wise Hidden-State MAE\n({model} under Quantization)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'precision_layerwise_mae_{model}.pdf'))
        plt.close()
        
def plot_p5_token_flip_probability(traces, output_dir="paper_plots"):
    """Figure P5: Analyzes token instability exclusively grouped by baseline Logit margins."""
    margins = []
    fp16_flips = []
    bf16_flips = []
    
    for t in traces:
        if 'token_step_dynamics' not in t:
            continue
        m = t['token_step_dynamics']['fp32_top2_margin']
        f16 = t['token_step_dynamics']['flipped_in_fp16']
        b16 = t['token_step_dynamics']['flipped_in_bf16']
        # Filter negative margins if any (mathematical anomalies or zero logits)
        for val, flip16, flipb16 in zip(m, f16, b16):
            if val >= 0:
                margins.append(val)
                fp16_flips.append(flip16)
                bf16_flips.append(flipb16)
                
    df = pd.DataFrame({
        'margin': margins,
        'FP16_Flip': fp16_flips,
        'BF16_Flip': bf16_flips
    })
    
    if df.empty:
        return
        
    # Strictly define bucket ranges natively mapping precision
    bins = [-0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0]
    labels = ['[0, 0.5)', '[0.5, 1)', '[1, 2)', '[2, 4)', '[4, 8)', '8+']
    df['margin_group'] = pd.cut(df['margin'], bins=bins, labels=labels)
    
    grouped = df.groupby('margin_group').agg({
        'FP16_Flip': ['sum', 'count'],
        'BF16_Flip': ['sum', 'count']
    }).reset_index()
    
    # Process probabilities
    probs = []
    for _, row in grouped.iterrows():
        count = row[('FP16_Flip', 'count')]
        grp = row['margin_group'].iloc[0] if isinstance(row['margin_group'], pd.Series) else row['margin_group']
        if count == 0:
            probs.append({'group': grp, 'Precision Pair': 'FP32 vs FP16', 'Flip Probability (%)': 0.0})
            probs.append({'group': grp, 'Precision Pair': 'FP32 vs BF16', 'Flip Probability (%)': 0.0})
        else:
            p_fp16 = (row[('FP16_Flip', 'sum')] / count) * 100
            p_bf16 = (row[('BF16_Flip', 'sum')] / count) * 100
            probs.append({'group': grp, 'Precision Pair': 'FP32 vs FP16', 'Flip Probability (%)': p_fp16})
            probs.append({'group': grp, 'Precision Pair': 'FP32 vs BF16', 'Flip Probability (%)': p_bf16})
            
    prob_df = pd.DataFrame(probs)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=prob_df, x='group', y='Flip Probability (%)', hue='Precision Pair')
    plt.xlabel('Original FP32 Top-1 vs Top-2 Logit Margin Bucket')
    plt.ylabel('Token Flip Probability (%)')
    plt.title('Token Flip Probability vs Logit Margin')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_flip_prob_vs_margin.pdf'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, default="trace")
    parser.add_argument("--output_dir", type=str, default="paper_plots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    traces = load_all_traces(args.trace_dir)
    print(f"Loaded {len(traces)} deep evaluation traces.")
    
    if not traces:
        print("No mathematical traces found. Aborting plotting phase.")
        return
        
    print("Graphing P4: Matrix MAE per layer...")
    plot_p4_layerwise_mae(traces, args.output_dir)
    
    print("Graphing P5: Discretized Logit Flips...")
    plot_p5_token_flip_probability(traces, args.output_dir)
    
    print(f"SUCCESS: Layer-Mechanism graphs natively injected to {args.output_dir}/")

if __name__ == "__main__":
    main()
