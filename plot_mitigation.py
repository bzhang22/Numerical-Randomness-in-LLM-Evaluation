import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob

RESULTS_DIR = "/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/results_mitigation"
TRACE_DIR = "/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/traces_mitigation"
PLOT_DIR = "/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/paper_plotting/mitigation_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Define color palette
VARIANT_COLORS = {
    "bf16_baseline": "tab:red",
    "attention": "tab:orange",
    "norm": "tab:green",
    "lm_head": "tab:blue",
    "attention_lm_head": "tab:purple",
    "fp32_reference": "gray"
}

def plot_m1(df):
    """Figure M1. Stability Recovery Relative to FP32"""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Aggregate across datasets/models to see raw trend
    agg_df = df.groupby('variant')['exact_match_vs_fp32'].mean().reset_index()
    # Ensure baseline is first
    order = ["bf16_baseline", "lm_head", "attention"]
    agg_df['variant_cat'] = pd.Categorical(agg_df['variant'], categories=order, ordered=True)
    agg_df = agg_df.sort_values('variant_cat').dropna()
    
    sns.barplot(data=agg_df, x='variant', y='exact_match_vs_fp32', palette=VARIANT_COLORS)
    plt.axhline(1.0, color='gray', linestyle='--', label='FP32 Reference (100%)')
    
    plt.title("Figure M1. Stability Recovery Relative to FP32\n(Cross-Model & Cross-Workload Average)")
    plt.ylabel("Exact Match Rate vs FP32")
    plt.xlabel("Mitigation Variant")
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "Figure_M1_Stability_Recovery.pdf"))
    plt.close()

def plot_m2(df):
    """Figure M2. Benchmark Score Recovery"""
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    order = ["bf16_baseline", "lm_head", "attention"]
    sns.barplot(data=df, x='dataset', y='benchmark_gap_vs_fp32', hue='variant', hue_order=order, palette=VARIANT_COLORS)
    plt.axhline(0.0, color='gray', linestyle='-', linewidth=2, label='FP32 Reference (0 Gap)')
    
    plt.title("Figure M2. Benchmark Score Deviation Recovery vs FP32")
    plt.ylabel("Accuracy Gap vs FP32 (%)")
    plt.xlabel("Workload")
    plt.legend(title="Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "Figure_M2_Benchmark_Recovery.pdf"))
    plt.close()

def plot_m3():
    """Figure M3. Layer-wise Hidden-State MAE"""
    trace_files = glob.glob(os.path.join(TRACE_DIR, "*_traces.json"))
    if not trace_files:
        print("No traces found for M3 yet.")
        return
        
    sns.set_theme(style="whitegrid")
    
    for tfile in trace_files:
        basename = os.path.basename(tfile).replace("_traces.json", "")
        parts = basename.rsplit("_", 1)
        if len(parts) == 2:
            model_name, dataset = parts
        else:
            model_name, dataset = basename, "unknown"
            
        with open(tfile, 'r') as f:
            tdata = json.load(f)
            
        all_trace_data = []
        for prompt_id, pdata in tdata.items():
            if not pdata.get("is_divergent", False): continue
            for variant, vdata in pdata.get("variants", {}).items():
                if variant not in ["bf16_baseline", "attention", "lm_head"]: continue
                if "layer_maes" not in vdata: continue
                maes = vdata["layer_maes"]
                for layer_idx, mae in enumerate(maes):
                    all_trace_data.append({
                        "Variant": variant,
                        "Layer": layer_idx,
                        "MAE": mae
                    })
                    
        if not all_trace_data: continue
        trace_df = pd.DataFrame(all_trace_data)
        
        VARIANT_MARKERS = {"bf16_baseline": "*", "attention": "s", "lm_head": "o"}
        VARIANT_DASHES = {"bf16_baseline": (4, 2), "attention": "", "lm_head": ""}
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=trace_df, x="Layer", y="MAE", hue="Variant", style="Variant", palette=VARIANT_COLORS, markers=VARIANT_MARKERS, dashes=VARIANT_DASHES, errorbar=None, markersize=9, linewidth=2.5)
        plt.yscale("log")
        plt.title(f"Figure M3. Layer-wise Hidden-State MAE vs FP32\nModel: {model_name} | Dataset: {dataset} (Divergent Cases)")
        plt.ylabel("Mean Absolute Error (Log Scale)")
        plt.xlabel("Transformer Layer Index")
        
        # Optionally, move the legend outside if it clutters the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        out_path = os.path.join(PLOT_DIR, f"Figure_M3_Layer_MAE_{model_name}_{dataset}.pdf")
        plt.savefig(out_path)
        plt.close()

def plot_m4(df):
    """Figure M4. Cost-Stability Tradeoff"""
    plt.figure(figsize=(8, 8))
    sns.set_theme(style="whitegrid")
    
    agg_df = df.groupby('variant')[['latency_overhead_percent', 'token_mismatch_rate']].mean().reset_index()
    agg_df = agg_df[agg_df['variant'].isin(["bf16_baseline", "attention", "lm_head"])]
    # Exclude baseline from overhead baseline calculation? Wait, baseline overhead is ~0.
    
    for i, row in agg_df.iterrows():
        plt.scatter(row['latency_overhead_percent'], row['token_mismatch_rate'], 
                    s=200, label=row['variant'], color=VARIANT_COLORS.get(row['variant'], 'black'))
        plt.text(row['latency_overhead_percent'] + 0.5, row['token_mismatch_rate'], row['variant'], fontsize=12)
        
    plt.title("Figure M4. Cost-Stability Tradeoff")
    plt.ylabel("Token Mismatch Rate vs FP32 (Lower is Better)")
    plt.xlabel("Latency Overhead % (Relative to BF16)")
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "Figure_M4_Cost_Tradeoff.pdf"))
    plt.close()

def main():
    csv_path = os.path.join(RESULTS_DIR, "mitigation_analysis_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Summary CSV not found at {csv_path}. Run analyze_mitigation.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    print("Plotting M1...")
    plot_m1(df)
    print("Plotting M2...")
    plot_m2(df)
    print("Plotting M4...")
    plot_m4(df)
    print("Plotting M3 Layer traces...")
    plot_m3()
    
    print(f"All Mitigation plots successfully dumped into {PLOT_DIR}.")

if __name__ == "__main__":
    main()
