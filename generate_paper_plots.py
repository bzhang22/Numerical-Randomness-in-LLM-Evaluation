import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import json
import glob
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found. Please run pairwise_compare.py first.")
    return pd.read_csv(filepath)

def extract_benchmark_scores(base_dir: str) -> pd.DataFrame:
    """Parses results_{timestamp}.json to extract accuracy/exact_match/pass@1 scores."""
    scores = []
    models = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    for model in models:
        for dtype in ["float32", "float16", "bfloat16"]:
            model_dir = os.path.join(base_dir, model, "precision", dtype)
            if not os.path.exists(model_dir):
                continue
                
            # Internal subdirectories created by lm-eval
            for subdir in os.listdir(model_dir):
                sub_path = os.path.join(model_dir, subdir)
                if not os.path.isdir(sub_path):
                    continue
                    
                json_files = glob.glob(os.path.join(sub_path, "results_*.json"))
                if not json_files:
                    continue
                    
                latest_json = sorted(json_files)[-1]
                with open(latest_json, 'r') as f:
                    try:
                        data = json.load(f)
                        res = data.get("results", {})
                    except Exception:
                        continue
                        
                    for benchmark, metrics in res.items():
                        # Extract the main metric
                        score = None
                        if "exact_match,strict-match" in metrics:
                            score = metrics["exact_match,strict-match"]
                        elif "pass@1,create_test" in metrics:
                            score = metrics["pass@1,create_test"]
                        elif "acc,none" in metrics:
                            score = metrics["acc,none"]
                        elif "word_perplexity,none" in metrics:
                            score = metrics["word_perplexity,none"] # For wikitext
                        
                        if score is not None:
                            # Normalize percentage metrics
                            if benchmark not in ["wikitext", "lambada_openai"]:
                                score *= 100
                            
                            scores.append({
                                "model": model,
                                "benchmark": benchmark.split("_")[0] if "cmmlu" in benchmark else benchmark, 
                                "precision": dtype.upper().replace("FLOAT", "FP").replace("BFP", "BF"),
                                "score": score
                            })
                            
    return pd.DataFrame(scores).groupby(["model", "benchmark", "precision"]).mean().reset_index()

def plot_p1_precision_heatmap(df: pd.DataFrame, output_dir: str):
    """Figure P1: Precision Pairwise Consistency Heatmap (Exact Match)"""
    # Filter only precision combinations
    prec_pairs = [("float32", "float16"), ("float32", "bfloat16"), ("float16", "bfloat16")]
    valid_pairs = []
    for a, b in prec_pairs:
        valid_pairs.append((a, b))
        valid_pairs.append((b, a))
        
    mask = df.apply(lambda row: (row['config_A'], row['config_B']) in valid_pairs, axis=1)
    prec_df = df[mask].copy()
    
    if prec_df.empty:
        print("No precision data found for P1.")
        return

    # Standardize pair naming (always A vs B)
    def standardize_pair(a, b):
        pairs_map = {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16"}
        items = sorted([pairs_map.get(a, a), pairs_map.get(b, b)])
        return tuple(items)

    prec_df['prec_A'] = prec_df.apply(lambda r: standardize_pair(r['config_A'], r['config_B'])[0], axis=1)
    prec_df['prec_B'] = prec_df.apply(lambda r: standardize_pair(r['config_A'], r['config_B'])[1], axis=1)
    
    summary = prec_df.groupby(['model', 'benchmark', 'prec_A', 'prec_B'])['exact_match'].mean().reset_index()
    summary['match_rate'] = summary['exact_match'] * 100
    
    # Version A: Map individual models
    models = summary['model'].unique()
    for model in models:
        mdl_data = summary[summary['model'] == model]
        pivot_data = mdl_data.groupby(['prec_A', 'prec_B'])['match_rate'].mean().unstack()
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0, vmax=100)
        plt.title(f'{model} - Precision Consistency (%)\nTarget: Temperature=0')
        plt.xlabel('Precision')
        plt.ylabel('Precision')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'precision_consistency_heatmap_{model}.pdf'))
        plt.close()
        
    # Version B: Overall aggregated heatmap
    overall = summary.groupby(['prec_A', 'prec_B'])['match_rate'].mean().unstack()
    plt.figure(figsize=(6, 5))
    sns.heatmap(overall, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0, vmax=100)
    plt.title('Overall Aggregated Precision Consistency (%)\nTarget: Temperature=0')
    plt.xlabel('Precision')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_consistency_heatmap_overall.pdf'))
    plt.close()

def plot_p2_mismatch_rate(df: pd.DataFrame, output_dir: str):
    """Figure P2: Precision-Induced Mismatch Rate (Per Dataset, Per Model, Per Precision)"""
    prec_pairs = [("float32", "float16"), ("float32", "bfloat16"), ("float16", "bfloat16")]
    valid_pairs = []
    for a, b in prec_pairs:
        valid_pairs.append((a, b))
        valid_pairs.append((b, a))
        
    mask = df.apply(lambda row: (row['config_A'], row['config_B']) in valid_pairs, axis=1)
    prec_df = df[mask].copy()
    
    if prec_df.empty:
        return
        
    prec_df['sequence_mismatch_rate'] = (1.0 - prec_df['exact_match'].astype(float)) * 100
    
    def get_color_pair(row):
        mapping = {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16"}
        a, b = row['config_A'], row['config_B']
        if mapping[a] == "FP32" and mapping[b] == "FP16" or mapping[b] == "FP32" and mapping[a] == "FP16":
            return "FP32 vs FP16"
        if mapping[a] == "FP32" and mapping[b] == "BF16" or mapping[b] == "FP32" and mapping[a] == "BF16":
            return "FP32 vs BF16"
        return "FP16 vs BF16"

    prec_df['pair'] = prec_df.apply(get_color_pair, axis=1)
    
    # Comprehensive Grid: per model (col), per dataset (x), per precision (hue)
    g = sns.catplot(
        data=prec_df, kind="bar",
        x="benchmark", y="sequence_mismatch_rate", hue="pair", col="model",
        col_wrap=3, height=4, aspect=1.2, errorbar=None
    )
    
    g.set_axis_labels("Benchmark Dataset", "Sequence Mismatch Rate (%)")
    g.set_titles("Model: {col_name}")
    
    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            
    plt.subplots_adjust(top=0.9, hspace=0.4, bottom=0.15)
    g.figure.suptitle('Comprehensive Precision Mismatch Rate (Per Dataset, Per Model, Per Precision)')
    
    g.savefig(os.path.join(output_dir, 'precision_mismatch_comprehensive.pdf'), bbox_inches='tight')
    plt.close()

def plot_p3_first_divergence_cdf(df: pd.DataFrame, output_dir: str):
    """Figure P3: First Divergence Position CDF"""
    div_df = df[df['first_divergence_pos'] >= 0].copy()
    if div_df.empty:
        return
        
    def get_color_pair(row):
        mapping = {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16"}
        # Fallbacks for non precision traces just in case
        a, b = mapping.get(row['config_A'], ""), mapping.get(row['config_B'], "")
        if "FP32" in (a,b) and "FP16" in (a,b): return "FP32 vs FP16"
        if "FP32" in (a,b) and "BF16" in (a,b): return "FP32 vs BF16"
        if "FP16" in (a,b) and "BF16" in (a,b): return "FP16 vs BF16"
        return "Other"

    div_df['pair'] = div_df.apply(get_color_pair, axis=1)
    div_df = div_df[div_df['pair'] != "Other"]

    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=div_df, x='first_divergence_pos', hue='pair', linewidth=2)
    plt.xlabel('First Divergence Position (Character Index)')
    plt.ylabel('Cumulative Distribution Function (CDF)')
    plt.title('CDF of Divergence Injection under Quantization Breakdown')
    plt.xlim(0, div_df['first_divergence_pos'].quantile(0.95)) # Cap at 95th proc to cut massive tails
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_first_divergence_cdf.pdf'))
    plt.close()

def plot_p6_benchmark_score_variation(df: pd.DataFrame, output_dir: str):
    """Figure P6: Benchmark Score Variation under Precision"""
    # Exclude wikitext since perplexity scale drastically mangles the bar chart compared to percentages
    acc_df = df[~df['benchmark'].isin(["wikitext", "lambada_openai", "lambada"])].copy()
    
    if acc_df.empty:
        print("No score data found for P6.")
        return
        
    plt.figure(figsize=(14, 7))
    sns.barplot(data=acc_df, x='benchmark', y='score', hue='precision', err_kws={'linewidth': 1})
    plt.ylabel('Benchmark Final Execution Score (%)')
    plt.xlabel('Standardized Benchmark Tasks')
    plt.title('Benchmark Baseline Deflection under Precision Types')
    plt.legend(title='Mathematical Space (Precision)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_benchmark_score_variation.pdf'))
    plt.close()

def gather_trace_data(trace_dir: str):
    """Gathers MAE and margin data from all trace JSONs."""
    mae_records = []
    margin_records = []
    if not os.path.exists(trace_dir):
        return pd.DataFrame(), pd.DataFrame()
        
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
                
                # P4: Layerwise MAE
                mae = data.get('layerwise_mae', {})
                fp32_fp16 = mae.get('fp32_vs_fp16', [])
                fp32_bf16 = mae.get('fp32_vs_bf16', [])
                
                for layer_idx, error in enumerate(fp32_fp16):
                    mae_records.append({'model': model, 'benchmark': benchmark, 'layer': layer_idx, 'precision_pair': 'FP32 vs FP16', 'mae': error})
                for layer_idx, error in enumerate(fp32_bf16):
                    mae_records.append({'model': model, 'benchmark': benchmark, 'layer': layer_idx, 'precision_pair': 'FP32 vs BF16', 'mae': error})
                    
                # P5: Token flip vs margin
                dyn = data.get('token_step_dynamics', {})
                margins = dyn.get('fp32_top2_margin', [])
                flip_bf16 = dyn.get('flipped_in_bf16', [])
                flip_fp16 = dyn.get('flipped_in_fp16', [])
                
                for m, f_bf, f_fp in zip(margins, flip_bf16, flip_fp16):
                    margin_records.append({'model': model, 'margin': m, 'precision_pair': 'FP32 vs BF16', 'flipped': f_bf})
                    margin_records.append({'model': model, 'margin': m, 'precision_pair': 'FP32 vs FP16', 'flipped': f_fp})
            except Exception as e:
                print(f"Error parsing {path}: {e}")
                continue
    return pd.DataFrame(mae_records), pd.DataFrame(margin_records)

def plot_p4_layerwise_mae(mae_df: pd.DataFrame, output_dir: str):
    """Figure P4: Layer-wise Hidden-State MAE Error"""
    if mae_df.empty: return
    
    models = mae_df['model'].unique()
    for model in models:
        df_model = mae_df[mae_df['model'] == model]
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_model, x='layer', y='mae', hue='precision_pair', errorbar=('ci', 95), marker='o')
        plt.title(f'{model} - Layer-wise Hidden-State Mean Absolute Error')
        plt.xlabel('Transformer Layer Depth')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'p4_layerwise_mae_{model.replace("/", "_")}.pdf'))
        plt.close()

def plot_p5_token_flip_probability_vs_margin(margin_df: pd.DataFrame, output_dir: str):
    """Figure P5: Token Flip Probability vs Logit Margin"""
    if margin_df.empty: return
    
    # Bin the margins for probability calculation
    bins = np.linspace(0, margin_df['margin'].quantile(0.95) if not margin_df.empty else 10, 20)
    margin_df['margin_bin'] = pd.cut(margin_df['margin'], bins=bins)
    
    prob_df = margin_df.groupby(['model', 'precision_pair', 'margin_bin'])['flipped'].mean().reset_index()
    prob_df['margin_center'] = prob_df['margin_bin'].apply(lambda x: x.mid if pd.notnull(x) else np.nan).astype(float)
    prob_df = prob_df.dropna(subset=['margin_center'])
    
    models = prob_df['model'].unique()
    for model in models:
        df_model = prob_df[prob_df['model'] == model]
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_model, x='margin_center', y='flipped', hue='precision_pair', marker='s', linewidth=2)
        plt.title(f'{model} - Token Flip Probability vs Logit Margin')
        plt.xlabel('Logit Margin (Top 1 probability - Top 2 probability)')
        plt.ylabel('Flip Probability under Low Precision')
        plt.xlim(bins[0], bins[-1])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'p5_token_flip_prob_{model.replace("/", "_")}.pdf'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Phase 8 Paper-Ready Precision Figures")
    parser.add_argument("--input_csv", type=str, default="pairwise_compare.csv")
    parser.add_argument("--output_dir", type=str, default="paper_plots")
    parser.add_argument("--base_dir", type=str, default="results")
    parser.add_argument("--trace_dir", type=str, default="trace")
    
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        df = load_data(args.input_csv)
        print(f"Aggregating {len(df)} discrete comparison rows...")
        
        print("Rendering P1 (Precision Heatmaps)...")
        plot_p1_precision_heatmap(df, args.output_dir)
        
        print("Rendering P2 (Mismatch Bar Graphs)...")
        plot_p2_mismatch_rate(df, args.output_dir)
        
        print("Rendering P3 (Divergence CDF)...")
        plot_p3_first_divergence_cdf(df, args.output_dir)
        
        print("Rendering P6 (Benchmark Raw Score Extraction)...")
        scores_df = extract_benchmark_scores(args.base_dir)
        plot_p6_benchmark_score_variation(scores_df, args.output_dir)
        
        print("Rendering P4 & P5 (Deep Tracing Hidden States and Logits)...")
        mae_df, margin_df = gather_trace_data(args.trace_dir)
        plot_p4_layerwise_mae(mae_df, args.output_dir)
        plot_p5_token_flip_probability_vs_margin(margin_df, args.output_dir)

        print(f"SUCCESS: All Paper-Ready Figures strictly compiled in {args.output_dir}/")
        
    except Exception as e:
        print(f"FATAL Engine Error: {e}")

if __name__ == "__main__":
    main()
