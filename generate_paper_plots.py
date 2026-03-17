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
    """Figure P2: Precision-Induced Mismatch Rate by Benchmark (Option A)"""
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
    
    # Calculate group means precisely
    agg = prec_df.groupby(['model', 'benchmark', 'pair'])['sequence_mismatch_rate'].mean().reset_index()
    
    # 1. Catplot grouping by Model (one subplot per model)
    g = sns.catplot(
        data=agg, kind='bar',
        x='benchmark', y='sequence_mismatch_rate', hue='pair',
        col='model', col_wrap=4, height=4, aspect=1.2,
        sharey=False
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Sequence Mismatch Rate per Benchmark by Model')
    g.set_axis_labels('Benchmark Task', 'Mismatch Rate (%)')
    g.set_titles('{col_name}')
    for axes in g.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
    g.savefig(os.path.join(output_dir, 'precision_mismatch_faceted_by_model.pdf'))
    plt.close()
    
    # 2. Catplot grouping by Benchmark (one subplot per benchmark)
    g2 = sns.catplot(
        data=agg, kind='bar',
        x='model', y='sequence_mismatch_rate', hue='pair',
        col='benchmark', col_wrap=2, height=4, aspect=1.5,
        sharey=False
    )
    g2.fig.subplots_adjust(top=0.9)
    g2.fig.suptitle('Sequence Mismatch Rate per Model by Benchmark')
    g2.set_axis_labels('Model Instance', 'Mismatch Rate (%)')
    g2.set_titles('{col_name}')
    for axes in g2.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
    g2.savefig(os.path.join(output_dir, 'precision_mismatch_faceted_by_benchmark.pdf'))
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

def main():
    parser = argparse.ArgumentParser(description="Generate Phase 8 Paper-Ready Precision Figures")
    parser.add_argument("--input_csv", type=str, default="pairwise_compare.csv")
    parser.add_argument("--output_dir", type=str, default="paper_plots")
    parser.add_argument("--base_dir", type=str, default="results")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        df = load_data(args.input_csv)
        print(f"Aggregating {len(df)} discrete comparison rows...")
        
        print("Rendering P1 (Precision Heatmaps)...")
        plot_p1_precision_heatmap(df, args.output_dir)
        print("P1 complete!")
        
        print("Rendering P2 (Mismatch Bar Graphs)...")
        plot_p2_mismatch_rate(df, args.output_dir)
        print("P2 complete!")
        
        print("Rendering P3 (Divergence CDF)...")
        plot_p3_first_divergence_cdf(df, args.output_dir)
        print("P3 complete!")
        
        print("Rendering P6 (Benchmark Raw Score Extraction)...")
        scores_df = extract_benchmark_scores(args.base_dir)
        plot_p6_benchmark_score_variation(scores_df, args.output_dir)
        print("P6 complete!")

        print(f"SUCCESS: All Paper-Ready Figures strictly compiled in {args.output_dir}/")
        
    except Exception as e:
        print(f"FATAL Engine Error: {e}")

if __name__ == "__main__":
    main()
