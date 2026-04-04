import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob

def load_fp32_references(results_dir):
    # Maps (Model, Dataset, id) -> Prediction
    ref_dict = {}
    files = glob.glob(os.path.join(results_dir, "*_*_fp32_reference.jsonl"))
    for filepath in files:
        if not os.path.exists(filepath): continue
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    m = obj.get("model", "").split("/")[-1]
                    d = obj.get("dataset", "")
                    i = obj.get("id", "")
                    pred = obj.get("prediction", None)
                    ref_dict[(m, d, i)] = pred
                except Exception:
                    pass
    return ref_dict

def parse_single_layer_results(results_dir, ref_dict):
    data = []
    files = glob.glob(os.path.join(results_dir, "*_*_*_layer*.jsonl"))
    
    for filepath in files:
        if not os.path.exists(filepath): continue
        if "_meta.json" in filepath: continue
        
        mismatches = 0
        total = 0
        correct = 0
        
        model = ""
        dataset = ""
        variant = ""
        layer_split = ""
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    model = obj.get("model", "").split("/")[-1]
                    dataset = obj.get("dataset", "")
                    variant = obj.get("variant", "")
                    layer_split = str(obj.get("layer_split", ""))
                    i = obj.get("id", "")
                    pred = obj.get("prediction", None)
                    is_correct = obj.get("correct", False)
                    
                    if is_correct: correct += 1
                    
                    # Compute mismatch against fp32 reference
                    ref_pred = ref_dict.get((model, dataset, i), None)
                    if pred != ref_pred:
                        mismatches += 1
                    total += 1
                except Exception:
                    pass
                    
        # Verify that layer_split is an integer or numeric
        if total > 0 and (layer_split.isdigit() or (layer_split.startswith('-') and layer_split[1:].isdigit())):
            data.append({
                "Model": model,
                "Dataset": dataset,
                "Variant": variant,
                "Target Layer": int(layer_split),
                "Accuracy": correct / total,
                "Mismatch Rate": mismatches / total
            })

    return pd.DataFrame(data)

def get_baselines(results_dir, ref_dict):
    baselines = []
    files = glob.glob(os.path.join(results_dir, "*_*_*baseline.jsonl")) + glob.glob(os.path.join(results_dir, "*_*_*reference.jsonl"))
    
    for filepath in files:
        if not os.path.exists(filepath): continue
        if "_meta.json" in filepath: continue
        
        mismatches = 0
        total = 0
        correct = 0
        
        model = ""
        dataset = ""
        variant = ""
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    model = obj.get("model", "").split("/")[-1]
                    dataset = obj.get("dataset", "")
                    variant = obj.get("variant", "")
                    if variant not in ["bf16_baseline", "fp32_reference", "fp16_baseline"]:
                        continue
                        
                    i = obj.get("id", "")
                    pred = obj.get("prediction", None)
                    is_correct = obj.get("correct", False)
                    
                    if is_correct: correct += 1
                    ref_pred = ref_dict.get((model, dataset, i), None)
                    if pred != ref_pred: mismatches += 1
                    total += 1
                except Exception:
                    pass
                    
        if total > 0 and variant in ["bf16_baseline", "fp32_reference", "fp16_baseline"]:
            baselines.append({
                "Model": model,
                "Dataset": dataset,
                "Variant": variant,
                "Accuracy": correct / total,
                "Mismatch Rate": mismatches / total
            })
    return pd.DataFrame(baselines)

def plot_single_layer_sweep(df, base_df, out_path_acc, out_path_diverge):
    if df.empty:
        print("No single layer data found!")
        return
        
    sns.set_theme(style="whitegrid", context="talk")
    datasets = ["cmmlu", "piqa"]
    
    # Plot 1: Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=False)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df[df["Dataset"] == dataset]
        base_subset = base_df[base_df["Dataset"] == dataset]
        if subset.empty: continue
            
        sns.lineplot(data=subset, x="Target Layer", y="Accuracy", hue="Model", marker="o", ax=ax, linewidth=3, markersize=8)
        
        models = subset["Model"].unique()
        colors = sns.color_palette()[:len(models)]
        for m_idx, m in enumerate(models):
            m_base = base_subset[base_subset["Model"] == m]
            fp32_val = m_base[m_base["Variant"] == "fp32_reference"]["Accuracy"].mean()
            if not pd.isna(fp32_val): ax.axhline(fp32_val, color=colors[m_idx], linestyle="--", alpha=0.7)
            bf16_val = m_base[m_base["Variant"] == "bf16_baseline"]["Accuracy"].mean()
            if not pd.isna(bf16_val): ax.axhline(bf16_val, color=colors[m_idx], linestyle=":", alpha=0.7)
                
        ax.set_title(f"Targeted Precision Efficacy ({dataset.upper()})")
        ax.set_xlabel("Protected Layer (FP32)")
        ax.set_ylabel("Final Model Accuracy")
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
    plt.tight_layout()
    plt.savefig(out_path_acc, dpi=300, bbox_inches="tight")
    
    # Plot 2: Divergence Mismatch
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=False)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df[df["Dataset"] == dataset]
        base_subset = base_df[base_df["Dataset"] == dataset]
        if subset.empty: continue
            
        sns.lineplot(data=subset, x="Target Layer", y="Mismatch Rate", hue="Model", marker="o", ax=ax, linewidth=3, markersize=8)
        
        for m_idx, m in enumerate(models):
            m_base = base_subset[base_subset["Model"] == m]
            bf16_val = m_base[m_base["Variant"] == "bf16_baseline"]["Mismatch Rate"].mean()
            if not pd.isna(bf16_val): ax.axhline(bf16_val, color=colors[m_idx], linestyle=":", alpha=0.7, label="BF16 Base Divergence" if m_idx==0 else "")
                
        ax.set_title(f"Output Divergence Under BF16 Noise ({dataset.upper()})")
        ax.set_xlabel("Protected Layer (FP32)")
        ax.set_ylabel("Mismatch Rate vs FP32 (%)")
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
    plt.tight_layout()
    plt.savefig(out_path_diverge, dpi=300, bbox_inches="tight")
    print(f"Saved accuracy plot to {out_path_acc} and diverge plot to {out_path_diverge}")

if __name__ == "__main__":
    RESULTS_DIR = "results_mitigation"
    ref_dict = load_fp32_references(RESULTS_DIR)
    df = parse_single_layer_results(RESULTS_DIR, ref_dict)
    
    if not df.empty:
        df.sort_values(["Model", "Dataset", "Target Layer"], inplace=True)
        df.to_csv("single_layer_sweep.csv", index=False)
        print("Saved raw data to single_layer_sweep.csv")
        
        base_df = get_baselines(RESULTS_DIR, ref_dict)
        plot_single_layer_sweep(df, base_df, "single_layer_acc_sweep.png", "single_layer_diverge_sweep.png")
    else:
        print("No exhaustive sweep data available yet.")
