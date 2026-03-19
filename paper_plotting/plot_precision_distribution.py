import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_confidence_log(filepath):
    data = {}
    current_model = None
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                model_match = re.search(r'Loading (.*?) \[(.*?)\]\.\.\.', line)
                if model_match:
                    current_model = f"{model_match.group(1)} ({model_match.group(2)})"
                    if current_model not in data:
                        data[current_model] = {'eager_margins': [], 'sdpa_margins': []}
                margin_match = re.search(r'Eager Margin Between Tokens: ([0-9\.\-]+) \| SDPA Margin Between Tokens: ([0-9\.\-]+)', line)
                if margin_match and current_model:
                    eager_m = float(margin_match.group(1))
                    sdpa_m = float(margin_match.group(2))
                    data[current_model]['eager_margins'].append(abs(eager_m))
                    data[current_model]['sdpa_margins'].append(abs(sdpa_m))
    except FileNotFoundError:
        pass
    return data

def plot_confidence_cdf(data, output_path):
    models_with_data = [m for m, m_data in data.items() if len(m_data['eager_margins']) > 0]
    if not models_with_data:
        return
        
    plt.figure(figsize=(10, 7))
    
    colors = {
        "Qwen/Qwen2.5-3B (float32)": "#3498db",
        "Qwen/Qwen2.5-3B (float16)": "#e74c3c",
        "Qwen/Qwen2.5-3B (bfloat16)": "#f1c40f"
    }
    fallback_colors = plt.cm.tab20.colors
    color_idx = 0
    
    for model_name in models_with_data:
        m_data = data[model_name]
        c = colors.get(model_name)
        if not c:
            c = fallback_colors[color_idx % len(fallback_colors)]
            color_idx += 1
            
        margins = np.array(m_data['eager_margins'])
        margins = np.sort(margins)
        # Cumulative probability
        p = 1.0 * np.arange(len(margins)) / (len(margins) - 1)
        
        plt.plot(margins, p, label=f"{model_name} (N={len(margins)})", color=c, linewidth=2)
        
    plt.title("Cumulative Distribution of Confidence Margin (Cross-Precision Flips)", fontsize=14, fontweight='bold')
    plt.xlabel("Confidence Difference between Top 2 Answers", fontsize=12)
    plt.ylabel("Cumulative Proportion of Flips", fontsize=12)
    
    # Optional: Log scale can sometimes help if values are squeezed near 0
    # plt.xscale('log')
    
    plt.xlim(xmin=0)
    plt.ylim(0, 1.05)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10, loc="lower right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=str, nargs='+', default=["/home/bohanzhang1/hooks_precision_test.log"])
    args = parser.parse_args()
    
    data = {}
    for log in args.logs:
        log_data = parse_confidence_log(log)
        for model, m_data in log_data.items():
            if model not in data:
                data[model] = {'eager_margins': [], 'sdpa_margins': []}
            data[model]['eager_margins'].extend(m_data['eager_margins'])
            data[model]['sdpa_margins'].extend(m_data['sdpa_margins'])
            
    output_path = "/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/precision_flip_cdf.png"
    plot_confidence_cdf(data, output_path)

if __name__ == "__main__":
    main()
