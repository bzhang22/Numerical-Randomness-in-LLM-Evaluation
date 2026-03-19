import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_inflection_log(filepath):
    models_data = {}
    current_model = None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("INFLECTION POINT ANALYSIS:"):
            match = re.search(r'INFLECTION POINT ANALYSIS: (.*?) \[(.*?)\]', line)
            if match:
                current_model = f"{match.group(1)} ({match.group(2)})"
                models_data[current_model] = {
                    'layers': [], 'mae': [], 'eager_max': [], 'sdpa_max': [], 'eager_var': [], 'sdpa_var': []
                }
        elif current_model and re.match(r'^\d+\s+\|', line):
            # Parse row: 0      | 5.67748e-03  | 3.51562e-02  | 3.50000e+00  | ...
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 7:
                try:
                    layer = int(parts[0])
                    mae = float(parts[1])
                    e_max = float(parts[3])
                    s_max = float(parts[4])
                    e_var = float(parts[5])
                    s_var = float(parts[6])
                    
                    models_data[current_model]['layers'].append(layer)
                    models_data[current_model]['mae'].append(mae)
                    models_data[current_model]['eager_max'].append(e_max)
                    models_data[current_model]['sdpa_max'].append(s_max)
                    models_data[current_model]['eager_var'].append(e_var)
                    models_data[current_model]['sdpa_var'].append(s_var)
                except ValueError:
                    pass
    return models_data

def plot_inflections(models_data, output_path):
    if not models_data:
        print("No valid inflection data to plot.")
        return
        
    num_models = len(models_data)
    fig, axes = plt.subplots(num_models, 3, figsize=(18, 5 * num_models), squeeze=False)
    
    for i, (model_name, data) in enumerate(models_data.items()):
        layers = data['layers']
        
        # Plot 1: MAE Growth
        ax1 = axes[i, 0]
        ax1.plot(layers, data['mae'], 'r-', linewidth=2, label='Mean Absolute Error')
        ax1.set_title(f"{model_name}\nHidden State Divergence (MAE)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Layer Depth")
        ax1.set_ylabel("MAE")
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Max Magnitude
        ax2 = axes[i, 1]
        ax2.plot(layers, data['eager_max'], 'b--', linewidth=2, label='Eager Max Element')
        ax2.plot(layers, data['sdpa_max'], 'g-', linewidth=2, alpha=0.7, label='SDPA Max Element')
        ax2.set_title("Maximum Tensor Magnitude Over Layers", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Layer Depth")
        ax2.set_ylabel("Max Absolute Value")
        # ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Variance
        ax3 = axes[i, 2]
        ax3.plot(layers, data['eager_var'], 'm--', linewidth=2, label='Eager Variance')
        ax3.plot(layers, data['sdpa_var'], 'c-', linewidth=2, alpha=0.7, label='SDPA Variance')
        ax3.set_title("Tensor Structural Variance Over Layers", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Layer Depth")
        ax3.set_ylabel("Variance")
        # ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved inflection analysis plots to {output_path}")

def main():
    import sys
    log_file = "/home/bohanzhang1/inflection_analysis.log"
    out_file = "/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/inflection_analysis_layer_trends.png"
    
    data = parse_inflection_log(log_file)
    plot_inflections(data, out_file)

if __name__ == "__main__":
    main()
