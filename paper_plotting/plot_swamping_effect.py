import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def simulate_swamping():
    # We simulate passing a small sensitive activation (e.g. 0.05) 
    # through a layer that has a massive outlier activation (M).
    # When added together (M + small), the limited mantissa of lower precisions
    # causes the 'small' value to be rounded off or completely destroyed.
    
    true_small_signal = 0.05
    
    # Range of outlier magnitudes observed in the model (e.g., up to 300)
    outlier_magnitudes = np.linspace(1, 400, 400)
    
    recovered_bf16 = []
    recovered_fp16 = []
    recovered_fp32 = []
    
    for M in outlier_magnitudes:
        # Simulate FP32 (Standard)
        val_fp32 = torch.tensor(M + true_small_signal, dtype=torch.float32)
        recovered_fp32.append(val_fp32.item() - M)
        
        # Simulate FP16
        val_fp16 = torch.tensor(M + true_small_signal, dtype=torch.float16)
        recovered_fp16.append(val_fp16.item() - M)
        
        # Simulate BF16
        val_bf16 = torch.tensor(M + true_small_signal, dtype=torch.bfloat16)
        recovered_bf16.append(val_bf16.item() - M)
        
    plt.figure(figsize=(10, 6))
    
    # Plotting the recovered signals
    plt.plot(outlier_magnitudes, [true_small_signal]*len(outlier_magnitudes), 'k--', lw=2, label="Original Small Signal (0.05)")
    
    plt.plot(outlier_magnitudes, recovered_fp32, 'g-', lw=3, alpha=0.8, label="Recovered in Float32 (100% matched)")
    plt.plot(outlier_magnitudes, recovered_fp16, 'b-', lw=2, alpha=0.7, label="Recovered in Float16")
    plt.scatter(outlier_magnitudes, recovered_bf16, color='red', s=10, alpha=0.5, label="Recovered in BFloat16")
    
    # Emphasize where the signal is completely destroyed (Recovered = 0)
    plt.axhline(0, color='gray', linewidth=1)
    
    plt.title("The 'Swamping' Effect: How Outliers Destroy Low-Precision Features", fontsize=14, fontweight='bold')
    plt.xlabel("Magnitude of Accompanying Outlier Feature in the same Layer", fontsize=12)
    plt.ylabel("Recovered Signal Value (True = 0.05)", fontsize=12)
    
    # Annotations to explain the math
    plt.annotate(
        "BFloat16 entirely erases the 0.05 signal\nwhen the outlier exceeds ~15\ndue to having only 7 bits of mantissa.", 
        xy=(50, 0), xytext=(80, 0.015),
        arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1)
    )
    plt.annotate(
        "Float16 starts stepping/losing precision\nbut lasts longer due to 10 bits of mantissa.",
        xy=(200, 0.045), xytext=(220, 0.03),
        arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=1)
    )
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    
    output_path = "/home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation/precision_swamping_illustration.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved swamping illustration to {output_path}")

if __name__ == "__main__":
    simulate_swamping()
