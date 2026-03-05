# Numerical Randomness and Evaluation Instability in LLMs

This document aggregates the comprehensive experimental findings regarding floating-point variability traversing different attention implementations (Eager vs. SDPA) across major Large Language Models.

## 1. Primary Vector: Attention Implementations & Token Flips
Modern LLM infrastructures (like vLLM) use heavily optimized FlashAttention/PagedAttention frameworks. Our experiments simulate these optimized layouts by forcing models to use `sdpa` (Scaled Dot-Product Attention) and comparing the outputs against the mathematically standard `eager` implementation.

**Key Finding:** Even when seeded entirely identically, passing the exact same prompt to the exact same model weights yields **different top chosen tokens** (Token Flips) in a measurable percentage of inferences simply by swapping the attention backend. 

## 2. Cross-Model and Dataset Scaled Evaluation
We executed benchmarking sweeps traversing three datasets (**Piqa, Commonsense QA, CMMLU**) across six major open-source checkpoints:
* `Qwen/Qwen2.5-0.5B` & `3B` & `7B`
* `Qwen/Qwen3-4B`
* `google/gemma-7b`
* `deepseek-ai/deepseek-llm-7b-base`
* `NousResearch/Meta-Llama-3-8B`

### The Role of Precision Format (FP32 vs FP16 vs BF16)
We ran the above matrices isolating the models loaded under entirely separate precision formats:

1. **Float32 (FP32 - 23 bit Mantissa):** Absolute Stability. We observed **0 Token Flips** across all models and all datasets under FP32. The precision bounds are high enough to completely absorb and erase the calculation path differences between Eager and SDPA block tilings.
2. **Float16 (FP16 - 10 bit Mantissa):** Moderate Instability. Token flips begin appearing due to the restricted mantissa resolution.
3. **BFloat16 (BF16 - 7 bit Mantissa):** High Instability. Models routinely flip top token predictions under `BF16`. This format allocates bits to the exponent (enabling massive dynamic ranges) at the direct expense of decimal precision.

## 3. Deep Dive Insights (Plotted In `experiment_results/plots/`)

### A. The Confidence Threshold (CDF Margins)
*Plots: `scaled_precision_flip_cdf.png` / `confidence_flip_distribution.png`*
We extracted the Softmax `logits` of the top two answers exclusively for prompts that experienced a flip. The CDF graphs reveal that flips overwhelmingly occur when the model is already "indecisive". If the absolute probability difference between *Token A* and *Token B* is under `1e-3`, the structural variance from BF16 SDPA acts as a tie-breaker, randomly elevating the runner-up token to the global maximum. 

### B. Internal Hidden State Divergence (Layer-by-Layer MAE)
*Plots: `scaled_precision_layer_trends.png` / `layer_variance_multi_dataset.png`*
Instead of just looking at the final token, we hooked into the individual Transformer layers and traced the Mean Absolute Error (MAE) of the hidden states (`Eager` vs `SDPA`). 
We mapped massive **inflection points**. Early layers output highly matched tensors (MAE ~1e-4). However, at specific deep bottleneck layers (e.g. Layer 26 for Qwen2.5-3B, Layer 28 for DeepSeek-7B), the MAE explodes logarithmically. 

### C. The Root Cause: Outliers and The "Swamping Effect"
*Plots: `inflection_analysis_layer_trends.png` / `precision_swamping_illustration.png`*
By tracing the maximum activation magnitudes concurrently with the MAE inflection points, we isolated the physical cause of the BF16 errors: **Activation Outliers**.

At the exact layer where MAE explodes, the internal hidden states produce massive numeric anomalies (jumping from maximums of ~15.0 to **200.0+**). 
Because `BFloat16` only has a 7-bit mantissa, it suffers from the **Swamping Effect**. When a small, sensitive feature value (e.g. `0.05`) is mathematically added/accumulated against a massive outlier (`250.0`), the limited fractional resolution of BF16 completely truncates and permanently erases the `0.05` signal. Because Eager and SDPA execute these aggregations using different memory chunking blocks, the truncation rounding applies differently to the hidden states, creating massive, compounding structural variance that ruins the output logits. 

### D. Contextual Sequences Triggering Outliers
By dynamically probing the token strings surrounding the BF16 outliners (via `extract_outlier_context.py`), we can map the linguistic triggers forming these mathematical anomalies to better understand the textual vulnerabilities of LLM inference pipelines.
