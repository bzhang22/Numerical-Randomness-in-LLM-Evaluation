# LLM Numerical Randomness - Paper Central Package

This directory serves as the centralized, independent repository representing the culmination of all scripts, experimental logic, and aggregated tables used for inference precision evaluation and batch randomness. 

## Python Plotting & Visualization Suite
*   **`generate_dataset_mae_plots.py`**: Maps dataset-specific trends tracking sequence layer divergences (MAE vs FP32 Baseline). Uses a SymLog metric scale to capture absolute stability mathematically.
*   **`plot_batch_layer_variance.py`** & **`plot_batch_variance.py`**: Reads JSONL logs formatting divergence across specific batch executions (`BS=1`, `BS=8`, etc.) into bar and CDF metrics.
*   **`plot_precision_distribution.py`** & **`plot_scaled_distribution.py`**: Evaluates rounding and casting variation when transitioning logic constraints from Float32 directly into Float16 and BFloat16 paradigms.
*   **`plot_inflection.py`** & **`plot_swamping_effect.py`**: Generates line analyses capturing exact precision saturation bounds.
*   **`generate_latex_tables.py`**: Merges raw mapping mismatches dynamically building publication-ready LaTeX parameter sheets.

## Core Evaluation Runners & Tracing Modules
*   **`run_benchmark.py`**: The underlying orchestrator natively triggering generation loops evaluating target models (Llama, Mistral, Qwen, Gemma) against PiQA, GSM8K, CMMLU, HumanEval.
*   **`extract_traces.py`**: Re-invokes matched sequences tracking explicit hidden states tensor per transformer layer, dumping massive raw target logs to `trace/`. 
*   **`intervention_experiments.py`**: Contains deterministic ablation parameters verifying specific rounding clamps and noise causalities.

## SLURM Workloads & Methods (How to Run)
*   **`submit_experiments.sh`** / **`submit_experiments_tier2.sh`**: Broad cross-evaluation arrays executing benchmarks against grid matrices covering `float16/float32/bfloat16` and batch iterations.
*   **`dispatch_traces.sh`** & **`submit_tracing.sh`**: The highly memory-intensive jobs pulling sequence arrays through `extract_traces.py` utilizing expanded segment configurations up to A100/H100 specs.
*   **`run_*_slurm.sh`**: Automated plotting allocators wrapping specific evaluations dynamically pushing visualizations across constrained memory nodes.

## Attached Processed Data
*   **`pairwise_compare.csv`**: A pre-processed exact-mismatch sheet summarizing basic outputs per target without hidden layer dimensions.
*   **`batch_layer_variance_results.jsonl`**, **`*core_mae.jsonl`**, **`[target]_layer_*_results.jsonl`**: Detailed per-sequence JSON array breakdowns tracking final MAE standard deviations and explicit token causality clamp behaviors.
*   **`batch_size_table.csv`** & **`batch_layer_mae_table.csv`**: Aggregated flip statistics isolated by precision modes directly readable by `Matplotlib / Seaborn`.
