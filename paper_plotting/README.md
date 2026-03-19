# LLM Numerical Randomness - Paper Plotting Suite

This directory serves as the centralized repository for all aggregated logs, result tables, and Python graphing scripts used to assess sequence variation, model divergence, and exact-mismatch rates.

## Core Contents
*   **`generate_dataset_mae_plots.py`**: Extracts `.json` token sequences tracking diverging layer states, filtering stability metrics utilizing a non-discarding `symlog` plot configuration without model exclusions.
*   **`plot_batch_layer_variance.py`** & **`plot_batch_variance.py`**: Reads `batch_layer_variance_results.jsonl` and native runtime execution logs to format bar charts showing sequence padding flips across batch iterations (`BS=1` vs `BS=8`).
*   **`generate_latex_tables.py`**: Combines data from `pairwise_compare.csv` into formal LaTex formatted metric sheets highlighting sequence stability percentage rates per model group.

## Data Sets Included
*   **`pairwise_compare.csv`**: A pre-processed map (~26MB) containing base cross-comparisons.
*   **`batch_layer_variance_results.jsonl`**: The aggregated trace outcomes summarizing final error deviations.
*   **`batch_size_table.csv`** & **`batch_layer_mae_table.csv`**: Post-processed outputs breaking down token deviations dynamically formatted.

## Execution Guide
Most graphing scripts can be natively executed if pointing correctly:
```bash
# E.g., re-running the flip charts
python plot_batch_variance.py --logs ../batch_variance_small.log ../batch_variance_large.log

# Extracting matching percentages
python generate_latex_tables.py --csv pairwise_compare.csv
```

> [!WARNING]
> Please note the script `generate_dataset_mae_plots.py` strictly attempts to read from the absolute parent `trace/` directory structure. Deep state tensors are gigabytes in size and cannot be committed via Git. To successfully reproduce MAE charts, you will first need to utilize `dispatch_traces.sh` to extract active sequence traces on SLURM resources beforehand.
