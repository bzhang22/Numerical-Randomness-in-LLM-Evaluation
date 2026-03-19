#!/bin/bash
#SBATCH --job-name="plot_mae"
#SBATCH --output="plot_mae.out"
#SBATCH --error="plot_mae.err"
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

cd /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation
/blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness/bin/python generate_dataset_mae_plots.py
