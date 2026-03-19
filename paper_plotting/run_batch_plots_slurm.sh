#!/bin/bash
#SBATCH --job-name="plot_batch"
#SBATCH --output="plot_batch.out"
#SBATCH --error="plot_batch.err"
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

cd /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation
/blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness/bin/python plot_batch_variance.py
/blue/liguanpeng/bohanzhang1/conda_envs/llm_randomness/bin/python plot_batch_layer_variance.py
