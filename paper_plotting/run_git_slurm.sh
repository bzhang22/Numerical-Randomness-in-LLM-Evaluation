#!/bin/bash
#SBATCH --job-name="git_push"
#SBATCH --output="git_push.out"
#SBATCH --error="git_push.err"
#SBATCH --time=00:05:00
#SBATCH --mem=2G

cd /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation
git add plot_batch_variance.py plot_batch_layer_variance.py batch_size_table.csv batch_layer_mae_table.csv batch_size_flips.png batch_layer_mae_trends.png batch_final_layer_cdf.png
git commit -m "Update batch size scripts to show all models, dataset splits, and generate data CSV tables"
git push origin master
