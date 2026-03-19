#!/bin/bash
#SBATCH --job-name="git_export"
#SBATCH --output="git_export.out"
#SBATCH --error="git_export.err"
#SBATCH --time=00:05:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --mem=2G

cd /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation

echo "Adding plotting directory to tracking..."
git add paper_plotting/

echo "Committing aggregation updates..."
git commit -m "Reorganize core Python plotters, logs, and LaTeX aggregators into paper_plotting module"

echo "Attempting to export upstream..."
git push origin master

echo "Done!"
