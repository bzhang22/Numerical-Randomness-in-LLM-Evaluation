#!/bin/bash

# Log the wait cycle
echo "[$(date)] Waiting for FlashAttn_Build to drop from the SLURM queue..."

# Loop every 60 seconds as long as 'FlashAtt' exists in the user's squeue
while squeue -u bohanzhang1 | grep -q "FlashAtt"; do
    sleep 60
done

echo "[$(date)] Compiler job completed or exited. Triggering Phase 6 evaluation arrays..."
cd /home/bohanzhang1/Numerical-Randomness-in-LLM-Evaluation

# Dispatch the missing structural data batches
./submit_attention.sh

echo "[$(date)] Sequence complete!"
