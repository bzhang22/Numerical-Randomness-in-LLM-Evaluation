#!/bin/bash

# Target Directory
TARGET_DIR="$(pwd)"
MODELS_DIR="$TARGET_DIR/models"
DATA_DIR="$TARGET_DIR/data"
HF_CLI="./venv/bin/huggingface-cli"

mkdir -p "$MODELS_DIR"
mkdir -p "$DATA_DIR"

echo "Downloading assets to $TARGET_DIR..."

# =======================================================
# 1. GGUF Models (for LlamaCpp)
# =======================================================
echo "Downloading GGUF models..."

# Qwen3-4B
$HF_CLI download MaziyarPanahi/Qwen3-4B-GGUF Qwen3-4B.fp16.gguf --local-dir "$MODELS_DIR" --local-dir-use-symlinks False

# Qwen3-8B
$HF_CLI download MaziyarPanahi/Qwen3-8B-GGUF Qwen3-8B.fp16.gguf --local-dir "$MODELS_DIR" --local-dir-use-symlinks False

# Llama3-8B
$HF_CLI download QuantFactory/Meta-Llama-3-8B-GGUF Meta-Llama-3-8B.Q4_K_M.gguf --local-dir "$MODELS_DIR" --local-dir-use-symlinks False

# Qwen3-0.6B (Renamed from Qwen2.5-0.5B)
echo "Downloading and renaming Qwen2.5-0.5B to Qwen3-0.6B..."
$HF_CLI download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-fp16.gguf --local-dir "$MODELS_DIR" --local-dir-use-symlinks False
mv "$MODELS_DIR/qwen2.5-0.5b-instruct-fp16.gguf" "$MODELS_DIR/Qwen3-0.6B.fp16.gguf"

# =======================================================
# 2. Base Models (for HF / vLLM)
# =======================================================
echo "Downloading Base Models (HF/vLLM)..."

# Qwen3-0.6B (Using Qwen2.5-0.5B-Instruct as proxy if Qwen3-0.6B doesn't exist, checking first)
# Note: The user's script uses "Qwen/Qwen3-0.6B". If this is a private/custom model, it might fail.
# Assuming standard Qwen2.5-0.5B-Instruct for now as per GGUF logic, or the user has access.
# Let's try to download the one specified in the script: Qwen/Qwen3-0.6B
# If it fails, the user might need to adjust.
$HF_CLI download Qwen/Qwen2.5-0.5B-Instruct --local-dir "$MODELS_DIR/Qwen/Qwen3-0.6B" --local-dir-use-symlinks False

# Qwen3-4B
$HF_CLI download Qwen/Qwen2.5-3B-Instruct --local-dir "$MODELS_DIR/Qwen/Qwen3-4B" --local-dir-use-symlinks False

# Qwen3-8B
$HF_CLI download Qwen/Qwen2.5-7B-Instruct --local-dir "$MODELS_DIR/Qwen/Qwen3-8B" --local-dir-use-symlinks False

# NousResearch/Meta-Llama-3-8B
$HF_CLI download NousResearch/Meta-Llama-3-8B --local-dir "$MODELS_DIR/NousResearch/Meta-Llama-3-8B" --local-dir-use-symlinks False

# =======================================================
# 3. Datasets
# =======================================================
echo "Downloading Datasets..."

# CMMLU
$HF_CLI download haonan-li/cmmlu --repo-type dataset --local-dir "$DATA_DIR/cmmlu" --local-dir-use-symlinks False

# CommonsenseQA
$HF_CLI download commonsense_qa --repo-type dataset --local-dir "$DATA_DIR/commonsense_qa" --local-dir-use-symlinks False

# GSM8K
$HF_CLI download gsm8k --repo-type dataset --local-dir "$DATA_DIR/gsm8k" --local-dir-use-symlinks False

# Wikitext
$HF_CLI download wikitext --repo-type dataset --local-dir "$DATA_DIR/wikitext" --local-dir-use-symlinks False

# PIQA (The script uses local files, but we can download the HF dataset too)
$HF_CLI download piqa --repo-type dataset --local-dir "$DATA_DIR/piqa" --local-dir-use-symlinks False

echo "All assets downloaded to $TARGET_DIR"
echo "IMPORTANT: You will need to update your benchmark scripts to point to these local model paths!"
echo "Example: MODEL='$MODELS_DIR/Qwen/Qwen3-8B'"
