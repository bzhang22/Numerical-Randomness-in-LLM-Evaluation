#!/bin/bash

# setup_env.sh
# Script to set up the environment for Numerical Randomness in LLM Evaluation

ENV_NAME="llm_randomness"
REQUIREMENTS_FILE="requirements.txt"

echo "Setting up environment for $ENV_NAME..."

# Check if Conda is available
if command -v conda &> /dev/null; then
    echo "Conda found. Creating Conda environment '$ENV_NAME'..."
    
    # Create conda env with python 3.10 (good compatibility for vllm/torch)
    conda create -n $ENV_NAME python=3.10 -y
    
    # Activate env - need to source conda.sh usually, but we can try direct run or tell user
    echo "Environment created. To activate and install dependencies, run:"
    echo "conda activate $ENV_NAME"
    echo "pip install -r $REQUIREMENTS_FILE"
    
    # Attempt to install if we can activate or use run
    # Try to find conda base
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate $ENV_NAME
    
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -r $REQUIREMENTS_FILE
    
    echo "Setup complete! Activate with: conda activate $ENV_NAME"

else
    echo "Conda not found. Trying Python venv..."
    
    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 not found."
        exit 1
    fi
    
    # Create venv
    VENV_DIR="venv"
    echo "Creating virtual environment in ./$VENV_DIR..."
    python3 -m venv $VENV_DIR
    
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo "Error: Failed to create venv. You might need to install python3-venv."
        echo "Try: sudo apt install python3-venv"
        exit 1
    fi
    
    source $VENV_DIR/bin/activate
    
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    # Attempt to find nvcc for llama-cpp-python
    if [ -z "$CUDACXX" ] && [ -f "/usr/local/cuda/bin/nvcc" ]; then
        export CUDACXX=/usr/local/cuda/bin/nvcc
        echo "Found nvcc at $CUDACXX"
    fi
    export CMAKE_ARGS="-DGGML_CUDA=on"
    pip install -r $REQUIREMENTS_FILE
    
    echo "Setup complete! Activate with: source $VENV_DIR/bin/activate"
fi
