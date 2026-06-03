#!/bin/bash
# setup_env.sh - Initialize environment for VLN Project

# Exit on error
set -e

echo "Creating Conda environment 'vln-alfred' from environment.yaml..."
conda env create -f environment.yaml

# Activate environment
# Note: 'conda activate' might not work directly inside a script depending on shell config
# Use 'source' or the full path to the env's bin
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate vln-alfred

echo "Installing core dependencies via UV (ultra-fast pip replacement)..."

# Install Torch with CUDA 12.1 (compatible with CUDA 12.0 drivers on Titan V)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Lightning, Hydra, and standard ML tools
uv pip install \
    pytorch-lightning \
    hydra-core \
    hydra-colorlog \
    hydra-optuna-sweeper \
    wandb \
    tensorboard

# Install VLN Specifics
uv pip install \
    ai2thor \
    alfworld \
    openai \
    transformers \
    py-trees \
    networkx \
    opencv-python \
    scikit-image

echo "Setup complete. Please activate the environment using: conda activate vln-alfred"
