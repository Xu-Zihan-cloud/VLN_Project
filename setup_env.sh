#!/bin/bash
set -e

echo "=== Step 1: Initializing Conda hooks for sub-shell ==="
# Automatically detect Conda base path to support miniconda/anaconda universally
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: Could not find conda.sh. Proceeding assuming conda is in PATH..."
fi

echo "=== Step 2: Creating Conda environment 'vln-alfred' ==="
# Force overwrite if environment exists to ensure a clean slate
conda env create -f environment.yaml --force

echo "=== Step 3: Activating environment ==="
conda activate vln-alfred

echo "=== Step 4: Installing core dependencies via UV ==="
# Install PyTorch optimized for CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Lightning & Hydra Frameworks
uv pip install pytorch-lightning hydra-core hydra-colorlog hydra-optuna-sweeper wandb tensorboard

# Install Embodied AI Simulation & LLM Orchestration
uv pip install ai2thor alfworld opencv-python scikit-image transformers py-trees networkx openai

echo "=== [SUCCESS] Setup Completed Successfully! ==="
