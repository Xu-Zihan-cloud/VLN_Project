#!/bin/bash
set -e

echo "=== Step 1: Initializing Conda hooks ==="
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

echo "=== Step 2: Creating Conda environment 'vln-alfred' ==="
conda env create -f environment.yaml --force

echo "=== Step 3: Activating environment ==="
conda activate vln-alfred

echo "=== Step 4: Installing CPU-Optimized Dependencies via UV ==="
# Note: Ensure system-level 'xvfb' is installed: sudo apt-get install -y xvfb

# Install CPU-specific PyTorch for AVX-512 support
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Frameworks
uv pip install pytorch-lightning hydra-core hydra-colorlog hydra-optuna-sweeper wandb tensorboard

# Install Simulation, LLM & Headless Tools
uv pip install ai2thor alfworld pyvirtualdisplay opencv-python scikit-image transformers py-trees networkx openai

echo "=== [SUCCESS] CPU-optimized setup completed! ==="
