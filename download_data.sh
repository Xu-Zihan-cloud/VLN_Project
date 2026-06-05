#!/bin/bash
# download_data.sh - Automate ALFRED/AlfWorld dataset preparation

set -e

# Ensure the conda environment is active to use alfworld-download
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate vln-alfred
fi

echo "=== Step 1: Downloading AlfWorld Trajectories via CLI ==="
# Using the official alfworld-download tool is the most reliable method
# It will download the JSONs and PDDL files needed for the tasks.

# Set the environment variable for alfworld data
export ALFWORLD_DATA="$(pwd)/data/alfworld"
mkdir -p $ALFWORLD_DATA

echo "Using alfworld-download to retrieve JSON trajectories..."
# Use the correct data-dir argument as per the help output
alfworld-download --data-dir "$ALFWORLD_DATA"

echo "=== Step 2: Verification ==="
echo "Data has been downloaded to: $ALFWORLD_DATA"
ls -R $ALFWORLD_DATA | head -n 20

echo "=== [SUCCESS] Official AlfWorld data preparation completed! ==="
echo "To run training, ensure your config data_dir points to $ALFWORLD_DATA"
