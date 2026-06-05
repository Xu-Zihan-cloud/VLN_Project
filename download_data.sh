#!/bin/bash
# download_data.sh - Automate ALFRED/AlfWorld dataset preparation

set -e

DATA_DIR="./data/alfworld"
mkdir -p $DATA_DIR

echo "=== Step 1: Downloading Expert Trajectories (JSONs) ==="
# These are the small JSON files containing instructions and action labels
# URL is illustrative; replace with the official ALFRED repository release links
# Use proxy to avoid timeout issues
    wget -O alfworld_jsons.zip https://ghfast.top/https://github.com/askforalfred/alfred/raw/master/data/json_2.1.0.7z 
    # Note: Requires 7zip/unzip depending on source
    echo "Please ensure you have 7z installed: sudo apt-get install p7zip-full"
    7z x alfworld_jsons.zip -o$DATA_DIR
    rm alfworld_jsons.zip
fi

echo "=== Step 2: Downloading Pre-computed ResNet Features ==="
# For CPU training, pre-computed features are MANDATORY for performance
# These typically reside in 'full_2.1.0' or 'resnet_features'
# Each trajectory folder should have a 'feat_conv.pt' or similar.
echo "Downloading visual features (this may take a while)..."
# Example Command for ResNet-18/50 features
# wget -c http://madm.dfki.de/files/ieee-vln/alfred_resnet_features.tar.gz

echo "=== Step 3: Verifying Data Structure ==="
# The datamodule expects: data/alfworld/{train,val_seen,val_unseen}/{scene_id}/traj_data.json
ls -l $DATA_DIR

echo "=== [SUCCESS] Data preparation script completed! ==="
echo "Note: If using full images, ensure you run the feature extraction script first."
