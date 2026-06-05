#!/bin/bash
# download_data.sh - Automate ALFRED/AlfWorld dataset preparation

set -e

DATA_DIR="./data/alfworld"
mkdir -p $DATA_DIR

echo "=== Step 1: Downloading Expert Trajectories (JSONs) ==="
# Using a more stable AWS S3 mirror for ALFRED data
if [ ! -d "$DATA_DIR/train" ]; then
    echo "Downloading JSON trajectories..."
    wget -O alfworld_jsons.7z https://ai2-thor-public.s3.amazonaws.com/alfred/json_2.1.0.7z
    
    echo "Please ensure you have 7z installed: sudo apt-get install p7zip-full"
    7z x alfworld_jsons.7z -o$DATA_DIR
    rm alfworld_jsons.7z
else
    echo "JSON trajectories already exist. Skipping Step 1."
fi

echo "=== Step 2: Downloading Pre-computed ResNet Features ==="
echo "Note: Full visual features are several GBs. Please download them manually if needed."
# Standard features location: https://ai2-thor-public.s3.amazonaws.com/alfred/full_2.1.0.7z

echo "=== Step 3: Verifying Data Structure ==="
ls -l $DATA_DIR

echo "=== [SUCCESS] Data preparation script completed! ==="
