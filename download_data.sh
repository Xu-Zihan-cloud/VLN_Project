#!/bin/bash
# download_data.sh - Automate ALFRED/AlfWorld dataset preparation

set -e

DATA_DIR="./data/alfworld"
mkdir -p $DATA_DIR

echo "=== Step 1: Downloading Expert Trajectories (JSONs) ==="
# Using the official DFKI mirror which is the primary source for ALFRED
if [ ! -d "$DATA_DIR/train" ]; then
    echo "Downloading JSON trajectories from DFKI mirror..."
    wget -c -O alfworld_jsons.7z http://madm.dfki.de/files/ieee-vln/json_2.1.0.7z
    
    echo "Extracting data... (This requires p7zip-full)"
    7z x alfworld_jsons.7z -o$DATA_DIR
    rm alfworld_jsons.7z
else
    echo "JSON trajectories already exist. Skipping Step 1."
fi

echo "=== Step 2: Visual Features Note ==="
echo "To run training, you also need pre-computed ResNet features."
echo "You can download them from the same mirror if needed:"
echo "URL: http://madm.dfki.de/files/ieee-vln/full_2.1.0.7z (Approx 50GB)"

echo "=== Step 3: Verifying Data Structure ==="
ls -l $DATA_DIR

echo "=== [SUCCESS] Data preparation script completed! ==="
