#!/bin/bash
# setup_ollama.sh - Setup the LLM backend for the Modular Agent

set -e

echo "=== Step 1: Installing Ollama (Linux) ==="
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

echo "=== Step 2: Starting Ollama Service ==="
# Start ollama in the background if not running
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve > /dev/null 2>&1 &
    sleep 5
fi

echo "=== Step 3: Pulling Qwen2.5-7B model ==="
# This is a highly efficient model for reasoning and planning
ollama pull qwen2.5:7b-instruct-q4_K_M

echo "=== Step 4: Verifying Connection ==="
curl -s http://localhost:11434/api/tags | grep "qwen2.5"

echo "=== [SUCCESS] Ollama is ready for the Modular Agent! ==="
