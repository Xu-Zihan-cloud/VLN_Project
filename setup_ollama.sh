#!/bin/bash
# setup_ollama.sh - Robust local installation of Ollama using GitHub proxy

set -e

PROJECT_ROOT=$(pwd)
BIN_DIR="$PROJECT_ROOT/bin"
mkdir -p "$BIN_DIR"

echo "=== Step 1: Downloading Ollama Binary via Proxy ==="
if [ ! -f "$BIN_DIR/ollama" ]; then
    echo "Downloading Ollama Linux binary from GitHub proxy..."
    # Using ghfast.top proxy to bypass network restrictions
    wget -O "$BIN_DIR/ollama" https://ghfast.top/https://github.com/ollama/ollama/releases/download/v0.1.48/ollama-linux-amd64
    chmod +x "$BIN_DIR/ollama"
else
    echo "Ollama binary already exists in $BIN_DIR."
fi

# Add local bin to PATH for this session
export PATH="$BIN_DIR:$PATH"

echo "=== Step 2: Starting Ollama Service (Background) ==="
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama server..."
    # Use a custom port if 11434 is blocked, but default is usually fine
    ollama serve > "$PROJECT_ROOT/ollama.log" 2>&1 &
    sleep 10 # Give it time to initialize
fi

echo "=== Step 3: Pulling Qwen2.5-7B model ==="
# Using the locally installed ollama
ollama pull qwen2.5:7b-instruct-q4_K_M

echo "=== Step 4: Verifying Connection ==="
if curl -s http://localhost:11434/api/tags | grep -q "qwen2.5"; then
    echo "=== [SUCCESS] Ollama is ready! ==="
    echo "Note: I've installed ollama locally in $BIN_DIR"
    echo "To use it in other terminals, run: export PATH=\"$BIN_DIR:\$PATH\""
else
    echo "Error: Ollama server did not start correctly. Check $PROJECT_ROOT/ollama.log"
    exit 1
fi
