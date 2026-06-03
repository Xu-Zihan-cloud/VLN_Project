# VLN Project: ALFRED/AlfWorld with AI2-THOR

This project implements a Visual Language Navigation (VLN) agent using the ALFRED/AlfWorld dataset.

## Architecture
- **Framework:** PyTorch Lightning + Hydra + UV
- **Environment:** AI2-THOR
- **Models:**
  - End-to-End: Episodic Transformer (E.T.)
  - Modular: LLM Planner (Qwen2.5-7B via Ollama) + Perception + Deterministic Policy

## Setup
```bash
bash setup_env.sh
conda activate vln-alfred
```
