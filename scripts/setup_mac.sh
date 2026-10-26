#!/usr/bin/env bash
# One-shot setup for macOS on Apple Silicon (M1+). Run from anywhere: bash scripts/setup_mac.sh
# Installs system tools (Homebrew), the venv, Python deps, the two TTS packages and the model weights.
# Re-runnable: each step skips what already exists. Needs ~15 GB free disk for weights.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This script is for Apple Silicon Macs (found $(uname -s) $(uname -m))." >&2
  exit 1
fi
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh" >&2; exit 1; }

echo "==> system tools"
brew list python@3.11 >/dev/null 2>&1 || brew install python@3.11
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
brew list cloudflared >/dev/null 2>&1 || brew install cloudflared   # optional: phone access

echo "==> venv"
[[ -d .venv ]] || "$(brew --prefix python@3.11)/bin/python3.11" -m venv .venv
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install --upgrade pip

echo "==> torch (MPS) + dependencies"
"$PY" -m pip install torch torchaudio
"$PY" -m pip install -r requirements.txt

echo "==> TTS packages (isolated folders, no dependency resolution)"
[[ -d libs_qwen/qwen_tts ]] || "$PY" -m pip install --no-deps --target libs_qwen qwen-tts==0.1.1 transformers==4.57.3 tokenizers==0.22.2
[[ -d libs_vox/voxcpm ]]    || "$PY" -m pip install --no-deps --target libs_vox voxcpm==2.0.3

echo "==> model weights (Hugging Face)"
mkdir -p models
"$PY" -m pip install -q "huggingface_hub[cli]"
HF="$ROOT/.venv/bin/hf"; [[ -x "$HF" ]] || HF="$ROOT/.venv/bin/huggingface-cli"
[[ -f models/voxcpm2/model.safetensors ]]            || "$HF" download openbmb/VoxCPM2 --local-dir models/voxcpm2
[[ -f models/qwen3-tts-1.7b-base/model.safetensors ]] || "$HF" download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/qwen3-tts-1.7b-base
[[ -d models/qwen3-tts-tokenizer ]]                  || "$HF" download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/qwen3-tts-tokenizer
mkdir -p models/whisper   # Whisper "small" downloads itself on first enroll

echo "==> doctor"
"$PY" scripts/vc.py doctor --fix || true
echo
echo "Done. Start the Studio:  $PY dashboard/app_server.py   ->  http://127.0.0.1:8787/?key=<dashboard/token.txt>"
