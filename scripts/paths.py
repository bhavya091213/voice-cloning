"""Single source of truth for repo paths, platform and torch device. Stdlib only (torch is imported lazily).

ROOT is the repo folder (override with VOICECLONE_ROOT). Works on Windows (ROCm/CUDA) and macOS Apple Silicon (MPS).
"""
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("VOICECLONE_ROOT") or Path(__file__).resolve().parent.parent)
IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")
MODELS = ROOT / "models"
REFERENCE = ROOT / "reference"
WORK = ROOT / "work"
OUTPUT = ROOT / "output"


def child_env():
    """Environment for pipeline subprocesses."""
    env = {**os.environ, "HF_HOME": str(MODELS / "hf_cache"), "PYTHONIOENCODING": "utf-8"}
    if IS_MAC:
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # ops MPS lacks fall back to CPU instead of raising
    return env


def device():
    """cuda (NVIDIA or ROCm) > mps (Apple Silicon) > cpu. VOICECLONE_DEVICE overrides."""
    import torch
    forced = os.environ.get("VOICECLONE_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype(dev=None):
    """bfloat16 on GPU backends, float32 on CPU. VOICECLONE_DTYPE (float32|float16|bfloat16) overrides."""
    import torch
    name = os.environ.get("VOICECLONE_DTYPE")
    if name:
        return getattr(torch, name)
    return torch.float32 if (dev or device()) == "cpu" else torch.bfloat16


def peak_mem_gb():
    """Peak accelerator memory in GB where torch can report it, else None."""
    import torch
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e9, 1)
    if getattr(torch, "mps", None) and torch.backends.mps.is_available():
        return round(torch.mps.driver_allocated_memory() / 1e9, 1)
    return None
