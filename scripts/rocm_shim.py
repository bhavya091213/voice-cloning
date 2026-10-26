"""Import this before transformers/indextts. (On macOS only the torchaudio and distributed stubs matter.)

The Windows ROCm torch build has no torch.distributed, but some transformers versions import
torch.distributed.tensor unconditionally. We never use distributed features, so stub it out.
"""
import sys
import types

import torch

# MIOpen on Windows/gfx1030 throws miopenStatusUnknownError (batch_norm etc.); use native kernels.
if sys.platform == "win32":
    torch.backends.cudnn.enabled = False

# torchaudio 2.9 routes load/save through torchcodec; use soundfile instead.
import soundfile as _sf
import torchaudio as _ta


def _save(path, src, sample_rate, **kw):
    _sf.write(str(path), src.detach().cpu().float().numpy().T if src.dim() > 1 else src.detach().cpu().float().numpy(),
              sample_rate)


def _load(path, **kw):
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


_ta.save, _ta.load = _save, _load

if not hasattr(torch.distributed, "ReduceOp"):
    class _ReduceOp:
        SUM, AVG, PRODUCT, MIN, MAX = "sum", "avg", "product", "min", "max"
    torch.distributed.ReduceOp = _ReduceOp

if not torch.distributed.is_available() or "torch.distributed.tensor" not in sys.modules:
    try:
        import torch.distributed.tensor  # noqa: F401
    except Exception:
        stub = types.ModuleType("torch.distributed.tensor")
        stub.DTensor = type("DTensor", (), {})
        stub.Replicate = stub.Shard = stub.Placement = type("Placement", (), {})
        sys.modules["torch.distributed.tensor"] = stub
        torch.distributed.tensor = stub
