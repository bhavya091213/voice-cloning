"""Live numbers for the pipeline dashboard: GPU / RAM, pipeline status and log, installed models, outputs, skill progress and
the last doctor report. Used by app_server.py (snapshot() for /api/status, gpu_loop() in a thread)."""
import ctypes
import json
import subprocess
import sys
import threading  # noqa: F401  (gpu_loop is started by the caller)
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS = ROOT / "work" / "status.json"
LOG = ROOT / "work" / "pipeline.log"
PORT = 8787
VRAM_TOTAL_GB = 16.0  # RX 6900 XT (Windows only; macOS has unified memory)

MODELS = [
    # name, role, params (read from the weight files, see /models), approx peak VRAM in GB (None = not measured), folder
    ("VoxCPM2", "Default TTS engine, 48 kHz", "2.38B", 6.7, ROOT / "models/voxcpm2"),
    ("Qwen3-TTS 1.7B Base", "Alternative TTS engine, 24 kHz", "2.10B", None, ROOT / "models/qwen3-tts-1.7b-base"),
    ("Qwen3-TTS Tokenizer 12Hz", "Speech codec for Qwen3-TTS", "-", None, ROOT / "models/qwen3-tts-tokenizer"),
    ("Whisper small", "Reference transcription and clip picking", "0.24B", None, ROOT / "models/whisper"),
    ("Qwen3-4B-Instruct-2507", "Director LLM (intonation tags)", "4.0B", None, ROOT / "models/qwen3-4b-instruct"),
]

state = {"gpu": {}, "t": 0}


def dir_gb(p):
    try:
        return round(sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9, 2)
    except Exception:
        return 0


def ram():
    if sys.platform != "win32":
        import psutil
        vm = psutil.virtual_memory()
        return {"used_gb": round((vm.total - vm.available) / 1e9, 1), "total_gb": round(vm.total / 1e9, 1)}

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                    ("tp", "ap", "tpf", "apf", "tv", "av", "ae")]
    m = MS(); m.l = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {"used_gb": round((m.tp - m.ap) / 1e9, 1), "total_gb": round(m.tp / 1e9, 1)}


PS = (r"$m=(Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples|Measure-Object CookedValue -Maximum;"
      r"$u=(Get-Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples|?{$_.InstanceName -match 'engtype_(3D|Compute)'}|Measure-Object CookedValue -Sum;"
      r"@{vram=$m.Maximum;util=$u.Sum}|ConvertTo-Json -Compress")


def gpu_loop():
    while True:
        if sys.platform != "win32":  # Apple Silicon: unified memory, no per-GPU counters from the shell
            state["gpu"] = {"error": "GPU stats unavailable on macOS (unified memory)"}
            time.sleep(5)
            continue
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", PS], capture_output=True, text=True, timeout=20).stdout
            d = json.loads(out)
            state["gpu"] = {"vram_used_gb": round(d["vram"] / 1e9, 2), "vram_total_gb": VRAM_TOTAL_GB,
                            "util": round(min(100, d["util"] or 0), 1)}
        except Exception as e:
            state["gpu"] = {"error": str(e)[:80]}
        time.sleep(2)


def _json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def snapshot():
    st = _json(STATUS, {"job": None, "stages": []})
    try:
        log = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
    except Exception:
        log = []
    models = [{"name": n, "role": r, "params": p, "peak_vram": v,
               "disk_gb": dir_gb(d) if Path(d).exists() else None} for n, r, p, v, d in MODELS]
    outs = sorted((ROOT / "output").glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)[:8]
    skill = _json(ROOT / "work" / "skill_progress.json")
    doc = _json(ROOT / "work" / "doctor.json")
    return {"time": time.strftime("%H:%M:%S"), "gpu": state["gpu"], "ram": ram(), "pipeline": st,
            "models": models, "log": log, "skill": skill,
            "doctor": ({"ok": doc["ok"], "checked": doc["checked"], "failing": [c["label"] for c in doc["checks"] if not c["ok"]]}
                       if doc else None),
            "outputs": [{"name": f.name, "mb": round(f.stat().st_size / 1e6, 1)} for f in outs]}
