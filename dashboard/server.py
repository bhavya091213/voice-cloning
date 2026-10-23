"""Read-only dashboard server. Binds to localhost only; cloudflared tunnels to it.
Run: python server.py   (serves http://127.0.0.1:8787)
"""
import ctypes
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(r"D:\VoiceClone")
STATUS = ROOT / "work" / "status.json"
LOG = ROOT / "work" / "pipeline.log"
PORT = 8787
VRAM_TOTAL_GB = 16.0  # RX 6900 XT

MODELS = [
    # name, role, approx params, approx weights on disk (GB), est. min VRAM (GB), dir
    ("IndexTTS-2.5 (GPT + S2Mel DiT + codec + BigVGAN)", "Voice-cloning TTS", "~1-2B total (est.)", 5.1, 6.3, ROOT / "models/index-tts/checkpoints"),
    ("Qwen3-4B-Instruct-2507", "Emotion/pace tagger LLM", "4.0B", 7.5, 8.3, ROOT / "models/qwen3-4b-instruct"),
    ("Qwen 0.6B emo (bundled)", "Text-emotion helper (optional)", "0.6B", 1.2, 2, None),
    ("w2v-BERT 2.0 (auto-fetched)", "Speaker/semantic features", "~0.6B", 2.3, 2, None),
    ("Enhancer (TBD)", "Reference cleanup", "<0.1B", 0.1, 1, None),
]

state = {"gpu": {}, "t": 0}


def dir_gb(p):
    try:
        return round(sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9, 2)
    except Exception:
        return 0


def ram():
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
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", PS], capture_output=True, text=True, timeout=20).stdout
            d = json.loads(out)
            state["gpu"] = {"vram_used_gb": round(d["vram"] / 1e9, 2), "vram_total_gb": VRAM_TOTAL_GB,
                            "util": round(min(100, d["util"] or 0), 1)}
        except Exception as e:
            state["gpu"] = {"error": str(e)[:80]}
        time.sleep(2)


def snapshot():
    try:
        st = json.loads(STATUS.read_text(encoding="utf-8"))
    except Exception:
        st = {"job": None, "stages": []}
    try:
        log = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
    except Exception:
        log = []
    models = [{"name": n, "role": r, "params": p, "disk_gb": g, "min_vram": v,
               "downloaded_gb": dir_gb(d) if d else None} for n, r, p, g, v, d in MODELS]
    outs = sorted((ROOT / "output").glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)[:8]
    return {"time": time.strftime("%H:%M:%S"), "gpu": state["gpu"], "ram": ram(), "pipeline": st,
            "models": models, "log": log,
            "outputs": [{"name": f.name, "mb": round(f.stat().st_size / 1e6, 1)} for f in outs]}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path.startswith("/api/status"):
            body, ct = json.dumps(snapshot()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, ct = (Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8"
        else:
            self.send_error(404); return
        self.send_response(200)
        self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    threading.Thread(target=gpu_loop, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

