"""Tiny helper the pipeline uses to report progress to the dashboard."""
import json
import time
from pathlib import Path

STATUS = Path(r"D:\VoiceClone\work\status.json")
LOG = Path(r"D:\VoiceClone\work\pipeline.log")
STAGES = ["Synthesize", "Master", "Output"]


def start_job(name: str):
    LOG.write_text("", encoding="utf-8")
    _write({"job": name, "started": time.time(), "stages": [{"name": s, "state": "pending", "detail": ""} for s in STAGES]})
    log(f"job started: {name}")


def stage(name: str, state: str, detail: str = ""):
    """state: pending | running | done | error"""
    d = json.loads(STATUS.read_text(encoding="utf-8"))
    for s in d["stages"]:
        if s["name"] == name:
            s["state"], s["detail"] = state, detail
    _write(d)
    log(f"{name}: {state} {detail}")


def log(msg: str):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')}  {msg}\n")


def _write(d):
    STATUS.write_text(json.dumps(d), encoding="utf-8")
