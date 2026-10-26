"""python run.py --speaker NAME --text-file T --job ID [--denoise 0] [--lufs -16]
Pipeline: Qwen3-TTS clone (whole text, natural chunking) -> master -> output."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402
import status  # noqa: E402

ROOT = paths.ROOT
PY = str(paths.VENV_PY)
ENV = paths.child_env()

ap = argparse.ArgumentParser()
ap.add_argument("--speaker", required=True)
ap.add_argument("--text-file", required=True)
ap.add_argument("--job", required=True)
ap.add_argument("--engine", choices=["qwen", "voxcpm"], default="voxcpm")
ap.add_argument("--style", default="")
ap.add_argument("--denoise", type=float, default=0.0)
ap.add_argument("--lufs", type=float, default=-16.0)
a = ap.parse_args()

jd = ROOT / "work" / "jobs" / a.job
jd.mkdir(parents=True, exist_ok=True)
import re  # noqa: E402
_txt = Path(a.text_file).read_text(encoding="utf-8-sig")
if a.engine == "qwen":  # Qwen3-TTS Base has no delivery control: drop the (instruction) prefixes
    _txt = chr(10).join(re.sub(r"^\s*\([^()]{1,120}\)\s*", "", ln) for ln in _txt.splitlines())
(jd / "text.txt").write_text(_txt, encoding="utf-8")
ENGINES = {"qwen": ("Qwen3-TTS 1.7B (24 kHz)", "qwen_synth.py"), "voxcpm": ("VoxCPM2 (48 kHz)", "vox_synth.py")}
meta = {"job": a.job, "speaker": a.speaker, "engine": ENGINES[a.engine][0], "style": a.style, "state": "running", "timings": {}}


def save(**kw):
    meta.update(kw)
    (jd / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


status.start_job(f"{a.job} · {a.speaker}")
save()
try:
    ref = ROOT / "reference" / a.speaker
    if not (ref / "ref.wav").exists():
        raise RuntimeError("speaker not enrolled")

    t = time.time(); status.stage("Synthesize", "running", f"loading {ENGINES[a.engine][0]}")
    cmd = [PY, "-W", "ignore", str(ROOT / "scripts" / ENGINES[a.engine][1]), str(jd / "text.txt"),
           str(ref / "ref.wav"), str(ref / "ref.txt"), str(jd / "raw.wav")]
    p = subprocess.run(cmd,
                       env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout.splitlines():
        if line.startswith(("chunk", "peak")):
            status.log(line)
    if p.returncode:
        status.log("STDERR: " + p.stderr.strip()[-800:])
        raise RuntimeError("synthesis failed")
    meta["timings"]["synth"] = round(time.time() - t, 1); save()
    status.stage("Synthesize", "done", f"{meta['timings']['synth']}s")

    t = time.time(); status.stage("Master", "running", "de-ess · EQ · multiband · bus comp · limit")
    import master
    stats = master.master(str(jd / "raw.wav"), str(jd / "master.wav"), denoise=a.denoise, target_lufs=a.lufs)
    meta["timings"]["master"] = round(time.time() - t, 1); save(master=stats)
    status.stage("Master", "done", f"{stats['lufs']} LUFS · {stats['true_peak_db']} dBTP · {stats['gain_reduction_db']} dB GR")

    shutil.copy(jd / "master.wav", ROOT / "output" / f"{a.job}.wav")
    status.stage("Output", "done", f"output/{a.job}.wav")
    save(state="done")
except Exception as e:
    status.log(traceback.format_exc()[-600:])
    for s in json.loads(status.STATUS.read_text())["stages"]:
        if s["state"] == "running":
            status.stage(s["name"], "error", str(e))
    save(state="error", error=str(e))
    sys.exit(1)
