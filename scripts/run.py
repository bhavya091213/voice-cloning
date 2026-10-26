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
ap.add_argument("--ref-tag", default="", help="use reference/<speaker>/ref_<tag>.wav/.txt (made by enroll.py --tag)")
ap.add_argument("--cfg", type=float, default=None, help="VoxCPM guidance (default 2.0; lower = more varied)")
ap.add_argument("--vcref", default="", help="voice bundle (.vcref): regex-split the text into takes, each with its own intonation clip")
ap.add_argument("--stack", action="store_true", help="with --vcref: prompt = neutral clip + the tag's clip")
ap.add_argument("--no-master", action="store_true", help="skip mastering; output is the raw synthesis")
ap.add_argument("--title", default="", help="display name for the Studio library")
ap.add_argument("--timed", action="store_true", help="text-file has `m:ss - m:ss  line` timestamps; fit each line to its slot (voxcpm)")
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
if a.timed:
    if a.engine != "voxcpm":
        raise SystemExit("--timed needs --engine voxcpm")
    ENGINES["voxcpm"] = ("VoxCPM2 timed (48 kHz)", "timed_synth.py")
meta = {"job": a.job, "speaker": a.speaker, "engine": ENGINES[a.engine][0], "style": a.style, "state": "running", "timings": {},
        "title": a.title,
        "options": {"multi": bool(a.vcref), "timed": a.timed, "master": not a.no_master, "cfg": a.cfg, "ref_tag": a.ref_tag,
                    "vcref": Path(a.vcref).name if a.vcref else ""}}


def save(**kw):
    meta.update(kw)
    (jd / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


status.start_job(f"{a.job} · {a.speaker}")
save()
try:
    ref = ROOT / "reference" / a.speaker
    stem = f"ref_{a.ref_tag}" if a.ref_tag else "ref"
    if not (ref / f"{stem}.wav").exists():
        raise RuntimeError("speaker not enrolled")
    if a.cfg is not None:
        ENV["VOICECLONE_CFG"] = str(a.cfg)
    if a.vcref:
        if a.engine != "voxcpm":
            raise RuntimeError("--vcref needs --engine voxcpm")
        ENV["VOICECLONE_VCREF"] = str(Path(a.vcref).resolve())
        if a.stack:
            ENV["VOICECLONE_STACK"] = "1"

    t = time.time(); status.stage("Synthesize", "running", f"loading {ENGINES[a.engine][0]}")
    cmd = [PY, "-W", "ignore", str(ROOT / "scripts" / ENGINES[a.engine][1]), str(jd / "text.txt"),
           str(ref / f"{stem}.wav"), str(ref / f"{stem}.txt"), str(jd / "raw.wav")]
    p = subprocess.run(cmd,
                       env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout.splitlines():
        if line.startswith(("chunk", "peak", "sample")):
            status.log(line)
    if p.returncode:
        status.log("STDERR: " + p.stderr.strip()[-800:])
        raise RuntimeError("synthesis failed")
    meta["timings"]["synth"] = round(time.time() - t, 1); save()
    status.stage("Synthesize", "done", f"{meta['timings']['synth']}s")

    if a.no_master:
        # No processing: the synthesis as it is, with ONE constant gain so the peak sits at -3 dBFS. The model's raw output
        # is very quiet (about -20 dBFS peak), so an unscaled file is nearly unusable. The untouched file stays as raw.wav.
        import numpy as np
        import soundfile as sf
        x, sr_ = sf.read(jd / "raw.wav", dtype="float32")
        peak = float(np.abs(x).max()) or 1.0
        g_db = round(-3.0 - 20 * np.log10(peak), 1)
        sf.write(jd / "master.wav", x * 10 ** (g_db / 20), sr_, subtype="FLOAT")  # keep the output path stable
        save(raw_gain_db=g_db)
        status.stage("Master", "skipped", f"raw output, gain only {g_db:+.1f} dB (--no-master)")
    else:
        t = time.time(); status.stage("Master", "running", "de-ess · EQ · multiband · bus comp · limit")
        import master
        stats = master.master(str(jd / "raw.wav"), str(jd / "master.wav"), denoise=a.denoise, target_lufs=a.lufs)
        meta["timings"]["master"] = round(time.time() - t, 1); save(master=stats)
        status.stage("Master", "done", f"{stats['lufs']} LUFS · {stats['true_peak_db']} dBTP · {stats['gain_reduction_db']} dB GR")

    if (jd / "timing.json").exists():
        save(timing=json.loads((jd / "timing.json").read_text(encoding="utf-8")))
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
