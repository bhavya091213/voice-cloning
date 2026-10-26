"""Reusable voice profiles + phase tracking for the VoiceClone skill.
python vc.py preflight | doctor [--fix] | profile list|show|create|set | enroll NAME | generate NAME --text-file T --job ID | progress init|set|show
Stdlib only. Profiles point at reference/<speaker>/ (never copied or overwritten); defaults live in profiles/<name>.json."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402

ROOT = paths.ROOT
PY = str(paths.VENV_PY)
IS_WIN, IS_MAC = paths.IS_WIN, paths.IS_MAC
PROFILES, PRESETS, REF = ROOT / "profiles", ROOT / "configs" / "presets", ROOT / "reference"
PROGRESS = ROOT / "work" / "skill_progress.json"
PHASES = ["Preflight", "Profile", "Enroll", "Config", "Script", "Generate", "Review"]
STATES = ("pending", "running", "done", "error", "skipped")
DEFAULTS = {"engine": "voxcpm", "style": "", "lufs": -16.0, "denoise": 0.0}


def out(o):
    print(json.dumps(o, indent=2, ensure_ascii=False))


def load_progress():
    return json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else None


def save_progress(d):
    PROGRESS.parent.mkdir(exist_ok=True)
    d["updated"] = time.time()
    PROGRESS.write_text(json.dumps(d), encoding="utf-8")


def p_init(label=""):
    save_progress({"label": label, "started": time.time(), "inputs": {},
                   "phases": [{"id": i + 1, "name": n, "state": "pending", "detail": ""} for i, n in enumerate(PHASES)]})


def p_set(phase, state, detail="", **inputs):
    d = load_progress()
    if d is None:
        p_init()
        d = load_progress()
    for ph in d["phases"]:
        if str(ph["id"]) == str(phase) or ph["name"].lower() == str(phase).lower():
            ph["state"], ph["detail"] = state, detail
    d["inputs"].update({k: v for k, v in inputs.items() if v})
    save_progress(d)


def profile_path(name):
    return PROFILES / f"{name}.json"


def get_profile(name):
    p = profile_path(name)
    if not p.exists():
        raise SystemExit(f"no profile '{name}' (see: profile list)")
    return json.loads(p.read_text(encoding="utf-8"))


def enrolled(speaker):
    return (REF / speaker / "ref.wav").exists() and (REF / speaker / "ref.txt").exists()


def cmd_preflight(_):
    import urllib.request
    checks = {"venv python": Path(PY).exists(), "ffmpeg": shutil.which("ffmpeg") is not None,
              "weights voxcpm2": (ROOT / "models" / "voxcpm2").exists(),
              "weights qwen3-tts": (ROOT / "models" / "qwen3-tts-1.7b-base").exists(),
              "weights whisper": (ROOT / "models" / "whisper").exists()}
    try:
        urllib.request.urlopen("http://127.0.0.1:8787/", timeout=2)
        checks["studio server :8787"] = True
    except Exception as e:
        checks["studio server :8787"] = hasattr(e, "code")  # an HTTP error still means it is up
    out(checks)
    bad = [k for k, v in checks.items() if not v]
    p_set(1, "error" if bad else "done", ("missing: " + ", ".join(bad)) if bad else "all checks passed")


def cmd_profile(a):
    PROFILES.mkdir(exist_ok=True)
    if a.action == "list":
        rows = []
        for f in sorted(PROFILES.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"name": f.stem, "speaker": d["speaker"], "enrolled": enrolled(d["speaker"]), **d["config"]})
        out({"profiles": rows,
             "unregistered_speakers": [s.name for s in REF.iterdir() if s.is_dir() and not profile_path(s.name).exists()],
             "presets": [f.stem for f in PRESETS.glob("*.json")]})
        return
    if not a.name:
        raise SystemExit("profile name required")
    if a.action == "show":
        d = get_profile(a.name)
        ok = enrolled(d["speaker"])
        out({**d, "enrolled": ok, "ref_text": (REF / d["speaker"] / "ref.txt").read_text(encoding="utf-8-sig") if ok else None})
        return
    p = profile_path(a.name)  # create | set
    if a.action == "create" and p.exists():
        raise SystemExit(f"profile '{a.name}' exists; use 'profile set'")
    if a.action == "set" and not p.exists():
        raise SystemExit(f"no profile '{a.name}'")
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"speaker": a.speaker or a.name, "created": time.strftime("%Y-%m-%d"), "config": dict(DEFAULTS)}
    if a.speaker:
        d["speaker"] = a.speaker
    if a.preset:
        d["config"].update(json.loads((PRESETS / f"{a.preset}.json").read_text(encoding="utf-8")))
    for k in DEFAULTS:
        v = getattr(a, k)
        if v is not None:
            d["config"][k] = v
    sp = REF / d["speaker"]
    for i, src in enumerate(a.audio or []):
        dst = sp / "parts" / f"{'ABC'[i]}.wav"
        if dst.exists():
            raise SystemExit(f"refusing to overwrite {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "48000", "-c:a", "pcm_f32le", str(dst)], check=True)
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    p_set(2, "done", f"{a.name} -> reference/{d['speaker']}", profile=a.name)
    if a.action == "set" or a.preset or any(getattr(a, k) is not None for k in DEFAULTS):
        p_set(4, "done", ", ".join(f"{k}={v}" for k, v in d["config"].items()))
    out(d)


def cmd_enroll(a):
    d = get_profile(a.name)
    if enrolled(d["speaker"]) and not a.force:
        p_set(3, "skipped", "already enrolled (reused)")
        return out({"enrolled": True, "reused": True})
    p_set(3, "running", "Whisper picking best 9-20 s window")
    r = subprocess.run([PY, "-W", "ignore", str(ROOT / "scripts" / "enroll.py"), d["speaker"]],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0 and enrolled(d["speaker"])
    p_set(3, "done" if ok else "error", "reference ready" if ok else r.stderr.strip()[-300:])
    print(r.stdout[-1500:])
    sys.exit(0 if ok else 1)


def cmd_generate(a):
    d = get_profile(a.name)
    c = {**d["config"], **{k: getattr(a, k) for k in DEFAULTS if getattr(a, k) is not None}}
    p_set(6, "running", f"{a.job} · {c['engine']}", job=a.job)
    cmd = [PY, str(ROOT / "scripts" / "run.py"), "--speaker", d["speaker"], "--text-file", a.text_file, "--job", a.job,
           "--engine", c["engine"], "--style", c["style"], "--denoise", str(c["denoise"]), "--lufs", str(c["lufs"])]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wav = ROOT / "output" / f"{a.job}.wav"
    p_set(6, "done" if wav.exists() else "error", str(wav) if wav.exists() else (r.stderr.strip()[-300:] or "see work/pipeline.log"))
    out({"output": str(wav) if wav.exists() else None, "config": c})
    sys.exit(0 if wav.exists() else 1)


def cmd_progress(a):
    if a.action == "init":
        p_init(a.label or "")
    elif a.action == "set":
        p_set(a.phase, a.state, a.detail or "")
    out(load_progress())


def _up():
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:8787/", timeout=2)
        return True
    except Exception as e:
        return hasattr(e, "code") or "Access key" in str(e)


def _tunnel():
    import re
    if IS_WIN:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq cloudflared.exe", "/NH"], capture_output=True, text=True)
    else:
        r = subprocess.run(["pgrep", "-x", "cloudflared"], capture_output=True, text=True)
        r.stdout = "cloudflared" if r.returncode == 0 else ""
    log = ROOT / "work" / "tunnel.log"
    urls = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", log.read_text(errors="replace")) if log.exists() else []
    return "cloudflared" in r.stdout.lower(), (urls[-1] if urls else "")


def cmd_doctor(a):
    """Dependency check; --fix applies safe repairs (create dirs, start server/tunnel). Never installs or deletes."""
    flags = (0x08000000 | 0x00000008) if IS_WIN else 0  # CREATE_NO_WINDOW | DETACHED_PROCESS
    detach = {} if IS_WIN else {"start_new_session": True}
    res = []

    def add(id_, label, ok, detail="", fix=""):
        res.append({"id": id_, "label": label, "ok": bool(ok), "detail": detail, "fix": fix})

    add("venv", "Python venv", Path(PY).exists(), PY, "recreate .venv (see README Setup)")
    if Path(PY).exists():
        names = ["numpy", "soundfile", "whisper", "torch", "torchaudio", "transformers"]
        code = "import importlib.util as u,json;print(json.dumps({n:u.find_spec(n) is not None for n in %r}))" % names
        r = subprocess.run([PY, "-I", "-c", code], capture_output=True, text=True)
        try:
            found = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            found = {}
        for n in names:
            add("pkg-" + n, f"package {n}", found.get(n), "", f"{PY} -m pip install {n}")
        if found.get("torch"):
            r = subprocess.run([PY, "-c", "import torch;c=torch.cuda.is_available();m=torch.backends.mps.is_available();print(torch.__version__, c, torch.cuda.get_device_name(0) if c else ('mps' if m else '-'))"],
                               capture_output=True, text=True, timeout=120)
            line = (r.stdout.strip().splitlines() or ["?"])[-1]
            add("gpu", "GPU (MPS)" if IS_MAC else "GPU (ROCm torch)", (" True " in f" {line} ") or "mps" in line, line, "reinstall torch (macOS: pip install torch torchaudio; Windows: AMD gfx103X ROCm index)")
    for lib, mod in (("libs_qwen", "qwen_tts"), ("libs_vox", "voxcpm"), ("libs_llm", "transformers")):
        add("lib-" + lib, f"{lib}/{mod}", (ROOT / lib / mod).exists(), str(ROOT / lib), f"pip install --target {ROOT / lib} <pkg>")
    add("ffmpeg", "ffmpeg on PATH", shutil.which("ffmpeg"), shutil.which("ffmpeg") or "", "brew install ffmpeg" if IS_MAC else "winget install ffmpeg")
    for w in ("voxcpm2", "qwen3-tts-1.7b-base", "qwen3-tts-tokenizer", "whisper"):
        d = ROOT / "models" / w
        add("w-" + w, f"weights {w}", d.exists() and any(d.iterdir()), str(d), "download the model into models/")
    free = shutil.disk_usage(ROOT).free / 1e9
    add("disk", "Disk free on D:", free > 10, f"{free:.0f} GB", "free up space; jobs write work/ and output/")
    for d_ in ("work", "output", "profiles", "reference", "configs/presets"):
        if not (ROOT / d_).exists() and a.fix:
            (ROOT / d_).mkdir(parents=True, exist_ok=True)
        add("dir-" + d_.replace("/", "-"), f"folder {d_}", (ROOT / d_).exists(), "", "doctor --fix creates it")
    up = _up()
    if not up and a.fix:
        subprocess.Popen([PY, str(ROOT / "dashboard" / "app_server.py")], cwd=str(ROOT), creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **detach)
        time.sleep(6)
        up = _up()
    add("server", "Studio server :8787", up, "", "doctor --fix starts it")
    run, url = _tunnel()
    if not run and a.fix and shutil.which("cloudflared") and up:
        subprocess.Popen(["cloudflared", "tunnel", "--url", "http://127.0.0.1:8787", "--no-autoupdate", "--protocol", "http2", "--logfile", str(ROOT / "work" / "tunnel.log")],
                         creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **detach)
        time.sleep(10)
        run, url = _tunnel()
    add("tunnel", "Cloudflare tunnel", run, url, "doctor --fix starts it (new URL each time)")
    tok = ROOT / "dashboard" / "token.txt"
    add("token", "Studio access key", tok.exists(), "", "start the server once to generate it")
    for f in sorted(PROFILES.glob("*.json")) if PROFILES.exists() else []:
        if f.stem.startswith("example"):
            continue
        sp = json.loads(f.read_text(encoding="utf-8"))["speaker"]
        add("prof-" + f.stem, f"profile {f.stem} enrolled", enrolled(sp), f"reference/{sp}", f"vc.py enroll {f.stem}")
    summary = {"checked": time.time(), "ok": all(r["ok"] for r in res), "link": f"{url}/?key={tok.read_text().strip()}" if url and tok.exists() else "", "checks": res}
    (ROOT / "work").mkdir(exist_ok=True)
    (ROOT / "work" / "doctor.json").write_text(json.dumps(summary), encoding="utf-8")
    out(summary)


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
sub.add_parser("preflight").set_defaults(f=cmd_preflight)
pp = sub.add_parser("profile")
pp.set_defaults(f=cmd_profile)
pp.add_argument("action", choices=["list", "show", "create", "set"])
pp.add_argument("name", nargs="?")
pp.add_argument("--speaker")
pp.add_argument("--audio", nargs="+")
pp.add_argument("--preset")
pp.add_argument("--engine", choices=["qwen", "voxcpm"])
pp.add_argument("--style")
pp.add_argument("--lufs", type=float)
pp.add_argument("--denoise", type=float)
pe = sub.add_parser("enroll")
pe.set_defaults(f=cmd_enroll)
pe.add_argument("name")
pe.add_argument("--force", action="store_true")
pg = sub.add_parser("generate")
pg.set_defaults(f=cmd_generate)
pg.add_argument("name")
pg.add_argument("--text-file", required=True)
pg.add_argument("--job", required=True)
pg.add_argument("--engine", choices=["qwen", "voxcpm"])
pg.add_argument("--style")
pg.add_argument("--lufs", type=float)
pg.add_argument("--denoise", type=float)
pd = sub.add_parser("doctor")
pd.set_defaults(f=cmd_doctor)
pd.add_argument("--fix", action="store_true")
pr = sub.add_parser("progress")
pr.set_defaults(f=cmd_progress)
pr.add_argument("action", choices=["init", "set", "show"])
pr.add_argument("phase", nargs="?")
pr.add_argument("state", nargs="?", choices=STATES)
pr.add_argument("detail", nargs="?")
pr.add_argument("--label")
a = ap.parse_args()
a.f(a)
