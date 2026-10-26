"""VoiceClone Studio: create speech, manage voices, browse outputs, check the system. One small stdlib HTTP server.
Binds to localhost only; protected by an access key (cookie or ?key= or X-Key header). cloudflared tunnels to it.
Run: python app_server.py   -> http://127.0.0.1:8787/?key=<token in token.txt>

Pages   /  Studio (app.html) | /dashboard live pipeline | /architecture | /models
API     GET  status, voices, presets, outputs, doctor, job, tag     POST part, enroll, refs, generate, tag, doctor
Files   GET  /files/<reference|work/jobs|output path>   (whitelisted by FILE_RE, Range supported, ?dl=1 downloads)
"""
import glob
import hmac
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import paths  # noqa: E402

ROOT = paths.ROOT
import stats as dash  # noqa: E402  (snapshot() + gpu_loop())
import clean  # noqa: E402
import director  # noqa: E402  (TAGS only; the LLM runs in a subprocess)
import soundfile as sf  # noqa: E402

TOKEN_FILE = HERE / "token.txt"
if not TOKEN_FILE.exists():
    TOKEN_FILE.write_text(secrets.token_urlsafe(18))
TOKEN = TOKEN_FILE.read_text().strip()
PY = str(paths.VENV_PY)
FFMPEG = shutil.which("ffmpeg") or next(iter(glob.glob(
    str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/**/bin/ffmpeg.exe"), recursive=True)
    + glob.glob("/opt/homebrew/bin/ffmpeg") + glob.glob("/usr/local/bin/ffmpeg")), "ffmpeg")
MAX_BODY = 200 * 1024 * 1024
NAME_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
FILE_RE = re.compile(r"^(reference/[a-z0-9_-]+/(raw|clean|ref|ref_[a-z0-9_]+|prompt_15s|prompt_expressive)\.wav|"
                     r"reference/[a-z0-9_-]+/parts/[ABC]\.wav|reference/[a-z0-9_-]+/refs/[a-z0-9_-]+\.wav|"
                     r"work/jobs/job_[a-z0-9_]+/(raw|master)\.wav|output/[A-Za-z0-9_.-]+\.wav)$")
TIMED_LINE = re.compile(r"^\s*\d+:\d{1,2}(?:\.\d+)?\s*-\s*\d+:\d{1,2}(?:\.\d+)?\s+\S")
TEST_RE = re.compile(r"(test|regress|demo)", re.I)

tag_jobs = {}  # id -> {"state": running|done|error, "tagged": str, "error": str}
job_lock = threading.Lock()
current = {"proc": None, "id": None}


def busy():
    p = current["proc"]
    return p is not None and p.poll() is None


def cli_job_running():
    """A job started outside the Studio (vc.py / run.py) shows as a running stage in work/status.json.
    Ignore a stale file (crashed or killed run) after 10 minutes without an update."""
    f = ROOT / "work" / "status.json"
    try:
        st = json.loads(f.read_text(encoding="utf-8"))
        return time.time() - f.stat().st_mtime < 600 and any(s.get("state") == "running" for s in st.get("stages", []))
    except Exception:
        return False


def gpu_busy():
    return busy() or cli_job_running() or any(t["state"] == "running" for t in tag_jobs.values())


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def env():
    e = paths.child_env()
    e["PATH"] = str(Path(FFMPEG).parent) + os.pathsep + e.get("PATH", "")
    return e


# ---- data views -----------------------------------------------------------------------------------------------------
def profile_for(speaker):
    for f in sorted((ROOT / "profiles").glob("*.json")):
        d = read_json(f, {})
        if d.get("speaker") == speaker and not f.stem.startswith("example"):
            return f.stem, d
    return None, None


def voices():
    out = []
    for d in sorted((ROOT / "reference").glob("*")):
        if not d.is_dir():
            continue
        pname, prof = profile_for(d.name)
        idx = read_json(d / "refs" / "index.json", {}) or {}
        bundle = ROOT / "voices" / f"{d.name}.vcref"
        out.append({"name": d.name, "parts": [p for p in "ABC" if (d / "parts" / f"{p}.wav").exists()],
                    "enrolled": (d / "ref.wav").exists(),
                    "ref_long": (d / "ref_long.wav").exists(),
                    "clips": [{"tag": t, "seconds": m.get("seconds"), "label": m.get("label", ""), "text": m.get("text", "")}
                              for t, m in (idx.get("clips") or {}).items()],
                    "vcref_mb": round(bundle.stat().st_size / 1e6, 1) if bundle.exists() else None,
                    "profile": pname, "config": (prof or {}).get("config")})
    return out


def presets():
    return {f.stem: read_json(f, {}) for f in sorted((ROOT / "configs" / "presets").glob("*.json"))}


def outputs():
    rows = []
    for f in sorted((ROOT / "output").glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = read_json(ROOT / "work" / "jobs" / f.stem / "meta.json", {}) or {}
        opts = meta.get("options") or {}
        try:
            info = sf.info(f)
            secs, sr = round(info.duration, 1), info.samplerate
        except Exception:
            secs, sr = None, None
        raw = (opts.get("master") is False) or f.stem.endswith("_raw") or (bool(meta) and "master" not in meta and "options" not in meta)
        rows.append({"file": f.name, "title": meta.get("title") or f.stem, "mtime": f.stat().st_mtime,
                     "mb": round(f.stat().st_size / 1e6, 1), "seconds": secs, "sr": sr,
                     "speaker": meta.get("speaker"), "engine": meta.get("engine"), "style": meta.get("style"),
                     "multi": bool(opts.get("multi")), "timed": bool(opts.get("timed")), "raw": raw,
                     "cfg": opts.get("cfg"), "lufs": (meta.get("master") or {}).get("lufs"),
                     "peak": (meta.get("master") or {}).get("true_peak_db"),
                     "test": bool(TEST_RE.search(f.stem))})
    return rows


def run_doctor(fix):
    cmd = [PY, str(ROOT / "scripts" / "vc.py"), "doctor"] + (["--fix"] if fix else [])
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240, env=env())
    if p.returncode:
        raise RuntimeError((p.stderr.strip().splitlines() or ["doctor failed"])[-1])
    return json.loads(p.stdout)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:30]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    # ---- helpers
    def _send(self, code, body=b"", ct="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        """Returns (ok, set_cookie_header_or_None)."""
        key = parse_qs(urlparse(self.path).query).get("key", [None])[0] or self.headers.get("X-Key")
        ck = SimpleCookie(self.headers.get("Cookie", ""))
        cookie = ck["vc"].value if "vc" in ck else None
        if key and hmac.compare_digest(key, TOKEN):
            return True, f"vc={TOKEN}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=2592000"
        if cookie and hmac.compare_digest(cookie, TOKEN):
            return True, None
        return False, None

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            raise ValueError("body too large")
        return self.rfile.read(n)

    def _q(self):
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    # ---- GET
    def do_GET(self):
        ok, setck = self._auth()
        if not ok:
            return self._send(401, "<h3>Access key required.</h3>", "text/html")
        extra = {"Set-Cookie": setck} if setck else {}
        path = urlparse(self.path).path
        pages = {"/": "app.html", "/models": "models.html", "/architecture": "architecture.html", "/dashboard": "index.html"}
        if path in pages:
            return self._send(200, (HERE / pages[path]).read_bytes(), "text/html; charset=utf-8", extra)
        if path == "/api/status":
            return self._send(200, dash.snapshot())
        if path == "/api/voices":
            return self._send(200, voices())
        if path == "/api/presets":
            return self._send(200, {"presets": presets(), "tags": director.TAGS})
        if path == "/api/outputs":
            return self._send(200, outputs())
        if path == "/api/doctor":
            d = read_json(ROOT / "work" / "doctor.json")
            return self._send(200, d or {"checks": [], "ok": None, "checked": None})
        if path == "/api/job":
            jid = self._q().get("id")
            jobs = sorted((ROOT / "work" / "jobs").glob("job_*"), key=lambda p: p.stat().st_mtime)
            jd = (ROOT / "work" / "jobs" / jid) if jid and re.fullmatch(r"job_[a-z0-9_]+", jid) else (jobs[-1] if jobs else None)
            if not jd or not (jd / "meta.json").exists():
                return self._send(404, {"error": "no job"})
            return self._send(200, json.loads((jd / "meta.json").read_text(encoding="utf-8")))
        if path == "/api/tag":
            t = tag_jobs.get(self._q().get("id", ""))
            return self._send(200, t) if t else self._send(404, {"error": "no such tag job"})
        if path.startswith("/files/"):
            return self._file(path[len("/files/"):])
        self._send(404, {"error": "not found"})

    def _file(self, rel):
        if not FILE_RE.match(rel) or not (ROOT / rel).exists():
            return self._send(404, {"error": "not found"})
        data = (ROOT / rel).read_bytes()
        extra = {"Accept-Ranges": "bytes"}
        if self._q().get("dl"):
            extra["Content-Disposition"] = f'attachment; filename="{Path(rel).name}"'
        rng = self.headers.get("Range")
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
            a = int(m.group(1) or 0)
            b = int(m.group(2)) if m.group(2) else len(data) - 1
            return self._send(206, data[a:b + 1], "audio/wav", {**extra, "Content-Range": f"bytes {a}-{b}/{len(data)}"})
        self._send(200, data, "audio/wav", extra)

    # ---- POST
    def do_POST(self):
        ok, _ = self._auth()
        if not ok:
            return self._send(401, {"error": "key required"})
        path, q = urlparse(self.path).path, self._q()
        try:
            body = self._body()
            if path == "/api/part":
                return self._send(200, self._part(q, body))
            if path == "/api/enroll":
                return self._send(200, self._enroll(self._spk(q.get("speaker", ""))))
            if path == "/api/refs":
                return self._send(200, self._refs(self._spk(q.get("speaker", ""))))
            if path == "/api/doctor":
                return self._send(200, run_doctor(q.get("fix") == "1"))
            if path == "/api/generate":
                return self._generate(json.loads(body))
            if path == "/api/tag":
                return self._tag(json.loads(body))
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        self._send(404, {"error": "not found"})

    @staticmethod
    def _spk(name):
        name = name.strip().lower()
        if not NAME_RE.match(name):
            raise ValueError("speaker name: 1-32 chars of a-z 0-9 _ -")
        return name

    def _part(self, q, body):
        spk, part = self._spk(q.get("speaker", "")), q.get("part", "")
        if part not in ("A", "B", "C") or not body:
            raise ValueError("bad part/body")
        d = ROOT / "reference" / spk / "parts"
        d.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".in", dir=ROOT / "work") as f:
            f.write(body)
        try:
            r = subprocess.run([FFMPEG, "-y", "-v", "error", "-i", f.name, "-ac", "1", "-ar", "48000",
                                "-c:a", "pcm_f32le", str(d / f"{part}.wav")], capture_output=True, text=True)
        finally:
            Path(f.name).unlink(missing_ok=True)
        if r.returncode:
            raise ValueError("could not decode audio: " + r.stderr[-200:])
        x, sr = sf.read(d / f"{part}.wav", dtype="float32")
        return clean.part_stats(x, sr)

    def _enroll(self, spk):
        p = subprocess.run([PY, "-W", "ignore", str(ROOT / "scripts" / "enroll.py"), spk], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=env())
        if p.returncode:
            raise ValueError("enroll failed: " + (p.stderr.strip().splitlines() or [""])[-1])
        return json.loads(p.stdout.strip().splitlines()[-1])

    def _refs(self, spk):
        """Build the per-intonation clips and pack voices/<spk>.vcref (creates a default profile if the speaker has none)."""
        if not (ROOT / "reference" / spk / "parts").exists():
            raise ValueError("record or upload the parts first")
        with job_lock:
            if gpu_busy():
                return self._send(409, {"error": "the GPU is busy with another job"})
        pname, _ = profile_for(spk)
        if not pname:
            pname = spk
            narr = presets().get("narration", {"engine": "voxcpm", "style": "", "lufs": -16, "denoise": 0.0})
            (ROOT / "profiles").mkdir(exist_ok=True)
            (ROOT / "profiles" / f"{pname}.json").write_text(json.dumps(
                {"speaker": spk, "created": time.strftime("%Y-%m-%d"), "config": narr}, indent=2), encoding="utf-8")
        p = subprocess.run([PY, "-W", "ignore", str(ROOT / "scripts" / "vc.py"), "refs", pname], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=env(), timeout=600)
        if p.returncode:
            raise ValueError("building clips failed: " + (p.stderr.strip().splitlines() or p.stdout.strip().splitlines() or [""])[-1])
        return next(v for v in voices() if v["name"] == spk)

    def _tag(self, req):
        text = (req.get("text") or "").strip()
        direction = (req.get("style") or "").strip()[:300]
        spk = self._spk(req.get("speaker", "me"))
        if not text or len(text) > 20000:
            raise ValueError("text must be 1-20000 chars")
        if not direction:
            raise ValueError("write a direction first")
        tags = [c["tag"] for v in voices() if v["name"] == spk for c in v["clips"]] or director.TAGS
        with job_lock:
            if gpu_busy():
                return self._send(409, {"error": "the GPU is busy with another job"})
            tid = "tag_" + time.strftime("%Y%m%d_%H%M%S")
            td = ROOT / "work" / "tags"
            td.mkdir(parents=True, exist_ok=True)
            (td / f"{tid}.txt").write_text(text, encoding="utf-8")
            (td / f"{tid}.dir").write_text(direction, encoding="utf-8")
            tag_jobs[tid] = {"state": "running"}

        def run():
            try:
                cmd = [PY, "-W", "ignore", str(ROOT / "scripts" / "director.py"), str(td / f"{tid}.txt"), str(td / f"{tid}.dir"),
                       str(td / f"{tid}.json"), "--tags", ",".join(tags), "--emit", str(td / f"{tid}.out.txt")]
                p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env())
                if p.returncode:
                    raise RuntimeError((p.stderr.strip().splitlines() or ["director failed"])[-1])
                res = json.loads((td / f"{tid}.json").read_text(encoding="utf-8"))
                tag_jobs[tid] = {"state": "done", "tagged": (td / f"{tid}.out.txt").read_text(encoding="utf-8"),
                                 "fixes": res["fixes"], "fallback_chunks": res["fallback_chunks"],
                                 "lines": [{"tag": r["tag"], "form": r.get("form", ""), "contour": r.get("contour", ""), "function": r.get("function", ""), "stress": r.get("stress", []),
                                            "text": r["text"]} for r in res["lines"] if not r.get("music")]}
            except Exception as e:
                tag_jobs[tid] = {"state": "error", "error": str(e)}

        threading.Thread(target=run, daemon=True).start()
        self._send(200, {"id": tid})

    def _generate(self, req):
        spk = self._spk(req.get("speaker", ""))
        text = (req.get("text") or "").strip()
        if not text or len(text) > 20000:
            raise ValueError("text must be 1-20000 chars")
        if not (ROOT / "reference" / spk / "ref.wav").exists():
            raise ValueError("speaker not enrolled yet")
        engine = "qwen" if req.get("engine") == "qwen" else "voxcpm"
        timed = bool(req.get("timed")) or sum(bool(TIMED_LINE.match(ln)) for ln in text.splitlines()) >= 2
        multi = bool(req.get("multi"))
        if (timed or multi) and engine != "voxcpm":
            raise ValueError("timed scripts and intonation clips need the VoxCPM2 engine")
        if multi and not (ROOT / "voices" / f"{spk}.vcref").exists():
            raise ValueError("this voice has no intonation clips yet (Voices tab: Build intonation clips)")
        title = (req.get("title") or "").strip()[:80]
        with job_lock:
            if gpu_busy():
                return self._send(409, {"error": "a job is already running"})
            jid = "job_" + time.strftime("%Y%m%d_%H%M%S") + (f"_{slug(title)}" if slug(title) else "")
            jd = ROOT / "work" / "jobs" / jid
            jd.mkdir(parents=True, exist_ok=True)
            (jd / "in.txt").write_text(text, encoding="utf-8")
            cmd = [PY, str(ROOT / "scripts" / "run.py"), "--speaker", spk, "--text-file", str(jd / "in.txt"),
                   "--job", jid, "--engine", engine, "--style", (req.get("style") or "")[:300].replace("(", "").replace(")", ""),
                   "--denoise", str(float(req.get("denoise", 0.0))), "--lufs", str(float(req.get("lufs", -16)))]
            if timed:
                cmd.append("--timed")
            if multi:
                cmd += ["--vcref", str(ROOT / "voices" / f"{spk}.vcref")]
            if req.get("cfg") not in (None, ""):
                cmd += ["--cfg", str(float(req["cfg"]))]
            if req.get("master") is False:
                cmd.append("--no-master")
            if title:
                cmd += ["--title", title]
            (jd / "meta.json").write_text(json.dumps({"job": jid, "state": "running", "title": title}))
            current["proc"], current["id"] = subprocess.Popen(cmd, cwd=str(ROOT), env=env()), jid
        self._send(200, {"job": jid})


if __name__ == "__main__":
    threading.Thread(target=dash.gpu_loop, daemon=True).start()
    print("ffmpeg:", FFMPEG)
    ThreadingHTTPServer(("127.0.0.1", dash.PORT), H).serve_forever()
