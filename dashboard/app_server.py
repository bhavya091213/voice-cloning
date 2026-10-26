"""VoiceClone web app: record/upload reference, enroll, generate, play results, live pipeline dashboard.
Binds to localhost only; protected by an access key (cookie or ?key= or X-Key header). cloudflared tunnels to it.
Run: python app_server.py   -> http://127.0.0.1:8787/?key=<token in token.txt>
"""
import glob
import hmac
import json
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
ROOT = Path(r"D:\VoiceClone")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
import server as dash  # noqa: E402  (snapshot() + gpu_loop())
import clean  # noqa: E402
import soundfile as sf  # noqa: E402

TOKEN_FILE = HERE / "token.txt"
if not TOKEN_FILE.exists():
    TOKEN_FILE.write_text(secrets.token_urlsafe(18))
TOKEN = TOKEN_FILE.read_text().strip()
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
FFMPEG = shutil.which("ffmpeg") or next(iter(glob.glob(
    str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/**/bin/ffmpeg.exe"), recursive=True)), "ffmpeg")
MAX_BODY = 200 * 1024 * 1024
NAME_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
FILE_RE = re.compile(r"^(reference/[a-z0-9_-]+/(raw|clean|ref|prompt_15s|prompt_expressive)\.wav|"
                     r"reference/[a-z0-9_-]+/parts/[ABC]\.wav|work/jobs/job_[a-z0-9_]+/(raw|master)\.wav)$")

tag_jobs = {}  # id -> {"state": running|done|error, "tagged": str, "error": str}
job_lock = threading.Lock()
current = {"proc": None, "id": None}


def busy():
    p = current["proc"]
    return p is not None and p.poll() is None


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

    # ---- routes
    def do_GET(self):
        ok, setck = self._auth()
        if not ok:
            return self._send(401, "<h3>Access key required.</h3>", "text/html")
        extra = {"Set-Cookie": setck} if setck else {}
        path = urlparse(self.path).path
        if path == "/":
            return self._send(200, (HERE / "app.html").read_bytes(), "text/html; charset=utf-8", extra)
        if path == "/models":
            return self._send(200, (HERE / "models.html").read_bytes(), "text/html; charset=utf-8", extra)
        if path == "/architecture":
            return self._send(200, (HERE / "architecture.html").read_bytes(), "text/html; charset=utf-8", extra)
        if path == "/dashboard":
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8", extra)
        if path == "/api/status":
            return self._send(200, dash.snapshot())
        if path == "/api/speakers":
            out = []
            for d in sorted((ROOT / "reference").glob("*")):
                if d.is_dir():
                    out.append({"name": d.name, "parts": [p for p in "ABC" if (d / "parts" / f"{p}.wav").exists()],
                                "enrolled": (d / "ref.wav").exists()})
            return self._send(200, out)
        if path == "/api/jobs":
            out = []
            for jd in sorted((ROOT / "work" / "jobs").glob("job_*"), key=lambda p: p.stat().st_mtime, reverse=True):
                if (jd / "master.wav").exists() and (jd / "meta.json").exists():
                    m = json.loads((jd / "meta.json").read_text(encoding="utf-8"))
                    out.append({"job": jd.name, "engine": m.get("engine"), "style": m.get("style"), "speaker": m.get("speaker"),
                                "text": (jd / "text.txt").read_text(encoding="utf-8")[:160] if (jd / "text.txt").exists() else "",
                                "master": m.get("master")})
                if len(out) >= 30:
                    break
            return self._send(200, out)
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
        rng = self.headers.get("Range")
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
            a = int(m.group(1) or 0)
            b = int(m.group(2)) if m.group(2) else len(data) - 1
            return self._send(206, data[a:b + 1], "audio/wav",
                              {"Content-Range": f"bytes {a}-{b}/{len(data)}", "Accept-Ranges": "bytes"})
        self._send(200, data, "audio/wav", {"Accept-Ranges": "bytes"})

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
                spk = self._spk(q.get("speaker", ""))
                p = subprocess.run([PY, "-W", "ignore", str(ROOT / "scripts" / "enroll.py"), spk], capture_output=True,
                                   text=True, encoding="utf-8", errors="replace",
                                   env={**__import__("os").environ, "HF_HOME": str(ROOT / "models" / "hf_cache"),
                                        "PATH": str(Path(FFMPEG).parent) + ";" + __import__("os").environ["PATH"]})
                if p.returncode:
                    raise ValueError("enroll failed: " + (p.stderr.strip().splitlines() or [""])[-1])
                return self._send(200, json.loads(p.stdout.strip().splitlines()[-1]))
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

    def _tag(self, req):
        text = (req.get("text") or "").strip()
        direction = (req.get("style") or "").strip()[:300]
        if not text or len(text) > 20000:
            raise ValueError("text must be 1-20000 chars")
        if not direction:
            raise ValueError("write a style direction first")
        with job_lock:
            if busy() or any(t["state"] == "running" for t in tag_jobs.values()):
                return self._send(409, {"error": "the GPU is busy with another job"})
            tid = "tag_" + time.strftime("%Y%m%d_%H%M%S")
            td = ROOT / "work" / "tags"
            td.mkdir(parents=True, exist_ok=True)
            (td / f"{tid}.txt").write_text(text, encoding="utf-8")
            (td / f"{tid}.dir").write_text(direction, encoding="utf-8")
            tag_jobs[tid] = {"state": "running"}

        def run():
            try:
                p = subprocess.run([PY, "-W", "ignore", str(ROOT / "scripts" / "tag_cli.py"), str(td / f"{tid}.txt"),
                                    str(td / f"{tid}.dir"), str(td / f"{tid}.json")], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   env={**__import__("os").environ, "HF_HOME": str(ROOT / "models" / "hf_cache")})
                if p.returncode:
                    raise RuntimeError((p.stderr.strip().splitlines() or ["tagger failed"])[-1])
                tag_jobs[tid] = {"state": "done", **json.loads((td / f"{tid}.json").read_text(encoding="utf-8"))}
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
        with job_lock:
            if busy() or any(t["state"] == "running" for t in tag_jobs.values()):
                return self._send(409, {"error": "a job is already running"})
            jid = "job_" + time.strftime("%Y%m%d_%H%M%S")
            jd = ROOT / "work" / "jobs" / jid
            jd.mkdir(parents=True, exist_ok=True)
            (jd / "in.txt").write_text(text, encoding="utf-8")
            cmd = [PY, str(ROOT / "scripts" / "run.py"), "--speaker", spk, "--text-file", str(jd / "in.txt"),
                   "--job", jid, "--engine", "qwen" if req.get("engine") == "qwen" else "voxcpm",
                   "--style", (req.get("style") or "")[:300].replace("(", "").replace(")", ""),
                   "--denoise", str(float(req.get("denoise", 0.0))), "--lufs", str(float(req.get("lufs", -16)))]
            (jd / "meta.json").write_text(json.dumps({"job": jid, "state": "running"}))
            current["proc"], current["id"] = subprocess.Popen(cmd, cwd=str(ROOT)), jid
        self._send(200, {"job": jid})


if __name__ == "__main__":
    threading.Thread(target=dash.gpu_loop, daemon=True).start()
    print("ffmpeg:", FFMPEG)
    ThreadingHTTPServer(("127.0.0.1", dash.PORT), H).serve_forever()
