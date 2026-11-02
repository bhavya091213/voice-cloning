"""python restitch.py JOB [--out NAME] [--gap 0.45] [--swap N=path.wav ...] [--no-master]
Re-assemble a finished vox_synth render without re-synthesizing it: split raw.wav at its exact-zero chunk gaps, trim the
model's hiss/silence pre- and post-roll off every chunk (that hiss is the "static" the mastering gain brings up), shorten
over-long pauses inside a chunk, fade the edges, join with short punctuation-aware gaps, then run master.py.
--swap N=file replaces chunk N (1-based) with a regenerated take (see regen_chunk.py)."""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402

FR = 0.01


def split_chunks(x, sr, min_zero=0.1):
    z = np.abs(x) < 1e-7
    d = np.diff(np.concatenate([[0], z.astype(np.int8), [0]]))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    cuts = [(a, b) for a, b in zip(s, e) if b - a >= min_zero * sr]
    out, pos = [], 0
    for a, b in cuts:
        out.append(x[pos:a])
        pos = b
    out.append(x[pos:])
    return [c for c in out if len(c)]


def frames_db(x, sr):
    n = int(sr * FR)
    k = len(x) // n
    return 20 * np.log10(np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-9)


def _runs(m):
    out, i = [], 0
    while i < len(m):
        j = i
        while j < len(m) and m[j] == m[i]:
            j += 1
        out.append((i, j, bool(m[i])))
        i = j
    return out


def speech_frames(x, sr, rel=-13.0, min_run=5, hole=4):
    """Speech = frames within `rel` dB of the loud frames, in runs of at least min_run frames (isolated hiss blips are
    not speech); holes of up to `hole` frames (stop closures) are bridged so words stay whole."""
    db = frames_db(x, sr)
    m = db > np.percentile(db, 95) + rel
    for a, b, v in _runs(m):
        if v and b - a < min_run:
            m[a:b] = False
    for a, b, v in _runs(m):
        if not v and b - a <= hole and a > 0 and b < len(m):
            m[a:b] = True
    return m


def tidy(x, sr, pre=0.30, post=0.35, max_pause=0.60, keep=0.40, fi=0.10, fo=0.28):
    """Cut the pre-roll hiss, keep a long tapering tail (quiet word endings sit at the hiss level), shorten dead holes."""
    m = speech_frames(x, sr)
    idx = np.where(m)[0]
    if not idx.size:
        return x, 0.0
    t0 = max(0.0, idx[0] * FR - pre)
    a = int(t0 * sr)
    b = min(len(x), int(((idx[-1] + 1) * FR + post) * sr))
    y = x[a:b].copy()
    # dead holes inside the chunk: frames well below speech level for longer than max_pause
    db = frames_db(y, sr)
    dead = db < np.percentile(db, 95) - 21
    f = int(0.015 * sr)
    pieces, last, removed = [], 0, 0.0
    for p, q, v in _runs(dead):
        if v and (q - p) * FR > max_pause and p > int(0.2 / FR) and q < len(dead) - int(0.2 / FR):
            ca, cb = int((p * FR + keep / 2) * sr), int((q * FR - keep / 2) * sr)
            if cb - ca > 2 * f:
                pieces.append(y[last:ca])
                last = cb
                removed += (cb - ca) / sr
    if pieces:
        pieces.append(y[last:])
        out = pieces[0]
        for pc in pieces[1:]:
            ramp = np.linspace(0, 1, f, dtype=np.float32)
            out[-f:] = out[-f:] * (1 - ramp) + pc[:f] * ramp
            out = np.concatenate([out, pc[f:]])
        y = out
    nfi, nfo = int(fi * sr), min(int(fo * sr), len(y))
    y[:nfi] *= np.linspace(0, 1, nfi, dtype=np.float32)
    y[-nfo:] *= np.linspace(1, 0, nfo, dtype=np.float32) ** 1.5
    return y, removed


def gap_after(text, base):
    t = text.strip()
    if t.endswith(","):
        return base * 0.6
    if t.endswith(("?", ":")):
        return base * 1.1
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--out", default="")
    ap.add_argument("--gap", type=float, default=0.45)
    ap.add_argument("--swap", action="append", default=[])
    ap.add_argument("--no-master", action="store_true")
    ap.add_argument("--lufs", type=float, default=-16.0)
    a = ap.parse_args()
    jd = paths.ROOT / "work" / "jobs" / a.job
    x, sr = sf.read(jd / "raw.wav", dtype="float32")
    chunks = split_chunks(x, sr)
    lines = [l for l in (jd / "text.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(chunks)} chunks, {len(lines)} script lines")
    for s in a.swap:
        n, p = s.split("=", 1)
        w, wsr = sf.read(p, dtype="float32")
        assert wsr == sr
        chunks[int(n) - 1] = w
    texts = [re.sub(r"^\s*\([^()]*\)\s*", "", l) for l in lines]
    if len(texts) != len(chunks):
        texts = [""] * len(chunks)
    parts, tot_old = [], len(x) / sr
    for i, c in enumerate(chunks):
        y, rem = tidy(c, sr)
        parts.append(y)
        if i < len(chunks) - 1:
            parts.append(np.zeros(int(sr * gap_after(texts[i], a.gap)), np.float32))
    y = np.concatenate(parts)
    name = a.out or a.job + "_clean"
    od = paths.ROOT / "work" / "jobs" / name
    od.mkdir(parents=True, exist_ok=True)
    sf.write(od / "raw.wav", y, sr, subtype="FLOAT")
    shutil.copy(jd / "text.txt", od / "text.txt")
    print(f"{tot_old:.1f}s -> {len(y) / sr:.1f}s")
    meta = json.loads((jd / "meta.json").read_text(encoding="utf-8"))
    meta.update(job=name, title=(meta.get("title") or a.job) + " - restitched (trimmed hiss, tight gaps)", state="done")
    if not a.no_master:
        import master
        meta["master"] = master.master(str(od / "raw.wav"), str(od / "master.wav"), denoise=0.0, target_lufs=a.lufs)
        print(meta["master"])
        shutil.copy(od / "master.wav", paths.ROOT / "output" / f"{name}.wav")
    (od / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


if __name__ == "__main__":
    main()
