"""Pure-numpy building blocks for stitching synthesized speech onto a timeline (no model, no GPU, easy to test).

trim / collapse_pauses      clean one take: cut leading/trailing silence, shorten over-long silences inside it
valley / silence_len_at     find where to cut a take between two script lines (the quietest point near a word start)
align_boundaries            map ASR words back to script lines (first-word times + how much of the text was heard)
plan                        choose speed and pauses so a sentence fits its time window with the fewest compromises
sentence_groups             group script lines into sentences
active_db / time_stretch    loudness of the speech frames; pitch-preserving speed change (ffmpeg atempo)
"""
import difflib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME = 0.01
MARK = re.compile(r"(?:^|\s)@[a-z][a-z0-9_-]{0,23}(?=\s|$)")


def strip_marks(text):
    return re.sub(r"\s+", " ", MARK.sub(" ", text)).strip()


def norm_word(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def frame_db(x, sr):
    n = int(sr * FRAME)
    k = len(x) // n
    if k == 0:
        return np.array([-120.0])
    return 20 * np.log10(np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-9)


def _runs(mask):
    """[(start, end_exclusive, value)] of equal consecutive values."""
    out, i = [], 0
    while i < len(mask):
        j = i
        while j < len(mask) and mask[j] == mask[i]:
            j += 1
        out.append((i, j, bool(mask[i])))
        i = j
    return out


def speech_mask(x, sr, rel_db=-38.0):
    """Frames that count as speech: within rel_db of the take's loud frames (90th percentile). Silent holes of two
    frames or less inside speech (a stop consonant) are filled so a word is not split."""
    db = frame_db(x, sr)
    m = db > np.percentile(db, 90) + rel_db
    for a, b, v in _runs(m):
        if not v and b - a <= 2 and a > 0 and b < len(m):
            m[a:b] = True
    return m


def trim(x, sr, pad=0.04, rel_db=-38.0):
    m = speech_mask(x, sr, rel_db)
    idx = np.where(m)[0]
    if idx.size == 0:
        return x
    a = max(0, int((idx[0] * FRAME - pad) * sr))
    b = min(len(x), int(((idx[-1] + 1) * FRAME + pad) * sr))
    return x[a:b]


def collapse_pauses(x, sr, max_pause=0.45, keep=0.30, fade=0.012, rel_db=-38.0):
    """Shorten every silence inside the take that is longer than max_pause to `keep` seconds (cut from its middle).
    Returns (audio, seconds_removed)."""
    m = speech_mask(x, sr, rel_db)
    pieces, last, removed = [], 0, 0.0
    f = int(fade * sr)
    for a, b, v in _runs(m):
        if v or a == 0 or b >= len(m) or (b - a) * FRAME <= max_pause:
            continue
        cut_a = int((a * FRAME + keep / 2) * sr)
        cut_b = int((b * FRAME - keep / 2) * sr)
        if cut_b - cut_a <= 2 * f:
            continue
        pieces.append(x[last:cut_a])
        last = cut_b
        removed += (cut_b - cut_a) / sr
    if not pieces:
        return x, 0.0
    pieces.append(x[last:])
    out = pieces[0].copy()
    for p in pieces[1:]:
        p = p.copy()
        if f and len(out) > f and len(p) > f:  # crossfade across the (silent) seam
            ramp = np.linspace(0, 1, f, dtype=np.float32)
            out[-f:] = out[-f:] * (1 - ramp) + p[:f] * ramp
            out = np.concatenate([out, p[f:]])
        else:
            out = np.concatenate([out, p])
    return out, removed


def silence_len_at(x, sr, t, rel_db=-38.0):
    """Length in seconds of the silent run that contains time t (0 if t is inside speech)."""
    m = speech_mask(x, sr, rel_db)
    k = min(len(m) - 1, max(0, int(t / FRAME)))
    if m[k]:
        return 0.0
    for a, b, v in _runs(m):
        if a <= k < b:
            return (b - a) * FRAME
    return 0.0


def valley(x, sr, t, window=0.15):
    """Time of the quietest spot within +-window of t; in a flat run of silence, the middle of the run."""
    db = frame_db(x, sr)
    k = np.convolve(db, np.ones(3) / 3, mode="same")
    a, b = max(0, int((t - window) / FRAME)), min(len(k), int((t + window) / FRAME) + 1)
    if b <= a:
        return t
    seg = k[a:b]
    low = np.where(seg <= seg.min() + 3.0)[0]
    return (a + int(low[len(low) // 2])) * FRAME + FRAME / 2


def align_boundaries(heard, lines_words):
    """heard: [(norm_word, start, end)] from ASR. lines_words: [[norm_word, ...]] per script line.
    Returns (first_word_start per line or None, matched_fraction)."""
    exp, owner = [], []
    for li, ws in enumerate(lines_words):
        exp += ws
        owner += [li] * len(ws)
    if not exp or not heard:
        return [None] * len(lines_words), 0.0
    sm = difflib.SequenceMatcher(None, exp, [h[0] for h in heard], autojunk=False)
    hit = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            hit[blk.a + k] = blk.b + k
    firsts = []
    for li in range(len(lines_words)):
        idxs = [i for i, o in enumerate(owner) if o == li]
        t = next((heard[hit[i]][1] for i in idxs if i in hit), None)
        firsts.append(t)
    return firsts, len(hit) / len(exp)


def plan(durs, nats, starts, end, mp=0.08, mp_pref=0.8, tail_ok=0.7, f_lo=0.88, f_hi=1.30):
    """Fit the speech segments of one sentence into its window.
    durs[i]   natural length of segment i in seconds      nats[i]   natural pause before segment i (nats[0] unused)
    starts[i] slot start of segment i's line               end       slot end of the last line
    Tries speeds f in [f_lo, f_hi]; segment i is placed at its slot start unless the previous one is still speaking.
    Cost: lateness vs slot start, pauses longer than mp_pref, deviation from the natural pause, overrunning the window
    (very expensive), a long silent tail, and changing the speed (speeding up costs more than slowing down).
    Returns dict(f, starts, over, tail, cost); `over` > 0 means the sentence still overruns at the fastest allowed speed."""
    best = None
    n_steps = int(round((f_hi - f_lo) / 0.01))
    for s in range(n_steps + 1):
        f = f_lo + s * 0.01
        d = [x / f for x in durs]
        p, cost = [starts[0]], 0.0
        for i in range(1, len(d)):
            prev_end = p[-1] + d[i - 1]
            pi = max(prev_end + mp, starts[i])
            pause = pi - prev_end
            cost += 4.0 * (pi - starts[i]) + 3.0 * max(0.0, pause - mp_pref) + 0.7 * abs(pause - min(nats[i], 0.6))
            p.append(pi)
        e = p[-1] + d[-1]
        over, tail = max(0.0, e - end), end - e
        cost += 100.0 * over + 4.0 * max(0.0, tail - tail_ok) + (5.0 * (1 - f) if f < 1 else 8.0 * (f - 1))
        if best is None or cost < best["cost"] - 1e-9:
            best = {"f": round(f, 3), "starts": p, "over": over, "tail": tail, "cost": cost}
    return best


def sentence_groups(rows):
    """rows: dicts with 'text' (spoken lines in order). A line continues the sentence of the previous line unless that
    line ends in . ? or !. Returns a list of lists of rows."""
    groups, cur = [], []
    for r in rows:
        cur.append(r)
        if re.search(r"[.?!][\"')\]]*$", strip_marks(r["text"])):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def active_db(x, sr, rel_db=-35.0):
    db = frame_db(x, sr)
    act = db[db > np.percentile(db, 90) + rel_db]
    return float(act.mean()) if act.size else -120.0


def time_stretch(x, sr, f):
    """Speed up by f (>1) or slow down (<1) keeping pitch."""
    if abs(f - 1.0) < 0.005 or len(x) == 0:
        return x
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(src, x, sr, subtype="FLOAT")
        subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-v", "error", "-i", str(src), "-filter:a", f"atempo={f:.5f}", str(dst)],
                       check=True)
        out, _ = sf.read(dst, dtype="float32")
    return out
