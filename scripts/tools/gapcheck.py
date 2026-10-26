"""python gapcheck.py <timed_script.txt> <rendered.wav>
Stitching diagnostics for a timed voiceover: silent gaps (flagging those in the middle of a sentence), the level of every
line, and the jumps between neighbouring lines. Numbers only; use it to compare renders before and after a change."""
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

LINE = re.compile(r"^\s*(\d+):(\d{1,2}(?:\.\d+)?)\s*-\s*(\d+):(\d{1,2}(?:\.\d+)?)\s+(.+?)\s*$")
FRAME = 0.01


def parse(path):
    rows = []
    for ln in Path(path).read_text(encoding="utf-8-sig").splitlines():
        m = LINE.match(ln)
        if m and not m[5].startswith("["):
            rows.append({"start": int(m[1]) * 60 + float(m[2]), "end": int(m[3]) * 60 + float(m[4]), "text": re.sub(r"^@\w+\s+", "", m[5])})
    for i, r in enumerate(rows):  # a line continues the previous sentence when that one does not end with . ? !
        r["cont"] = i > 0 and not re.search(r"[.?!][\"')\]]*$", rows[i - 1]["text"].strip())
    return rows


def levels(x, sr):
    n = int(sr * FRAME)
    k = len(x) // n
    return 20 * np.log10(np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-9)


def main(script, wav, min_gap=0.2):
    rows = parse(script)
    x, sr = sf.read(wav, dtype="float32")
    x = x.mean(1) if x.ndim > 1 else x
    lv = levels(x, sr)
    thr = np.percentile(lv[lv > -80], 90) - 35
    speech = lv > thr
    t0, t1 = rows[0]["start"], rows[-1]["end"]
    gaps, i = [], int(t0 / FRAME)
    end = int(t1 / FRAME)
    while i < end:
        if not speech[i]:
            j = i
            while j < end and not speech[j]:
                j += 1
            if (j - i) * FRAME >= min_gap:
                gaps.append((i * FRAME, j * FRAME))
            i = j
        else:
            i += 1
    bounds = [r["start"] for r in rows if r["cont"]]
    print(f"{'gap at':>8} {'length':>7}  kind")
    mid = []
    for a, b in gaps:
        inside = any(a - 0.1 <= t <= b + 0.1 for t in bounds)
        if inside:
            mid.append(b - a)
        print(f"{a:8.1f} {b - a:6.2f}s  {'MID-SENTENCE' if inside else 'between sentences'}")
    print(f"\nmid-sentence gaps >= {min_gap}s: {len(mid)}  (longest {max(mid, default=0):.2f}s, total {sum(mid):.1f}s)")
    print(f"all gaps >= {min_gap}s: {len(gaps)}  (longest {max((b - a for a, b in gaps), default=0):.2f}s)")
    lines = []
    for r in rows:
        a, b = int(r["start"] / FRAME), int(r["end"] / FRAME)
        act = lv[a:b][speech[a:b]]
        lines.append(float(np.mean(act)) if act.size else float("nan"))
    jumps = [abs(lines[k] - lines[k - 1]) for k in range(1, len(lines)) if not np.isnan(lines[k]) and not np.isnan(lines[k - 1])]
    print(f"line level (mean active dB): min {np.nanmin(lines):.1f}  max {np.nanmax(lines):.1f}  spread {np.nanmax(lines) - np.nanmin(lines):.1f} dB")
    print(f"level jump between neighbouring lines: mean {np.mean(jumps):.1f} dB, max {max(jumps):.1f} dB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
