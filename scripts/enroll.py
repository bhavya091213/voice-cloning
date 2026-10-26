"""python enroll.py <speaker>  -> prints JSON.
Minimal enrollment: pick the best ~15 s sentence-aligned span of the raw recording and transcribe it.
No denoising, no EQ - the model hears your voice exactly as recorded.
"""
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import whisper  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402

REF = paths.REFERENCE


def main(spk):
    d = REF / spk
    parts = {p: d / "parts" / f"{p}.wav" for p in "ABC" if (d / "parts" / f"{p}.wav").exists()}
    if not parts:
        raise SystemExit("no recorded parts")
    src_name = "A" if "A" in parts else max(parts, key=lambda p: sf.info(parts[p]).duration)
    src = parts[src_name]
    model = whisper.load_model("small", download_root=str(paths.MODELS / "whisper"))
    r = model.transcribe(str(src), language="en", word_timestamps=True, fp16=False)
    words = [w for s in r["segments"] for w in s["words"]]
    sents, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.!?][\"')]*$", w["word"].strip()):
            sents.append(cur); cur = []
    if cur:
        sents.append(cur)

    best, best_score = None, -1e9
    for i in range(len(sents)):
        for j in range(i, len(sents)):
            t0, t1 = sents[i][0]["start"], sents[j][-1]["end"]
            dur = t1 - t0
            if dur > 20:
                break
            if dur < 9:
                continue
            wps = sum(len(s) for s in sents[i:j + 1]) / dur
            txt = " ".join(w["word"].strip() for s in sents[i:j + 1] for w in s)
            toks = [t.lower().strip(".,;:?!\"'") for t in txt.split()]
            allit = sum(a[:1] == b[:1] for a, b in zip(toks, toks[1:])) / max(1, len(toks) - 1)  # tongue-twister penalty
            score = -abs(dur - 15) / 5 - abs(wps - 2.7) - (3 if re.search(r"\d", txt) else 0) - 12 * allit
            if score > best_score:
                best, best_score = (i, j, t0, t1, txt), score
    if best is None:  # fall back to the first ~15 s
        i = j = 0
        t0, t1 = words[0]["start"], min(words[-1]["end"], words[0]["start"] + 15)
        txt = " ".join(w["word"].strip() for w in words if w["end"] <= t1)
        best = (0, 0, t0, t1, txt)
    i, j, t0, t1, txt = best

    x, sr = sf.read(src, dtype="float32")
    prev_end = sents[i - 1][-1]["end"] if i > 0 else 0.0
    next_start = sents[j + 1][0]["start"] if j + 1 < len(sents) else t1 + 0.6
    a = max(0.0, t0 - min(0.25, max(0.02, (t0 - prev_end) / 2)))
    b = t1 + min(0.3, max(0.02, (next_start - t1) / 2))
    clip = x[int(a * sr):int(b * sr)].copy()
    f = int(0.01 * sr)
    clip[:f] *= np.linspace(0, 1, f); clip[-f:] *= np.linspace(1, 0, f)
    sf.write(d / "ref.wav", clip, sr, subtype="PCM_24")
    (d / "ref.txt").write_text(txt, encoding="utf-8")
    print(json.dumps({"seconds": round(len(clip) / sr, 1), "transcript": txt, "source": src_name}))


if __name__ == "__main__":
    main(sys.argv[1])
