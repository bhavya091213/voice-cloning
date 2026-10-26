"""python enroll_multi.py <speaker>
Build one reference clip per intonation from the enrollment parts already recorded (docs/recording_script.md):
  A (neutral narration)  -> neutral   best sentence-aligned ~20 s window
  B (six acted lines)    -> excited, warm, serious, sad, annoyed, playful   one clip per acted line
  C (short utterances)   -> question  the span covering the questions
Writes reference/<speaker>/refs/<tag>.wav|txt + index.json. Never touches parts/ or the existing ref.wav.
Transcripts come from Whisper (what was actually said), labels from the recording script.
"""
import difflib
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import whisper  # noqa: E402

B_LINES = [  # tag, label, the line as scripted (used only to match what Whisper heard to the right emotion)
    ("excited", "Excited, fast", "No way! You actually got it done? That is absolutely incredible, I knew you could do it!"),
    ("warm", "Warm, soft, slow", "Take your time. There's no rush at all. Everything is going to be just fine."),
    ("serious", "Serious, low, deliberate", "Listen carefully. What happens next matters more than anything we've done so far."),
    ("sad", "Sad, quiet", "I really thought it would be different this time. I miss how things used to be."),
    ("annoyed", "Annoyed, sharp", "Seriously? Again? I told you twice already. Just put it back where it belongs."),
    ("playful", "Playful, teasing", "Oh, you think you're so clever, don't you? Well, we'll just see about that."),
]
norm = lambda w: re.sub(r"[^a-z']", "", w.lower())  # noqa: E731
norm_words = lambda s: [t for t in map(norm, s.split()) if t]  # noqa: E731


def words_of(model, path):
    r = model.transcribe(str(path), language="en", word_timestamps=True, fp16=False)
    return [w for s in r["segments"] for w in s["words"]]


def sentences(words):
    out, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.!?][\"')]*$", w["word"].strip()):
            out.append(cur); cur = []
    return out + ([cur] if cur else [])


def text_of(ws):
    return " ".join(w["word"].strip() for w in ws)


def cut(x, sr, t0, t1, prev_end=0.0, next_start=None):
    next_start = t1 + 0.6 if next_start is None else next_start
    a = max(0.0, t0 - min(0.25, max(0.02, (t0 - prev_end) / 2)))
    b = t1 + min(0.3, max(0.02, (next_start - t1) / 2))
    clip = x[int(a * sr):int(b * sr)].copy()
    f = int(0.01 * sr)
    clip[:f] *= np.linspace(0, 1, f)
    clip[-f:] *= np.linspace(1, 0, f)
    return clip


def neutral_window(sents, target=20.0):
    best, best_score = None, -1e9
    for i in range(len(sents)):
        for j in range(i, len(sents)):
            t0, t1 = sents[i][0]["start"], sents[j][-1]["end"]
            dur = t1 - t0
            if dur > target * 4 / 3:
                break
            if dur < target * 0.6:
                continue
            txt = text_of([w for s in sents[i:j + 1] for w in s])
            toks = [norm(t) for t in txt.split()]
            allit = sum(a[:1] == b[:1] for a, b in zip(toks, toks[1:])) / max(1, len(toks) - 1)
            wps = sum(len(s) for s in sents[i:j + 1]) / dur
            score = -abs(dur - target) / (target / 3) - abs(wps - 2.7) - (3 if re.search(r"\d", txt) else 0) - 12 * allit
            if score > best_score:
                best, best_score = (i, j), score
    return best


def main(spk):
    d = paths.REFERENCE / spk
    out = d / "refs"
    out.mkdir(exist_ok=True)
    model = whisper.load_model("small", download_root=str(paths.MODELS / "whisper"))
    clips, skipped = {}, []

    def save(tag, label, source, clip, sr, text):
        sf.write(out / f"{tag}.wav", clip, sr, subtype="PCM_24")
        (out / f"{tag}.txt").write_text(text, encoding="utf-8")
        clips[tag] = {"text": text, "seconds": round(len(clip) / sr, 1), "label": label, "source": source}

    def load(part):
        p = d / "parts" / f"{part}.wav"
        if not p.exists():
            skipped.append(f"part {part} missing")
            return None
        x, sr = sf.read(p, dtype="float32")
        return p, x, sr

    a = load("A")
    if a:
        p, x, sr = a
        sents = sentences(words_of(model, p))
        win = neutral_window(sents)
        if win:
            i, j = win
            ws = [w for s in sents[i:j + 1] for w in s]
            prev_end = sents[i - 1][-1]["end"] if i else 0.0
            nxt = sents[j + 1][0]["start"] if j + 1 < len(sents) else None
            save("neutral", "Neutral narration", "A", cut(x, sr, ws[0]["start"], ws[-1]["end"], prev_end, nxt), sr, text_of(ws))
        else:
            skipped.append("neutral: no suitable window in A")

    b = load("B")
    if b:
        p, x, sr = b
        sents = sentences(words_of(model, p))
        lines = [{n: 0 for n in set(map(norm, l.split()))} for _, _, l in B_LINES]
        groups, last = {}, 0
        for k, s in enumerate(sents):
            toks = [norm(w["word"]) for w in s]
            scores = [sum(t in ln for t in toks) / max(1, len(toks)) for ln in lines]
            li = max(range(last, len(B_LINES)), key=lambda i: scores[i])  # lines were read in order
            last = li
            groups.setdefault(li, []).append(k)
        for li, (tag, label, _) in enumerate(B_LINES):
            ks = groups.get(li)
            if not ks:
                skipped.append(f"{tag}: not found in B")
                continue
            ws = [w for k in ks for w in sents[k]]
            prev_end = sents[ks[0] - 1][-1]["end"] if ks[0] else 0.0
            nxt = sents[ks[-1] + 1][0]["start"] if ks[-1] + 1 < len(sents) else None
            heard, scripted = text_of(ws), B_LINES[li][2]
            sim = difflib.SequenceMatcher(None, norm_words(heard), norm_words(scripted)).ratio()
            # 0.6-0.9: Whisper most likely garbled the scripted line, so trust the script. >= 0.9: faithful (or a small
            # deliberate change), keep what was heard. < 0.6: the speaker said something else, keep what was heard.
            text = scripted if 0.6 <= sim < 0.9 else heard
            if text is scripted:
                print(f"{tag}: transcript corrected from the script (Whisper heard: {heard!r})", file=sys.stderr)
            save(tag, label, "B", cut(x, sr, ws[0]["start"], ws[-1]["end"], prev_end, nxt), sr, text)

    c = load("C")
    if c:
        p, x, sr = c
        sents = sentences(words_of(model, p))
        q = [k for k, s in enumerate(sents) if s[-1]["word"].strip().endswith("?")]
        if q:
            k0, k1 = q[0], q[-1]
            while sum(sents[k][-1]["end"] - sents[k][0]["start"] for k in range(k0, k1 + 1)) < 4.0 and k1 + 1 < len(sents):
                k1 += 1  # pad with the next utterance so the clip is long enough to condition on
            ws = [w for k in range(k0, k1 + 1) for w in sents[k]]
            prev_end = sents[k0 - 1][-1]["end"] if k0 else 0.0
            nxt = sents[k1 + 1][0]["start"] if k1 + 1 < len(sents) else None
            save("question", "Questions (short)", "C", cut(x, sr, ws[0]["start"], ws[-1]["end"], prev_end, nxt), sr, text_of(ws))
        else:
            skipped.append("question: no '?' found in C")

    if not clips:
        raise SystemExit("no clips could be built")
    (out / "index.json").write_text(json.dumps({"default": "neutral" if "neutral" in clips else next(iter(clips)),
                                                "clips": clips}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"clips": {t: {"seconds": c["seconds"], "text": c["text"][:70]} for t, c in clips.items()}, "skipped": skipped},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
