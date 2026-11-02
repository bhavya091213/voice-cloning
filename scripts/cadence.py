"""cadence: measure how a voice paces speech and keep it as a few numbers inside a .vcref.

Per recording we keep (about 300 bytes of JSON per clip, nothing is re-synthesized from it):
    artic    syllables/second while speaking (silences > 0.12 s removed)    rate   syllables/second including pauses
    pauses   median / p90 silence by what ended the word before it: comma (, ; :), stop (. !), ask (?), word (none)
    final    how much longer the last word of a phrase is than the average word (phrase-final lengthening)
    lead     speech rate of the first / middle / last third of a phrase relative to its mean (rate contour)
    f0       median pitch in Hz and 10th-90th percentile spread in semitones      db   loudness spread (p10-p90) in dB
A bundle stores `cadence = {"v":1, "tags": {tag: stats}, "global": stats, "sections": {name: stats}}`.
`tags` come from the intonation clips (one per emotion), `sections` from a long recording split by what the sentence
does (opening, body, closing, question, clause-run), `global` is the duration-weighted pool of everything measured.

predict() turns text + tag into the speaking time this voice would need, so synthesis can pace itself by cadence.

CLI:  python cadence.py wav <audio> [--text T]            measure one recording
      python cadence.py video <audio> <asr.json> -o c.json    long recording + Whisper word timestamps -> sections
      python cadence.py embed <file.vcref> <cadence.json>      write cadence into an existing bundle
      python cadence.py show <file.vcref>
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402
import timeline as tl  # noqa: E402

PAUSE_MIN = 0.12   # silences shorter than this are articulation, not pauses
KINDS = ("comma", "stop", "ask", "word")
_CMU = None


def _cmu():
    global _CMU
    if _CMU is None:
        from nltk.corpus import cmudict
        try:
            _CMU = cmudict.dict()
        except LookupError:
            import nltk
            nltk.download("cmudict", quiet=True)
            _CMU = cmudict.dict()
    return _CMU


def syllables(text):
    """Syllable count of a text: CMU dictionary, vowel-group fallback for unknown words."""
    n, cmu = 0, _cmu()
    for w in re.findall(r"[A-Za-z']+", text):
        w = w.lower().strip("'")
        if not w:
            continue
        if w in cmu:
            n += sum(1 for p in cmu[w][0] if p[-1].isdigit())
        else:
            n += max(1, len(re.findall(r"[aeiouy]+", w)) - (1 if w.endswith("e") and len(w) > 3 else 0))
    return n


def kind_of(word):
    """Pause class from the punctuation that ends a word."""
    w = word.strip().rstrip("\"')]")
    if w.endswith("?"):
        return "ask"
    if w.endswith((".", "!", "…")):
        return "stop"
    if w.endswith((",", ";", ":", "—", "-")):
        return "comma"
    return "word"


def _mono(x):
    return x.mean(1) if x.ndim > 1 else x


def _mask(x, sr):
    """Speech frames. timeline.speech_mask looks 38 dB below the loud frames, which is right for clean synthesis but
    never sees a pause in a real recording (room noise sits around -25 dB), so the floor is lifted to the noise level."""
    db = tl.frame_db(x, sr)
    m = db > max(np.percentile(db, 90) - 38.0, np.percentile(db, 10) + 8.0)
    for a, b, v in tl._runs(m):
        if not v and b - a <= 2 and a > 0 and b < len(m):
            m[a:b] = True
    return m


def _f0(x, sr):
    """Median Hz and 10-90 percentile spread in semitones of the voiced frames (pYIN at 16 kHz)."""
    import librosa
    y = resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x.astype(np.float32)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=420, sr=16000, frame_length=1024, hop_length=320)
    f = f0[voiced & np.isfinite(f0)]
    if f.size < 20:
        return {"med": 0.0, "spread": 0.0, "sd": 0.0}
    st = 12 * np.log2(f / np.median(f))
    return {"med": round(float(np.median(f)), 1), "spread": round(float(np.percentile(st, 90) - np.percentile(st, 10)), 2),
            "sd": round(float(st.std()), 2)}


def measure(x, sr, words, with_f0=True):
    """x audio, words [(token_with_punctuation, start, end)] in seconds -> stats dict (see module doc)."""
    x = _mono(x).astype(np.float32)
    if len(words) < 3:
        raise ValueError("need at least 3 words to measure cadence")
    t0, t1 = words[0][1], words[-1][2]
    seg = x[int(t0 * sr):int(t1 * sr)]
    mask = _mask(seg, sr)
    for w0, w1 in zip(words, words[1:]):  # Whisper saw no word for 0.2 s or more: that is a pause whatever the noise floor says
        if w1[1] - w0[2] >= 0.2:
            mask[int((w0[2] - t0 + 0.05) / tl.FRAME):int((w1[1] - t0 - 0.05) / tl.FRAME)] = False
    pauses = {k: [] for k in KINDS}
    spoken = 0.0
    for a, b, v in tl._runs(mask):
        d = (b - a) * tl.FRAME
        if v:
            spoken += d
        elif a > 0 and b < len(mask):
            if d >= PAUSE_MIN:
                t = t0 + a * tl.FRAME
                prev = min(words, key=lambda w: abs(w[2] - t))
                pauses[kind_of(prev[0])].append(d)
            else:
                spoken += d
    total = t1 - t0
    spoken = max(spoken, 0.3 * total)
    syl = [max(1, syllables(w[0])) for w in words]
    artic = sum(syl) / spoken
    # phrase-final lengthening: last word of every phrase (before a pause or punctuation) vs mean seconds/syllable
    per_syl = [(w[2] - w[1]) / s for w, s in zip(words, syl)]
    ends = [i for i, w in enumerate(words) if kind_of(w[0]) != "word" or i == len(words) - 1]
    final = float(np.mean([per_syl[i] for i in ends]) / np.mean(per_syl)) if ends else 1.0
    # rate contour inside phrases: split at punctuation, compare thirds (by syllable position)
    thirds = [[], [], []]
    start = 0
    for i in ends:
        ph = words[start:i + 1]
        if len(ph) >= 3:
            for k, third in enumerate(np.array_split(np.arange(len(ph)), 3)):
                if len(third):
                    d = ph[third[-1]][2] - ph[third[0]][1]
                    s = sum(syl[start + j] for j in third)
                    if d > 0:
                        thirds[k].append(s / d)
        start = i + 1
    mean_r = np.mean([np.mean(t) for t in thirds if t]) if any(thirds) else artic
    lead = [round(float(np.mean(t) / mean_r), 3) if t else 1.0 for t in thirds]
    db = tl.frame_db(seg, sr)
    act = db[db > np.percentile(db, 90) - 35]
    out = {"seconds": round(total, 2), "syll": int(sum(syl)), "artic": round(artic, 3), "rate": round(sum(syl) / total, 3),
           "pauses": {k: {"n": len(v), "med": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3)}
                      for k, v in pauses.items() if v},
           "final": round(final, 3), "lead": lead,
           "db": round(float(np.percentile(act, 90) - np.percentile(act, 10)), 1) if act.size > 10 else 0.0}
    if with_f0:
        out["f0"] = _f0(seg, sr)
    return out


def pool(stats):
    """Duration-weighted merge of several measure() results (pause counts add, medians weighted by count)."""
    stats = [s for s in stats if s]
    w = np.array([s["seconds"] for s in stats])
    def avg(key):
        return round(float(np.average([s[key] for s in stats], weights=w)), 3)
    pauses = {}
    for k in KINDS:
        have = [s["pauses"][k] for s in stats if k in s["pauses"]]
        if have:
            n = np.array([h["n"] for h in have])
            pauses[k] = {"n": int(n.sum()), "med": round(float(np.average([h["med"] for h in have], weights=n)), 3),
                         "p90": round(float(np.average([h["p90"] for h in have], weights=n)), 3)}
    out = {"seconds": round(float(w.sum()), 1), "syll": int(sum(s["syll"] for s in stats)), "artic": avg("artic"),
           "rate": avg("rate"), "pauses": pauses, "final": avg("final"),
           "lead": [round(float(np.average([s["lead"][i] for s in stats], weights=w)), 3) for i in range(3)], "db": avg("db")}
    f0s = [s["f0"] for s in stats if s.get("f0") and s["f0"]["med"]]
    if f0s:
        out["f0"] = {k: round(float(np.average([f[k] for f in f0s], weights=[s["seconds"] for s in stats if s.get("f0") and s["f0"]["med"]])), 2)
                     for k in ("med", "spread", "sd")}
    return out


def whisper_words(x, sr, model=None):
    """[(word, start, end)] from Whisper (small, local) for a recording."""
    import whisper
    model = model or whisper.load_model("small", download_root=str(paths.MODELS / "whisper"))
    y = resample_poly(_mono(x), 16000, sr).astype(np.float32) if sr != 16000 else _mono(x).astype(np.float32)
    r = model.transcribe(y, language="en", word_timestamps=True, fp16=False)
    return [(w["word"].strip(), float(w["start"]), float(w["end"])) for s in r["segments"] for w in s["words"] if w["word"].strip()]


def with_script_punctuation(words, text):
    """Whisper's punctuation is a guess; when the clip's real transcript is known, copy its punctuation onto the words
    (sequence-aligned on the letters)."""
    import difflib
    toks = text.split()
    a = [tl.norm_word(w[0]) for w in words]
    b = [tl.norm_word(t) for t in toks]
    out = list(words)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                out[i1 + k] = (toks[j1 + k], out[i1 + k][1], out[i1 + k][2])
    return out


def analyze_clip(wav, text=None, model=None):
    x, sr = sf.read(wav, dtype="float32")
    words = whisper_words(x, sr, model)
    if text:
        words = with_script_punctuation(words, text)
    return measure(x, sr, words)


def from_refs(refs_dir, model=None):
    """Cadence of every intonation clip in a refs folder (<tag>.wav + index.json) -> {"v":1,"tags":..,"global":..}."""
    refs_dir = Path(refs_dir)
    idx = json.loads((refs_dir / "index.json").read_text(encoding="utf-8"))
    tags = {}
    for tag, meta in idx["clips"].items():
        try:
            tags[tag] = analyze_clip(refs_dir / f"{tag}.wav", meta.get("text"), model)
        except ValueError:
            pass
    return {"v": 1, "tags": tags, "global": pool(list(tags.values()))}


# --- long recordings: split into sections by what the sentence does ---------------------------------------------

def sentence_spans(words):
    """Group words into sentences (ending . ? !) -> [(first_idx, last_idx)]."""
    spans, s = [], 0
    for i, w in enumerate(words):
        if kind_of(w[0]) in ("stop", "ask") or i == len(words) - 1:
            spans.append((s, i))
            s = i + 1
    return spans


def section_of(idx, n, words, span):
    """Opening / closing by position in the recording, question by '?', clause-run for a sentence of 3+ comma clauses,
    otherwise body."""
    a, b = span
    sent = words[a:b + 1]
    if kind_of(words[b][0]) == "ask":
        return "question"
    if idx == 0:
        return "opening"
    if idx >= n - 3:
        return "closing"
    if sum(1 for w in sent if kind_of(w[0]) == "comma") >= 2 and b - a >= 18:
        return "clause_run"
    return "body"


def analyze_long(wav, words, model=None):
    """words: [(token, start, end)] with punctuation (Whisper of a long recording). Returns (global, sections)."""
    x, sr = sf.read(wav, dtype="float32")
    x = _mono(x)
    spans = sentence_spans(words)
    groups = {}
    for i, sp in enumerate(spans):
        groups.setdefault(section_of(i, len(spans), words, sp), []).append(sp)
    sections = {}
    for name, sps in groups.items():
        parts = []
        for a, b in sps:
            if b - a >= 3:
                try:
                    parts.append(measure(x, sr, words[a:b + 1], with_f0=False))
                except ValueError:
                    pass
        if parts:
            sections[name] = pool(parts)
    glob = pool(list(sections.values()))
    # sentence gaps: silence between the end of one sentence and the start of the next (inside the recording)
    gaps = [words[b + 1][1] - words[b][2] for (a, b) in spans[:-1] if b + 1 < len(words)]
    glob["sentence_gap"] = {"n": len(gaps), "med": round(float(np.median(gaps)), 3), "p25": round(float(np.percentile(gaps, 25)), 3),
                            "p90": round(float(np.percentile(gaps, 90)), 3)}
    glob["f0"] = _f0(x, sr)
    return glob, sections


# --- use --------------------------------------------------------------------------------------------------------

def pick(cad, tag=None, section=None):
    """Most specific stats available: section, else tag, else global."""
    if not cad:
        return None
    for key, table in ((section, "sections"), (tag, "tags")):
        if key and key in cad.get(table, {}):
            return cad[table][key]
    return cad.get("global") or next(iter(cad.get("tags", {}).values()), None)


def pause_for(cad, kind, scale=1.0):
    """Typical pause (s) after a word of this class, from the pooled global stats; 0 when unknown."""
    g = (cad or {}).get("global") or {}
    p = g.get("pauses", {}).get(kind)
    return round(p["med"] * scale, 3) if p else 0.0


def predict(text, cad, tag=None, section=None, pace=1.0):
    """Seconds this voice would take to say `text` at its own cadence (x pace): syllables / artic + pauses at punctuation."""
    s = pick(cad, tag, section)
    if not s:
        return None
    speak = syllables(text) / (s["artic"] * pace)
    pz = 0.0
    for w in text.split():
        k = kind_of(w)
        if k != "word":
            p = s["pauses"].get(k) or (cad.get("global") or {}).get("pauses", {}).get(k)
            pz += (p["med"] if p else 0.0) / pace
    return speak + pz


def classify_text(text, index=None, total=None):
    """Section name for a script sentence (same rules as the recording analysis)."""
    ws = [(w, 0, 0) for w in text.split()]
    if not ws:
        return "body"
    if kind_of(ws[-1][0]) == "ask":
        return "question"
    if index == 0:
        return "opening"
    if index is not None and total and index >= total - 3:
        return "closing"
    if sum(1 for w in ws if kind_of(w[0]) == "comma") >= 2 and len(ws) >= 18:
        return "clause_run"
    return "body"


def _cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("wav"); p.add_argument("audio"); p.add_argument("--text")
    p = sub.add_parser("video"); p.add_argument("audio"); p.add_argument("asr"); p.add_argument("-o", "--out", required=True)
    p = sub.add_parser("embed"); p.add_argument("vcref"); p.add_argument("json")
    sub.add_parser("show").add_argument("vcref")
    a = ap.parse_args()
    if a.cmd == "wav":
        print(json.dumps(analyze_clip(a.audio, a.text), indent=1))
    elif a.cmd == "video":
        asr = json.loads(Path(a.asr).read_text(encoding="utf-8"))
        words = [(w["word"].strip(), w["start"], w["end"]) for s in asr["segments"] for w in s.get("words", []) if w["word"].strip()]
        g, sec = analyze_long(a.audio, words)
        Path(a.out).write_text(json.dumps({"v": 1, "global": g, "sections": sec}, indent=1), encoding="utf-8")
        names = ", ".join("%s %.2f syl/s" % (k, v["artic"]) for k, v in sec.items())
        print(f"{len(words)} words -> {a.out}; sections: {names}")
    elif a.cmd == "embed":
        import vcref
        vcref.set_cadence(a.vcref, json.loads(Path(a.json).read_text(encoding="utf-8")))
    else:
        import vcref
        print(json.dumps(vcref.Bundle(a.vcref).cadence, indent=1))


if __name__ == "__main__":
    _cli()
