"""python timed_synth.py <timed_script.txt> <ref.wav> <ref.txt> <out.wav>
VoxCPM2 voiceover to a timeline. Script lines look like `0:03.0 - 0:07.5   Text to say`; lines whose text starts
with `[` (music cues) stay silent. Output length = the last end time. Writes timing.json next to the output.

How a sentence is made (see scripts/timeline.py for the pieces):
  1. Lines are grouped into sentences (a line that does not end in . ? ! continues into the next), and each sentence
     is synthesized as ONE take, so the intonation runs through it instead of falling at every line break.
  2. Each take is checked: trimmed, over-long inner silences shortened, then Whisper listens to it. A take that lost or
     garbled words is discarded; up to VOICECLONE_TAKES (3) takes are tried and the best one is kept.
  3. The take is cut back into its script lines at the quietest point near each line's first word (forced alignment).
  4. plan() picks one speed (0.88x to 1.30x) and the pauses at the line breaks, so each line starts at its slot start
     when it can, no pause drags on, and the sentence ends inside its window. A sentence that cannot fit may spill into
     the silence before the next sentence; it is never cut mid-word unless even that is not enough (flagged).
  5. All sentences are levelled to the same speech loudness (at most 12 dB of gain either way), faded at the edges, mixed.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402
sys.path.insert(0, str(paths.ROOT / "libs_vox"))
import rocm_shim  # noqa: F401,E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import status  # noqa: E402
import takes  # noqa: E402
import timeline as tl  # noqa: E402
import vcref  # noqa: E402
import re  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

MODEL = str(paths.MODELS / "voxcpm2")
LINE = re.compile(r"^\s*(\d+):(\d{1,2}(?:\.\d+)?)\s*-\s*(\d+):(\d{1,2}(?:\.\d+)?)\s+(.+?)\s*$")
CFG = float(os.environ.get("VOICECLONE_CFG", 2.0))  # lower = looser, more varied delivery; higher = closer to the reference
MAX_TAKES = int(os.environ.get("VOICECLONE_TAKES", 3))
MIN_ASR = 0.92       # fraction of the script's words Whisper must hear (in order) for a take to count as faithful
TAIL_GAP = 0.25      # the silence we aim to leave after a sentence when choosing how long its speech should be
MAX_GAIN_DB = 12.0
FADE = 0.015


def parse(text):
    rows = []
    for ln in text.splitlines():
        m = LINE.match(ln)
        if not m:
            continue
        a, b = int(m[1]) * 60 + float(m[2]), int(m[3]) * 60 + float(m[4])
        rows.append({"start": a, "end": b, "text": m[5], "music": m[5].startswith("[")})
    return rows


def main(script, ref_wav, ref_txt, out):
    rows = parse(Path(script).read_text(encoding="utf-8-sig"))
    if not rows:
        raise SystemExit("no timed lines found (expected `m:ss.s - m:ss.s  text`)")
    ref_text = Path(ref_txt).read_text(encoding="utf-8").strip()
    bundle = vcref.Bundle(os.environ["VOICECLONE_VCREF"]) if os.environ.get("VOICECLONE_VCREF") else None
    if bundle:
        bundle.verify()
    status.stage("Synthesize", "running", "loading VoxCPM2")
    extra = {"device": paths.device(), "optimize": False} if paths.IS_MAC else {}
    model = VoxCPM.from_pretrained(MODEL, load_denoiser=False, **extra)
    sr = model.tts_model.sample_rate
    import whisper  # after the TTS model so a failure above is reported first
    asr = whisper.load_model("small", download_root=str(paths.MODELS / "whisper"))
    total = max(r["end"] for r in rows)
    spoken = [r for r in rows if not r["music"]]
    groups = tl.sentence_groups(spoken)

    def take(text, clip=None):
        w, t = (ref_wav, ref_text) if clip is None else (str(clip.wav), clip.text)
        wav = model.generate(text=text, prompt_wav_path=w, prompt_text=t,
                             reference_wav_path=w, cfg_value=CFG, inference_timesteps=10)
        return np.asarray(wav, dtype=np.float32).squeeze()

    def say(text):
        """Script text -> (audio, tags used). With a .vcref bundle the text is split into takes (takes.py), each spoken
        with the reference clip its tag picks (a sentence normally has one tag, so one take), then joined."""
        if bundle is None:
            return take(text), ["-"]
        pieces, used = [], []
        for tk in takes.split(text, bundle.rules(), set(bundle.tags), bundle.default):
            clip = bundle.resolve(tk.tag, stack=bool(os.environ.get("VOICECLONE_STACK")))
            pieces += [tl.trim(take(tk.text, clip), sr), np.zeros(int(0.12 * sr), np.float32)]
            used.append(clip.tag)
        return np.concatenate(pieces[:-1]), used

    def heard_words(a):
        try:
            r = asr.transcribe(resample_poly(a, 16000, sr).astype(np.float32), language="en", word_timestamps=True, fp16=False)
            return [(tl.norm_word(w["word"]), w["start"], w["end"]) for s in r["segments"] for w in s["words"] if tl.norm_word(w["word"])]
        except Exception as e:  # keep going: the take is then judged on length alone
            print(f"asr failed: {e}", flush=True)
            return None

    placed = []  # per sentence: dict(segs=[(start_sec, audio)], db=level)
    for gi, g in enumerate(groups):
        status.stage("Synthesize", "running", f"sentence {gi + 1}/{len(groups)}")
        t0 = time.time()
        texts = [r["text"] for r in g]
        words = [[w for w in map(tl.norm_word, tl.strip_marks(t).split()) if w] for t in texts]
        S, E = g[0]["start"], g[-1]["end"]
        goal = max(0.8, E - S - TAIL_GAP)
        flags, cands = [], []
        for _ in range(MAX_TAKES):
            a, used = say(" ".join(texts))
            a = tl.trim(a, sr)
            a, removed = tl.collapse_pauses(a, sr)
            heard = heard_words(a)
            firsts, ratio = tl.align_boundaries(heard, words) if heard is not None else ([None] * len(g), 1.0)
            D = len(a) / sr
            cands.append({"a": a, "used": used, "firsts": firsts, "ratio": ratio, "D": D, "removed": removed,
                          "score": 12 * (1 - ratio) + abs(math.log(max(D, 0.1) / goal))})
            if ratio >= MIN_ASR and 0.88 <= D / goal <= 1.2:
                break
        c = min(cands, key=lambda k: k["score"])
        a, D, ratio = c["a"], c["D"], c["ratio"]
        if ratio < MIN_ASR:
            flags.append(f"UNVERIFIED words heard {ratio:.0%}")
        if c["removed"] > 0.05:
            flags.append(f"shortened inner silence by {c['removed']:.1f}s")

        # cut the take back into its script lines
        n = len(g)
        chars = np.cumsum([0] + [len(w) + 1 for w in map(" ".join, words)])
        cuts = []
        for k in range(1, n):
            t = c["firsts"][k]
            if t is None:
                t = D * chars[k] / max(1, chars[-1])
                flags.append(f"line {k + 1} boundary estimated")
            cuts.append(tl.valley(a, sr, max(0.0, t - 0.05), 0.12))
        if any(b - a_ < 0.15 for a_, b in zip([0.0] + cuts, cuts + [D])):  # alignment gave nonsense: split by length
            cuts = [D * chars[k] / max(1, chars[-1]) for k in range(1, n)]
            flags.append("boundaries by length")
        edges = [0.0] + cuts + [D]
        segs = [tl.trim(a[int(edges[k] * sr):int(edges[k + 1] * sr)], sr) for k in range(n)]
        nats = [0.0] + [tl.silence_len_at(a, sr, t) for t in cuts]
        durs = [len(s) / sr for s in segs]
        starts = [r["start"] for r in g]

        end = E
        pl = tl.plan(durs, nats, starts, end)
        if pl["over"] > 0.02:  # let it spill into the quiet before the next sentence
            nxt = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else total
            if nxt - 0.15 > E:
                end = nxt - 0.15
                pl = tl.plan(durs, nats, starts, end)
                flags.append(f"spills {max(0.0, pl['starts'][-1] + durs[-1] / pl['f'] - E):.1f}s past its window")
        f = pl["f"]
        try:
            segs = [tl.time_stretch(s, sr, f) for s in segs]
        except Exception as e:
            flags.append(f"stretch failed ({e}); speed left at 1.0")
            f = 1.0
        pos, prev_end = [], 0.0
        for k, s in enumerate(segs):
            p = max(pl["starts"][k], prev_end + (0.06 if k else 0.0))
            pos.append(p)
            prev_end = p + len(s) / sr
        if prev_end > end + 0.02:  # last resort: even the extended window is too short; trim the final tail with a fade
            keep = int((end - pos[-1]) * sr)
            segs[-1] = segs[-1][:max(keep, int(0.2 * sr))]
            flags.append(f"OVERRUN cut {prev_end - end:.1f}s from the end")
        placed.append({"segs": list(zip(pos, segs)), "db": tl.active_db(np.concatenate(segs), sr)})

        for k, r in enumerate(g):
            r.update(natural=round(durs[k], 2), final=round(len(segs[k]) / sr, 2), slot=round(r["end"] - r["start"], 2),
                     stretch=round(f, 3), late=round(pos[k] - r["start"], 2), group=gi, asr=round(ratio, 2),
                     tags=c["used"], flags=flags)
        tag = "  " + "; ".join(flags) if flags else ""
        print(f"chunk {gi + 1}/{len(groups)} {S:.1f}s window {E - S:.1f}s speech {D:.1f}s x{f:.2f} words {ratio:.0%} "
              f"takes {len(cands)} in {time.time() - t0:.1f}s [{','.join(c['used'])}]{tag}", flush=True)

    # level every sentence to the median speech loudness, then mix with short fades
    target = float(np.median([p["db"] for p in placed]))
    timeline = np.zeros(int(round(total * sr)) + sr, np.float32)
    for gi, p in enumerate(placed):
        gain_db = float(np.clip(target - p["db"], -MAX_GAIN_DB, MAX_GAIN_DB))
        for r in groups[gi]:
            r["gain_db"] = round(gain_db, 1)
        for start, s in p["segs"]:
            s = s * 10 ** (gain_db / 20)
            n = min(len(s) // 2, int(FADE * sr))
            if n:
                s = s.copy()
                s[:n] *= np.linspace(0, 1, n, dtype=np.float32)
                s[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
            s0 = int(round(start * sr))
            timeline[s0:s0 + len(s)] += s[: len(timeline) - s0]
    peak = float(np.abs(timeline).max())
    if peak > 0.99:
        timeline *= 0.99 / peak
    sf.write(out, timeline[: int(round(total * sr))], sr, subtype="FLOAT")
    Path(out).with_name("timing.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("sample rate", sr, "duration", round(total, 1), "peak VRAM GB", paths.peak_mem_gb())


if __name__ == "__main__":
    main(*sys.argv[1:5])
