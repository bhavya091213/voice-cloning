"""python regen_chunk.py JOB N [--tries 5] [--speaker me_v2]
Re-synthesize script line N (1-based; one chunk) of a finished vox_synth job several times, transcribe every take with
Whisper, and keep the take whose words were all heard clearly. Writes work/jobs/JOB/regen/chunkNN.wav (the winner) and
prints the ranking. The original take is scored too, so a swap only happens if a candidate is actually better."""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402
sys.path.insert(0, str(paths.ROOT / "libs_vox"))
import rocm_shim  # noqa: F401,E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import restitch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("job")
ap.add_argument("n", type=int)
ap.add_argument("--tries", type=int, default=5)
ap.add_argument("--speaker", default="me_v2")
a = ap.parse_args()

jd = paths.ROOT / "work" / "jobs" / a.job
lines = [l for l in (jd / "text.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
text = re.sub(r"^\s*\([^()]*\)\s*", "", lines[a.n - 1]).strip()
orig_raw, sr = sf.read(jd / "raw.wav", dtype="float32")
orig = restitch.split_chunks(orig_raw, sr)[a.n - 1]
ref = paths.ROOT / "reference" / a.speaker
ref_text = (ref / "ref.txt").read_text(encoding="utf-8").strip()
out = jd / "regen"
out.mkdir(exist_ok=True)


def norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


import whisper  # noqa: E402
wm = whisper.load_model("small", download_root=str(paths.MODELS / "whisper"), device="cpu")


def score(wav):
    """(lowest word probability, number of missed script words, shortest word) - higher first is better."""
    tmp = out / "_tmp.wav"
    sf.write(tmp, wav, sr)
    import librosa
    y = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    r = wm.transcribe(y, language="en", word_timestamps=True, condition_on_previous_text=False, fp16=False)
    words = [w for s in r["segments"] for w in s["words"]]
    heard = [norm(w["word"]) for w in words]
    want = [norm(w) for w in text.split()]
    import difflib
    sm = difflib.SequenceMatcher(None, want, heard, autojunk=False)
    hit = sum(b.size for b in sm.get_matching_blocks())
    probs = [w["probability"] for w in words]
    durs = [w["end"] - w["start"] for w in words]
    return (len(want) - hit, -min(probs), -float(np.mean(sorted(probs)[:3])), r["text"]), words


print("script:", text)
base, bw = score(orig)
print(f"original: missed={base[0]} minprob={-base[1]:.2f} | {base[3]}")
for w in bw:
    if w["probability"] < 0.6:
        print(f"   weak word {w['word']!r} p={w['probability']:.2f} {w['end'] - w['start']:.2f}s")

from voxcpm import VoxCPM  # noqa: E402
extra = {"device": paths.device(), "optimize": False} if paths.IS_MAC else {}
model = VoxCPM.from_pretrained(str(paths.MODELS / "voxcpm2"), load_denoiser=False, **extra)
best, best_s = None, None
cands = [("original", orig, base)]
for i in range(a.tries):
    w = np.asarray(model.generate(text=text, prompt_wav_path=str(ref / "ref.wav"), prompt_text=ref_text,
                                  reference_wav_path=str(ref / "ref.wav"), cfg_value=2.0, inference_timesteps=10),
                   dtype=np.float32).squeeze()
    s, ws = score(w)
    sf.write(out / f"chunk{a.n:02d}_try{i}.wav", w, sr, subtype="FLOAT")
    print(f"try {i}: {len(w) / sr:.1f}s missed={s[0]} minprob={-s[1]:.2f} | {s[3]}", flush=True)
    for x in ws:
        if x["probability"] < 0.6:
            print(f"   weak word {x['word']!r} p={x['probability']:.2f} {x['end'] - x['start']:.2f}s")
    cands.append((f"try{i}", w, s))
name, wav, s = min(cands, key=lambda c: c[2][:3])
print("winner:", name)
sf.write(out / f"chunk{a.n:02d}.wav", wav, sr, subtype="FLOAT")
