"""python prep_clip.py <in_audio> <out_wav> <out_txt> [max_gap_s=0.22]
Decode to 48 kHz mono, transcribe with Whisper (word timestamps), cut silences longer than max_gap down to max_gap
(10 ms crossfades), write the tightened wav and the transcript."""
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import whisper  # noqa: E402

SR = 48000
src, out_wav, out_txt = sys.argv[1:4]
max_gap = float(sys.argv[4]) if len(sys.argv) > 4 else 0.22
tmp = Path(out_wav).with_suffix(".full.wav")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1", "-ar", str(SR), "-c:a", "pcm_f32le", str(tmp)], check=True)
x, _ = sf.read(tmp, dtype="float32")
m = whisper.load_model("small", download_root=r"D:\VoiceClone\models\whisper")
r = m.transcribe(str(tmp), language="en", word_timestamps=True, fp16=False)
words = [w for s in r["segments"] for w in s["words"]]
text = " ".join(s["text"].strip() for s in r["segments"])

# keep [start, end] spans of speech; a gap longer than max_gap keeps max_gap/2 of room on each side of the words
pad = 0.04
spans = []
for w in words:
    a, b = max(0.0, w["start"] - pad), min(len(x) / SR, w["end"] + pad)
    if spans and a - spans[-1][1] <= max_gap:
        spans[-1][1] = b
    else:
        if spans:  # keep a short natural pause between kept spans
            spans[-1][1] += max_gap / 2
            a = max(spans[-1][1], a - max_gap / 2)
        spans.append([a, b])
fade = int(0.01 * SR)
out = []
for a, b in spans:
    seg = x[int(a * SR):int(b * SR)].copy()
    if len(seg) > 2 * fade:
        seg[:fade] *= np.linspace(0, 1, fade); seg[-fade:] *= np.linspace(1, 0, fade)
    out.append(seg)
y = np.concatenate(out)
sf.write(out_wav, y, SR, subtype="FLOAT")
Path(out_txt).write_text(text.strip(), encoding="utf-8")
tmp.unlink()
print(f"{src}: {len(x)/SR:.1f}s -> {len(y)/SR:.1f}s, {len(words)} words\n{text}")
