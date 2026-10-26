"""python synth.py <segments.json> <prompt.wav> <out.wav>   (IndexTTS-2.5, one call per tagged segment)"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "models" / "index-tts"))
import rocm_shim  # noqa: F401,E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import status  # noqa: E402

EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]

segs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prompt, out = sys.argv[2], sys.argv[3]

os.chdir(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "models" / "index-tts"))
from indextts.infer_v2_5 import IndexTTS2  # noqa: E402

tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=True)
pieces, sr = [], 22050
for i, s in enumerate(segs):
    status.stage("Synthesize", "running", f"segment {i + 1}/{len(segs)} ({s['emotion']})")
    vec = [0.0] * 8
    vec[EMOTIONS.index(s["emotion"])] = round(min(0.9, max(0.05, s["intensity"] * 0.9)), 2)
    tmp = tempfile.mktemp(suffix=".wav")
    tts.infer(spk_audio_prompt=prompt, text=s["text"], output_path=tmp, lang="EN", emo_vector=vec,
              duration_factor=round(1 / s["pace"], 3), use_random=False, verbose=False)
    a, sr = sf.read(tmp, dtype="float32")
    os.remove(tmp)
    pieces += [a if a.ndim == 1 else a.mean(1), np.zeros(int(sr * s["pause_ms"] / 1000), np.float32)]
sf.write(out, np.concatenate(pieces), sr, subtype="FLOAT")
print("synth done")
