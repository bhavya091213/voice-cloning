"""python qwen_synth.py <text_file> <ref.wav> <ref.txt> <out.wav>
Qwen3-TTS-12Hz-1.7B-Base voice clone. Text is split only at sentence/paragraph boundaries into
natural-sized chunks; gaps between chunks are short and fixed (no per-sentence pause injection).
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402
sys.path.insert(0, str(paths.ROOT / "libs_qwen"))
import rocm_shim  # noqa: F401,E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import status  # noqa: E402
from qwen_tts import Qwen3TTSModel  # noqa: E402

MODEL = str(paths.MODELS / "qwen3-tts-1.7b-base")
MAX_CHARS = 280
GAP_SENT, GAP_PARA = 0.18, 0.55


def chunks(text):
    """Yield (chunk, gap_after_seconds)."""
    out = []
    for para in re.split(r"\n\s*\n|\n", text.strip()):
        sents = re.findall(r"[^.!?]+[.!?]+[\"')\]]*|[^.!?]+$", para.strip())
        cur = ""
        for s in sents:
            s = s.strip()
            if cur and len(cur) + len(s) + 1 > MAX_CHARS:
                out.append([cur, GAP_SENT]); cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            out.append([cur, GAP_PARA])
    if out:
        out[-1][1] = 0.0
    return out


def main(text_file, ref_wav, ref_txt, out):
    text = Path(text_file).read_text(encoding="utf-8-sig").replace("\ufeff", "")
    ref_text = Path(ref_txt).read_text(encoding="utf-8").strip()
    status.stage("Synthesize", "running", "loading Qwen3-TTS")
    model = Qwen3TTSModel.from_pretrained(MODEL, device_map=paths.device(), dtype=paths.dtype(), attn_implementation="sdpa")
    prompt = model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=ref_text, x_vector_only_mode=False)
    parts, sr = [], 24000
    cs = chunks(text)
    for i, (c, gap) in enumerate(cs):
        status.stage("Synthesize", "running", f"chunk {i + 1}/{len(cs)}")
        t = time.time()
        wavs, sr = model.generate_voice_clone(text=c, language="English", voice_clone_prompt=prompt)
        a = np.asarray(wavs[0], dtype=np.float32)
        print(f"chunk {i + 1}/{len(cs)} {len(a) / sr:.1f}s audio in {time.time() - t:.1f}s", flush=True)
        parts += [a, np.zeros(int(sr * gap), np.float32)]
    sf.write(out, np.concatenate(parts), sr, subtype="FLOAT")
    print("peak VRAM GB", paths.peak_mem_gb())


if __name__ == "__main__":
    main(*sys.argv[1:5])
