"""python vox_synth.py <text_file> <ref.wav> <ref.txt> <out.wav> [style]
VoxCPM2 (48 kHz out). 'Ultimate cloning': reference audio + its transcript. Optional style prefix in parentheses.
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\VoiceClone\libs_vox")
sys.path.insert(0, str(Path(__file__).parent))
import rocm_shim  # noqa: F401,E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import status  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

MODEL = r"D:\VoiceClone\models\voxcpm2"
MAX_CHARS = 280
GAP_SENT, GAP_PARA = 0.18, 0.55


def chunks(text):
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


def main(text_file, ref_wav, ref_txt, out, style=""):
    text = Path(text_file).read_text(encoding="utf-8-sig").replace("\ufeff", "")
    ref_text = Path(ref_txt).read_text(encoding="utf-8").strip()
    status.stage("Synthesize", "running", "loading VoxCPM2")
    model = VoxCPM.from_pretrained(MODEL, load_denoiser=False)
    sr = model.tts_model.sample_rate
    parts = []
    cs = chunks(text)
    for i, (c, gap) in enumerate(cs):
        status.stage("Synthesize", "running", f"chunk {i + 1}/{len(cs)}")
        t = time.time()
        wav = model.generate(text=f"({style}){c}" if style else c, prompt_wav_path=ref_wav, prompt_text=ref_text,
                             reference_wav_path=ref_wav, cfg_value=2.0, inference_timesteps=10)
        a = np.asarray(wav, dtype=np.float32).squeeze()
        print(f"chunk {i + 1}/{len(cs)} {len(a) / sr:.1f}s audio in {time.time() - t:.1f}s", flush=True)
        parts += [a, np.zeros(int(sr * gap), np.float32)]
    sf.write(out, np.concatenate(parts), sr, subtype="FLOAT")
    print("sample rate", sr, "peak VRAM GB", round(torch.cuda.max_memory_allocated() / 1e9, 1))


if __name__ == "__main__":
    main(*sys.argv[1:6])
