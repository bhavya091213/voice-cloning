"""python mlx_synth.py <text_file> <ref.wav> <ref.txt> <out.wav>
Apple Silicon only: voice clone through MLX (mlx-audio) instead of PyTorch/MPS. Model: VOICECLONE_MLX_MODEL
(default mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16; any mlx-audio model that clones from ref_audio + ref_text works).
Same chunking and gaps as qwen_synth.py so the output slots into the same mastering pipeline.
UNTESTED on real hardware: written against the mlx-audio API without a Mac to run it on."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import status  # noqa: E402
from qwen_synth import chunks  # noqa: E402  (text splitting only; importing it loads no model)

MODEL = os.environ.get("VOICECLONE_MLX_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")


def main(text_file, ref_wav, ref_txt, out):
    if not paths.IS_MAC:
        raise SystemExit("the MLX engine runs on Apple Silicon Macs only (use --engine qwen or voxcpm here)")
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError:
        raise SystemExit("mlx-audio is not installed: .venv/bin/python -m pip install mlx-audio")
    text = Path(text_file).read_text(encoding="utf-8-sig").replace("﻿", "")
    ref_text = Path(ref_txt).read_text(encoding="utf-8").strip()
    status.stage("Synthesize", "running", f"loading {MODEL} (MLX)")
    model = load_model(MODEL)
    sr = getattr(model, "sample_rate", 24000)
    parts, cs = [], chunks(text)
    for i, (c, gap) in enumerate(cs):
        status.stage("Synthesize", "running", f"chunk {i + 1}/{len(cs)}")
        t = time.time()
        segs = [np.asarray(r.audio, dtype=np.float32).reshape(-1)
                for r in model.generate(text=c, ref_audio=ref_wav, ref_text=ref_text)]
        a = np.concatenate(segs)
        print(f"chunk {i + 1}/{len(cs)} {len(a) / sr:.1f}s audio in {time.time() - t:.1f}s", flush=True)
        parts += [a, np.zeros(int(sr * gap), np.float32)]
    sf.write(out, np.concatenate(parts), sr, subtype="FLOAT")
    print("sample rate", sr)


if __name__ == "__main__":
    main(*sys.argv[1:5])
