"""python vox_synth.py <text_file> <ref.wav> <ref.txt> <out.wav> 
VoxCPM2 (48 kHz out). 'Ultimate cloning': reference audio + its transcript. Text lines may start with a (delivery instruction) written by the tagger.
"""
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402
sys.path.insert(0, str(paths.ROOT / "libs_vox"))
import rocm_shim  # noqa: F401,E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import status  # noqa: E402
import takes  # noqa: E402
import vcref  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

MODEL = str(paths.MODELS / "voxcpm2")
CFG = float(os.environ.get("VOICECLONE_CFG", 2.0))
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


TAG_RE = re.compile(r"^\(([^()]{1,120})\)\s*(.*)$")


def segments(text):
    """Tagged script -> [(style, text, gap_after)]. A line '(style)text' is directed; plain lines are not.
    A blank line after a line makes the pause paragraph-sized."""
    lines = text.replace(chr(13), "").split(chr(10))
    out = []
    for i, ln in enumerate(lines):
        ln = ln.strip()
        if not ln:
            if out:
                out[-1][2] = GAP_PARA
            continue
        m = TAG_RE.match(ln)
        style, body = (m.group(1).strip(), m.group(2).strip()) if m else ("", ln)
        if not body:
            continue
        cs = chunks(body)
        for j, (c, gap) in enumerate(cs):
            out.append([style, c, GAP_SENT if j < len(cs) - 1 else 0.3])
    if out:
        out[-1][2] = 0.0
    return out


def main(text_file, ref_wav, ref_txt, out):
    text = Path(text_file).read_text(encoding="utf-8-sig").replace(chr(0xFEFF), "")
    ref_text = Path(ref_txt).read_text(encoding="utf-8").strip()
    status.stage("Synthesize", "running", "loading VoxCPM2")
    # macOS: explicit MPS device, no torch.compile (flaky on MPS); elsewhere keep the library defaults
    extra = {"device": paths.device(), "optimize": False} if paths.IS_MAC else {}
    model = VoxCPM.from_pretrained(MODEL, load_denoiser=False, **extra)
    sr = model.tts_model.sample_rate
    parts = []
    if os.environ.get("VOICECLONE_VCREF"):  # multi-reference: regex-split takes, each with its own intonation clip
        bundle = vcref.Bundle(os.environ["VOICECLONE_VCREF"])
        bundle.verify()
        tk = takes.split(re.sub(r"(?m)^\s*\([^()]{1,120}\)\s*", "", text), bundle.rules(), set(bundle.tags), bundle.default)
        for i, t in enumerate(tk):
            status.stage("Synthesize", "running", f"take {i + 1}/{len(tk)} [{t.tag}]")
            clip = bundle.resolve(t.tag, stack=bool(os.environ.get("VOICECLONE_STACK")))
            t0 = time.time()
            wav = model.generate(text=t.text, prompt_wav_path=str(clip.wav), prompt_text=clip.text,
                                 reference_wav_path=str(clip.wav), cfg_value=CFG, inference_timesteps=10)
            a = np.asarray(wav, dtype=np.float32).squeeze()
            print(f"chunk {i + 1}/{len(tk)} {len(a) / sr:.1f}s audio in {time.time() - t0:.1f}s  [{clip.tag}]", flush=True)
            parts += [a, np.zeros(int(sr * t.gap), np.float32)]
        sf.write(out, np.concatenate(parts), sr, subtype="FLOAT")
        print("sample rate", sr, "peak VRAM GB", paths.peak_mem_gb())
        return
    segs = segments(text)
    for i, (style, c, gap) in enumerate(segs):
        status.stage("Synthesize", "running", f"chunk {i + 1}/{len(segs)}")
        t = time.time()
        if style:
            # controllable cloning: the parenthetical is only read as an instruction when the reference is NOT
            # also given as a transcribed prompt (continuation mode treats it as text to speak)
            wav = model.generate(text=f"({style}){c}", reference_wav_path=ref_wav, cfg_value=CFG, inference_timesteps=10)
        else:
            wav = model.generate(text=c, prompt_wav_path=ref_wav, prompt_text=ref_text,
                                 reference_wav_path=ref_wav, cfg_value=CFG, inference_timesteps=10)
        a = np.asarray(wav, dtype=np.float32).squeeze()
        print(f"chunk {i + 1}/{len(segs)} {len(a) / sr:.1f}s audio in {time.time() - t:.1f}s  [{style}]", flush=True)
        parts += [a, np.zeros(int(sr * gap), np.float32)]
    sf.write(out, np.concatenate(parts), sr, subtype="FLOAT")
    print("sample rate", sr, "peak VRAM GB", paths.peak_mem_gb())


if __name__ == "__main__":
    main(*sys.argv[1:5])
