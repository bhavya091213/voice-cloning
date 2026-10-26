"""Enrollment: combine recorded parts, denoise (gently, keeping high end), pick the best 15 s cloning prompt."""
import os
from pathlib import Path

import numpy as np
import noisereduce as nr
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

REF_ROOT = str(Path(os.environ.get("VOICECLONE_ROOT") or Path(__file__).resolve().parent.parent) / "reference")
SR = 48000
PROMPT_SEC = 15  # IndexTTS only uses the first 15 s of the speaker prompt


def _levels(x, sr, ms=20):
    n = int(sr * ms / 1000)
    k = len(x) // n
    return 20 * np.log10(np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-9)


def noise_floor_db(x, sr=SR):
    """Level of the quietest 0.5 s window - a robust room-tone estimate."""
    lv = _levels(x, sr, 50)
    w = max(1, int(0.5 / 0.05))
    if len(lv) < w:
        return float(lv.min())
    return float(min(lv[i:i + w].mean() for i in range(0, len(lv) - w + 1, 2)))


def _quietest_clip(x, sr, sec=1.0):
    n = int(sr * sec)
    if len(x) <= n:
        return x
    hop = n // 2
    best = min(range(0, len(x) - n, hop), key=lambda i: np.sqrt((x[i:i + n] ** 2).mean()))
    return x[best:best + n]


def clean_part(x, sr=SR, strength=0.7):
    x = sosfiltfilt(butter(4, 60, "high", fs=sr, output="sos"), x)
    y = nr.reduce_noise(y=x, sr=sr, y_noise=_quietest_clip(x, sr), stationary=True,
                        prop_decrease=strength, n_fft=2048, freq_mask_smooth_hz=400, time_mask_smooth_ms=80)
    return y.astype(np.float32)


def trim(x, sr=SR, margin=0.25):
    lv = _levels(x, sr)
    thr = noise_floor_db(x, sr) + 10
    idx = np.where(lv > thr)[0]
    if idx.size == 0:
        return x
    a = max(0, int(idx[0] * 0.02 * sr - margin * sr))
    b = min(len(x), int((idx[-1] + 1) * 0.02 * sr + margin * sr))
    return x[a:b]


def _peak_norm(x, db=-1.0):
    p = np.abs(x).max()
    return x * (10 ** (db / 20) / p) if p > 0 else x


def pick_prompt(x, sr=SR, sec=PROMPT_SEC):
    """Window with the highest fraction of speech frames; start snapped to a quiet frame."""
    n = int(sec * sr)
    if len(x) <= n:
        return x
    lv = _levels(x, sr)
    speech = (lv > noise_floor_db(x, sr) + 12).astype(float)
    fw = int(sec / 0.02)
    cs = np.concatenate([[0], np.cumsum(speech)])
    starts = range(0, len(speech) - fw, 25)  # 0.5 s step
    best = max(starts, key=lambda s: cs[s + fw] - cs[s])
    lo, hi = max(0, best - 25), min(len(lv) - fw, best + 25)
    best = lo + int(np.argmin(lv[lo:hi + 1])) if hi >= lo else best
    return x[best * int(0.02 * sr): best * int(0.02 * sr) + n]


def enroll(speaker: str, strength: float = 0.7) -> dict:
    from pathlib import Path
    d = Path(REF_ROOT) / speaker
    parts = {p: d / "parts" / f"{p}.wav" for p in "ABC" if (d / "parts" / f"{p}.wav").exists()}
    if not parts:
        raise RuntimeError("no recorded parts found")
    gap = np.zeros(int(0.5 * SR), np.float32)
    raw_all, clean_all, clean_parts = [], [], {}
    for name, path in parts.items():
        x, sr = sf.read(path, dtype="float32")
        assert sr == SR, f"{path} must be {SR} Hz"
        raw_all += [x, gap]
        c = _peak_norm(trim(clean_part(x, SR, strength)))
        clean_parts[name] = c
        clean_all += [c, gap]
    sf.write(d / "raw.wav", np.concatenate(raw_all), SR, subtype="FLOAT")
    sf.write(d / "clean.wav", np.concatenate(clean_all), SR, subtype="FLOAT")
    base = clean_parts["A"] if "A" in clean_parts else next(iter(clean_parts.values()))
    sf.write(d / "prompt_15s.wav", pick_prompt(base), SR, subtype="PCM_24")
    if "B" in clean_parts:
        sf.write(d / "prompt_expressive.wav", pick_prompt(clean_parts["B"]), SR, subtype="PCM_24")
    raw_cat = np.concatenate(raw_all)
    return {"parts": list(parts), "seconds": round(sum(len(c) for c in clean_parts.values()) / SR, 1),
            "noise_floor_before_db": round(noise_floor_db(raw_cat), 1),
            "noise_floor_after_db": round(noise_floor_db(np.concatenate(clean_all)), 1)}


def part_stats(x, sr=SR) -> dict:
    peak = float(np.abs(x).max())
    lv = _levels(x, sr)
    nf = noise_floor_db(x, sr)
    voiced = lv > nf + 12
    return {"seconds": round(len(x) / sr, 1), "peak_db": round(20 * np.log10(peak + 1e-9), 1),
            "clip_pct": round(float((np.abs(x) > 0.995).mean() * 100), 3), "noise_floor_db": round(nf, 1),
            "speech_sec": round(float(voiced.sum() * 0.02), 1),
            "snr_db": round(float(np.percentile(lv[voiced], 75) - nf), 1) if voiced.any() else 0.0}
