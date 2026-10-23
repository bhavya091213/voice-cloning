"""Mastering chain: HPF -> light denoise -> de-ess -> EQ -> multiband comp -> bus comp (~3 dB GR) -> loudness -> -1 dBTP limit."""
import numpy as np
import noisereduce as nr
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import Compressor, HighpassFilter, HighShelfFilter, Limiter, LowShelfFilter, Pedalboard, PeakFilter
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import butter, resample_poly, sosfiltfilt

OUT_SR = 48000


def _pb(x, sr, *plugins):
    return Pedalboard(list(plugins))(x.astype(np.float32)[None, :], sr)[0]


def _levels(x, sr, ms=10):
    n = int(sr * ms / 1000)
    k = len(x) // n
    return 20 * np.log10(np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-9)


def _adaptive_th(x, sr, pct):
    lv = _levels(x, sr)
    active = lv[lv > lv.max() - 45]
    return float(np.percentile(active, pct)) if active.size else -40.0


def _split(x, sr, f):
    lo = sosfiltfilt(butter(4, f, "low", fs=sr, output="sos"), x)
    return lo, x - lo  # complementary: sums back exactly


def deess(x, sr, freq=5500, pct=88, ratio=4.0):
    lo, hi = _split(x, sr, freq)
    th = _adaptive_th(hi, sr, pct)
    return lo + _pb(hi, sr, Compressor(threshold_db=th, ratio=ratio, attack_ms=1, release_ms=40))


def multiband(x, sr, pct=60, ratio=2.0):
    low, rest = _split(x, sr, 180)
    mid, high = _split(rest, sr, 3500)
    out = 0
    for band, atk, rel in ((low, 20, 150), (mid, 12, 120), (high, 6, 80)):
        out = out + _pb(band, sr, Compressor(threshold_db=_adaptive_th(band, sr, pct), ratio=ratio,
                                             attack_ms=atk, release_ms=rel))
    return out


def _mean_gr(pre, post, sr):
    a, b = _levels(pre, sr, 50), _levels(post, sr, 50)
    m = a > a.max() - 30
    return float(np.mean(a[m] - b[m]))


def tune_bus(x, sr, ref, target_gr=3.0, ratio=2.5):
    """Bisect the bus-compressor threshold so total gain reduction (vs `ref`) averages target_gr dB."""
    lo, hi, best = -50.0, 0.0, None
    for _ in range(10):
        th = (lo + hi) / 2
        y = _pb(x, sr, Compressor(threshold_db=th, ratio=ratio, attack_ms=10, release_ms=100))
        g = _mean_gr(ref, y, sr)
        best = (y, g)
        lo, hi = (lo, th) if g < target_gr else (th, hi)
    return best


def limit(x, sr, ceiling_db, lookahead_ms=5):
    """Lookahead peak limiter: gain never exceeds ceiling/|x|; min-filter then smooth so gain moves gradually."""
    w = int(sr * lookahead_ms / 1000)
    c = 10 ** (ceiling_db / 20)
    need = np.minimum(1.0, c / np.maximum(np.abs(x), 1e-9))
    g = uniform_filter1d(minimum_filter1d(need, size=2 * w + 1), size=w + 1)
    return x * np.minimum(g, need)


def true_peak_db(x):
    return float(20 * np.log10(np.abs(resample_poly(x, 4, 1)).max() + 1e-12))


def master(in_path, out_path, denoise=0.25, target_lufs=-16.0, ceiling_db=-1.0, target_gr=3.0) -> dict:
    x, sr = sf.read(in_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != OUT_SR:
        g = np.gcd(sr, OUT_SR)
        x = resample_poly(x, OUT_SR // g, sr // g).astype(np.float32)
    sr = OUT_SR
    x = sosfiltfilt(butter(4, 70, "high", fs=sr, output="sos"), x)
    if denoise > 0:
        n = int(0.4 * sr)
        quiet = min(range(0, max(1, len(x) - n), n // 2), key=lambda i: np.abs(x[i:i + n]).mean())
        x = nr.reduce_noise(y=x, sr=sr, y_noise=x[quiet:quiet + n], stationary=True, prop_decrease=denoise)
    x = deess(x, sr)
    x = _pb(x, sr, LowShelfFilter(120, 1.0), PeakFilter(250, -1.5, 0.9), PeakFilter(3500, 1.2, 0.8),
            HighShelfFilter(8500, 1.5))
    pre = x.copy()
    x = multiband(x, sr)
    x, gr = tune_bus(x, sr, pre, target_gr)
    meter = pyln.Meter(sr)
    x = pyln.normalize.loudness(x.astype(np.float64), meter.integrated_loudness(x.astype(np.float64)), target_lufs)
    for _ in range(3):  # limiter trims loudness slightly; nudge back to target and re-limit
        x = limit(x, sr, ceiling_db - 0.3)
        err = target_lufs - meter.integrated_loudness(x.astype(np.float64))
        if abs(err) < 0.15:
            break
        x = x * 10 ** (err / 20)
    x = limit(x, sr, ceiling_db - 0.3)
    tp = true_peak_db(x)
    if tp > ceiling_db:
        x = x * 10 ** ((ceiling_db - tp) / 20)
    sf.write(out_path, x, sr, subtype="PCM_24")
    return {"lufs": round(float(meter.integrated_loudness(x.astype(np.float64))), 1),
            "true_peak_db": round(true_peak_db(x), 2), "gain_reduction_db": round(gr, 2), "sample_rate": sr}
