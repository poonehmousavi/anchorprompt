"""Render noise and masking variants for benchmarks that ship none.

SAKURA's variants were pre-rendered; MMAR's are not. Reproducing the same scheme here
keeps a transfer number comparable to the dev number -- a different corruption recipe
would make Phase 3 measure the recipe rather than the transfer.

  noise_XdB  additive white Gaussian at a measured SNR of X dB over the whole signal
  mask_P     random distributed chunks zeroed until P% of samples are silent

Answer-PRESERVING channel transformations (the "unseen" generalisation set, 2026-09-11):
nothing the question asks about is removed, so the only faithful output is the clean
answer and any rise in declining is damage, never calibration.

  gain_XdB     scale by X dB (negative: quieter)
  pad_Xs       X seconds of digital silence prepended AND appended (looks locally like
               the trained mask; probes the abstain reflex)
  reverb_Xs    convolve with a synthetic room impulse response of RT60 = X s
  band_tel     Butterworth band-pass 300-3400 Hz (telephone)
  band_4k      Butterworth low-pass 4 kHz (8 kHz telephone bandwidth)

Rendered once and cached on disk; the seed is the file path, so a variant is identical
across runs and across phases.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

SR = 16000
CHUNK_MS = 100


def _rng(path: str, tag: str) -> np.random.Generator:
    h = hashlib.blake2b(f"{path}|{tag}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "big"))


def add_noise(y: np.ndarray, snr_db: float, rng) -> np.ndarray:
    sig = float(np.mean(y ** 2))
    if sig <= 0:
        return y
    noise = rng.standard_normal(y.shape)
    noise *= np.sqrt(sig / (10 ** (snr_db / 10.0)) / max(float(np.mean(noise ** 2)), 1e-12))
    return (y + noise).astype(np.float32)


def apply_mask(y: np.ndarray, pct: float, rng) -> np.ndarray:
    """Zero distributed chunks until pct% of samples are silent."""
    if pct <= 0:
        return y
    if pct >= 100:
        return np.zeros_like(y)
    chunk = max(1, int(SR * CHUNK_MS / 1000))
    n_chunks = max(1, int(np.ceil(len(y) / chunk)))
    n_mask = int(round(n_chunks * pct / 100.0))
    out = y.copy()
    for c in rng.permutation(n_chunks)[:n_mask]:
        out[c * chunk:(c + 1) * chunk] = 0.0
    return out.astype(np.float32)


def apply_gain(y: np.ndarray, db: float) -> np.ndarray:
    return (y * (10.0 ** (db / 20.0))).astype(np.float32)


def apply_pad(y: np.ndarray, seconds: float) -> np.ndarray:
    """Digital silence of `seconds` on BOTH ends; the middle is bit-identical to `y`."""
    n = int(round(seconds * SR))
    z = np.zeros(n, dtype=np.float32)
    return np.concatenate([z, y.astype(np.float32), z])


def synth_rir(rt60: float, rng, sr: int = SR) -> np.ndarray:
    """Exponentially decaying Gaussian noise: energy falls 60 dB over `rt60` seconds."""
    n = max(1, int(rt60 * sr))
    t = np.arange(n) / sr
    decay = np.exp(-6.9078 * t / rt60)          # ln(1e3): -60 dB in amplitude over rt60
    h = rng.standard_normal(n) * decay
    h[0] = 1.0                                   # direct path first, at full level
    return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)


def apply_reverb(y: np.ndarray, rt60: float, rng) -> np.ndarray:
    """Convolve with a synthetic RIR, then rescale so the RMS matches the source."""
    from scipy.signal import fftconvolve
    rms_in = float(np.sqrt(np.mean(y ** 2)))
    if rms_in <= 0:
        return y.astype(np.float32)
    out = fftconvolve(y, synth_rir(rt60, rng))
    rms_out = float(np.sqrt(np.mean(out ** 2)))
    return (out * (rms_in / max(rms_out, 1e-12))).astype(np.float32)


BANDS = {"tel": (300.0, 3400.0), "4k": (None, 4000.0)}


def apply_band(y: np.ndarray, band: str) -> np.ndarray:
    """Zero-phase Butterworth (order 4) band-pass or low-pass; RMS rescaled to the source."""
    from scipy.signal import butter, sosfiltfilt
    lo, hi = BANDS[band]
    nyq = SR / 2.0
    if lo is None:
        sos = butter(4, hi / nyq, btype="low", output="sos")
    else:
        sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    if len(y) < 64:
        return y.astype(np.float32)
    out = sosfiltfilt(sos, y.astype(np.float64))
    rms_in, rms_out = float(np.sqrt(np.mean(y ** 2))), float(np.sqrt(np.mean(out ** 2)))
    if rms_in > 0 and rms_out > 0:
        out = out * (rms_in / rms_out)
    return out.astype(np.float32)


def transform(y: np.ndarray, condition: str, rng) -> np.ndarray:
    """Dispatch on the condition name. Raises ValueError for anything unknown, so a typo
    cannot silently evaluate clean audio under an attack label."""
    if condition.startswith("noise_"):
        return add_noise(y, float(condition[len("noise_"):-2]), rng)
    if condition.startswith("mask_"):
        return apply_mask(y, float(condition[len("mask_"):]), rng)
    if condition.startswith("gain_"):
        return apply_gain(y, float(condition[len("gain_"):-2]))
    if condition.startswith("pad_"):
        return apply_pad(y, float(condition[len("pad_"):-1]))
    if condition.startswith("reverb_"):
        return apply_reverb(y, float(condition[len("reverb_"):-1]), rng)
    if condition.startswith("band_"):
        return apply_band(y, condition[len("band_"):])
    raise ValueError(f"cannot render {condition!r}")


def render(src: str, condition: str, cache_root: Path) -> str:
    """Path to the rendered variant, creating it on first use."""
    import soundfile as sf

    dest = Path(cache_root) / condition / (
        hashlib.blake2b(str(src).encode(), digest_size=12).hexdigest() + ".wav")
    if _complete_wav(dest):
        return str(dest)
    import librosa

    y, _ = librosa.load(src, sr=SR, mono=True)
    rng = _rng(str(src), condition)
    out = transform(y, condition, rng)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:                      # keep it representable; SNR is unchanged by scale
        out = out / peak
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publish: several jobs render the same cache concurrently (the token sweep runs
    # L1/L4/L8 on identical items). A reader that saw `dest` while it was being written got
    # a header-only wav, zero audio frames, and Qwen's encoder raised
    # "index is out of bounds for dimension with size 0" (job 10751891).
    tmp = dest.with_name(f"{dest.stem}.{os.getpid()}.tmp.wav")
    sf.write(tmp, out, SR)
    os.replace(tmp, dest)
    return str(dest)


def _complete_wav(path: Path) -> bool:
    """A cached render counts only if it is readable and holds audio (a job preempted
    mid-write, before the atomic publish existed, could leave a truncated file)."""
    if not path.exists():
        return False
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames > 0
    except Exception:
        return False
