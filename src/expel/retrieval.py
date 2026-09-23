"""Similarity keys over (audio, instruction) — what an insight is retrieved BY.

Scope-based retrieval was too coarse to work. Every item in a track received the same
block, so per-insight credit was collinear (measured: Gender insights had identical
damage counts, 59/300 and 55/300) and a rule learned on a barking dog was injected into
a question about a violin. A key computed from the audio itself lets a cluster form
around what the recording actually sounds like.

The key is legal at inference: it is computed from the audio and the instruction, the
only two things the actor is given. Nothing about which intervention was applied enters
it — the fingerprint of a masked clip is distinctive because masked audio SOUNDS
different, not because it is labelled.

That property is what makes the abstention route deployable. Silence and broadband noise
sit far from every clean recording in mel-statistics space, so unanswerable inputs
cluster on their own and retrieve the insight that tells the model to decline — without
anyone telling the model that this input was masked.

Audio is fingerprinted by log-mel statistics rather than a learned encoder: it is
CPU-cheap, needs no extra GPU pass, and is at its most discriminative exactly where this
has to work — silence has near-zero band energy and white noise is spectrally flat.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

SR = 16000
N_MELS = 32
TEXT_DIM = 64
AUDIO_WEIGHT = 0.5          # audio vs instruction share of the key


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _block(v: np.ndarray) -> np.ndarray:
    """Standardise one feature block so blocks contribute comparably to the cosine."""
    v = np.asarray(v, dtype=np.float32).ravel()
    sd = float(v.std())
    return (v - v.mean()) / sd if sd > 1e-6 else v - v.mean()


def audio_fingerprint(path: str) -> np.ndarray:
    """Log-mel statistics + spectral shape. Deterministic, L2-normalised.

    Blocks are standardised BEFORE concatenation. Raw units differ by three orders of
    magnitude -- spectral centroid and rolloff are in Hz, flatness is on [0, 1] -- so an
    unstandardised vector is essentially the Hz features alone. Measured: audio at
    -17 dB SNR scored 0.985 against its own clean source, while spectral flatness, the
    descriptor that actually detects broadband noise, moved 0.012 -> 0.559 and was worth
    2 of 74 dimensions.
    """
    import librosa

    y, _ = librosa.load(path, sr=SR, mono=True)
    dim = 2 * N_MELS + 10
    if y.size == 0 or not np.any(np.isfinite(y)):
        return np.zeros(dim, dtype=np.float32)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=512, hop_length=160)
    logmel = librosa.power_to_db(mel + 1e-10)

    flat = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    # Hz-valued descriptors go in as log-frequency, which is both scale-free and closer
    # to how the difference between these sounds is actually perceived.
    hz = [np.log1p(fn(y=y, sr=SR)) for fn in (librosa.feature.spectral_centroid,
                                              librosa.feature.spectral_bandwidth,
                                              librosa.feature.spectral_rolloff)]
    shape = np.concatenate([[float(x.mean()), float(x.std())] for x in ([flat, zcr] + hz)])

    v = np.concatenate([_block(logmel.mean(axis=1)), _block(logmel.std(axis=1)),
                        _block(shape)]).astype(np.float32)
    v[~np.isfinite(v)] = 0.0
    return _l2(v)


_WORD = re.compile(r"[a-z']+")


def instruction_fingerprint(text: str, dim: int = TEXT_DIM) -> np.ndarray:
    """Hashed bag of words. No vocabulary to fit, so train and test share a space."""
    v = np.zeros(dim, dtype=np.float32)
    for w in _WORD.findall(text.lower()):
        if len(w) < 3:
            continue
        h = int(hashlib.blake2b(w.encode(), digest_size=8).hexdigest(), 16)
        v[h % dim] += 1.0
    return _l2(v)


class KeyEncoder:
    """(audio, instruction) -> one L2-normalised vector. Audio fingerprints are cached.

    The cache is keyed by absolute path, so a variant wav and its clean source are
    separate entries — which is the point; they must not collapse to the same key.
    """

    def __init__(self, cache_path: str | Path | None = None,
                 audio_weight: float = AUDIO_WEIGHT):
        self.audio_weight = audio_weight
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict = {}
        if self.cache_path and self.cache_path.exists():
            with np.load(self.cache_path, allow_pickle=False) as z:
                self._cache = {k: z[k] for k in z.files}

    def audio(self, path: str) -> np.ndarray:
        key = str(Path(path).resolve())
        if key not in self._cache:
            self._cache[key] = audio_fingerprint(key)
        return self._cache[key]

    def encode(self, audio_path: str, instruction: str) -> np.ndarray:
        a = self.audio(audio_path) * self.audio_weight
        t = instruction_fingerprint(instruction) * (1.0 - self.audio_weight)
        return _l2(np.concatenate([a, t]))

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.cache_path, **self._cache)


class HashKeyEncoder:
    """Deterministic stand-in that reads no audio. Tests and CPU smoke runs only."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, audio_path: str, instruction: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in (Path(audio_path).parent.name, Path(audio_path).stem[:3]):
            h = int(hashlib.blake2b(token.encode(), digest_size=8).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return _l2(np.concatenate([v * AUDIO_WEIGHT,
                                   instruction_fingerprint(instruction, self.dim) * (1 - AUDIO_WEIGHT)]))

    def save(self) -> None:
        pass


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if a.shape != b.shape or not a.size:
        return 0.0
    return float(np.dot(a, b))          # both are already L2-normalised
