"""Audio loading, resampling and chunking."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np

try:  # scipy gives better resampling; fall back to NumPy if it is not installed
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover
    resample_poly = None

from .features import CHUNK_SECONDS, SAMPLE_RATE


def to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        if resample_poly is not None:
            g = gcd(SAMPLE_RATE, sample_rate)
            audio = resample_poly(audio, SAMPLE_RATE // g, sample_rate // g).astype(np.float32)
        else:
            n_out = int(round(len(audio) * SAMPLE_RATE / sample_rate))
            audio = np.interp(np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio).astype(np.float32)
    return audio


def load_audio(path: str | Path) -> np.ndarray:
    """Load a WAV/FLAC/OGG file as 16 kHz mono float32."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono_16k(audio, sr)


def chunk_audio(audio: np.ndarray, seconds: int = CHUNK_SECONDS) -> list[tuple[float, np.ndarray]]:
    """Split audio into Whisper-sized chunks. Returns (start_seconds, chunk) pairs."""
    size = seconds * SAMPLE_RATE
    if audio.shape[0] == 0:
        return []
    return [(start / SAMPLE_RATE, audio[start : start + size]) for start in range(0, audio.shape[0], size)]
