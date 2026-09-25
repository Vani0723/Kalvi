"""Whisper log-mel spectrogram in pure NumPy.

This reproduces Hugging Face's ``WhisperFeatureExtractor`` (which Qualcomm AI
Hub's Whisper models are trained and exported against) without depending on
PyTorch or transformers at runtime, which keeps the Windows ARM64 install small.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_SECONDS = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS  # 480000
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    """Slaney-style mel scale (linear below 1 kHz, logarithmic above)."""
    freq = np.asarray(freq, dtype=np.float64)
    min_log_hz, min_log_mel = 1000.0, 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    log_region = freq >= min_log_hz
    mels = np.where(log_region, min_log_mel + np.log(np.maximum(freq, 1e-10) / min_log_hz) * logstep, mels)
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    min_log_hz, min_log_mel = 1000.0, 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    return np.where(log_region, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freq)


@lru_cache(maxsize=4)
def mel_filters(n_mels: int = 80, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT) -> np.ndarray:
    """Slaney-normalised triangular mel filter bank, shape (n_freq_bins, n_mels)."""
    n_freqs = 1 + n_fft // 2
    fft_freqs = np.linspace(0, sample_rate // 2, n_freqs)
    mel_min, mel_max = _hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(sample_rate / 2))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    filter_freqs = _mel_to_hz(mel_points)

    filter_diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / filter_diff[:-1]
    up = slopes[:, 2:] / filter_diff[1:]
    filters = np.maximum(0.0, np.minimum(down, up))

    enorm = 2.0 / (filter_freqs[2 : n_mels + 2] - filter_freqs[:n_mels])
    return (filters * enorm[None, :]).astype(np.float64)


@lru_cache(maxsize=1)
def _hann(n_fft: int = N_FFT) -> np.ndarray:
    # Periodic Hann window, as used by Whisper.
    return np.hanning(n_fft + 1)[:-1]


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.shape[0] >= length:
        return audio[:length]
    return np.pad(audio, (0, length - audio.shape[0]))


def log_mel_spectrogram(audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """Compute Whisper input features for up to 30 s of 16 kHz mono audio.

    Returns an array of shape (1, n_mels, 3000), dtype float32.
    """
    audio = pad_or_trim(audio).astype(np.float64)

    # Centre the frames with reflect padding (like torch.stft(center=True)).
    pad = N_FFT // 2
    padded = np.pad(audio, (pad, pad), mode="reflect")
    n_frames = 1 + (padded.shape[0] - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, N_FFT),
        strides=(padded.strides[0] * HOP_LENGTH, padded.strides[0]),
        writeable=False,
    )
    spectrum = np.fft.rfft(frames * _hann(), n=N_FFT, axis=-1)
    power = np.abs(spectrum) ** 2  # (n_frames, n_freqs)

    mel = power @ mel_filters(n_mels)  # (n_frames, n_mels)
    log_spec = np.log10(np.maximum(mel, 1e-10)).T  # (n_mels, n_frames)
    log_spec = log_spec[:, :-1]  # drop the last frame -> 3000 frames
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec[None, :, :].astype(np.float32)
