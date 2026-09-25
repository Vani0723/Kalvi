import numpy as np
import pytest

from kalvi.asr.audio import chunk_audio, to_mono_16k
from kalvi.asr.features import N_FRAMES, log_mel_spectrogram


def _speechlike(seconds: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(1)
    return (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t)) + 0.02 * rng.normal(size=t.shape)).astype(np.float32)


@pytest.mark.parametrize("n_mels", [80, 128])
def test_shape(n_mels):
    feats = log_mel_spectrogram(_speechlike(4.0), n_mels=n_mels)
    assert feats.shape == (1, n_mels, N_FRAMES)
    assert feats.dtype == np.float32


@pytest.mark.parametrize("n_mels", [80, 128])
def test_matches_huggingface_feature_extractor(n_mels):
    transformers = pytest.importorskip("transformers")
    audio = _speechlike(7.5)
    ref = transformers.WhisperFeatureExtractor(feature_size=n_mels)(
        audio, sampling_rate=16000, return_tensors="np"
    )["input_features"]
    ours = log_mel_spectrogram(audio, n_mels=n_mels)
    assert ref.shape == ours.shape
    np.testing.assert_allclose(ours, ref, atol=1e-4)


def test_resample_and_mono():
    stereo = np.stack([_speechlike(1.0, 44100), _speechlike(1.0, 44100)], axis=1)
    mono = to_mono_16k(stereo, 44100)
    assert mono.ndim == 1
    assert abs(mono.shape[0] - 16000) <= 1


def test_chunking():
    chunks = chunk_audio(np.zeros(16000 * 61, dtype=np.float32))
    assert [start for start, _ in chunks] == [0.0, 30.0, 60.0]
    assert chunks[-1][1].shape[0] == 16000
