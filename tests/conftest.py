import numpy as np
import pytest

from kalvi.asr.whisper_onnx import WhisperONNX
from kalvi.config import Settings

from .fake_whisper import build_fake_whisper

SR = 16000


def speech_bursts(pattern: list[tuple[str, float]]) -> np.ndarray:
    """Build audio from ("speech"|"silence", seconds) pairs."""
    rng = np.random.default_rng(0)
    parts = []
    for kind, seconds in pattern:
        n = int(seconds * SR)
        if kind == "speech":
            t = np.arange(n) / SR
            parts.append((0.2 * np.sin(2 * np.pi * 200 * t) + 0.02 * rng.normal(size=n)).astype(np.float32))
        else:
            parts.append((0.001 * rng.normal(size=n)).astype(np.float32))
    return np.concatenate(parts)


class FakeMic:
    """Plays a fixed audio array in 100 ms blocks, then goes quiet."""

    def __init__(self, audio: np.ndarray):
        self.audio = audio
        self.pos = 0
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def read(self, timeout: float = 0.5):
        if self.pos >= len(self.audio):
            import time

            time.sleep(0.01)
            return None
        block = self.audio[self.pos : self.pos + SR // 10]
        self.pos += len(block)
        return block

    def stop(self):
        self.stopped = True


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.models_dir = tmp_path / "models"
    s.device = "cpu"
    s.language = "auto"
    return s


@pytest.fixture
def engine(tmp_path):
    return WhisperONNX(build_fake_whisper(tmp_path / "models" / "whisper_base" / "cpu"), device="cpu")
