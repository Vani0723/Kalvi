"""Speech recognition engines."""

from __future__ import annotations

import time

import numpy as np

from .features import SAMPLE_RATE
from .whisper_onnx import Timings, Transcription, WhisperONNX


class MockASR:
    """Stand-in engine for working on the UI without model files."""

    device = "mock"
    load_seconds = 0.0

    def __init__(self) -> None:
        self._count = 0

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> Transcription:
        self._count += 1
        seconds = len(audio) / SAMPLE_RATE
        start = time.perf_counter()
        text = f"[mock caption {self._count}: {seconds:.1f} s of speech]"
        timings = Timings(encoder_ms=(time.perf_counter() - start) * 1000)
        return Transcription(text, language if language != "auto" else "en", [], seconds, timings)


def load_engine(settings) -> "WhisperONNX | MockASR":
    """Create the speech engine selected by KALVI_DEVICE."""
    from .. import runtime

    if settings.device == "mock":
        return MockASR()
    device = runtime.resolve_device(settings.device)
    return WhisperONNX(settings.model_path(device), device)


__all__ = ["MockASR", "Timings", "Transcription", "WhisperONNX", "load_engine"]
