"""Runtime settings, read from environment variables with sensible defaults.

Every setting can be overridden with an environment variable, for example:

    set KALVI_DEVICE=cpu
    set KALVI_LANGUAGE=ta
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # Where model folders live. Each model has one sub-folder per device,
    # for example models/whisper_base/npu and models/whisper_base/cpu.
    models_dir: Path = field(default_factory=lambda: Path(_env("KALVI_MODELS_DIR", str(ROOT / "models"))))
    # Where lectures (SQLite database and audio recordings) are stored.
    data_dir: Path = field(default_factory=lambda: Path(_env("KALVI_DATA_DIR", str(ROOT / "data"))))

    # Speech model folder name inside models_dir.
    asr_model: str = field(default_factory=lambda: _env("KALVI_ASR_MODEL", "whisper_base"))
    # "npu", "cpu", "auto" (NPU when available, else CPU) or "mock" (no model, for UI work).
    device: str = field(default_factory=lambda: _env("KALVI_DEVICE", "auto"))
    # Whisper language code ("en", "ta", "hi", ...) or "auto" to let the model detect it.
    language: str = field(default_factory=lambda: _env("KALVI_LANGUAGE", "auto"))

    # Voice activity detection for live captions.
    sample_rate: int = 16000
    vad_threshold: float = field(default_factory=lambda: float(_env("KALVI_VAD_THRESHOLD", "0.012")))
    min_segment_s: float = 0.5  # minimum voiced speech per caption
    max_segment_s: float = 20.0
    silence_hangover_s: float = 0.6

    host: str = field(default_factory=lambda: _env("KALVI_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("KALVI_PORT", "8765")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "kalvi.db"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    def model_path(self, device: str) -> Path:
        return self.models_dir / self.asr_model / device


settings = Settings()
