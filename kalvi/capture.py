"""Microphone capture and speech segmentation for live captions.

Audio arrives in small blocks. ``Segmenter`` groups them into utterances with
a simple energy-based voice activity detector, so Whisper runs on whole
phrases (better accuracy) and silence never reaches the NPU (saves power).
"""

from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np

FRAME_S = 0.03  # 30 ms analysis frames


@dataclass
class Segment:
    start_s: float
    audio: np.ndarray
    sample_rate: int = 16000

    @property
    def end_s(self) -> float:
        return self.start_s + len(self.audio) / self.sample_rate


class Segmenter:
    """Energy-based voice activity detection with hangover and a max length."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.012,
        min_segment_s: float = 0.5,
        max_segment_s: float = 20.0,
        silence_hangover_s: float = 0.6,
        pre_roll_s: float = 0.3,
    ):
        self.sr = sample_rate
        self.threshold = threshold
        self.frame = int(FRAME_S * sample_rate)
        self.min_len = int(min_segment_s * sample_rate)
        self.max_len = int(max_segment_s * sample_rate)
        self.hangover_frames = max(1, int(silence_hangover_s / FRAME_S))
        self.pre_roll = int(pre_roll_s * sample_rate)

        self._pending = np.zeros(0, dtype=np.float32)  # samples not yet framed
        self._history = np.zeros(0, dtype=np.float32)  # recent silence, for pre-roll
        self._speech: list[np.ndarray] = []
        self._speech_len = 0
        self._voiced_len = 0  # voiced samples in the current segment
        self._speech_start = 0
        self._silent_frames = 0
        self._consumed = 0  # samples processed so far
        self.last_level = 0.0

    @property
    def in_speech(self) -> bool:
        return bool(self._speech)

    def _emit(self) -> Segment | None:
        audio = np.concatenate(self._speech) if self._speech else np.zeros(0, dtype=np.float32)
        start, voiced = self._speech_start, self._voiced_len
        self._speech, self._speech_len, self._voiced_len, self._silent_frames = [], 0, 0, 0
        if voiced < self.min_len:  # too little actual speech (a cough, a door)
            return None
        return Segment(start / self.sr, audio, self.sr)

    def feed(self, block: np.ndarray) -> list[Segment]:
        out: list[Segment] = []
        self._pending = np.concatenate([self._pending, np.asarray(block, dtype=np.float32).reshape(-1)])
        n_frames = len(self._pending) // self.frame
        for i in range(n_frames):
            frame = self._pending[i * self.frame : (i + 1) * self.frame]
            rms = float(np.sqrt(np.mean(frame**2)))
            self.last_level = rms
            voiced = rms >= self.threshold
            frame_start = self._consumed

            if self._speech:
                self._speech.append(frame)
                self._speech_len += len(frame)
                if voiced:
                    self._voiced_len += len(frame)
                self._silent_frames = 0 if voiced else self._silent_frames + 1
                if self._silent_frames >= self.hangover_frames or self._speech_len >= self.max_len:
                    seg = self._emit()
                    if seg:
                        out.append(seg)
            elif voiced:
                pre = self._history[-self.pre_roll :] if self.pre_roll else np.zeros(0, dtype=np.float32)
                self._speech = [pre, frame] if len(pre) else [frame]
                self._speech_len = len(pre) + len(frame)
                self._speech_start = frame_start - len(pre)
                self._voiced_len = len(frame)
                self._silent_frames = 0
            else:
                self._history = np.concatenate([self._history, frame])[-max(self.pre_roll, 1) :]
            self._consumed += len(frame)
        self._pending = self._pending[n_frames * self.frame :]
        return out

    def flush(self) -> list[Segment]:
        if self._speech:
            if len(self._pending):
                self._speech.append(self._pending)
                self._speech_len += len(self._pending)
            self._pending = np.zeros(0, dtype=np.float32)
            seg = self._emit()
            return [seg] if seg else []
        return []


class MicSource:
    """16 kHz mono microphone stream. Blocks are read with ``read()``."""

    def __init__(self, sample_rate: int = 16000, block_s: float = 0.1, device: int | str | None = None):
        import sounddevice as sd  # imported lazily: needs PortAudio

        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self.sample_rate = sample_rate

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            self._q.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=sample_rate, channels=1, dtype="float32",
            blocksize=int(sample_rate * block_s), callback=callback, device=device,
        )

    def start(self) -> None:
        self._stream.start()

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()
