"""Lecture recording: microphone -> speech segments -> Whisper -> store.

Two worker threads per live lecture:

* capture: reads the microphone, writes the full recording to a WAV file
  (used later for Doubt Replay) and cuts speech into segments;
* transcribe: runs each segment through Whisper and stores the caption.

Events (captions, mic level, status) are passed to ``on_event`` so the web
server can push them to the browser over a WebSocket.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .asr.audio import chunk_audio, load_audio
from .capture import MicSource, Segment, Segmenter
from .config import Settings
from .store import Store

log = logging.getLogger(__name__)
EventFn = Callable[[dict[str, Any]], None]


class Recorder:
    def __init__(
        self,
        store: Store,
        engine,
        settings: Settings,
        source_factory: Callable[[], Any] | None = None,
        on_event: EventFn | None = None,
    ):
        self.store = store
        self.engine = engine
        self.settings = settings
        self.source_factory = source_factory or (lambda: MicSource(settings.sample_rate))
        self.on_event = on_event or (lambda event: None)
        self.lecture_id: int | None = None
        self.language = settings.language
        self._engine_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._segments: queue.Queue[Segment | None] = queue.Queue()
        self._started_at = 0.0
        self.error: str | None = None

    # ------------------------------------------------------------ helpers
    @property
    def recording(self) -> bool:
        return self.lecture_id is not None

    def _emit(self, event: dict[str, Any]) -> None:
        try:
            self.on_event(event)
        except Exception:  # never let a UI problem stop a recording
            log.exception("Event handler failed")

    def _transcribe(self, audio: np.ndarray, language: str):
        with self._engine_lock:
            return self.engine.transcribe(audio, language)

    def _audio_path(self, lecture_id: int) -> Path:
        self.settings.audio_dir.mkdir(parents=True, exist_ok=True)
        return self.settings.audio_dir / f"lecture_{lecture_id}.wav"

    def _store_result(self, lecture_id: int, start_s: float, end_s: float, result) -> dict[str, Any] | None:
        text = result.text.strip()
        if not text:
            return None
        return self.store.add_segment(lecture_id, round(start_s, 2), round(end_s, 2), text, result.language, round(result.timings.total_ms, 1))

    # --------------------------------------------------------- live lecture
    def start(self, title: str | None = None, language: str | None = None) -> dict[str, Any]:
        if self.recording:
            raise RuntimeError("A lecture is already being recorded.")
        self.language = language or self.settings.language
        title = title or time.strftime("Lecture %d %b %Y, %H:%M")
        source = self.source_factory()  # fail early if the microphone is missing

        lecture_id = self.store.create_lecture(title, self.language, self.engine.device)
        audio_path = self._audio_path(lecture_id)
        self.store.set_audio_path(lecture_id, str(audio_path))
        self.lecture_id = lecture_id
        self.error = None
        self._stop.clear()
        self._segments = queue.Queue()
        self._started_at = time.time()

        self._threads = [
            threading.Thread(target=self._capture_loop, args=(source, audio_path), daemon=True, name="kalvi-capture"),
            threading.Thread(target=self._transcribe_loop, args=(lecture_id,), daemon=True, name="kalvi-asr"),
        ]
        for t in self._threads:
            t.start()
        self._emit({"type": "status", "recording": True, "lecture_id": lecture_id})
        return self.store.get_lecture(lecture_id)

    def _capture_loop(self, source, audio_path: Path) -> None:
        import soundfile as sf

        s = self.settings
        segmenter = Segmenter(s.sample_rate, s.vad_threshold, s.min_segment_s, s.max_segment_s, s.silence_hangover_s)
        last_level = 0.0
        try:
            source.start()
            with sf.SoundFile(str(audio_path), "w", samplerate=s.sample_rate, channels=1, subtype="PCM_16") as wav:
                while not self._stop.is_set():
                    block = source.read()
                    if block is None:
                        continue
                    wav.write(block)
                    for seg in segmenter.feed(block):
                        self._segments.put(seg)
                    now = time.time()
                    if now - last_level > 0.15:
                        last_level = now
                        self._emit({"type": "level", "rms": round(segmenter.last_level, 4), "speaking": segmenter.in_speech,
                                    "elapsed_s": round(now - self._started_at, 1)})
                for seg in segmenter.flush():
                    self._segments.put(seg)
        except Exception as exc:
            log.exception("Audio capture failed")
            self.error = f"Microphone error: {exc}"
            self._emit({"type": "error", "message": self.error})
        finally:
            try:
                source.stop()
            except Exception:
                pass
            self._segments.put(None)

    def _transcribe_loop(self, lecture_id: int) -> None:
        while True:
            seg = self._segments.get()
            if seg is None:
                break
            self._emit({"type": "transcribing", "start_s": round(seg.start_s, 2)})
            try:
                result = self._transcribe(seg.audio, self.language)
            except Exception as exc:
                log.exception("Transcription failed")
                self._emit({"type": "error", "message": f"Transcription error: {exc}"})
                continue
            row = self._store_result(lecture_id, seg.start_s, seg.end_s, result)
            if row:
                self._emit({"type": "segment", "segment": row, "device": self.engine.device})

    def stop(self) -> dict[str, Any] | None:
        if not self.recording:
            return None
        lecture_id = self.lecture_id
        self._stop.set()
        for t in self._threads:
            t.join(timeout=120)
        self.store.end_lecture(lecture_id)
        self.lecture_id = None
        self._emit({"type": "status", "recording": False, "lecture_id": lecture_id})
        return self.store.get_lecture(lecture_id)

    # ---------------------------------------------------------- file import
    def import_file(self, path: str | Path, title: str | None = None, language: str | None = None, background: bool = True) -> int:
        """Transcribe an existing recording (e.g. a lecture recorded on a phone)."""
        language = language or self.settings.language
        audio = load_audio(path)
        lecture_id = self.store.create_lecture(title or Path(path).stem, language, self.engine.device, source="import")
        audio_path = self._audio_path(lecture_id)

        import soundfile as sf

        sf.write(str(audio_path), audio, self.settings.sample_rate, subtype="PCM_16")
        self.store.set_audio_path(lecture_id, str(audio_path))

        def work() -> None:
            s = self.settings
            segmenter = Segmenter(s.sample_rate, s.vad_threshold, s.min_segment_s, s.max_segment_s, s.silence_hangover_s)
            block = s.sample_rate // 10
            segments: list[Segment] = []
            for i in range(0, len(audio), block):
                segments += segmenter.feed(audio[i : i + block])
            segments += segmenter.flush()
            if not segments:  # very quiet recording: fall back to fixed 30 s chunks
                segments = [Segment(start, chunk, s.sample_rate) for start, chunk in chunk_audio(audio)]
            total_s = len(audio) / s.sample_rate
            for seg in segments:
                try:
                    result = self._transcribe(seg.audio, language)
                except Exception as exc:
                    log.exception("Import transcription failed")
                    self._emit({"type": "error", "message": f"Transcription error: {exc}"})
                    continue
                row = self._store_result(lecture_id, seg.start_s, seg.end_s, result)
                if row:
                    self._emit({"type": "segment", "segment": row, "device": self.engine.device})
                self._emit({"type": "import_progress", "lecture_id": lecture_id, "done_s": round(seg.end_s, 1), "total_s": round(total_s, 1)})
            self.store.end_lecture(lecture_id)
            self._emit({"type": "import_done", "lecture_id": lecture_id})

        if background:
            threading.Thread(target=work, daemon=True, name="kalvi-import").start()
        else:
            work()
        return lecture_id
