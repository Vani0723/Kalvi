"""Kalvi local web server (FastAPI). Serves the UI on http://127.0.0.1:8765.

The server only listens on localhost: the browser is just the UI, and all
audio and text stay on the machine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, runtime
from .asr import load_engine
from .asr.tokenizer import LANGUAGE_NAMES
from .config import ROOT, Settings, settings as default_settings
from .recorder import Recorder
from .store import Store

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"


class StartRequest(BaseModel):
    title: str | None = None
    language: str | None = None


class Hub:
    """Fan-out of recorder events (from worker threads) to WebSocket clients."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()

    def publish(self, event: dict[str, Any]) -> None:
        if self.loop is None:
            return
        for q in list(self.clients):
            self.loop.call_soon_threadsafe(q.put_nowait, event)


def create_app(settings: Settings | None = None, engine=None, source_factory=None) -> FastAPI:
    settings = settings or default_settings
    hub = Hub()
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        eng = engine if engine is not None else await asyncio.to_thread(load_engine, settings)
        store = Store(settings.db_path)
        state["engine"] = eng
        state["store"] = store
        state["recorder"] = Recorder(store, eng, settings, source_factory=source_factory, on_event=hub.publish)
        log.info("Kalvi %s ready on %s", __version__, runtime.device_label(eng.device))
        yield
        rec: Recorder = state["recorder"]
        if rec.recording:
            rec.stop()
        store.close()

    app = FastAPI(title="Kalvi", version=__version__, lifespan=lifespan)

    def store() -> Store:
        return state["store"]

    def recorder() -> Recorder:
        return state["recorder"]

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        eng = state["engine"]
        rec = recorder()
        return {
            "version": __version__,
            "device": eng.device,
            "device_label": runtime.device_label(eng.device),
            "model": settings.asr_model,
            "model_load_s": round(getattr(eng, "load_seconds", 0.0), 2),
            "language": settings.language,
            "languages": LANGUAGE_NAMES,
            "recording": rec.recording,
            "lecture_id": rec.lecture_id,
            "qnn_available": runtime.qnn_available(),
        }

    @app.get("/api/lectures")
    def lectures() -> list[dict[str, Any]]:
        return store().list_lectures()

    @app.get("/api/lectures/{lecture_id}")
    def lecture(lecture_id: int) -> dict[str, Any]:
        found = store().get_lecture(lecture_id)
        if not found:
            raise HTTPException(404, "Lecture not found")
        return found

    @app.delete("/api/lectures/{lecture_id}")
    def delete_lecture(lecture_id: int) -> dict[str, Any]:
        if recorder().lecture_id == lecture_id:
            raise HTTPException(409, "Stop the recording before deleting it.")
        found = store().delete_lecture(lecture_id)
        if not found:
            raise HTTPException(404, "Lecture not found")
        if found.get("audio_path"):
            Path(found["audio_path"]).unlink(missing_ok=True)
        return {"deleted": lecture_id}

    @app.get("/api/lectures/{lecture_id}/audio")
    def lecture_audio(lecture_id: int):
        found = store().get_lecture(lecture_id)
        if not found or not found.get("audio_path") or not Path(found["audio_path"]).exists():
            raise HTTPException(404, "No audio for this lecture")
        return FileResponse(found["audio_path"], media_type="audio/wav")

    @app.post("/api/record/start")
    def start(req: StartRequest) -> dict[str, Any]:
        try:
            return recorder().start(req.title, req.language)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:  # e.g. no microphone / PortAudio missing
            raise HTTPException(500, f"Could not start recording: {exc}") from exc

    @app.post("/api/record/stop")
    def stop() -> dict[str, Any]:
        lecture = recorder().stop()
        if lecture is None:
            raise HTTPException(409, "Not recording.")
        return lecture

    @app.post("/api/import")
    async def import_audio(file: UploadFile = File(...), title: str | None = Form(None), language: str | None = Form(None)) -> dict[str, Any]:
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            lecture_id = await asyncio.to_thread(
                recorder().import_file, tmp_path, title or Path(file.filename or "Imported lecture").stem, language
            )
        except Exception as exc:
            raise HTTPException(400, f"Could not read audio file: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)
        return {"lecture_id": lecture_id}

    @app.get("/api/benchmarks")
    def benchmarks() -> dict[str, Any]:
        path = ROOT / "benchmarks" / "results.json"
        if not path.exists():
            return {"results": []}
        return json.loads(path.read_text(encoding="utf-8"))

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        q: asyncio.Queue = asyncio.Queue()
        hub.clients.add(q)
        try:
            while True:
                await socket.send_json(await q.get())
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(q)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
