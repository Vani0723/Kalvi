import time

import soundfile as sf
from fastapi.testclient import TestClient

from kalvi.recorder import Recorder
from kalvi.server import create_app
from kalvi.store import Store

from .conftest import SR, FakeMic, speech_bursts

LECTURE = [("silence", 0.5), ("speech", 2), ("silence", 1.2), ("speech", 2.5), ("silence", 1)]


def wait_for(predicate, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_live_recording_produces_captions_and_audio(settings, engine):
    audio = speech_bursts(LECTURE)
    events = []
    store = Store(settings.db_path)
    mic = FakeMic(audio)
    rec = Recorder(store, engine, settings, source_factory=lambda: mic, on_event=events.append)

    lecture = rec.start("Thermodynamics", "auto")
    assert wait_for(lambda: mic.pos >= len(audio))
    assert wait_for(lambda: sum(e["type"] == "segment" for e in events) >= 1)
    saved = rec.stop()

    assert mic.started and mic.stopped
    assert saved["title"] == "Thermodynamics" and saved["ended_at"]
    assert len(saved["segments"]) == 2
    assert saved["segments"][0]["text"] == "hello world kalvi"
    assert saved["segments"][0]["start_s"] < saved["segments"][1]["start_s"]
    info = sf.info(saved["audio_path"])
    assert info.samplerate == SR and abs(info.duration - len(audio) / SR) < 0.2
    assert any(e["type"] == "level" for e in events)
    assert lecture["id"] == saved["id"]


def test_server_endpoints(settings, engine, tmp_path):
    audio = speech_bursts(LECTURE)
    app = create_app(settings, engine=engine, source_factory=lambda: FakeMic(audio))
    with TestClient(app) as client:
        status = client.get("/api/status").json()
        assert status["device"] == "cpu" and not status["recording"]
        assert "ta" in status["languages"]
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200

        with client.websocket_connect("/ws") as ws:
            started = client.post("/api/record/start", json={"title": "Maths"}).json()
            assert client.post("/api/record/start", json={}).status_code == 409
            seen = []
            while not any(e["type"] == "segment" for e in seen):
                seen.append(ws.receive_json())
            time.sleep(1.0)
            stopped = client.post("/api/record/stop").json()
        assert stopped["id"] == started["id"]
        assert len(stopped["segments"]) >= 1

        # Import an existing recording.
        wav = tmp_path / "phone_recording.wav"
        sf.write(wav, audio, SR)
        with open(wav, "rb") as f:
            res = client.post("/api/import", files={"file": ("phone_recording.wav", f, "audio/wav")}, data={"language": "ta"})
        imported_id = res.json()["lecture_id"]
        assert wait_for(lambda: client.get(f"/api/lectures/{imported_id}").json()["ended_at"] is not None)
        imported = client.get(f"/api/lectures/{imported_id}").json()
        assert imported["title"] == "phone_recording"
        assert [s["text"] for s in imported["segments"]] == ["kalvi", "kalvi"]
        assert client.get(f"/api/lectures/{imported_id}/audio").status_code == 200

        titles = [l["title"] for l in client.get("/api/lectures").json()]
        assert titles == ["phone_recording", "Maths"]
        assert client.delete(f"/api/lectures/{imported_id}").status_code == 200
        assert client.get(f"/api/lectures/{imported_id}").status_code == 404
        assert client.post("/api/record/stop").status_code == 409


def test_bad_upload_is_rejected(settings, engine):
    app = create_app(settings, engine=engine, source_factory=lambda: FakeMic(speech_bursts(LECTURE)))
    with TestClient(app) as client:
        res = client.post("/api/import", files={"file": ("notes.wav", b"not audio", "audio/wav")})
        assert res.status_code == 400
