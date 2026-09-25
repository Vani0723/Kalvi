import numpy as np

from kalvi.capture import Segmenter
from kalvi.store import Store

from .conftest import SR, speech_bursts


def feed_all(seg: Segmenter, audio: np.ndarray):
    out = []
    for i in range(0, len(audio), SR // 10):
        out += seg.feed(audio[i : i + SR // 10])
    return out + seg.flush()


def test_segmenter_splits_on_silence():
    audio = speech_bursts([("silence", 1), ("speech", 2), ("silence", 1.5), ("speech", 3), ("silence", 1)])
    segs = feed_all(Segmenter(), audio)
    assert len(segs) == 2
    # Starts include ~0.3 s pre-roll before the speech onset.
    assert 0.6 <= segs[0].start_s <= 1.05
    assert 4.1 <= segs[1].start_s <= 4.55
    assert 2.0 <= len(segs[0].audio) / SR <= 3.0


def test_segmenter_drops_short_noise_and_caps_length():
    audio = speech_bursts([("silence", 1), ("speech", 0.3), ("silence", 2), ("speech", 25), ("silence", 1)])
    segs = feed_all(Segmenter(max_segment_s=10), audio)
    assert all(len(s.audio) / SR <= 10.05 for s in segs)
    assert len(segs) == 3  # the 0.3 s blip is dropped; 25 s of speech -> 10 + 10 + 5


def test_store_roundtrip(tmp_path):
    store = Store(tmp_path / "k.db")
    lid = store.create_lecture("Physics", "en", "npu")
    store.add_segment(lid, 1.0, 3.5, "Entropy is a measure of disorder.", "en", 120.0)
    store.add_segment(lid, 0.0, 0.9, "Good morning.", "en", 80.0)
    lecture = store.get_lecture(lid)
    assert [s["text"] for s in lecture["segments"]] == ["Good morning.", "Entropy is a measure of disorder."]
    listing = store.list_lectures()
    assert listing[0]["segment_count"] == 2 and listing[0]["duration_s"] == 3.5
    store.delete_lecture(lid)
    assert store.get_lecture(lid) is None
    assert store.segments(lid) == []
