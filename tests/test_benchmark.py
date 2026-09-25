import json

import numpy as np

from kalvi.benchmark import benchmark_device, save_results

from .fake_whisper import build_fake_whisper


def test_benchmark_writes_results(tmp_path):
    model_dir = build_fake_whisper(tmp_path / "whisper_base" / "cpu")
    audio = np.zeros(16000 * 5, dtype=np.float32)
    result = benchmark_device(model_dir, "cpu", audio, runs=2)
    assert result.text == "hello world kalvi"
    assert result.model == "whisper_base"
    assert result.real_time_factor > 0

    json_path, md_path = save_results([result], tmp_path / "bench")
    data = json.loads(json_path.read_text())
    assert data["results"][0]["device"] == "cpu"
    assert "| Real-time factor (lower is better) |" in md_path.read_text()
