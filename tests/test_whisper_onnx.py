import numpy as np
import pytest

from kalvi.asr.whisper_onnx import WhisperONNX

from .fake_whisper import DECODE_LEN, LAYERS, N_MELS, build_fake_whisper


@pytest.fixture(params=[False, True], ids=["float32", "float16"])
def engine(request, tmp_path):
    build_fake_whisper(tmp_path / "whisper", float16=request.param)
    return WhisperONNX(tmp_path / "whisper", device="cpu")


def test_reads_model_dimensions(engine):
    assert engine.num_layers == LAYERS
    assert engine.n_mels == N_MELS
    assert engine.decode_len == DECODE_LEN
    assert len(engine.cross_map) == 2 * LAYERS


def test_auto_language_detects_and_transcribes(engine):
    audio = np.random.default_rng(0).normal(0, 0.1, 16000 * 3).astype(np.float32)
    result = engine.transcribe(audio, language="auto")
    assert result.text == "hello world kalvi"
    assert result.language == "en"
    assert result.timings.decoder_steps == 5
    assert result.audio_seconds == pytest.approx(3.0)


def test_forced_language_skips_prompt_tokens(engine):
    audio = np.zeros(16000 * 2, dtype=np.float32)
    result = engine.transcribe(audio, language="ta")
    # Prompt is 4 tokens, so the first generated token comes from table row 3.
    assert result.text == "kalvi"
    assert result.language == "ta"


def test_long_audio_is_chunked(engine):
    audio = np.zeros(16000 * 65, dtype=np.float32)  # 3 chunks: 30 + 30 + 5 s
    result = engine.transcribe(audio, language="auto")
    assert result.text == " ".join(["hello world kalvi"] * 3)
    assert result.real_time_factor > 0


def test_missing_models_give_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_models"):
        WhisperONNX(tmp_path, device="cpu")
