"""Whisper speech recognition with ONNX Runtime on the Snapdragon NPU or CPU.

The encoder/decoder interface matches Qualcomm AI Hub's Whisper export
(``qai_hub_models.models.templates.hf_whisper``):

Encoder
    input_features            (1, n_mels, 3000)
    -> k_cache_cross_{i}, v_cache_cross_{i}   one pair per decoder layer

Decoder (one token per call, static shapes so it can run on the NPU)
    input_ids                 (1, 1)            int32
    attention_mask            (1, 1, 1, L)      float, MASK_NEG = masked
    k_cache_self_{i}_in       (heads, 1, head_dim, L - 1)
    v_cache_self_{i}_in       (heads, 1, L - 1, head_dim)
    k_cache_cross_{i}, v_cache_cross_{i}
    position_ids              (1,)              int32
    -> logits (1, vocab, 1, 1), k_cache_self_{i}_out, v_cache_self_{i}_out

Model dimensions (layers, heads, decode length, mel bins) are read from the
ONNX graphs, so the same code runs Whisper Base, Small or Large-V3-Turbo.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import runtime
from .audio import chunk_audio
from .features import SAMPLE_RATE, log_mel_spectrogram
from .tokenizer import WhisperTokenizer

log = logging.getLogger(__name__)

MASK_NEG = -100.0
_SELF_IN = re.compile(r"^([kv])_cache_self_(\d+)_in$")
_SELF_OUT = re.compile(r"^([kv])_cache_self_(\d+)_out$")
_CROSS = re.compile(r"^([kv])_cache_cross_(\d+)$")


@dataclass
class Timings:
    features_ms: float = 0.0
    encoder_ms: float = 0.0
    decoder_ms: float = 0.0
    decoder_steps: int = 0

    @property
    def total_ms(self) -> float:
        return self.features_ms + self.encoder_ms + self.decoder_ms

    @property
    def ms_per_token(self) -> float:
        return self.decoder_ms / self.decoder_steps if self.decoder_steps else 0.0

    def add(self, other: "Timings") -> None:
        self.features_ms += other.features_ms
        self.encoder_ms += other.encoder_ms
        self.decoder_ms += other.decoder_ms
        self.decoder_steps += other.decoder_steps


@dataclass
class Transcription:
    text: str
    language: str | None
    tokens: list[int]
    audio_seconds: float
    timings: Timings = field(default_factory=Timings)

    @property
    def real_time_factor(self) -> float:
        """Processing time divided by audio length. Below 1.0 = faster than real time."""
        return (self.timings.total_ms / 1000.0) / self.audio_seconds if self.audio_seconds else 0.0


def _order_key(name: str, pattern: re.Pattern[str]) -> tuple[int, int]:
    m = pattern.match(name)
    assert m
    return int(m.group(2)), 0 if m.group(1) == "k" else 1


def find_model_files(model_dir: Path) -> tuple[Path, Path]:
    """Locate encoder and decoder ONNX files inside an AI Hub export folder."""
    onnx_files = sorted(model_dir.rglob("*.onnx"))
    enc = [p for p in onnx_files if "encoder" in p.name.lower()]
    dec = [p for p in onnx_files if "decoder" in p.name.lower()]
    if not enc or not dec:
        raise FileNotFoundError(
            f"Expected Whisper encoder and decoder .onnx files in {model_dir}. "
            "Run scripts/fetch_models.py (NPU) or scripts/export_cpu_baseline.py (CPU)."
        )
    return enc[0], dec[0]


def find_tokenizer(model_dir: Path) -> Path:
    for candidate in (model_dir / "tokenizer.json", model_dir.parent / "tokenizer.json"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"tokenizer.json not found in {model_dir} or its parent folder.")


class WhisperONNX:
    """Greedy Whisper transcription with a static-shape KV-cache decoder."""

    def __init__(self, model_dir: str | Path, device: str = "auto", perf_mode: str = "burst"):
        self.model_dir = Path(model_dir)
        self.device = runtime.resolve_device(device)
        enc_path, dec_path = find_model_files(self.model_dir)
        self.tokenizer = WhisperTokenizer(find_tokenizer(self.model_dir))

        t0 = time.perf_counter()
        self.encoder = runtime.create_session(enc_path, self.device, perf_mode)
        self.decoder = runtime.create_session(dec_path, self.device, perf_mode)
        self.load_seconds = time.perf_counter() - t0

        self._enc_dtypes = runtime.input_dtypes(self.encoder)
        self._dec_dtypes = runtime.input_dtypes(self.decoder)
        for name, dtype in {**self._enc_dtypes, **self._dec_dtypes}.items():
            if dtype in (np.uint8, np.uint16, np.int8, np.int16):
                raise ValueError(
                    f"Input {name} uses quantized I/O ({dtype.__name__}). Download the "
                    "float precision Whisper assets from AI Hub instead."
                )
        self._inspect()
        log.info(
            "Whisper ready on %s: %d decoder layers, %d mels, decode length %d (loaded in %.1fs)",
            self.device, self.num_layers, self.n_mels, self.decode_len, self.load_seconds,
        )

    # ------------------------------------------------------------------ setup
    def _inspect(self) -> None:
        enc_in = self.encoder.get_inputs()[0]
        self.encoder_input = enc_in.name
        self.n_mels = int(enc_in.shape[1])
        self.encoder_outputs = [o.name for o in self.encoder.get_outputs()]

        dec_inputs = {i.name: i for i in self.decoder.get_inputs()}
        dec_names = [i.name for i in self.decoder.get_inputs()]
        self.decoder_outputs = [o.name for o in self.decoder.get_outputs()]

        self_in = [n for n in dec_names if _SELF_IN.match(n)]
        cross_in = [n for n in dec_names if _CROSS.match(n)]
        if self_in and cross_in and "input_ids" in dec_inputs:
            self.self_inputs = sorted(self_in, key=lambda n: _order_key(n, _SELF_IN))
            self.cross_inputs = sorted(cross_in, key=lambda n: _order_key(n, _CROSS))
            self.ids_input, self.mask_input, self.pos_input = "input_ids", "attention_mask", "position_ids"
        else:
            # Fall back to the positional order used by AI Hub:
            # input_ids, attention_mask, self caches..., cross caches..., position_ids
            n_cache = (len(dec_names) - 3) // 2
            self.ids_input, self.mask_input, self.pos_input = dec_names[0], dec_names[1], dec_names[-1]
            self.self_inputs = dec_names[2 : 2 + n_cache]
            self.cross_inputs = dec_names[2 + n_cache : -1]
        self.num_layers = len(self.self_inputs) // 2
        self.decode_len = int(dec_inputs[self.mask_input].shape[-1])
        self.self_shapes = {n: tuple(int(d) for d in dec_inputs[n].shape) for n in self.self_inputs}

        # Encoder outputs -> decoder cross-attention inputs.
        if set(self.encoder_outputs) >= set(self.cross_inputs):
            self.cross_map = {n: n for n in self.cross_inputs}
        else:
            ordered = sorted(self.encoder_outputs, key=lambda n: _order_key(n, _CROSS)) if all(
                _CROSS.match(n) for n in self.encoder_outputs
            ) else self.encoder_outputs
            self.cross_map = dict(zip(self.cross_inputs, ordered))

        # Decoder self-cache outputs -> next step's self-cache inputs.
        self.logits_output = self.decoder_outputs[0]
        out_self = [n for n in self.decoder_outputs if _SELF_OUT.match(n)]
        if len(out_self) == len(self.self_inputs):
            by_key = {_order_key(n, _SELF_OUT): n for n in out_self}
            self.self_map = {n: by_key[_order_key(n, _SELF_IN)] for n in self.self_inputs}
        else:
            self.self_map = dict(zip(self.self_inputs, self.decoder_outputs[1:]))

    # -------------------------------------------------------------- inference
    def _cast(self, name: str, value: np.ndarray, dtypes: dict[str, np.dtype]) -> np.ndarray:
        dtype = dtypes.get(name, np.float32)
        return value if value.dtype == dtype else value.astype(dtype)

    def _run_encoder(self, features: np.ndarray) -> dict[str, np.ndarray]:
        feed = {self.encoder_input: self._cast(self.encoder_input, features, self._enc_dtypes)}
        outputs = self.encoder.run(self.encoder_outputs, feed)
        by_name = dict(zip(self.encoder_outputs, outputs))
        return {dec_name: by_name[enc_name] for dec_name, enc_name in self.cross_map.items()}

    def _decode(self, cross: dict[str, np.ndarray], prompt: list[int], timings: Timings) -> list[int]:
        L = self.decode_len
        output_ids = list(prompt)
        mask = np.full((1, 1, 1, L), MASK_NEG, dtype=np.float32)
        position = np.zeros((1,), dtype=np.int32)
        caches = {n: np.zeros(shape, dtype=self._dec_dtypes.get(n, np.float32)) for n, shape in self.self_shapes.items()}
        cross_feed = {n: self._cast(n, v, self._dec_dtypes) for n, v in cross.items()}
        eot = self.tokenizer.eot

        for n in range(L - 1):
            mask[..., L - n - 1] = 0.0
            feed = {
                self.ids_input: self._cast(self.ids_input, np.array([[output_ids[n]]], dtype=np.int32), self._dec_dtypes),
                self.mask_input: self._cast(self.mask_input, mask, self._dec_dtypes),
                self.pos_input: self._cast(self.pos_input, position, self._dec_dtypes),
                **caches,
                **cross_feed,
            }
            t0 = time.perf_counter()
            outputs = self.decoder.run(None, feed)
            timings.decoder_ms += (time.perf_counter() - t0) * 1000
            timings.decoder_steps += 1

            out = dict(zip(self.decoder_outputs, outputs))
            caches = {n: out[o] for n, o in self.self_map.items()}
            position += 1

            if n < len(prompt) - 1:
                continue  # still feeding the forced prompt tokens
            next_id = int(np.argmax(out[self.logits_output].reshape(-1).astype(np.float32)))
            output_ids.append(next_id)
            if next_id == eot:
                break
        return output_ids

    def transcribe_chunk(self, audio: np.ndarray, language: str = "auto") -> Transcription:
        """Transcribe up to 30 s of 16 kHz mono audio."""
        timings = Timings()
        t0 = time.perf_counter()
        features = log_mel_spectrogram(audio, n_mels=self.n_mels)
        timings.features_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cross = self._run_encoder(features)
        timings.encoder_ms = (time.perf_counter() - t0) * 1000

        prompt = self.tokenizer.prompt(language)
        tokens = self._decode(cross, prompt, timings)
        detected = language if language != "auto" else (
            self.tokenizer.token_to_language(tokens[1]) if len(tokens) > 1 else None
        )
        return Transcription(
            text=self.tokenizer.decode(tokens[len(prompt):] if language != "auto" else tokens[1:]),
            language=detected,
            tokens=tokens,
            audio_seconds=len(audio) / SAMPLE_RATE,
            timings=timings,
        )

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> Transcription:
        """Transcribe audio of any length by splitting it into 30 s chunks."""
        total = Timings()
        texts, tokens, lang = [], [], None
        for _, chunk in chunk_audio(audio):
            result = self.transcribe_chunk(chunk, language)
            total.add(result.timings)
            if result.text:
                texts.append(result.text)
            tokens.extend(result.tokens)
            lang = lang or result.language
        return Transcription(" ".join(texts), lang, tokens, len(audio) / SAMPLE_RATE, total)
