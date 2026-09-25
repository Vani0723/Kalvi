"""Tiny ONNX models with the same interface as Qualcomm AI Hub's Whisper export.

They let the test suite exercise the real decode loop through ONNX Runtime
without downloading any weights.

The fake decoder picks its next token from a lookup table indexed by
``(cache_step + position_ids) // 2``, where ``cache_step`` is the maximum value
in ``k_cache_self_0_in``. Each call returns the self-attention caches plus one,
so the expected token sequence only comes out if the decoder loop threads the
caches and positions correctly from step to step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HEADS, HEAD_DIM, DECODE_LEN, LAYERS, N_MELS, AUDIO_EMB = 2, 4, 8, 2, 80, 1500

VOCAB = {
    "hello": 0,
    "world": 1,
    "kalvi": 2,
    "<|endoftext|>": 3,
    "<|startoftranscript|>": 4,
    "<|en|>": 5,
    "<|ta|>": 6,
    "<|translate|>": 7,
    "<|transcribe|>": 8,
    "<|notimestamps|>": 9,
}
# Table row -> token produced at that step.
SCRIPT = ["<|en|>", "hello", "world", "kalvi", "<|endoftext|>"]


def write_tokenizer(path: Path) -> None:
    special = [t for t in VOCAB if t.startswith("<|")]
    tok = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": VOCAB[t], "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
            for t in special
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": VOCAB, "unk_token": "<|endoftext|>"},
    }
    path.write_text(json.dumps(tok))


def _float_type(float16: bool) -> int:
    return TensorProto.FLOAT16 if float16 else TensorProto.FLOAT


def build_encoder(path: Path, float16: bool = False) -> None:
    ft = _float_type(float16)
    np_ft = np.float16 if float16 else np.float32
    inputs = [helper.make_tensor_value_info("input_features", ft, [1, N_MELS, 3000])]
    outputs, nodes, inits = [], [], []
    for i in range(LAYERS):
        for kind, shape in (("k", [HEADS, 1, HEAD_DIM, AUDIO_EMB]), ("v", [HEADS, 1, AUDIO_EMB, HEAD_DIM])):
            name = f"{kind}_cache_cross_{i}"
            inits.append(numpy_helper.from_array(np.zeros(shape, dtype=np_ft), f"{name}_const"))
            nodes.append(helper.make_node("Identity", [f"{name}_const"], [name]))
            outputs.append(helper.make_tensor_value_info(name, ft, shape))
    graph = helper.make_graph(nodes, "fake_encoder", inputs, outputs, inits)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), str(path))


def build_decoder(path: Path, float16: bool = False) -> None:
    ft = _float_type(float16)
    np_ft = np.float16 if float16 else np.float32
    vocab_size = len(VOCAB)
    L = DECODE_LEN

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT32, [1, 1]),
        helper.make_tensor_value_info("attention_mask", ft, [1, 1, 1, L]),
    ]
    self_shapes = {}
    for i in range(LAYERS):
        self_shapes[f"k_cache_self_{i}_in"] = [HEADS, 1, HEAD_DIM, L - 1]
        self_shapes[f"v_cache_self_{i}_in"] = [HEADS, 1, L - 1, HEAD_DIM]
    for name, shape in self_shapes.items():
        inputs.append(helper.make_tensor_value_info(name, ft, shape))
    for i in range(LAYERS):
        inputs.append(helper.make_tensor_value_info(f"k_cache_cross_{i}", ft, [HEADS, 1, HEAD_DIM, AUDIO_EMB]))
        inputs.append(helper.make_tensor_value_info(f"v_cache_cross_{i}", ft, [HEADS, 1, AUDIO_EMB, HEAD_DIM]))
    inputs.append(helper.make_tensor_value_info("position_ids", TensorProto.INT32, [1]))

    table = np.full((L, vocab_size), -10.0, dtype=np_ft)
    for row in range(L):
        token = SCRIPT[row] if row < len(SCRIPT) else "<|endoftext|>"
        table[row, VOCAB[token]] = 10.0

    inits = [
        numpy_helper.from_array(table, "table"),
        numpy_helper.from_array(np.array(1.0, dtype=np_ft), "one"),
        numpy_helper.from_array(np.array([2], dtype=np.int32), "two"),
        numpy_helper.from_array(np.array([1, vocab_size, 1, 1], dtype=np.int64), "logits_shape"),
    ]
    nodes = [
        helper.make_node("ReduceMax", ["k_cache_self_0_in"], ["cache_max"], keepdims=0),
        helper.make_node("Cast", ["cache_max"], ["cache_step_scalar"], to=TensorProto.INT32),
        helper.make_node("Unsqueeze", ["cache_step_scalar", "axes0"], ["cache_step"]),
        helper.make_node("Add", ["cache_step", "position_ids"], ["step_sum"]),
        helper.make_node("Div", ["step_sum", "two"], ["row"]),
        helper.make_node("Gather", ["table", "row"], ["row_logits"], axis=0),
        helper.make_node("Reshape", ["row_logits", "logits_shape"], ["logits"]),
    ]
    inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), "axes0"))
    outputs = [helper.make_tensor_value_info("logits", ft, [1, vocab_size, 1, 1])]
    for name, shape in self_shapes.items():
        out = name.replace("_in", "_out")
        nodes.append(helper.make_node("Add", [name, "one"], [out]))
        outputs.append(helper.make_tensor_value_info(out, ft, shape))

    graph = helper.make_graph(nodes, "fake_decoder", inputs, outputs, inits)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9), str(path))


def build_fake_whisper(model_dir: Path, float16: bool = False) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    build_encoder(model_dir / "WhisperEncoder.onnx", float16)
    build_decoder(model_dir / "WhisperDecoder.onnx", float16)
    write_tokenizer(model_dir / "tokenizer.json")
    return model_dir
