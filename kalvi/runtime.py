"""ONNX Runtime session creation for the Snapdragon NPU (QNN) and the CPU.

On a Snapdragon X PC, `onnxruntime-qnn` provides the QNNExecutionProvider,
which runs models on the Hexagon NPU through the HTP backend. Models exported
from Qualcomm AI Hub as ``precompiled_qnn_onnx`` contain a compiled NPU
context and can only run with this provider.

The CPU path uses the standard CPUExecutionProvider and float ONNX models
exported with ``scripts/export_cpu_baseline.py``. It is used as the baseline
in benchmarks and as a fallback on machines without an NPU.
"""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

QNN = "QNNExecutionProvider"
CPU = "CPUExecutionProvider"

# ONNX tensor type strings -> numpy dtypes.
ONNX_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
    "tensor(uint16)": np.uint16,
    "tensor(int16)": np.int16,
    "tensor(bool)": np.bool_,
}


@dataclass
class DeviceInfo:
    requested: str
    provider: str
    label: str


def qnn_available() -> bool:
    return QNN in ort.get_available_providers()


def qnn_provider_options(perf_mode: str = "burst") -> dict[str, str]:
    """Options for the QNN execution provider using the HTP (NPU) backend."""
    backend = "QnnHtp.dll" if sys.platform == "win32" else "libQnnHtp.so"
    return {
        "backend_path": backend,
        # burst = lowest latency; power_saver / low_power_saver trade speed for battery.
        "htp_performance_mode": perf_mode,
        "htp_graph_finalization_optimization_mode": "3",
    }


def resolve_device(requested: str) -> str:
    """Turn "auto" into "npu" or "cpu" depending on what this machine supports."""
    requested = requested.lower()
    if requested == "auto":
        return "npu" if qnn_available() else "cpu"
    if requested == "npu" and not qnn_available():
        raise RuntimeError(
            "NPU requested but QNNExecutionProvider is not available. Install "
            "`onnxruntime-qnn` in a native Windows ARM64 Python on a Snapdragon PC, "
            "or set KALVI_DEVICE=cpu."
        )
    return requested


def create_session(model_path: str | Path, device: str, perf_mode: str = "burst") -> ort.InferenceSession:
    """Create an inference session on the NPU or CPU."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if device == "npu":
        # Never silently fall back to CPU: benchmarks must reflect the real NPU.
        opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers = [(QNN, qnn_provider_options(perf_mode))]
    elif device == "cpu":
        providers = [CPU]
    else:
        raise ValueError(f"Unknown device {device!r}; expected 'npu' or 'cpu'.")
    log.info("Loading %s on %s", Path(model_path).name, device)
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)


def device_label(device: str) -> str:
    if device == "npu":
        return "Snapdragon Hexagon NPU (QNN HTP)"
    if device == "mock":
        return "Mock engine (no model)"
    return f"CPU ({platform.machine() or 'unknown'})"


def input_dtypes(session: ort.InferenceSession) -> dict[str, np.dtype]:
    return {i.name: ONNX_DTYPES.get(i.type, np.float32) for i in session.get_inputs()}
