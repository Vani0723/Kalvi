"""Check that this machine is ready to run Kalvi on the Snapdragon NPU.

    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
problems = 0


def report(level: str, msg: str) -> None:
    global problems
    if level == FAIL:
        problems += 1
    print(f"[{level}] {msg}")


def main() -> None:
    machine = platform.machine()
    arch = "ARM64" if machine.upper() in ("ARM64", "AARCH64") else machine
    print(f"Python {platform.python_version()} ({arch}) on {platform.platform()}\n")

    if sys.platform == "win32" and arch != "ARM64":
        report(WARN, "This Python is not ARM64. On a Snapdragon PC it runs under emulation and cannot load the NPU "
                     "provider. Install the Windows ARM64 Python from python.org for running Kalvi.")
    else:
        report(OK, f"Native {arch} Python")

    for module, why in [
        ("numpy", "maths"), ("scipy", "audio resampling (optional)"), ("onnxruntime", "model inference"),
        ("tokenizers", "Whisper tokenizer"), ("soundfile", "reading/writing audio"),
        ("sounddevice", "microphone capture"), ("fastapi", "local web UI"), ("uvicorn", "local web server"),
        ("multipart", "file import (python-multipart)"), ("psutil", "battery measurement (optional)"),
    ]:
        try:
            mod = importlib.import_module(module)
            report(OK, f"{module} {getattr(mod, '__version__', '')} - {why}")
        except Exception as exc:  # noqa: BLE001
            optional = "optional" in why
            report(WARN if optional else FAIL, f"{module} not usable ({exc.__class__.__name__}: {exc}) - {why}")

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "QNNExecutionProvider" in providers:
            report(OK, "QNNExecutionProvider available: models can run on the Hexagon NPU")
        else:
            report(WARN, f"No QNNExecutionProvider (providers: {providers}). Install onnxruntime-qnn on a "
                         "Snapdragon PC for NPU inference; Kalvi will use the CPU.")
    except Exception:
        pass

    try:
        import sounddevice as sd

        default_in = sd.query_devices(kind="input")
        report(OK, f"Microphone: {default_in['name']}")
    except Exception as exc:  # noqa: BLE001
        report(WARN, f"No microphone found ({exc}). You can still import recordings.")

    for device in ("npu", "cpu"):
        folder = ROOT / "models" / "whisper_base" / device
        found = list(folder.rglob("*.onnx")) if folder.exists() else []
        level = OK if found else WARN
        hint = "" if found else (" - run scripts/fetch_models.py" if device == "npu" else " - run scripts/export_cpu_baseline.py")
        report(level, f"Whisper {device.upper()} models: {len(found)} ONNX files in {folder.relative_to(ROOT)}{hint}")
    tok = ROOT / "models" / "whisper_base" / "tokenizer.json"
    report(OK if tok.exists() else WARN, f"Tokenizer {'found' if tok.exists() else 'missing'}: {tok.relative_to(ROOT)}")

    print("\nReady." if problems == 0 else f"\n{problems} required item(s) missing. See README.md > Setup.")


if __name__ == "__main__":
    main()
