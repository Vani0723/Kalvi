"""Benchmark Whisper on the Snapdragon NPU against the CPU.

    python scripts/benchmark.py --audio samples/lecture.wav
    python scripts/benchmark.py --audio samples/lecture.wav --devices npu cpu --power-minutes 10

Writes benchmarks/results.json and benchmarks/results.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kalvi.asr.audio import load_audio  # noqa: E402
from kalvi.benchmark import benchmark_device, save_results, to_markdown, system_info  # noqa: E402
from kalvi.runtime import qnn_available  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, help="Speech recording to transcribe (WAV/FLAC). Up to 30 s is typical.")
    parser.add_argument("--model", default="whisper_base")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--devices", nargs="+", default=["npu", "cpu"], choices=["npu", "cpu"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--power-minutes", type=float, default=0.0, help="Also measure battery drain over N minutes per device.")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.audio:
        audio = load_audio(args.audio)
    else:
        print("No --audio given: using 20 s of synthetic audio. Use a real lecture clip for published numbers.")
        t = np.arange(20 * 16000) / 16000
        audio = (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    results = []
    for device in args.devices:
        if device == "npu" and not qnn_available():
            print("Skipping NPU: QNNExecutionProvider not available (needs onnxruntime-qnn on a Snapdragon PC).")
            continue
        model_dir = args.models_dir / args.model / device
        if not model_dir.exists():
            print(f"Skipping {device.upper()}: {model_dir} not found.")
            continue
        print(f"\nBenchmarking {device.upper()} ...")
        result = benchmark_device(model_dir, device, audio, runs=args.runs, language=args.language, power_minutes=args.power_minutes)
        print(f"  {result.total_ms_mean:.0f} ms per clip, real-time factor {result.real_time_factor}")
        print(f"  Transcript: {result.text[:120]}")
        results.append(result)

    if not results:
        raise SystemExit("Nothing was benchmarked. See the setup steps in README.md.")
    json_path, md_path = save_results(results, args.out)
    print("\n" + to_markdown(results, system_info()))
    print(f"Saved {json_path} and {md_path}")


if __name__ == "__main__":
    main()
