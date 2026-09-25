"""NPU vs CPU benchmark for the Whisper speech pipeline.

Measures, per device: model load time, encoder latency, decoder latency per
token, end-to-end transcription time, real-time factor and (optionally)
battery drain during a sustained run. Results are written as JSON and as a
Markdown table ready for the README.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .asr.features import SAMPLE_RATE
from .asr.whisper_onnx import WhisperONNX


@dataclass
class DeviceResult:
    device: str
    model: str
    load_s: float
    runs: int
    audio_s: float
    encoder_ms: float
    ms_per_token: float
    tokens_per_s: float
    total_ms_mean: float
    total_ms_p50: float
    total_ms_p90: float
    real_time_factor: float
    text: str
    battery_drop_pct_per_hour: float | None = None
    notes: list[str] = field(default_factory=list)


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(np.array(values), pct)) if values else 0.0


def system_info() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "python_arch": platform.architecture()[0],
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def battery_percent() -> tuple[float, bool] | None:
    try:
        import psutil
    except ImportError:
        return None
    battery = psutil.sensors_battery()
    if battery is None:
        return None
    return float(battery.percent), bool(battery.power_plugged)


def benchmark_device(
    model_dir: Path,
    device: str,
    audio: np.ndarray,
    runs: int = 5,
    warmup: int = 1,
    language: str = "auto",
    power_minutes: float = 0.0,
) -> DeviceResult:
    engine = WhisperONNX(model_dir, device)
    for _ in range(warmup):
        engine.transcribe(audio, language)

    totals, encoders, per_token, steps = [], [], [], []
    text = ""
    for _ in range(runs):
        result = engine.transcribe(audio, language)
        totals.append(result.timings.total_ms)
        encoders.append(result.timings.encoder_ms)
        per_token.append(result.timings.ms_per_token)
        steps.append(result.timings.decoder_steps)
        text = result.text

    audio_s = len(audio) / SAMPLE_RATE
    mean_total = statistics.fmean(totals)
    decoder_s = sum(p * s for p, s in zip(per_token, steps)) / 1000
    out = DeviceResult(
        device=device,
        model=model_dir.parent.name,
        load_s=round(engine.load_seconds, 3),
        runs=runs,
        audio_s=round(audio_s, 2),
        encoder_ms=round(statistics.fmean(encoders), 2),
        ms_per_token=round(statistics.fmean(per_token), 2),
        tokens_per_s=round(sum(steps) / decoder_s, 1) if decoder_s else 0.0,
        total_ms_mean=round(mean_total, 1),
        total_ms_p50=round(_percentile(totals, 50), 1),
        total_ms_p90=round(_percentile(totals, 90), 1),
        real_time_factor=round((mean_total / 1000) / audio_s, 4) if audio_s else 0.0,
        text=text,
    )

    if power_minutes > 0:
        start = battery_percent()
        if start is None:
            out.notes.append("No battery reading available on this machine.")
        elif start[1]:
            out.notes.append("Laptop is plugged in; unplug it for a battery measurement.")
        else:
            t_end = time.time() + power_minutes * 60
            t0 = time.time()
            while time.time() < t_end:
                engine.transcribe(audio, language)
            end = battery_percent()
            hours = (time.time() - t0) / 3600
            if end is not None and hours > 0:
                out.battery_drop_pct_per_hour = round((start[0] - end[0]) / hours, 2)
                out.notes.append(
                    f"Continuous transcription for {power_minutes:g} min, battery {start[0]:.0f}% -> {end[0]:.0f}%."
                )
    return out


def to_markdown(results: list[DeviceResult], info: dict) -> str:
    lines = [
        f"Measured on {info['processor'] or info['machine']} ({info['platform']}), "
        f"ONNX Runtime {info['onnxruntime']}, {info['timestamp']}.",
        "",
        "| Metric | " + " | ".join(r.device.upper() for r in results) + " |",
        "| --- | " + " | ".join("---" for _ in results) + " |",
    ]
    rows = [
        ("Model load (s)", "load_s"),
        ("Encoder latency (ms)", "encoder_ms"),
        ("Decoder latency per token (ms)", "ms_per_token"),
        ("Decoder speed (tokens/s)", "tokens_per_s"),
        ("End-to-end, mean (ms)", "total_ms_mean"),
        ("End-to-end, p90 (ms)", "total_ms_p90"),
        ("Real-time factor (lower is better)", "real_time_factor"),
        ("Battery drain (% per hour)", "battery_drop_pct_per_hour"),
    ]
    for label, key in rows:
        values = [getattr(r, key) for r in results]
        if all(v is None for v in values):
            continue
        lines.append(f"| {label} | " + " | ".join("n/a" if v is None else str(v) for v in values) + " |")

    by_dev = {r.device: r for r in results}
    if "npu" in by_dev and "cpu" in by_dev and by_dev["npu"].total_ms_mean:
        speedup = by_dev["cpu"].total_ms_mean / by_dev["npu"].total_ms_mean
        lines += ["", f"**NPU speed-up over CPU: {speedup:.1f}x end to end.**"]
    lines += ["", f"Audio clip: {results[0].audio_s} s, {results[0].runs} timed runs after warm-up."]
    return "\n".join(lines) + "\n"


def save_results(results: list[DeviceResult], out_dir: Path) -> tuple[Path, Path]:
    info = system_info()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    md_path = out_dir / "results.md"
    json_path.write_text(json.dumps({"system": info, "results": [asdict(r) for r in results]}, indent=2, ensure_ascii=False))
    md_path.write_text(to_markdown(results, info), encoding="utf-8")
    return json_path, md_path
