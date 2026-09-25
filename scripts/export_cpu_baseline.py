"""Export the CPU baseline: the same AI Hub Whisper model as float ONNX.

The NPU assets from AI Hub are compiled QNN contexts and cannot run on the
CPU. For a fair NPU vs CPU benchmark, this script exports Qualcomm's own
PyTorch Whisper wrapper (identical graph and inputs/outputs) to plain ONNX.

Run it in an x64 Python (Qualcomm's `qai_hub_models` package does not install
on Windows ARM64 Python), then copy the folder to the Snapdragon PC if needed:

    pip install -r requirements-export.txt
    python scripts/export_cpu_baseline.py --model whisper_base

Result:
    models/<model>/cpu/WhisperEncoder.onnx
    models/<model>/cpu/WhisperDecoder.onnx
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def export_component(component, out_path: Path) -> None:
    import torch

    spec = component.get_input_spec()
    inputs = component.sample_inputs(spec)
    tensors = tuple(torch.tensor(v[0]) for v in inputs.values())
    output_names = list(component.get_output_spec().keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        component,
        tensors,
        str(out_path),
        input_names=list(spec.keys()),
        output_names=output_names,
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="whisper_base")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()

    module = importlib.import_module(f"qai_hub_models.models.{args.model}")
    whisper = module.Model.from_pretrained()
    out_dir = args.models_dir / args.model / "cpu"
    export_component(whisper.encoder, out_dir / "WhisperEncoder.onnx")
    export_component(whisper.decoder, out_dir / "WhisperDecoder.onnx")

    tokenizer = args.models_dir / args.model / "tokenizer.json"
    if not tokenizer.exists():
        from transformers import WhisperTokenizerFast

        WhisperTokenizerFast.from_pretrained(whisper.hf_source).backend_tokenizer.save(str(tokenizer))
        print(f"Saved {tokenizer}")


if __name__ == "__main__":
    main()
