"""Download Qualcomm AI Hub models compiled for the Snapdragon X NPU.

Uses Qualcomm's lightweight `qai_hub_models_cli` (pure Python, works on
Windows ARM64) to fetch the pre-compiled QNN ONNX assets, and saves the
matching Whisper tokenizer next to them so Kalvi runs fully offline.

    pip install qai_hub_models_cli huggingface_hub
    python scripts/fetch_models.py                      # Whisper Base, X Elite
    python scripts/fetch_models.py --model whisper_large_v3_turbo
    python scripts/fetch_models.py --chipset qualcomm-snapdragon-x2-elite

Result:
    models/<model>/npu/...onnx (+ .bin context files)
    models/<model>/tokenizer.json
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# AI Hub model id -> Hugging Face repo holding the matching tokenizer.
HF_TOKENIZERS = {
    "whisper_tiny": "openai/whisper-tiny",
    "whisper_base": "openai/whisper-base",
    "whisper_small": "openai/whisper-small",
    "whisper_large_v3_turbo": "openai/whisper-large-v3-turbo",
}


def find_cli() -> list[str]:
    exe = shutil.which("qai-hub-models")
    if exe:
        return [exe]
    return [sys.executable, "-m", "qai_hub_models_cli.cli"]


def fetch_npu_assets(model: str, chipset: str, out_dir: Path, info_only: bool) -> None:
    cmd = [*find_cli(), "fetch", model, "--runtime", "precompiled_qnn_onnx", "--precision", "float", "--chipset", chipset]
    if info_only:
        cmd.append("--info")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd += ["--output-dir", str(out_dir)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def fetch_tokenizer(model: str, model_root: Path) -> None:
    from huggingface_hub import hf_hub_download

    repo = HF_TOKENIZERS.get(model)
    if repo is None:
        raise SystemExit(f"No tokenizer mapping for {model}; add it to HF_TOKENIZERS.")
    path = hf_hub_download(repo, "tokenizer.json")
    model_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, model_root / "tokenizer.json")
    print(f"Saved tokenizer from {repo} to {model_root / 'tokenizer.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="whisper_base", choices=sorted(HF_TOKENIZERS))
    parser.add_argument("--chipset", default="qualcomm-snapdragon-x-elite")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--info", action="store_true", help="Only list the available assets.")
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    model_root = args.models_dir / args.model
    fetch_npu_assets(args.model, args.chipset, model_root / "npu", args.info)
    if not args.info and not args.skip_tokenizer:
        fetch_tokenizer(args.model, model_root)
    if not args.info:
        onnx = sorted((model_root / "npu").rglob("*.onnx"))
        print("\nNPU model files:")
        for p in onnx:
            print("  ", p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
        if not onnx:
            print("   none found; run with --info to check which assets exist for this chipset.")


if __name__ == "__main__":
    main()
