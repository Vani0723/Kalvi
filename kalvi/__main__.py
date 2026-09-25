"""Start Kalvi: ``python -m kalvi`` then open http://127.0.0.1:8765."""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser

import uvicorn

from .config import settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="kalvi", description="Offline AI lecture companion")
    parser.add_argument("--device", choices=["auto", "npu", "cpu", "mock"], help="Where to run the models")
    parser.add_argument("--language", help="Whisper language code (en, ta, hi, ...) or auto")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.device:
        settings.device = args.device
    if args.language:
        settings.language = args.language
    if args.port:
        settings.port = args.port

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .server import create_app

    url = f"http://{settings.host}:{settings.port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Kalvi is starting at {url}")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
