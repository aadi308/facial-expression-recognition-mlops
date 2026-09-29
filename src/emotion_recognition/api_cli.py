"""Command-line launcher for the FastAPI application."""

import argparse
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the facial expression recognition HTTP API."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    import uvicorn

    uvicorn.run(
        "emotion_recognition.api:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
