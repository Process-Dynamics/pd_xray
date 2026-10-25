"""Command line: `pd-xray serve`."""

import argparse
import logging
import threading
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from pd_xray.server.app import create_app
from pd_xray.server.settings import Settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pd-xray", description="pd_xray web application.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the web server.")
    serve.add_argument("--host", default="127.0.0.1", help="Address; 0.0.0.0 for all (default: %(default)s).")
    serve.add_argument("--port", type=int, default=8000, help="Port (default: %(default)s).")
    serve.add_argument(
        "--root",
        action="append",
        type=Path,
        help="Folder users may browse and open. Repeat for several. Default: the current folder.",
    )
    serve.add_argument("--cache-gb", type=float, default=2.0, help="Memory for cached frames (default: %(default)s).")
    serve.add_argument("--workers", type=int, default=2, help="Background jobs run at once (default: %(default)s).")
    serve.add_argument("--no-browser", action="store_true", help="Don't open a web browser.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the `pd-xray` command."""
    import uvicorn

    args = _parser().parse_args(argv)
    settings = Settings(
        roots=tuple(args.root or [Path.cwd()]), cache_bytes=int(args.cache_gb * 1024**3), job_workers=args.workers
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    url = f"http://{'localhost' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/"
    logging.getLogger(__name__).info("Serving %s at %s", ", ".join(map(str, settings.roots)), url)
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, (url,)).start()
    uvicorn.run(create_app(settings), host=args.host, port=args.port)
