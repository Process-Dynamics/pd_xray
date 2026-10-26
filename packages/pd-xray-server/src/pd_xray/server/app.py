"""FastAPI application: file browsing, datasets, frame tiles, background jobs and a WebSocket for job events.

Endpoints that touch files or pixels are plain `def` functions, which FastAPI runs in a thread pool, so slow reads never
block the event loop. Long work runs as jobs (see `jobs.py`).
"""

import asyncio
import contextlib
import json
import logging
import re
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pd_xray.core import CancelToken, OperationCancelledError, PdXrayError, ProgressCallback
from pd_xray.io import compute_value_range, supported_extensions
from pd_xray.io.hdf5 import inspect_hdf5
from pd_xray.processing import OPERATIONS, Pipeline, tone_map
from pd_xray.server.colormaps import COLORMAPS
from pd_xray.server.connections import BROWSER_COOKIE, ConnectionStore, list_remote, new_browser_id
from pd_xray.server.datasets import DatasetStore, NotFoundError, jsonable
from pd_xray.server.jobs import JobManager
from pd_xray.server.paths import AccessDeniedError, list_directory, resolve_path
from pd_xray.server.render import MEDIA_TYPES, ImageFormat, parse_tone
from pd_xray.server.runs import PIPELINE_EXTENSIONS, check_run, run_pipeline, save_pipeline
from pd_xray.server.settings import Settings
from pd_xray.storage import HostKeyError, LocalStorage, Storage, copy, join_path, normalise_path

logger = logging.getLogger(__name__)

_HDF5_EXTENSIONS = (".h5", ".hdf5", ".hdf", ".nxs")
STATIC_DIR = Path(__file__).parent / "static"
"""Built frontend (`npm run build` in `frontend/` writes here)."""


class OpenRequest(BaseModel):
    """Body of POST /api/datasets. `dataset` is for HDF5 files, `pattern` for folders (file sequences). With
    `connection`, `path` is on that connection (from its root) and is read lazily from there."""

    path: str
    connection: str | None = None
    dataset: str | None = None
    pattern: str | None = None
    axes: list[str] | None = None


class PreviewRequest(BaseModel):
    """Body of POST …/preview. `steps` as in a pipeline file; `client` identifies one browser tab, so a new preview
    from the same tab cancels its previous one."""

    steps: list[dict[str, Any]]
    client: str


class ConnectionRequest(BaseModel):
    """Body of POST /api/connections. `settings` (host, bucket, …) are returned by GET; `credentials` never are."""

    kind: str
    name: str = ""
    settings: dict[str, Any]
    credentials: dict[str, str]


class Location(BaseModel):
    """A file or folder on a connection, or on the server (`connection` None: an absolute path inside the roots)."""

    connection: str | None = None
    path: str


class CopyRequest(BaseModel):
    """Body of POST /api/jobs/copy: copy the file or folder `source` into the folder `destination`, keeping its name."""

    source: Location
    destination: Location
    overwrite: bool = False
    verify: bool = False


class SavePipelineRequest(BaseModel):
    """Body of POST /api/pipelines. With `dataset_id`, references inside the dataset's folder are saved relative."""

    path: str
    steps: list[dict[str, Any]]
    dataset_id: str | None = None
    overwrite: bool = False


class RunRequest(BaseModel):
    """Body of POST /api/jobs/run. `frames` is (start, stop, step) along the first axis, as a Python slice."""

    dataset_id: str
    steps: list[dict[str, Any]]
    destination: str
    dtype: str = "float32"
    value_range: tuple[float, float] | None = None
    frames: tuple[int | None, int | None, int | None] | None = None
    overwrite: bool = False


class ValueRangeRequest(BaseModel):
    """Body of POST /api/jobs/value-range."""

    dataset_id: str


def create_app(settings: Settings) -> FastAPI:
    """Build the app. All state (open datasets, cache, jobs) lives in the returned app, not in globals."""
    store = DatasetStore(settings.cache_bytes, settings.tile_size, settings.roots)
    jobs = JobManager(settings.job_workers)
    connections = ConnectionStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        jobs.shutdown()
        store.close_all()
        connections.close_all()

    app = FastAPI(title="pd_xray", lifespan=lifespan)
    app.state.settings, app.state.store, app.state.jobs = settings, store, jobs
    app.state.connections = connections

    @app.middleware("http")
    async def browser_id(request: Request, call_next: Any) -> Response:
        """Gives each browser a random id cookie, which keeps its connections apart from other browsers'."""
        browser = request.cookies.get(BROWSER_COOKIE)
        new = not browser
        request.state.browser = new_browser_id() if new else browser
        response: Response = await call_next(request)
        if new:
            response.set_cookie(
                BROWSER_COOKIE, request.state.browser, max_age=365 * 86400, httponly=True, samesite="strict"
            )
        return response

    def error(status: int) -> Any:
        async def handler(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse({"detail": str(exc), "error": type(exc).__name__}, status_code=status)

        return handler

    app.add_exception_handler(PdXrayError, error(400))
    app.add_exception_handler(PermissionError, error(403))
    app.add_exception_handler(IsADirectoryError, error(400))

    @app.exception_handler(HostKeyError)
    async def host_key_error(request: Request, exc: HostKeyError) -> JSONResponse:
        detail = {"detail": str(exc), "error": "HostKeyError", "fingerprint": exc.fingerprint}
        return JSONResponse(detail, status_code=400)

    app.add_exception_handler(AccessDeniedError, error(403))
    app.add_exception_handler(NotFoundError, error(404))
    app.add_exception_handler(FileNotFoundError, error(404))
    app.add_exception_handler(NotADirectoryError, error(400))
    app.add_exception_handler(OperationCancelledError, error(409))
    app.add_exception_handler(FileExistsError, error(409))
    previews: dict[str, CancelToken] = {}  # running preview per browser tab
    previews_lock = threading.Lock()

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        """Server configuration the UI needs."""
        return {
            "roots": [str(root) for root in settings.roots],
            "tile_size": settings.tile_size,
            "formats": list(MEDIA_TYPES),
            "colormaps": list(COLORMAPS),
            "extensions": supported_extensions(),
        }

    @app.get("/api/files")
    def files(path: str | None = None, show_hidden: bool = False) -> dict[str, Any]:
        """Folder listing. Without `path`, lists the root folders."""
        if path is None:
            entries = [{"name": str(root), "path": str(root), "kind": "dir", "size": None} for root in settings.roots]
            return {"path": None, "parent": None, "entries": entries}
        return list_directory(resolve_path(path, settings.roots), settings.roots, show_hidden)

    @app.get("/api/connections")
    def list_connections(request: Request) -> list[dict[str, Any]]:
        """This browser's remote connections."""
        return [c.info() for c in connections.list(request.state.browser)]

    @app.post("/api/connections")
    def add_connection(request: Request, body: ConnectionRequest) -> dict[str, Any]:
        """Test and open a connection (`kind` "sftp" or "s3"). Fails with the reason; an unknown SSH host key gives
        error "HostKeyError" with its `fingerprint`, to be sent back as settings.host_key once the user trusts it."""
        connection = connections.add(request.state.browser, body.kind, body.name, body.settings, body.credentials)
        logger.info("Browser connected to %s", connection.storage.url)
        return connection.info()

    @app.delete("/api/connections/{connection_id}", status_code=204)
    def remove_connection(request: Request, connection_id: str) -> None:
        """Disconnect. Datasets opened from it are closed."""
        connections.get(request.state.browser, connection_id)  # 404 for another browser's connection
        for entry in store.list():
            if entry.options.get("connection") == connection_id:
                store.close(entry.id)
        connections.remove(request.state.browser, connection_id)

    @app.get("/api/connections/{connection_id}/files")
    def remote_files(request: Request, connection_id: str, path: str = "", show_hidden: bool = False) -> dict[str, Any]:
        """Folder listing on a connection. `path` is relative to the connection's root ("" for the root)."""
        connection = connections.get(request.state.browser, connection_id)
        return list_remote(connection, normalise_path(path), show_hidden)

    @app.get("/api/inspect")
    def inspect(request: Request, path: str, connection: str | None = None) -> dict[str, Any]:
        """What a file contains before opening it. HDF5: every dataset, with `is_image` marking candidates. With
        `connection`, `path` is on that connection."""
        storage = None if connection is None else connections.get(request.state.browser, connection).storage
        if storage is None:
            resolved = resolve_path(path, settings.roots)
            if not resolved.is_file():
                raise FileNotFoundError(f"No such file: {resolved}")
        else:
            resolved = Path("/" + normalise_path(path))
            if storage.stat(str(resolved)).is_dir:
                raise IsADirectoryError(f"Is a folder: {resolved}")
        if resolved.suffix.lower() not in _HDF5_EXTENSIONS:
            return {"path": str(resolved), "format": resolved.suffix.lower().lstrip("."), "datasets": []}
        datasets = [
            {
                "name": d.name,
                "shape": list(d.shape),
                "dtype": d.dtype,
                "chunks": d.chunks,
                "compression": d.compression,
                "is_image": d.is_image,
                "attrs": jsonable(d.attrs),
                "warnings": list(d.warnings),
            }
            for d in inspect_hdf5(resolved, storage)
        ]
        return {"path": str(resolved), "format": "hdf5", "datasets": datasets}

    @app.post("/api/datasets")
    def open_dataset(http: Request, request: OpenRequest) -> dict[str, Any]:
        """Open a file, or a folder as a sequence of single-frame files, on the server or on a connection."""
        if request.connection is None:
            path = resolve_path(request.path, settings.roots)
            return store.info(store.open(path, request.dataset, request.pattern, request.axes))
        storage = connections.get(http.state.browser, request.connection).storage
        path = Path("/" + normalise_path(request.path))
        entry = store.open(path, request.dataset, request.pattern, request.axes, storage, request.connection)
        return store.info(entry)

    @app.get("/api/datasets")
    def list_datasets() -> list[dict[str, Any]]:
        return [store.info(entry) for entry in store.list()]

    @app.get("/api/datasets/{dataset_id}")
    def get_dataset(dataset_id: str) -> dict[str, Any]:
        return store.info(store.get(dataset_id))

    @app.delete("/api/datasets/{dataset_id}", status_code=204)
    def close_dataset(dataset_id: str) -> None:
        store.close(dataset_id)

    @app.get("/api/datasets/{dataset_id}/frames/{frame}/stats")
    def frame_stats(dataset_id: str, frame: str, pipeline: str | None = None) -> dict[str, float]:
        """min/max of the frame and automatic contrast limits low/high. `frame` is e.g. "12", "3,40", or "-".

        `pipeline` (from …/preview) gives the statistics of the processed frame.
        """
        entry = store.get(dataset_id)
        return store.stats(entry, store.parse_frame(entry, frame), pipeline)

    @app.get("/api/datasets/{dataset_id}/frames/{frame}/histogram")
    def frame_histogram(
        dataset_id: str,
        frame: str,
        bins: int = 256,
        low: float | None = None,
        high: float | None = None,
        pipeline: str | None = None,
    ) -> dict[str, Any]:
        """Pixel counts of the full frame in `bins` equal bins over `low`..`high` (default: the frame's min..max).

        `pipeline` (from …/preview) gives the histogram of the processed frame.
        """
        entry = store.get(dataset_id)
        return store.histogram(entry, store.parse_frame(entry, frame), bins, low, high, pipeline)

    @app.get("/api/tone-curve")
    def tone_curve(tone: str | None = None, points: int = 256) -> dict[str, list[float]]:
        """`tone` settings (JSON, as on tiles) sampled at `points` inputs evenly over 0..1, for drawing.

        Returns {"curve": the tone curve alone, "response": every adjustment together}.
        """
        if not 2 <= points <= 4096:
            raise PdXrayError(f"Points must be 2 to 4096, got {points}.")
        settings = json.loads(parse_tone(tone) or "{}")
        x = np.linspace(0, 1, points, dtype=np.float32)
        return {
            "curve": tone_map(x, curve=settings.get("curve")).tolist(),
            "response": tone_map(x, **settings).tolist(),
        }

    @app.get("/api/operations")
    def operations() -> list[dict[str, Any]]:
        """The processing operations and their parameters, for building the processing panel."""
        return [
            {
                "name": op.name,
                "title": op.title,
                "description": op.description,
                "params": [jsonable(vars(param)) for param in op.params],
            }
            for op in OPERATIONS.values()
        ]

    @app.post("/api/datasets/{dataset_id}/frames/{frame}/preview")
    def preview(dataset_id: str, frame: str, request: PreviewRequest) -> dict[str, Any]:
        """Process one frame with the given steps (full resolution, exact) and describe the result for tile requests.

        Slow steps can take seconds. A newer preview from the same `client` cancels this one between steps (409).
        """
        entry = store.get(dataset_id)
        index = store.parse_frame(entry, frame)
        token = CancelToken()
        with previews_lock:
            previous = previews.get(request.client)
            previews[request.client] = token
        if previous is not None:
            previous.cancel()
        try:
            return store.preview(entry, index, request.steps, token)
        finally:
            with previews_lock:
                if previews.get(request.client) is token:
                    del previews[request.client]

    @app.get("/api/datasets/{dataset_id}/frames/{frame}/tiles/{level}/{column}_{row}.{fmt}")
    def tile(
        dataset_id: str,
        frame: str,
        level: int,
        column: int,
        row: int,
        fmt: ImageFormat,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "grey",
        pipeline: str | None = None,
        tone: str | None = None,
    ) -> Response:
        """8-bit tile of a frame. Contrast is `vmin`..`vmax` (default: the frame's automatic limits), colour map `cmap`.

        Level 0 is full resolution; level n is shrunk by 2**n (block mean). See /api/datasets/{id} for `levels`.
        `pipeline` (from …/preview) shows the processed frame; its size and levels come from the preview. `tone` is
        JSON of display tone adjustments (exposure, contrast, highlights, shadows, whites, blacks, curve), applied
        between vmin and vmax.
        """
        entry = store.get(dataset_id)
        index = store.parse_frame(entry, frame)
        content = store.tile(entry, index, level, column, row, fmt, vmin, vmax, cmap, pipeline, parse_tone(tone))
        # Ids are new on every open, so a tile URL never changes meaning while the dataset is open.
        return Response(content, media_type=MEDIA_TYPES[fmt], headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/pipelines")
    def load_pipeline(path: str) -> dict[str, Any]:
        """Steps of a pipeline file (.yaml/.yml), as saved: references may be relative to the dataset's folder."""
        resolved = resolve_path(path, settings.roots)
        if resolved.suffix.lower() not in PIPELINE_EXTENSIONS:
            raise PdXrayError(f"Pipeline files end in .yaml or .yml, got '{resolved.name}'.")
        if not resolved.is_file():
            raise FileNotFoundError(f"No such file: {resolved}")
        return {"path": str(resolved), "steps": Pipeline.load(resolved).to_dict()["steps"]}

    @app.post("/api/pipelines")
    def save_pipeline_file(request: SavePipelineRequest) -> dict[str, Any]:
        """Save steps as a pipeline file. Existing files are kept (409) unless `overwrite`."""
        resolved = resolve_path(request.path, settings.roots)
        if not resolved.parent.is_dir():
            raise NotADirectoryError(f"Not a folder: {resolved.parent}")
        base = store.get(request.dataset_id).folder if request.dataset_id else None
        return {"path": str(save_pipeline(resolved, request.steps, base, request.overwrite))}

    @app.post("/api/jobs/run")
    def run_job(request: RunRequest) -> dict[str, Any]:
        """Job: apply a pipeline to a whole dataset (or `frames`) and write it to `destination`, a file in the allowed
        folders (format from its extension), plus `<destination>.pipeline.yaml`. See `runs.py`."""
        entry = store.get(request.dataset_id)
        Pipeline.from_dict({"pd_xray_pipeline": 1, "steps": request.steps})  # invalid steps fail now, not in the job
        destination = resolve_path(request.destination, settings.roots)
        if not destination.parent.is_dir():
            raise NotADirectoryError(f"Not a folder: {destination.parent}")
        check_run(entry, destination, request.dtype, request.overwrite)
        frames = None if request.frames is None else slice(*request.frames)
        if frames is not None and frames.step == 0:
            raise PdXrayError("The frame step can't be 0.")
        value_range = request.value_range

        def work(progress: ProgressCallback, cancel: CancelToken) -> dict[str, Any]:
            return run_pipeline(
                store, entry, request.steps, destination, request.dtype, value_range, frames, progress, cancel
            )

        return jobs.submit(f"Process {entry.path.name} into {destination.name}", work)

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return jobs.list()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return jobs.get(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        return jobs.cancel(job_id)

    @app.post("/api/jobs/value-range")
    def value_range_job(request: ValueRangeRequest) -> dict[str, Any]:
        """Job: min and max over every frame of a dataset (for whole-dataset display contrast). Reads all data."""
        entry = store.get(request.dataset_id)

        def work(progress: ProgressCallback, cancel: CancelToken) -> dict[str, float]:
            low, high = compute_value_range(entry.dataset, progress=progress, cancel=cancel)
            return {"min": low, "max": high}

        return jobs.submit(f"Value range of {entry.path.name}", work)

    def locate(browser: str, location: Location) -> tuple[Storage, str, str, str]:
        """(storage, path in it, name, label for messages) of a location. Local paths are inside the roots."""
        if location.connection is None:
            path = resolve_path(location.path, settings.roots)
            root = next(r for r in settings.roots if path.is_relative_to(r))
            return LocalStorage(root), normalise_path(path.relative_to(root).as_posix()), path.name, str(path)
        connection = connections.get(browser, location.connection)
        remote = normalise_path(location.path)
        name = remote.rsplit("/", 1)[-1] or re.sub(r"[^\w.-]+", "_", connection.name).strip("_")  # root: its name
        return connection.storage, remote, name, f"{connection.name}: /{remote}"

    @app.post("/api/jobs/copy")
    def copy_job(request: Request, body: CopyRequest) -> dict[str, Any]:
        """Job: copy a file or folder into a folder, between any two places: server folders and connections (see
        `pd_xray.storage.copy`). Identical files are skipped and interrupted copies resume; differing files fail the
        job unless `overwrite`. Remote-to-remote copies stream through the server."""
        source, source_path, name, _ = locate(request.state.browser, body.source)
        destination, folder, _, label = locate(request.state.browser, body.destination)
        if not destination.stat(folder).is_dir:
            raise NotADirectoryError(f"Not a folder: {label}")
        target = join_path(folder, name)

        def work(progress: ProgressCallback, cancel: CancelToken) -> dict[str, Any]:
            result = copy(
                source,
                source_path,
                destination,
                target,
                overwrite=body.overwrite,
                verify=body.verify,
                progress=progress,
                cancel=cancel,
            )
            return {**vars(result), "destination": f"{label.rstrip('/')}/{name}"}

        return jobs.submit(f"Copy {name} to {label}", work, unit="bytes")

    @app.websocket("/api/events")
    async def events(websocket: WebSocket) -> None:
        """Pushes `{"type": "job", "job": snapshot}` for every job on connect, then on each update.

        Updates that arrive faster than they can be sent are merged, so only the latest snapshot per job is sent.
        """
        await websocket.accept()
        loop = asyncio.get_running_loop()
        pending: dict[str, dict[str, Any]] = {}
        wake = asyncio.Event()

        def add(snapshot: dict[str, Any]) -> None:
            pending[snapshot["id"]] = snapshot
            wake.set()

        def listener(snapshot: dict[str, Any]) -> None:  # called from worker threads
            with contextlib.suppress(RuntimeError):  # loop closed during shutdown
                loop.call_soon_threadsafe(add, snapshot)

        unsubscribe = jobs.subscribe(listener)
        for snapshot in jobs.list():
            add(snapshot)

        async def send() -> None:
            while True:
                await wake.wait()
                wake.clear()
                batch = list(pending.values())
                pending.clear()
                for snapshot in batch:
                    await websocket.send_json({"type": "job", "job": snapshot})

        sender = asyncio.create_task(send())
        try:
            while True:  # client messages are ignored; this loop ends when the client disconnects
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            unsubscribe()
            sender.cancel()

    if (STATIC_DIR / "index.html").is_file():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")  # last: API routes match first
    else:

        @app.get("/", include_in_schema=False)
        def index() -> RedirectResponse:
            logger.warning("Frontend not built (no %s); showing the API docs instead.", STATIC_DIR / "index.html")
            return RedirectResponse("/docs")

    return app
