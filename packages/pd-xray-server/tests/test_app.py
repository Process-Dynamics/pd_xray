import io
import json
import threading
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pd_xray.core import CancelToken, OperationCancelledError, ProgressCallback, track
from pd_xray.server import Settings, create_app
from pd_xray.server import app as app_module
from pd_xray.server.cache import LRUCache
from pd_xray.server.colormaps import COLORMAPS

DATA = (np.arange(3 * 600 * 700) % 50_000).astype(np.uint16).reshape(3, 600, 700)
FINISHED = ("done", "failed", "cancelled")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    with h5py.File(tmp_path / "scan.h5", "w") as f:
        f["images"] = DATA
        f["dark"] = DATA[0]
    (tmp_path / "sub").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    return tmp_path


@pytest.fixture
def client(root: Path) -> Any:
    with TestClient(create_app(Settings(roots=(root,), tile_size=256))) as client:
        yield client


def test_browse_and_path_checks(client: TestClient, root: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    listing = client.get("/api/files", params={"path": str(root)}).json()
    assert [(e["name"], e["openable"]) for e in listing["entries"]] == [
        ("sub", False),
        ("notes.txt", False),
        ("scan.h5", True),
    ]
    assert listing["parent"] is None
    assert client.get("/api/files", params={"path": str(root / "sub")}).json()["parent"] == str(root)
    assert client.get("/api/files", params={"path": str(root / "sub" / ".." / "..")}).status_code == 403
    outside = tmp_path_factory.mktemp("outside")
    (root / "link").symlink_to(outside)
    assert client.get("/api/files", params={"path": str(root / "link")}).status_code == 403
    assert client.get("/api/files", params={"path": "relative/path"}).status_code == 403


def test_open_and_tiles(client: TestClient, root: Path) -> None:
    inspected = client.get("/api/inspect", params={"path": str(root / "scan.h5")}).json()
    assert {d["name"] for d in inspected["datasets"]} == {"images", "dark"}
    response = client.post("/api/datasets", json={"path": str(root / "scan.h5")})
    assert response.status_code == 400 and "choose one" in response.json()["detail"]

    info = client.post("/api/datasets", json={"path": str(root / "scan.h5"), "dataset": "images"}).json()
    assert info["frame_shape"] == [3] and (info["height"], info["width"], info["levels"]) == (600, 700, 3)
    base = f"/api/datasets/{info['id']}/frames"

    tile = client.get(f"{base}/1/tiles/0/1_0.png", params={"vmin": 20_000, "vmax": 20_255})
    assert tile.status_code == 200 and tile.headers["content-type"] == "image/png"
    pixels = np.asarray(Image.open(io.BytesIO(tile.content)))
    expected = np.clip(DATA[1, :256, 256:512].astype(int) - 20_000, 0, 255)
    assert 0 < (expected == 0).sum() < expected.size  # tile spans the contrast window
    np.testing.assert_array_equal(pixels, expected)

    top = np.asarray(Image.open(io.BytesIO(client.get(f"{base}/2/tiles/2/0_0.jpeg").content)))
    assert top.shape == (150, 175)  # ceil(600 / 4), ceil(700 / 4): whole frame in one tile
    stats = client.get(f"{base}/2/stats").json()
    assert (stats["min"], stats["max"]) == (DATA[2].min(), DATA[2].max())

    assert client.get(f"{base}/3/stats").status_code == 400
    assert client.get(f"{base}/0,0/stats").status_code == 400
    assert client.get(f"{base}/0/tiles/0/3_0.png").status_code == 404
    assert client.get(f"{base}/0/tiles/3/0_0.png").status_code == 404
    assert client.delete(f"/api/datasets/{info['id']}").status_code == 204
    assert client.get(f"{base}/0/stats").status_code == 404


def test_value_range_job_reports_progress(client: TestClient, root: Path) -> None:
    info = client.post("/api/datasets", json={"path": str(root / "scan.h5"), "dataset": "images"}).json()
    with client.websocket_connect("/api/events") as ws:
        job = client.post("/api/jobs/value-range", json={"dataset_id": info["id"]}).json()
        states = []
        while not states or states[-1]["state"] not in FINISHED:
            states.append(ws.receive_json()["job"])
    assert states[-1]["state"] == "done" and states[-1]["id"] == job["id"]
    assert states[-1]["result"] == {"min": 0, "max": 49_999}
    assert states[-1]["current"] == states[-1]["total"] == 3


def test_job_cancels_without_blocking_requests(client: TestClient) -> None:
    started = threading.Event()

    def slow(progress: ProgressCallback, cancel: CancelToken) -> None:
        for i in track(range(1000), progress=progress, cancel=cancel):
            if i == 2:
                started.set()
            time.sleep(0.01)

    jobs = client.app.state.jobs  # type: ignore[attr-defined]
    with client.websocket_connect("/api/events") as ws:
        job = jobs.submit("slow", slow)
        assert started.wait(5)
        t = time.perf_counter()
        assert client.get("/api/info").status_code == 200
        assert time.perf_counter() - t < 1
        assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
        last = ws.receive_json()["job"]
        while last["state"] not in FINISHED:
            last = ws.receive_json()["job"]
    assert last["state"] == "cancelled" and 0 < last["current"] < 1000
    assert client.post("/api/jobs/nope/cancel").status_code == 404


def test_cache_computes_once_under_concurrency() -> None:
    cache = LRUCache(10_000)
    calls = []

    def compute() -> bytes:
        calls.append(1)
        time.sleep(0.05)
        return b"tile"

    threads = [threading.Thread(target=cache.get_or_compute, args=("k", compute)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 1


def test_colour_maps_and_sequences(client: TestClient, root: Path) -> None:
    assert "viridis" in client.get("/api/info").json()["colormaps"]
    info = client.post("/api/datasets", json={"path": str(root / "scan.h5"), "dataset": "dark"}).json()
    base = f"/api/datasets/{info['id']}/frames/-/tiles/0/0_0.png"
    grey = np.asarray(Image.open(io.BytesIO(client.get(base, params={"vmin": 0, "vmax": 255}).content)))
    rgb = np.asarray(
        Image.open(io.BytesIO(client.get(base, params={"vmin": 0, "vmax": 255, "cmap": "viridis"}).content))
    )
    assert grey.ndim == 2 and rgb.shape == grey.shape + (3,)
    np.testing.assert_array_equal(rgb, COLORMAPS["viridis"][grey])
    assert client.get(base, params={"cmap": "rainbow"}).status_code == 400

    for i in (1, 2, 10):
        Image.fromarray(DATA[0, :8, :8]).save(root / "sub" / f"scan_{i}.tif")
    Image.fromarray(DATA[0, :8, :8]).save(root / "sub" / "other.tif")
    listing = client.get("/api/files", params={"path": str(root / "sub")}).json()
    assert listing["sequences"] == [{"pattern": "scan_*.tif", "count": 3, "first": "scan_1.tif"}]
    opened = client.post("/api/datasets", json={"path": str(root / "sub"), "pattern": "scan_*.tif"}).json()
    assert opened["frame_shape"] == [3]


def test_serves_built_frontend(root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "STATIC_DIR", tmp_path / "missing")  # frontend not built
    with TestClient(create_app(Settings(roots=(root,)))) as client:
        assert client.get("/", follow_redirects=False).headers["location"] == "/docs"
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html>pd_xray</html>")
    (static / "assets" / "app.js").write_text("console.log(1)")
    monkeypatch.setattr(app_module, "STATIC_DIR", static)
    with TestClient(create_app(Settings(roots=(root,)))) as client:
        assert client.get("/").text == "<html>pd_xray</html>"
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/api/info").status_code == 200  # API routes still win


def test_preview_with_folder_and_file_references(client: TestClient, root: Path) -> None:
    ops = {op["name"]: op for op in client.get("/api/operations").json()}
    assert ops["flat_field"]["params"][0]["kind"] == "reference" and ops["median"]["params"][0]["minimum"] == 1
    image = np.full((6, 8), 50, np.uint16)
    (root / "dark").mkdir()
    for i, value in enumerate((9, 11)):  # mean 10
        Image.fromarray(np.full((6, 8), value, np.uint16)).save(root / "dark" / f"d{i}.tif")
    for name, value in (("f0.tif", 25), ("f1.tif", 35)):  # mean 30
        Image.fromarray(np.full((6, 8), value, np.uint16)).save(root / "sub" / name)
    Image.fromarray(image).save(root / "sub" / "image.tif")
    info = client.post("/api/datasets", json={"path": str(root / "sub" / "image.tif")}).json()
    steps = [
        {"op": "flat_field", "params": {"dark": str(root / "dark"), "flat": ["f0.tif", "f1.tif"]}},  # relative
        {"op": "crop", "params": {"width": 5}},
    ]
    url = f"/api/datasets/{info['id']}/frames/-/preview"
    result = client.post(url, json={"steps": steps, "client": "tab"}).json()
    assert (result["height"], result["width"]) == (6, 5)  # crop changes the size
    assert result["stats"]["min"] == result["stats"]["max"] == 2.0  # (50 - 10) / (30 - 10)
    assert [t["op"] for t in result["timings"]] == ["flat_field", "crop"]
    tile = client.get(
        f"/api/datasets/{info['id']}/frames/-/tiles/0/0_0.png",
        params={"pipeline": result["pipeline"], "vmin": 0, "vmax": 2},
    )
    assert np.asarray(Image.open(io.BytesIO(tile.content))).shape == (6, 5)
    stats = client.get(f"/api/datasets/{info['id']}/frames/-/stats", params={"pipeline": result["pipeline"]}).json()
    assert stats["max"] == 2.0

    bad = client.post(url, json={"steps": [{"op": "median", "params": {"size": 0}}], "client": "tab"})
    assert bad.status_code == 400 and "Step 1 (median)" in bad.json()["detail"]
    outside = client.post(url, json={"steps": [{"op": "flat_field", "params": {"dark": "/"}}], "client": "tab"})
    assert outside.status_code == 403
    expired = client.get(f"/api/datasets/{info['id']}/frames/-/stats", params={"pipeline": "nope"})
    assert expired.status_code == 404


def test_newer_preview_from_the_same_tab_cancels_the_older(client: TestClient, root: Path) -> None:
    store = client.app.state.store  # type: ignore[attr-defined]
    info = client.post("/api/datasets", json={"path": str(root / "scan.h5"), "dataset": "images"}).json()
    entry = store.get(info["id"])
    stop = CancelToken()
    stop.cancel()
    with pytest.raises(OperationCancelledError):
        store.preview(entry, (0,), [{"op": "invert"}], stop)
    # Over HTTP: the endpoint maps cancellation to 409.
    started, release = threading.Event(), threading.Event()
    original = store.preview

    def slow_preview(*args: Any) -> Any:
        started.set()
        release.wait(5)
        args[3].raise_if_cancelled()
        return original(*args)

    store.preview = slow_preview
    url = f"/api/datasets/{info['id']}/frames/0/preview"
    body = {"steps": [{"op": "invert"}], "client": "tab"}
    first: list[int] = []
    thread = threading.Thread(target=lambda: first.append(client.post(url, json=body).status_code))
    thread.start()
    assert started.wait(5)
    store.preview = original
    second = client.post(url, json=body)  # cancels the first one's token
    release.set()
    thread.join()
    assert first == [409] and second.status_code == 200


def test_histogram_and_display_tone_match_the_tone_step(client: TestClient, root: Path) -> None:
    info = client.post("/api/datasets", json={"path": str(root / "scan.h5"), "dataset": "images"}).json()
    base = f"/api/datasets/{info['id']}/frames/1"
    hist = client.get(f"{base}/histogram", params={"bins": 10}).json()
    assert (hist["low"], hist["high"]) == (DATA[1].min(), DATA[1].max()) and sum(hist["counts"]) == DATA[1].size
    window = client.get(f"{base}/histogram", params={"bins": 4, "low": 0, "high": 4}).json()
    counts = [int((DATA[1] == v).sum()) for v in range(5)]
    assert window["counts"] == [*counts[:3], counts[3] + counts[4]]  # bins of width 1; the last includes 4
    assert client.get(f"{base}/histogram", params={"bins": 1}).status_code == 400

    tone = {"exposure": 0.5, "contrast": 40, "shadows": 30, "curve": [[0, 0], [0.5, 0.4], [1, 1]]}
    params = {"vmin": 10_000, "vmax": 40_000}
    shown = client.get(f"{base}/tiles/0/0_0.png", params={**params, "tone": json.dumps(tone)})
    pixels = np.asarray(Image.open(io.BytesIO(shown.content))).astype(int)
    step = {"op": "tone", "params": {"black": 10_000, "white": 40_000, **tone}}
    preview = client.post(f"{base}/preview", json={"steps": [step], "client": "tab"}).json()
    stepped = client.get(f"{base}/tiles/0/0_0.png", params={**params, "pipeline": preview["pipeline"]})
    np.testing.assert_allclose(np.asarray(Image.open(io.BytesIO(stepped.content))), pixels, atol=1)  # same mapping
    plain = np.asarray(Image.open(io.BytesIO(client.get(f"{base}/tiles/0/0_0.png", params=params).content)))
    assert np.abs(plain.astype(int) - pixels).max() > 20  # the tone did change the tile
    default = client.get(f"{base}/tiles/0/0_0.png", params={**params, "tone": json.dumps({"contrast": 0})})
    assert default.content == client.get(f"{base}/tiles/0/0_0.png", params=params).content
    bad = client.get(f"{base}/tiles/0/0_0.png", params={**params, "tone": '{"gamma": 2}'})
    assert bad.status_code == 400 and "keys from" in bad.json()["detail"]

    curves = client.get("/api/tone-curve", params={"tone": json.dumps({"exposure": 1}), "points": 5}).json()
    np.testing.assert_allclose(curves["curve"], [0, 0.25, 0.5, 0.75, 1])  # the curve alone is the identity
    np.testing.assert_allclose(curves["response"], [0, 0.5, 1, 1, 1])


def test_connections_per_browser(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    moto = pytest.importorskip("moto")
    import boto3

    from pd_xray.server import connections as connections_module
    from pd_xray.storage import HostKeyError

    app = create_app(Settings(roots=(root,)))
    s3 = {"bucket": "lab", "region": "us-east-1", "endpoint_url": ""}
    keys = {"access_key": "testing", "secret_key": "testing"}
    with moto.mock_aws(), TestClient(app) as browser, TestClient(app) as other:
        client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x")
        client.create_bucket(Bucket="lab")
        client.put_object(Bucket="lab", Key="scans/b10.tif", Body=b"12")
        client.put_object(Bucket="lab", Key="scans/b9.tif", Body=b"1")
        client.put_object(Bucket="lab", Key="scans/sub/x", Body=b"")

        added = browser.post("/api/connections", json={"kind": "s3", "settings": s3, "credentials": keys}).json()
        assert added["url"] == added["name"] == "s3://lab/" and "credentials" not in added
        assert added["settings"] == {"bucket": "lab", "region": "us-east-1"}  # empty fields dropped
        listing = browser.get(f"/api/connections/{added['id']}/files", params={"path": "/scans/"}).json()
        assert [(e["name"], e["kind"], e["size"]) for e in listing["entries"]] == [
            ("sub", "dir", None),
            ("b9.tif", "file", 1),
            ("b10.tif", "file", 2),
        ]
        assert (listing["path"], listing["parent"], listing["connection"]) == ("scans", "", added["id"])

        assert [c["id"] for c in browser.get("/api/connections").json()] == [added["id"]]
        assert other.get("/api/connections").json() == []  # another browser sees nothing
        assert other.get(f"/api/connections/{added['id']}/files").status_code == 404
        missing = browser.get(f"/api/connections/{added['id']}/files", params={"path": "nope"})
        assert missing.status_code == 404
        browser.delete(f"/api/connections/{added['id']}")
        assert browser.get("/api/connections").json() == []

        bad = browser.post("/api/connections", json={"kind": "s3", "settings": s3, "credentials": {}})
        assert bad.status_code == 400 and "access key" in bad.json()["detail"]
        bad = browser.post("/api/connections", json={"kind": "s3", "settings": {"x": 1}, "credentials": keys})
        assert bad.status_code == 400 and "Unknown s3 fields: x" in bad.json()["detail"]
        bad = browser.post("/api/connections", json={"kind": "sftp", "settings": {"host": "h"}, "credentials": {}})
        assert bad.status_code == 400 and "password or a private key" in bad.json()["detail"]

        def untrusted(*args: Any) -> None:
            raise HostKeyError("Unknown host key", "SHA256:abc")

        monkeypatch.setattr(connections_module, "open_storage", untrusted)
        sftp = {"host": "nas", "username": "me"}
        response = browser.post(
            "/api/connections", json={"kind": "sftp", "settings": sftp, "credentials": {"password": "p"}}
        )
        assert response.status_code == 400 and response.json()["fingerprint"] == "SHA256:abc"


def test_copy_job(root: Path) -> None:
    moto = pytest.importorskip("moto")
    import boto3

    keys = {"access_key": "testing", "secret_key": "testing"}
    with moto.mock_aws(), TestClient(create_app(Settings(roots=(root,)))) as client:
        s3 = boto3.client("s3", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x")
        s3.create_bucket(Bucket="lab")
        s3.put_object(Bucket="lab", Key="scans/a.bin", Body=b"a" * 1000)
        s3.put_object(Bucket="lab", Key="scans/sub/b.bin", Body=b"b")
        body = {"kind": "s3", "settings": {"bucket": "lab", "region": "us-east-1"}, "credentials": keys}
        connection = client.post("/api/connections", json=body).json()["id"]
        bucket = {"connection": connection, "path": "scans"}

        def copy(source: dict[str, Any], destination: dict[str, Any], **options: Any) -> dict[str, Any]:
            job = client.post("/api/jobs/copy", json={"source": source, "destination": destination, **options})
            assert job.status_code == 200, job.json()
            assert job.json()["unit"] == "bytes"
            for _ in range(200):
                snapshot = client.get(f"/api/jobs/{job.json()['id']}").json()
                if snapshot["state"] in FINISHED:
                    return snapshot  # type: ignore[no-any-return]
                time.sleep(0.02)
            raise AssertionError("job didn't finish")

        sub = {"path": str(root / "sub")}
        done = copy(bucket, sub)  # download
        assert done["state"] == "done" and done["result"]["copied"] == 2 and done["result"]["checksummed"] == 2
        assert done["result"]["destination"] == str(root / "sub" / "scans")
        assert (root / "sub" / "scans" / "a.bin").read_bytes() == b"a" * 1000
        assert (root / "sub" / "scans" / "sub" / "b.bin").read_bytes() == b"b"
        assert copy(bucket, sub)["result"]["skipped"] == 2
        (root / "sub" / "scans" / "a.bin").write_bytes(b"x")
        failed = copy(bucket, sub)
        assert failed["state"] == "failed" and "overwrite" in failed["error"]
        assert copy(bucket, sub, overwrite=True)["result"]["copied"] == 1

        (root / "sub" / "new.bin").write_bytes(b"new")
        done = copy({"path": str(root / "sub" / "new.bin")}, bucket)  # upload
        assert done["state"] == "done" and done["result"]["copied"] == 1
        assert s3.get_object(Bucket="lab", Key="scans/new.bin")["Body"].read() == b"new"
        assert copy({"path": str(root / "sub" / "scans")}, {"connection": connection, "path": ""})["result"] == {
            "copied": 0,  # the downloaded files carry the objects' dates, so they are the same
            "skipped": 2,
            "bytes_copied": 0,
            "checksummed": 0,
            "destination": "s3://lab/: /scans",
        }

        request = {"source": bucket, "destination": {"path": "/tmp"}}
        assert client.post("/api/jobs/copy", json=request).status_code == 403
        request = {"source": bucket, "destination": {"connection": connection, "path": "scans/a.bin"}}
        response = client.post("/api/jobs/copy", json=request)
        assert response.status_code == 400 and "Not a folder" in response.json()["detail"]


def run_and_wait(client: TestClient, body: dict[str, Any], cancel_at: int | None = None) -> dict[str, Any]:
    with client.websocket_connect("/api/events") as ws:
        response = client.post("/api/jobs/run", json=body)
        assert response.status_code == 200, response.text
        job = response.json()
        while job["state"] not in FINISHED:
            update = ws.receive_json()["job"]
            job = update if update["id"] == job["id"] else job  # the socket also replays earlier jobs
            if cancel_at is not None and job["current"] >= cancel_at:
                client.post(f"/api/jobs/{job['id']}/cancel")
    return job


def test_pipeline_files_and_run_job(client: TestClient, root: Path) -> None:
    from pd_xray.io import convert, map_frames, open_dataset, reference_frames
    from pd_xray.processing import Pipeline, average_frames

    Image.fromarray(np.full((600, 700), 1000, np.uint16)).save(root / "sub" / "dark.tif")
    info = client.post("/api/datasets", json={"path": str(root / "scan.h5"), "dataset": "images"}).json()
    steps = [
        {"op": "flat_field", "params": {"dark": str(root / "sub" / "dark.tif"), "flat": None}},
        {"op": "crop", "params": {"top": 10, "height": 500}},
        {"op": "median", "params": {"size": 3}, "enabled": False},
    ]

    # Save and load: references inside the dataset's folder are stored relative (D10).
    saved = client.post("/api/pipelines", json={"path": str(root / "p.yaml"), "steps": steps, "dataset_id": info["id"]})
    assert saved.status_code == 200
    assert "dark: sub/dark.tif" in (root / "p.yaml").read_text()
    again = client.post("/api/pipelines", json={"path": str(root / "p.yaml"), "steps": steps})
    assert again.status_code == 409
    loaded = client.get("/api/pipelines", params={"path": str(root / "p.yaml")}).json()
    assert loaded["steps"][0]["params"]["dark"] == "sub/dark.tif" and loaded["steps"][2]["enabled"] is False
    assert client.post("/api/pipelines", json={"path": str(root / "p.txt"), "steps": steps}).status_code == 400

    # Run with the loaded (relative) steps; the job writes the output and its pipeline file.
    body = {"dataset_id": info["id"], "steps": loaded["steps"], "destination": str(root / "out" / "x.tif")}
    assert client.post("/api/jobs/run", json=body).status_code == 400  # no such folder
    body["destination"] = str(root / "sub" / "out.tif")
    job = run_and_wait(client, body)
    assert job["state"] == "done", job["error"]
    assert job["result"]["frames"] == 3 and job["current"] == job["total"] == 3
    expected = DATA[:, 10:510].astype(np.float32) - 1000  # no flat: frame - dark
    with open_dataset(root / "sub" / "out.tif") as out:
        assert out.dtype == np.float32
        np.testing.assert_allclose(out[...], expected, rtol=1e-6)
        assert out.metadata["pd_xray_run"]["source"] == str(root / "scan.h5")

    # The pipeline file next to the output reproduces it from plain Python.
    pipeline = Pipeline.load(root / "sub" / "out.tif.pipeline.yaml")
    with open_dataset(root / "scan.h5", dataset="images") as ds:
        resolved = pipeline.resolve(lambda ref: average_frames(reference_frames(ref, base=root)))
        convert(map_frames(ds, resolved, workers=2), root / "again.tif")
    with open_dataset(root / "again.tif") as a, open_dataset(root / "sub" / "out.tif") as b:
        np.testing.assert_array_equal(a[...], b[...])

    # Existing output: refused unless overwrite. Integer output scales from the value range.
    assert client.post("/api/jobs/run", json=body).status_code == 409
    scaled = {**body, "dtype": "uint8", "value_range": [0, 10_000], "frames": [1, None, None], "overwrite": True}
    job = run_and_wait(client, scaled)
    assert job["state"] == "done", job["error"]
    with open_dataset(root / "sub" / "out.tif") as out:
        assert (out.dtype, out.shape) == (np.uint8, (2, 500, 700))
        scaled_values = expected[1:].astype(np.float64) / (10_000 / 255)  # as convert computes it
        np.testing.assert_array_equal(out[...], np.rint(np.clip(scaled_values, 0, 255)))
    bad = client.post("/api/jobs/run", json={**body, "destination": str(root / "o.png"), "dtype": "float32"})
    assert bad.status_code == 400 and "can't store" in bad.json()["detail"]

    # A cancelled run leaves no output (D8).
    slow = [{"op": "median", "params": {"size": 25}}]
    many = np.zeros((40, 600, 700), np.uint16)
    with h5py.File(root / "many.h5", "w") as f:
        f["images"] = many
    info = client.post("/api/datasets", json={"path": str(root / "many.h5")}).json()
    job = run_and_wait(client, {"dataset_id": info["id"], "steps": slow, "destination": str(root / "c.h5")}, 2)
    assert job["state"] == "cancelled"
    assert not [p.name for p in root.iterdir() if p.name.startswith(("c.h5", ".c.h5"))]
