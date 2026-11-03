"""Small local UI for rotating one ACCESS dendrite and rendering its X-ray view.

Double-click start_access_phase_viewer.cmd to open it in a browser. Stop it
with the page button or by closing the launcher console. No source data or
rendered image is written automatically.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import threading
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion


HERE = Path(__file__).resolve().parent
HTML = HERE / "access_phase_viewer.html"
LOW_RES: Path | None = None
HIGH_RES: Path | None = None
TIMES: list[float] = []
DETECTOR = 512
SOURCE_EDGE = 511
VOXEL_UM = 0.20
PIXEL_UM = SOURCE_EDGE * VOXEL_UM / (0.75 * DETECTOR)
_render_lock = threading.Lock()
_sparse = None


def configure_data(explicit: Path | None = None) -> Path:
    """Locate generated phase frames without copying simulation data into Git."""
    global LOW_RES, HIGH_RES, TIMES, _sparse
    configured = explicit or os.environ.get("PD_TOMO_VIEWER_DATA")
    candidates = ([Path(configured)] if configured else [
        HERE.parents[1] / "data",  # pd_xray/data (ignored by Git)
        HERE.parents[2] / "access_phase_data" / "cache",  # existing local workspace cache
    ])
    for root in candidates:
        high = root / "downsampled_511" / "phase"
        if all((high / f"phase_{step:04d}.h5").is_file() for step in range(7)):
            HIGH_RES = high
            low = root / "downsampled_63" / "phase"
            LOW_RES = low if low.is_dir() else None
            TIMES = []
            for step in range(7):
                with h5py.File(high / f"phase_{step:04d}.h5", "r") as handle:
                    dataset = handle["phase_field"]
                    if dataset.shape != (SOURCE_EDGE,) * 3:
                        raise ValueError(f"Expected a 511^3 phase field: {high / f'phase_{step:04d}.h5'}")
                    TIMES.append(float(dataset.attrs["time_s"]))
            if not all(a < b for a, b in zip(TIMES, TIMES[1:])):
                raise ValueError("Phase frame times must increase")
            _sparse = None
            preview_points.cache_clear()
            return root.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find downsampled_511/phase/phase_0000.h5 ... phase_0006.h5 under: {searched}. "
        "Pass --data-dir PATH or set PD_TOMO_VIEWER_DATA."
    )


def rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Object-to-view rotation: yaw about x, pitch about z, roll about y."""
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rx = np.array([[1, 0, 0], [0, cy, -sy], [0, sy, cy]])
    rz = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
    ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    return ry @ rz @ rx


def maxpool_preview_phase(step: int) -> np.ndarray:
    """Make a 64^3 preview when the optional 63^3 cache is absent."""
    assert HIGH_RES is not None
    output = np.zeros((64, 64, 64), dtype=np.float32)
    with h5py.File(HIGH_RES / f"phase_{step:04d}.h5", "r") as handle:
        source = handle["phase_field"]
        for block in range(64):
            start = block * 8
            slab = source[start:min(start + 8, SOURCE_EDGE), :, :]
            padded = np.zeros((8, 512, 512), dtype=np.float32)
            padded[:slab.shape[0], :SOURCE_EDGE, :SOURCE_EDGE] = slab
            output[block] = padded.reshape(8, 64, 8, 64, 8).max(axis=(0, 2, 4))
    return output


@lru_cache(maxsize=7)
def preview_points(step: int) -> list[list[float]]:
    assert HIGH_RES is not None
    if step == 0:
        return []  # the t=0 field is effectively empty
    if step <= 4:
        # The earliest seeds disappear at 63^3, so sample a small central ROI
        # from the 511^3 cache while keeping the original global scale.
        offset = 160
        with h5py.File(HIGH_RES / f"phase_{step:04d}.h5", "r") as handle:
            phase = handle["phase_field"][offset:351, offset:351, offset:351]
        solid = phase > 0.3
        coordinate_offset = offset
        coordinate_scale = 1
        center = 255
        radius = 255
    elif LOW_RES is not None and (LOW_RES / f"phase_{step:04d}.h5").is_file():
        with h5py.File(LOW_RES / f"phase_{step:04d}.h5", "r") as handle:
            phase = handle["phase_field"][:]
        solid = phase > 0.4
        coordinate_offset = 0
        coordinate_scale = 1
        center = 31
        radius = 31
    else:
        phase = maxpool_preview_phase(step)
        solid = phase > 0.4
        coordinate_offset = 3.5
        coordinate_scale = 8
        center = 255
        radius = 255
    surface = solid & ~binary_erosion(solid)
    zyx = np.argwhere(surface)
    if len(zyx) > 8500:
        rng = np.random.default_rng(42)
        zyx = zyx[np.sort(rng.choice(len(zyx), 8500, replace=False))]
    xyz = zyx[:, ::-1].astype(np.float32) * coordinate_scale + coordinate_offset
    xyz = (xyz - center) / radius
    return np.round(xyz, 4).tolist()


def sparse_phase(step: int) -> tuple[np.ndarray, np.ndarray]:
    global _sparse
    assert HIGH_RES is not None
    if _sparse is None or _sparse[0] != step:
        with h5py.File(HIGH_RES / f"phase_{step:04d}.h5", "r") as handle:
            phase = handle["phase_field"][:]
        mask = phase > 0.01
        zyx = np.argwhere(mask).astype(np.int16)
        density = phase[mask].astype(np.float32)
        # HDF5 axes are z,y,x. The projection uses world x,y,z.
        xyz = np.ascontiguousarray(zyx[:, ::-1])
        _sparse = (step, xyz, density)
    return _sparse[1:]


def render_projection(step: int, yaw: float, pitch: float, roll: float,
                      object_mu: float, background_mu: float) -> tuple[np.ndarray, np.ndarray, dict]:
    xyz, density = sparse_phase(step)
    r = rotation(yaw, pitch, roll)
    beam_world = r.T @ np.array([0.0, 1.0, 0.0])
    path_bg = SOURCE_EDGE * VOXEL_UM * float(np.abs(beam_world).sum())
    tau_bg = background_mu * path_bg
    voxel_path = VOXEL_UM * float(np.abs(beam_world).sum())
    tau_object = np.zeros(DETECTOR * DETECTOR, dtype=np.float64)
    for start in range(0, len(xyz), 750_000):
        stop = min(start + 750_000, len(xyz))
        world = (xyz[start:stop].astype(np.float32) - 255) * VOXEL_UM
        viewed = world @ r.T
        # Front view: detector horizontal=z, vertical=x, beam=+y.
        fu = viewed[:, 2] / PIXEL_UM + DETECTOR / 2
        fv = viewed[:, 0] / PIXEL_UM + DETECTOR / 2
        iu = np.floor(fu).astype(np.int32)
        iv = np.floor(fv).astype(np.int32)
        valid = (iu >= 0) & (iu < DETECTOR - 1) & (iv >= 0) & (iv < DETECTOR - 1)
        iu, iv = iu[valid], iv[valid]
        wu, wv = fu[valid] - iu, fv[valid] - iv
        contrib = density[start:stop][valid] * object_mu * voxel_path
        index = iv * DETECTOR + iu
        for offset, weight in (
            (0, (1 - wu) * (1 - wv)),
            (1, wu * (1 - wv)),
            (DETECTOR, (1 - wu) * wv),
            (DETECTOR + 1, wu * wv),
        ):
            tau_object += np.bincount(
                index + offset, weights=contrib * weight, minlength=DETECTOR * DETECTOR
            )
    tau_object = tau_object.reshape(DETECTOR, DETECTOR)
    raw = 1 - np.exp(-(tau_object + tau_bg))
    # Same view correction as the six-step inspection figure. Raw contains bg.
    corrected = np.clip((raw - (1 - np.exp(-tau_bg))) / np.exp(-tau_bg), 0, 1)
    enhanced = 0.15 + 0.85 * np.sqrt(corrected)
    return raw, enhanced, {
        "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll,
        "beam_world_xyz": np.round(beam_world, 5).tolist(),
        "background_tau": round(tau_bg, 4),
        "object_mu_per_um": object_mu,
        "background_mu_per_um": background_mu,
        "detector_px": DETECTOR,
        "source_frame": step,
        "source_time_s": TIMES[step],
    }


def png_base64(image: np.ndarray) -> str:
    pixels = np.uint8(np.round(np.clip(image, 0, 1) * 255))
    buffer = BytesIO()
    Image.fromarray(pixels, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class Handler(BaseHTTPRequestHandler):
    def send_payload(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/":
            self.send_payload(HTML.read_bytes(), "text/html; charset=utf-8")
        elif parsed.path == "/model":
            try:
                step = int(parse_qs(parsed.query).get("step", ["6"])[0])
                if not 0 <= step < len(TIMES):
                    raise ValueError("Invalid time step")
                data = json.dumps({
                    "points": preview_points(step), "step": step,
                    "source_time_s": TIMES[step], "timesteps": TIMES,
                }).encode()
                self.send_payload(data, "application/json")
            except ValueError as exc:
                self.send_payload(json.dumps({"error": str(exc)}).encode(), "application/json", 400)
        else:
            self.send_payload(b"Not found", "text/plain", 404)

    def do_POST(self) -> None:
        if self.path == "/shutdown":
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send_payload(b"Expected JSON", "text/plain", 415)
                return
            self.send_payload(b'{"stopped": true}', "application/json")
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if self.path != "/render":
            self.send_payload(b"Not found", "text/plain", 404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size < 4096:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(size))
            step = int(request["step"])
            if not 0 <= step < len(TIMES):
                raise ValueError("Invalid time step")
            yaw = float(request["yaw"])
            pitch = float(request["pitch"])
            roll = float(request["roll"])
            object_mu = float(request["object_mu"])
            background_mu = float(request["background_mu"])
            if not all(map(math.isfinite, (yaw, pitch, roll, object_mu, background_mu))):
                raise ValueError("Parameters must be finite")
            if not 0 < object_mu <= 0.2 or not 0 <= background_mu <= 0.2:
                raise ValueError("Attenuation must be between 0 and 0.2 /um")
            with _render_lock:
                raw, enhanced, info = render_projection(
                    step, yaw, pitch, roll, object_mu, background_mu
                )
            # Raw uses the same fixed [0.88, 1.0] window as the inspection figure.
            raw_preview = np.clip((raw - 0.88) / 0.12, 0, 1)
            data = json.dumps({
                "enhanced_png": png_base64(enhanced),
                "raw_png": png_base64(raw_preview),
                "info": info,
            }).encode()
            self.send_payload(data, "application/json")
        except Exception as exc:
            self.send_payload(json.dumps({"error": str(exc)}).encode(), "application/json", 400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local ACCESS dendrite X-ray viewer")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path,
        help="Directory containing downsampled_511/phase/phase_0000.h5 ... phase_0006.h5",
    )
    args = parser.parse_args()
    source = configure_data(args.data_dir)
    url = "http://127.0.0.1:8765/"
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print(f"ACCESS phase viewer: {url}", flush=True)
    print(f"Phase data: {source}", flush=True)
    print("Close this window or click Stop server in the page to quit.", flush=True)
    if args.open_browser:
        opener = threading.Timer(0.5, lambda: webbrowser.open(url))
        opener.daemon = True
        opener.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
