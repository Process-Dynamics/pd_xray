"""Frame-level visualisation and saving utilities for array selections.

Designed for use with data loaded via HDF5Reader / S3Backend, where the user
has already sliced a section of a larger array and wants to inspect it or
save it for later analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from pd_xray.core.logging import get_logger

if TYPE_CHECKING:
    from pd_xray.processing import ImageProcessor

logger = get_logger(__name__)


def save_selection(arr: NDArray, path: str | Path) -> None:
    """Save an array selection to disk using np.save.

    Args:
        arr  : Any NumPy array, typically a slice of a larger dataset
               (e.g. ``lazy_arr[100:200]``).
        path : Destination path. A ``.npy`` extension is added if not present.

    Usage::

        with reader.lazy_open_remote(backend, "scan.h5") as arr:
            section = arr[100:200]
        save_selection(section, "/scratch/scan_frames_100_200.npy")
    """
    path = Path(path)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    logger.info("Saved array  shape=%s  dtype=%s  -> %s", arr.shape, arr.dtype, path)


def _is_jupyter() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


_MAX_DISPLAY_PX = 600


def _encode_frame(frame: NDArray, cmap: str) -> str:
    """Encode a 2D array as a base64 PNG using PIL. Much faster than matplotlib for large images."""
    import base64
    import io

    from PIL import Image

    f = np.asarray(frame, dtype=np.float32)
    vmin, vmax = float(f.min()), float(f.max())
    span = vmax - vmin
    normed = (f - vmin) / span if span > 0 else np.zeros_like(f)

    if cmap == "gray":
        img = Image.fromarray((normed * 255).astype(np.uint8), mode="L")
    else:
        from matplotlib import colormaps
        rgba = colormaps[cmap](normed, bytes=True)
        img = Image.fromarray(rgba[:, :, :3], mode="RGB")

    h, w = normed.shape
    if h > _MAX_DISPLAY_PX:
        img = img.resize((int(w * _MAX_DISPLAY_PX / h), _MAX_DISPLAY_PX), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _view_frames_jupyter(arr: NDArray, title: str, cmap: str) -> None:
    """Render a frame slider as an HTML/JS widget — works with the default inline backend."""
    from IPython.display import HTML, display

    n_frames, h, w = arr.shape
    disp_h = min(h, _MAX_DISPLAY_PX)
    disp_w = int(w * disp_h / h)

    frames_b64 = [_encode_frame(arr[i], cmap) for i in range(n_frames)]

    uid = f"fv{abs(id(arr)):x}"
    frames_json = "[" + ",".join(f'"{f}"' for f in frames_b64) + "]"

    html = f"""
<div style="font-family:sans-serif">
  <b>{title}</b>
  <br>
  <img id="{uid}i"
       src="data:image/png;base64,{frames_b64[0]}"
       style="width:{disp_w}px;display:block;margin-top:6px">
  <div style="margin-top:8px;display:flex;align-items:center;gap:12px">
    <input id="{uid}s" type="range" min="0" max="{n_frames - 1}" value="0"
           style="width:{min(disp_w, 500)}px"
           oninput="var v=parseInt(this.value);
                    document.getElementById('{uid}i').src='data:image/png;base64,'+{uid}f[v];
                    document.getElementById('{uid}l').textContent='Frame '+v+' / {n_frames - 1}';">
    <span id="{uid}l">Frame 0 / {n_frames - 1}</span>
  </div>
</div>
<script>var {uid}f = {frames_json};</script>
"""
    display(HTML(html))


def _view_frames_matplotlib(arr: NDArray, title: str, cmap: str) -> None:
    """Render a frame slider using matplotlib.widgets — for scripts and non-notebook use."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    n_frames = arr.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.canvas.manager.set_window_title(title)
    plt.subplots_adjust(bottom=0.12)

    im = ax.imshow(arr[0], cmap=cmap, aspect="equal")
    ax.set_title(f"frame 0 / {n_frames - 1}")
    ax.axis("off")

    ax_slider = fig.add_axes([0.15, 0.03, 0.70, 0.04])
    slider = Slider(ax_slider, "Frame", 0, n_frames - 1, valinit=0, valstep=1)

    def _update(val: Any) -> None:
        idx = int(slider.val)
        im.set_data(arr[idx])
        im.set_clim(arr[idx].min(), arr[idx].max())
        ax.set_title(f"frame {idx} / {n_frames - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(_update)
    plt.show()


def view_frames(arr: NDArray, title: str = "Frame viewer", cmap: str = "gray") -> None:
    """Display a 2D image or a 3D stack with an interactive frame slider.

    In Jupyter notebooks, renders an HTML/JavaScript slider that works with
    the default inline backend — no extra packages required.
    Outside notebooks, uses a matplotlib Slider widget.

    Args:
        arr   : 2D ``(H, W)`` or 3D ``(Z, H, W)`` NumPy array.
        title : Figure title / window title.
        cmap  : Matplotlib colormap name. Default ``"gray"``.

    Usage::

        with reader.lazy_open_remote(backend, "scan.h5") as lazy:
            section = lazy[100:120]
        view_frames(section)
    """
    import matplotlib.pyplot as plt

    arr = np.asarray(arr)

    if arr.ndim == 2:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(arr, cmap=cmap, aspect="equal")
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        if _is_jupyter():
            from IPython.display import display
            display(fig)
            plt.close(fig)
        else:
            fig.canvas.manager.set_window_title(title)
            plt.show()
        return

    if arr.ndim != 3:
        raise ValueError(f"view_frames expects a 2D or 3D array, got shape {arr.shape}")

    if _is_jupyter():
        _view_frames_jupyter(arr, title=title, cmap=cmap)
    else:
        _view_frames_matplotlib(arr, title=title, cmap=cmap)


def apply_and_view(
    arr: NDArray,
    processor: "ImageProcessor",
    title: str = "Processed frame viewer",
    cmap: str = "gray",
) -> NDArray:
    """Apply an ImageProcessor pipeline to an array then open the frame viewer.

    Args:
        arr       : 2D or 3D NumPy array.
        processor : A configured :class:`~pd_xray.processing.ImageProcessor`.
        title     : Figure window title passed to :func:`view_frames`.
        cmap      : Matplotlib colormap passed to :func:`view_frames`.

    Returns:
        The processed array so you can inspect or save it.

    Usage::

        from pd_xray.processing import ImageProcessor

        proc = ImageProcessor().gaussian_blur(sigma=1.5).normalise()

        with reader.lazy_open_remote(backend, "scan.h5") as lazy:
            section = lazy[100:120]

        processed = apply_and_view(section, proc)
        save_selection(processed, "/scratch/processed_100_120.npy")
    """
    result = processor(arr)
    view_frames(result, title=title, cmap=cmap)
    return result
