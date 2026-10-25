# `pd_xray.core`: axes, progress, cancelling, errors

Shared by all packages. Needs only NumPy.

## Axis names

```python
from pd_xray.core import validate_axes, frame_axes

validate_axes(("angle", "y", "x"), 3)        # ('angle', 'y', 'x')
validate_axes(("x", "y"), 2)                 # ValueError: must end with ('y', 'x')
frame_axes(("time", "angle", "y", "x"))      # ('time', 'angle'): the axes that index frames
```

Arrays always end with `("y", "x")`, or `("y", "x", "channel")` for colour. Constants: `X`, `Y`, `Z`, `TIME`,
`ANGLE`, `CHANNEL`.

## Progress and cancelling

Long functions (`convert`, `compute_value_range`, `apply_pipeline`, `average_frames`, `copy`, …) accept two optional
arguments:

- `progress`: a function called with `Progress(current, total, message)`. `total` may be None; `fraction` gives
  0..1.
- `cancel`: a `CancelToken`. Calling `token.cancel()` (from any thread) makes the function stop at its next safe
  point with `OperationCancelledError`.

```python
import threading
from pd_xray.core import CancelToken, OperationCancelledError, Progress, track

def show(p: Progress) -> None:
    print(f"{p.current}/{p.total} {p.message}")

stop = CancelToken()
threading.Timer(10, stop.cancel).start()      # e.g. give up after 10 s
try:
    convert(ds, "out.tif", progress=show, cancel=stop)
except OperationCancelledError:
    print("stopped")
```

`track` adds the same to your own loops:

```python
for i in track(range(ds.shape[0]), message="frames", progress=show, cancel=stop):
    ...
```

The libraries never print; showing progress is up to you (a print, a tqdm bar, the web app's status bar).

## Errors

All pd_xray errors derive from `PdXrayError`, so `except PdXrayError` catches them. `MissingDependencyError` says
which extra to install, e.g. `pip install "pd-xray-io[hdf5]"`. Messages say what went wrong and what to do.

## Logging

The packages log through Python's `logging` (e.g. "Resuming … at 939524096 bytes") and never configure it. To see
messages in a script:

```python
import logging
logging.basicConfig(level=logging.INFO)
```
