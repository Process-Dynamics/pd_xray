from pathlib import Path

import numpy as np
import pytest

from pd_xray.core import CancelToken, OperationCancelledError, PdXrayError
from pd_xray.processing import Pipeline, Step, apply_pipeline

FRAMES = np.arange(3 * 6 * 8, dtype=np.uint16).reshape(3, 6, 8)


def test_round_trip_keeps_defaults_and_disabled_steps(tmp_path: Path) -> None:
    pipeline = Pipeline([Step("flat_field", {"dark": "scan/*_dar.tif"}), Step("gaussian", {"sigma": 2}, False)])
    loaded = Pipeline.load(pipeline.save(tmp_path / "p.yaml"))
    assert loaded == pipeline
    assert loaded.steps[1] == Step("gaussian", {"sigma": 2.0}, False)
    assert "pd_xray_pipeline: 1" in (tmp_path / "p.yaml").read_text()


def test_bad_files_raise_clear_errors() -> None:
    with pytest.raises(PdXrayError, match="version 1"):
        Pipeline.from_yaml("steps: []")
    with pytest.raises(PdXrayError, match="Step 1: unknown operation 'blur'"):
        Pipeline.from_yaml("pd_xray_pipeline: 1\nsteps:\n  - op: blur\n")
    with pytest.raises(PdXrayError, match=r"Step 2 \(median\)"):
        Pipeline([Step("invert"), Step("median", {"size": 0})])


def test_apply_resolves_references_and_skips_disabled() -> None:
    pipeline = Pipeline(
        [Step("flat_field", {"dark": "d"}), Step("invert", enabled=False), Step("crop", {"width": 4}), Step("flip")]
    )
    with pytest.raises(PdXrayError, match="Unresolved"):
        apply_pipeline(pipeline, FRAMES)
    resolved = pipeline.resolve(lambda reference: np.ones((6, 8)))
    with pytest.raises(PdXrayError, match="loaded array"):
        resolved.to_dict()
    out = list(apply_pipeline(resolved, FRAMES))
    assert len(out) == 3 and out[0].shape == (6, 4) and out[0].dtype == np.float32
    np.testing.assert_array_equal(out[2], FRAMES[2, :, :4][:, ::-1] - 1.0)  # choice strings aren't references


def test_apply_reports_progress_and_cancels() -> None:
    seen, stop = [], CancelToken()

    def progress(p: object) -> None:
        seen.append(p)
        stop.cancel()

    with pytest.raises(OperationCancelledError):
        list(apply_pipeline(Pipeline(), FRAMES, progress=progress, cancel=stop))
    assert len(seen) == 1


def test_list_references_round_trip_and_resolve(tmp_path: Path) -> None:
    pipeline = Pipeline([Step("flat_field", {"dark": ["a.tif", "b.tif"], "flat": "flats"})])
    loaded = Pipeline.load(pipeline.save(tmp_path / "p.yaml"))
    assert loaded == pipeline and loaded.references == {("a.tif", "b.tif"), "flats"}
    calls = []

    def load(reference: object) -> np.ndarray:
        calls.append(reference)
        return np.full((6, 8), 2.0 if reference == "flats" else 1.0)

    resolved = loaded.resolve(load)
    assert sorted(map(str, calls)) == sorted(map(str, [("a.tif", "b.tif"), "flats"]))
    np.testing.assert_array_equal(resolved(FRAMES[1]), FRAMES[1] - 1.0)  # (frame - 1) / (2 - 1)
    with pytest.raises(PdXrayError, match="non-empty list"):
        Pipeline([Step("flat_field", {"dark": []})])


def test_call_reports_steps_and_cancels_between_them() -> None:
    pipeline = Pipeline([Step("invert"), Step("median", enabled=False), Step("gaussian")])
    seen: list[tuple[int, int | None, str]] = []
    pipeline(FRAMES[0], progress=lambda p: seen.append((p.current, p.total, p.message)))
    assert seen == [(0, 2, "invert"), (1, 2, "gaussian"), (2, 2, "")]
    stop = CancelToken()

    def cancel_after_first(p: object) -> None:
        stop.cancel()

    with pytest.raises(OperationCancelledError):
        pipeline(FRAMES[0], progress=cancel_after_first, cancel=stop)
