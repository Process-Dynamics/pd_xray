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
    pipeline = Pipeline([Step("flat_field", {"dark": "d"}), Step("invert", enabled=False), Step("crop", {"width": 4})])
    with pytest.raises(PdXrayError, match="Unresolved"):
        apply_pipeline(pipeline, FRAMES)
    resolved = pipeline.resolve(lambda reference: np.ones((6, 8)))
    with pytest.raises(PdXrayError, match="loaded array"):
        resolved.to_dict()
    out = list(apply_pipeline(resolved, FRAMES))
    assert len(out) == 3 and out[0].shape == (6, 4) and out[0].dtype == np.float32
    np.testing.assert_array_equal(out[2], FRAMES[2, :, :4] - 1.0)


def test_apply_reports_progress_and_cancels() -> None:
    seen, stop = [], CancelToken()

    def progress(p: object) -> None:
        seen.append(p)
        stop.cancel()

    with pytest.raises(OperationCancelledError):
        list(apply_pipeline(Pipeline(), FRAMES, progress=progress, cancel=stop))
    assert len(seen) == 1
