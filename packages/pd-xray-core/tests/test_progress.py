import pytest

from pd_xray.core import CancelToken, OperationCancelledError, Progress, track


def test_track_reports_start_and_each_item() -> None:
    updates: list[Progress] = []
    assert list(track(range(3), message="frames", progress=updates.append)) == [0, 1, 2]
    assert [(u.current, u.total) for u in updates] == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert updates[-1].fraction == 1.0
    assert updates[-1].message == "frames"


def test_track_unknown_total() -> None:
    updates: list[Progress] = []
    list(track(iter("ab"), progress=updates.append))
    assert updates[-1].total is None
    assert updates[-1].fraction is None


def test_track_stops_when_cancelled() -> None:
    cancel = CancelToken()
    seen = []
    with pytest.raises(OperationCancelledError):
        for i in track(range(10), cancel=cancel):
            seen.append(i)
            if i == 2:
                cancel.cancel()
    assert seen == [0, 1, 2]


def test_progress_fraction_is_clamped() -> None:
    assert Progress(5, 4).fraction == 1.0
    assert Progress(1, 0).fraction is None
