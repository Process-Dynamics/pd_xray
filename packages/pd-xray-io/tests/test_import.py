import pd_xray.io


def test_import() -> None:
    assert pd_xray.io.__all__ is not None
