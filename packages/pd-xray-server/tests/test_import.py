import pd_xray.server


def test_import() -> None:
    assert pd_xray.server.__all__ is not None
