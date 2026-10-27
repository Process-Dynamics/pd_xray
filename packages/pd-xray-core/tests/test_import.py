import pd_xray.core


def test_import() -> None:
    assert pd_xray.core.__all__ is not None
