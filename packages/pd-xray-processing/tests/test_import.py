import pd_xray.processing


def test_import() -> None:
    assert pd_xray.processing.__all__ is not None
