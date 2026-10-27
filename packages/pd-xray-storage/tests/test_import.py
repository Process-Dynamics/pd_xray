import pd_xray.storage


def test_import() -> None:
    assert pd_xray.storage.__all__ is not None
