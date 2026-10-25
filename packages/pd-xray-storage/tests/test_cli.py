from pathlib import Path

import pytest

from pd_xray.storage.cli import main


def test_copy_and_ls(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "scan").mkdir()
    (tmp_path / "scan" / "a.bin").write_bytes(b"abc")
    (tmp_path / "out").mkdir()
    main(["copy", str(tmp_path / "scan"), str(tmp_path / "out")])  # existing folder: copied inside it, like cp
    assert (tmp_path / "out" / "scan" / "a.bin").read_bytes() == b"abc"
    assert "Copied 1 files" in capsys.readouterr().out
    main(["copy", str(tmp_path / "scan"), str(tmp_path / "out")])  # the same command again
    assert "skipped 1 identical" in capsys.readouterr().out
    main(["ls", str(tmp_path / "out" / "scan")])
    assert capsys.readouterr().out == "a.bin\t3\n"
    with pytest.raises(SystemExit, match="Unknown location type 'ftp://'"):
        main(["ls", "ftp://host/x"])
