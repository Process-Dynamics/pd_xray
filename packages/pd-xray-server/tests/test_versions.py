"""The five packages are released together: same version, and each requires its siblings at that version."""

import re
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[2]


def test_versions_and_inner_pins_match() -> None:
    files = {p.parent.name: p.read_text() for p in PACKAGES.glob("pd-xray-*/pyproject.toml")}
    if len(files) != 5:
        pytest.skip("not run from the repository")
    versions = {name: re.search(r'^version = "(.+)"$', text, re.M).group(1) for name, text in files.items()}  # type: ignore[union-attr]
    assert len(set(versions.values())) == 1, versions
    version = next(iter(versions.values()))
    for name, text in files.items():
        for sibling, pin in re.findall(r'"(pd-xray-[a-z]+)(?:\[[a-z,]+\])?([^"]*)"', text):
            if sibling != name:
                assert pin == f"~={version}", f"{name} requires {sibling}{pin}; use ~={version}"
