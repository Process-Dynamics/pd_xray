import pytest
from pathlib import Path

from pd_xray.core.config import Config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def cfg():
    return Config(FIXTURES / "valid_config.yaml")


def test_load_valid(cfg):
    assert isinstance(cfg._config, dict)


def test_get_existing_key(cfg):
    assert cfg.get("data.source.port") == 22