"""Shared pytest setup.

Two things every test in this directory needs:

* the project root on sys.path, since the engines are top-level modules rather
  than a package;
* an exports directory that is not the real one. `app` reads
  REELFORGE_EXPORTS_DIR at import time, so it has to be set before the first
  import of anything that imports app -- which is why this lives in conftest
  rather than in a fixture.
"""
from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_SANDBOX = tempfile.mkdtemp(prefix="reelforge_tests_")
os.environ.setdefault("REELFORGE_EXPORTS_DIR", _SANDBOX)
os.environ.setdefault("REELFORGE_USERS_FILE", os.path.join(_SANDBOX, "users.json"))

import pytest  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--network", action="store_true", default=False,
                     help="run the tests that download from the live platforms")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network: needs the internet")
    config.addinivalue_line("markers", "slow: runs ffmpeg, takes tens of seconds")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--network"):
        return
    skip = pytest.mark.skip(reason="needs --network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def sandbox() -> str:
    """The throwaway exports directory this run is using."""
    return _SANDBOX


@pytest.fixture()
def workdir(tmp_path) -> str:
    return str(tmp_path)


@pytest.fixture(scope="session")
def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
