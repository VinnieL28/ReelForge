"""
Runs the standalone verification suites under pytest.

The `suite_*.py` files are scripts, not pytest modules -- they print a readable
narrative of what they measured, which is the point of them, and they are meant
to be run directly while working on the engine they cover. They are named
`suite_` rather than `test_` deliberately: pytest imports every `test_*.py` at
collection time, so as `test_*.py` their entire bodies executed during
`--collect-only`, which made collection take 30 seconds and would have fired a
live TikTok download before a single test ran.

Here they are run as subprocesses instead, so a failure is reported as a
failure rather than a collection error, and the slow and networked ones are
behind markers.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

SUITES = os.path.dirname(os.path.abspath(__file__))


def _run(name: str, timeout: int) -> None:
    path = os.path.join(SUITES, name)
    assert os.path.exists(path), path

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run([sys.executable, path], capture_output=True, text=True,
                          timeout=timeout, env=env, cwd=os.path.dirname(SUITES))

    # The suites print what they measured; surface it on failure.
    if proc.returncode != 0:
        pytest.fail(f"{name} failed\n\n--- stdout ---\n{proc.stdout[-3000:]}"
                    f"\n\n--- stderr ---\n{proc.stderr[-2000:]}")
    print(proc.stdout[-1500:])


def test_narrative_unit():
    """Entity extraction, storyboard JSON parsing, retiming."""
    _run("suite_narrative_unit.py", timeout=300)


@pytest.mark.slow
def test_vector_scenes():
    """The rig, all fifteen templates, the palette and the easing."""
    _run("suite_vector_scenes.py", timeout=600)


@pytest.mark.slow
def test_narrative_ffmpeg():
    """Ken Burns, the dissolve chain arithmetic, subtitle burn-in."""
    _run("suite_narrative_ffmpeg.py", timeout=900)


@pytest.mark.network
def test_url_ingest():
    """URL sanitising and a live TikTok download."""
    _run("suite_url_ingest.py", timeout=600)
