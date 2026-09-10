"""
Every path the app touches, resolved from the project root rather than the
working directory.

Streamlit is usually started from the project folder, but a container, a
systemd unit or a scheduled task will not be, and a font that resolves on
Windows by bare filename resolves nowhere on Debian. Both problems land here so
the rest of the code can ask for a thing rather than a location.
"""

from __future__ import annotations

import os
import platform
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT: Path = Path(__file__).resolve().parent
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
FONTS_DIR: Path = ASSETS_DIR / "fonts"
ENV_FILE: Path = PROJECT_ROOT / ".env"


def _from_env(name: str, default: Path) -> Path:
    """
    An override from the environment, or the project-relative default.

    Both of these have to leave the image in a container: renders and accounts
    must survive a rebuild, which means they live on a mounted volume, which
    means the path cannot be baked into the source.
    """
    override = os.environ.get(name, "").strip()
    return Path(override).expanduser().resolve() if override else default


# REELFORGE_EXPORTS_DIR / REELFORGE_USERS_FILE, read at import so a container
# can point both at a volume without touching the code.
EXPORTS_ROOT: Path = _from_env("REELFORGE_EXPORTS_DIR", PROJECT_ROOT / "exports")
USERS_FILE: Path = _from_env("REELFORGE_USERS_FILE", PROJECT_ROOT / "users.json")

# Scratch lives in the system temp directory on every platform. In a container
# that is a tmpfs or the overlay, which is what we want: intermediates should
# never survive a restart, and they must not land in a mounted exports volume.
TEMP_DIR: Path = Path(tempfile.gettempdir())


def project_path(*parts: str) -> str:
    """An absolute path inside the project, as a string for os-level APIs."""
    return str(PROJECT_ROOT.joinpath(*parts))


def ensure_dir(path: Path | str) -> str:
    """Creates a directory if it is missing and returns it as a string."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


# ---------------------------------------------------------------------------
# Fonts
#
# PIL resolves a bare "arialbd.ttf" through the Windows font directory and
# nowhere else, so the same call that works on a laptop silently falls back to
# the 11px bitmap default on a Linux server. The index below makes the lookup
# explicit: project assets first, then whatever the OS actually ships.
# ---------------------------------------------------------------------------

def _system_font_dirs() -> list[Path]:
    system = platform.system()
    candidates: list[Path] = [FONTS_DIR]

    if system == "Windows":
        candidates += [
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Fonts",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts",
        ]
    elif system == "Darwin":
        candidates += [Path("/System/Library/Fonts"), Path("/Library/Fonts"),
                       Path.home() / "Library" / "Fonts"]
    else:
        candidates += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
                       Path.home() / ".fonts", Path.home() / ".local/share/fonts"]

    return [path for path in candidates if path and str(path) != "." and path.is_dir()]


@lru_cache(maxsize=1)
def font_index() -> dict[str, str]:
    """
    Maps lowercase font filename -> absolute path, for every font on the box.

    Walked once and cached: the Debian font tree is a few hundred files, and
    this is called for every text size the renderer asks for.
    """
    found: dict[str, str] = {}
    for directory in reversed(_system_font_dirs()):     # project assets win
        try:
            for path in directory.rglob("*"):
                if path.suffix.lower() in (".ttf", ".otf", ".ttc") and path.is_file():
                    found[path.name.lower()] = str(path)
        except (OSError, PermissionError):
            continue
    return found


def resolve_font(names: Sequence[str]) -> str | None:
    """
    First of `names` that exists, as an absolute path.

    Accepts bare filenames ("Montserrat-Bold.ttf"), which is what the callers
    have, and absolute paths, which is what a Docker image can pin.
    """
    index = font_index()
    for name in names:
        if os.path.isabs(name) and os.path.exists(name):
            return name
        hit = index.get(os.path.basename(name).lower())
        if hit:
            return hit
    return None


def installed_families(names: Iterable[str]) -> list[str]:
    """
    Which of these font *families* libass will actually find.

    libass takes a family name, not a path, and silently substitutes when the
    family is missing -- which is how a caption ends up in a default serif on a
    server. Matching on filename is a good enough proxy to pick a name that is
    really there.
    """
    index = font_index()
    stems = {Path(key).stem.replace("-", "").replace("_", "").lower() for key in index}
    available: list[str] = []
    for family in names:
        squashed = family.replace(" ", "").replace("-", "").lower()
        if any(squashed in stem or stem.startswith(squashed) for stem in stems):
            available.append(family)
    return available
