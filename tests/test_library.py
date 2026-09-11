"""
The Exports Library: classification, filtering, and a delete that stays inside
the signed-in account's own folder.

The library replaced a four-item strip that used to sit under every creation
page. Two properties matter and are asserted here rather than assumed:

* **Scratch stays out.** Downloaded sources, muted intermediates and narration
  WAVs live in the same folder as finished renders. Showing them turns the
  gallery into a file listing.
* **Delete cannot escape the sandbox.** Per-user isolation is the whole reason
  exports are namespaced by account; a delete that follows a path out of that
  folder would undo it.
"""
from __future__ import annotations

import os
import subprocess

import pytest

import app


@pytest.fixture()
def library(workdir, monkeypatch):
    """A workspace with one render per engine, plus scratch that must be hidden."""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    files = [
        ("commentary_1788900001.mp4", "128x228"),
        ("minimalist_1788900002.mp4", "128x228"),
        ("narrative_1788900003.mp4", "128x228"),
        ("batch_1788900004.mp4", "128x228"),
        ("reel_1788900005.mp4", "128x128"),
        ("duel_1788900006.mp4", "128x228"),
        ("atmosphere_1788900007.mp4", "228x128"),
        ("source_1788900008.mp4", "64x64"),            # scratch
        ("narration_1788900009.mp4", "64x64"),         # scratch
        ("muted_1788900010.mp4", "64x64"),             # scratch
        ("notes.txt", None),                           # not a video at all
    ]
    for index, (name, size) in enumerate(files):
        path = os.path.join(workdir, name)
        if size is None:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("not a video")
            continue
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", f"testsrc2=size={size}:rate=12:duration=1", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-an", path],
            capture_output=True, check=True)
        # Distinct mtimes so the sort order is well defined.
        os.utime(path, (1788900000 + index * 60, 1788900000 + index * 60))

    monkeypatch.setattr(app, "user_exports", lambda: workdir)
    return workdir


class TestClassification:

    @pytest.mark.parametrize("name,expected", [
        ("commentary_123.mp4", "🎙️ Commentary"),
        ("minimalist_123.mp4", "◼️ Minimalist"),
        ("narrative_123.mp4", "📖 Narrative"),
        ("batch_123.mp4", "📦 Batch"),
        ("reel_123.mp4", "🎬 Reel"),
        ("duel_123.mp4", "⚔️ Duel"),
        ("atmosphere_123.mp4", "🌙 Atmosphere"),
        ("something_else.mp4", "📄 Other"),
    ])
    def test_the_prefix_identifies_the_engine(self, name, expected):
        assert app.export_kind(name) == expected

    @pytest.mark.parametrize("width,height,expected", [
        (1080, 1920, "9:16"),
        (1920, 1080, "16:9"),
        (1080, 1080, "1:1"),
        (1080, 1350, "4:5"),
        (0, 0, ""),
    ])
    def test_the_aspect_tag(self, width, height, expected):
        assert app.aspect_tag(width, height) == expected

    def test_an_unusual_ratio_reports_itself(self):
        assert app.aspect_tag(2560, 1080) == "2.37:1"


class TestListing:

    def test_scratch_and_non_video_files_are_hidden(self, library):
        names = [item["name"] for item in app.list_exports()]
        assert len(names) == 7, names
        for hidden in ("source_", "narration_", "muted_", "notes"):
            assert not any(n.startswith(hidden) for n in names), f"{hidden} leaked in"

    def test_scratch_can_be_asked_for_explicitly(self, library):
        """The command centre's purge needs to see them even though the
        gallery does not."""
        assert len(app.list_exports(include_scratch=True)) == 10

    @pytest.mark.parametrize("sort,first", [
        ("newest", "atmosphere_1788900007.mp4"),
        ("oldest", "commentary_1788900001.mp4"),
        ("name", "atmosphere_1788900007.mp4"),
    ])
    def test_sorting(self, library, sort, first):
        assert app.list_exports(sort=sort)[0]["name"] == first

    def test_largest_first(self, library):
        sizes = [item["bytes"] for item in app.list_exports(sort="largest")]
        assert sizes == sorted(sizes, reverse=True)

    def test_an_unknown_sort_falls_back_to_newest(self, library):
        assert app.list_exports(sort="nonsense")[0]["name"] == \
            app.list_exports(sort="newest")[0]["name"]

    def test_filtering_by_engine(self, library):
        duels = app.list_exports(kind="⚔️ Duel")
        assert len(duels) == 1
        assert duels[0]["name"].startswith("duel_")

    def test_every_engine_is_represented(self, library):
        kinds = {item["kind"] for item in app.list_exports()}
        assert len(kinds) == 7, kinds
        assert "📄 Other" not in kinds

    def test_an_empty_workspace_is_not_an_error(self, workdir, monkeypatch):
        empty = os.path.join(workdir, "nothing-here")
        monkeypatch.setattr(app, "user_exports", lambda: empty)
        assert app.list_exports() == []


class TestDelete:

    def test_a_file_outside_the_workspace_is_refused(self, library, workdir):
        """Per-user isolation is the reason exports are namespaced by account.
        A delete that follows a path out of that folder would undo it."""
        outsider = os.path.join(os.path.dirname(workdir), "not_mine.mp4")
        with open(outsider, "wb") as handle:
            handle.write(b"someone else's render")
        try:
            problem = app.delete_export(outsider)
            assert problem, "a path outside the user folder was accepted"
            assert os.path.exists(outsider), "it was deleted anyway"
        finally:
            os.remove(outsider)

    def test_a_traversal_path_is_refused(self, library, workdir):
        escape = os.path.join(workdir, "..", "escape.mp4")
        with open(os.path.abspath(escape), "wb") as handle:
            handle.write(b"x")
        try:
            assert app.delete_export(escape), "..\\ escaped the workspace"
            assert os.path.exists(os.path.abspath(escape))
        finally:
            os.remove(os.path.abspath(escape))

    def test_a_real_file_is_removed(self, library, workdir):
        victim = os.path.join(workdir, "batch_1788900004.mp4")
        assert app.delete_export(victim) == ""
        assert not os.path.exists(victim)
        assert len(app.list_exports()) == 6

    def test_deleting_something_already_gone_says_so(self, library, workdir):
        problem = app.delete_export(os.path.join(workdir, "never_existed.mp4"))
        assert "already gone" in problem


class TestDetails:

    def test_resolution_and_duration_are_read_from_the_file(self, library, workdir):
        path = os.path.join(workdir, "atmosphere_1788900007.mp4")
        stat = os.stat(path)
        details = app.export_details(path, stat.st_mtime, stat.st_size)
        assert (details["width"], details["height"]) == (228, 128)
        assert app.aspect_tag(details["width"], details["height"]) == "16:9"
        assert 0.5 < details["duration"] < 2.0

    def test_a_vertical_render_tags_as_nine_sixteen(self, library, workdir):
        path = os.path.join(workdir, "duel_1788900006.mp4")
        stat = os.stat(path)
        details = app.export_details(path, stat.st_mtime, stat.st_size)
        assert app.aspect_tag(details["width"], details["height"]) == "9:16"

    def test_an_unreadable_file_does_not_raise(self, workdir):
        """A truncated or still-being-written file must not take the page down."""
        broken = os.path.join(workdir, "broken.mp4")
        with open(broken, "wb") as handle:
            handle.write(b"not really an mp4")
        details = app.export_details(broken, 0.0, 17)
        assert details["thumb"] is None
        assert details["duration"] == 0.0


class TestPageWiring:

    def test_the_library_is_not_a_production_mode(self):
        """It makes nothing, so it carries no 'best for' card and no ETA."""
        assert "library" in app.MODE_LABELS
        assert "library" not in app.MODE_GUIDES

    def test_the_download_guard_is_set_below_a_gigabyte(self):
        """st.download_button reads the whole file into memory and pushes it
        through the websocket. An 8-hour Atmosphere render is several GB."""
        assert 0 < app.DOWNLOAD_LIMIT_BYTES <= 1024 ** 3

    def test_every_sort_mode_has_a_label(self):
        for key in app.SORT_MODES:
            assert app.SORT_MODES[key].strip()
