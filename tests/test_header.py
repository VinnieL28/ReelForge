"""
The header: one per page, a subtitle that names the mode, and type that is not
clipped.

The clipping was a CSS arithmetic mistake worth pinning. `.rf-title` is filled
with `background-clip: text`, so the only ink is inside the line box -- there
is no overflow to spill. At 1.85rem with `line-height: 1.1` the box is ~32.6px
around glyphs that need ~35.5px, and the ascenders were simply cut off.
"""
from __future__ import annotations

import re

import pytest

import app


def _css() -> str:
    return app.THEME_CSS


def _rule(selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{(.*?)\}", _css(), re.DOTALL)
    assert match, f"{selector} is not in the stylesheet"
    return match.group(1)


class TestSubtitles:

    def test_every_mode_has_one(self):
        assert set(app.MODE_SUBTITLES) == set(app.MODE_LABELS), (
            "a mode without a subtitle falls back to an empty header line")

    @pytest.mark.parametrize("mode", list(app.MODE_LABELS))
    def test_each_subtitle_describes_its_own_engine(self, mode):
        text = app.MODE_SUBTITLES[mode]
        assert text.strip() and text.endswith("."), mode
        assert 40 < len(text) < 200, f"{mode}: {len(text)} characters"

    def test_the_subtitles_are_distinct(self):
        """The whole point is that the header stops describing the Commentary
        Machine on every page."""
        assert len(set(app.MODE_SUBTITLES.values())) == len(app.MODE_SUBTITLES)

    @pytest.mark.parametrize("mode,phrase", [
        ("commentary", "visual beats"),
        ("minimalist", "zero copyright risk"),
        ("narrative", "consistent characters"),
        ("batch", "in bulk"),
        ("reel", "kinetic subtitles"),
        ("duel", "split-screen"),
        ("atmosphere", "multi-hour ambient"),
        ("library", "reveal in explorer"),
        ("dashboard", "every engine"),
    ])
    def test_the_wording_matches_what_the_mode_does(self, mode, phrase):
        assert phrase in app.MODE_SUBTITLES[mode].lower()


class TestHeaderLayout:

    def test_the_container_has_top_breathing_room(self):
        rule = _rule(".block-container")
        assert "padding-top: 2.5rem !important" in rule, rule

    def test_the_logotype_line_box_fits_its_glyphs(self):
        """line-height 1.1 on a background-clip:text fill is what clipped the
        ascenders: no ink exists outside the line box to overflow."""
        rule = _rule(".rf-title")
        line_height = float(re.search(r"line-height:\s*([\d.]+)", rule).group(1))
        assert line_height >= 1.25, f"line-height {line_height} still clips"
        assert "display: inline-block" in rule, (
            "background-clip:text needs a block box to paint the full letterform")

    @pytest.mark.parametrize("selector", [".rf-brand", ".rf-title", ".rf-logo", ".rf-sub"])
    def test_no_negative_offsets_on_any_header_element(self, selector):
        rule = _rule(selector)
        assert "margin-top: -" not in rule, f"{selector} pulls itself up off-screen"
        assert "translateY(-" not in rule, f"{selector} transforms itself up off-screen"
        assert not re.search(r"(?<!-)\btop:\s*-", rule), f"{selector} has a negative top"

    def test_the_only_negative_transforms_are_hover_lifts(self):
        """A 1px lift on :hover is a button micro-interaction, not a layout
        offset -- it must survive a sweep for negative transforms."""
        for match in re.finditer(r"([^{}]+)\{([^}]*translateY\(-[^}]*)\}", _css()):
            selector = match.group(1).strip().splitlines()[-1].strip()
            assert ":hover" in selector, f"{selector} translates up outside a hover"
            assert "translateY(-1px)" in match.group(2), selector


class TestOneHeaderPerPage:

    def test_only_the_main_header_draws_the_brand_block(self):
        """Batch Studio used to draw a second brand header under the first, so
        that page showed two logos and two titles."""
        source = open(app.__file__, encoding="utf-8").read()
        # One in the stylesheet, one in main(). Anything else is a duplicate.
        drawn = re.findall(r"'<div class=\"rf-brand\">", source)
        assert len(drawn) == 1, f"{len(drawn)} places draw a brand header"
