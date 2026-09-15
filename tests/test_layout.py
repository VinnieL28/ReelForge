"""
Text never gets drawn through text.

The report that prompted this file described the fault three different ways --
"counter text overlaps", "labels overlap figures", "doubled/ghosted titles" --
and the first of them had a cause worth writing down.

scene_vessel asked for its counter at y=1478 and its axis caption at y=1552.
At the call site that is two lines 74 pixels apart, which is fine. Both were
below TEXT_SAFE_Y, so Frame.safe_y hoisted each of them to the caption
ceiling independently, and they arrived 21 pixels apart -- 38px type drawn
through 62px type, neither readable. Nothing raised. Every one of 660 tests
passed. It was visible only in a rendered frame.

That is the shape of the whole class: *the numbers a template asks for are not
the numbers that get drawn*, because the safe-area clamps sit in between. So a
check that reads the call sites proves nothing, and a check that counts bright
pixels cannot tell a label from a chart line. What works is asking the frame
what it actually drew, which is what Frame.text_boxes records.
"""
from __future__ import annotations

import pytest

import minimalist_engine as me

# One act's worth of time, sampled across its whole run. The collisions that
# matter are not all present at t=0: a counter climbs into its neighbour, a
# label fades in under a figure that has walked somewhere.
ACT_SECONDS = 11.0
SAMPLES = 22

COPY = {
    "title": "The Illusion Of Cost",
    "subtitle": "Why the obvious answer fails here",
    "thesis": "word " * 24,
}


def _act(template: str, **extra) -> dict:
    act = me.normalise_act({"template": template, **COPY, **extra})
    act.update({"seconds": ACT_SECONDS, "start": 0.0,
                "climax": ACT_SECONDS * 0.75})
    return act


def _boxes_at(template: str, t: float, **extra) -> list:
    act = _act(template, **extra)
    frame = me.Frame()
    me.TEMPLATES[template]["fn"](frame, t, act, ACT_SECONDS)
    me.draw_titles(frame, act, t)
    return frame.text_boxes


def _collisions(boxes: list) -> list[tuple[str, str]]:
    """
    Overlapping pairs, ignoring lines of the same wrapped paragraph.

    A paragraph's lines are set on a leading of 1.28, and the ink bound used
    for these boxes reserves 0.85 of the point size either side of the anchor
    -- a deliberately conservative figure for the gap between two separate
    elements. Applied within a paragraph it reports every normally-set block of
    type as a collision, which is a statement about the bound and not about the
    typography.
    """
    hits = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[5] == b[5]:
                continue
            if me.text_boxes_overlap(a[1:5], b[1:5]):
                hits.append((a[0], b[0]))
    return hits


class TestTheFrameRecordsWhatItDrew:
    """Without this the check has nothing to measure. `text_boxes_overlap`
    already existed and had no boxes to compare."""

    def test_a_drawn_line_is_recorded(self):
        frame = me.Frame()
        frame.text("A LINE", (540, 800), 48, me.WHITE)
        assert len(frame.text_boxes) == 1
        body, x0, y0, x1, y1, _ = frame.text_boxes[0]
        assert body == "A LINE"
        assert x0 < 540 < x1 and y0 < 800 < y1

    def test_an_empty_line_records_nothing(self):
        frame = me.Frame()
        frame.text("", (540, 800), 48, me.WHITE)
        assert frame.text_boxes == []

    def test_a_tracked_line_records_its_full_span(self):
        """Tracking spaces the letters, so the box is wider than the advance
        of the same string set solid."""
        solid, tracked = me.Frame(), me.Frame()
        solid.text("SPREAD OUT", (540, 800), 40, me.WHITE, tracking=0)
        tracked.text("SPREAD OUT", (540, 800), 40, me.WHITE, tracking=8)
        wide = tracked.text_boxes[0][3] - tracked.text_boxes[0][1]
        narrow = solid.text_boxes[0][3] - solid.text_boxes[0][1]
        assert wide > narrow

    def test_it_records_where_the_clamp_put_the_line_not_where_it_was_asked_for(self):
        """The whole reason this is measured on the frame. A line asked for
        below the safe area is drawn above it, and the box has to say so."""
        frame = me.Frame()
        frame.text("BELOW THE RAIL", (540, 1900), 44, me.WHITE)
        y1 = frame.text_boxes[0][4]
        assert y1 <= me.SAFE_Y, "the record kept the asked-for y, not the drawn one"

    def test_a_wrapped_paragraph_records_every_line(self):
        frame = me.Frame()
        frame.wrapped("one two three four five six seven eight nine ten eleven "
                      "twelve thirteen fourteen fifteen sixteen",
                      (540, 800), size=44, max_width=420)
        assert len(frame.text_boxes) >= 3

    def test_the_lines_of_one_paragraph_share_a_group(self):
        """So the check can tell leading from a collision. Without this every
        normally-set paragraph in the engine reports as broken."""
        frame = me.Frame()
        frame.wrapped("one two three four five six seven eight nine ten eleven "
                      "twelve thirteen fourteen", (540, 800), size=44, max_width=420)
        groups = {box[5] for box in frame.text_boxes}
        assert len(groups) == 1, "a paragraph was split across groups"

    def test_two_separate_calls_get_separate_groups(self):
        frame = me.Frame()
        frame.text("FIRST", (300, 700), 40, me.WHITE)
        frame.text("SECOND", (700, 900), 40, me.WHITE)
        assert frame.text_boxes[0][5] != frame.text_boxes[1][5]

    def test_a_paragraph_does_not_report_its_own_leading_as_a_collision(self):
        frame = me.Frame()
        frame.wrapped("one two three four five six seven eight nine ten eleven "
                      "twelve thirteen fourteen", (540, 800), size=44, max_width=420)
        assert _collisions(frame.text_boxes) == []


class TestNoTemplateDrawsTextThroughText:

    @pytest.mark.parametrize("template", me.METAPHOR_TYPES)
    def test_across_the_whole_act(self, template):
        for step in range(1, SAMPLES + 1):
            t = step * ACT_SECONDS / SAMPLES
            hits = _collisions(_boxes_at(template, t))
            assert not hits, (
                f"{template} at t={t:.1f}s drew "
                + "; ".join(f"{a!r} through {b!r}" for a, b in hits))

    def test_the_vessel_is_clear_in_both_directions(self):
        """It is the one template whose readout changes length as it runs --
        '1.00x' to '37.78x', '100%' to '0%' -- so the pair that collided is
        also the pair most likely to collide again."""
        for direction, axis in (("fill", "d"), ("drain", "y")):
            for step in range(1, SAMPLES + 1):
                t = step * ACT_SECONDS / SAMPLES
                hits = _collisions(_boxes_at(
                    "compounding_jar", t, direction=direction,
                    axis_suffix=axis, axis_max=30 if axis == "y" else 365))
                assert not hits, (hits, direction, t)

    def test_the_check_would_have_caught_the_bug_it_was_written_for(self):
        """
        Not a tautology: it pins that the detector fires on the exact geometry
        that shipped. 38px and 62px type 21px apart, which is what the clamp
        produced from y=1478 and y=1552.
        """
        frame = me.Frame()
        frame.text("DAY 185", (540, 1302), 38, me.GREY)
        frame.text("6.30x", (540, 1281), 62, me.WHITE)
        assert _collisions(frame.text_boxes), (
            "two lines 21px apart at 38px and 62px read as clear")


class TestTitlesNeverShareAFrame:
    """
    The other half of the report's overlap complaint, and the one that looked
    worst: at 19.0s of the delivered file two act titles ghosted through each
    other because the handover was a cross-dissolve. It is guaranteed by the
    compositor now rather than by the templates being careful, so this is where
    that guarantee is checked end to end.
    """

    @staticmethod
    def _plan(count: int = 3) -> dict:
        spec = me.normalise_spec({
            "title": "T", "payoff": "P", "duration": 66.0,
            "acts": [{"template": key, "title": f"ACT {i}",
                      "subtitle": "a subtitle", "thesis": "word " * 24}
                     for i, key in enumerate(me.METAPHOR_TYPES[:count], 1)],
        })
        spec["acts"] = me.allocate_acts(spec, 66.0)
        return spec

    def test_no_frame_in_a_handover_is_lit_by_two_acts(self):
        import numpy as np

        spec = self._plan()
        for act in spec["acts"][1:]:
            edge = float(act["start"])
            for step in range(int(me.ACT_OVERLAP / 0.05) + 1):
                t = edge + step * 0.05
                previous = me._outgoing_act(spec, t)
                if previous is None:
                    continue
                older, older_t, older_s, blend = previous
                under = me._draw_act(spec, older, older_t, older_s, t, 66.0)
                over = me._draw_act(spec, act, t - edge,
                                    float(act["seconds"]), t, 66.0)
                mixed = me._mix_frames(under, over, blend).astype(np.int32)
                # The mix can only be a scaling of one of its two inputs.
                from_under = np.abs(mixed - under.astype(np.int32)).max()
                from_over = np.abs(mixed - over.astype(np.int32)).max()
                lit_by_both = (mixed > 8).sum() and from_under and from_over
                if lit_by_both:
                    scaled_under = (mixed.max() <= under.astype(np.int32).max() + 1)
                    scaled_over = (mixed.max() <= over.astype(np.int32).max() + 1)
                    assert scaled_under or scaled_over, (
                        f"the frame at {t:.2f}s is brighter than either act")

    def test_the_closing_card_never_shares_the_frame_with_a_scene(self):
        """The same fault, at the end of the video: the payoff was drawn across
        two lit doorways with the CTA on top of a label."""
        import numpy as np

        spec = self._plan()
        duration = 66.0
        for step in range(int(me.CLOSING_FADE / 0.05) + 2):
            t = duration - me.CLOSING_SECONDS + step * 0.05
            closing = me.closing_alpha_at(t, duration)
            scene = max(0.0, 1.0 - closing / me.TRANSITION_BLACK_AT)
            card = max(0.0, (closing - me.TRANSITION_BLACK_AT)
                       / (1.0 - me.TRANSITION_BLACK_AT))
            assert scene == 0.0 or card == 0.0, (
                f"at {t:.2f}s the scene is at {scene:.2f} and the card at {card:.2f}")
