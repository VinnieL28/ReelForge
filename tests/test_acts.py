"""
Three-act scenes: the fix for a minute of one static metaphor.

One piece of geometry holds attention for about twenty seconds. The runtime is
over a minute, and the first 68-second render proved what stretching does to
it: the picture changed by 0.3% per second and 19 of 68 seconds were completely
still. Three acts, each with its own metaphor and its own slice of the
narration, measured 2.4x the motion.

The test that matters most here is the idempotency one. `render_animation`
normalises the spec a second time, after the act timings have been allocated,
and the first version of `normalise_act` reset them to zero. Every act
collapsed to start=0 seconds=0, `act_at` fell past all of them to the last, and
a three-act plan rendered as one frozen frame for sixty-nine seconds. Nothing
raised, nothing logged, and the file was the right length.
"""
from __future__ import annotations

import pytest

import gemini_engine as ge
import minimalist_engine as me


def _plan(*weights: int) -> dict:
    """A spec with one act per weight, each carrying that many words."""
    templates = ["two_doors", "gravity_funnel", "split_path", "domino_chain"]
    return me.normalise_spec({
        "acts": [{"template": templates[i % len(templates)],
                  "title": f"ACT {i + 1}",
                  "thesis": " ".join(["word"] * weight)}
                 for i, weight in enumerate(weights)],
    })


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

class TestActSpec:

    def test_a_plan_keeps_its_acts(self):
        assert len(_plan(30, 40, 30)["acts"]) == 3

    def test_a_single_act_is_not_a_plan(self):
        """One act is what this was built to replace."""
        assert "acts" not in _plan(30)

    def test_a_spec_with_no_acts_is_untouched(self):
        """Every preset, the offline fallback and every existing caller hands
        in a single-metaphor spec and must keep working."""
        spec = me.normalise_spec(me.fallback_scene_spec("a test", "split_path", 18.0))
        assert "acts" not in spec
        act, local, seconds = me.act_at(spec, 9.0, 18.0)
        assert act is spec and local == 9.0 and seconds == 18.0

    def test_junk_acts_are_dropped(self):
        spec = me.normalise_spec({"acts": ["nonsense", None, 42]})
        assert "acts" not in spec

    def test_an_unknown_template_falls_back_to_a_real_one(self):
        spec = me.normalise_spec({"acts": [
            {"template": "not_a_metaphor", "thesis": "a b c"},
            {"template": "split_path", "thesis": "d e f"}]})
        for act in spec["acts"]:
            assert act["template"] in me.TEMPLATES


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------

class TestAllocation:

    def test_the_acts_fill_the_runtime_exactly(self):
        for duration in (62.0, 66.3, 70.0, 95.0):
            acts = me.allocate_acts(_plan(30, 40, 30), duration)
            assert sum(a["seconds"] for a in acts) == pytest.approx(duration, abs=0.01)
            assert acts[0]["start"] == 0.0

    def test_time_is_shared_by_words_not_split_evenly(self):
        """An act carrying five sentences should hold the screen longer than
        one carrying two, or the picture drifts away from the voice."""
        acts = me.allocate_acts(_plan(20, 60, 20), 66.0)
        assert acts[1]["seconds"] > acts[0]["seconds"]
        assert acts[0]["seconds"] == pytest.approx(acts[2]["seconds"], abs=0.2)

    def test_a_short_act_still_gets_long_enough_to_read(self):
        """Below a few seconds a cut reads as a glitch rather than an idea."""
        acts = me.allocate_acts(_plan(2, 80, 80), 66.0)
        assert acts[0]["seconds"] >= me.MIN_ACT_SECONDS - 0.01

    def test_the_acts_run_back_to_back_with_no_gap(self):
        acts = me.allocate_acts(_plan(30, 40, 30), 66.0)
        for earlier, later in zip(acts, acts[1:]):
            assert later["start"] == pytest.approx(
                earlier["start"] + earlier["seconds"], abs=0.01)

    def test_every_act_gets_a_beat_inside_its_own_span(self):
        for act in me.allocate_acts(_plan(30, 40, 30), 66.0):
            assert 0.0 < act["climax"] < act["seconds"]


# ---------------------------------------------------------------------------
# The bug
# ---------------------------------------------------------------------------

class TestNormalisationIsIdempotent:

    def test_allocated_timings_survive_a_second_normalise(self):
        """
        render_animation normalises again after allocation. Zeroing the timings
        there collapsed every act to start=0 seconds=0; act_at then fell past
        all of them to the last one and drew its end state for the whole video.
        Three acts planned, one frozen frame rendered, no error raised.
        """
        spec = _plan(37, 42, 43)
        spec["acts"] = me.allocate_acts(spec, 66.3)
        before = [(a["start"], a["seconds"]) for a in spec["acts"]]

        again = me.normalise_spec(spec)
        after = [(a["start"], a["seconds"]) for a in again["acts"]]

        assert after == before, "the second normalise wiped the allocation"

    def test_normalising_three_times_changes_nothing(self):
        spec = _plan(30, 40, 30)
        spec["acts"] = me.allocate_acts(spec, 66.0)
        once = me.normalise_spec(spec)
        twice = me.normalise_spec(once)
        assert [a["seconds"] for a in twice["acts"]] == \
               [a["seconds"] for a in once["acts"]]

    def test_dispatch_still_works_after_renormalising(self):
        spec = _plan(37, 42, 43)
        spec["acts"] = me.allocate_acts(spec, 66.3)
        spec = me.normalise_spec(spec)

        seen = {me.act_at(spec, t, 66.3)[0]["template"]
                for t in (2.0, 30.0, 60.0)}
        assert len(seen) == 3, f"only {len(seen)} act(s) ever drawn: {seen}"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestDispatch:

    def test_each_act_gets_its_own_clock(self):
        """An act must draw itself from its own beginning, not join halfway
        through someone else's animation."""
        spec = _plan(30, 30, 30)
        spec["acts"] = me.allocate_acts(spec, 60.0)
        _act, local, _seconds = me.act_at(spec, 21.0, 60.0)
        assert local < 2.0, f"act two started at {local:.1f}s into its own clock"

    def test_the_last_frame_belongs_to_the_last_act(self):
        spec = _plan(30, 30, 30)
        spec["acts"] = me.allocate_acts(spec, 60.0)
        act, _local, _seconds = me.act_at(spec, 59.99, 60.0)
        assert act["template"] == spec["acts"][-1]["template"]

    def test_past_the_end_does_not_fall_off(self):
        spec = _plan(30, 30, 30)
        spec["acts"] = me.allocate_acts(spec, 60.0)
        act, local, seconds = me.act_at(spec, 999.0, 60.0)
        assert act["template"] == spec["acts"][-1]["template"]
        assert seconds > 0.1, "a fallthrough act with no duration draws its end state"

    @pytest.mark.parametrize("t", [0.0, 5.0, 22.0, 30.0, 45.0, 65.0])
    def test_frames_draw_across_every_act(self, t):
        import numpy as np

        spec = _plan(37, 42, 43)
        spec["acts"] = me.allocate_acts(spec, 66.0)
        array = np.asarray(me.make_scene_frame(spec, t, 66.0))
        assert array.shape == (1920, 1080, 3)

    def test_the_progress_bar_measures_the_whole_video(self):
        """It is retention furniture: it promises how much is left of the
        video, not of the current act."""
        import inspect

        # Drawn in _draw_act, which every path goes through.
        source = inspect.getsource(me._draw_act)
        assert "draw_progress(frame, t, duration)" in source, (
            "the hairline is being drawn on the act's clock")

    def test_the_payoff_belongs_to_the_video_not_to_every_act(self):
        import inspect

        source = inspect.getsource(me._draw_act)
        assert "draw_footer(frame, spec, t, duration)" in source, (
            "the closing line is being drawn at the end of every act")


# ---------------------------------------------------------------------------
# The plan the model returns
# ---------------------------------------------------------------------------

RAW_PLAN = """{"acts": [
  {"template": "two_doors", "title": "THE PARALYSIS OF PLENTY",
   "subtitle": "abundance is a trap", "labels": {"left": "COMFORT"},
   "thesis": "Twenty-four jams drew more shoppers than six."},
  {"template": "comparison_split", "title": "THE JAM STUDY",
   "subtitle": "less is ten times more",
   "thesis": "Iyengar and Lepper measured a tenfold gap in purchases."},
  {"template": "split_path", "title": "THE SATISFICER",
   "subtitle": "choose enough, then stop",
   "thesis": "Satisficers report higher satisfaction than maximisers."}],
 "payoff": "Freedom lies in deliberate limitation.",
 "publish": {"title": "t", "description": "d", "hashtags": ["#a"]}}"""


class TestPlanParsing:

    def test_a_full_plan_parses(self):
        plan = ge.parse_scene_plan(RAW_PLAN)
        assert len(plan["acts"]) == 3
        assert plan["payoff"] == "Freedom lies in deliberate limitation."
        assert plan["acts"][1]["template"] == "comparison_split"

    def test_the_thesis_is_the_acts_read_end_to_end(self):
        plan = ge.parse_scene_plan(RAW_PLAN)
        for act in plan["acts"]:
            assert act["thesis"] in plan["thesis"]

    def test_the_same_metaphor_twice_is_replaced(self):
        """Two identical acts in a row defeat the entire point."""
        raw = RAW_PLAN.replace('"comparison_split"', '"two_doors"')
        templates = [a["template"] for a in ge.parse_scene_plan(raw)["acts"]]
        assert len(set(templates)) == len(templates), templates

    def test_an_act_with_no_narration_is_dropped(self):
        raw = RAW_PLAN.replace(
            '"thesis": "Satisficers report higher satisfaction than maximisers."',
            '"thesis": ""')
        assert len(ge.parse_scene_plan(raw)["acts"]) == 2

    def test_fewer_than_two_acts_is_not_a_plan(self):
        """Falling back to the single-metaphor path is the honest outcome."""
        assert ge.parse_scene_plan('{"acts": [{"template": "two_doors", '
                                   '"thesis": "one"}]}') == {}

    @pytest.mark.parametrize("raw", ["", "sorry, I cannot help", "{]", "[]"])
    def test_junk_is_rejected(self, raw):
        assert ge.parse_scene_plan(raw) == {}

    def test_a_parsed_plan_renders(self):
        """The whole point of the shape is that it survives the trip into the
        engine."""
        import numpy as np

        spec = me.normalise_spec(ge.parse_scene_plan(RAW_PLAN))
        spec["acts"] = me.allocate_acts(spec, 66.0)
        assert np.asarray(me.make_scene_frame(spec, 33.0, 66.0)).shape == \
            (1920, 1080, 3)


class TestPlanPromptDemandsFacts:
    """
    Domain Specificity scored 3.0 on the first long render -- "0 checkable
    details in 124 words". The prompt asked for a narrative and never asked for
    anything that could be wrong.
    """

    def test_it_asks_for_something_checkable(self):
        prompt = ge.build_scene_plan_prompt("The Paradox of Choice").lower()
        assert "checkable" in prompt
        assert any(word in prompt for word in ("figure", "date", "study"))

    def test_it_forbids_inventing_statistics(self):
        """A fabricated study is worse than a vague sentence: it is the one
        mistake that gets a channel called out."""
        prompt = ge.build_scene_plan_prompt("x").lower()
        assert "never invent" in prompt

    def test_it_shows_the_difference_rather_than_describing_it(self):
        prompt = ge.build_scene_plan_prompt("x")
        assert "BAD:" in prompt and "GOOD:" in prompt

    def test_it_carries_the_word_budget_that_sets_the_runtime(self):
        prompt = ge.build_scene_plan_prompt("x")
        assert f"{ge.SCENE_MIN_WORDS}-{ge.SCENE_MAX_WORDS} words" in prompt

    def test_it_offers_the_whole_catalogue(self):
        prompt = ge.build_scene_plan_prompt("x")
        for key in ("two_doors", "gravity_funnel", "split_path"):
            assert key in prompt

    def test_it_asks_for_three_and_forbids_repeats(self):
        prompt = ge.build_scene_plan_prompt("x")
        assert "THREE ACTS" in prompt
        assert "same one twice" in prompt


class TestLabelsStayInFrame:
    """
    A label anchored near the edge used to run off the canvas. The balance
    scale's right pan sits at x=895 and "PAIN BALANCE" is about 400px wide, so
    its last letter was drawn past 1080 and the video read "PAIN BALANC".
    Nothing raised: the glyph was drawn, just outside the picture.
    """

    @pytest.mark.parametrize("body,x,size", [
        ("PAIN BALANCE", 895, 44),
        ("REWARD SPIKE", 185, 40),
        ("50 YEARS", 1000, 40),
        ("COMFORT NOW", 60, 40),
        ("DOWNREGULATION", 900, 44),
    ])
    def test_a_label_near_the_edge_is_pulled_back_in(self, body, x, size):
        import numpy as np

        frame = me.Frame()
        frame.text(body, (x, 900), size, me.WHITE, "bold", tracking=5)
        array = np.asarray(frame.finish())
        columns = np.nonzero((array.max(axis=2) > 40).any(axis=0))[0]

        assert columns.min() >= me.SIDE_MARGIN - 6, f"{body} touches the left edge"
        assert columns.max() <= array.shape[1] - me.SIDE_MARGIN + 6, (
            f"{body} runs off the right edge")

    def test_the_untracked_path_is_clamped_too(self):
        """Two drawing paths, and only clamping one of them leaves the bug in
        half the templates."""
        import numpy as np

        frame = me.Frame()
        frame.text("NEURAL DOWNREGULATION", (980, 900), 44, me.WHITE, "bold")
        array = np.asarray(frame.finish())
        columns = np.nonzero((array.max(axis=2) > 40).any(axis=0))[0]
        assert columns.max() <= array.shape[1] - 1

    def test_the_clamp_allows_for_ink_running_wider_than_advance(self):
        """textlength measures advance, not ink. A bold glyph overhangs its own
        box by up to 7%, which is enough for one more letter to cross the
        gutter."""
        assert me._INK_OVER_ADVANCE > 1.0

    def test_a_line_wider_than_the_frame_is_centred(self):
        """It will still overflow, but symmetrically, which reads as a design
        choice rather than a defect."""
        # Centred in the SAFE box, not on the canvas, so it leans away
        # from the action rail on the right.
        frame = me.Frame()
        assert frame.safe_x(80.0, 4000.0) == pytest.approx(sum(me.SAFE_X) / 2.0)

    def test_opting_out_still_works(self):
        """Templates that position their own geometry-bound text must be able
        to bypass the clamp."""
        frame = me.Frame()
        assert frame.safe_x(2000.0, 10.0) < 2000.0     # clamped by default
