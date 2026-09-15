"""
The seven render defects found by going through an output frame by frame.

Every number in here was measured on a real render before and after, because
five of the seven were invisible to the test suite: the file was the right
length, in sync, and passed every existing assertion while being visibly
broken.

  * Black flash at act boundaries -- 5 near-black frames, mean brightness
    falling 26.8 -> 1.9 and climbing back.
  * The closing caption composited over a chart label.
  * Acts long enough that each finished its reveals and then held.
  * No ending: final frame at a flat mean of 21.0 with audio at full level.
  * -22.8 LUFS integrated against a platform target near -14.
  * Content at x 78-1000, y 249-1694, under the action rail and the scrubber.
  * A subtitle reading "why decision fails".
"""
from __future__ import annotations

import pytest

import minimalist_engine as me
import video_engine as ve


def _plan(*weights: int) -> dict:
    templates = ["two_doors", "gravity_funnel", "split_path", "balance_scale"]
    spec = me.normalise_spec({
        "payoff": "CHOOSE YOUR DISCOMFORT",
        "acts": [{"template": templates[i % len(templates)],
                  "title": f"ACT {i + 1}", "subtitle": "a subtitle",
                  "thesis": " ".join(["word"] * w)}
                 for i, w in enumerate(weights)],
    })
    spec["acts"] = me.allocate_acts(spec, 64.0)
    return spec


# ---------------------------------------------------------------------------
# BUG 1 -- the black flash
# ---------------------------------------------------------------------------

class TestActCrossfade:

    def test_the_acts_overlap(self):
        assert me.ACT_OVERLAP > 0

    def test_both_acts_are_drawn_across_a_boundary(self):
        spec = _plan(30, 30, 30)
        edge = float(spec["acts"][1]["start"])
        assert me._outgoing_act(spec, edge + 0.01) is not None
        assert me._outgoing_act(spec, edge + me.ACT_OVERLAP + 0.5) is None

    def test_the_blend_runs_from_nothing_to_everything(self):
        spec = _plan(30, 30, 30)
        edge = float(spec["acts"][1]["start"])
        early = me._outgoing_act(spec, edge + 0.01)[3]
        late = me._outgoing_act(spec, edge + me.ACT_OVERLAP - 0.01)[3]
        assert early < 0.1 and late > 0.9

    def test_the_boundary_is_not_a_black_flash(self):
        """
        The frame at the cut must be at least as bright as the outgoing act
        was, because the outgoing picture is still on screen underneath.
        Measured before this: brightness fell to 1.9 for five frames.
        """
        import numpy as np

        spec = _plan(30, 30, 30)
        edge = float(spec["acts"][1]["start"])
        before = np.asarray(me.make_scene_frame(spec, edge - 0.2, 64.0)).mean()
        at_cut = np.asarray(me.make_scene_frame(spec, edge + 0.05, 64.0)).mean()
        assert at_cut > before * 0.5, (
            f"the cut dropped from {before:.1f} to {at_cut:.1f}")

    def test_a_single_metaphor_scene_has_nothing_to_blend(self):
        spec = me.normalise_spec(me.fallback_scene_spec("x", "split_path", 18.0))
        assert me._outgoing_act(spec, 9.0) is None

    def test_mixing_goes_through_black_rather_than_across(self):
        """
        A cross-dissolve was the wrong tool. These frames are white line art on
        pure black, so blending two of them does not merge them -- it shows
        both. At 19.0s of minimalist_1789502356.mp4 two titles ghosted through
        each other and act 3's figure stood inside act 2's beaker.

        The property that replaces it: at no point do both acts contribute
        light.
        """
        import numpy as np

        under = np.full((4, 4, 3), 200, np.uint8)
        over = np.full((4, 4, 3), 200, np.uint8)

        assert me._mix_frames(under, over, 0.0).mean() == pytest.approx(200, abs=1)
        assert me._mix_frames(under, over, 1.0).mean() == pytest.approx(200, abs=1)
        # Two full-brightness frames that would sum to 200 under a dissolve.
        assert me._mix_frames(under, over, 0.5).mean() == pytest.approx(0, abs=1)

    def test_only_one_act_is_ever_lit(self):
        """The guarantee stated as the compositor's own invariant, so no
        template has to be careful about it."""
        import numpy as np

        lit = np.full((2, 2, 3), 255, np.uint8)
        dark = np.zeros((2, 2, 3), np.uint8)
        for step in range(21):
            blend = step / 20.0
            from_under = me._mix_frames(lit, dark, blend).mean()
            from_over = me._mix_frames(dark, lit, blend).mean()
            assert from_under < 1 or from_over < 1, (
                f"both acts contributed light at blend {blend:.2f}")

    def test_the_handover_has_no_step_in_it(self):
        """A dip is a transition; a jump is a glitch. Measured across the whole
        window, no single frame may move more than a fifth of full scale."""
        import numpy as np

        spec = _plan(30, 30, 30)
        edge = float(spec["acts"][1]["start"])
        frames = [np.asarray(me.make_scene_frame(spec, edge - 0.3 + i * 0.05, 64.0)).mean()
                  for i in range(int((me.ACT_OVERLAP + 0.6) / 0.05))]
        steps = [abs(b - a) for a, b in zip(frames, frames[1:])]
        assert max(steps) < 51.0, f"largest single-frame step was {max(steps):.1f}"


def _card(spec):
    frame = me.Frame()
    me.draw_closing_card(frame, spec, 1.0)
    return frame.finish()


# ---------------------------------------------------------------------------
# BUG 2 -- the caption band
# ---------------------------------------------------------------------------

class TestCaptionBand:

    def test_the_band_sits_inside_the_safe_box(self):
        assert me.SAFE_Y_BOX[0] < me.CAPTION_BAND[0] < me.CAPTION_BAND[1]
        assert me.CAPTION_BAND[1] <= me.TEXT_SAFE_Y

    @pytest.mark.parametrize("size", [32, 40, 52])
    def test_an_annotation_is_lifted_out_of_the_band(self, size):
        """This is the invariant the fix rests on: a label can ask for a y
        inside the band and will not get it."""
        frame = me.Frame()
        placed = frame.safe_y(me.CAPTION_BAND[1] - 10.0, size)
        assert placed + size * 0.85 <= me.CAPTION_BAND[0] + 0.5

    def test_the_caption_itself_keeps_the_band(self):
        frame = me.Frame()
        frame.reserved = True
        placed = frame.safe_y(me.TEXT_SAFE_Y, 52)
        assert placed > me.CAPTION_BAND[0]

    def test_the_annotation_layer_ducks_when_the_caption_arrives(self):
        assert me.annotation_alpha_at(10.0, 64.0) == 1.0
        assert me.annotation_alpha_at(64.0 - me.CAPTION_LEAD_SECONDS, 64.0) == 1.0
        mid = me.annotation_alpha_at(
            64.0 - me.CAPTION_LEAD_SECONDS + me.CAPTION_DUCK_SECONDS / 2, 64.0)
        assert 0.2 < mid < 0.8
        assert me.annotation_alpha_at(63.0, 64.0) == 0.0

    def test_the_duck_starts_exactly_when_the_closing_card_does(self):
        """Two different leads would either fade the labels early or leave them
        under the first word."""
        assert me.CAPTION_LEAD_SECONDS == me.CLOSING_SECONDS

    def test_the_closing_card_is_held_long_enough_to_act_on(self):
        """It is the only beat in the video that asks for something. Fully up,
        perfectly still, for at least four seconds."""
        hold = me.CLOSING_SECONDS - me.CLOSING_FADE - me.END_FADE_SECONDS
        assert hold == pytest.approx(4.0, abs=0.05) or hold > 4.0, (
            f"the payoff is only still for {hold:.1f}s")

    def test_the_closing_card_sits_inside_the_safe_area(self):
        """Centred in open frame rather than on the bottom rail -- but the rail
        is still the rail."""
        import numpy as np

        frame = me.Frame()
        me.draw_closing_card(frame, {"payoff": "STOP PAYING TO UNDERPERFORM",
                                     "cta": "Follow for more"}, 1.0)
        rows = np.asarray(frame.finish()).mean(axis=(1, 2))
        lit = np.nonzero(rows > 2)[0]
        assert lit.size, "the closing card drew nothing"
        assert lit.min() >= me.SAFE_Y_BOX[0], lit.min()
        assert lit.max() <= me.SAFE_Y, lit.max()

    def test_every_closing_card_asks_for_something(self):
        """
        Not "the CTA renders when supplied" -- a spec that carries no CTA falls
        back to the default rather than ending on a full stop, so the ask is
        not something a model can forget to include.
        """
        import numpy as np

        supplied = np.asarray(_card({"payoff": "A LINE", "cta": "Subscribe now"}))
        defaulted = np.asarray(_card({"payoff": "A LINE"}))
        payoff_only = np.asarray(_card({"payoff": "A LINE", "cta": "", "_": 1}))

        assert defaulted.sum() > 0
        # The default is drawn whether the key is missing or empty.
        assert defaulted.sum() == payoff_only.sum()
        # And a real CTA replaces it rather than being ignored.
        assert supplied.sum() != defaulted.sum()

    def test_the_default_cta_is_the_one_the_engine_names(self):
        assert me.DEFAULT_CTA and not me.DEFAULT_CTA.endswith(".")

    def test_overlapping_boxes_are_detected(self):
        assert me.text_boxes_overlap((0, 0, 100, 50), (50, 20, 150, 70))
        assert me.text_boxes_overlap((0, 0, 100, 50), (10, 10, 20, 20))
        assert not me.text_boxes_overlap((0, 0, 100, 50), (100, 0, 200, 50))
        assert not me.text_boxes_overlap((0, 0, 100, 50), (0, 50, 100, 100))

    def test_the_caption_cannot_share_its_band_with_a_label(self):
        """
        Checked through the placement API rather than by counting pixels in the
        band: bright geometry passes through there by design -- a curve behind
        the caption is the picture -- so pixels cannot tell a label from a
        chart line. The placement can.
        """
        frame = me.Frame()

        # Where the caption goes.
        frame.reserved = True
        caption_y = frame.safe_y(me.TEXT_SAFE_Y, 52)
        caption = (me.SAFE_X[0], caption_y - 52 * 0.85,
                   me.SAFE_X[1], caption_y + 52 * 0.85)

        # Where a label asking for the same place actually lands.
        frame.reserved = False
        for size in (32, 40, 44, 52):
            label_y = frame.safe_y(caption_y, size)
            label = (me.SAFE_X[0], label_y - size * 0.85,
                     me.SAFE_X[1], label_y + size * 0.85)
            assert not me.text_boxes_overlap(caption, label), (
                f"a {size}px label still lands on the caption")

    def test_a_label_that_ducked_puts_no_ink_on_screen(self):
        """The other half: once the caption is up the annotation layer is
        scaled to nothing, so even a label drawn in a legal position is gone."""
        import numpy as np

        frame = me.Frame()
        frame.annotation_alpha = me.annotation_alpha_at(63.5, 64.0)
        frame.text("REWARD SPIKE", (500, 1200), 44, me.WHITE, "bold")
        array = np.asarray(frame.finish())
        assert int((array.max(axis=2) > 40).sum()) == 0


# ---------------------------------------------------------------------------
# BUG 3 -- act length
# ---------------------------------------------------------------------------

class TestActLength:

    def test_there_is_a_cap(self):
        assert me.MIN_ACT_SECONDS < me.MAX_ACT_SECONDS <= 12.0

    def test_the_act_count_comes_from_the_runtime(self):
        """Three acts over a 66-second narration is 22 seconds each: six of
        animation and sixteen of a held picture."""
        assert me.plan_act_count(24.0) == 2
        assert me.plan_act_count(66.0) == 6
        assert me.plan_act_count(74.0) == 7

    def test_it_never_asks_for_a_silly_number(self):
        assert me.plan_act_count(1.0) >= 2
        assert me.plan_act_count(100000.0) <= 8

    def test_overflow_is_recorded_rather_than_clipped(self):
        """The acts have to cover the narration -- shortening them would end
        the picture before the voice. The overflow is the signal that the plan
        needed more acts."""
        acts = me.allocate_acts(_plan(30, 30, 30), 64.0)
        assert all("over_cap" in act for act in acts)
        assert sum(a["seconds"] for a in acts) == pytest.approx(64.0, abs=0.01)

    def test_enough_acts_means_no_overflow(self):
        acts = me.allocate_acts(_plan(*([20] * 6)), 64.0)
        assert max(a["over_cap"] for a in acts) == 0.0


# ---------------------------------------------------------------------------
# BUG 4 -- an ending
# ---------------------------------------------------------------------------

class TestEnding:

    def test_there_is_an_end_fade(self):
        assert me.END_FADE_SECONDS > 0

    def test_both_picture_and_sound_fade_together(self):
        import inspect

        source = inspect.getsource(me.render_animation)
        assert "vfx.FadeOut" in source, "the picture does not fade"
        assert "afx.AudioFadeOut" in source, "the sound does not fade"
        assert source.count("fade > 0") >= 2, "one of the two is unguarded"

    def test_a_very_short_clip_is_not_all_fade(self):
        import inspect

        source = inspect.getsource(me.render_animation)
        assert "duration * 0.25" in source, (
            "a 2s test render would be a quarter-second of picture and a fade")


# ---------------------------------------------------------------------------
# BUG 5 -- loudness
# ---------------------------------------------------------------------------

class TestLoudness:

    def test_the_targets_are_named_constants(self):
        assert ve.TARGET_LUFS == -14.0
        assert ve.TARGET_TRUE_PEAK_DB == -1.5
        assert ve.TARGET_LRA > 0

    def test_the_limiter_is_asked_for_less_than_the_ceiling(self):
        """Requesting exactly -1.5 produced -1.2 in the finished file: the
        limiter runs before the AAC encode, which adds its own intersample
        peaks."""
        assert ve._TRUE_PEAK_HEADROOM_DB > 0

    def test_it_measures_before_it_corrects(self):
        """Single-pass loudnorm works from a running estimate and lands near
        the target: measured -15.6 LUFS against a -14 request."""
        import inspect

        source = inspect.getsource(ve.normalise_loudness)
        assert "_loudnorm_measure" in source
        assert "measured_I" in source

    def test_a_failed_measurement_leaves_the_file_alone(self):
        """A video that is too quiet is a worse video. A video that is gone is
        a lost render."""
        import inspect

        source = inspect.getsource(ve.normalise_loudness)
        assert '"applied": False' in source
        assert source.count("return {") >= 3

    def test_silence_is_not_normalised(self):
        """loudnorm reports -inf for a silent track and the filter then fails
        on the value it was given."""
        import inspect

        source = inspect.getsource(ve._loudnorm_measure)
        assert "inf" in source

    def test_an_unmeasurable_file_reports_zeroes(self, tmp_path):
        broken = tmp_path / "not-a-video.mp4"
        broken.write_bytes(b"nope")
        assert ve.measure_loudness(str(broken))["lufs"] == 0.0

    def test_the_video_stream_is_copied_not_re_encoded(self):
        import inspect

        source = inspect.getsource(ve.normalise_loudness)
        assert '"-c:v", "copy"' in source


# ---------------------------------------------------------------------------
# BUG 6 -- the safe box
# ---------------------------------------------------------------------------

class TestSafeBox:

    def test_the_box_is_the_one_specified(self):
        assert me.SAFE_X == (100.0, 900.0)
        assert me.SAFE_Y_BOX == (320.0, 1500.0)

    def test_the_right_gutter_clears_the_action_rail(self):
        """The rail is on the right and about 180px wide, which is why the box
        is not symmetric."""
        assert me.CANVAS[0] - me.SAFE_X[1] >= 180

    def test_the_progress_bar_is_above_the_safe_line(self):
        assert me.PROGRESS_Y < me.SAFE_Y_BOX[1]

    def test_type_never_reaches_the_progress_bar(self):
        assert me.TEXT_SAFE_Y < me.PROGRESS_Y

    @pytest.mark.parametrize("x", [-500, 0, 40, 540, 1040, 2000])
    def test_text_is_clamped_into_the_box_horizontally(self, x):
        import numpy as np

        frame = me.Frame()
        frame.text("REWARD SPIKE", (x, 900), 44, me.WHITE, "bold", tracking=5)
        array = np.asarray(frame.finish())
        columns = np.nonzero((array.max(axis=2) > 40).any(axis=0))[0]
        assert columns.min() >= me.SAFE_X[0] - 8, f"x={x} spilled left"
        assert columns.max() <= me.SAFE_X[1] + 8, f"x={x} spilled right"

    @pytest.mark.parametrize("y", [0, 100, 250, 900, 1600, 1900])
    def test_text_is_clamped_into_the_box_vertically(self, y):
        import numpy as np

        frame = me.Frame()
        frame.text("REWARD SPIKE", (500, y), 44, me.WHITE, "bold")
        array = np.asarray(frame.finish())
        rows = np.nonzero((array.max(axis=2) > 40).any(axis=1))[0]
        assert rows.min() >= me.SAFE_Y_BOX[0] - 8, f"y={y} spilled above"
        assert rows.max() <= me.SAFE_Y_BOX[1] + 8, f"y={y} spilled below"


# ---------------------------------------------------------------------------
# BUG 7 -- the copy
# ---------------------------------------------------------------------------

class TestCopyRules:
    """
    "why decision fails" is not in any source file -- it was written by the
    model for that render and will differ on the next one. The only durable
    lever is the prompt.
    """

    def test_the_string_is_not_hardcoded_anywhere(self):
        import glob

        for path in glob.glob("*.py"):
            source = open(path, encoding="utf-8").read().lower()
            assert "why decision fails" not in source, path

    def test_the_prompt_demands_subject_verb_agreement(self):
        import gemini_engine as ge

        prompt = ge.build_scene_plan_prompt("Dopamine Detox")
        assert "Subject and verb must agree" in prompt
        assert "why decisions fail" in prompt
