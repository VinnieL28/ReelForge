"""
The round that came out of a frame-by-frame review of
minimalist_1789509234.mp4 -- a short about fund fees that rendered with
WHO YOU ARE / WHO YOU THINK, COMFORT / THE HARD ONE and BASIC / HARD / SMART
stamped on four of its six scenes, and scored 8.8.

Every test here is pinned to something that actually shipped.
"""
from __future__ import annotations

import pytest

import compliance
import gemini_engine as ge
import minimalist_engine as me


# ---------------------------------------------------------------------------
# The labels
# ---------------------------------------------------------------------------

class TestThePromptListsTheRealSlots:
    """
    The root cause, and it was not the instruction.

    The slot list was built from SCENE_METAPHORS, which carries "label" and
    "suits" and has never had a "labels" key, so the comprehension filtered
    everything out and the prompt read "Which slots exist: (none)". The model
    was shown an empty list and then told to fill it in. Adding FILL THE LABELS
    FOR EVERY ACT the round before changed nothing, because nothing was missing
    from the asking.
    """

    def test_the_slot_table_is_not_empty(self):
        assert ge.LABEL_SLOTS_FOR["delusion_mirror"]
        assert set(ge.LABEL_SLOTS_FOR["delusion_mirror"]) == {"real", "imagined", "gap"}

    def test_every_labelled_template_reaches_the_prompt(self):
        prompt = ge.build_scene_plan_prompt("why index funds beat active management")
        for key, wanted in ge.LABEL_SLOTS_FOR.items():
            if not wanted:
                continue
            for slot in wanted:
                assert f'"{slot}"' in prompt, f"{key}.{slot} is not in the prompt"

    def test_the_prompt_never_says_there_are_no_slots(self):
        assert "(none)" not in ge.build_scene_plan_prompt("x")

    def test_the_slots_come_with_an_example_of_the_kind_of_phrase(self):
        """A slot named "near" says nothing about what goes in it."""
        prompt = ge.build_scene_plan_prompt("x")
        assert '"near" (e.g.' in prompt


class TestAPlannedActNeverBorrowsPlaceholderCopy:
    """
    The defaults are written for the metaphor, not for any subject. In a preset
    render that is a complete short with no model needed. In a planned act it
    is a confident statement about something else.
    """

    def test_a_planned_act_with_no_label_draws_nothing(self):
        act = me.normalise_act({"template": "delusion_mirror", "planned": True,
                                "title": "T", "thesis": "x"})
        assert me.label(act, "real") == ""

    def test_a_preset_still_gets_its_own_copy(self):
        """Every preset, the offline fallback and every existing caller."""
        act = me.normalise_act({"template": "delusion_mirror",
                                "title": "T", "thesis": "x"})
        assert me.label(act, "real") == "WHO YOU ARE"

    def test_a_supplied_label_always_wins(self):
        act = me.normalise_act({"template": "delusion_mirror", "planned": True,
                                "labels": {"real": "what you own"}, "thesis": "x"})
        assert me.label(act, "real") == "WHAT YOU OWN"

    def test_a_callers_own_fallback_still_applies(self):
        """scene_vessel derives the counter's unit from the axis suffix."""
        act = me.normalise_act({"template": "compounding_jar", "planned": True,
                                "thesis": "x"})
        assert me.label(act, "unit", fallback="YEAR") == "YEAR"

    def test_the_flag_survives_normalisation_twice(self):
        """render_animation normalises the spec a second time."""
        act = me.normalise_act(me.normalise_act(
            {"template": "two_doors", "planned": True, "thesis": "x"}))
        assert act["planned"] is True
        assert me.label(act, "left") == ""


class TestUnlabelledActsAreNoticed:

    RAW = """{"acts": [
      {"template": "two_doors", "title": "A", "subtitle": "s",
       "labels": {}, "thesis": "one"},
      {"template": "compounding_jar", "title": "B", "subtitle": "s",
       "labels": {"unit": "YEAR", "meter": "%"}, "thesis": "two"},
      {"template": "delusion_mirror", "title": "C", "subtitle": "s",
       "labels": {}, "thesis": "three"}
     ], "payoff": "p"}"""

    def test_the_parser_reports_them(self):
        plan = ge.parse_scene_plan(self.RAW, acts=3)
        assert plan["unlabelled"] == ["A", "C"]

    def test_a_fully_labelled_plan_reports_none(self):
        raw = self.RAW.replace('"labels": {}', '"labels": {"left": "X", "right": "Y", '
                                               '"through": "Z", "real": "P", '
                                               '"imagined": "Q", "gap": "R"}')
        assert ge.parse_scene_plan(raw, acts=3)["unlabelled"] == []

    def test_every_act_of_a_plan_is_marked_planned(self):
        plan = ge.parse_scene_plan(self.RAW, acts=3)
        assert all(act["planned"] for act in plan["acts"])

    def test_the_retry_note_names_the_acts(self):
        note = ge.relabel_note(["A", "C"])
        assert '"A"' in note and '"C"' in note
        assert "placeholder" in note.lower()

    def test_the_retry_is_wired_into_the_planner(self):
        import inspect

        source = inspect.getsource(ge.generate_scene_plan)
        assert "unlabelled" in source and "relabel_note" in source


# ---------------------------------------------------------------------------
# The vessel's curve
# ---------------------------------------------------------------------------

class TestTheDrainFollowsRealFeeDrag:
    """
    "At year 24 it shows 75%, but real compounding at those rates gives about
    64%. The drain eases in rather than following the actual curve."

    A drain is the ratio of two compounding series -- what you keep against
    what you would have kept -- and that ratio is ((1+r-f)/(1+r))^t, which is
    exponential decay. The fill's own 1.01^n curve agrees with it only at the
    ends.
    """

    @staticmethod
    def _level(u: float, end_value: float = 0.56) -> float:
        return end_value ** u

    def test_it_lands_where_the_plan_says(self):
        assert self._level(1.0) == pytest.approx(0.56, abs=0.001)
        assert self._level(0.0) == pytest.approx(1.0, abs=0.001)

    @pytest.mark.parametrize("year", [6, 12, 18, 24, 30])
    def test_it_tracks_a_two_percent_fee_on_seven_percent_returns(self, year):
        """Within one percentage point at every year, not just at the ends."""
        truth = (1.05 / 1.07) ** year
        drawn = self._level(year / 30.0, end_value=(1.05 / 1.07) ** 30)
        assert drawn == pytest.approx(truth, abs=0.01), year

    def test_the_old_curve_was_wrong_in_the_middle(self):
        """Pinned so the shape is not quietly reverted to the fill's."""
        growth = 1.01 ** (0.8 * 365)
        ceiling = 1.01 ** 365
        old = 1.0 - (1.0 - 0.56) * ((growth - 1.0) / (ceiling - 1.0))
        assert old > 0.70, "the old interpolation did not overstate year 24"
        assert self._level(0.8) < 0.65

    def test_the_engine_uses_it(self):
        import inspect

        source = inspect.getsource(me.scene_vessel)
        assert "end ** u" in source


# ---------------------------------------------------------------------------
# Frame rate
# ---------------------------------------------------------------------------

class TestTheFrameRateFloor:
    """
    Changing `st.session_state.get("render_fps") or 30` did nothing, because
    the sidebar writes the key on every run and a st.pills widget whose key
    already exists ignores its default. The floor belongs to the engine that
    needs it.
    """

    def test_nothing_renders_below_thirty(self):
        assert me.render_fps(24) == 30
        assert me.render_fps(0) == 30
        assert me.render_fps(None) == 30

    def test_a_higher_request_is_honoured(self):
        assert me.render_fps(60) == 60

    def test_rubbish_does_not_raise(self):
        assert me.render_fps("nonsense") == 30

    def test_both_entry_points_apply_it(self):
        import inspect

        for fn in (me.render_animation, me.build_minimalist_video):
            assert "fps = render_fps(fps)" in inspect.getsource(fn), fn.__name__


# ---------------------------------------------------------------------------
# The tail
# ---------------------------------------------------------------------------

class TestTheCardWaitsForTheNarration:
    """
    Runtime was `spoken + 1.6` and the closing card owns the last
    CLOSING_SECONDS, so on a 74.33s render the card came up at 68.7s while the
    narration ran to 72.7 -- act 6's geometry wiped off the screen with four
    seconds of its own thesis still to go.
    """

    def test_the_tail_is_at_least_as_long_as_the_card(self):
        import inspect

        source = inspect.getsource(me.build_minimalist_video)
        assert "tail = CLOSING_SECONDS + 0.4" in source
        assert "spoken + tail" in source

    def test_the_old_tail_would_have_overlapped(self):
        assert 1.6 < me.CLOSING_SECONDS, "the old tail was longer than the card"

    def test_a_74_second_render_now_clears_it(self):
        spoken = 72.7
        duration = spoken + me.CLOSING_SECONDS + 0.4
        assert duration - me.CLOSING_SECONDS >= spoken


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

class TestCaptions:
    """
    "Most short-form viewers watch muted. Right now the only text is a small
    subtitle line per scene, so the numbers in the narration never appear on
    screen."
    """

    def test_they_sit_in_the_band_reserved_for_them(self):
        """CAPTION_BAND has been kept clear of labels since it was introduced,
        for a caption that did not exist yet. Measured off a burned frame: a
        single 75px line lands at y=1359-1416."""
        assert me.CAPTION_BAND[0] <= 1359 and 1416 <= me.CAPTION_BAND[1]

    def test_they_clear_the_progress_hairline(self):
        assert 1416 < me.PROGRESS_Y

    def test_a_phrase_fits_on_one_line(self):
        """Two lines is 142px of ink against a 132px band, so the second runs
        through the hairline and into the zone labels are clamped to."""
        assert me.CAPTION_WORDS_PER_PHRASE == 3
        assert me.CAPTION_FONT_SCALE <= 0.072

    def test_the_word_timings_are_shifted_by_the_trim(self):
        """They are measured on the untrimmed take. Dropping the shift puts the
        captions a beat late for the whole video."""
        import inspect

        source = inspect.getsource(me.build_minimalist_video)
        assert "narration_trimmed" in source
        assert 'float(word.get("start", 0.0)) - narration_trimmed' in source

    def test_a_failed_burn_keeps_the_render(self):
        import inspect

        source = inspect.getsource(me.render_animation)
        assert "except Exception" in source
        assert "keeping the clean render" in source

    def test_captions_go_on_before_the_loudness_pass(self):
        """Loudness copies the video stream, so burning first is one re-encode
        rather than two."""
        import inspect

        source = inspect.getsource(me.render_animation)
        assert source.index("burn_ass_subtitles") < source.index("normalise_loudness")


# ---------------------------------------------------------------------------
# The scorecard's honesty
# ---------------------------------------------------------------------------

class TestRenderChecks:
    """
    "The 8.8 grades only the script. Every problem above is visual, so the
    score overstates how ready the video is."
    """

    DELIVERED = {
        "duration": 74.3, "fps": 24, "captions": False,
        "loudness": {"after": {"lufs": -14.5}},
        "spec": {"acts": [
            {"template": "delusion_mirror", "title": "The Downside Myth", "labels": {}},
            {"template": "two_doors", "title": "Audit Your Ratio", "labels": {}},
            {"template": "compounding_jar", "title": "The Silent Bleed",
             "labels": {"unit": "YEAR", "meter": "%"}},
        ]},
    }

    def test_it_catches_what_the_review_caught(self):
        failed = {c["key"] for c in compliance.failing_checks(
            compliance.render_checks(self.DELIVERED))}
        assert {"act_labels", "captions", "fps"} <= failed

    def test_it_names_the_unlabelled_scenes(self):
        detail = next(c["detail"] for c in compliance.render_checks(self.DELIVERED)
                      if c["key"] == "act_labels")
        assert "The Downside Myth" in detail and "Audit Your Ratio" in detail

    def test_a_good_render_passes_everything(self):
        good = {
            "duration": 68.0, "fps": 30, "captions": True, "caption_words": 134,
            "loudness": {"after": {"lufs": -14.0}},
            "spec": {"acts": [
                {"template": "compounding_jar", "title": "A",
                 "labels": {"unit": "YEAR", "meter": "%"}},
                {"template": "two_doors", "title": "B",
                 "labels": {"left": "X", "right": "Y", "through": "Z"}},
                {"template": "split_path", "title": "C",
                 "labels": {"near": "X", "far": "Y", "easy": "Z"}},
                {"template": "balance_scale", "title": "D",
                 "labels": {"left": "X", "right": "Y"}},
            ]},
        }
        assert compliance.failing_checks(compliance.render_checks(good)) == []

    def test_it_says_nothing_about_what_it_cannot_see(self):
        """An empty result must not produce a wall of false failures."""
        assert compliance.render_checks({}) == []

    def test_the_payout_floor_is_a_check(self):
        short = dict(self.DELIVERED, duration=56.3)
        failed = {c["key"] for c in compliance.failing_checks(
            compliance.render_checks(short))}
        assert "payout_length" in failed


class TestTheScorerIsGivenItsNumbers:
    """
    "It contradicts itself: the specificity note says 147 words, while the
    pacing tip says 156."

    One was counted and one was written by a model that had been shown the
    script and the duration and no measured figures at all.
    """

    def test_the_prompt_carries_the_counts(self):
        assert "{words}" in ge.VIRAL_PROMPT
        assert "{sentences}" in ge.VIRAL_PROMPT
        assert "{wpm:.0f}" in ge.VIRAL_PROMPT

    def test_it_forbids_the_model_inventing_its_own(self):
        assert "use exactly these numbers" in ge.VIRAL_PROMPT

    def test_the_counts_are_actually_computed(self):
        import inspect

        source = inspect.getsource(ge.score_virality)
        assert "word_count = len(" in source
        assert "words=word_count" in source
