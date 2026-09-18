"""
Captions on the voice, phrases on the sense, and an ending that is said.

A review of minimalist_1789579603.mp4 scored it 5/10, and the first thing it
named was sync: captions finishing 0.3-0.8s before the voice at the end of
every sentence and starting late on the next. That was the estimate doing what
its own docstring said it did. Gemini TTS returns no word timings, so words
were laid across the speech by letter count; measured against edge-tts's real
boundaries on a 133-word script, 85 of those words landed more than a quarter
of a second from the voice. faster-whisper, matched back onto the script, put
none of them there.

The same review read captions like "DRAIN STANDARD AND" and "REAL RETURN A",
a label cut to "ACTIVE 88 PERCEN", and a closing card that sat in silence for
six seconds. Those are here too.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

import audio_engine as ae
import compliance
import minimalist_engine as me
import video_engine as ve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tone_file(path, pattern, rate=16000):
    """A WAV of tone and silence: pattern is [(seconds, voiced), ...]."""
    chunks = []
    for seconds, voiced in pattern:
        n = int(seconds * rate)
        t = np.arange(n) / rate
        wave = 0.6 * np.sin(2 * math.pi * 220 * t) if voiced else np.zeros(n)
        chunks.append(wave)
    mono = np.concatenate(chunks).astype(np.float32)
    ae.save_wav_to_file(np.stack([mono, mono], axis=1), rate, str(path))
    return str(path)


class _Word:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _Segment:
    def __init__(self, words):
        self.words = words


class _Model:
    """Stands in for faster-whisper: returns the words it is given."""

    def __init__(self, heard):
        self.heard = heard

    def transcribe(self, samples, **kwargs):
        assert kwargs.get("word_timestamps"), "alignment needs word timestamps"
        return iter([_Segment([_Word(*w) for w in self.heard])]), None


@pytest.fixture()
def voiced(tmp_path):
    """Four seconds of continuous speech-like tone."""
    return _tone_file(tmp_path / "voiced.wav", [(4.0, True)])


def _align(monkeypatch, audio, text, heard):
    monkeypatch.setattr(ae, "_aligner", lambda: _Model(heard))
    return ae.align_word_timings(audio, text)


# ---------------------------------------------------------------------------
# Measured alignment
# ---------------------------------------------------------------------------

class TestAlignment:

    def test_matched_words_take_the_times_that_were_heard(self, monkeypatch, voiced):
        words = _align(monkeypatch, voiced, "alpha beta gamma",
                       [(" alpha", 0.10, 0.50), (" beta", 0.60, 1.00), (" gamma", 1.20, 1.70)])
        assert [w["start"] for w in words] == [0.10, 0.60, 1.20]
        assert [w["end"] for w in words] == [0.50, 1.00, 1.70]

    def test_every_script_word_gets_a_time(self, monkeypatch, voiced):
        words = _align(monkeypatch, voiced, "one two three four five",
                       [(" one", 0.1, 0.4), (" five", 2.0, 2.4)])
        assert len(words) == 5

    def test_a_number_heard_as_digits_shares_its_time_across_the_words(self, monkeypatch, voiced):
        """ "eighty-eight percent" comes back from the recogniser as "88%"."""
        words = _align(monkeypatch, voiced, "eighty-eight percent of funds",
                       [(" 88%", 0.20, 0.90), (" of", 1.00, 1.10), (" funds", 1.20, 1.60)])
        assert words[0]["start"] == pytest.approx(0.20)
        assert 0.20 < words[1]["start"] < 0.90
        assert words[1]["end"] == pytest.approx(0.90, abs=0.01)
        assert words[2]["start"] == pytest.approx(1.00)

    def test_words_nobody_heard_fill_the_gap_between_their_neighbours(self, monkeypatch, voiced):
        words = _align(monkeypatch, voiced, "one two three four",
                       [(" one", 0.10, 0.40), (" four", 2.00, 2.40)])
        assert 0.40 <= words[1]["start"] < words[2]["start"] < 2.00

    def test_the_order_is_always_forward(self, monkeypatch, voiced):
        """A mismatched stretch can hand back a time before its predecessor's;
        captions cannot go backwards."""
        words = _align(monkeypatch, voiced, "red green blue",
                       [(" red", 1.00, 1.40), (" green", 0.50, 0.90), (" blue", 1.60, 2.00)])
        starts = [w["start"] for w in words]
        assert starts == sorted(starts)

    def test_the_script_keeps_its_spelling_and_punctuation(self, monkeypatch, voiced):
        words = _align(monkeypatch, voiced, "Fees compound, quietly.",
                       [(" fees", 0.1, 0.4), (" compound", 0.5, 1.0), (" quietly", 1.3, 1.8)])
        assert [w["text"] for w in words] == ["Fees", "compound", "quietly"]
        assert [w["raw"] for w in words] == ["Fees", "compound,", "quietly."]

    def test_extra_words_the_voice_added_are_ignored(self, monkeypatch, voiced):
        """Gemini TTS is steered with a style prefix, and sometimes says it."""
        words = _align(monkeypatch, voiced, "fees compound",
                       [(" calm", 0.0, 0.3), (" certain", 0.4, 0.8),
                        (" fees", 1.0, 1.3), (" compound", 1.4, 1.9)])
        assert [w["start"] for w in words] == [1.0, 1.4]

    def test_a_word_after_a_pause_starts_when_the_voice_does(self, monkeypatch, tmp_path):
        """
        The recogniser hands the pause before a word to that word. On the
        benchmark "On", after a full stop, started 0.85s early -- the whole
        pause -- while no other word was more than 0.22s out.
        """
        audio = _tone_file(tmp_path / "pause.wav", [(1.0, True), (1.0, False), (1.0, True)])
        words = _align(monkeypatch, audio, "year. On",
                       [(" year", 0.40, 0.95), (" On", 1.02, 2.40)])
        assert words[1]["start"] == pytest.approx(2.0, abs=0.05)

    def test_a_word_heard_late_after_a_pause_starts_when_the_voice_does(self, monkeypatch, tmp_path):
        """
        The other direction. On a delivered render the voice came back from a
        pause 0.28-0.46s before the recogniser said the next word started, and
        the caption arrived after the voice it belonged to.
        """
        audio = _tone_file(tmp_path / "late.wav", [(1.0, True), (0.5, False), (1.5, True)])
        words = _align(monkeypatch, audio, "reveal. That ninety",
                       [(" reveal", 0.40, 0.98), (" that", 1.90, 2.10), (" ninety", 2.20, 2.60)])
        assert words[1]["start"] == pytest.approx(1.5, abs=0.05)
        assert words[2]["start"] == pytest.approx(2.20)

    def test_a_word_inside_continuous_speech_is_not_moved(self, monkeypatch, voiced):
        words = _align(monkeypatch, voiced, "steady speech here",
                       [(" steady", 0.4, 0.8), (" speech", 1.3, 1.7), (" here", 2.0, 2.3)])
        assert [w["start"] for w in words] == [0.4, 1.3, 2.0]

    def test_nothing_heard_is_an_error_for_the_caller_to_handle(self, monkeypatch, voiced):
        with pytest.raises(RuntimeError):
            _align(monkeypatch, voiced, "some words", [])


class TestFallback:

    def test_a_missing_model_is_an_estimate_not_a_failed_render(self, monkeypatch, voiced):
        def broken():
            raise ImportError("faster_whisper is not installed")

        monkeypatch.setattr(ae, "_aligner", broken)
        words, source = ae.timed_words(voiced, "one two three")
        assert source == "estimated"
        assert len(words) == 3

    def test_a_working_model_is_reported_as_measured(self, monkeypatch, voiced):
        monkeypatch.setattr(ae, "_aligner", lambda: _Model([(" one", 0.1, 0.3)]))
        _, source = ae.timed_words(voiced, "one")
        assert source == "measured"

    def test_the_estimate_keeps_punctuation_too(self, voiced):
        """So a fallback render still breaks its phrases at clauses."""
        words = ae.estimate_word_timings(voiced, "fees rise. slowly")
        assert words[1]["raw"] == "rise."

    def test_gemini_narration_is_timed_through_it(self):
        source = inspect.getsource(ae.synthesize_gemini)
        assert "timed_words(" in source
        assert '"timing_source"' in source
        assert "estimate_word_timings(" not in source

    def test_the_model_is_the_measured_choice(self):
        """small.en: median 44ms, 95% within 0.14s on the benchmark, against
        84ms and 0.24s for base.en."""
        assert ae.ALIGN_MODEL == "small.en"

    def test_it_is_a_pinned_dependency(self):
        import pathlib

        text = (pathlib.Path(__file__).resolve().parents[1] / "requirements.txt").read_text(
            encoding="utf-8")
        assert "faster-whisper==" in text


# ---------------------------------------------------------------------------
# Phrases
# ---------------------------------------------------------------------------

def _timed(text, gap=0.05):
    words, t = [], 0.0
    for token in text.split():
        words.append({"text": token.strip(".,;:!?"), "raw": token,
                      "start": round(t, 3), "end": round(t + 0.35, 3)})
        t += 0.35 + gap
    return words


def _phrases(text, **kwargs):
    words = ve.spoken_numbers_as_figures(_timed(text))
    return [" ".join(w["text"].upper() for w in phrase)
            for phrase in ve.group_words_into_phrases(words, max_words=3, **kwargs)]


class TestPhrases:

    def test_a_phrase_never_runs_across_a_full_stop(self):
        """"DRAIN STANDARD AND" straddled the end of one sentence and the
        start of the next."""
        phrases = _phrases("before you notice the drain. Standard and Poor's data")
        assert not any("DRAIN STANDARD" in p for p in phrases)
        assert any(p.endswith("DRAIN") for p in phrases)

    def test_a_comma_closes_a_phrase_too(self):
        phrases = _phrases("fund managers fail, compounding lost growth")
        assert any(p.endswith("FAIL") for p in phrases)

    def test_a_full_phrase_does_not_end_on_a_word_that_leads_into_the_next(self):
        """"POTENTIAL AGAINST YOUR" and "REAL RETURN A" both did."""
        phrases = _phrases("compounding lost growth potential against your principal every year")
        for phrase in phrases:
            assert phrase.split()[-1].lower() not in ve.WEAK_PHRASE_ENDINGS, phrases

    def test_a_figure_stays_with_its_unit(self):
        phrases = _phrases("managers fail over fifteen years and more")
        assert "OVER 15 YEARS" in phrases or any("15 YEARS" in p for p in phrases)

    def test_the_length_cap_still_holds(self):
        """The vector engine's caption must fit on one line."""
        text = " ".join(f"word{i}" for i in range(40))
        assert all(len(p.split()) <= 3 for p in _phrases(text))

    def test_timings_without_punctuation_still_group(self):
        """edge-tts strips punctuation from its boundaries; the pause is the
        only signal, and it still works."""
        words = [{"text": "one", "start": 0.0, "end": 0.3},
                 {"text": "two", "start": 0.35, "end": 0.6},
                 {"text": "three", "start": 1.5, "end": 1.8}]
        phrases = ve.group_words_into_phrases(words, max_words=3)
        assert [len(p) for p in phrases] == [2, 1]


class TestFigures:

    @pytest.mark.parametrize("spoken, shown", [
        ("eighty-eight percent", ["88%"]),
        ("two hundred thousand dollars", ["$200,000"]),
        ("a hundred thousand dollar portfolio", ["$100,000", "portfolio"]),
        ("seven percent", ["7%"]),
        ("a one percent fee", ["a", "1%", "fee"]),
        ("year twenty-five", ["year", "25"]),
        ("one point five percent", ["1.5%"]),
        ("in twenty twenty-four", ["in", "2024"]),
        ("three basis points", ["3", "basis", "points"]),
        ("five thousand dollars", ["$5,000"]),
        ("a 1.5 percent fee", ["a", "1.5%", "fee"]),
        ("that 92 percent of", ["that", "92%", "of"]),
        ("from 2,500 dollars", ["from", "$2,500"]),
    ])
    def test_spoken_numbers_become_figures(self, spoken, shown):
        words = ve.spoken_numbers_as_figures(_timed(spoken))
        assert [w["text"] for w in words] == shown

    @pytest.mark.parametrize("text", [
        "the one that wins", "no one noticed", "one of them", "a fund",
    ])
    def test_words_that_are_not_quantities_are_left_alone(self, text):
        words = ve.spoken_numbers_as_figures(_timed(text))
        assert [w["text"] for w in words] == text.split()

    def test_a_figure_is_highlighted_for_as_long_as_it_is_said(self):
        words = _timed("two hundred thousand dollars")
        merged = ve.spoken_numbers_as_figures(words)
        assert merged[0]["start"] == words[0]["start"]
        assert merged[0]["end"] == words[-1]["end"]

    def test_a_clause_mark_after_a_figure_still_ends_the_phrase(self):
        merged = ve.spoken_numbers_as_figures(_timed("costs seven percent, every year"))
        assert merged[1]["raw"] == "7%,"

    def test_the_vector_engine_asks_for_figures(self):
        assert "figures=True" in inspect.getsource(me.build_minimalist_video)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

class TestLabels:

    def test_a_label_is_never_cut_through_a_word(self):
        shown, shortened = me.fit_label("THE MANAGER YOU TRUST MOST")
        assert shortened
        assert "THE MANAGER YOU TRUST MOST".startswith(shown)
        assert " ".join(shown.split()) == shown
        assert all(word in "THE MANAGER YOU TRUST MOST".split() for word in shown.split())

    @pytest.mark.parametrize("given, shown", [
        ("ACTIVE 88 PERCENT", "ACTIVE 88%"),
        ("1.5 PERCENT FEE", "1.5% FEE"),
        ("3 BASIS POINTS", "3 BPS"),
        ("THREE BASIS POINTS", "3 BPS"),
        ("200 THOUSAND DOLLARS", "$200K"),
        ("EIGHTY-EIGHT PERCENT", "88%"),
    ])
    def test_quantities_are_written_short(self, given, shown):
        assert me.fit_label(given) == (shown, False)

    def test_the_label_that_shipped_now_fits_whole(self):
        """"ACTIVE 88 PERCEN" was on screen for ten seconds."""
        spec = {"template": "comparison_split", "labels": {"tier1": "ACTIVE 88 PERCENT"}}
        assert me.label(spec, "tier1") == "ACTIVE 88%"

    def test_the_vessel_counts_in_its_axis_unit(self):
        """"74%" over "PERCENT 21", beside an axis in years."""
        act = me.normalise_act({"template": "compounding_jar", "thesis": "x",
                                "direction": "drain", "axis_max": 30,
                                "axis_suffix": "y", "labels": {"unit": "PERCENT"}})
        act.update({"seconds": 11.0, "start": 0.0, "climax": 8.0})
        frame = me.Frame()
        me.scene_vessel(frame, 8.0, act, 11.0)
        bodies = [box[0] for box in frame.text_boxes]
        assert any(body.startswith("YEAR ") for body in bodies), bodies
        assert not any(body.startswith("PERCENT ") for body in bodies), bodies

    def test_a_render_check_reports_a_shortened_label(self):
        checks = compliance.render_checks(
            {"duration": 66.0},
            {"acts": [{"template": "split_path", "title": "T",
                       "labels": {"easy": "THE MANAGER YOU TRUST MOST"}}]})
        fit = next(c for c in checks if c["key"] == "labels_fit")
        assert not fit["ok"]
        assert "THE MANAGER" in fit["detail"]


class TestAlignment_Frame:

    def test_left_aligned_text_starts_at_its_anchor(self):
        frame = me.Frame()
        frame.text("LEFT EDGE", (300, 800), 40, me.WHITE, align="left")
        x0 = frame.text_boxes[0][1]
        assert x0 == pytest.approx(300, abs=12)

    def test_right_aligned_text_ends_at_its_anchor(self):
        frame = me.Frame()
        frame.text("RIGHT EDGE", (700, 800), 40, me.WHITE, align="right")
        x1 = frame.text_boxes[0][3]
        assert x1 == pytest.approx(700, abs=12)

    def test_centred_is_still_the_default(self):
        frame = me.Frame()
        frame.text("MIDDLE", (540, 800), 40, me.WHITE)
        x0, x1 = frame.text_boxes[0][1], frame.text_boxes[0][3]
        assert (x0 + x1) / 2 == pytest.approx(540, abs=2)


# ---------------------------------------------------------------------------
# The spoken close
# ---------------------------------------------------------------------------

ACTS = [{"template": "split_path", "thesis": "Fees look small."},
        {"template": "compounding_jar", "thesis": "They compound every year."}]


class TestSpokenClose:

    def test_the_payoff_is_appended_as_a_sentence(self):
        line, index = me.spoken_close(ACTS, "Pay three basis points")
        assert line == "Pay three basis points."
        assert index == 7

    def test_a_payoff_the_last_act_already_says_is_not_said_twice(self):
        acts = ACTS[:1] + [{"thesis": "They compound. Pay three basis points."}]
        line, index = me.spoken_close(acts, "pay three basis points")
        assert line == ""
        assert index == 3 + 2

    def test_no_payoff_means_nothing_appended(self):
        assert me.spoken_close(ACTS, "") == ("", 7)

    def test_the_card_can_start_when_the_voice_does(self):
        assert me.closing_alpha_at(49.9, 70.0, 50.0) == 0.0
        assert me.closing_alpha_at(50.0 + me.CLOSING_FADE, 70.0, 50.0) == 1.0

    def test_without_a_start_the_card_owns_the_last_seconds_as_before(self):
        assert me.closing_alpha_at(70.0 - me.CLOSING_SECONDS - 0.1, 70.0) == 0.0
        assert me.closing_alpha_at(69.0, 70.0) == 1.0

    def test_the_labels_clear_out_when_the_card_does(self):
        assert me.annotation_alpha_at(49.0, 70.0, 50.0) == 1.0
        assert me.annotation_alpha_at(51.0, 70.0, 50.0) == 0.0

    def test_the_card_is_drawn_from_the_spec_start(self):
        spec = me.normalise_spec({"payoff": "P", "closing_at": 30.0, "duration": 40.0})
        assert spec["closing_at"] == 30.0
        early = np.asarray(me._draw_act(spec, spec, 29.0, 40.0, 29.0, 40.0))
        late = np.asarray(me._draw_act(spec, spec, 31.0, 40.0, 31.0, 40.0))
        assert not np.array_equal(early, late)


class TestActsOnTheVoice:

    def _spec(self):
        return me.normalise_spec({"title": "T", "payoff": "P", "duration": 66.0,
                                  "acts": [{"template": "split_path", "thesis": "a " * 10},
                                           {"template": "compounding_jar", "thesis": "b " * 10},
                                           {"template": "two_doors", "thesis": "c " * 10}]})

    def test_each_act_starts_just_before_its_first_word(self):
        acts = me.allocate_acts(self._spec(), 40.0, [0.2, 12.3, 25.0])
        lead = me.ACT_OVERLAP * me.TRANSITION_BLACK_AT
        assert acts[0]["start"] == 0.0
        assert acts[1]["start"] == pytest.approx(12.3 - lead)
        assert acts[2]["start"] == pytest.approx(25.0 - lead)

    def test_they_tile_the_span_exactly(self):
        acts = me.allocate_acts(self._spec(), 40.0, [0.2, 12.3, 25.0])
        assert sum(a["seconds"] for a in acts) == pytest.approx(40.0)
        for first, second in zip(acts, acts[1:]):
            assert first["start"] + first["seconds"] == pytest.approx(second["start"])

    def test_a_wrong_count_of_starts_falls_back_to_word_share(self):
        by_words = me.allocate_acts(self._spec(), 40.0)
        assert me.allocate_acts(self._spec(), 40.0, [0.0, 5.0]) == by_words

    def test_starts_too_close_together_still_give_each_act_time(self):
        acts = me.allocate_acts(self._spec(), 40.0, [0.0, 0.5, 0.9])
        assert all(a["seconds"] >= 1.9 for a in acts[:-1])


class TestBuildSpeaksTheClose:
    """
    End to end through build_minimalist_video, with the voice and the drawing
    stubbed: what matters here is what gets said, where the card goes, and
    where the acts cut.
    """

    PAUSE = 0.6
    # Realistic act lengths: twenty words each is eight seconds here. With a
    # two-word act the minimum act length, not the voice, decides the cut.
    ACTS = [{"template": "split_path", "thesis": " ".join(["fees"] * 20)},
            {"template": "compounding_jar", "thesis": " ".join(["compound"] * 20)}]

    def _run(self, monkeypatch, tmp_path, captured):
        def narrate(text, **kwargs):
            captured["text"] = text
            tokens = text.split()
            payoff_at = len(" ".join(a["thesis"] for a in self.ACTS).split())
            words, t = [], 0.0
            for index, token in enumerate(tokens):
                if index == payoff_at:
                    t += self.PAUSE
                words.append({"start": round(t, 3), "end": round(t + 0.3, 3),
                              "text": token.strip(".,"), "raw": token})
                t += 0.4
            path = _tone_file(tmp_path / "voice.wav", [(t, True)])
            captured["words"] = words
            return {"path": path, "duration": t, "words": words,
                    "timings_exact": True, "timing_source": "measured"}

        def draw(spec, output_path, **kwargs):
            captured["spec"] = spec
            return {"output_path": output_path, "duration": float(spec["duration"]),
                    "fps": 30, "frames": 1, "template": spec["template"],
                    "climax": spec["climax"]}

        monkeypatch.setattr(ae, "synthesize_narration", narrate)
        monkeypatch.setattr(ae, "trim_leading_silence",
                            lambda path, **kw: {"path": path, "trimmed": 0.0,
                                                "duration": captured["words"][-1]["end"] + 0.1})
        monkeypatch.setattr(me, "render_animation", draw)
        spec = {"title": "T", "payoff": "Pay three basis points",
                "acts": [dict(a) for a in self.ACTS], "duration": 66.0}
        return me.build_minimalist_video(spec, str(tmp_path / "out.mp4"), bgm=False,
                                         sfx=False, narrate=True)

    def test_the_payoff_is_the_last_thing_said(self, monkeypatch, tmp_path):
        captured = {}
        self._run(monkeypatch, tmp_path, captured)
        assert captured["text"].endswith("Pay three basis points.")

    def test_the_card_comes_up_as_the_voice_starts_the_payoff(self, monkeypatch, tmp_path):
        captured = {}
        result = self._run(monkeypatch, tmp_path, captured)
        said = captured["words"][40]["start"]
        closing_at = captured["spec"]["closing_at"]
        visible = closing_at + me.CLOSING_FADE * me.TRANSITION_BLACK_AT
        assert visible == pytest.approx(said, abs=0.01)
        assert result["payoff_spoken"] is True

    def test_each_act_cuts_on_its_own_first_word(self, monkeypatch, tmp_path):
        captured = {}
        self._run(monkeypatch, tmp_path, captured)
        acts = captured["spec"]["acts"]
        lead = me.ACT_OVERLAP * me.TRANSITION_BLACK_AT
        second_act_first_word = captured["words"][20]["start"]
        assert acts[1]["start"] == pytest.approx(second_act_first_word - lead)

    def test_the_captions_say_how_they_were_timed(self, monkeypatch, tmp_path):
        captured = {}
        result = self._run(monkeypatch, tmp_path, captured)
        assert result["caption_timing"] == "measured"

    def test_it_still_clears_the_payout_floor(self, monkeypatch, tmp_path):
        captured = {}
        result = self._run(monkeypatch, tmp_path, captured)
        assert result["duration"] >= me.PAYOUT_FLOOR_SECONDS


# ---------------------------------------------------------------------------
# The Dashboard sees the facts
# ---------------------------------------------------------------------------

class TestTheDashboardSeesTheRender:

    def test_the_one_click_result_carries_the_engine_result(self):
        """The checks panel was reading `result.get("render")`, and the
        one-click path never set it -- so on the Dashboard it only ever saw
        the runtime."""
        import app

        assert '"render":' in inspect.getsource(app._magic_minimalist)
        assert '"render": result.get("render")' in inspect.getsource(app.start_magic_job)

    def test_estimated_captions_are_flagged(self):
        checks = compliance.render_checks(
            {"duration": 66.0, "captions": True, "caption_words": 120,
             "caption_timing": "estimated"})
        timing = next(c for c in checks if c["key"] == "caption_timing")
        assert not timing["ok"]

    def test_measured_captions_pass(self):
        checks = compliance.render_checks(
            {"duration": 66.0, "captions": True, "caption_words": 120,
             "caption_timing": "measured"})
        assert next(c for c in checks if c["key"] == "caption_timing")["ok"]

    def test_a_silent_card_is_flagged(self):
        checks = compliance.render_checks({"duration": 66.0, "payoff_spoken": False})
        assert not next(c for c in checks if c["key"] == "payoff_spoken")["ok"]


# ---------------------------------------------------------------------------
# The script writer's rules
# ---------------------------------------------------------------------------

class TestPromptRules:

    @pytest.fixture(scope="class")
    def prompt(self):
        import gemini_engine as ge

        return ge.build_scene_plan_prompt("the year your fees overtake your returns")

    @pytest.mark.parametrize("rule", [
        "ONE SET OF NUMBERS", "WORK THE NUMBERS OUT", "NO ABSOLUTES", "OPEN ON THE FIGURE",
    ])
    def test_the_rule_is_there(self, prompt, rule):
        assert rule in prompt

    def test_labels_are_asked_for_as_figures(self, prompt):
        assert '"88%"' in prompt

    def test_the_split_path_says_which_way_it_argues(self, prompt):
        """A fees video put "1.5% FEE" on the grey path and a reviewer read the
        climbing white curve as the fee."""
        assert "BRIGHT steep path" in prompt and "GREY flat path" in prompt

    def test_the_payoff_is_written_to_be_said(self, prompt):
        assert "payoff is read aloud" in prompt

class TestAFailedPlanCannotHide:
    """
    The six-act plan failed once, transiently -- the same concept planned fine
    on the next attempt -- and the render fell back to a single metaphor held
    for 71 seconds, with a script from the single-scene prompt that carries
    none of the evidence rules. Domain Specificity came back 3.0: one
    checkable detail in 150 words.

    The card looked normal. Both checks that would have caught it lived inside
    `if acts:` and simply were not rendered, because there were no acts. A
    check that disappears when the thing it guards is missing is not a check.
    """

    @staticmethod
    def _checks(spec):
        return compliance.render_checks({
            "duration": 71.3, "captions": True, "caption_words": 150,
            "caption_timing": "measured", "fps": 30,
            "loudness": {"after": {"lufs": -14.3}}, "spec": spec})

    def test_a_render_with_no_acts_fails_loudly(self):
        checks = self._checks({"template": "comparison_split"})
        failing = {check["key"] for check in compliance.failing_checks(checks)}
        assert "act_count" in failing

    def test_the_failure_says_what_happened(self):
        checks = self._checks({"template": "comparison_split"})
        detail = next(c["detail"] for c in checks if c["key"] == "act_count")
        assert "1 scene" in detail and "single-scene prompt" in detail

    def test_it_stays_quiet_when_there_was_no_render_at_all(self):
        """The opposite mistake: a false failure about something this cannot
        see. An empty result must produce no checks."""
        assert compliance.render_checks({}) == []
        assert compliance.render_checks({"spec": {"template": "two_doors"}}) == []

    def test_a_planned_render_still_passes(self):
        acts = [{"template": "compounding_jar", "title": "A", "labels": {"unit": "YEAR"}},
                {"template": "two_doors", "title": "B",
                 "labels": {"left": "L", "right": "R", "through": "T"}},
                {"template": "balance_scale", "title": "C", "labels": {"left": "L", "right": "R"}},
                {"template": "gravity_funnel", "title": "D", "labels": {"pull": "P"}},
                {"template": "split_path", "title": "E",
                 "labels": {"near": "N", "far": "F", "easy": "E"}},
                {"template": "domino_chain", "title": "F",
                 "labels": {"first": "1", "last": "2"}}]
        checks = self._checks({"acts": acts})
        assert not [c for c in compliance.failing_checks(checks)
                    if c["key"] == "act_count"]

    def test_the_planner_asks_twice_before_giving_up(self):
        """The caller's fallback is so much worse than any plan that another
        round of asking is cheap by comparison."""
        import inspect

        import gemini_engine as ge

        source = inspect.getsource(ge.generate_scene_plan)
        assert "MODEL_CANDIDATES * 2" in source

    def test_the_fallback_tells_the_user(self):
        import inspect

        import app

        source = inspect.getsource(app._magic_minimalist)
        assert "fell_back = True" in source
        assert "weaker than usual" in source
