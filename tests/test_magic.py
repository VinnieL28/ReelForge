"""
Magic Studio, the render credits, and the scorecard that stopped saying 5.5.

The one-click flow is a planner plus the existing pipelines. The planner is
what is tested here, because it is where a prompt becomes concrete inputs and
where a wrong answer produces a confidently wrong video rather than an error:
a matchup split down the wrong word, a storm rendered as light rain, an eight
hour request rendered as a minute.

The scorecard section pins the behaviour the flat mid-fives came from. The
offline card seeded every hook at 5.0 and adjusted from there, so a script with
nothing measurable in it came back "average" -- and averaging that into the
model's read dragged every real verdict back toward it.
"""
from __future__ import annotations

import time

import pytest

import compliance
import magic_studio as ms


# ---------------------------------------------------------------------------
# Reading the prompt
# ---------------------------------------------------------------------------

class TestMatchupSplit:

    @pytest.mark.parametrize("prompt,left,right", [
        ("Range Rover vs Porsche Cayenne", "Range Rover", "Porsche Cayenne"),
        ("iPhone 17 Pro VS Galaxy S26", "iPhone 17 Pro", "Galaxy S26"),
        ("Rolex versus Omega", "Rolex", "Omega"),
        ("Which is better: Tesla vs Rivian", "Tesla", "Rivian"),
        ("compare Nikon Z8 vs Sony A7 IV", "Nikon Z8", "Sony A7 IV"),
    ])
    def test_it_splits_a_matchup(self, prompt, left, right):
        assert ms.split_matchup(prompt) == (left, right)

    @pytest.mark.parametrize("prompt", [
        "Why Rome fell in 60s", "8-hour rain on window", "", "   ",
        "the history of the roman empire",
    ])
    def test_a_non_matchup_returns_nothing(self, prompt):
        """Empty is the signal to ask the model for a matchup. Guessing at a
        split would build a duel between half a sentence and the other half."""
        assert ms.split_matchup(prompt) == ("", "")


class TestStyleDetection:

    @pytest.mark.parametrize("prompt,expected", [
        ("Range Rover vs Porsche Cayenne", "duel"),
        ("8-hour rain on window", "atmosphere"),
        ("thunderstorm for sleep 3 hours", "atmosphere"),
        ("Why Rome fell in 60s", "commentary"),
        ("the rise of the roman empire", "commentary"),
        ("Why consistency beats intensity", "minimalist"),
        ("how compounding works", "minimalist"),
        ("why you cant focus anymore", "minimalist"),
    ])
    def test_a_prompt_preselects_its_style(self, prompt, expected):
        assert ms.detect_style(prompt) == expected

    def test_an_abstract_concept_does_not_go_looking_for_footage(self):
        """'Why consistency beats intensity' is the canonical vector-engine
        subject. Routing it to commentary sends it hunting for stock footage of
        an idea."""
        assert ms.detect_style("Why consistency beats intensity") == "minimalist"

    def test_an_empty_prompt_falls_back(self):
        assert ms.detect_style("") == ms.DEFAULT_STYLE


class TestAtmospherePlan:

    @pytest.mark.parametrize("prompt,bed", [
        ("8-hour rain on window", "rain_window"),
        ("heavy thunderstorm all night", "thunderstorm"),
        ("crackling fireplace", "fireplace"),
        ("gentle stream in a forest", "stream"),
        ("brown noise for studying", "brown_noise"),
    ])
    def test_the_bed_matches_the_words(self, prompt, bed):
        assert ms.atmosphere_plan(prompt)["bed"] == bed

    def test_a_storm_is_not_filed_as_rain(self):
        """Both words are in the prompt. The more specific one has to win or
        every storm renders as light rain."""
        assert ms.atmosphere_plan("thunderstorm and rain")["bed"] == "thunderstorm"

    @pytest.mark.parametrize("prompt,key", [
        ("8-hour rain", "8hours"), ("3 hours of rain", "3hours"),
        ("1 hour of rain", "1hour"), ("30 min rain", "30min"),
        ("rain", "8hours"),
    ])
    def test_it_hears_the_length(self, prompt, key):
        assert ms.atmosphere_plan(prompt)["duration"] == key

    def test_every_plan_names_real_catalogue_entries(self):
        import ambient_engine

        for prompt in ("rain", "thunderstorm", "fireplace", "stream", "brown noise",
                       "night forest", "white noise 3 hours", "campfire"):
            plan = ms.atmosphere_plan(prompt)
            assert plan["bed"] in ambient_engine.PRIMARY_BEDS, prompt
            assert plan["texture"] in ambient_engine.SECONDARY_TEXTURES, prompt
            assert plan["duration"] in ambient_engine.DURATIONS, prompt


class TestSceneConcept:

    @pytest.mark.parametrize("prompt,concept", [
        ("make me a video about compound interest", "compound interest"),
        ("create a short on discipline", "discipline"),
        ("Why consistency beats intensity", "Why consistency beats intensity"),
    ])
    def test_it_strips_the_request_and_keeps_the_subject(self, prompt, concept):
        assert ms.scene_concept(prompt) == concept

    def test_it_never_returns_nothing(self):
        assert ms.scene_concept("make a video") .strip()


# ---------------------------------------------------------------------------
# Styles and stages
# ---------------------------------------------------------------------------

class TestStyles:

    def test_the_four_styles_the_brief_names(self):
        assert ms.STYLE_KEYS == ("minimalist", "duel", "commentary", "atmosphere")

    def test_every_style_maps_to_a_real_mode(self):
        import app

        for entry in ms.STYLES:
            assert entry["mode"] in app.MODE_LABELS, entry["key"]

    def test_every_style_has_an_example_that_routes_to_itself(self):
        """The example in the placeholder has to be a prompt that actually
        lands on the style it is advertising."""
        for entry in ms.STYLES:
            assert ms.detect_style(entry["example"]) == entry["key"], entry["key"]

    def test_the_placeholder_carries_all_three_examples(self):
        for phrase in ("Why Rome fell", "Range Rover vs Porsche Cayenne",
                       "8-hour rain on window"):
            assert phrase in ms.PLACEHOLDER

    def test_an_unknown_style_key_does_not_explode(self):
        assert ms.style("nope")["key"] in ms.STYLE_KEYS


class TestStages:

    def test_the_four_stages_the_bar_promises(self):
        assert [label for _k, label, _w in ms.STAGES] == [
            "Writing Hook", "Synthesizing Voice", "Animating", "Finalizing Render"]

    def test_the_weights_are_a_whole_job(self):
        assert abs(sum(weight for _k, _l, weight in ms.STAGES) - 1.0) < 1e-9

    def test_progress_only_moves_forward(self):
        seen = [ms.stage_fraction(key, 0.0) for key, _l, _w in ms.STAGES]
        assert seen == sorted(seen)
        assert seen[0] == 0.0

    def test_progress_inside_a_stage_moves_the_bar(self):
        """The render is half the job. A bar that sits still through it is the
        one people assume has hung."""
        start = ms.stage_fraction("render", 0.0)
        mid = ms.stage_fraction("render", 0.5)
        end = ms.stage_fraction("render", 1.0)
        assert start < mid < end
        assert end == pytest.approx(1.0)

    def test_the_caption_marks_where_it_is(self):
        caption = ms.stage_caption("voice")
        assert "**Synthesizing Voice**" in caption
        assert caption.count("→") == 3


# ---------------------------------------------------------------------------
# The duel brief
# ---------------------------------------------------------------------------

class TestDuelBrief:

    RAW = """{"a": {"name": "Range Rover", "hook": "The luxury benchmark"},
              "b": {"name": "Porsche Cayenne", "hook": "Sports car DNA"},
              "headline": "Range Rover vs Porsche Cayenne",
              "cta": "Which would you pick?",
              "rounds": [
                {"metric": "Starting Price", "a_score": 107400, "b_score": 79200,
                 "unit": "$", "winner": "B", "note": "Cayenne saves 28 grand"},
                {"metric": "Horsepower", "a_score": 395, "b_score": 468,
                 "unit": "hp", "winner": "B", "note": "73 more horses"},
                {"metric": "0-60 mph", "a_score": 5.5, "b_score": 4.6,
                 "unit": "s", "winner": "B", "note": "Nearly a second quicker"}]}"""

    def test_it_parses_a_full_brief(self):
        brief = ms.parse_duel_brief(self.RAW)
        assert brief["a"]["name"] == "Range Rover"
        assert len(brief["rounds"]) == 3
        assert brief["rounds"][0]["unit"] == "$"

    def test_a_missing_winner_is_derived_from_the_numbers(self):
        raw = self.RAW.replace('"winner": "B", "note": "Cayenne saves 28 grand"',
                               '"winner": "", "note": "Cayenne saves 28 grand"')
        brief = ms.parse_duel_brief(raw)
        # Price: the lower number wins, whatever a naive higher-is-better rule
        # would say.
        assert brief["rounds"][0]["winner"] == "B"

    def test_higher_wins_where_higher_is_better(self):
        raw = self.RAW.replace('"winner": "B", "note": "73 more horses"',
                               '"winner": "", "note": "73 more horses"')
        assert ms.parse_duel_brief(raw)["rounds"][1]["winner"] == "B"

    def test_a_round_with_no_numbers_is_dropped_not_faked(self):
        raw = self.RAW.replace('"a_score": 395', '"a_score": "loads"')
        brief = ms.parse_duel_brief(raw)
        assert len(brief["rounds"]) == 2
        assert all(isinstance(r["a_score"], float) for r in brief["rounds"])

    def test_a_brief_with_no_names_is_rejected(self):
        assert ms.parse_duel_brief('{"a": {}, "b": {}, "rounds": []}') == {}

    def test_junk_is_rejected(self):
        for raw in ("", "sorry, I cannot help with that", "{]"):
            assert ms.parse_duel_brief(raw) == {}

    def test_it_builds_slides_the_renderer_accepts(self):
        """The point of the brief is the render, so the shape has to survive
        the trip into build_duel_slides."""
        import app
        from PIL import Image

        brief = ms.parse_duel_brief(self.RAW)
        duel = {
            "layout": "stacked", "headline": brief["headline"], "cta": brief["cta"],
            "rounds": brief["rounds"],
            "a": {**brief["a"], "image": Image.new("RGB", (64, 64), (20, 30, 40))},
            "b": {**brief["b"], "image": Image.new("RGB", (64, 64), (40, 30, 20))},
        }
        slides = app.build_duel_slides(duel)
        kinds = [slide["kind"] for slide in slides]
        assert kinds[0] == "duel_intro" and kinds[-1] == "duel_winner"
        assert kinds.count("duel_round") == 3


# ---------------------------------------------------------------------------
# Credits
# ---------------------------------------------------------------------------

class TestCredits:

    def _render(self, root, name, when):
        path = root / name
        path.write_bytes(b"\x00" * 32)
        import os

        os.utime(path, (when, when))

    def test_it_counts_this_month_only(self, tmp_path):
        import dashboard_view as dv

        now = time.time()
        self._render(tmp_path, "reel_1.mp4", now)
        self._render(tmp_path, "duel_2.mp4", now)
        self._render(tmp_path, "old_3.mp4", now - 86400 * 70)   # two months back

        status = dv.credit_status(str(tmp_path), {"tier": "pro"}, now=now)
        assert status["used"] == 2
        assert status["quota"] == 500
        assert status["remaining"] == 498

    def test_scratch_is_not_a_render(self, tmp_path):
        """Intermediates live in the same folder. Counting them would bill
        someone for their own narration WAVs."""
        import dashboard_view as dv

        now = time.time()
        self._render(tmp_path, "reel_1.mp4", now)
        for scratch in ("source_9.mp4", "muted_9.mp4", "preview_9.mp4",
                        "reframed_9.mp4", "clip_gemini.mp4"):
            self._render(tmp_path, scratch, now)

        assert dv.credit_status(str(tmp_path), {}, now=now)["used"] == 1

    def test_a_missing_folder_is_zero_not_an_error(self, tmp_path):
        import dashboard_view as dv

        status = dv.credit_status(str(tmp_path / "nope"), {"tier": "free"})
        assert status["used"] == 0 and status["remaining"] == 50

    def test_the_tier_falls_back_rather_than_failing(self):
        import dashboard_view as dv

        for record in (None, {}, {"tier": "bogus"}, {"tier": ""}, "nonsense"):
            assert dv.tier_of(record) in dv.TIERS

    def test_remaining_never_goes_negative(self, tmp_path):
        import dashboard_view as dv

        now = time.time()
        for index in range(4):
            self._render(tmp_path, f"reel_{index}.mp4", now)
        status = dv.credit_status(str(tmp_path), {"tier": "over"}, now=now)
        assert status["remaining"] >= 0


# ---------------------------------------------------------------------------
# The scorecard, after the mid-fives
# ---------------------------------------------------------------------------

TEMPLATE = ("You're doing money wrong, here's why. Mistake 1: this quietly costs you "
            "more than you think. Mistake 2: you're optimizing the thing that matters "
            "least. Mistake 3: quitting right before the compounding kicks in. Fix "
            "these and thank me later")

RESEARCHED = ("Stop buying the Range Rover Vogue. At 108,000 dollars it costs 29,000 "
              "more than a Porsche Cayenne that makes 468 horsepower to the Rover's "
              "395, and hits 60 mph in 4.6 seconds against 5.5. Land Rover ranked "
              "last of 32 brands in the 2024 What Car reliability survey.")

CLEAN = {"licence": "cc0", "duration": 62.0, "tts_provider": "edge",
         "ai_disclosed": True}


class TestNoStaticScores:

    def test_the_hook_has_no_neutral_midpoint(self):
        """The 5.0 seed was the bug: an opening with nothing measurable in it
        came back 'average' rather than 'gives nobody a reason to stay'."""
        import inspect

        source = inspect.getsource(compliance.score_hook)
        assert "score = 5.0" not in source
        assert "HOOK_FLOOR" in source

    def test_an_empty_opening_scores_low_not_middling(self):
        bare = ("This is a video about productivity. It covers some ideas. "
                "They are useful ideas. Thanks for watching.")
        assert compliance.score_hook(bare)["score"] < 5.0

    def test_the_two_ends_of_the_range_are_far_apart(self):
        """A scorer whose answers cluster is a scorer nobody can act on."""
        low = compliance.viral_scorecard(TEMPLATE, CLEAN)["overall"]
        high = compliance.viral_scorecard(RESEARCHED, CLEAN)["overall"]
        assert high - low >= 3.0, f"{low} to {high} is not a usable spread"

    def test_every_card_carries_one_actionable_sentence(self):
        for script in (TEMPLATE, RESEARCHED):
            tip = compliance.viral_scorecard(script, CLEAN)["retention_tip"]
            assert tip and tip.endswith("."), tip
            assert len(tip.split()) >= 8, "a tip too short to be actionable"
            assert "add more detail" not in tip.lower()

    def test_the_tip_names_the_thing_to_change(self):
        """Generic advice is what the tip exists to replace."""
        tip = compliance.viral_scorecard(TEMPLATE, CLEAN)["retention_tip"].lower()
        assert any(word in tip for word in ("cut", "delete", "lead with", "put a",
                                            "reframe", "give the"))


class TestMonetizationIsCategorical:

    def test_uncleared_footage_cannot_read_as_publishable(self):
        """A researched script over an unverified licence used to average out
        to exactly 8.0 and report 'good enough to publish'."""
        risky = {**CLEAN, "licence": "unverified"}
        card = compliance.viral_scorecard(RESEARCHED, risky)
        assert card["overall"] < compliance.VIRAL_TARGET_SCORE
        assert card["needs_rewrite"]
        assert "monetization" in card["verdict"].lower()

    def test_the_ceiling_tracks_the_axis_rather_than_parking_on_a_number(self):
        """A fixed "blocked" score would read the same for a fair-use claim and
        for footage nobody has cleared at all. They are not the same problem."""
        worse = compliance.viral_scorecard(RESEARCHED, {**CLEAN, "licence": "unverified"})
        better = compliance.viral_scorecard(RESEARCHED, {**CLEAN, "licence": "fair_use"})
        assert worse["overall"] < better["overall"], (worse["overall"], better["overall"])
        assert worse["monetization"]["score"] < better["monetization"]["score"]

    def test_a_clean_entry_is_not_capped(self):
        card = compliance.viral_scorecard(RESEARCHED, CLEAN)
        assert card["monetization"]["score"] >= compliance.MONETIZATION_FLOOR


class TestVerdictIsShared:

    def test_one_verdict_function_serves_both_paths(self):
        """A 7.4 must not read 'workable' on one path and 'not ready' on the
        other. Both call the same function."""
        card = {"hook": {"score": 7.0}, "density": {"score": 7.0},
                "monetization": {"score": 8.0}}
        assert compliance.verdict_for(9.0, card) == "Strong on all three axes."
        assert compliance.verdict_for(8.2, card) == "Good enough to publish."
        assert "Workable" in compliance.verdict_for(7.0, card)
        assert "Not ready" in compliance.verdict_for(3.0, card)

    def test_the_verdict_names_the_weakest_axis(self):
        card = {"hook": {"score": 9.0}, "density": {"score": 9.0},
                "monetization": {"score": 2.0}}
        assert compliance.weakest_axis(card) == "monetization"
        assert "monetization" in compliance.verdict_for(4.0, card).lower()


# ---------------------------------------------------------------------------
# Monetization-eligible by default
# ---------------------------------------------------------------------------

class TestRewardsDuration:

    def test_the_default_band_clears_the_tiktok_threshold(self):
        """Creator Rewards pays nothing for anything under a minute, so a
        default that renders 20-30s videos is a default that earns nothing."""
        import gemini_engine

        band = gemini_engine.DURATION_TARGETS[gemini_engine.DEFAULT_TARGET]
        assert band["low"] > compliance.TIKTOK_REWARDS_MIN_SECONDS
        assert band["high"] <= 90

    def test_the_floor_has_margin_over_the_bar(self):
        """TTS lands within a few percent of a word budget, not on it. A band
        starting at exactly 60 puts about half its renders under the bar."""
        import gemini_engine

        band = gemini_engine.DURATION_TARGETS["rewards"]
        assert band["low"] - compliance.TIKTOK_REWARDS_MIN_SECONDS >= 2

    def test_the_shorter_bands_still_exist(self):
        """Defaulting to rewards-eligible must not remove the choice."""
        import gemini_engine

        for key in ("short", "standard", "deep"):
            assert key in gemini_engine.DURATION_TARGETS


# ---------------------------------------------------------------------------
# Platform compliance guarantees
# ---------------------------------------------------------------------------

class TestOriginalByConstruction:
    """
    Minimalist Motion is the answer to YouTube's reused-content rule, so the
    two properties that make it one are worth pinning: the picture is computed,
    not fetched, and the sound is several generated layers rather than a
    licensed track.
    """

    def test_the_picture_is_drawn_not_fetched(self):
        import inspect

        import minimalist_engine

        source = inspect.getsource(minimalist_engine)
        for borrowed in ("requests.get", "urlopen", "pexels", "unsplash",
                         "download_licensed_clip", "fetch_photo"):
            assert borrowed not in source, (
                f"{borrowed} would put third-party material in a mode whose whole "
                f"claim is that there is none")

    def test_the_mix_has_more_than_one_generated_layer(self):
        """A single bed is a soundtrack. Bed plus impact plus narration is a
        mix, and it is what makes the audio track unique to this render."""
        import inspect

        import minimalist_engine

        source = inspect.getsource(minimalist_engine.build_minimalist_video)
        for layer in ("bgm", "sfx", "narrate"):
            assert layer in source, f"no {layer} layer in the mix"
        assert "CompositeAudioClip" in source, "the layers are never combined"

    def test_the_ledger_records_it_as_own_work(self):
        """Provenance is what answers a claim months later. A procedural render
        filed under a stock licence is a render nobody can defend."""
        import inspect

        import app

        source = inspect.getsource(app._magic_minimalist)
        assert '"licence": "own"' in source
        assert '"source_provider": "reelforge"' in source


class TestEncoderFallback:

    def test_the_gpu_path_is_nvenc(self):
        import video_engine

        assert video_engine.GPU_CODEC == "h264_nvenc"

    def test_the_cpu_fallback_is_veryfast(self):
        """A failed NVENC probe must not leave the render on a preset nobody
        waits out. Measured 2.18x faster than medium at 1080x1920."""
        import video_engine

        assert video_engine.CPU_PRESET == "veryfast"

    def test_a_gpu_failure_retries_on_the_cpu_rather_than_giving_up(self):
        import inspect

        import video_engine

        source = inspect.getsource(video_engine.write_clip)
        assert "veryfast" in source, "no CPU preset in the retry"
        assert "yuv420p" in source, (
            "the fallback can pick yuv444p, which plays as a green screen on iOS")

    def test_every_magic_render_goes_through_the_shared_writer(self):
        """Four styles, one encoder policy. A style with its own write call is
        a style that misses the GPU probe and the fallback."""
        import inspect

        import app

        for runner in (app._magic_duel, app._magic_minimalist,
                       app._magic_commentary, app._magic_atmosphere):
            source = inspect.getsource(runner)
            assert "write_videofile" not in source, (
                f"{runner.__name__} encodes directly instead of via the engine")


@pytest.mark.slow
def test_loading_a_preset_does_not_inherit_the_last_matchup(tmp_path, monkeypatch):
    """
    `st.session_state.slides` is shared by Reel Studio and Versus Duel and it
    holds PIL images. Nothing used to clear it on a preset switch, so loading
    Luxury SUVs and going straight to Preview or Export rendered the *previous*
    pairing's photographs under the new names.
    """
    import auth
    from streamlit.testing.v1 import AppTest

    import app

    monkeypatch.setenv("REELFORGE_EXPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("REELFORGE_USERS_FILE", str(tmp_path / "users.json"))
    monkeypatch.setenv("REELFORGE_ADMIN_USER", "admin")
    monkeypatch.setenv("REELFORGE_ADMIN_PASSWORD_HASH", auth.hash_password("duel-test-pw"))
    monkeypatch.setattr(auth, "USERS_FILE", str(tmp_path / "users.json"))

    at = AppTest.from_file(app.__file__.replace("app.py", "app.py"), default_timeout=180)
    at.run()
    at.text_input(key="login_user").set_value("admin")
    at.text_input(key="login_pass").set_value("duel-test-pw")
    at.button[0].click().run()
    assert "auth" in at.session_state

    at.session_state["app_mode"] = "duel"
    at.run()

    # Stand in for a previous build: slides from the last matchup, and its
    # rendered video still sitting in state.
    at.session_state["slides"] = [{"kind": "duel_intro", "stale": True}]
    at.session_state["duel_video_path"] = "/old/duel.mp4"
    at.session_state["duel_video_bytes"] = b"stale"
    at.run()

    presets = [b for b in at.button if (b.key or "").startswith("preset_")]
    assert presets, "no quick-fill presets on the duel page"
    presets[0].click().run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    assert at.session_state["slides"] == [], (
        "the previous matchup's slides survived the preset switch")
    for key in ("duel_video_path", "duel_video_bytes"):
        assert key not in at.session_state, f"{key} survived the preset switch"
