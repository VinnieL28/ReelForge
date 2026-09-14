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

    def test_the_placeholder_names_the_subjects_the_engine_can_draw(self):
        """The strip is one engine now. The prompt has to ask for the kind of
        topic its geometry can actually argue -- an abstract claim with a
        mechanism, not a news event there would be nothing to draw for."""
        low = ms.PLACEHOLDER.lower()
        assert "psychology" in low and "finance" in low and "discipline" in low

    def test_the_quick_topics_suit_the_vector_metaphors(self):
        assert ms.QUICK_TOPICS == (
            "The Paradox of Choice", "Why Smart People Fail",
            "The Cost of Procrastination", "Dopamine Detox")
        # Each has to route to the engine it is a shortcut for.
        for topic in ms.QUICK_TOPICS:
            assert ms.detect_style(topic) == ms.PRIMARY_STYLE, topic

    def test_the_primary_style_is_the_claim_proof_one(self):
        assert ms.PRIMARY_STYLE == "minimalist"
        assert ms.style(ms.PRIMARY_STYLE)["mode"] == "minimalist"

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


# ---------------------------------------------------------------------------
# Minimalist Motion: the engine this project leans on hardest
# ---------------------------------------------------------------------------

class TestCaptionSafeArea:
    """
    Captions must clear the action rail -- like, comment, share, sound -- on
    both Shorts and TikTok, or the payoff line is read with a thumb over it.

    The margin is enforced by a clamp inside the text drawing itself rather
    than by each template remembering to stay above a line: fourteen templates
    each doing their own arithmetic is fourteen chances to get it wrong.
    """

    def test_the_margin_is_deep_enough_for_a_tall_phone(self):
        import minimalist_engine as me

        assert me.SAFE_BOTTOM >= 220, (
            f"{me.SAFE_BOTTOM}px does not clear the rail on a 20:9 screen")

    def test_the_safe_line_moves_with_the_margin(self):
        """A hardcoded y that does not track SAFE_BOTTOM is how the payoff
        ended up six pixels inside the rail when the margin was raised."""
        import minimalist_engine as me

        assert me.SAFE_Y == me.CANVAS[1] - me.SAFE_BOTTOM
        assert me.TEXT_SAFE_Y < me.SAFE_Y

    def test_the_payoff_is_not_anchored_to_a_fixed_y(self):
        import inspect

        import minimalist_engine as me

        source = inspect.getsource(me.draw_footer)
        assert "TEXT_SAFE_Y" in source, "the payoff is not anchored to the safe line"
        assert "1720" not in source, "the payoff is back on a hardcoded y"

    @pytest.mark.parametrize("size", [36, 44, 52, 64, 72])
    def test_a_clamped_line_never_paints_into_the_rail(self, size):
        """The clamp reserves size * _INK_BELOW_ANCHOR below the anchor. That
        figure was 0.62, a baseline-to-descender ratio, which is not what a
        centred anchor needs: measured ink ran 36px below a 52px line against
        the 32px reserved, and six of those landed in the rail."""
        import numpy as np

        import minimalist_engine as me

        frame = me.Frame()
        # Ask for a line below the safe area and let the clamp pull it up.
        frame.text("QUIETLY COMPOUNDING, ypqjg", (540, 1900), size, me.WHITE, "bold")
        array = np.asarray(frame.finish())
        below = array[int(me.SAFE_Y):, :, :]
        assert int((below.max(axis=2) > 40).sum()) == 0, (
            f"{size}px text painted into the action rail")

    @pytest.mark.parametrize("metaphor", [
        "staircase_progress", "compounding_jar", "balance_scale", "split_path",
        "gravity_funnel", "domino_chain", "comparison_split", "steep_staircase",
        "delusion_mirror", "chain_anchor", "growth_consistency",
        "sisyphus_boulder", "discipline_iceberg", "two_doors",
    ])
    def test_every_metaphor_draws_without_error(self, metaphor):
        """All fourteen, across the whole timeline. A template that raises only
        at the impact frame passes any test that renders one frame."""
        import numpy as np

        import minimalist_engine as me

        spec = me.normalise_spec(me.fallback_scene_spec("a test", metaphor, 8.0))
        for t in (0.5, 2.0, 4.0, 6.0, 7.5):
            array = np.asarray(me.make_scene_frame(spec, t, 8.0))
            assert array.shape == (1920, 1080, 3)
            assert array.dtype.name == "uint8"

    @pytest.mark.parametrize("metaphor", [
        "compounding_jar", "balance_scale", "split_path", "domino_chain",
        "growth_consistency",
    ])
    def test_template_geometry_only_grazes_the_boundary(self, metaphor):
        """
        Five templates put line-work within a few pixels of the safe line.
        That is antialiasing on a stroke rather than a caption, and it leaves
        the buttons 217 of their 220 pixels -- but it is pinned here so a
        redesign that walks real geometry into the rail is caught.
        """
        import numpy as np

        import minimalist_engine as me

        spec = me.normalise_spec(me.fallback_scene_spec("a test", metaphor, 8.0))
        deepest = int(me.SAFE_Y)
        for t in (0.5, 2.0, 4.0, 6.0, 7.5):
            array = np.asarray(me.make_scene_frame(spec, t, 8.0))
            ys, _ = np.nonzero(array[int(me.SAFE_Y):, :, :].max(axis=2) > 40)
            if len(ys):
                deepest = max(deepest, int(me.SAFE_Y) + int(ys.max()))
        assert deepest - int(me.SAFE_Y) <= 8, (
            f"{metaphor} reaches {deepest - int(me.SAFE_Y)}px into the action rail")


class TestMonetizableLength:

    def test_scene_scripts_target_a_payable_runtime(self):
        """TikTok Creator Rewards counts nothing at or under 60 seconds. The
        band was 15-25s, so every vector render earned exactly zero there."""
        import gemini_engine

        assert gemini_engine.SCENE_MIN_SECONDS > compliance.TIKTOK_REWARDS_MIN_SECONDS
        assert gemini_engine.SCENE_MIN_SECONDS >= 62
        # Locked to a narrow band: long enough to be paid for, short enough
        # that nobody scrolls before the payoff.
        assert gemini_engine.SCENE_MAX_SECONDS <= 70

    def test_the_word_budget_fills_that_runtime(self):
        import gemini_engine as ge

        # The budget must land inside the band at the rate this engine really
        # speaks -- measured at 1.875 words/second by rendering one, not
        # estimated. 150 words came back at 80.0s and was clipped.
        rate = ge.SCENE_WORDS_PER_SECOND
        assert ge.SCENE_MIN_WORDS / rate >= ge.SCENE_MIN_SECONDS - 1, (
            "the minimum budget speaks for less than the minimum runtime")
        assert ge.SCENE_MAX_WORDS / rate <= ge.SCENE_MAX_SECONDS, (
            f"{ge.SCENE_MAX_WORDS} words runs {ge.SCENE_MAX_WORDS / rate:.0f}s, "
            f"past the {ge.SCENE_MAX_SECONDS:.0f}s band")

    def test_the_prompt_actually_asks_for_it(self):
        import gemini_engine as ge

        prompt = ge.build_scene_prompt("a test concept", "auto", 70.0)
        assert f"{ge.SCENE_MIN_WORDS}-{ge.SCENE_MAX_WORDS} words" in prompt
        assert f"{ge.SCENE_MIN_SECONDS:.0f}" in prompt
        assert f"{ge.SCENE_MAX_SECONDS:.0f}" in prompt

    def test_the_render_ceiling_clears_the_target_band(self):
        """MAX_DURATION was 60.0 -- precisely the number that earns nothing --
        so the cap silently clipped every scene to just short of payable."""
        import gemini_engine as ge
        import minimalist_engine as me

        assert me.MAX_DURATION >= ge.SCENE_MAX_SECONDS


class TestNoBlockingRenders:

    def test_every_ffmpeg_call_has_a_timeout(self):
        """A wedged ffmpeg inside a Streamlit script run is a browser tab that
        never comes back, and it looks identical to a slow render."""
        import glob
        import re

        untimed = []
        for path in sorted(glob.glob("*.py")):
            source = open(path, encoding="utf-8").read()
            for match in re.finditer(
                    r"subprocess\.(run|check_output)\((?:[^()]|\([^()]*\))*\)",
                    source, re.DOTALL):
                if "timeout=" not in match.group(0):
                    line = source[:match.start()].count("\n") + 1
                    untimed.append(f"{path}:{line}")
        assert not untimed, f"subprocess calls with no timeout: {untimed}"

    def test_the_one_click_render_runs_off_the_script_thread(self):
        import inspect

        import app

        source = inspect.getsource(app.start_magic_job)
        assert "threading.Thread" in source
        assert "daemon=True" in source

    def test_no_runner_touches_streamlit(self):
        """There is no ScriptRunContext on the worker thread. An st.* call
        there is at best a logged warning and at worst a wrong answer:
        session_state would resolve to a different account's export folder."""
        import inspect

        import app

        for runner in (app._magic_minimalist, app._magic_duel,
                       app._magic_commentary, app._magic_atmosphere):
            source = inspect.getsource(runner)
            assert "session_state" not in source, (
                f"{runner.__name__} reads session_state off-thread")
            assert "user_exports()" not in source, (
                f"{runner.__name__} resolves the export folder off-thread")
            assert "st.rerun" not in source and "st.markdown" not in source, (
                f"{runner.__name__} draws Streamlit from the worker thread")

    def test_the_context_is_captured_on_the_main_thread(self):
        import inspect

        import app

        source = inspect.getsource(app.magic_context)
        assert "user_exports()" in source and "session_state" in source

    def test_a_dead_worker_does_not_poll_forever(self):
        """A thread killed without reporting would otherwise leave the page
        refreshing every 0.7s for the rest of the session."""
        import threading

        job = ms.MagicJob("test", "minimalist")
        job.thread = threading.Thread(target=lambda: None)
        job.thread.start()
        job.thread.join()
        state = job.snapshot()
        assert state["done"] and state["error"]


class TestPrimaryEngineEncoding:
    """
    Minimalist Motion is the engine the Dashboard renders, so its encode path
    is the one that has to survive a bad night on the GPU.
    """

    def test_it_goes_through_the_shared_writer(self):
        """A direct write_videofile gets NVENC but no retry: when the driver
        gives up on a 1080x1920 stream -- another process holding the session,
        or a consumer card's concurrent-session cap -- the failure lands at the
        end of a render that now runs over a minute, and is thrown away."""
        import inspect

        import minimalist_engine as me

        source = inspect.getsource(me.render_animation)
        assert "write_clip(" in source, "the encode bypasses the shared writer"
        assert "clip.write_videofile" not in source, (
            "the encode calls write_videofile directly, so there is no CPU retry")

    def test_the_writer_forces_a_pixel_format_every_phone_decodes(self):
        """libx264 picks yuv444p for some inputs and the file then plays as a
        green screen on iOS -- which for a TikTok-first engine is the audience."""
        import inspect

        import video_engine

        source = inspect.getsource(video_engine.write_clip)
        assert source.count("yuv420p") >= 2, "only one path forces yuv420p"

    def test_the_result_reports_the_encoder_that_actually_ran(self):
        """A fallback poisons the probe, so the flag has to be read after the
        write rather than before it."""
        import inspect

        import minimalist_engine as me

        source = inspect.getsource(me.render_animation)
        write_at = source.index("write_clip(")
        probe_at = source.rindex("video_encoder()")
        assert probe_at > write_at, (
            "the gpu flag is read before the encode, so a CPU fallback is "
            "reported as a GPU render")
