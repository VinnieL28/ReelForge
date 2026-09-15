"""
The regression suite for this overhaul.

Each test here corresponds to a defect that was reproduced first and is now
asserted against, so a later change that reintroduces it fails loudly:

  * the duel's asset/state bleed between presets
  * the duel's colliding titles and doubled WINNER
  * the duel's per-frame cost, which is what froze it at Stage 4/4
  * Gemini File API rejections caused by the container rather than the content
  * the viral scorecard rating a pure-template script as highly as a researched one
  * the Minimalist Motion safe area
  * the Narrative Studio's Ken Burns move and the Shorts cap
"""
from __future__ import annotations

import os
import subprocess
import time

import numpy as np
import pytest
from PIL import Image

import compliance
import duel_engine
import minimalist_engine as me
import narrative_engine as ne
import reel_engine
import vector_rig as rig
import video_engine as ve

SIZE = (1080, 1920)


def _panel(tone: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (1400, 1400), tone)


ITEM_A = {"name": "Range Rover Vogue", "hook": "British flagship luxury",
          "image": _panel((64, 72, 86))}
ITEM_B = {"name": "Porsche Cayenne", "hook": "The driver's SUV",
          "image": _panel((86, 70, 54))}
ROUND = {"metric": "Horsepower", "a_score": 395, "b_score": 468, "unit": "hp",
         "winner": "B", "note": "Peak output"}


# ---------------------------------------------------------------------------
# 1. Versus Duel
# ---------------------------------------------------------------------------

class TestDuelTitleSlot:
    """The intro hook, each round's metric header and the winner card all want
    the top of the frame. A crossfade blends whole frames, so two of them lit
    at once ghost through each other for the length of the transition."""

    def test_gate_is_closed_at_both_ends(self):
        duration = 5.5
        assert duel_engine.title_gate(0.0, duration) == 0.0
        assert duel_engine.title_gate(duration, duration) == 0.0
        # and open through the middle
        assert duel_engine.title_gate(duration / 2, duration) == 1.0

    def test_gate_clears_the_default_transition(self):
        # The UI offers transitions up to 1.5s; the slot must be empty at the
        # midpoint of the blend, which is transition/2 from each edge.
        duration = 5.0
        for transition in (0.5, 1.0, 1.5):
            half = transition / 2
            assert duel_engine.title_gate(duration - half, duration) < 0.9, transition

    def test_intro_headline_is_gone_before_the_cut(self):
        clip = duel_engine.create_duel_intro_clip(
            SIZE, ITEM_A, ITEM_B, headline="Range Rover Vogue vs Porsche Cayenne",
            duration=4.0, layout="stacked")
        band = slice(0, 180)          # where the headline lives

        mid = np.asarray(clip.get_frame(2.0))[band].max(axis=2)
        tail = np.asarray(clip.get_frame(3.98))[band].max(axis=2)

        lit_mid = float((mid > 180).mean())
        lit_tail = float((tail > 180).mean())
        assert lit_mid > 0.01, "the headline never appeared at all"
        assert lit_tail < lit_mid * 0.1, (
            f"headline still lit at the cut: {lit_tail:.4f} vs {lit_mid:.4f}")

    def test_round_header_arrives_and_leaves_empty(self):
        clip = duel_engine.create_duel_round_clip(
            SIZE, ITEM_A, ITEM_B, ROUND, duration=5.5, layout="stacked")
        band = slice(0, 200)

        head = np.asarray(clip.get_frame(0.01))[band].max(axis=2)
        mid = np.asarray(clip.get_frame(2.75))[band].max(axis=2)
        tail = np.asarray(clip.get_frame(5.48))[band].max(axis=2)

        assert float((mid > 180).mean()) > 0.01, "the metric pill never appeared"
        assert float((head > 180).mean()) < 0.004, "the pill is lit on frame one"
        assert float((tail > 180).mean()) < 0.004, "the pill is still lit at the cut"


class TestDuelWinnerCard:
    """One WINNER, in one place."""

    def test_round_winner_tag_retires_before_the_cut(self):
        """Isolates the tag by differencing against the same round with no
        winner -- panel B's stat card is amber too, so a colour threshold over
        the card region measures the card, not the tag."""
        lit = duel_engine.create_duel_round_clip(
            SIZE, ITEM_A, ITEM_B, ROUND, duration=5.5, layout="stacked")
        none = duel_engine.create_duel_round_clip(
            SIZE, ITEM_A, ITEM_B, {**ROUND, "winner": ""}, duration=5.5, layout="stacked")

        def tag_ink(t: float) -> float:
            delta = np.abs(np.asarray(lit.get_frame(t), dtype=np.int16)
                           - np.asarray(none.get_frame(t), dtype=np.int16))
            return float(delta.max(axis=2).mean())

        reveal = tag_ink(5.0)
        tail = tag_ink(5.49)
        assert reveal > 0.5, f"the WINNER tag never showed ({reveal:.3f})"
        assert tail < reveal * 0.35, (
            f"the WINNER tag survives into the cut: {tail:.3f} vs {reveal:.3f}")

    def test_winner_card_has_exactly_one_wordmark(self):
        clip = duel_engine.create_winner_clip(
            SIZE, ITEM_B, (3, 1), duration=4.5, is_a=False,
            headline="Which one would you pick?")
        frame = np.asarray(clip.get_frame(2.5))

        gold = (frame[:, :, 0] > 200) & (frame[:, :, 1] > 150) & (frame[:, :, 2] < 130)
        rows = np.where(gold.sum(axis=1) > 40)[0]
        assert len(rows), "no gold wordmark or trophy found at all"

        # Group into bands, then keep only the substantial ones. The trophy's
        # three concentric glow rings are 2-3 rows each and are not wordmarks.
        bands, start = [], rows[0]
        for prev, cur in zip(rows, rows[1:]):
            if cur - prev > 12:
                bands.append((start, prev))
                start = cur
        bands.append((start, rows[-1]))
        solid = [(a, b) for a, b in bands if b - a >= 25]

        assert len(solid) == 2, f"expected trophy badge + one wordmark, got {solid}"
        # The lower of the two is the wordmark; the upper is the badge.
        badge, wordmark = sorted(solid)
        assert badge[1] < wordmark[0], "the wordmark is above the trophy"
        assert 30 <= wordmark[1] - wordmark[0] <= 110, (
            f"the WINNER band is {wordmark[1] - wordmark[0]}px tall — "
            "that is not one line of type")

    def test_victory_block_is_vertically_centred(self):
        clip = duel_engine.create_winner_clip(
            SIZE, ITEM_B, (3, 1), duration=4.5, is_a=False, headline="Vote below")
        frame = np.asarray(clip.get_frame(2.5)).max(axis=2)

        rows = np.where((frame > 170).sum(axis=1) > 25)[0]
        # Ignore the gold vignette, which runs the full height.
        rows = rows[(rows > 120) & (rows < 1800)]
        centre = float(rows.mean())
        assert 700 < centre < 1200, (
            f"the victory block centres at y={centre:.0f}, not near the middle of the frame")

    def test_card_builds_in_rather_than_snapping(self):
        clip = duel_engine.create_winner_clip(
            SIZE, ITEM_B, (3, 1), duration=4.5, is_a=False, headline="Vote below")
        first = float(np.asarray(clip.get_frame(0.0)).max(axis=2).mean())
        held = float(np.asarray(clip.get_frame(2.5)).max(axis=2).mean())
        assert first < held, "the card is at full opacity on frame one"


class TestDuelPerformance:
    """The freeze at Stage 4/4 was the per-frame cost of rebuilding static
    layers: the panel scrims and the divider glow do not depend on t, but were
    redrawn 3,845 Python-level lines at a time for every frame."""

    def test_static_layer_is_cached(self):
        duel_engine._static_layer.cache_clear()
        duel_engine._static_layer(SIZE, "stacked",
                                  duel_engine.DUEL_ACCENT_A, duel_engine.DUEL_ACCENT_B)
        for _ in range(5):
            duel_engine._static_layer(SIZE, "stacked",
                                      duel_engine.DUEL_ACCENT_A, duel_engine.DUEL_ACCENT_B)
        info = duel_engine._static_layer.cache_info()
        assert info.hits == 5 and info.misses == 1, info

    def test_the_sequencer_matches_the_composite_it_replaced(self):
        """crossfade_sequence exists purely for speed, so it has to be
        indistinguishable from CompositeVideoClip + vfx.CrossFadeIn -- same
        length, same start times, same picture."""
        from moviepy import CompositeVideoClip, vfx

        transition = 0.5

        def build():
            return [
                duel_engine.create_duel_intro_clip(SIZE, ITEM_A, ITEM_B,
                                                   headline="A vs B", duration=2.0),
                duel_engine.create_duel_round_clip(SIZE, ITEM_A, ITEM_B, ROUND,
                                                   duration=2.5),
                duel_engine.create_winner_clip(SIZE, ITEM_B, (1, 0), duration=2.0,
                                               is_a=False, headline="Vote below"),
            ]

        raw = build()
        timeline, cursor, old_starts = [], 0.0, []
        for i, clip in enumerate(raw):
            if i == 0:
                old_starts.append(0.0)
                timeline.append(clip.with_start(0.0))
                cursor = clip.duration - transition
            else:
                old_starts.append(cursor)
                timeline.append(clip.with_start(cursor)
                                .with_effects([vfx.CrossFadeIn(transition)]))
                cursor += clip.duration - transition
        old = CompositeVideoClip(timeline)
        new, new_starts = duel_engine.crossfade_sequence(build(), transition)

        assert abs(old.duration - new.duration) < 1e-6
        assert all(abs(x - y) < 1e-6 for x, y in zip(old_starts, new_starts)), \
            "the start times moved, which slides the audio off the picture"

        worst = 0
        for t in (0.4, 1.6, 1.75, 1.9, 2.6, 3.9, 4.1, 5.0):
            a = np.asarray(old.get_frame(t), dtype=np.int16)
            b = np.asarray(new.get_frame(t), dtype=np.int16)
            worst = max(worst, int(np.abs(a - b).max()))
        assert worst <= 2, f"pictures differ by up to {worst} code values"

    def test_the_static_layers_are_built_once_not_per_frame(self):
        """
        This was an absolute budget -- 85ms a frame, from 99.9 before the fix
        and 55.5 after. The trouble with a wall-clock number is that it is a
        measurement of the machine as much as of the code: it fails on a loaded
        laptop and on a slow CI box while the caching it exists to protect is
        working perfectly, and both of those are false alarms about someone
        else's work.

        What it actually cares about is that the panel scrims and the divider
        glow are baked once and reused. `_static_layer` is the lru_cache that
        holds them, so asking it how many times it was missed answers the
        question exactly -- and counts the same on any machine, under any load.
        """
        duel_engine._static_layer.cache_clear()
        clip = duel_engine.create_duel_round_clip(
            SIZE, ITEM_A, ITEM_B, ROUND, duration=5.5, layout="stacked")

        for i in range(12):
            clip.get_frame(0.4 + i * 0.4)

        info = duel_engine._static_layer.cache_info()
        assert info.misses <= 1, (
            f"the static overlay was built {info.misses} times across 12 "
            "frames -- it is back to being rebuilt per frame")
        assert info.hits >= 11, (
            f"only {info.hits} of 12 frames reused the baked overlay")


class TestDuelPresetIsolation:
    """Selecting 'Luxury SUVs' after 'Flagship Phones' showed phone images,
    because a Streamlit widget with an existing key ignores its `value=`."""

    def test_preset_load_clears_every_widget_key(self, monkeypatch):
        import app

        stale = {
            "duel_name_a": "iPhone 15 Pro Max", "duel_q_a": "iPhone 15 Pro",
            "duel_hook_b": "The spec monster", "rm_0": "Battery", "ra_0": 4441.0,
            "duel_headline": "iPhone vs Galaxy", "duel_rounds_n": 4,
        }
        for key, value in stale.items():
            app.st.session_state[key] = value

        duel = app.duel_from_preset("🚙 Luxury SUVs")

        survivors = [k for k in stale if k in app.st.session_state]
        assert not survivors, f"stale keys would overwrite the preset: {survivors}"
        assert "Range Rover" in duel["a"]["name"]
        assert "Range Rover" in duel["a"]["query"]
        assert [r["metric"] for r in duel["rounds"]] == \
            ["Price", "Horsepower", "0-60 mph", "Off-Road"]

    def test_rounds_are_independent_copies(self):
        import app
        from demo_data import DUEL_PRESETS

        duel = app.duel_from_preset("🚙 Luxury SUVs")
        duel["rounds"][0]["a_score"] = 999_999
        assert DUEL_PRESETS["🚙 Luxury SUVs"]["rounds"][0]["a_score"] == 108_000


# ---------------------------------------------------------------------------
# 2. Gemini File API pre-flight
# ---------------------------------------------------------------------------

class TestGeminiNormaliser:
    """`GeminiError: Gemini could not process this clip` was always the
    container, never the content."""

    @staticmethod
    def _make(ffmpeg: str, path: str, *args: str) -> str | None:
        command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                   "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=1",
                   "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                   *args, path]
        proc = subprocess.run(command, capture_output=True, text=True)
        return path if proc.returncode == 0 else None

    def test_baseline_h264_passes_through_untouched(self, ffmpeg, workdir):
        path = self._make(ffmpeg, os.path.join(workdir, "ok.mp4"),
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac")
        assert path
        result = ve.normalize_for_gemini(path, dest_dir=workdir)
        assert result["transcoded"] is False
        assert result["path"] == path, "a clean file was needlessly copied"

    @pytest.mark.slow
    @pytest.mark.parametrize("name,args,why", [
        ("hevc.mp4", ["-c:v", "libx265", "-pix_fmt", "yuv420p", "-r", "30",
                      "-c:a", "aac", "-tag:v", "hvc1"], "hevc"),
        ("vp9.webm", ["-c:v", "libvpx-vp9", "-b:v", "200k", "-pix_fmt", "yuv420p",
                      "-c:a", "libopus"], "vp9"),
        ("yuv444.mp4", ["-c:v", "libx264", "-pix_fmt", "yuv444p", "-r", "30",
                        "-c:a", "aac"], "yuv444p"),
        ("fps60.mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "60",
                       "-c:a", "aac"], "60fps"),
        ("mp3.mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                     "-c:a", "mp3"], "mp3"),
        ("hdr.mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p10le", "-r", "30",
                     "-color_trc", "smpte2084", "-color_primaries", "bt2020",
                     "-c:a", "aac"], "yuv420p10le"),
    ])
    def test_awkward_containers_become_baseline(self, ffmpeg, workdir, name, args, why):
        path = self._make(ffmpeg, os.path.join(workdir, name), *args)
        if path is None:
            pytest.skip(f"this ffmpeg cannot produce {name}")

        needed, reason = ve.needs_gemini_normalise(ve.probe_stream_info(path))
        assert needed, f"{name} was waved through: {reason}"
        assert why in reason, f"{name} flagged for the wrong reason: {reason}"

        out = ve.normalize_for_gemini(path, dest_dir=workdir)
        assert out["transcoded"]
        info = out["output_info"]
        assert info["codec"] == "h264"
        assert info["pix_fmt"] == "yuv420p"
        assert abs(info["fps"] - ve.GEMINI_SAFE_FPS) < 0.1
        assert info["audio_codec"] == "aac"
        assert info["audio_rate"] == ve.GEMINI_SAFE_AUDIO_RATE

    def test_a_clip_with_no_audio_is_handled(self, ffmpeg, workdir):
        path = os.path.join(workdir, "silent.mp4")
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=1",
                        "-c:v", "libx265", "-pix_fmt", "yuv420p", "-an", path],
                       capture_output=True, check=True)
        out = ve.normalize_for_gemini(path, dest_dir=workdir)
        assert out["transcoded"]
        assert out["output_info"]["codec"] == "h264"

    def test_an_unprobeable_file_is_normalised_rather_than_trusted(self):
        needed, reason = ve.needs_gemini_normalise({})
        assert needed and "probe" in reason

    def test_upload_runs_the_preflight(self, monkeypatch, ffmpeg, workdir):
        import gemini_engine

        path = self._make(ffmpeg, os.path.join(workdir, "in.mp4"),
                          "-c:v", "libx264", "-pix_fmt", "yuv444p", "-r", "30", "-c:a", "aac")
        assert path
        uploaded: dict[str, str] = {}

        class _Files:
            def upload(self, file):
                uploaded["path"] = file
                return type("F", (), {"state": "ACTIVE", "name": "files/x"})()

        client = type("C", (), {"files": _Files()})()
        gemini_engine.upload_video(client, path)

        assert uploaded["path"] != path, "the original yuv444p file was uploaded as-is"
        assert ve.probe_stream_info(uploaded["path"])["pix_fmt"] == "yuv420p"


# ---------------------------------------------------------------------------
# 3. Viral scorecard and factual scripting
# ---------------------------------------------------------------------------

TEMPLATE_SCRIPTS = {
    "listicle": ("5 things about money that nobody told you. Number 1 will change how "
                 "you see money forever. Number 2 is the one almost everyone gets "
                 "wrong. Number 3 is what the experts stay quiet about. Follow for "
                 "part two, you don't want to miss it"),
    "mistakes": ("You're doing money wrong, here's why. Mistake 1: this quietly costs "
                 "you more than you think. Mistake 2: you're optimizing the thing "
                 "that matters least. Mistake 3: quitting right before the "
                 "compounding kicks in. Fix these and thank me later"),
    "secrets": ("Stop scrolling, the truth about fitness. Secret 1: this is what "
                "actually moves the needle with fitness. Secret 2: everyone chases "
                "the opposite of this, and loses. Secret 3: the pros built their "
                "whole system around it. Save this before it disappears"),
}

RESEARCHED = ("Stop buying the Range Rover Vogue. At 108,000 dollars it costs 29,000 "
              "more than a Porsche Cayenne that makes 468 horsepower to the Rover's "
              "395, and hits 60 mph in 4.6 seconds against 5.5. Land Rover ranked last "
              "of 32 brands in the 2024 What Car reliability survey. You are paying a "
              "29,000 dollar premium for mud you will never drive through.")

CLEAN_ENTRY = {"licence": "cc0", "duration": 30.0, "tts_provider": "edge",
               "ai_disclosed": True}


class TestViralScorecard:

    @pytest.mark.parametrize("name", sorted(TEMPLATE_SCRIPTS))
    def test_placeholder_templates_fail(self, name):
        card = compliance.viral_scorecard(TEMPLATE_SCRIPTS[name], CLEAN_ENTRY)
        assert card["overall"] < 6.0, f"{name} scored {card['overall']}"
        assert card["needs_rewrite"]

    def test_a_researched_script_passes(self):
        card = compliance.viral_scorecard(RESEARCHED, CLEAN_ENTRY)
        assert card["overall"] >= compliance.VIRAL_TARGET_SCORE, card["overall"]
        assert not card["needs_rewrite"]

    def test_list_numbering_is_not_counted_as_fact(self):
        """The original metric rated the listicle template 8.7 facts per 100
        words on nothing but its own "Number 1 / Number 2" scaffolding."""
        card = compliance.score_density(TEMPLATE_SCRIPTS["listicle"])
        assert card["per_100"] == 0.0, card["per_100"]

    def test_spelled_out_numbers_count_as_facts(self):
        """These scripts are read aloud, so a writer aiming at TTS legitimately
        writes 'twenty-three minutes'. A digits-only metric rated a researched
        narration script at 60% sentence coverage."""
        spoken = ("UC Irvine research shows context switching costs you twenty-three "
                  "minutes to refocus. Gloria Mark found our average screen attention "
                  "span has fallen to forty-seven seconds.")
        card = compliance.score_density(spoken)
        assert card["covered"] == 1.0, card["covered"]
        assert card["per_100"] > 8.0, card["per_100"]

    def test_small_spelled_numbers_are_not_facts(self):
        """'one of the reasons' and 'the first thing' are not figures, and
        counting them would let vague copy score as specific."""
        vague = ("Here are three reasons your videos are not growing. The first is "
                 "your hook. The second is your pacing. The third is your topic.")
        assert compliance.score_density(vague)["per_100"] == 0.0

    def test_filler_scores_below_mere_vagueness(self):
        vague = ("Here are three reasons your videos are not growing. The first is "
                 "your hook. The second is your pacing. The third is your topic.")
        assert (compliance.score_density(TEMPLATE_SCRIPTS["mistakes"])["score"]
                < compliance.score_density(vague)["score"])

    def test_unlicensed_footage_sinks_a_good_script(self):
        risky = {**CLEAN_ENTRY, "licence": "unverified", "tts_provider": "gemini",
                 "ai_disclosed": False}
        good = compliance.viral_scorecard(RESEARCHED, CLEAN_ENTRY)
        bad = compliance.viral_scorecard(RESEARCHED, risky)
        assert bad["overall"] < good["overall"]
        assert bad["weakest"] == "monetization"

    def test_the_verdict_names_the_weakest_axis(self):
        risky = {**CLEAN_ENTRY, "licence": "unverified"}
        card = compliance.viral_scorecard(RESEARCHED, risky)
        assert "monetization" in card["verdict"].lower()

    def test_every_axis_stays_in_range(self):
        for script in (*TEMPLATE_SCRIPTS.values(), RESEARCHED, "", "one two three"):
            card = compliance.viral_scorecard(script, CLEAN_ENTRY)
            for axis in ("hook", "density", "monetization"):
                assert 1.0 <= card[axis]["score"] <= 10.0
            assert 1.0 <= card["overall"] <= 10.0


class TestFactualScripting:

    @pytest.mark.parametrize("topic,expected", [
        ("5 money mistakes keeping you broke", "finance"),
        ("how much protein to build muscle", "fitness"),
        ("why you cant focus anymore", "productivity"),
        ("Range Rover vs Porsche Cayenne", "cars"),
        ("mind-blowing facts about space", "science"),
        ("the fall of the roman empire", "history"),
    ])
    def test_topics_route_to_the_right_domain(self, topic, expected):
        assert reel_engine.classify_topic(topic) == expected

    def test_the_offline_bank_beats_the_old_templates(self):
        script = reel_engine.build_fact_script(
            "5 money mistakes keeping you broke", count=3, use_ai=False)
        spoken = " ".join(line for _role, line in reel_engine.script_lines(script))

        banked = compliance.viral_scorecard(spoken, CLEAN_ENTRY)
        template = compliance.viral_scorecard(TEMPLATE_SCRIPTS["mistakes"], CLEAN_ENTRY)
        assert banked["density"]["score"] > template["density"]["score"] + 2.0

    def test_no_line_is_spoken_twice(self):
        script = reel_engine.build_fact_script("how much protein to build muscle",
                                               count=3, use_ai=False)
        lines = [line for _role, line in reel_engine.script_lines(script)]
        assert len(lines) == len(set(lines))

    def test_every_banked_fact_carries_a_source(self):
        for domain, facts in reel_engine.FACT_BANK.items():
            for claim, source in facts:
                assert claim.strip(), domain
                assert len(source.strip()) > 8, f"{domain}: {claim[:40]!r} has no source"

    def test_an_uncovered_topic_admits_it(self):
        script = reel_engine.build_fact_script("how to make sourdough bread",
                                               count=3, use_ai=False)
        assert script["needs_editing"] is True
        assert script["source"] == "skeleton"
        assert script["sources"] == []


# ---------------------------------------------------------------------------
# 4. Minimalist Motion
# ---------------------------------------------------------------------------

class TestMinimalistSafeArea:
    """The bottom 200px belongs to the platform's own caption and buttons."""

    @pytest.mark.parametrize("template", list(me.TEMPLATES))
    def test_nothing_is_drawn_in_the_bottom_band(self, template):
        spec = me.fallback_scene_spec("", template, 18.0)
        duration = float(spec["duration"])
        band = me.CANVAS[1] - me.SAFE_BOTTOM

        for frac in (0.15, 0.55, 0.95):
            frame = me.make_scene_frame(spec, duration * frac, duration)
            ink = float((frame[band:, :, :].max(axis=2) > 60).mean())
            assert ink < 0.002, (
                f"{template} at {frac:.0%}: {ink:.3%} ink inside the safe area")

    def test_the_clamp_is_enforced_by_the_frame_not_by_convention(self):
        frame = me.Frame()
        frame.text("CAPTION COLLISION", (540, 1900), size=60)
        arr = frame.finish(glow=0.0)
        band = me.CANVAS[1] - me.SAFE_BOTTOM
        assert float((arr[band:, :, :].max(axis=2) > 60).mean()) < 0.001

    def test_a_wrapped_block_is_lifted_whole(self):
        """Clamping each line separately would close the leading on the last
        one and the paragraph would read as a typo. The block must move as a
        unit, so a clamped render is a pure vertical translation of a free one."""
        body = ("A payoff line long enough to wrap onto three separate lines "
                "of type at this size")

        def render(y: float) -> np.ndarray:
            frame = me.Frame()
            frame.wrapped(body, (540, y), size=52, max_width=700)
            return frame.finish(glow=0.0).max(axis=2)

        clamped = render(1880)          # well inside the safe area
        free = render(900)              # nowhere near it

        band = me.CANVAS[1] - me.SAFE_BOTTOM
        assert float((clamped[band:] > 60).mean()) < 0.001, "still inside the safe area"

        top_clamped = int(np.where((clamped > 90).sum(axis=1) > 20)[0][0])
        top_free = int(np.where((free > 90).sum(axis=1) > 20)[0][0])
        shift = top_clamped - top_free
        assert shift > 0, "the clamp did not move the block down from y=900"

        rolled = np.roll(free, shift, axis=0)
        overlap = float((np.abs(clamped.astype(np.int16)
                                - rolled.astype(np.int16)) > 24).mean())
        assert overlap < 0.0008, (
            f"the clamped block is not a pure translation ({overlap:.4%} of pixels differ) "
            "— the lines were moved independently")

    def test_the_progress_hairline_is_inside_the_safe_area(self):
        assert me.PROGRESS_Y < me.SAFE_Y
        assert me.PROGRESS_Y > me.TEXT_SAFE_Y, "the hairline can collide with the payoff"


class TestMinimalistRig:
    """All eight character metaphors must actually put the rig on screen."""

    @pytest.mark.parametrize("template", list(me.CHARACTER_TYPES))
    def test_a_figure_is_visible(self, template):
        spec = me.fallback_scene_spec("", template, 18.0)
        duration = float(spec["duration"])
        with_rig = me.make_scene_frame(spec, duration * 0.6, duration)

        drawn: list[str] = []
        real = rig.draw_figure

        def spy(*args, **kwargs):
            drawn.append(str(args[1]) if len(args) > 1 else "?")
            return real(*args, **kwargs)

        rig.draw_figure = spy
        try:
            me.make_scene_frame(spec, duration * 0.6, duration)
        finally:
            rig.draw_figure = real

        assert drawn, f"{template} never calls the rig"
        assert with_rig.max() > 60, f"{template} renders nothing"

    @pytest.mark.parametrize("template", list(me.CHARACTER_TYPES))
    def test_the_figure_moves(self, template):
        """A pose that never changes is a sticker, not a character."""
        spec = me.fallback_scene_spec("", template, 18.0)
        duration = float(spec["duration"])
        early = me.make_scene_frame(spec, duration * 0.45, duration).astype(np.int16)
        later = me.make_scene_frame(spec, duration * 0.62, duration).astype(np.int16)
        assert float(np.abs(later - early).mean()) > 0.4, f"{template} is static"

    def test_every_pose_solves(self):
        for pose in rig.POSES:
            skeleton = rig.build_skeleton(pose, 0.25, (0.0, 0.0), 1.0)
            for joint in ("head", "neck", "chest", "hip", "hand_l", "hand_r",
                          "foot_l", "foot_r", "knee_l", "knee_r"):
                point = getattr(skeleton, joint)
                assert len(point) == 2 and all(np.isfinite(point)), f"{pose}.{joint}"


# ---------------------------------------------------------------------------
# 5. Narrative Studio
# ---------------------------------------------------------------------------

class TestKenBurns:

    def test_the_zoom_is_eight_percent(self):
        assert abs(ne.KEN_BURNS_ZOOM - 0.08) < 1e-9, "the move is not 1.00x to 1.08x"

    @pytest.mark.slow
    @pytest.mark.parametrize("move", list(ne._MOVES))
    def test_every_move_renders_and_moves(self, ffmpeg, workdir, move):
        from moviepy import VideoFileClip

        still = os.path.join(workdir, "src.png")
        pattern = Image.new("RGB", (1080, 1920), (12, 12, 16))
        px = pattern.load()
        for y in range(0, 1920, 24):
            for x in range(0, 1080, 24):
                if (x // 24 + y // 24) % 2 == 0:
                    for dy in range(24):
                        for dx in range(24):
                            px[x + dx, y + dy] = (235, 228, 214)
        pattern.save(still)

        out = os.path.join(workdir, f"{move}.mp4")
        ne.ken_burns_clip(still, out, 2.0, (360, 640), fps=24, move=move, ffmpeg=ffmpeg)
        assert os.path.exists(out)

        with VideoFileClip(out) as clip:
            first = np.asarray(clip.get_frame(0.0), dtype=np.float32)
            last = np.asarray(clip.get_frame(1.9), dtype=np.float32)
        assert float(np.abs(last - first).mean()) > 1.0, f"{move} does not move"

    @pytest.mark.slow
    def test_the_zoom_is_eased_not_linear(self, ffmpeg, workdir):
        from moviepy import VideoFileClip

        still = os.path.join(workdir, "src.png")
        pattern = Image.new("RGB", (1080, 1920), (12, 12, 16))
        px = pattern.load()
        for y in range(0, 1920, 24):
            for x in range(0, 1080, 24):
                if (x // 24 + y // 24) % 2 == 0:
                    for dy in range(24):
                        for dx in range(24):
                            px[x + dx, y + dy] = (235, 228, 214)
        pattern.save(still)

        out = os.path.join(workdir, "ease.mp4")
        ne.ken_burns_clip(still, out, 4.0, (360, 640), fps=24, move="in_center",
                          ffmpeg=ffmpeg)
        with VideoFileClip(out) as clip:
            seq = [np.asarray(clip.get_frame(f / 24), dtype=np.float32)
                   for f in range(0, 94, 6)]
        deltas = [float(np.abs(seq[i + 1] - seq[i]).mean()) for i in range(len(seq) - 1)]
        ends = (deltas[0] + deltas[-1]) / 2
        middle = max(deltas[len(deltas) // 2 - 1:len(deltas) // 2 + 2])
        assert middle > ends * 1.25, (
            f"linear, not eased: ends {ends:.2f} vs middle {middle:.2f}")


class TestShortsCap:

    @staticmethod
    def _board(count: int, each: float) -> list[dict]:
        cursor = 0.0
        out = []
        for i in range(count):
            out.append({"index": i + 1, "duration": each, "start": cursor,
                        "end": cursor + each, "line": f"beat {i + 1}"})
            cursor += each
        return out

    def test_a_short_board_is_left_alone(self):
        board = self._board(12, 4.0)          # 48s
        capped, trimmed = ne.cap_for_shorts(board)
        assert not trimmed
        assert ne.storyboard_runtime(capped) == pytest.approx(48.0, abs=0.1)

    def test_a_slightly_long_board_is_compressed(self):
        board = self._board(12, 5.5)          # 66s
        capped, trimmed = ne.cap_for_shorts(board)
        assert not trimmed, "compression alone should have been enough"
        assert len(capped) == 12, "beats were dropped when they did not need to be"
        assert ne.storyboard_runtime(capped) <= ne.SHORTS_MAX_SECONDS + 0.05

    def test_a_very_long_board_drops_from_the_end_and_says_so(self):
        board = self._board(60, 4.0)          # 240s, a long-form script
        capped, trimmed = ne.cap_for_shorts(board)
        assert trimmed, "beats were dropped silently"
        assert ne.storyboard_runtime(capped) <= ne.SHORTS_MAX_SECONDS + 0.05
        assert min(s["duration"] for s in capped) >= ne.MIN_SEGMENT_SECONDS - 0.01, \
            "shots were compressed below the flash-frame threshold"
        # Dropped from the tail, so the opening survives intact.
        assert capped[0]["line"] == "beat 1"

    def test_the_cap_lands_inside_the_eligibility_window(self):
        for count, each in ((12, 5.5), (20, 4.0), (40, 4.0)):
            capped, _ = ne.cap_for_shorts(self._board(count, each))
            runtime = ne.storyboard_runtime(capped)
            assert runtime <= ne.SHORTS_MAX_SECONDS + 0.05, (count, each, runtime)

    def test_the_window_clears_the_payout_floor(self):
        """
        This used to assert 55-58s, deliberately *under* sixty, because
        YouTube Shorts capped at sixty seconds. That ceiling went to three
        minutes in late 2024, and the binding constraint is now the other end:
        TikTok Creator Rewards pays nothing at or under sixty, so a band that
        sat just below it earned nothing on every episode.
        """
        import compliance

        assert ne.SHORTS_MIN_SECONDS > compliance.TIKTOK_REWARDS_MIN_SECONDS
        assert ne.SHORTS_MIN_SECONDS < ne.SHORTS_MAX_SECONDS <= 70.0


# ---------------------------------------------------------------------------
# 6. Encoder
# ---------------------------------------------------------------------------

class TestEncoder:

    def test_the_cpu_profile_is_a_real_fallback(self):
        cpu = ve.video_encoder(force_cpu=True)
        assert cpu["codec"] == "libx264"
        assert cpu["gpu"] is False
        assert "-crf" in cpu["ffmpeg_params"]

    def test_the_cpu_preset_is_fast_enough_to_finish(self):
        """On a machine with no usable NVENC -- the Docker image included --
        every render takes this path. Measured at 1080x1920 CRF 20: veryfast is
        2.18x quicker than medium and 12% smaller."""
        assert ve.CPU_PRESET == "veryfast"
        assert "-preset veryfast" in " ".join(ve.video_encoder(force_cpu=True)["cli"])

    def test_the_detected_encoder_is_usable(self):
        enc = ve.video_encoder()
        assert enc["codec"] in (ve.GPU_CODEC, ve.CPU_CODEC)
        assert isinstance(enc["ffmpeg_params"], list)

    @pytest.mark.slow
    def test_write_clip_falls_back_when_the_gpu_fails(self, monkeypatch, workdir):
        """NVENC is probed with one 256x256 frame, which is a weak promise: the
        driver can still refuse a 1080x1920 stream at the end of a long render."""
        from moviepy import VideoClip

        calls: list[str] = []
        real_write = VideoClip.write_videofile

        def flaky(self, path, **kwargs):
            calls.append(str(kwargs.get("codec")))
            if kwargs.get("codec") == ve.GPU_CODEC:
                raise RuntimeError("NVENC: no capable devices found")
            return real_write(self, path, **kwargs)

        monkeypatch.setattr(VideoClip, "write_videofile", flaky)
        monkeypatch.setattr(ve, "_encoder_cache", {
            "codec": ve.GPU_CODEC, "preset": ve.GPU_PRESET,
            "ffmpeg_params": ["-cq", ve.GPU_CQ], "cli": [], "gpu": True,
        })

        clip = VideoClip(lambda t: np.full((64, 64, 3), int(t * 40) % 255, np.uint8),
                         duration=0.5)
        out = os.path.join(workdir, "fallback.mp4")
        ve.write_clip(clip, out, fps=12, with_audio=False)

        assert calls == [ve.GPU_CODEC, ve.CPU_CODEC], calls
        assert os.path.exists(out) and os.path.getsize(out) > 0
        assert ve.probe_stream_info(out)["pix_fmt"] == "yuv420p"
