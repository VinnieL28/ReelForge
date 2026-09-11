"""
Niche Scout: the parts that are computed rather than generated.

The model's opinions are not testable and are not tested. What is tested is
everything the module deliberately refuses to ask a model for -- the CPM band,
the saturation score, the recon arithmetic -- plus the handoffs, because a
handoff that silently fails to populate the target mode looks exactly like one
that worked.

None of this needs a network. The YouTube calls go through one seam, `_yt_get`,
which is replaced with recorded payloads.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import niche_engine as ne


def _stamp(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _channel(name: str, subs: int, median: int, videos: int = 8) -> dict:
    return {"channel_id": name, "name": name, "subscribers": subs,
            "hidden_subs": False, "total_views": median * 50,
            "videos_in_window": videos, "median_views": median,
            "view_to_sub": (median / subs) if subs else 0.0,
            "top_videos": [{"title": f"{name} hit", "views": median * 3,
                            "published": "2026-08-01", "video_id": "x"}],
            "url": ""}


# ---------------------------------------------------------------------------
# CPM
# ---------------------------------------------------------------------------

class TestCpmBands:

    @pytest.mark.parametrize("topic,band", [
        ("Ancient Mysteries and lost cities", "history"),
        ("Dark Psychology and manipulation tactics", "psychology"),
        ("Military logistics explained", "education"),
        ("Megaprojects and civil engineering", "education"),
        ("Passive income and dividend investing", "finance"),
        ("AI tools for small business", "tech_b2b"),
        ("Rain sounds for sleeping", "relaxation"),
        ("Gaming reaction compilations", "entertainment"),
        ("Knitting patterns", "general"),
    ])
    def test_topics_land_in_the_right_band(self, topic, band):
        assert ne.classify_band(topic) == band

    def test_word_boundaries_are_respected(self):
        """`"ai" in "military logistics explained"` is True, which filed
        documentaries under enterprise software."""
        assert ne.classify_band("Military logistics explained") != "tech_b2b"
        assert ne.classify_band("The Spanish Armada explained") != "tech_b2b"

    def test_a_decisive_hint_beats_an_ambiguous_one(self):
        """"AI tools for small business" matches both `ai` and `business`; a
        length heuristic picked the longer, ambiguous word."""
        assert ne.classify_band("AI tools for small business") == "tech_b2b"

    def test_every_band_gives_a_range_and_never_a_number(self):
        for key, spec in ne.CPM_BANDS.items():
            assert spec["low"] < spec["high"], key
            assert len(spec["why"]) > 30, f"{key} does not say why"

    def test_the_estimate_carries_its_own_caveat(self):
        """A CPM without its basis reads as a measurement of this niche."""
        estimate = ne.cpm_estimate("ancient mysteries")
        assert "not a measurement" in estimate["caveat"]
        assert estimate["low"] < estimate["high"]

    def test_an_unknown_band_falls_back_rather_than_raising(self):
        assert ne.cpm_estimate("x", band="not_a_band")["band"] == ne.classify_band("x")

    def test_relaxation_pays_less_than_finance(self):
        """The ordering is the useful part -- if it inverted, the whole card
        would be pointing people at the wrong niches."""
        assert ne.CPM_BANDS["relaxation"]["high"] < ne.CPM_BANDS["finance"]["low"]


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------

class TestSaturation:

    def test_small_channels_with_big_reach_read_as_open(self):
        recon = {"channels": [_channel("A", 3_000, 420_000), _channel("B", 8_000, 310_000),
                              _channel("C", 22_000, 280_000), _channel("D", 41_000, 350_000)]}
        result = ne.saturation_from_recon(recon)
        assert result["score"] <= 35, result
        assert result["measured"] is True
        assert "Open" in result["verdict"]

    def test_one_giant_and_some_minnows_reads_as_closed(self):
        recon = {"channels": [_channel("A", 4_200_000, 2_400_000),
                              _channel("B", 900_000, 180_000),
                              _channel("C", 1_100_000, 140_000),
                              _channel("D", 780_000, 120_000)]}
        result = ne.saturation_from_recon(recon)
        assert result["score"] >= 61, result
        assert "Crowded" in result["verdict"]

    def test_the_ordering_holds_across_all_three_shapes(self):
        open_ = ne.saturation_from_recon({"channels": [
            _channel("A", 3_000, 420_000), _channel("B", 8_000, 310_000),
            _channel("C", 22_000, 280_000), _channel("D", 41_000, 350_000)]})["score"]
        middle = ne.saturation_from_recon({"channels": [
            _channel("A", 180_000, 190_000), _channel("B", 240_000, 150_000),
            _channel("C", 90_000, 120_000), _channel("D", 310_000, 210_000)]})["score"]
        closed = ne.saturation_from_recon({"channels": [
            _channel("A", 4_200_000, 2_400_000), _channel("B", 900_000, 180_000),
            _channel("C", 1_100_000, 140_000), _channel("D", 780_000, 120_000)]})["score"]
        assert open_ < middle < closed, (open_, middle, closed)

    def test_no_data_reads_as_unknown_not_as_open(self):
        """The dangerous failure: a missing API key rendering as a wide-open
        niche, which is an invitation to enter a market blind."""
        for empty in ({}, {"channels": []}, {"channels": [_channel("A", 1, 1)]}):
            result = ne.saturation_from_recon(empty)
            assert result["score"] == 50, result
            assert result["measured"] is False
            assert "Not enough data" in result["verdict"]

    def test_the_score_stays_in_range(self):
        for subs, median in ((1, 10_000_000), (50_000_000, 1), (1, 1)):
            result = ne.saturation_from_recon({"channels": [
                _channel("A", subs, median), _channel("B", subs, median)]})
            assert 1 <= result["score"] <= 99, (subs, median, result)


# ---------------------------------------------------------------------------
# Competitor recon
# ---------------------------------------------------------------------------

class TestRecon:

    @pytest.fixture()
    def recorded(self, monkeypatch):
        """Recorded API payloads, so the parsing is exercised without network."""
        videos = {
            "UU_big": [("Rome's Lost Legion", 900_000, 10),
                       ("The Vanished Fleet", 700_000, 40),
                       ("Ancient Engineering", 500_000, 70),
                       ("Far Too Old To Count", 100, 400)],     # outside the window
            "UU_small": [("Iram of the Pillars", 620_000, 20),
                         ("The Desert City", 310_000, 55),
                         ("Sand and Silence", 180_000, 85)],
        }
        state = {"playlist": None}

        def fake_get(endpoint, params):
            if endpoint == "search":
                return {"items": [{"snippet": {"channelId": c}} for c in
                                  ("UC_big", "UC_big", "UC_big", "UC_small", "UC_small")]}
            if endpoint == "channels":
                return {"items": [
                    {"id": "UC_big", "snippet": {"title": "Big History Co"},
                     "statistics": {"subscriberCount": "1200000", "viewCount": "300000000"},
                     "contentDetails": {"relatedPlaylists": {"uploads": "UU_big"}}},
                    {"id": "UC_small", "snippet": {"title": "Small Upstart"},
                     "statistics": {"subscriberCount": "9000", "viewCount": "4000000"},
                     "contentDetails": {"relatedPlaylists": {"uploads": "UU_small"}}}]}
            if endpoint == "playlistItems":
                state["playlist"] = params["playlistId"]
                return {"items": [{"contentDetails": {"videoId": f"v{i}"}}
                                  for i in range(len(videos[state["playlist"]]))]}
            if endpoint == "videos":
                return {"items": [
                    {"id": f"v{i}", "snippet": {"title": t, "publishedAt": _stamp(d)},
                     "statistics": {"viewCount": str(v)}}
                    for i, (t, v, d) in enumerate(videos[state["playlist"]])]}
            raise AssertionError(endpoint)

        monkeypatch.setattr(ne, "_yt_get", fake_get)
        return videos

    def test_videos_outside_the_window_are_excluded(self, recorded):
        recon = ne.competitor_recon("ancient mysteries")
        big = next(c for c in recon["channels"] if c["name"] == "Big History Co")
        assert big["videos_in_window"] == 3, "the 400-day-old video was counted"
        assert big["median_views"] == 700_000

    def test_the_top_three_are_the_top_three(self, recorded):
        recon = ne.competitor_recon("ancient mysteries")
        big = next(c for c in recon["channels"] if c["name"] == "Big History Co")
        assert len(big["top_videos"]) == 3
        views = [v["views"] for v in big["top_videos"]]
        assert views == sorted(views, reverse=True)
        assert views[0] == 900_000

    def test_the_view_to_sub_ratio_is_what_reveals_the_upstart(self, recorded):
        """A 9k-subscriber channel with a 310k median is the signal that the
        niche is open. Subscriber count alone would rank it last."""
        recon = ne.competitor_recon("ancient mysteries")
        small = next(c for c in recon["channels"] if c["name"] == "Small Upstart")
        big = next(c for c in recon["channels"] if c["name"] == "Big History Co")
        assert small["view_to_sub"] > big["view_to_sub"] * 10

    def test_the_quota_cost_is_reported(self, recorded):
        """search.list costs 100 of 10,000 a day. A user who does not know that
        will wonder why the tool stopped working by lunchtime."""
        recon = ne.competitor_recon("ancient mysteries")
        assert recon["quota_units"] == ne.SEARCH_COST_UNITS + 1 + 4

    def test_an_empty_search_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(ne, "_yt_get", lambda e, p: {"items": []})
        recon = ne.competitor_recon("a niche nobody covers")
        assert recon["channels"] == []
        assert ne.saturation_from_recon(recon)["measured"] is False

    def test_an_empty_query_is_refused(self):
        with pytest.raises(ne.NicheError):
            ne.competitor_recon("   ")

    def test_a_missing_key_explains_which_key(self, monkeypatch):
        """A Gemini key returns 401 here, and the message has to say so or the
        user will paste the wrong one twice."""
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        monkeypatch.delenv("YT_API_KEY", raising=False)
        with pytest.raises(ne.NicheError, match="separate key"):
            ne._yt_get("search", {})


# ---------------------------------------------------------------------------
# SEO packaging
# ---------------------------------------------------------------------------

class TestChapters:

    def test_chapters_on_their_own_lines_are_read(self):
        body = "A lost city.\n\nChapters:\n0:00 Intro\n2:15 The account\n4:40 Ubar"
        assert ne.chapters_of(body) == [
            ("0:00", "Intro"), ("2:15", "The account"), ("4:40", "Ubar")]

    def test_an_inline_chapter_run_is_read(self):
        """Models routinely return the whole run on one line, and a
        line-anchored parser finds none of it."""
        body = ("Cahokia was larger than London. Chapters: 0:00 The Pyramid City, "
                "2:15 Rise of a Power, 4:40 Stress, 7:10 Abandonment")
        found = ne.chapters_of(body)
        assert len(found) == 4, found
        assert found[0] == ("0:00", "The Pyramid City")

    def test_an_inline_run_is_rewritten_onto_separate_lines(self):
        """Not cosmetic: YouTube only builds a clickable chapter list when each
        timestamp starts its own line, so the copied description would produce
        no chapters at all."""
        body = ("Cahokia was larger than London. Chapters: 0:00 The Pyramid City, "
                "2:15 Rise of a Power, 4:40 Stress, 7:10 Abandonment")
        fixed = ne.normalise_chapters(body)
        lines = fixed.splitlines()
        starts = [line for line in lines if line.startswith(("0:00", "2:15", "4:40", "7:10"))]
        assert len(starts) == 4, fixed
        assert lines[0].startswith("Cahokia"), "the prose was lost"
        assert "Chapters:" in fixed

    def test_a_correct_description_is_left_alone(self):
        body = "A lost city.\n\nChapters:\n0:00 Intro\n2:15 The account\n4:40 Ubar"
        assert ne.normalise_chapters(body) == body

    def test_a_description_with_no_chapters_is_left_alone(self):
        assert ne.normalise_chapters("Just prose.") == "Just prose."

    def test_tags_are_trimmed_to_youtube_s_budget(self):
        tags = ne._trim_tags([f"a-fairly-long-search-tag-number-{i}" for i in range(40)])
        assert sum(len(t) for t in tags) + max(0, len(tags) - 1) <= ne.TAGS_TOTAL_LIMIT

    def test_the_seo_block_carries_the_whole_package(self):
        topic = {"hook_title": "Iram of the Pillars", "why_it_works": "Curiosity gap.",
                 "seo_title": "Iram Explained", "description": "Body.\n0:00 Intro",
                 "tags": ["iram", "lost cities"]}
        block = ne.seo_block(topic)
        for section in ("TITLE", "THUMBNAIL / HOOK", "WHY IT WORKS", "DESCRIPTION", "TAGS"):
            assert section in block, section
        assert "iram, lost cities" in block


# ---------------------------------------------------------------------------
# Handoffs
# ---------------------------------------------------------------------------

TOPIC = {
    "hook_title": "Iram of the Pillars: The City the Desert Ate",
    "why_it_works": "Curiosity gap: a named place with no agreed location.",
    "seo_title": "Iram of the Pillars Explained",
    "description": "A lost city.\n\nChapters:\n0:00 Intro\n2:15 Ubar",
    "tags": ["iram of the pillars", "lost cities"],
}


class TestHandoff:

    def test_batch_gets_the_hook_title(self):
        assert ne.batch_handoff(TOPIC) == TOPIC["hook_title"]

    def test_batch_falls_back_to_the_seo_title(self):
        assert ne.batch_handoff({"seo_title": "Fallback"}) == "Fallback"

    def test_narrative_carries_the_premise_and_the_angle(self):
        payload = ne.narrative_handoff(TOPIC, "history")
        assert TOPIC["hook_title"] in payload["topic"]
        assert TOPIC["why_it_works"] in payload["topic"]

    @pytest.mark.parametrize("band,aesthetic", [
        ("history", "nordic_noir"),
        ("education", "flat_minimal"),
        ("psychology", "vector_comic"),
        ("relaxation", "lofi_anime"),
    ])
    def test_the_look_follows_the_band(self, band, aesthetic):
        """A history channel and a finance channel want different films.
        Defaulting both to vector comic is how every episode ends up the same."""
        assert ne.narrative_handoff(TOPIC, band)["aesthetic"] == aesthetic

    def test_an_unknown_band_still_produces_a_usable_payload(self):
        payload = ne.narrative_handoff(TOPIC, "not_a_band")
        assert payload["aesthetic"] and payload["tone"] and payload["format"]

    def test_the_payload_matches_what_narrative_studio_expects(self):
        import narrative_engine as narrative

        payload = ne.narrative_handoff(TOPIC, "history")
        assert payload["aesthetic"] in narrative.VISUAL_AESTHETICS
        assert payload["tone"] in narrative.NARRATIVE_TONES
        assert payload["format"] in narrative.DURATION_FORMATS


class TestIsolation:

    def test_the_engine_imports_no_rendering_module(self):
        """Niche Scout is a feeder. It must not be able to break a renderer."""
        source = open(ne.__file__, encoding="utf-8").read()
        head = source[:source.index("def _gemini")]
        for engine in ("video_engine", "duel_engine", "minimalist_engine",
                       "ambient_engine", "reel_engine", "audio_engine"):
            assert engine not in head, f"{engine} is imported at module scope"

    def test_narrative_is_only_touched_for_its_vocabulary(self):
        """The handoff names an aesthetic and a tone; it does not call the
        engine. The import in the test above is the test's, not the module's."""
        source = open(ne.__file__, encoding="utf-8").read()
        assert "narrative_engine" not in source
