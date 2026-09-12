"""
The Dashboard: the landing view, and the grouped menu that reaches every mode.

Three things here are load-bearing and none of them are visible in a
screenshot:

* **Every card opens something.** A card whose `mode` is not registered is a
  dead "Open Engine" button, and a mode with no card is an engine that can only
  be found by scrolling the sidebar. Both directions are asserted.
* **The menu covers the registry.** `NAV_GROUPS` is a hand-written grouping, so
  adding a mode without giving it a home would silently drop it out of the
  sidebar entirely -- it would still route, and still be unreachable.
* **The landing page makes no network calls.** It is drawn on every sign-in and
  every refresh. `pexels_key_status()` costs an HTTP round trip on its first
  call, so reading the cached verdict rather than calling it is the difference
  between a dashboard and a stall.
"""
from __future__ import annotations

import os

import pytest

import app
import auth
import dashboard_view as dv


# ---------------------------------------------------------------------------
# The catalogue against the registry
# ---------------------------------------------------------------------------

class TestCatalogue:

    def test_every_card_names_a_registered_mode(self):
        for engine in dv.ENGINES:
            assert engine["mode"] in app.MODE_LABELS, (
                f"{engine['mode']} has a dashboard card but is not a mode")

    def test_every_production_mode_has_a_card(self):
        """A mode that renders something has to be reachable from the landing
        page, or the dashboard is a partial index of the app."""
        from test_modes import ALL_MODES, NON_PRODUCTION_MODES

        carded = {engine["mode"] for engine in dv.ENGINES}
        for mode in ALL_MODES:
            if mode in NON_PRODUCTION_MODES and mode != "scout":
                continue
            assert mode in carded, f"{mode} has no dashboard card"

    def test_there_are_eight_cards(self):
        assert len(dv.ENGINES) == 8

    def test_no_card_is_listed_twice(self):
        modes = [engine["mode"] for engine in dv.ENGINES]
        assert len(modes) == len(set(modes))

    @pytest.mark.parametrize("engine", dv.ENGINES, ids=lambda e: e["mode"])
    def test_each_card_is_complete(self, engine):
        assert engine["icon"] and engine["name"] and engine["format"]
        assert engine["lane"] in ("short", "long", "research")
        use = engine["use"]
        assert use.endswith("."), engine["mode"]
        # One sentence of real use case, not a label and not an essay.
        assert 60 < len(use) < 220, f"{engine['mode']}: {len(use)} characters"

    def test_the_lanes_split_the_catalogue(self):
        assert "atmosphere" in dv.LONG_FORM
        assert "minimalist" in dv.SHORT_FORM
        assert "scout" not in dv.SHORT_FORM and "scout" not in dv.LONG_FORM

    def test_a_creator_is_only_offered_what_they_can_open(self):
        """A card for an engine the role cannot reach is a button that lands on
        an access error."""
        creator = auth.allowed_modes("creator")
        offered = {engine["mode"] for engine in dv.engines_for(creator)}
        assert offered <= set(creator)
        assert "duel" not in offered, "duel is admin-only but was offered"
        assert "minimalist" in offered

    def test_engine_card_looks_up_by_mode(self):
        assert dv.engine_card("atmosphere")["format"] == "16:9 Long-form"
        assert dv.engine_card("nope") is None


# ---------------------------------------------------------------------------
# The stepper
# ---------------------------------------------------------------------------

class TestStepper:

    def test_there_are_three_steps_in_order(self):
        assert [step["number"] for step in dv.STEPS] == ["1", "2", "3"]

    def test_every_target_is_a_real_mode(self):
        for step in dv.STEPS:
            for target in step["targets"]:
                assert target in app.MODE_LABELS, f"step {step['number']} -> {target}"
            for _label, lane_modes in step["lanes"]:
                for mode in lane_modes:
                    assert mode in app.MODE_LABELS

    def test_step_two_offers_both_lengths(self):
        lanes = dict(dv.STEPS[1]["lanes"])
        assert set(lanes["Short-form"]) == {"minimalist", "commentary",
                                            "narrative", "duel"}
        assert lanes["Long-form"] == ("atmosphere",)

    def test_the_path_starts_at_research_and_ends_at_publishing(self):
        assert dv.STEPS[0]["targets"] == ("scout",)
        assert "batch" in dv.STEPS[2]["targets"]


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

class TestNavigation:

    def test_the_menu_covers_every_mode_exactly_once(self):
        from test_modes import ALL_MODES

        listed = [mode for _heading, group in app.NAV_GROUPS for mode in group]
        assert len(listed) == len(set(listed)), "a mode is in two groups"
        assert set(listed) == set(ALL_MODES), (
            f"menu and registry disagree: {set(listed) ^ set(ALL_MODES)}")

    def test_dashboard_is_first_and_alone(self):
        heading, group = app.NAV_GROUPS[0]
        assert group == ("dashboard",)
        assert heading == "", "the landing view sits above the group headings"

    def test_the_groups_are_named_as_specified(self):
        headings = [heading for heading, _ in app.NAV_GROUPS if heading]
        assert headings == ["Creation Engines", "Tools & Strategy"]

    def test_nothing_that_renders_is_filed_under_tools(self):
        from test_modes import NON_PRODUCTION_MODES

        tools = dict((h, g) for h, g in app.NAV_GROUPS)["Tools & Strategy"]
        for mode in tools:
            assert mode in NON_PRODUCTION_MODES, f"{mode} renders but is filed as a tool"

    def test_every_role_can_reach_the_default_landing_view(self):
        for role in ("admin", "creator"):
            assert app.DEFAULT_MODE in auth.allowed_modes(role), role

    def test_the_default_is_the_dashboard(self):
        assert app.DEFAULT_MODE == "dashboard"


# ---------------------------------------------------------------------------
# Connectivity, without the network
# ---------------------------------------------------------------------------

class TestConnectivity:

    def test_it_reports_a_row_per_service(self):
        names = [row["name"] for row in dv.api_status()]
        assert names == ["Gemini API", "YouTube Upload", "YouTube Data", "Pexels"]

    def test_every_row_explains_what_it_gates(self):
        for row in dv.api_status():
            assert row["state"] and row["detail"].endswith(".")
            assert row["tone"] in ("", "green", "amber", "cyan", "violet")

    def test_a_missing_gemini_key_is_flagged_not_hidden(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        row = dv.api_status()[0]
        assert row["state"] == "No key" and row["tone"] == "amber"

    def test_a_present_gemini_key_reads_connected(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x" * 12)
        assert dv.api_status()[0]["state"] == "Connected"

    def test_whitespace_is_not_a_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "   ")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert dv.api_status()[0]["state"] == "No key"

    def test_a_stored_token_is_what_makes_youtube_connected(self, monkeypatch):
        import publisher

        monkeypatch.setattr(publisher, "has_client_secrets", lambda: True)
        monkeypatch.setattr(publisher, "has_token", lambda: False)
        assert dv.api_status()[1]["state"] == "Unauthenticated"

        monkeypatch.setattr(publisher, "has_token", lambda: True)
        assert dv.api_status()[1]["state"] == "Connected"

    def test_the_youtube_data_row_says_it_is_not_the_gemini_key(self, monkeypatch):
        """The 401 this row exists to pre-empt: an AI Studio key is rejected by
        the YouTube Data API, and the error does not say why."""
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        detail = dv.api_status()[2]["detail"].lower()
        assert "not the gemini key" in detail

    def test_an_unverified_pexels_key_is_not_claimed_as_verified(self, monkeypatch):
        import demo_data

        monkeypatch.setenv("PEXELS_API_KEY", "x" * 12)
        monkeypatch.setitem(demo_data.PEXELS_STATE, "checked", False)
        row = dv.api_status()[3]
        assert row["state"] == "Active"
        assert "not yet verified" in row["detail"].lower() or "first footage" in row["detail"].lower()

    def test_a_rejected_pexels_key_says_so(self, monkeypatch):
        import demo_data

        monkeypatch.setenv("PEXELS_API_KEY", "x" * 12)
        monkeypatch.setitem(demo_data.PEXELS_STATE, "checked", True)
        monkeypatch.setitem(demo_data.PEXELS_STATE, "ok", False)
        monkeypatch.setitem(demo_data.PEXELS_STATE, "reason", "Pexels rejected the key.")
        row = dv.api_status()[3]
        assert row["state"] == "Rejected" and "rejected" in row["detail"].lower()

    def test_it_never_probes_the_network(self, monkeypatch):
        """The landing page is drawn on every refresh. A probe here is a stall
        in front of the whole app."""
        import demo_data

        def explode(*_args, **_kwargs):
            raise AssertionError("the dashboard made a network call")

        monkeypatch.setattr(demo_data, "pexels_key_status", explode)
        monkeypatch.setattr(demo_data.requests, "get", explode)
        monkeypatch.setenv("PEXELS_API_KEY", "x" * 12)
        dv.api_status()


# ---------------------------------------------------------------------------
# Reference content
# ---------------------------------------------------------------------------

class TestMonetizationReference:

    def test_it_covers_both_platforms_and_licensing(self):
        platforms = [group["platform"] for group in dv.platform_rules()]
        assert any("YouTube" in p for p in platforms)
        assert any("TikTok" in p for p in platforms)
        assert any("licens" in p.lower() for p in platforms)

    def test_the_thresholds_come_from_compliance_not_a_second_copy(self):
        """Two copies of a monetization threshold is one copy that goes stale
        the next time a platform moves the bar."""
        from compliance import MONETIZATION_NOTES

        youtube = next(g for g in dv.platform_rules() if "YouTube" in g["platform"])
        assert youtube["threshold"] == MONETIZATION_NOTES["YouTube"]

    def test_the_tiktok_length_gate_matches_the_publish_gate(self):
        from compliance import TIKTOK_REWARDS_MIN_SECONDS

        tiktok = next(g for g in dv.platform_rules() if "TikTok" in g["platform"])
        joined = " ".join(tiktok["rules"])
        assert f"{TIKTOK_REWARDS_MIN_SECONDS:.0f} seconds" in joined

    def test_it_names_the_three_traps_the_brief_called_out(self):
        text = " ".join(
            rule for group in dv.platform_rules() for rule in group["rules"]).lower()
        assert "templated" in text, "no warning about templated content"
        assert "second" in text, "no pacing or length threshold"
        assert "content id" in text, "nothing about audio licensing claims"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestDurationChip:

    @pytest.mark.parametrize("seconds,expected", [
        (28800.0, "8.0 h"),      # an eight-hour soundscape
        (3600.0, "1.0 h"),
        (1800.0, "30 min"),
        (125.0, "2 min"),
        (42.0, "42s"),
        (7.4, "7.4s"),
    ])
    def test_it_scales_to_the_runtime(self, seconds, expected):
        assert dv.duration_chip(seconds) == expected

    @pytest.mark.parametrize("value", [0, -1.0, None, "", "abc"])
    def test_an_unprobeable_duration_prints_nothing(self, value):
        """Better an absent chip than '0.0s' stamped on a video that is fine."""
        assert dv.duration_chip(value) == ""


# ---------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------

class TestPageStyles:

    def test_the_container_keeps_its_breathing_room(self):
        assert "padding-top: 2.5rem !important" in app.THEME_CSS

    @pytest.mark.parametrize("selector", [
        ".rf-hero", ".rf-hero-title", ".rf-step", ".rf-step-n",
        ".rf-engine", ".rf-engine-icon", ".rf-api",
    ])
    def test_the_dashboard_classes_are_defined(self, selector):
        assert selector + " {" in app.THEME_CSS or selector + "{" in app.THEME_CSS

    def test_the_dashboard_does_not_draw_a_second_kpi_bar(self):
        """render_command_center() is already painted above every page. A
        second copy collides on the purge button's widget key."""
        import inspect

        source = inspect.getsource(app.render_dashboard)
        assert "render_command_center" not in source


# ---------------------------------------------------------------------------
# The page, running
#
# One real app boot. It is the only way to catch the two failures that no
# amount of unit testing sees: landing somewhere other than the dashboard, and
# a card whose button does not actually move the router.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_app_lands_on_the_dashboard_and_the_cards_route(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("REELFORGE_EXPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("REELFORGE_USERS_FILE", str(tmp_path / "users.json"))
    monkeypatch.setenv("REELFORGE_ADMIN_USER", "admin")
    monkeypatch.setenv("REELFORGE_ADMIN_PASSWORD_HASH",
                       auth.hash_password("dashboard-test-pw"))
    monkeypatch.setattr(auth, "USERS_FILE", str(tmp_path / "users.json"))

    at = AppTest.from_file(os.path.join(os.path.dirname(app.__file__), "app.py"),
                           default_timeout=120)
    at.run()
    at.text_input(key="login_user").set_value("admin")
    at.text_input(key="login_pass").set_value("dashboard-test-pw")
    at.button[0].click().run()
    assert "auth" in at.session_state, "could not sign in"

    # Nothing chose a mode, so the landing view is the default one.
    assert at.session_state["app_mode"] == "dashboard"
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    opens = [button for button in at.button if (button.key or "").startswith("open_")]
    assert len(opens) == len(dv.ENGINES), "the engine grid is not fully drawn"

    # Developer telemetry is not on the page a customer lands on.
    assert not [b for b in at.button if (b.key or "") == "purge_scratch"], (
        "the scratch purge is still on the consumer dashboard")

    # ...but it is still reachable, in Admin, where it belongs.
    at.session_state["app_mode"] = "admin"
    at.run()
    assert [b for b in at.button if (b.key or "") == "purge_scratch"], (
        "the scratch purge did not move to Admin, it just disappeared")
    at.session_state["app_mode"] = "dashboard"
    at.run()

    at.button(key="open_atmosphere").click().run()
    assert at.session_state["app_mode"] == "atmosphere", "the card did not route"
    assert any(app.MODE_SUBTITLES["atmosphere"] in markdown.value
               for markdown in at.markdown), "the header did not follow the mode"

    at.button(key="nav_dashboard").click().run()
    assert at.session_state["app_mode"] == "dashboard"
