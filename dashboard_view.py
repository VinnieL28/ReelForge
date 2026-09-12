"""
The catalogue behind the Dashboard landing screen.

Deliberately free of Streamlit. `app.py` draws the page; everything here is
data or a cheap local probe, which means the catalogue can be tested without an
AppTest harness and cannot silently drift out of step with the mode registry.

Two rules this module exists to enforce:

  * Every engine card names a real mode. A card whose `mode` is not in
    `app.MODE_LABELS` is a dead "Open Engine" button, and a production mode with
    no card is an engine nobody can find from the landing page. A test asserts
    both directions.
  * The connectivity panel never touches the network. It is drawn on every
    visit to the default landing page, so a probe costing an HTTP round trip
    would put a stall in front of the whole app -- and a key that is merely
    *present* is the honest thing to report when nothing has verified it yet.
"""
from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# The engine catalogue
#
# `lane` groups the engines the way the production stepper talks about them:
# short-form vertical, long-form horizontal, and the research mode that makes
# no video at all.
# ---------------------------------------------------------------------------

ENGINES: tuple[dict[str, str], ...] = (
    {
        "mode": "minimalist",
        "icon": "◼️",
        "name": "Minimalist Motion",
        "format": "Vector Code",
        "lane": "short",
        "use": "100% original 2D vector stickman animations drawn from code, so there "
               "is nothing to license and nothing for anyone to claim.",
    },
    {
        "mode": "commentary",
        "icon": "\U0001f399️",
        "name": "Commentary Machine",
        "format": "9:16 Vertical",
        "lane": "short",
        "use": "Drop in a raw clip and get editorial commentary written against what "
               "Gemini actually watched happen, not against the title.",
    },
    {
        "mode": "narrative",
        "icon": "\U0001f4d6",
        "name": "Narrative Studio",
        "format": "Multi-Scene",
        "lane": "short",
        "use": "Episodic story videos whose storyboard is re-cut against the real "
               "narration timings, so voice and picture cannot drift apart.",
    },
    {
        "mode": "duel",
        "icon": "⚔️",
        "name": "Versus Duel",
        "format": "Multi-Layer",
        "lane": "short",
        "use": "Split-screen comparisons whose stat meters count up on the same clock "
               "the sound effects are scheduled against.",
    },
    {
        "mode": "reel",
        "icon": "\U0001f3ac",
        "name": "Reel Studio",
        "format": "9:16 Vertical",
        "lane": "short",
        "use": "Fact reels and listicles where every beat carries a researched figure "
               "with its source still attached.",
    },
    {
        "mode": "batch",
        "icon": "\U0001f4e6",
        "name": "Batch Studio",
        "format": "Unattended Queue",
        "lane": "short",
        "use": "Hand it a list of topics and it renders the whole list back to back, "
               "turning one sitting into a week of uploads.",
    },
    {
        "mode": "atmosphere",
        "icon": "\U0001f319",
        "name": "Atmosphere Studio",
        "format": "16:9 Long-form",
        "lane": "long",
        "use": "Synthesizes 30 minutes to 8 hours of seamless ambient soundscape and "
               "publishes it straight to YouTube with its own SEO.",
    },
    {
        "mode": "scout",
        "icon": "\U0001f3af",
        "name": "Niche Scout",
        "format": "Research",
        "lane": "research",
        "use": "Finds a faceless niche worth entering, measures how crowded it actually "
               "is, and feeds the winner into the production engines.",
    },
)

# Mode keys by lane, in catalogue order.
SHORT_FORM: tuple[str, ...] = tuple(e["mode"] for e in ENGINES if e["lane"] == "short")
LONG_FORM: tuple[str, ...] = tuple(e["mode"] for e in ENGINES if e["lane"] == "long")


def engines_for(allowed: Any) -> list[dict[str, str]]:
    """The catalogue filtered to the modes this role can actually open."""
    permitted = set(allowed or ())
    return [dict(engine) for engine in ENGINES if engine["mode"] in permitted]


def engine_card(mode: str) -> dict[str, str] | None:
    for engine in ENGINES:
        if engine["mode"] == mode:
            return dict(engine)
    return None


# ---------------------------------------------------------------------------
# The three-step production path
#
# The order a channel is actually built in, which is not the order the sidebar
# lists the engines in: research first, render second, publish third.
# ---------------------------------------------------------------------------

STEPS: tuple[dict[str, Any], ...] = (
    {
        "number": "1",
        "title": "Discover & Validate",
        "blurb": "Pick a niche that pays, and prove it is not already saturated before "
                 "you render a single frame. Scored on evergreen longevity, a CPM band "
                 "with its basis stated, and saturation measured from real channels.",
        "targets": ("scout",),
        "lanes": (),
    },
    {
        "number": "2",
        "title": "Script & Synthesize",
        "blurb": "Choose the shape of the output. Short-form is the vertical feed -- "
                 "Minimalist, Commentary, Narrative and Duel. Long-form is Atmosphere "
                 "Studio, where one render covers a whole night of watch time.",
        "targets": ("minimalist", "atmosphere"),
        "lanes": (("Short-form", ("minimalist", "commentary", "narrative", "duel")),
                  ("Long-form", ("atmosphere",))),
    },
    {
        "number": "3",
        "title": "Publish & Scale",
        "blurb": "Clear the viral scorecard before anything leaves the machine, queue "
                 "the rest of the season in Batch Studio, and let Atmosphere upload on "
                 "a schedule. Finished files live in the Exports Library.",
        "targets": ("batch", "library"),
        "lanes": (),
    },
)


# ---------------------------------------------------------------------------
# Connectivity
#
# Local checks only -- see the module docstring. "Present" and "verified" are
# reported as different states rather than collapsed into one green tick,
# because an invalid Pexels key behind a CDN returns HTTP 200 and used to read
# as working right up until the footage search came back empty.
# ---------------------------------------------------------------------------

def gemini_key_present() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip()
                or os.environ.get("GOOGLE_API_KEY", "").strip())


def youtube_data_key_present() -> bool:
    return bool(os.environ.get("YOUTUBE_API_KEY", "").strip())


def _pexels_row() -> dict[str, str]:
    if not os.environ.get("PEXELS_API_KEY", "").strip():
        return {"name": "Pexels", "state": "No key", "tone": "",
                "detail": "Licensed stock search falls back to Commons and its "
                          "share-alike terms."}

    # Read the cached verdict without triggering the probe: pexels_key_status()
    # makes an HTTP request the first time it is called.
    try:
        from demo_data import PEXELS_STATE

        checked = bool(PEXELS_STATE.get("checked"))
        ok = bool(PEXELS_STATE.get("ok"))
        reason = str(PEXELS_STATE.get("reason") or "")
    except Exception:
        checked, ok, reason = False, False, ""

    if checked and ok:
        return {"name": "Pexels", "state": "Active", "tone": "green",
                "detail": "Key verified against the live API."}
    if checked and not ok:
        return {"name": "Pexels", "state": "Rejected", "tone": "amber",
                "detail": reason or "The key did not authenticate."}
    return {"name": "Pexels", "state": "Active", "tone": "cyan",
            "detail": "Key present. Verified on the first footage search."}


def api_status() -> list[dict[str, str]]:
    """One row per external service: name, state, tone, and what it gates."""
    rows: list[dict[str, str]] = []

    if gemini_key_present():
        rows.append({"name": "Gemini API", "state": "Connected", "tone": "green",
                     "detail": "Scripting, niche research, scorecards and narration."})
    else:
        rows.append({"name": "Gemini API", "state": "No key", "tone": "amber",
                     "detail": "Set GEMINI_API_KEY in .env -- every scripted mode "
                               "needs it."})

    # Publishing is OAuth rather than a key, so a stored refresh token is the
    # difference between authorized and merely configured.
    try:
        from publisher import has_client_secrets, has_token

        secrets, token = has_client_secrets(), has_token()
    except Exception:
        secrets, token = False, False

    if token:
        rows.append({"name": "YouTube Upload", "state": "Connected", "tone": "green",
                     "detail": "A channel is authorized. Atmosphere Studio can publish."})
    elif secrets:
        rows.append({"name": "YouTube Upload", "state": "Unauthenticated", "tone": "amber",
                     "detail": "Client secret stored, no channel linked yet. "
                               "Authorize in Atmosphere Studio."})
    else:
        rows.append({"name": "YouTube Upload", "state": "Not configured", "tone": "",
                     "detail": "Upload an OAuth client secret in Atmosphere Studio to "
                               "publish without leaving the app."})

    if youtube_data_key_present():
        rows.append({"name": "YouTube Data", "state": "Active", "tone": "green",
                     "detail": "Competitor recon in Niche Scout. 10,000 quota units a day."})
    else:
        rows.append({"name": "YouTube Data", "state": "No key", "tone": "",
                     "detail": "Competitor recon is disabled. This is a Cloud project key, "
                               "not the Gemini key -- an AI Studio key returns 401 here."})

    rows.append(_pexels_row())
    return rows


# ---------------------------------------------------------------------------
# Platform compliance reference
# ---------------------------------------------------------------------------

def platform_rules() -> list[dict[str, Any]]:
    """
    The monetization rules worth reading before a faceless channel is built.

    The numeric thresholds are imported rather than restated: they already live
    in compliance.MONETIZATION_NOTES, which is what the publish gate checks
    against, and two copies of a threshold is one copy that goes stale.
    """
    try:
        from compliance import MONETIZATION_NOTES, TIKTOK_REWARDS_MIN_SECONDS

        youtube = MONETIZATION_NOTES.get("YouTube", "")
        tiktok = MONETIZATION_NOTES.get("TikTok", "")
        floor = float(TIKTOK_REWARDS_MIN_SECONDS)
    except Exception:                                    # pragma: no cover
        youtube, tiktok, floor = "", "", 60.0

    return [
        {
            "platform": "YouTube Partner Programme",
            "threshold": youtube,
            "rules": [
                "**Templated output is the rule that catches faceless channels.** The "
                "reused-content policy is not only about other people's footage -- the "
                "same skeleton with the nouns swapped across forty uploads reads as "
                "inauthentic even when every asset is licensed. Vary the structure, "
                "not just the subject.",
                "**Commentary has to add something of its own.** A clip plus a voice "
                "narrating what is already on screen is the textbook refusal; a clip "
                "plus an argument is not.",
                "**Shorts are judged on watched-versus-swiped.** The hook has to land "
                "in the first two seconds or the retention curve never recovers, which "
                "is exactly what the scorecard's Hook Intrigue axis measures.",
            ],
        },
        {
            "platform": "TikTok Creator Rewards",
            "threshold": tiktok,
            "rules": [
                f"**Length is a hard gate, not a preference.** Nothing under "
                f"{floor:.0f} seconds qualifies at all, however well it performs. A "
                f"45-second cut of a great idea earns nothing.",
                "**Qualifying views are not raw views.** Very short watch times are "
                "discounted, so a video that hooks and then sags can post a large view "
                "count against a small payout.",
                "**Original audio matters here too.** Trending-sound videos monetize "
                "worse than videos carrying their own audio, and a synthesized bed "
                "counts as your own.",
            ],
        },
        {
            "platform": "Audio and footage licensing",
            "threshold": "Every audio layer ReelForge synthesizes is original by "
                         "construction -- there is no licence to honour and no claim "
                         "to answer.",
            "rules": [
                "**Minimalist Motion and Atmosphere Studio are claim-proof by design.** "
                "Frames are drawn from code and every audio layer is generated, so "
                "there is no third-party rights holder anywhere in the chain.",
                "**A Content ID claim is not a strike, which is why it gets missed.** "
                "The video stays up and keeps earning -- for the claimant, across its "
                "whole runtime, not just the seconds the music plays.",
                "**Licensed stock still carries terms.** Pexels clips are cleared for "
                "commercial reuse, but presenting stock as your own footage of a real "
                "event is a separate problem from licensing, and it is the one that "
                "draws a misinformation flag.",
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def duration_chip(seconds: float) -> str:
    """
    A runtime short enough to sit on a thumbnail.

    Hours for an 8-hour soundscape, minutes for an episode, seconds for a reel.
    Returns "" for a file whose duration could not be probed, so the caller can
    omit the chip rather than print "0.0s" over a video that is fine.
    """
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    if value >= 3600:
        return f"{value / 3600:.1f} h"
    if value >= 120:
        return f"{value / 60:.0f} min"
    if value >= 10:
        return f"{value:.0f}s"
    return f"{value:.1f}s"


# ---------------------------------------------------------------------------
# Tiers and render credits
#
# Derived from the account's own finished renders this calendar month rather
# than stored as a number somewhere, because a stored counter and a folder full
# of videos disagree the first time anything goes wrong -- and the folder is
# the one telling the truth.
#
# Display only. Nothing here blocks a render: a tool that refuses to work
# because a number says so needs a billing system behind it to be honest, and
# there is not one.
# ---------------------------------------------------------------------------

TIERS: dict[str, dict[str, Any]] = {
    "free": {"label": "Free", "quota": 50},
    "pro": {"label": "Pro", "quota": 500},
    "studio": {"label": "Studio", "quota": 2000},
}
DEFAULT_TIER = "pro"

# Files that are not finished work. Mirrors app.SCRATCH_PREFIXES; kept here so
# this module does not import the app.
_SCRATCH = ("source_", "muted_", "reframed_", "narration_", "music_", "preview_")


def tier_of(user: Any) -> str:
    """The tier on an account record, defaulting rather than failing."""
    record = user if isinstance(user, dict) else {}
    name = str(record.get("tier") or DEFAULT_TIER).strip().lower()
    return name if name in TIERS else DEFAULT_TIER


def renders_this_month(root: str, now: float | None = None) -> int:
    """Finished renders in `root` whose mtime falls in the current month."""
    import os
    import time

    stamp = time.localtime(now if now is not None else time.time())
    year, month = stamp.tm_year, stamp.tm_mon

    count = 0
    if not os.path.isdir(root):
        return 0
    for entry in os.scandir(root):
        name = entry.name.lower()
        if not entry.is_file() or not name.endswith(".mp4"):
            continue
        if name.startswith(_SCRATCH) or name.endswith("_gemini.mp4"):
            continue
        try:
            made = time.localtime(entry.stat().st_mtime)
        except OSError:
            continue
        if (made.tm_year, made.tm_mon) == (year, month):
            count += 1
    return count


def credit_status(root: str, user: Any = None, now: float | None = None) -> dict[str, Any]:
    """{"tier", "tier_label", "quota", "used", "remaining"} for the status pill."""
    tier = tier_of(user)
    spec = TIERS[tier]
    quota = int(spec["quota"])
    used = renders_this_month(root, now)
    return {
        "tier": tier,
        "tier_label": str(spec["label"]),
        "quota": quota,
        "used": used,
        "remaining": max(0, quota - used),
    }
