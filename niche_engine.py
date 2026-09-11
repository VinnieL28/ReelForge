# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""
Niche Scout -- research and strategy for a faceless channel.

Standalone: no rendering engine imports this, and it imports none of them. It
produces text and numbers, and hands them to the production modes as plain
dicts.

Two things here are deliberately *not* asked of a language model, because a
model will answer confidently either way and the answer would be fiction:

* **CPM.** A model does not know what a niche pays. It will happily invent
  "$18-24 RPM" for anything. What is real is that advertiser categories have
  broadly reported ranges, so the number comes from a table below with its
  basis stated, and the model's job is reduced to classifying the niche into a
  category -- which it can actually do.

* **Saturation.** This is computed from YouTube search results, not guessed.
  The signal that a niche is open is not "how many videos exist"; it is whether
  small channels are getting large views. A 2,000-subscriber channel with
  400,000 views on a 3-week-old upload says more about the opportunity than any
  opinion could.

Competitor recon needs a YouTube Data API v3 key, which is not the same as a
Gemini key -- an AI Studio key returns 401 against this API. Everything else
works without one.
"""

from __future__ import annotations

import os
import re
import statistics
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

ProgressFn = Callable[[str], None]

YOUTUBE_API = "https://www.googleapis.com/youtube/v3"
RECON_WINDOW_DAYS = 90

# search.list costs 100 quota units against a 10,000/day default; every other
# call here costs 1. One recon is therefore ~110 units, and a project can run
# roughly 90 of them a day before the quota resets at midnight Pacific.
SEARCH_COST_UNITS = 100


class NicheError(RuntimeError):
    """Raised with a message meant for the person looking at the screen."""


# ---------------------------------------------------------------------------
# CPM reference
#
# These are advertiser-category ranges, widely reported by creators and ad
# networks for a US/UK/CA/AU-weighted audience. They are NOT a measurement of
# any particular channel, and the spread inside a band is larger than the gap
# between bands: the same niche pays perhaps 3x more in Q4 than in January, and
# perhaps 5x more to a US audience than to a global one.
#
# They live here rather than in a prompt because a model asked for a CPM will
# invent a precise-sounding number for anything, including niches that do not
# monetize at all.
# ---------------------------------------------------------------------------

CPM_BANDS: dict[str, dict[str, Any]] = {
    "finance": {
        "label": "Finance, investing, business",
        "low": 12.0, "high": 35.0,
        "why": "Brokerages, banks and B2B software bid hard for this audience, "
               "and a viewer who watches investing content is worth a lot to them.",
    },
    "tech_b2b": {
        "label": "Technology, software, AI",
        "low": 8.0, "high": 24.0,
        "why": "SaaS and hardware advertisers with real customer lifetime value.",
    },
    "health": {
        "label": "Health, fitness, longevity",
        "low": 6.0, "high": 18.0,
        "why": "Supplements and insurance pay well; medical claims policy caps "
               "how far it can be pushed.",
    },
    "education": {
        "label": "Education, science, documentary",
        "low": 4.0, "high": 12.0,
        "why": "Course platforms and universities, and an audience that watches "
               "to the end — high watch time compounds the effective rate.",
    },
    "history": {
        "label": "History, mystery, true crime",
        "low": 3.5, "high": 10.0,
        "why": "Broad consumer advertisers. Long watch time is the earner here, "
               "not the rate.",
    },
    "psychology": {
        "label": "Psychology, self-improvement, stoicism",
        "low": 4.0, "high": 14.0,
        "why": "Apps, courses and coaching, with a receptive audience — though "
               "the field is crowded enough to depress the auction.",
    },
    "entertainment": {
        "label": "Entertainment, gaming, reaction",
        "low": 1.5, "high": 6.0,
        "why": "Large audiences, young skew, low advertiser intent.",
    },
    "relaxation": {
        "label": "Sleep, ambient, meditation",
        "low": 1.0, "high": 5.0,
        "why": "Watch time is enormous and rates are low, because a sleeping "
               "viewer converts for nobody. It earns on volume, not on rate.",
    },
    "general": {
        "label": "General interest",
        "low": 2.0, "high": 8.0,
        "why": "No advertiser category bids specifically for this audience, so "
               "the rate is whatever the general auction pays that month.",
    },
}
DEFAULT_BAND = "general"

# Keywords that place a topic in a band without asking a model, weighted by how
# decisive each one is. The weights are the point: a length heuristic filed "AI
# tools for small business" under finance, because "business" is longer than
# "ai" -- when "ai" is the decisive word and "business" is the ambiguous one.
#
# 3 = this word alone settles it. 1 = a hint that needs company.
_BAND_HINTS: dict[str, dict[str, int]] = {
    "finance": {"investing": 3, "stock market": 3, "dividend": 3, "passive income": 3,
                "crypto": 3, "real estate": 3, "personal finance": 3, "wealth": 2,
                "money": 2, "millionaire": 2, "mortgage": 2, "debt": 2, "tax": 2,
                "finance": 2, "business": 1, "entrepreneur": 1},
    "tech_b2b": {"ai": 3, "software": 3, "saas": 3, "programming": 3, "coding": 3,
                 "cyber": 3, "startup": 2, "tech": 2, "gadget": 2, "computer": 1},
    "health": {"nutrition": 3, "workout": 3, "weight loss": 3, "supplement": 3,
               "longevity": 3, "fitness": 2, "diet": 2, "muscle": 2, "health": 2},
    "education": {"megaproject": 3, "engineering": 3, "logistics": 3, "documentary": 3,
                  "physics": 3, "how it works": 3, "geography": 2, "science": 2,
                  "space": 2, "biology": 2, "explained": 1},
    "history": {"ancient": 3, "archaeolog": 3, "true crime": 3, "civilisation": 3,
                "civilization": 3, "empire": 2, "unsolved": 2, "mystery": 2,
                "military": 2, "history": 2, "war": 1},
    "psychology": {"dark psychology": 3, "stoic": 3, "manipulat": 3, "psychology": 3,
                   "self-improvement": 3, "mindset": 2, "discipline": 2, "habit": 2,
                   "motivation": 1},
    "entertainment": {"gaming": 3, "reaction": 3, "prank": 3, "anime": 2, "meme": 2,
                      "celebrity": 2, "drama": 1, "movie": 1},
    "relaxation": {"rain sounds": 3, "white noise": 3, "meditation": 3, "asmr": 3,
                   "ambient": 2, "lofi": 2, "sleep": 2, "relax": 1},
}


def classify_band(topic: str) -> str:
    """
    Places a topic in a CPM band from its own vocabulary.

    Two things this has to get right, both of which a naive substring match
    gets wrong:

    * **Word boundaries.** `"ai" in "military logistics explained"` is True,
      which quietly filed documentary topics under enterprise software.
    * **Specificity.** "AI tools for small business" matches `ai` and
      `business` equally, and a tie resolved by dictionary order sent it to
      finance. Hints carry explicit weights, so the decisive word wins over the
      ambiguous one regardless of length.
    """
    low = str(topic or "").lower()
    best, best_score = DEFAULT_BAND, 0

    for band, hints in _BAND_HINTS.items():
        score = sum(
            weight for word, weight in hints.items()
            if re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", low))
        if score > best_score:
            best, best_score = band, score
    return best


def cpm_estimate(topic: str, band: str = "") -> dict[str, Any]:
    """
    The CPM band for a topic, with its basis attached.

    Returns a range and never a single number, because a single number would be
    a lie of precision: the same niche pays several times more in December than
    in January, and several times more to a US audience than a global one.
    """
    key = band if band in CPM_BANDS else classify_band(topic)
    spec = CPM_BANDS[key]
    return {
        "band": key,
        "label": str(spec["label"]),
        "low": float(spec["low"]),
        "high": float(spec["high"]),
        "why": str(spec["why"]),
        "caveat": "An advertiser-category range for a US/UK/CA/AU-weighted "
                  "audience, not a measurement of this niche. Q4 runs 2-3x "
                  "January, and a global audience runs a fraction of a US one.",
    }


# ---------------------------------------------------------------------------
# Competitor recon
# ---------------------------------------------------------------------------

def youtube_key() -> str:
    """The YouTube Data API v3 key, which is not the Gemini key."""
    return (os.environ.get("YOUTUBE_API_KEY")
            or os.environ.get("YT_API_KEY") or "").strip()


def has_youtube_key() -> bool:
    return bool(youtube_key())


def _yt_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    One YouTube Data API call. The seam the tests replace.

    Errors are translated here rather than at the call site, because the two
    that actually happen -- a key with the API disabled, and an exhausted daily
    quota -- both arrive as a 403 with the reason buried in the payload.
    """
    import requests

    key = youtube_key()
    if not key:
        raise NicheError(
            "No YouTube Data API key. Competitor recon needs one — it is a "
            "separate key from the Gemini key, and an AI Studio key will not "
            "work (the API returns 401 for it). In Google Cloud Console: enable "
            "YouTube Data API v3, create an API key, and put it in .env as "
            "YOUTUBE_API_KEY.")

    try:
        response = requests.get(f"{YOUTUBE_API}/{endpoint}",
                                params={**params, "key": key}, timeout=30)
    except Exception as exc:
        raise NicheError(f"Could not reach YouTube: {type(exc).__name__}: {exc}") from exc

    if response.status_code == 200:
        return response.json()

    try:
        error = response.json().get("error", {})
        reasons = [e.get("reason", "") for e in error.get("errors", [])]
        message = str(error.get("message") or "")
    except Exception:
        reasons, message = [], response.text[:200]

    if "quotaExceeded" in reasons:
        raise NicheError(
            f"The project's daily YouTube quota is spent. One recon costs about "
            f"{SEARCH_COST_UNITS + 15} units of the default 10,000/day, so that is "
            f"roughly 90 searches. It resets at midnight Pacific.")
    if response.status_code == 401 or "required" in reasons:
        raise NicheError(
            "YouTube rejected the key. An AI Studio (Gemini) key does not work "
            "here — it needs a Google Cloud API key from a project with YouTube "
            "Data API v3 enabled.")
    if "accessNotConfigured" in reasons or "forbidden" in reasons:
        raise NicheError(
            "YouTube Data API v3 is not enabled on that key's project. Enable it "
            "in Google Cloud Console under APIs & Services → Library.")
    raise NicheError(f"YouTube returned {response.status_code}: {message or reasons}")


def _iso_days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(stamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def competitor_recon(niche: str, channels: int = 6,
                     window_days: int = RECON_WINDOW_DAYS,
                     progress: ProgressFn | None = None) -> dict[str, Any]:
    """
    Real channels succeeding in a niche, and what they are succeeding with.

    The call pattern matters for cost. A naive implementation searches per
    channel, at 100 units each; this searches once, then walks each channel's
    uploads playlist at 1 unit a page. Six channels cost ~113 units rather than
    ~700.

    Returns {"channels": [...], "query", "window_days", "quota_units"}.
    """
    query = str(niche or "").strip()
    if not query:
        raise NicheError("Give it a niche to look up.")

    if progress:
        progress(f"Searching YouTube for '{query}'...")

    found = _yt_get("search", {
        "part": "snippet", "q": query, "type": "video",
        "order": "viewCount", "maxResults": 50,
        "publishedAfter": _iso_days_ago(window_days),
        "relevanceLanguage": "en",
    })
    units = SEARCH_COST_UNITS

    # Rank candidate channels by how often they appear in the top results --
    # one viral video is luck, three is a channel that has worked the niche out.
    appearances: dict[str, int] = {}
    for item in found.get("items", []):
        channel_id = str(item.get("snippet", {}).get("channelId") or "")
        if channel_id:
            appearances[channel_id] = appearances.get(channel_id, 0) + 1
    if not appearances:
        return {"channels": [], "query": query, "window_days": window_days,
                "quota_units": units}

    ranked = sorted(appearances, key=lambda cid: -appearances[cid])[:channels]

    if progress:
        progress(f"Reading {len(ranked)} channels...")
    detail = _yt_get("channels", {
        "part": "snippet,statistics,contentDetails", "id": ",".join(ranked),
        "maxResults": 50,
    })
    units += 1

    out: list[dict[str, Any]] = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    for channel in detail.get("items", []):
        stats = channel.get("statistics", {})
        uploads = (channel.get("contentDetails", {})
                   .get("relatedPlaylists", {}).get("uploads", ""))
        if not uploads:
            continue

        if progress:
            progress(f"Reading uploads from {channel.get('snippet', {}).get('title', '?')}...")
        playlist = _yt_get("playlistItems", {
            "part": "contentDetails", "playlistId": uploads, "maxResults": 50})
        units += 1

        video_ids = [str(i.get("contentDetails", {}).get("videoId") or "")
                     for i in playlist.get("items", [])]
        video_ids = [v for v in video_ids if v][:50]
        if not video_ids:
            continue

        videos = _yt_get("videos", {
            "part": "snippet,statistics", "id": ",".join(video_ids), "maxResults": 50})
        units += 1

        recent: list[dict[str, Any]] = []
        for video in videos.get("items", []):
            published = _parse_time(video.get("snippet", {}).get("publishedAt", ""))
            if not published or published < cutoff:
                continue
            recent.append({
                "title": str(video.get("snippet", {}).get("title") or ""),
                "views": int(video.get("statistics", {}).get("viewCount") or 0),
                "published": published.strftime("%Y-%m-%d"),
                "video_id": str(video.get("id") or ""),
            })

        if not recent:
            continue

        views = [v["views"] for v in recent]
        subs = int(stats.get("subscriberCount") or 0)
        median_views = int(statistics.median(views))

        out.append({
            "channel_id": str(channel.get("id") or ""),
            "name": str(channel.get("snippet", {}).get("title") or ""),
            "subscribers": subs,
            "hidden_subs": bool(stats.get("hiddenSubscriberCount")),
            "total_views": int(stats.get("viewCount") or 0),
            "videos_in_window": len(recent),
            "median_views": median_views,
            # The number that actually says whether the algorithm is pushing
            # this channel past the people who already follow it.
            "view_to_sub": (median_views / subs) if subs > 0 else 0.0,
            "top_videos": sorted(recent, key=lambda v: -v["views"])[:3],
            "url": f"https://www.youtube.com/channel/{channel.get('id')}",
        })

    out.sort(key=lambda c: -c["median_views"])
    return {"channels": out, "query": query, "window_days": window_days,
            "quota_units": units}


def saturation_from_recon(recon: dict[str, Any]) -> dict[str, Any]:
    """
    A saturation score computed from what the recon actually found.

    Lower is more open. The three signals, in order of how much they say:

      * **Small-channel breakthrough.** If channels under 50k subs are landing
        in the top results, the niche rewards the video rather than the brand.
        This is the strongest signal there is and it is weighted accordingly.
      * **View-to-subscriber ratio.** A median above 1.0 means the algorithm is
        pushing these videos well past the existing audience -- the niche has
        demand the incumbents are not saturating.
      * **Incumbent concentration.** When one channel's median dwarfs the rest,
        the niche has an owner and a new entrant is competing with a brand.

    Returns 50 with `measured: False` when there is nothing to compute from, so
    a missing API key reads as "unknown" rather than as "wide open".
    """
    channels = list(recon.get("channels") or [])
    if len(channels) < 2:
        return {"score": 50, "verdict": "Not enough data to judge.",
                "measured": False, "signals": {}}

    subs = [c["subscribers"] for c in channels if c["subscribers"] > 0]
    medians = [c["median_views"] for c in channels if c["median_views"] > 0]
    ratios = [c["view_to_sub"] for c in channels if c["view_to_sub"] > 0]

    small = sum(1 for s in subs if s < 50_000) / len(subs) if subs else 0.0
    ratio = statistics.median(ratios) if ratios else 0.0
    # Concentration: the leader's median against the median of the rest.
    concentration = 1.0
    if len(medians) >= 2:
        leader = max(medians)
        rest = statistics.median(sorted(medians, reverse=True)[1:])
        concentration = leader / rest if rest > 0 else 4.0

    # Each term is 0 (open) to 1 (closed), then weighted.
    breakthrough_term = 1.0 - min(1.0, small * 1.6)
    demand_term = 1.0 - min(1.0, ratio / 2.0)
    concentration_term = min(1.0, (concentration - 1.0) / 4.0)

    score = int(round(100 * (breakthrough_term * 0.45
                             + demand_term * 0.35
                             + concentration_term * 0.20)))
    score = max(1, min(99, score))

    if score <= 35:
        verdict = "Open. Small channels are breaking through here."
    elif score <= 60:
        verdict = "Competitive but enterable with a sharper angle."
    else:
        verdict = "Crowded. Established channels own the results."

    return {
        "score": score,
        "verdict": verdict,
        "measured": True,
        "signals": {
            "small_channel_share": round(small, 2),
            "median_view_to_sub": round(ratio, 2),
            "leader_concentration": round(concentration, 2),
            "channels_sampled": len(channels),
        },
    }


# ---------------------------------------------------------------------------
# Gemini research
# ---------------------------------------------------------------------------

def _gemini(prompt: str, progress: ProgressFn | None = None) -> dict[str, Any]:
    """One structured-JSON call through the shared model chain."""
    import gemini_engine as ge

    client = ge.get_client()
    last: Exception | None = None
    for model in ge.MODEL_CANDIDATES:
        try:
            if progress:
                progress(f"Asking {model}...")
            response = ge.generate_with_retry(client, model, prompt, progress=None)
            parsed = ge.parse_scene_response(getattr(response, "text", "") or "")
            if parsed:
                parsed["_model"] = model
                return parsed
        except Exception as exc:
            last = exc
            continue
    raise NicheError(
        f"No model returned usable research: {type(last).__name__}: {last}" if last
        else "No model returned usable research.")


IDEAS_PROMPT = """You advise people starting faceless YouTube and TikTok channels. Return ONE
JSON object and nothing else.

Suggest exactly 12 niches that are genuinely workable for someone with no face
on camera, no crew, and a small budget. Spread them across topic areas -- do not
give twelve variations of self-improvement.

For each, be specific about the *format*, not just the subject. "History" is not
a niche; "Declassified military logistics, 8-12 minute archival-narration
documentaries" is.

Each entry:
  "name": 3-8 words, the niche as someone would describe it
  "format": the video shape that works in it -- length, structure, visual source
  "pros": 2-3 short concrete reasons it is worth entering
  "cons": 2-3 honest reasons it is hard. Do not soften these; the value of this
          list is that it is not a pitch.
  "monetization": how the channel actually earns beyond AdSense -- the specific
          sponsor category, product or affiliate that fits
  "faceless_fit": one line on why it works without a presenter
  "cpm_band": exactly one of: finance, tech_b2b, health, education, history,
          psychology, entertainment, relaxation, general

Do not state CPM or RPM figures. Do not claim view counts. Those are supplied
from a reference table, and a number you invent would look identical to one you
know.

JSON shape:
{{"niches": [{{"name": "...", "format": "...", "pros": ["..."], "cons": ["..."],
  "monetization": "...", "faceless_fit": "...", "cpm_band": "history"}}]}}"""


VALIDATE_PROMPT = """You assess whether a faceless channel niche is worth entering. Return ONE
JSON object and nothing else.

NICHE: {topic}

Be the person who talks someone out of a bad idea. A niche that does not work
should read as not working.

Fields:

"summary": 2-3 sentences on what this niche actually is as a channel, and who
  watches it.

"evergreen": {{
   "score": 1-10, where 10 means an episode still earns in three years and 1
     means it is dead in a fortnight,
   "why": one or two sentences naming the specific thing that dates it or does
     not. "News cycle dependent" and "the search demand is permanent" are the
     two poles.
}}

"faceless_fit": {{"score": 1-10, "why": "..."}} -- can this be made with stock,
  archival, AI imagery or code-drawn visuals, with no presenter?

"production_cost": {{"score": 1-10, "why": "..."}} -- 10 means cheap and fast per
  episode. Research-heavy niches score low here even when they are good niches.

"risks": 2-4 specific things that go wrong in this niche. Copyright strikes on
  archival footage, reused-content flags, medical claims policy, a topic that
  advertisers avoid. Name the actual mechanism.

"angles": 3 specific positioning angles that would differentiate a new channel,
  each one sentence and each genuinely distinct from the others.

"verdict": one honest sentence. If the answer is "do not", say "do not".

Do not state CPM, RPM or revenue figures.

JSON shape:
{{"summary": "...", "evergreen": {{"score": 0, "why": "..."}},
  "faceless_fit": {{"score": 0, "why": "..."}},
  "production_cost": {{"score": 0, "why": "..."}},
  "risks": ["..."], "angles": ["..."], "verdict": "..."}}"""


BUCKETS_PROMPT = """You plan the content strategy for a faceless channel. Return ONE JSON object
and nothing else.

NICHE: {topic}
{context}

Break the niche into exactly 7 content pillars -- recurring episode *formats*,
not subjects. A pillar is something you could publish fifty times: "Vanished
Without Explanation", "The Engineering That Failed", "What The Records Actually
Say". Each pillar should feel like a series a viewer would subscribe for.

Under each pillar, write exactly 3 specific episode concepts. Real subjects, not
placeholders -- name the event, the place, the person, the object.

Each pillar:
  "pillar": 2-5 words
  "premise": one sentence on what every episode in it does
  "topics": 3 entries, each:
     "hook_title": the title as it would appear on the thumbnail card. Specific
        and concrete, with a curiosity gap. Like "Iram of the Pillars: The City
        the Desert Ate" -- a real subject plus the thing you do not yet know.
        Under 70 characters. No ALL CAPS, no clickbait punctuation.
     "why_it_works": the psychological mechanism, named. Curiosity gap,
        unresolved ending, scale violation, taboo, identity threat, pattern
        interrupt. One sentence saying which and why it applies here.
     "seo_title": the searchable variant, under 70 characters, leading with the
        phrase someone would actually type.
     "description": 3-5 sentences, then a "Chapters:" line followed by 4-6
        timestamped chapters starting at 0:00. Realistic timings for a 10-minute
        episode.
     "tags": 8-12 lowercase search tags, total under 450 characters.

Only propose subjects that exist. If you are unsure a subject is real, choose a
different one -- a fabricated historical event is worse than a dull true one.

JSON shape:
{{"pillars": [{{"pillar": "...", "premise": "...", "topics": [
  {{"hook_title": "...", "why_it_works": "...", "seo_title": "...",
    "description": "...", "tags": ["..."]}}]}}]}}"""


def suggest_niches(progress: ProgressFn | None = None) -> dict[str, Any]:
    """Twelve workable faceless niches, with honest downsides."""
    parsed = _gemini(IDEAS_PROMPT, progress)
    niches = parsed.get("niches") or []
    if not isinstance(niches, list) or not niches:
        raise NicheError("The model returned no niches.")

    out: list[dict[str, Any]] = []
    for entry in niches[:12]:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        band = str(entry.get("cpm_band") or "").strip()
        out.append({
            "name": name,
            "format": str(entry.get("format") or "").strip(),
            "pros": [str(p).strip() for p in (entry.get("pros") or []) if str(p).strip()][:3],
            "cons": [str(c).strip() for c in (entry.get("cons") or []) if str(c).strip()][:3],
            "monetization": str(entry.get("monetization") or "").strip(),
            "faceless_fit": str(entry.get("faceless_fit") or "").strip(),
            # The model's band is accepted only if it is a real one; otherwise
            # the local classifier decides from the niche's own words.
            "cpm": cpm_estimate(name, band if band in CPM_BANDS else ""),
        })

    if not out:
        raise NicheError("The model's niches came back unusable.")
    return {"niches": out, "model": str(parsed.get("_model") or "")}


def validate_niche(topic: str, recon: dict[str, Any] | None = None,
                   progress: ProgressFn | None = None) -> dict[str, Any]:
    """
    The opportunity scorecard for one niche.

    The qualitative half comes from the model; the CPM band and the saturation
    score do not, for the reasons in the module docstring.
    """
    topic = str(topic or "").strip()
    if not topic:
        raise NicheError("Give it a niche to validate.")

    parsed = _gemini(VALIDATE_PROMPT.format(topic=topic), progress)

    def axis(key: str) -> dict[str, Any]:
        raw = parsed.get(key) or {}
        if not isinstance(raw, dict):
            raw = {}
        try:
            score = max(1.0, min(10.0, float(raw.get("score") or 0)))
        except (TypeError, ValueError):
            score = 5.0
        return {"score": score, "why": str(raw.get("why") or "").strip()}

    saturation = saturation_from_recon(recon or {})
    cpm = cpm_estimate(topic)
    evergreen = axis("evergreen")

    # The overall verdict weighs what a faceless channel actually lives on:
    # whether the library keeps earning, whether it can be made at all without a
    # presenter, and whether there is room to be seen. Cost matters least --
    # it is a constraint on pace, not on whether the niche works.
    openness = (100 - saturation["score"]) / 10.0 if saturation["measured"] else 5.0
    overall = round(
        evergreen["score"] * 0.30
        + axis("faceless_fit")["score"] * 0.25
        + openness * 0.25
        + min(10.0, cpm["high"] / 3.5) * 0.20, 1)

    return {
        "topic": topic,
        "summary": str(parsed.get("summary") or "").strip(),
        "evergreen": evergreen,
        "faceless_fit": axis("faceless_fit"),
        "production_cost": axis("production_cost"),
        "risks": [str(r).strip() for r in (parsed.get("risks") or []) if str(r).strip()][:4],
        "angles": [str(a).strip() for a in (parsed.get("angles") or []) if str(a).strip()][:3],
        "verdict": str(parsed.get("verdict") or "").strip(),
        "cpm": cpm,
        "saturation": saturation,
        "overall": overall,
        "model": str(parsed.get("_model") or ""),
    }


def content_buckets(topic: str, assessment: dict[str, Any] | None = None,
                    progress: ProgressFn | None = None) -> dict[str, Any]:
    """Seven pillars, three researched episode concepts under each."""
    topic = str(topic or "").strip()
    if not topic:
        raise NicheError("Give it a niche to plan.")

    context = ""
    if assessment:
        angles = assessment.get("angles") or []
        if angles:
            context = ("POSITIONING ANGLES ALREADY CHOSEN FOR THIS CHANNEL:\n"
                       + "\n".join(f"- {a}" for a in angles)
                       + "\nLean the pillars into these rather than ignoring them.")

    parsed = _gemini(BUCKETS_PROMPT.format(topic=topic, context=context), progress)
    pillars = parsed.get("pillars") or []
    if not isinstance(pillars, list) or not pillars:
        raise NicheError("The model returned no content pillars.")

    out: list[dict[str, Any]] = []
    for pillar in pillars[:7]:
        if not isinstance(pillar, dict):
            continue
        name = str(pillar.get("pillar") or "").strip()
        if not name:
            continue

        topics: list[dict[str, Any]] = []
        for entry in (pillar.get("topics") or [])[:3]:
            if not isinstance(entry, dict):
                continue
            hook = str(entry.get("hook_title") or "").strip()
            if not hook:
                continue
            tags = entry.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",")]
            topics.append({
                "hook_title": hook,
                "why_it_works": str(entry.get("why_it_works") or "").strip(),
                "seo_title": str(entry.get("seo_title") or hook).strip()[:100],
                # Normalised on the way in, so the text the user copies into
                # YouTube actually produces clickable chapters.
                "description": normalise_chapters(
                    str(entry.get("description") or "").strip()),
                "tags": _trim_tags([str(t).strip().lower() for t in tags if str(t).strip()]),
            })

        if topics:
            out.append({"pillar": name,
                        "premise": str(pillar.get("premise") or "").strip(),
                        "topics": topics})

    if not out:
        raise NicheError("The pillars came back unusable.")
    return {"topic": topic, "pillars": out, "model": str(parsed.get("_model") or "")}


# YouTube's limit across all tags on a video.
TAGS_TOTAL_LIMIT = 450


def _trim_tags(tags: list[str]) -> list[str]:
    """Keeps tags inside YouTube's total-character budget."""
    kept: list[str] = []
    total = 0
    for tag in tags:
        cost = len(tag) + (1 if kept else 0)
        if total + cost > TAGS_TOTAL_LIMIT:
            break
        kept.append(tag)
        total += cost
    return kept


def seo_block(topic: dict[str, Any]) -> str:
    """The full package for one episode, as text to copy into an upload form."""
    tags = topic.get("tags") or []
    lines = [
        f"TITLE\n{topic.get('seo_title') or topic.get('hook_title', '')}",
        "",
        f"THUMBNAIL / HOOK\n{topic.get('hook_title', '')}",
        "",
        f"WHY IT WORKS\n{topic.get('why_it_works', '')}",
        "",
        "DESCRIPTION",
        str(topic.get("description") or ""),
        "",
        f"TAGS ({sum(len(t) for t in tags) + max(0, len(tags) - 1)}/{TAGS_TOTAL_LIMIT} chars)",
        ", ".join(tags),
    ]
    return "\n".join(lines)


# A timestamp, and everything up to the next one.
_CHAPTER_RUN = re.compile(
    r"(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—:]?\s*([^\n]*?)(?=\s*\d{1,2}:\d{2}|\s*$)")


def chapters_of(description: str) -> list[tuple[str, str]]:
    """
    Pulls the timestamped chapters out of a description, however they are laid out.

    Line-per-chapter is the shape YouTube needs, but models routinely return the
    whole run inline -- "Chapters: 0:00 Intro, 2:15 The Rise, 4:40 Collapse" --
    and a line-anchored parser finds none of them. Both shapes are read here.
    """
    found: list[tuple[str, str]] = []
    for line in str(description or "").splitlines():
        stripped = line.strip()
        match = re.match(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—:]?\s*(.+)$", stripped)
        if match:
            found.append((match.group(1), match.group(2).strip()))
            continue
        # An inline run, with or without a "Chapters:" label in front of it.
        for stamp, title in _CHAPTER_RUN.findall(stripped):
            title = title.strip(" ,;–—-")
            if title:
                found.append((stamp, title))
    return found


def normalise_chapters(description: str) -> str:
    """
    Rewrites an inline chapter run onto one line each.

    This is not cosmetic. YouTube only builds a clickable chapter list when each
    timestamp starts its own line and the first is 0:00 -- a description reading
    "Chapters: 0:00 Intro, 2:15 The Rise" produces no chapters at all. Since the
    description is copied straight into the upload form, fixing the layout here
    is the difference between working chapters and none.
    """
    body = str(description or "")
    chapters = chapters_of(body)
    if not chapters:
        return body

    # Already one per line? Leave it alone.
    line_anchored = sum(
        1 for line in body.splitlines()
        if re.match(r"^\s*\d{1,2}:\d{2}", line))
    if line_anchored >= len(chapters):
        return body

    # Cut the description off where the chapter run begins.
    head = body
    label = re.search(r"\n?\s*chapters\s*:?", body, re.IGNORECASE)
    first_stamp = re.search(r"\d{1,2}:\d{2}", body)
    cut = min([m.start() for m in (label, first_stamp) if m] or [len(body)])
    head = body[:cut].rstrip()

    lines = [head, "", "Chapters:"] if head else ["Chapters:"]
    lines += [f"{stamp} {title}" for stamp, title in chapters]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handoff payloads
#
# Plain dicts. The app writes them into the target mode's session state; this
# module never touches Streamlit.
# ---------------------------------------------------------------------------

# Which narrative look and tone suit which CPM band. A history channel and a
# finance channel want genuinely different films, and defaulting both to the
# same vector-comic look is how every episode ends up feeling the same.
_NARRATIVE_STYLE: dict[str, tuple[str, str]] = {
    "history": ("nordic_noir", "suspense"),
    "education": ("flat_minimal", "reflection"),
    "psychology": ("vector_comic", "stoic"),
    "finance": ("flat_minimal", "stoic"),
    "tech_b2b": ("flat_minimal", "reflection"),
    "health": ("lofi_anime", "reflection"),
    "entertainment": ("vector_comic", "reflection"),
    "relaxation": ("lofi_anime", "reflection"),
    "general": ("vector_comic", "reflection"),
}


def narrative_handoff(topic: dict[str, Any], band: str = "general") -> dict[str, Any]:
    """What Narrative Studio needs to open pre-filled on this episode."""
    aesthetic, tone = _NARRATIVE_STYLE.get(band, _NARRATIVE_STYLE["general"])
    premise = str(topic.get("hook_title") or "").strip()
    angle = str(topic.get("why_it_works") or "").strip()
    return {
        "topic": f"{premise}\n\n{angle}" if angle else premise,
        "aesthetic": aesthetic,
        "tone": tone,
        "format": "short",
    }


def batch_handoff(topic: dict[str, Any]) -> str:
    """Batch Studio takes a topic line; the hook title is the searchable one."""
    return str(topic.get("hook_title") or topic.get("seo_title") or "").strip()
