"""
YouTube Analytics: what actually happened after the upload.

Everything upstream of this module is a prediction. The scorecard says a hook
should hold; the CPM band says a niche should pay; the saturation score says a
market should have room. None of it is ever contradicted, because nothing has
ever measured the result. This module closes that loop: it reads the channel's
own numbers back and joins them to the render that produced them, using the
video id the publisher now writes into the ledger.

Two things it is careful about.

**It does not invent metric names.** Impressions and click-through rate are
the two numbers everyone wants, and for most of this API's life they have been
Studio-only -- not exposed to the public Analytics API at all. Asking for them
unconditionally earns a 400 that takes the whole report down with it. So the
optional metrics are tried once, and if the API rejects them the request is
retried with the core set and the result says plainly which numbers were not
available. If Google exposes them later this starts working with no change.

**It reports "not enough data" rather than a number built on two videos.** A
correlation across three uploads is noise with a decimal point on it.
"""
from __future__ import annotations

import statistics
from datetime import date, timedelta
from typing import Any, Callable

ProgressFn = Callable[[str], None]


class AnalyticsError(RuntimeError):
    """Raised when the report cannot be fetched or the channel is not linked."""


# The read-only Analytics scope. Distinct from the upload scope, so a channel
# authorized before this existed carries a token that cannot query reports --
# which is why missing_scopes() exists rather than letting a 403 surface raw.
ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"

# Metrics the Analytics API has exposed consistently.
CORE_METRICS: tuple[str, ...] = (
    "views",
    "estimatedMinutesWatched",
    "averageViewDuration",
    "averageViewPercentage",
    "subscribersGained",
    "likes",
    "comments",
    "shares",
)

# Wanted, and historically Studio-only. Tried, then dropped if refused.
OPTIONAL_METRICS: tuple[str, ...] = (
    "impressions",
    "impressionClickThroughRate",
)

# The window a short lives in. Most of a Short's lifetime views land inside a
# month, and a longer window mixes in videos that have had unequal time to run.
DEFAULT_WINDOW_DAYS = 28

# Below this many published videos, any relationship between the predicted
# score and the measured retention is noise.
MIN_VIDEOS_FOR_CORRELATION = 5

# The hook is the first three seconds. Retention is reported against a ratio of
# elapsed video time, so the sample point depends on how long the video is.
HOOK_SECONDS = 3.0


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def missing_scopes() -> list[str]:
    """
    Scopes the stored token lacks. Empty when it can query reports.

    A token minted before analytics existed is still perfectly valid for
    uploading, so this cannot just check "is there a token". Without this the
    first report returns a 403 whose message does not say the word "scope".
    """
    import json
    import os

    import publisher

    if not os.path.exists(publisher.TOKEN_PATH):
        return list(publisher.SCOPES)

    try:
        with open(publisher.TOKEN_PATH, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, ValueError):
        return list(publisher.SCOPES)

    granted = set(stored.get("scopes") or [])
    return [scope for scope in publisher.SCOPES if scope not in granted]


def _client() -> Any:
    """The Analytics API client, or a reason it cannot be built."""
    from googleapiclient.discovery import build

    import publisher

    lacking = missing_scopes()
    if lacking:
        raise AnalyticsError(
            "This channel was authorized before performance data was added, so "
            "its token cannot read reports. Disconnect and reconnect the "
            "channel in Atmosphere Studio to grant the extra permission — "
            "nothing else changes, and your uploads are untouched."
            if len(lacking) < len(publisher.SCOPES)
            else "No channel is connected. Authorize one in Atmosphere Studio.")

    creds = publisher._load_credentials()
    if creds is None:
        raise AnalyticsError("No channel is connected.")
    return build("youtubeAnalytics", "v2", credentials=creds,
                 cache_discovery=False)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_report(response: Any) -> list[dict[str, Any]]:
    """
    Rows keyed by column name.

    The API answers with a header list and a list of positional rows, which is
    compact and unreadable. Everything downstream wants dicts.
    """
    payload = response if isinstance(response, dict) else {}
    headers = [str(col.get("name") or "")
               for col in (payload.get("columnHeaders") or [])]
    if not headers:
        return []

    rows: list[dict[str, Any]] = []
    for raw in payload.get("rows") or []:
        if not isinstance(raw, (list, tuple)):
            continue
        row: dict[str, Any] = {}
        for name, value in zip(headers, raw):
            row[name] = value
        rows.append(row)
    return rows


def _is_unknown_metric_error(exc: BaseException) -> bool:
    """True when the API refused a metric name rather than the request."""
    text = str(exc).lower()
    return "400" in text and any(
        word in text for word in ("metric", "unknown", "invalid", "unsupported"))


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _window(days: int) -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=max(1, int(days)))
    return start.isoformat(), end.isoformat()


def video_report(video_ids: list[str] | tuple[str, ...] = (),
                 days: int = DEFAULT_WINDOW_DAYS,
                 progress: ProgressFn | None = None,
                 client: Any = None) -> dict[str, Any]:
    """
    Per-video performance over the window.

    Returns {"rows", "metrics", "unavailable", "days", "start", "end"}.
    `unavailable` names the metrics this API version would not serve, so the UI
    can say why a column is missing instead of showing a silent zero.
    """
    service = client if client is not None else _client()
    start, end = _window(days)
    ids = [str(v).strip() for v in video_ids if str(v).strip()]

    def run(metrics: tuple[str, ...]) -> Any:
        request = {
            "ids": "channel==MINE",
            "startDate": start,
            "endDate": end,
            "metrics": ",".join(metrics),
            "dimensions": "video",
            "sort": "-views",
            "maxResults": 50,
        }
        if ids:
            # The filter caps at 500 characters; an 11-character id plus a
            # separator is 12, so 40 is a safe ceiling.
            request["filters"] = "video==" + ",".join(ids[:40])
        return service.reports().query(**request).execute()

    wanted = CORE_METRICS + OPTIONAL_METRICS
    unavailable: list[str] = []

    if progress:
        progress(f"Reading the last {days} days...")

    try:
        response = run(wanted)
    except Exception as exc:                                  # noqa: BLE001
        if not _is_unknown_metric_error(exc):
            raise AnalyticsError(f"Could not read the report: {exc}") from exc
        # The optional metrics are the only ones that can be refused by name.
        unavailable = list(OPTIONAL_METRICS)
        if progress:
            progress("Impressions and CTR are not served by this API; "
                     "reading the rest.")
        try:
            response = run(CORE_METRICS)
        except Exception as inner:                            # noqa: BLE001
            raise AnalyticsError(f"Could not read the report: {inner}") from inner

    served = CORE_METRICS if unavailable else wanted
    return {
        "rows": parse_report(response),
        "metrics": list(served),
        "unavailable": unavailable,
        "days": int(days),
        "start": start,
        "end": end,
    }


def retention_curve(video_id: str, days: int = DEFAULT_WINDOW_DAYS,
                    client: Any = None) -> list[dict[str, float]]:
    """
    The audience-retention curve for one video.

    Each point is (elapsedVideoTimeRatio, audienceWatchRatio): the fraction of
    the video, and the fraction of viewers still watching there.
    """
    service = client if client is not None else _client()
    start, end = _window(days)

    try:
        response = service.reports().query(
            ids="channel==MINE",
            startDate=start,
            endDate=end,
            metrics="audienceWatchRatio",
            dimensions="elapsedVideoTimeRatio",
            filters=f"video=={str(video_id).strip()}",
        ).execute()
    except Exception as exc:                                  # noqa: BLE001
        raise AnalyticsError(f"Could not read retention: {exc}") from exc

    points: list[dict[str, float]] = []
    for row in parse_report(response):
        try:
            points.append({
                "ratio": float(row.get("elapsedVideoTimeRatio") or 0.0),
                "watching": float(row.get("audienceWatchRatio") or 0.0),
            })
        except (TypeError, ValueError):
            continue
    points.sort(key=lambda p: p["ratio"])
    return points


def hook_retention(curve: list[dict[str, float]], duration: float,
                   at_seconds: float = HOOK_SECONDS) -> float:
    """
    The share of viewers still watching at the three-second mark, 0-1.

    This is the number the scorecard's hook axis has been predicting blind.
    Retention is reported against a ratio of elapsed time, so where three
    seconds falls depends on the runtime: on a 70-second short it is 4.3% in.

    Returns 0.0 when there is no curve or no duration to place the mark
    against, which the caller must read as "unknown" rather than "nobody
    stayed".
    """
    if not curve or duration <= 0:
        return 0.0

    target = max(0.0, min(1.0, float(at_seconds) / float(duration)))

    # The curve is sampled coarsely, so take the first point at or past the
    # mark rather than expecting one exactly on it.
    for point in curve:
        if point["ratio"] >= target:
            return max(0.0, min(1.0, point["watching"]))
    return max(0.0, min(1.0, curve[-1]["watching"]))


# ---------------------------------------------------------------------------
# Joining performance back to the render that produced it
# ---------------------------------------------------------------------------

def published_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ledger entries that carry a YouTube video id."""
    return [dict(entry) for entry in entries
            if str(entry.get("youtube_video_id") or "").strip()]


def join_performance(entries: list[dict[str, Any]],
                     report: dict[str, Any]) -> list[dict[str, Any]]:
    """
    One record per published render: what it was, and how it did.

    Rows the channel reports but the ledger has never heard of are dropped
    rather than shown: they are uploads made outside this tool, and the point
    of the join is to grade what this tool produced.
    """
    by_id = {str(row.get("video") or ""): row for row in report.get("rows") or []}

    joined: list[dict[str, Any]] = []
    for entry in published_entries(entries):
        video_id = str(entry.get("youtube_video_id") or "")
        row = by_id.get(video_id)
        if row is None:
            continue

        duration = float(entry.get("duration") or 0.0)
        watched = float(row.get("averageViewDuration") or 0.0)
        joined.append({
            "video_id": video_id,
            "video_name": str(entry.get("video_name") or ""),
            "title": str(entry.get("youtube_title") or entry.get("video_name") or ""),
            "url": f"https://youtu.be/{video_id}",
            "script": str(entry.get("script") or ""),
            "duration": duration,
            "views": int(row.get("views") or 0),
            "average_view_seconds": watched,
            "average_view_percent": float(row.get("averageViewPercentage") or 0.0),
            "subscribers_gained": int(row.get("subscribersGained") or 0),
            "likes": int(row.get("likes") or 0),
            "comments": int(row.get("comments") or 0),
            "impressions": int(row.get("impressions") or 0),
            "ctr": float(row.get("impressionClickThroughRate") or 0.0),
        })

    joined.sort(key=lambda item: -item["views"])
    return joined


# ---------------------------------------------------------------------------
# Grading the scorecard against reality
# ---------------------------------------------------------------------------

def score_vs_retention(joined: list[dict[str, Any]],
                       hook_scores: dict[str, float]) -> dict[str, Any]:
    """
    Does a high hook score actually mean people stayed?

    `hook_scores` maps video id to the score the scorecard gave that script.
    Returns a verdict with `measured` False until there are enough videos to
    say anything -- a correlation over three uploads is noise with a decimal
    point on it.

    The measure is Pearson's r between the predicted hook score and the
    measured average view percentage. It is a weak instrument on a small
    sample, which is exactly why the threshold exists and why the wording says
    "so far" rather than stating a fact about the world.
    """
    pairs = [(hook_scores[item["video_id"]], item["average_view_percent"])
             for item in joined
             if item["video_id"] in hook_scores
             and item["average_view_percent"] > 0]

    if len(pairs) < MIN_VIDEOS_FOR_CORRELATION:
        return {
            "measured": False,
            "videos": len(pairs),
            "needed": MIN_VIDEOS_FOR_CORRELATION,
            "verdict": (f"{len(pairs)} of {MIN_VIDEOS_FOR_CORRELATION} videos "
                        f"needed before this can say anything."),
        }

    predicted = [p for p, _ in pairs]
    actual = [a for _, a in pairs]

    try:
        r = statistics.correlation(predicted, actual)
    except (statistics.StatisticsError, ValueError):
        # Happens when every score is identical: there is no variation to
        # correlate, which is a real answer rather than an error.
        return {
            "measured": False, "videos": len(pairs),
            "needed": MIN_VIDEOS_FOR_CORRELATION,
            "verdict": "Every video scored the same, so there is nothing to "
                       "compare against.",
        }

    if r >= 0.5:
        verdict = ("The scorecard is predicting retention so far: the videos it "
                   "rated higher are the ones being watched longer.")
    elif r >= 0.2:
        verdict = ("Weak agreement so far. The scorecard is not wrong, but it is "
                   "not the main thing deciding retention either.")
    elif r > -0.2:
        verdict = ("No relationship so far. The scorecard is grading something "
                   "that is not moving retention on this channel.")
    else:
        verdict = ("Inverted so far: the videos the scorecard rated highest are "
                   "being watched least. Trust the measurement over the score.")

    return {
        "measured": True,
        "videos": len(pairs),
        "needed": MIN_VIDEOS_FOR_CORRELATION,
        "r": round(float(r), 2),
        "best": max(pairs, key=lambda p: p[1]),
        "worst": min(pairs, key=lambda p: p[1]),
        "verdict": verdict,
    }


def summarise(joined: list[dict[str, Any]]) -> dict[str, Any]:
    """Channel-level totals across the joined videos."""
    if not joined:
        return {"videos": 0, "views": 0, "subscribers": 0,
                "median_view_percent": 0.0, "best": None}

    percents = [item["average_view_percent"] for item in joined
                if item["average_view_percent"] > 0]
    return {
        "videos": len(joined),
        "views": sum(item["views"] for item in joined),
        "subscribers": sum(item["subscribers_gained"] for item in joined),
        "median_view_percent": round(statistics.median(percents), 1) if percents else 0.0,
        "best": joined[0],
    }
