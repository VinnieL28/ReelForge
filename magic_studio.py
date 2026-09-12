"""
Magic Studio: one prompt, one style, one finished video.

This module is the planner, not the renderer. Given a sentence someone typed
and a style they picked, it works out exactly what the existing engine for that
style needs to be handed -- a scene concept, a matchup with its metric rounds,
a duration target, a bed and a texture -- and nothing more. `app.py` then calls
the same pipelines the manual modes call.

That split is deliberate. Forking the render pipelines to get a one-click flow
would mean two code paths to keep in sync, and the second one would be the one
without the tests. Everything here is pure: no Streamlit, no network except the
one optional Gemini call in `duel_brief`, and no file system. It can be tested
by calling it.

The stage names are fixed and shared, because the progress bar promises a
specific sequence and a stage that appears in one style but not another makes
the bar jump.
"""
from __future__ import annotations

import re
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

STYLES: tuple[dict[str, str], ...] = (
    {
        "key": "minimalist",
        "label": "📐 Minimalist Vector",
        "sub": "100% original, YouTube/TikTok safe",
        "mode": "minimalist",
        "example": "Why consistency beats intensity",
        "blurb": "Every frame drawn from code and every audio layer synthesized, so "
                 "there is no third-party rights holder anywhere in the video.",
    },
    {
        "key": "duel",
        "label": "⚔️ Versus Duel",
        "sub": "High comment debate reel",
        "mode": "duel",
        "example": "Range Rover vs Porsche Cayenne",
        "blurb": "Two things, three measured rounds and a winner. The format people "
                 "argue with in the comments.",
    },
    {
        "key": "commentary",
        "label": "🎙️ Faceless Commentary",
        "sub": "Viral breakdown",
        "mode": "commentary",
        "example": "Why Rome fell in 60s",
        "blurb": "Licensed footage under an editorial script, with kinetic captions "
                 "burned in.",
    },
    {
        "key": "atmosphere",
        "label": "🌙 8-Hour Atmosphere",
        "sub": "Sleep/study YouTube long-form",
        "mode": "atmosphere",
        "example": "8-hour rain on window",
        "blurb": "One render covers a whole night of watch time, and every layer of "
                 "it is synthesized.",
    },
)

STYLE_KEYS: tuple[str, ...] = tuple(style["key"] for style in STYLES)
DEFAULT_STYLE = "minimalist"

PLACEHOLDER = ("What video do you want to create? (e.g., 'Why Rome fell in 60s', "
               "'Range Rover vs Porsche Cayenne', '8-hour rain on window')")


def style(key: str) -> dict[str, str]:
    for entry in STYLES:
        if entry["key"] == key:
            return dict(entry)
    return dict(STYLES[0])


# ---------------------------------------------------------------------------
# Stages
#
# Weighted by measured share of wall clock, not evenly: the render dominates
# every style, and a bar that gives four stages a quarter each sits at 75% for
# most of the wait.
# ---------------------------------------------------------------------------

STAGES: tuple[tuple[str, str, float], ...] = (
    ("script", "Writing Hook", 0.12),
    ("voice", "Synthesizing Voice", 0.18),
    ("visual", "Animating", 0.20),
    ("render", "Finalizing Render", 0.50),
)

STAGE_LABELS: dict[str, str] = {key: label for key, label, _ in STAGES}


def stage_fraction(stage_key: str, within: float = 0.0) -> float:
    """
    How far along the whole job a stage is, 0-1.

    `within` is progress inside the stage itself, so a render that reports 40%
    of its frames moves the overall bar rather than sitting still through the
    longest part of the job.
    """
    done = 0.0
    for key, _label, weight in STAGES:
        if key == stage_key:
            return min(1.0, done + weight * max(0.0, min(1.0, within)))
        done += weight
    return min(1.0, done)


def stage_caption(stage_key: str) -> str:
    """'Writing Hook → Synthesizing Voice → Animating → Finalizing Render'."""
    parts = []
    seen = False
    for key, label, _weight in STAGES:
        if key == stage_key:
            seen = True
            parts.append(f"**{label}**")
        else:
            parts.append(label if not seen else f"_{label}_")
    return " → ".join(parts)


# ---------------------------------------------------------------------------
# Reading the prompt
# ---------------------------------------------------------------------------

_VS_RE = re.compile(r"^\s*(.+?)\s+(?:vs\.?|versus|or)\s+(.+?)\s*$", re.IGNORECASE)

# Word -> (bed, texture). Ordered most specific first: "thunderstorm" has to be
# tested before "rain", or every storm becomes plain rain.
_BED_HINTS: tuple[tuple[str, str, str], ...] = (
    ("thunderstorm", "thunderstorm", "distant_thunder"),
    ("thunder", "thunderstorm", "distant_thunder"),
    ("storm", "thunderstorm", "distant_thunder"),
    ("fireplace", "fireplace", "room_wind"),
    ("fire", "fireplace", "room_wind"),
    ("campfire", "fireplace", "crickets"),
    ("brown noise", "brown_noise", "none"),
    ("white noise", "brown_noise", "none"),
    ("noise", "brown_noise", "none"),
    ("stream", "stream", "crickets"),
    ("river", "stream", "crickets"),
    ("creek", "stream", "crickets"),
    ("water", "stream", "none"),
    ("forest", "stream", "crickets"),
    ("night", "rain_window", "crickets"),
    ("rain", "rain_window", "room_wind"),
    ("window", "rain_window", "room_wind"),
)

# Spoken and written forms of the durations Atmosphere Studio offers.
_DURATION_HINTS: tuple[tuple[str, str], ...] = (
    ("8 hour", "8hours"), ("8-hour", "8hours"), ("eight hour", "8hours"),
    ("3 hour", "3hours"), ("3-hour", "3hours"), ("three hour", "3hours"),
    ("1 hour", "1hour"), ("1-hour", "1hour"), ("one hour", "1hour"),
    ("30 min", "30min"), ("30-min", "30min"), ("half hour", "30min"),
    ("test", "test"),
)


def split_matchup(prompt: str) -> tuple[str, str]:
    """
    ("Range Rover", "Porsche Cayenne") from "Range Rover vs Porsche Cayenne".

    Returns ("", "") when the prompt is not a matchup, which is the signal to
    ask the model to invent one rather than guessing at a split.
    """
    text = str(prompt or "").strip()
    # Strip a leading framing clause so "Which is better: A vs B" still splits.
    text = re.sub(r"^(?:which is better|compare|comparison of)\s*[:\-]?\s*", "",
                  text, flags=re.IGNORECASE)

    match = _VS_RE.match(text)
    if not match:
        return "", ""

    left = match.group(1).strip(" ,.?!")
    right = match.group(2).strip(" ,.?!")
    if not left or not right:
        return "", ""
    # "cheap or expensive" is a question, not a matchup. Require both sides to
    # look like things rather than bare adjectives.
    if len(left) < 2 or len(right) < 2:
        return "", ""
    return left, right


def detect_style(prompt: str) -> str:
    """
    The style a prompt is asking for, when it is obvious from the words.

    Only used to pre-select a pill -- the person can always override it, so a
    wrong guess costs one click rather than a wrong render.
    """
    text = str(prompt or "").lower()
    if not text.strip():
        return DEFAULT_STYLE

    if any(word in text for word in ("hour", "sleep", "asmr", "ambient", "study",
                                     "rain", "noise", "fireplace", "thunderstorm")):
        return "atmosphere"
    if split_matchup(text) != ("", ""):
        return "duel"
    # Abstract-concept prompts go to the vector engine before the "why"/"how"
    # test, or "Why consistency beats intensity" -- the canonical Minimalist
    # Motion subject -- routes itself to a footage-based commentary.
    if any(word in text for word in _MINIMALIST_HINTS):
        return "minimalist"
    if any(word in text for word in ("why", "how", "history", "explained",
                                     "breakdown", "story of", "fell", "rise of")):
        return "commentary"
    return DEFAULT_STYLE


# Subjects with nothing to film: they are metaphors, and the vector engine
# draws metaphors. A stock clip of someone at a laptop illustrates none of them.
_MINIMALIST_HINTS: tuple[str, ...] = (
    "consistency", "discipline", "habit", "compound", "compounding", "mindset",
    "motivation", "procrastinat", "focus", "burnout", "patience", "stoic",
    "psychology", "self-improvement", "overthink", "comfort zone",
    "delayed gratification", "beats intensity", "saving", "investing early",
)


def atmosphere_plan(prompt: str, default_duration: str = "8hours") -> dict[str, str]:
    """Bed, texture and length for an ambient prompt."""
    text = str(prompt or "").lower()

    bed, texture = "rain_window", "room_wind"
    for word, bed_key, texture_key in _BED_HINTS:
        if word in text:
            bed, texture = bed_key, texture_key
            break

    duration = default_duration
    for word, key in _DURATION_HINTS:
        if word in text:
            duration = key
            break

    return {"bed": bed, "texture": texture, "duration": duration}


def scene_concept(prompt: str) -> str:
    """
    The prompt as a scene concept for Minimalist Motion.

    Strips the imperative framing people type into a box that asks what they
    want -- "make me a video about X" is a request, "X" is the subject.
    """
    text = str(prompt or "").strip()
    text = re.sub(r"^(?:make|create|generate|build)\s+(?:me\s+)?"
                  r"(?:an?\s+)?(?:video|reel|short|animation)?\s*"
                  r"(?:about|on|explaining)?\s*", "", text, flags=re.IGNORECASE)
    return text.strip() or str(prompt or "").strip()


# ---------------------------------------------------------------------------
# The duel brief
# ---------------------------------------------------------------------------

DUEL_PROMPT = """Build a head-to-head comparison for a short-form video.

REQUEST: {prompt}

Return ONE JSON object and nothing else:

{{"a": {{"name": "...", "hook": "..."}},
  "b": {{"name": "...", "hook": "..."}},
  "headline": "...",
  "cta": "...",
  "rounds": [{{"metric": "...", "a_score": 0, "b_score": 0, "unit": "...",
               "winner": "A", "note": "..."}}]}}

RULES:
- Exactly 3 rounds. Each metric must be numeric and genuinely comparable -- a
  price, a figure of output, a measured time. Never a rating out of ten and
  never a subjective score.
- a_score and b_score are plain numbers with no units in them. The unit goes in
  "unit" ("$", "hp", "s", "mpg", "kg").
- "winner" is "A" or "B", and must agree with the numbers: for a metric where
  less is better, like price or 0-60 time, the lower number wins.
- Use real figures you are confident about. If you are not confident about a
  metric, choose a different metric rather than inventing a number.
- "hook" is at most 5 words. "note" is at most 8 and says why the gap matters.
- If the request names only one thing, choose its most obvious real rival."""


def parse_duel_brief(raw: str) -> dict[str, Any]:
    """Validates a model duel brief into something build_duel_slides can use."""
    from gemini_engine import parse_scene_response

    parsed = parse_scene_response(raw)
    if not isinstance(parsed, dict):
        return {}

    def side(key: str) -> dict[str, str]:
        item = parsed.get(key)
        item = item if isinstance(item, dict) else {}
        return {"name": str(item.get("name") or "").strip(),
                "hook": str(item.get("hook") or "").strip()}

    a, b = side("a"), side("b")
    if not a["name"] or not b["name"]:
        return {}

    rounds: list[dict[str, Any]] = []
    for entry in (parsed.get("rounds") or [])[:3]:
        if not isinstance(entry, dict):
            continue
        try:
            a_score = float(entry.get("a_score"))
            b_score = float(entry.get("b_score"))
        except (TypeError, ValueError):
            continue
        metric = str(entry.get("metric") or "").strip()
        if not metric:
            continue

        winner = str(entry.get("winner") or "").strip().upper()
        if winner not in ("A", "B"):
            # Fall back to the numbers rather than dropping the round: higher
            # wins unless the metric is one where less obviously is better.
            lower_wins = any(word in metric.lower() for word in
                             ("price", "cost", "0-60", "0 to 60", "time", "weight"))
            if a_score == b_score:
                winner = ""
            elif lower_wins:
                winner = "A" if a_score < b_score else "B"
            else:
                winner = "A" if a_score > b_score else "B"

        rounds.append({
            "metric": metric,
            "a_score": a_score,
            "b_score": b_score,
            "unit": str(entry.get("unit") or "").strip(),
            "winner": winner,
            "note": str(entry.get("note") or "").strip(),
        })

    if not rounds:
        return {}

    return {
        "a": a, "b": b, "rounds": rounds,
        "headline": str(parsed.get("headline") or f"{a['name']} vs {b['name']}").strip(),
        "cta": str(parsed.get("cta") or "Which one would you pick?").strip(),
    }


def duel_brief(prompt: str,
               progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """
    A full matchup for a typed prompt.

    Raises on failure rather than returning a stub. A duel built from invented
    numbers is worse than no duel: the whole format rests on the figures being
    checkable, and a confident wrong horsepower number is what the comments
    will be about.
    """
    from gemini_engine import (MODEL_CANDIDATES, GeminiError, generate_with_retry,
                               get_client)

    text = str(prompt or "").strip()
    if not text:
        raise ValueError("Type what you want to compare first.")

    client = get_client()
    body = DUEL_PROMPT.format(prompt=text)

    last: Exception | None = None
    for model in MODEL_CANDIDATES:
        try:
            if progress:
                progress(f"Building the matchup with {model}...")
            response = generate_with_retry(client, model, body, progress=progress)
            brief = parse_duel_brief(getattr(response, "text", "") or "")
            if brief:
                brief["model"] = model
                return brief
        except Exception as exc:                              # noqa: BLE001
            last = exc

    raise GeminiError(
        f"Could not build a matchup for {text!r}"
        + (f": {type(last).__name__}: {last}" if last else ".")
    )


# ---------------------------------------------------------------------------
# Running a render off the Streamlit thread
#
# A Streamlit script run is a request. Rendering inside one holds that request
# open for the length of the render -- minutes, now that Minimalist scenes are
# written to a monetizable 62-75 seconds -- and a browser does not wait
# minutes. The tab stops responding, the websocket drops, and what the person
# sees is indistinguishable from a crash.
#
# So the work happens on a worker thread and the script run polls this object.
# Each run then lasts a fraction of a second: draw the current stage, schedule
# another run, return.
#
# The worker must never touch st.session_state or call any st.* function.
# There is no ScriptRunContext on that thread, so those calls are at best
# no-ops that log a warning and at worst raise. Everything the render needs is
# copied into a plain dict on the main thread and handed over; everything it
# produces comes back through here.
# ---------------------------------------------------------------------------

class MagicJob:
    """Thread-safe progress and result for one one-click render."""

    def __init__(self, prompt: str, style_key: str) -> None:
        import threading
        import time

        self.prompt = str(prompt)
        self.style = str(style_key)
        self.started = time.time()
        self.thread: Any = None
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "stage": STAGES[0][0],
            "within": 0.0,
            "message": "Starting...",
            "done": False,
            "error": "",
            "result": None,
        }

    # -- called from the worker ---------------------------------------------

    def report(self, stage: str, within: float = 0.0, message: str = "") -> None:
        with self._lock:
            self._state["stage"] = stage
            self._state["within"] = max(0.0, min(1.0, float(within)))
            self._state["message"] = message or STAGE_LABELS.get(stage, stage)

    def finish(self, result: dict[str, Any]) -> None:
        with self._lock:
            self._state["result"] = result
            self._state["done"] = True
            self._state["within"] = 1.0

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            self._state["error"] = f"{type(exc).__name__}: {exc}"
            self._state["done"] = True

    # -- called from the script run -----------------------------------------

    def snapshot(self) -> dict[str, Any]:
        import time

        with self._lock:
            state = dict(self._state)
        state["elapsed"] = time.time() - self.started
        state["fraction"] = (1.0 if state["done"] and not state["error"]
                             else stage_fraction(state["stage"], state["within"]))
        state["caption"] = stage_caption(state["stage"])

        # An ETA is only meaningful once there is enough of the job behind it
        # to extrapolate from. Before that it swings by minutes between frames.
        eta = 0.0
        if state["fraction"] > 0.08 and not state["done"]:
            eta = state["elapsed"] * (1.0 - state["fraction"]) / state["fraction"]
        state["eta"] = eta

        # A worker that died without reporting -- an OS-level kill, an
        # unhandled exit -- would otherwise leave the page polling forever.
        thread = self.thread
        if (not state["done"] and thread is not None and not thread.is_alive()):
            state["done"] = True
            state["error"] = state["error"] or (
                "The render thread stopped without reporting a result.")
        return state
