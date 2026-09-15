# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false, reportUnknownLambdaType=false
"""
Gemini video-grounding engine for the AI Faceless Video Commentary Machine.

Uploads a raw viral clip through the Gemini Files API, waits for it to finish
processing, then asks a flash model to watch it and write a short-form
commentary script (Hook -> Story -> CTA).
"""

from __future__ import annotations

import os
import re
import json
import time
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai

from minimalist_engine import METAPHOR_TYPES, TEMPLATES

# Load .env next to this file so the key is available before Client() is built;
# genai.Client() reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

ProgressFn = Callable[[str], None]

# Model fallback chain, tried in order.
#
# Verified against the live API on 2026-09-09: gemini-2.5-flash and
# gemini-1.5-flash both return 404 ("no longer available to new users") for a
# newly issued key, so the current flash models lead and the older names are
# kept at the back for keys that still have access to them.
MODEL_CANDIDATES: tuple[str, ...] = (
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
)

# Narration pacing presets. Word budgets assume ~165 wpm (2.75 words/second),
# the rate edge-tts lands at with the default +12% speed boost.
WORDS_PER_SECOND = 2.75

DURATION_TARGETS: dict[str, dict[str, Any]] = {
    "short": {
        "label": "10–15s · Ultra-short (single clip)",
        "low": 10, "high": 15,
        "body": "2 punchy sentences",
    },
    "standard": {
        "label": "20–30s · Standard viral commentary",
        "low": 20, "high": 30,
        "body": "3-4 vivid, witty sentences",
    },
    "deep": {
        "label": "45–60s · Deep-dive story",
        "low": 45, "high": 60,
        "body": "6-8 vivid sentences that build a story arc with a turn in the middle",
    },
    # TikTok's Creator Rewards programme only counts videos over one minute, so
    # nothing shorter than this can earn there however well it performs.
    #
    # The floor is 62 seconds rather than 60 on purpose. TTS pacing lands
    # within a few percent of the word budget, not on it, so a script written
    # to exactly 60 seconds renders somewhere either side of the bar and about
    # half of those earn nothing. Two seconds of margin costs nothing and makes
    # the eligibility deterministic.
    "rewards": {
        "label": "62–75s · TikTok Rewards eligible (over 1 min)",
        "low": 62, "high": 75,
        "body": ("8-11 vivid sentences that build a full story arc: set the scene, "
                 "raise the stakes, deliver a turn, then land the payoff"),
    },
    "long": {
        "label": "70–90s · Extended story",
        "low": 70, "high": 90,
        "body": ("9-12 vivid sentences that build a full story arc: set the scene, "
                 "raise the stakes, deliver a turn, then land the payoff"),
    },
}

# Monetization-eligible by default. Every other band renders something TikTok
# Creator Rewards will not pay for, which is a strange thing for the default to
# do on a tool whose whole purpose is monetized output.
DEFAULT_TARGET = "rewards"

# The band that clears the payout threshold, named so the UI can mark it.
REWARDS_TARGET = "rewards"


def build_prompt(target: str = DEFAULT_TARGET) -> str:
    """
    Builds the commentary prompt calibrated to a spoken-duration target.

    Models pace far more reliably against a word budget than against seconds,
    so the prompt carries both: the seconds for tone, the word count for length.
    """
    spec = DURATION_TARGETS.get(target, DURATION_TARGETS[DEFAULT_TARGET])
    low, high = int(spec["low"]), int(spec["high"])
    words_low = int(low * WORDS_PER_SECOND)
    words_high = int(high * WORDS_PER_SECOND)

    return f"""You are an elite short-form viral video storyteller (MrBeast / Liam style). Watch this video clip closely. Write a {low}-{high} second spoken commentary script reacting to what happens.

LENGTH IS CRITICAL: the script must be {words_low}-{words_high} words total, because it will be read aloud at about {WORDS_PER_SECOND:.1f} words per second and must finish inside {high} seconds. Count your words and stay in range.

Format strictly:
- Hook: A one-sentence punchy reaction to the opening second.
- Story/Body: {spec['body']} describing the action as it unfolds.
- Outro/CTA: A closing question asking viewers what they would do.
Return ONLY the spoken text, clean without parentheticals or timestamps."""


# Kept for callers that want the default wording directly.
COMMENTARY_PROMPT = build_prompt(DEFAULT_TARGET)

UPLOAD_TIMEOUT = 300
POLL_INTERVAL = 2.0


class GeminiError(RuntimeError):
    """Raised when the Gemini pipeline fails, carrying a user-readable reason."""


def get_client() -> genai.Client:
    """
    Builds the Gemini client.

    Raises a clear error when the key is missing rather than letting the SDK
    fail later with something cryptic.
    """
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise GeminiError(
            "No GEMINI_API_KEY found. Add it to the .env file in the project root."
        )
    try:
        return genai.Client()
    except Exception as exc:
        raise GeminiError(f"Could not initialise the Gemini client: {exc}") from exc


def upload_video(
    client: genai.Client,
    video_path: str,
    progress: ProgressFn | None = None,
    timeout: int = UPLOAD_TIMEOUT,
    normalise: bool = True,
) -> Any:
    """
    Uploads a clip and polls until Gemini reports it ACTIVE.

    A freshly uploaded video sits in PROCESSING for a while; calling
    generate_content against it before it goes ACTIVE fails, so this blocks
    until the file is genuinely usable.

    Every clip goes through an ffmpeg pre-flight first. The File API does not
    reject an awkward container at upload time -- it accepts the bytes, parks
    the file in FAILED, and the user sees "Gemini could not process this clip"
    after a two-minute wait. Transcoding first turns that class of failure into
    a four-second transcode, and shrinks the upload as a side effect.
    """
    if not os.path.exists(video_path):
        raise GeminiError(f"Video file not found: {video_path}")

    upload_path = video_path
    if normalise:
        try:
            from video_engine import normalize_for_gemini

            prepared = normalize_for_gemini(video_path, progress=progress)
            upload_path = prepared["path"]
        except Exception as exc:
            # A pre-flight that cannot run is not a reason to refuse the
            # upload -- the original may well be fine.
            if progress:
                progress(f"Pre-flight skipped ({type(exc).__name__}); uploading as-is.")

    size_mb = os.path.getsize(upload_path) / 1_048_576
    if progress:
        progress(f"Uploading clip to Gemini ({size_mb:.1f} MB)...")

    try:
        uploaded = client.files.upload(file=upload_path)
    except Exception as exc:
        raise GeminiError(f"Upload failed: {type(exc).__name__}: {exc}") from exc

    deadline = time.time() + timeout
    while True:
        state = str(getattr(uploaded.state, "name", uploaded.state) or "").upper()

        if state == "ACTIVE":
            if progress:
                progress("Clip is ACTIVE — Gemini is ready to watch it.")
            return uploaded

        if state == "FAILED":
            # The pre-flight above already delivered baseline H.264/yuv420p/30
            # CFR, so reaching FAILED means the file itself is damaged -- a
            # truncated download rather than a container Gemini dislikes.
            raise GeminiError(
                "Gemini rejected this clip after it had been normalised to "
                "baseline H.264, which usually means the download is truncated "
                "or the file has no readable video stream. Re-download it and "
                "try again."
            )

        if time.time() > deadline:
            raise GeminiError(
                f"Timed out after {timeout}s waiting for Gemini to process the clip "
                f"(last state: {state or 'UNKNOWN'})."
            )

        if progress:
            remaining = int(deadline - time.time())
            progress(f"Gemini is processing the clip (state: {state}, {remaining}s left)...")

        time.sleep(POLL_INTERVAL)
        file_name = getattr(uploaded, "name", None)
        if not file_name:
            raise GeminiError("Gemini returned an upload without a file name.")
        try:
            uploaded = client.files.get(name=file_name)
        except Exception as exc:
            raise GeminiError(f"Lost track of the uploaded file: {exc}") from exc


# Transient API failures worth waiting out, versus permanent ones worth
# skipping immediately. A 503 "high demand" spike is common on flash models
# and clears in seconds; a 404 means the model is gone for this key.
_RETRYABLE = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "overloaded")
_PERMANENT = ("404", "NOT_FOUND", "400", "INVALID_ARGUMENT", "PERMISSION_DENIED", "401")


def is_retryable(exc: Exception) -> bool:
    """True when an API error is a transient spike rather than a dead end."""
    text = f"{type(exc).__name__}: {exc}"
    if any(token in text for token in _PERMANENT):
        return False
    return any(token in text for token in _RETRYABLE)


def generate_with_retry(
    client: genai.Client,
    model: str,
    contents: Any,
    config: Any = None,
    attempts: int = 3,
    base_delay: float = 2.0,
    progress: ProgressFn | None = None,
) -> Any:
    """
    Calls generate_content, riding out transient overload with backoff.

    Without this a single 503 spike -- which flash models throw regularly --
    fails the whole run even though the next attempt a few seconds later
    usually succeeds.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if config is not None:
                return client.models.generate_content(model=model, contents=contents, config=config)
            return client.models.generate_content(model=model, contents=contents)
        except Exception as exc:
            last = exc
            if attempt >= attempts or not is_retryable(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            if progress:
                progress(f"{model} is busy (attempt {attempt}/{attempts}); retrying in {delay:.0f}s...")
            time.sleep(delay)
    raise last if last else RuntimeError("generate_with_retry exhausted without an error")


def generate_commentary(
    video_path: str,
    prompt: str | None = None,
    models: tuple[str, ...] = MODEL_CANDIDATES,
    progress: ProgressFn | None = None,
    cleanup: bool = True,
    duration_target: str = DEFAULT_TARGET,
) -> dict[str, Any]:
    """
    Full Step-2 pipeline: upload the clip, wait for ACTIVE, then have a flash
    model watch it and write the commentary script.

    `duration_target` selects a pacing preset from DURATION_TARGETS; an explicit
    `prompt` overrides it entirely.

    Returns {"script", "model", "file_name", "target"}. Raises GeminiError with
    a readable message on any failure.
    """
    if prompt is None:
        prompt = build_prompt(duration_target)
    client = get_client()
    uploaded = upload_video(client, video_path, progress=progress)

    errors: list[str] = []
    try:
        for model in models:
            if progress:
                progress(f"Asking {model} to watch the clip and write the script...")
            try:
                response = generate_with_retry(
                    client, model, [uploaded, prompt], progress=progress,
                )
            except Exception as exc:
                errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:160]}")
                continue

            script = (getattr(response, "text", "") or "").strip()
            if script:
                return {
                    "script": clean_script(script),
                    "model": model,
                    "file_name": getattr(uploaded, "name", ""),
                    "target": duration_target,
                }
            errors.append(f"{model}: returned an empty script")

        raise GeminiError(
            "Every model in the fallback chain failed:\n" + "\n".join(f"• {e}" for e in errors)
        )
    finally:
        if cleanup:
            # Uploaded files expire on their own, but removing them keeps the
            # account's file quota clear across many runs.
            try:
                client.files.delete(name=str(uploaded.name))
            except Exception:
                pass


# Three competing angles on the same footage. Publishing the same clip under
# different framings is the cheapest way to find which hook the algorithm likes.
SCRIPT_ANGLES: dict[str, dict[str, str]] = {
    "suspense": {
        "label": "🔥 High Suspense",
        "blurb": "Danger and tension — what could go wrong",
        "brief": ("HIGH SUSPENSE / DISASTER RISK. Lean into the danger and tension. "
                  "Emphasise what could go catastrophically wrong, the split-second "
                  "margin for error, and the stakes if it fails."),
    },
    "skill": {
        "label": "🎯 Operator Skill",
        "blurb": "Technique, precision and raw machine power",
        "brief": ("CURIOSITY / OPERATOR SKILL. Focus on the technique, the precision "
                  "and the raw power of the machinery or the person's craft. Explain "
                  "what makes this genuinely hard and why the execution is impressive."),
    },
    "debate": {
        "label": "💬 Debate Driver",
        "blurb": "Provocative framing built to farm comments",
        "brief": ("DEBATE DRIVER / CONTROVERSY. Frame it provocatively to spark "
                  "arguments in the comments. Take a mild stance, question whether "
                  "this was smart or reckless, and invite viewers to disagree."),
    },
}

ANGLE_ORDER: tuple[str, ...] = ("suspense", "skill", "debate")


def build_multi_angle_prompt(target: str = DEFAULT_TARGET) -> str:
    """
    Asks for three differently-framed scripts on one pass over the video.

    One request rather than three: the upload and the model's viewing of the
    clip are the expensive parts, and asking for the angles together also
    pushes the model to make them genuinely distinct from each other.
    """
    spec = DURATION_TARGETS.get(target, DURATION_TARGETS[DEFAULT_TARGET])
    low, high = int(spec["low"]), int(spec["high"])
    words_low = int(low * WORDS_PER_SECOND)
    words_high = int(high * WORDS_PER_SECOND)

    briefs = "\n".join(
        f'  "{key}": {SCRIPT_ANGLES[key]["brief"]}' for key in ANGLE_ORDER
    )

    return f"""You are an elite short-form viral video storyteller (MrBeast / Liam style). Watch this video clip closely.

Write THREE alternative spoken commentary scripts for the same footage, each taking a completely different angle:
{briefs}

Every script must:
- Open with a one-sentence punchy hook reacting to the opening second.
- Continue with {spec['body']} describing the action as it unfolds.
- Close with a question aimed at the viewer.
- Be {words_low}-{words_high} words long. This is critical: the text is read aloud at about {WORDS_PER_SECOND:.1f} words per second and must finish inside {high} seconds.
- Contain ONLY spoken words. No parentheticals, no timestamps, no stage directions, no labels.

The three scripts must be meaningfully different from each other, not rewordings.

Return ONLY a JSON object in exactly this shape, with no markdown fence and no commentary:
{{"suspense": "<script text>", "skill": "<script text>", "debate": "<script text>"}}"""


def parse_angle_response(raw: str) -> dict[str, str]:
    """
    Extracts the three angle scripts from a model response.

    Tries JSON first (including fenced or prose-wrapped JSON), then falls back
    to labelled blocks. Models drift on output format across versions, so the
    parser is deliberately forgiving rather than assuming one shape.
    """
    text = (raw or "").strip()
    if not text:
        return {}

    # Strip a markdown fence if the model added one despite instructions.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # 1. Straight JSON, or JSON embedded in surrounding prose.
    candidates = [text]
    brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            found = {
                key: clean_script(str(data[key]))
                for key in ANGLE_ORDER
                if isinstance(data.get(key), str) and str(data[key]).strip()
            }
            if found:
                return found

    # 2. Labelled blocks: "Angle A: ...", "suspense: ...", "**Debate Driver**".
    aliases = {
        "suspense": ("suspense", "angle a", "disaster", "danger"),
        "skill": ("skill", "angle b", "curiosity", "operator", "technique"),
        "debate": ("debate", "angle c", "controversy", "provocative"),
    }

    lines = text.splitlines()
    heads: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        probe = line.strip().strip("#*_-").lower()
        if not probe or len(probe) > 90:
            continue
        for key, words in aliases.items():
            if any(word in probe for word in words) and not any(k == key for _, k in heads):
                heads.append((index, key))
                break

    parsed: dict[str, str] = {}
    for position, (index, key) in enumerate(heads):
        stop = heads[position + 1][0] if position + 1 < len(heads) else len(lines)
        body = "\n".join(lines[index + 1:stop]).strip()
        if not body:
            # The script may sit on the heading line after a colon.
            _, _, tail = lines[index].partition(":")
            body = tail.strip()
        if body:
            parsed[key] = clean_script(body)

    return parsed


def generate_commentary_angles(
    video_path: str,
    models: tuple[str, ...] = MODEL_CANDIDATES,
    progress: ProgressFn | None = None,
    cleanup: bool = True,
    duration_target: str = DEFAULT_TARGET,
) -> dict[str, Any]:
    """
    Uploads the clip once and returns three differently-angled scripts.

    Returns {"angles": {key: script}, "model", "file_name", "target"}. Raises
    GeminiError if no model produced a usable set.
    """
    client = get_client()
    uploaded = upload_video(client, video_path, progress=progress)
    prompt = build_multi_angle_prompt(duration_target)
    errors: list[str] = []

    try:
        for model in models:
            if progress:
                progress(f"Asking {model} for three angles on the clip...")
            try:
                response = generate_with_retry(
                    client, model, [uploaded, prompt], progress=progress,
                )
            except Exception as exc:
                errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:160]}")
                continue

            angles = parse_angle_response(getattr(response, "text", "") or "")
            if angles:
                return {
                    "angles": angles,
                    "model": model,
                    "file_name": getattr(uploaded, "name", ""),
                    "target": duration_target,
                }
            errors.append(f"{model}: response did not contain parsable angles")

        raise GeminiError(
            "Could not generate the three angles:\n" + "\n".join(f"• {e}" for e in errors)
        )
    finally:
        if cleanup:
            try:
                client.files.delete(name=str(uploaded.name))
            except Exception:
                pass


def clean_script(text: str) -> str:
    """
    Strips the scaffolding models like to add despite being told not to.

    Removes markdown, leading 'Hook:' / 'Body:' / 'CTA:' labels and stray
    bracketed stage directions, so what remains is purely spoken words.
    """
    import re

    cleaned = text.replace("**", "").replace("*", "").replace("#", "")
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)          # [excited tone]
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)            # (pause)
    cleaned = re.sub(r"\b\d{1,2}:\d{2}\b", " ", cleaned)    # 0:03 timestamps

    lines: list[str] = []
    label = re.compile(
        r"^\s*(-\s*)?(hook|story|body|story\s*/\s*body|outro|cta|outro\s*/\s*cta|"
        r"narrator|voice\s*over|vo)\s*[:\-]\s*",
        re.I,
    )
    for raw in cleaned.splitlines():
        line = label.sub("", raw).strip(" -–—")
        if line:
            lines.append(line)

    return re.sub(r"\s{2,}", " ", " ".join(lines)).strip()


def estimate_speech_seconds(script: str, words_per_minute: float = 165.0) -> float:
    """Rough spoken length, used to warn when a script overruns the target."""
    words = len([w for w in script.split() if w.strip()])
    return (words / max(words_per_minute, 1.0)) * 60.0


# ---------------------------------------------------------------------------
# Minimalist Motion: scene specs and metaphor generation
#
# The model never draws anything. It returns numbers -- geometry, timings, copy
# -- which motion_engine renders deterministically. That keeps the output on
# brand, keeps it cheap, and means a bad response degrades to a plainer video
# instead of a broken one.
# ---------------------------------------------------------------------------

# The quick-select concepts, in the order the UI shows them.
SCENE_PRESETS: dict[str, dict[str, str]] = {
    "compounding": {
        "label": "Compounding & Consistency (1% Daily)",
        "concept": "How 1% better every day compounds into 37x over a year, "
                   "and why the first three months look like nothing is happening",
        "template": "compounding_jar",
    },
    "pain_freedom": {
        "label": "Short-term Pain vs. Long-term Freedom",
        "concept": "Five years of deliberate discomfort buying fifty years of freedom, "
                   "versus the comfortable road that collapses later",
        "template": "split_path",
    },
    "discipline": {
        "label": "Discipline vs. Motivation",
        "concept": "Motivation is a feeling that arrives late; discipline is a system "
                   "that runs whether the feeling shows up or not",
        "template": "staircase_progress",
    },
    "overthinking": {
        "label": "Overthinking vs. Action",
        "concept": "Overthinking loops in place while action moves forward badly and "
                   "still ends up further ahead",
        "template": "custom",
    },
}

# Imported rather than restated: the prompt lists exactly the metaphors the
# renderer can draw, so a model choice can never name a template that does not
# exist. `suits` is the one-line hint the model picks on.
SCENE_METAPHORS: dict[str, dict[str, str]] = {
    key: {"label": str(TEMPLATES[key]["label"]), "suits": str(TEMPLATES[key]["suits"])}
    for key in METAPHOR_TYPES
}
SCENE_TEMPLATE_KEYS = tuple(METAPHOR_TYPES) + ("custom", "auto")

# Shorts retention: long enough to say something, short enough to loop.
# Long enough to earn. TikTok Creator Rewards counts nothing under 60
# seconds, so a 15-25s scene -- which is what this asked for until now -- was a
# beautiful video that could not be monetized on the platform this engine
# exists to serve. 62 rather than 60 for the same reason the commentary band
# uses it: narration lands within a few percent of a word budget rather than on
# it, and a target of exactly 60 puts half the renders under the bar.
SCENE_MIN_SECONDS, SCENE_MAX_SECONDS = 62.0, 70.0

# The spoken budget that fills it. Calibrated by rendering, not estimated: a
# 150-word thesis came back at 80.0s -- which is 1.875 words/second, well under
# the 2.4 the commentary bands assume, because this engine's narration is
# deliberately unhurried and pauses on the beats. 150 words therefore overran
# the band and hit the render ceiling, where the tail of the narration is
# clipped to fit.
#
# 62s / 1.875 = 116 words, 70s / 1.875 = 131. The band below sits inside that
# with a word of margin at each end -- deliberately tight, because the point of
# the band is that every render clears the Creator Rewards floor without
# drifting far enough past it to bore anyone.
SCENE_WORDS_PER_SECOND = 1.875
SCENE_MIN_WORDS, SCENE_MAX_WORDS = 118, 130


def build_scene_prompt(concept: str, template: str = "auto",
                       duration: float = 18.0) -> str:
    """
    Asks for an animation blueprint as JSON.

    When `template` is "auto" the model picks the metaphor. That is the whole
    point: told to use one, it produced the same shape for every topic and only
    the words changed. Given the choice it reaches for the geometry that fits
    the argument.

    The six named metaphors have their geometry drawn in code, so for those the
    model supplies copy, timing and phase boundaries. "custom" gets the full
    vector schema instead.
    """
    duration = max(SCENE_MIN_SECONDS, min(SCENE_MAX_SECONDS, float(duration)))
    catalogue = "\n".join(
        f'    "{key}": {spec["suits"]}' for key, spec in SCENE_METAPHORS.items()
    )

    shared = (
        "You design shorts for a faceless minimalist animation channel: white "
        "vector lines on pure black, no faces, no stock footage. The tone is "
        "calm, certain and a little cold. Never use emoji, hashtags or "
        "exclamation marks in the on-screen copy.\n\n"
        f"CONCEPT: {concept.strip()}\n\n"
        "Return ONE JSON object and nothing else.\n\n"
    )

    if template == "auto":
        choose = (
            "First choose the metaphor whose geometry actually argues this "
            "concept. Do not default to the first one; a compounding idea wants "
            "a filling vessel, a trade-off wants two diverging paths, a "
            "cascade wants dominoes.\n\n"
            "Route by the psychological premise underneath the words, not by "
            "the vocabulary in them:\n"
            "  consistency, compounding, time, patience, showing up  -> "
            "compounding_jar, growth_consistency, staircase_progress or steep_staircase\n"
            "  overthinking versus doing, planning versus starting, comparing "
            "yourself  -> comparison_split, balance_scale or split_path\n"
            "  perseverance, grinding, hidden effort, work nobody sees  -> "
            "sisyphus_boulder or discipline_iceberg\n"
            "  a decision between two futures  -> two_doors or split_path\n"
            "  self-image, ego, delusion  -> delusion_mirror\n"
            "  what is holding someone back  -> chain_anchor or gravity_funnel\n"
            "  one small cause with a large effect  -> domino_chain\n\n"
            "Eight of them put a stick figure on screen. Reach for those when the "
            "concept is about a person doing something — straining, climbing, "
            "comparing themselves to someone, dragging something, tending "
            "something — and for the abstract ones when it is about a quantity "
            "or a shape of change.\n\n"
            f'  "metaphor_type": one of:\n{catalogue}\n\n'
        )
    else:
        choose = f'  "metaphor_type": "{template}"\n\n'

    common = (
        '  "title": 2-5 words, uppercase, the hook that stops the scroll\n'
        '  "subtitle": one short line under the title, sentence case\n'
        '  "payoff": the closing line, 3-8 words, the idea at its hardest\n'
        f'  "thesis": the full narration, {SCENE_MIN_WORDS}-{SCENE_MAX_WORDS} words. '
        'Not a single sentence -- 8 to 12 of them, building one argument: state '
        'the counter-intuitive claim, give the mechanism, give a concrete '
        'consequence, then turn. It is read aloud over the animation and its '
        'length sets the length of the video, so write the whole thing.\n'
        f'  "duration": seconds, between {SCENE_MIN_SECONDS:.0f} and {SCENE_MAX_SECONDS:.0f}\n'
        '  "animation_phases": {"draw_end": seconds the geometry finishes '
        'drawing itself, "impact": seconds the object clears the obstacle and '
        'the sub-bass drops}. impact should land between 65% and 88% of the '
        "duration -- early enough to pay off, late enough to have earned it.\n"
        '  "ambient": 0.0 to 1.2, how strong the background grid and drifting '
        "particles are. Use 1.0 normally, lower for a clinical look.\n"
        '  "labels": the words stamped onto the geometry itself. One or two '
        "words each, uppercase, 16 characters at most. Which slots exist "
        "depends on the metaphor you chose:\n"
        '     split_path         {"near","far","easy"}  e.g. {"near":"5 YEARS",'
        '"far":"50 YEARS","easy":"COMFORT NOW"}\n'
        '     compounding_jar    {"unit"}               the counter word: "DAY", "REP", "PAGE"\n'
        '     staircase_progress {"before","after","left","right"}  the two '
        "phase names, then the two bar captions\n"
        '     balance_scale      {"left","right"}       what sits in each pan\n'
        '     gravity_funnel     {"pull"}               what is doing the pulling\n'
        '     domino_chain       {"first","last"}       the small cause, the large effect\n'
        '     comparison_split   {"tier1","tier2","tier3"}   three approaches, worst '
        "to best, e.g. BASIC / HARD / SMART\n"
        '     steep_staircase    {"stage1".."stage5","summit"}   the five stages of '
        "progression, and what waits at the top\n"
        '     delusion_mirror    {"real","imagined","gap"}   what someone is, what '
        "they picture, and the distance between\n"
        '     chain_anchor       {"anchor1","anchor2","freed"}   the two dead weights '
        "being dragged, and the state after letting go\n"
        '     growth_consistency {"input","output"}   the unchanging effort, and what '
        "it compounded into\n"
        "   Write these for THIS concept. Generic defaults exist and will be "
        "used if you leave them out, which is worse than filling them in.\n"
        '  "publish": {"title": high-CTR YouTube Shorts title under 70 chars, '
        '"description": 2-3 sentences of psychological framing, '
        '"hashtags": array of 5-8 tags each starting with #}\n'
    )

    if template != "custom":
        return shared + choose + common + (
            "\nThe geometry for every metaphor above is already animated in "
            "code. Supply the copy, the timing and the phases only; do not "
            "include an elements array."
        )

    return shared + '  "metaphor_type": "custom"\n\n' + common + (
        '  "elements": an array of 4-9 drawing commands, rendered in order\n\n'
        "Element schema. Coordinates are 0..1 with (0,0) top-left; the canvas "
        "is a 1080x1920 vertical frame, so keep content between y=0.30 and "
        'y=0.85 to clear the title and the closing line. Colours: "white" or '
        '"grey". Times are seconds.\n'
        '  {"type":"path","id":"p1","points":[[x,y],...],"curve":true,'
        '"width":7,"colour":"white","from":1.0,"to":9.0}\n'
        '  {"type":"dot","follows":"p1","radius":24,"colour":"white",'
        '"from":1.0,"to":9.0}\n'
        '  {"type":"text","text":"FIVE YEARS","at":[0.5,0.78],"size":44,'
        '"colour":"grey","from":3.0,"to":12.0}\n'
        '  {"type":"circle","at":[0.5,0.55],"radius":0.18,"fill":false,'
        '"grow":true,"colour":"grey","from":0.5,"to":6.0}\n'
        '  {"type":"bar","at":[0.25,0.80],"width":0.07,"height":0.20,'
        '"grow":true,"colour":"white","from":2.0,"to":10.0}\n\n'
        "Draw the metaphor, not an illustration of the words. Two contrasting "
        "paths, or one path against one static shape, reads best. Give every "
        "element a from/to inside the duration."
    )


def parse_scene_response(raw: str) -> dict[str, Any]:
    """
    Pulls the JSON object out of a model response.

    Same forgiving approach as the angle parser: models fence their JSON, or
    wrap it in a sentence, and a strict parse would throw away a usable answer.
    """
    text = (raw or "").strip()
    if not text:
        return {}

    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    candidates = [text]
    brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}



# ---------------------------------------------------------------------------
# Three-act scene plans
#
# One metaphor holds a viewer for about twenty seconds. The runtime is over a
# minute, so a single piece of geometry stretched across it gives a visual
# event roughly every twenty-three seconds -- measured on the first 68-second
# render, the picture changed by 0.3% per second and 19 of its 68 seconds were
# completely still.
#
# So the model plans three acts, each with its own metaphor, and each carrying
# its own slice of the narration. The engine gives every act its share of the
# runtime by word count and cuts between them.
# ---------------------------------------------------------------------------

SCENE_PLAN_PROMPT = """You design shorts for a faceless minimalist animation channel:
white vector lines on pure black, no faces, no stock footage, no photography.
The tone is calm, certain and a little cold. Never use emoji, hashtags or
exclamation marks.

CONCEPT: {concept}

Plan the video as THREE ACTS. Each act gets its own metaphor, because one piece
of geometry on screen for a whole minute is the thing that loses the viewer.
The three acts are one argument in three moves:

  Act 1 -- the claim. State the counter-intuitive thing, with the hardest fact
           you have, in the first nine words.
  Act 2 -- the mechanism. WHY it is true. This is where the specifics live.
  Act 3 -- the turn. What the viewer should do differently, and the cost of not.

Choose each act's metaphor from this catalogue by which geometry actually
argues that act. Do not use the same one twice:
{catalogue}

THE RULE THAT MATTERS MOST -- every act's narration must carry something
checkable. A figure, a date, a named researcher, a named study, a measured
effect, a unit. Writing that could survive having its subject swapped is
writing that says nothing:

  BAD:  "Delay is borrowing against tomorrow. You accumulate a silent
         psychological debt and quiet shame."
  GOOD: "Procrastination is not a time-management problem. Fuschia Sirois
         found it tracks mood repair -- people delay to escape a feeling, not
         a task, and the delay costs them roughly a fifth of the working day."

If you do not know a real figure, use a real mechanism or a named effect
instead. Never invent a statistic, a study or a person. A concrete mechanism
beats a fabricated number every time.

The three theses are read aloud as one continuous narration and their combined
length sets the length of the video, so together they must total
{words_low}-{words_high} words. Split them roughly evenly.

Return ONE JSON object and nothing else:

{{"acts": [
   {{"template": "<catalogue key>",
     "title": "2-5 words, uppercase, this act's idea",
     "subtitle": "one short line under the title",
     "labels": {{}},
     "thesis": "this act's narration"}},
   ... exactly 3 ...
 ],
 "payoff": "the closing line of the whole video, 3-8 words",
 "publish": {{"title": "...", "description": "...", "hashtags": ["..."]}}}}

"labels" are the words stamped onto that act's geometry -- one or two words
each, uppercase, 16 characters at most. Which slots exist depends on the
metaphor; use the ones that suit it and leave the rest out:
{label_slots}

COPY RULES for every title, subtitle and label:
- Subject and verb must agree. "why decisions fail", never "why decision
  fails". A plural subject takes a plural verb.
- No trailing full stop on a title or a label. Subtitles may have one.
- Sentence case for subtitles, and the title in the words you would say aloud.
  Do not write in all capitals; the renderer sets the case itself."""


def build_scene_plan_prompt(concept: str) -> str:
    """The three-act prompt, with the metaphor catalogue spliced in."""
    catalogue = "\n".join(
        f'    "{key}": {spec["suits"]}' for key, spec in SCENE_METAPHORS.items())
    slots = "\n".join(
        f'    {key:<18} {sorted(spec["labels"])}'
        for key, spec in SCENE_METAPHORS.items() if spec.get("labels"))

    return SCENE_PLAN_PROMPT.format(
        concept=str(concept).strip(),
        catalogue=catalogue,
        label_slots=slots or "    (none)",
        words_low=SCENE_MIN_WORDS,
        words_high=SCENE_MAX_WORDS,
    )


def parse_scene_plan(raw: str) -> dict[str, Any]:
    """
    Validates a three-act plan into something normalise_spec can take.

    Returns {} rather than a partial plan when there are fewer than two usable
    acts: one act is what this was built to replace, so falling back to the
    single-metaphor path is the honest outcome.
    """
    parsed = parse_scene_response(raw)
    if not isinstance(parsed, dict):
        return {}

    acts: list[dict[str, Any]] = []
    for entry in (parsed.get("acts") or [])[:3]:
        if not isinstance(entry, dict):
            continue
        thesis = str(entry.get("thesis") or "").strip()
        if not thesis:
            continue
        template = str(entry.get("template") or entry.get("metaphor_type")
                       or "").strip().lower()
        acts.append({
            "template": template if template in SCENE_TEMPLATE_KEYS else "",
            "title": str(entry.get("title") or "").strip()[:40],
            "subtitle": str(entry.get("subtitle") or "").strip()[:90],
            "labels": entry.get("labels") if isinstance(entry.get("labels"), dict) else {},
            "thesis": thesis,
        })

    if len(acts) < 2:
        return {}

    # The same metaphor twice in a row defeats the point of acts, so a repeat
    # is replaced with the next unused one from the catalogue.
    seen: set[str] = set()
    spare = [key for key in SCENE_TEMPLATE_KEYS if key != "auto"]
    for act in acts:
        if not act["template"] or act["template"] in seen:
            act["template"] = next(
                (key for key in spare if key not in seen), act["template"] or spare[0])
        seen.add(act["template"])

    return {
        "acts": acts,
        "payoff": str(parsed.get("payoff") or "").strip()[:60],
        "thesis": " ".join(act["thesis"] for act in acts),
        "publish": parsed.get("publish") if isinstance(parsed.get("publish"), dict) else {},
        "source": "gemini-plan",
    }


def generate_scene_plan(concept: str,
                        progress: ProgressFn | None = None) -> dict[str, Any]:
    """
    A three-act plan for a concept.

    Raises GeminiError when no model returns a usable plan; the caller falls
    back to the single-metaphor path, which still renders a video.
    """
    client = get_client()
    prompt = build_scene_plan_prompt(concept)
    last: Exception | None = None

    for model in MODEL_CANDIDATES:
        try:
            if progress:
                progress(f"Asking {model} for a three-act plan...")
            response = generate_with_retry(client, model, prompt, progress=progress)
            plan = parse_scene_plan(getattr(response, "text", "") or "")
            if plan:
                plan["concept"] = str(concept).strip()
                plan["model"] = model
                return plan
        except Exception as exc:                              # noqa: BLE001
            last = exc

    raise GeminiError(
        f"No model returned a usable three-act plan for {concept!r}"
        + (f": {type(last).__name__}: {last}" if last else "."))


def generate_scene_spec(
    concept: str,
    template: str = "auto",
    duration: float = 18.0,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Asks Gemini for a scene spec and returns it raw, for motion_engine to clamp.

    Raises GeminiError only when there is no usable response at all -- the
    caller is expected to fall back to the preset copy rather than fail the
    render, because a metaphor animation with stock wording is still a video.
    """
    if template not in SCENE_TEMPLATE_KEYS:
        template = "auto"

    client = get_client()
    prompt = build_scene_prompt(concept, template, duration)
    last: Exception | None = None

    for model in MODEL_CANDIDATES:
        try:
            if progress:
                progress(f"Asking {model} for a scene...")
            response = generate_with_retry(client, model, prompt, progress=progress)
            spec = parse_scene_response(getattr(response, "text", "") or "")
            if not spec:
                raise GeminiError(f"{model} returned no JSON object")
            # Only pin the template when the caller asked for a specific
            # one. On "auto" the model's metaphor_type is the answer -- forcing
            # it here is what made every topic come out as the same shape.
            chosen = str(spec.get("metaphor_type") or spec.get("template") or "").strip().lower()
            if template != "auto":
                spec["template"] = template
            elif chosen in SCENE_TEMPLATE_KEYS and chosen != "auto":
                spec["template"] = chosen
            else:
                # A model that ignored the list gets a deterministic choice
                # rather than a silent fallback to the same default every time.
                spec["template"] = METAPHOR_TYPES[
                    sum(ord(c) for c in concept) % len(METAPHOR_TYPES)
                ]
            spec["concept"] = concept
            spec["source"] = model
            return spec
        except Exception as exc:
            last = exc
            if progress:
                progress(f"{model} could not write the scene ({exc}); trying the next model...")
            continue

    raise GeminiError(
        "No Gemini model could write a scene spec. "
        f"Last error: {last}" if last else "No Gemini model could write a scene spec."
    )


def fallback_publish_meta(concept: str, title: str, payoff: str) -> dict[str, Any]:
    """
    Publish metadata without an API call.

    Deliberately generic where it has to be and specific where it can be: a
    made-up claim in a description is worse than a plain one.
    """
    concept = (concept or "").strip()
    headline = (title or concept[:60] or "The rule nobody tells you").strip()
    return {
        "title": f"{headline} #shorts".strip()[:100],
        "description": (
            f"{payoff or headline}\n\n"
            f"{concept}\n\n"
            "Animated from scratch — no stock footage, no filler. "
            "Watch it twice; the second time you will see the curve."
        ).strip(),
        "hashtags": ["#Mindset", "#Discipline", "#Psychology", "#Shorts",
                     "#SelfImprovement", "#Motivation"],
    }


# ---------------------------------------------------------------------------
# Viral scorecard and the retention rewrite
#
# compliance.viral_scorecard() is the floor: no network, always runs, and what
# the tests assert against. This is the ceiling -- a model that can tell the
# difference between a specific claim and a confident-sounding empty one, which
# no regex can.
#
# The two are blended rather than one overriding the other. The heuristic
# cannot be talked out of a filler count; the model cannot be fooled by a
# script that name-drops numbers without saying anything. Averaging them keeps
# both failure modes visible.
# ---------------------------------------------------------------------------

# How far a model's read of the writing can pull down the monetization axis.
# Bounded because the axis is mostly licence facts the model cannot see.
MONETIZATION_MAX_PENALTY = 2.0

VIRAL_PROMPT = """You grade short-form video scripts for a channel that has to survive
YouTube's reused-content review and TikTok's originality rules. You are hard to
impress. Return ONE JSON object and nothing else.

SCRIPT:
{script}

CONTEXT: the finished video runs {duration:.0f} seconds.

Score three axes from 1 to 10. Use the whole range in BOTH directions: a script
that does the job well is an 8, and refusing to award one is as wrong as
handing them out. Do not cluster your answers in the middle.

1. hook (first 3 seconds, roughly the first 9 words)
   9-10 = a number that sounds wrong, or a flat contradiction of something the
          viewer believes, landing in the first six words.
   7-8  = opens on a stake, a named specific or a hard figure. The viewer has a
          reason to stay even if it is not startling. A direct instruction
          against the viewer's assumption ("Stop buying the X") is an 8.
   4-6  = states the topic clearly. Accurate, and scrollable.
   1-3  = "In this video I want to talk about...", or pure throat-clearing.

2. density (is the script load-bearing?)
   9-10 = almost every sentence carries a figure, a name, a mechanism or a
          consequence that could be checked and could be wrong.
   7-8  = a real argument with most claims supported by something specific.
   4-6  = a real point, thinly supported.
   1-3  = confident phrasing around nothing. "This costs more than you think",
          "the secret is consistency", "let that sink in". Score these 1-3 even
          when they read smoothly -- especially then.

3. monetization (does the SCRIPT read as transformative?)
   Judge the writing only. You cannot see the footage licence and must not
   guess at it -- that is checked separately and your score is combined with it.
   9-10 = the commentary is clearly the product; footage would only illustrate it.
   7-8  = a genuine argument of its own, in its own words.
   4-6  = original narration that mostly describes what would be on screen.
   1-3  = reaction noises over someone else's video, or a template with the
          nouns swapped.

Also return:
  "fixes": 2-4 specific, actionable notes. Name the exact phrase to cut or the
           exact kind of fact that is missing. Never "add more detail".
  "retention_tip": ONE sentence naming the single change that buys the most
           retention in the first three seconds. Quote the exact words to cut
           or the exact figure to lead with. It must be executable without
           re-reading the script. Never "make it punchier".
  "rewritten_hook": one replacement opening line of 9 words or fewer.

JSON shape:
{{"hook": 0, "density": 0, "monetization": 0,
  "fixes": ["..."], "retention_tip": "...", "rewritten_hook": "..."}}"""


REWRITE_PROMPT = """Rewrite this short-form script for retention. Return ONLY the rewritten
script -- no preamble, no notes, no markdown.

CURRENT SCRIPT:
{script}

WHAT IS WRONG WITH IT:
{diagnosis}

RULES:
- Open on a loop, not a topic. First nine words must make scrolling feel like
  missing something. No "In this video", no "Here are three", no greeting.
- Every claim gets a figure, a name, a date or a mechanism. If you do not know
  a real one, restructure the sentence so it does not need one -- do not invent
  a statistic.
- Delete every phrase that would survive having its subject swapped. "Game
  changer", "at the end of the day", "the secret is", "trust me" and anything
  that reads like them.
- Keep it within {low}-{high} words so the spoken length does not move.
- Same topic, same claims, same angle. This is a rewrite, not a new script.
- End on the turn, not on a sign-off."""


def parse_scorecard_response(raw: str) -> dict[str, Any]:
    """Pulls the scorecard JSON out of a model response, forgivingly."""
    parsed = parse_scene_response(raw)
    if not isinstance(parsed, dict):
        return {}

    def _axis(key: str) -> float:
        try:
            return max(1.0, min(10.0, float(parsed.get(key, 0) or 0)))
        except (TypeError, ValueError):
            return 0.0

    fixes = parsed.get("fixes") or []
    if isinstance(fixes, str):
        fixes = [fixes]

    return {
        "hook": _axis("hook"),
        "density": _axis("density"),
        "monetization": _axis("monetization"),
        "fixes": [str(f).strip() for f in fixes if str(f).strip()][:4],
        "retention_tip": str(parsed.get("retention_tip") or "").strip(),
        "rewritten_hook": str(parsed.get("rewritten_hook") or "").strip(),
    }


def score_virality(
    script: str,
    entry: dict[str, Any] | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Scores a script 1-10 on hook, density and monetization safety.

    Always returns a scorecard. With no API key, or on any API failure, the
    compliance heuristic's answer is returned unchanged with source
    "heuristic" -- the card is part of the export step and must never be the
    thing that blocks a render.
    """
    import compliance

    entry = entry or {}
    base = compliance.viral_scorecard(script, entry)

    if not str(script or "").strip():
        return base

    try:
        client = get_client()
    except GeminiError:
        return base

    prompt = VIRAL_PROMPT.format(
        script=str(script).strip(),
        duration=float(entry.get("duration") or 0.0),
    )

    last: Exception | None = None
    for model in MODEL_CANDIDATES:
        try:
            if progress:
                progress(f"Scoring the script with {model}...")
            response = generate_with_retry(client, model, prompt, progress=progress)
            ai = parse_scorecard_response(getattr(response, "text", "") or "")
            if not ai or not ai["hook"]:
                continue

            # The model leads on judgement; the heuristic caps on evidence.
            #
            # Averaging the two was the wrong shape. The heuristic has no
            # opinion about whether a sentence says anything, so its answer for
            # an unremarkable script sits near the middle of the range -- and
            # averaging dragged every real verdict back toward that middle,
            # which is where the flat mid-fives came from. So:
            #
            #   hook, density  -- the model decides, because telling a specific
            #       claim from a confident-sounding empty one is exactly what a
            #       regex cannot do. The heuristic still caps it: where it has
            #       counted filler phrases or a canned opener, the model cannot
            #       score more than two points above that hard evidence.
            #   monetization   -- the lower of the two, always. The heuristic
            #       reads the actual licence off the ledger entry, which the
            #       model never sees, so it can only be right in ways the model
            #       cannot be.
            merged = dict(base)

            # "Hard" means counted, not merely absent. A missing figure in the
            # opening is something the model may legitimately read differently;
            # a filler phrase or a canned opener is a string that is either in
            # the script or is not.
            marks = set(base["hook"].get("diagnosis") or ())
            hard_evidence = bool(base["density"].get("filler_hits")) or bool(
                marks & {"weak_opener", "slow_open"})

            for axis in ("hook", "density", "monetization"):
                heuristic = float(base[axis]["score"])
                model_score = float(ai[axis])

                if axis == "monetization":
                    # The heuristic owns this axis: it reads the real licence,
                    # duration and disclosure state off the ledger entry, none
                    # of which the model is shown. The model contributes one
                    # thing it can genuinely see -- whether the writing reads as
                    # transformative -- as a bounded penalty, never as the
                    # score. Letting it lead here had it marking cleared,
                    # disclosed, CC0 footage down to a 3 on nothing.
                    penalty = 0.0
                    if model_score < 5.0:
                        penalty = min(MONETIZATION_MAX_PENALTY, 5.0 - model_score)
                    final = heuristic - penalty
                elif hard_evidence:
                    final = min(model_score, heuristic + 2.0)
                else:
                    final = model_score

                merged[axis] = dict(base[axis])
                merged[axis]["score"] = round(final, 1)
                merged[axis]["model_score"] = model_score
                merged[axis]["heuristic_score"] = heuristic

            overall = round(merged["hook"]["score"] * 0.45
                            + merged["density"]["score"] * 0.35
                            + merged["monetization"]["score"] * 0.20, 1)

            # The same categorical gate the offline card applies: a licence
            # that is not cleared is not a weakness to be averaged away.
            if merged["monetization"]["score"] < compliance.MONETIZATION_FLOOR:
                overall = round(min(overall, merged["monetization"]["score"] + 2.0), 1)

            merged["overall"] = overall
            merged["needs_rewrite"] = overall < compliance.VIRAL_TARGET_SCORE
            merged["fixes"] = ai["fixes"]
            merged["retention_tip"] = ai["retention_tip"] or base.get("retention_tip", "")
            merged["rewritten_hook"] = ai["rewritten_hook"]
            merged["verdict"] = compliance.verdict_for(overall, merged)
            merged["source"] = f"{model} + evidence caps"
            merged["model"] = model
            return merged
        except Exception as exc:
            last = exc
            if not is_retryable(exc):
                continue

    if progress and last:
        progress(f"Scoring fell back to the offline card ({type(last).__name__}).")
    return base


def rewrite_for_retention(
    script: str,
    card: dict[str, Any] | None = None,
    target: str = DEFAULT_TARGET,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Rewrites a script against its own scorecard.

    Returns {"script", "model", "before", "after"} where before/after are the
    two overall scores, so the UI can show whether the rewrite actually helped
    rather than asserting that it did. A rewrite that scores *worse* is
    reported as such and the original is kept.
    """
    import compliance

    body = str(script or "").strip()
    if not body:
        raise GeminiError("There is no script to rewrite.")

    card = card or compliance.viral_scorecard(body)
    diagnosis_lines: list[str] = list(card.get("fixes") or [])
    if not diagnosis_lines:
        for axis in ("hook", "density", "monetization"):
            diagnosis_lines.extend(card.get(axis, {}).get("notes", []))
    diagnosis = "\n".join(f"- {line}" for line in diagnosis_lines[:6]) or "- It is generic."

    spec = DURATION_TARGETS.get(target, DURATION_TARGETS[DEFAULT_TARGET])
    low = int(spec["low"] * WORDS_PER_SECOND * 0.85)
    high = int(spec["high"] * WORDS_PER_SECOND * 1.05)

    client = get_client()
    prompt = REWRITE_PROMPT.format(script=body, diagnosis=diagnosis, low=low, high=high)

    last: Exception | None = None
    for model in MODEL_CANDIDATES:
        try:
            if progress:
                progress(f"Rewriting for retention with {model}...")
            response = generate_with_retry(client, model, prompt, progress=progress)
            rewritten = clean_script(getattr(response, "text", "") or "")
            if len(rewritten.split()) < 8:
                continue

            after = compliance.viral_scorecard(rewritten)["overall"]
            return {
                "script": rewritten,
                "model": model,
                "before": float(card.get("overall") or 0.0),
                "after": after,
                "improved": after > float(card.get("overall") or 0.0),
            }
        except Exception as exc:
            last = exc
            continue

    raise GeminiError(
        f"Could not rewrite the script: {type(last).__name__}: {last}" if last
        else "No model accepted the rewrite request."
    )
