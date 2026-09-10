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

from motion_engine import METAPHOR_TYPES, TEMPLATES

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
    "rewards": {
        "label": "70–90s · TikTok Rewards eligible (over 1 min)",
        "low": 70, "high": 90,
        "body": ("9-12 vivid sentences that build a full story arc: set the scene, "
                 "raise the stakes, deliver a turn, then land the payoff"),
    },
}
DEFAULT_TARGET = "standard"


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
) -> Any:
    """
    Uploads a clip and polls until Gemini reports it ACTIVE.

    A freshly uploaded video sits in PROCESSING for a while; calling
    generate_content against it before it goes ACTIVE fails, so this blocks
    until the file is genuinely usable.
    """
    if not os.path.exists(video_path):
        raise GeminiError(f"Video file not found: {video_path}")

    size_mb = os.path.getsize(video_path) / 1_048_576
    if progress:
        progress(f"Uploading clip to Gemini ({size_mb:.1f} MB)...")

    try:
        uploaded = client.files.upload(file=video_path)
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
            raise GeminiError(
                "Gemini could not process this clip. Try a standard H.264 MP4 "
                "(some MOV/HEVC files are rejected)."
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
SCENE_MIN_SECONDS, SCENE_MAX_SECONDS = 15.0, 25.0


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
            f'  "metaphor_type": one of:\n{catalogue}\n\n'
        )
    else:
        choose = f'  "metaphor_type": "{template}"\n\n'

    common = (
        '  "title": 2-5 words, uppercase, the hook that stops the scroll\n'
        '  "subtitle": one short line under the title, sentence case\n'
        '  "payoff": the closing line, 3-8 words, the idea at its hardest\n'
        '  "thesis": one spoken sentence, 18-32 words, what a narrator reads\n'
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
