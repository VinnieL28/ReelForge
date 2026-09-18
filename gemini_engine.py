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

from minimalist_engine import (LABEL_SLOTS, METAPHOR_TYPES, TEMPLATES,
                               missing_label_slots, plan_act_count)

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
# The moves an argument can make, in the order they earn their place.
#
# An act is a beat, not a third of the runtime. Adding acts to a longer video
# should add *moves* -- the evidence, the objection -- rather than stretch the
# same three, which is what produced 22-second acts where the animation
# finished in six.
ACT_BEATS: tuple[tuple[str, str], ...] = (
    ("CLAIM", "the counter-intuitive thing, with the hardest fact you have, "
              "in the first nine words"),
    ("EVIDENCE", "the figure, named study or named effect that makes the claim "
                 "impossible to wave away"),
    ("MECHANISM", "WHY it is true -- the causal step, not a restatement"),
    ("SCALE", "what it costs at size: over a career, across a population, "
              "compounded over years"),
    ("OBJECTION", "the obvious objection a sharp viewer is already forming, "
                  "and why it does not hold"),
    ("TURN", "what the viewer does differently, concretely, and what it costs "
             "them not to"),
)

# Which beats a plan of N acts uses. CLAIM always opens and TURN always closes;
# what fills the middle is what the extra runtime buys.
_BEAT_LADDER: dict[int, tuple[int, ...]] = {
    2: (0, 5),
    3: (0, 2, 5),
    4: (0, 1, 2, 5),
    5: (0, 1, 2, 3, 5),
    6: (0, 1, 2, 3, 4, 5),
    7: (0, 1, 2, 3, 4, 2, 5),
    8: (0, 1, 2, 3, 4, 1, 2, 5),
}


def act_beats(count: int) -> list[tuple[str, str]]:
    """The named beats for a plan of `count` acts."""
    count = max(2, min(8, int(count)))
    return [ACT_BEATS[i] for i in _BEAT_LADDER[count]]


# Which metaphors each domain reaches for first, and what counts as evidence
# there.
#
# Two videos on unrelated topics were coming out looking like the same video,
# because the model was given the same fourteen shapes in the same order with
# no sense of what the topic was. Routing by domain is what makes a finance
# short and a psychology short *look* different before the words are written --
# and the evidence line is what stops the narration being true-sounding filler
# in either one.
DOMAIN_GUIDE: dict[str, dict[str, Any]] = {
    "finance": {
        "metaphors": ("compounding_jar", "split_path", "comparison_split",
                      "gravity_funnel", "balance_scale", "growth_consistency",
                      "domino_chain", "staircase_progress"),
        "evidence": "Real instruments and real institutions: SPIVA, expense "
                    "ratios, the 4% rule, a 7% real return, VOO or VTI by "
                    "name, actual basis points. Compounding arguments are made "
                    "over decades, never over days -- set axis_max to the "
                    "number of years and axis_suffix to \"y\".",
    },
    "psychology": {
        "metaphors": ("delusion_mirror", "chain_anchor", "two_doors",
                      "balance_scale", "split_path", "gravity_funnel",
                      "comparison_split", "domino_chain"),
        "evidence": "Named effects and the people who found them: Iyengar and "
                    "Lepper on jam, Baumeister on depletion, Sirois on mood "
                    "repair, Zeigarnik, Dunning and Kruger. Name the effect, "
                    "then say what it predicts.",
    },
    "health": {
        "metaphors": ("growth_consistency", "staircase_progress",
                      "compounding_jar", "sisyphus_boulder", "discipline_iceberg",
                      "comparison_split", "split_path", "steep_staircase"),
        "evidence": "Measured quantities with their units: VO2 max, grams of "
                    "protein per kilo, hours of sleep, resting heart rate, "
                    "all-cause mortality. Never a percentage without what it "
                    "is a percentage of.",
    },
    "education": {
        "metaphors": ("domino_chain", "gravity_funnel", "steep_staircase",
                      "comparison_split", "staircase_progress", "split_path",
                      "compounding_jar", "balance_scale"),
        "evidence": "The mechanism, in order, with the real numbers: how much, "
                    "how long, how many. Name the system or the project.",
    },
    "history": {
        "metaphors": ("domino_chain", "chain_anchor", "steep_staircase",
                      "gravity_funnel", "split_path", "comparison_split",
                      "sisyphus_boulder", "two_doors"),
        "evidence": "Dates, places and named people. One decision, its year, "
                    "and what followed from it.",
    },
    "tech_b2b": {
        "metaphors": ("comparison_split", "domino_chain", "steep_staircase",
                      "split_path", "gravity_funnel", "balance_scale",
                      "compounding_jar", "growth_consistency"),
        "evidence": "Named systems, versions and measured limits: latency in "
                    "milliseconds, cost per unit, adoption figures with a date "
                    "on them.",
    },
    "general": {
        "metaphors": ("split_path", "compounding_jar", "delusion_mirror",
                      "domino_chain", "two_doors", "balance_scale",
                      "sisyphus_boulder", "comparison_split"),
        "evidence": "A figure, a date, a named researcher, a named study, a "
                    "measured effect or a unit.",
    },
}


def domain_guide(concept: str) -> dict[str, Any]:
    """
    The metaphor pool and evidence standard for this concept's domain.

    Classification lives in niche_engine because that is where the CPM bands
    are, and a topic's domain is the same question in both places.
    """
    try:
        from niche_engine import classify_band

        band = classify_band(concept)
    except Exception:
        band = "general"
    guide = DOMAIN_GUIDE.get(band) or DOMAIN_GUIDE["general"]
    return {"band": band, **guide}


SCENE_METAPHORS: dict[str, dict[str, str]] = {
    key: {"label": str(TEMPLATES[key]["label"]), "suits": str(TEMPLATES[key]["suits"])}
    for key in METAPHOR_TYPES
}
SCENE_TEMPLATE_KEYS = tuple(METAPHOR_TYPES) + ("custom", "auto")

# Which slots each template stamps onto its geometry.
#
# Read from LABEL_SLOTS, which is where they live. The prompt used to build
# this list from SCENE_METAPHORS -- which has no "labels" key at all -- so it
# came out empty and told the model there were no slots to fill. Two things
# depend on getting it from the right table: the prompt lists them, and the
# parser uses them to tell an act that was labelled from one that was not.
LABEL_SLOTS_FOR: dict[str, dict[str, str]] = {
    key: dict(LABEL_SLOTS.get(key) or {}) for key in METAPHOR_TYPES
}

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
# Two measurements, and the first one was wrong. A 150-word thesis "came back
# at 80.0s" -- but 80.0 was MAX_DURATION at the time, so that take was clipped
# and the real rate was somewhere below the 1.875 it implied, not at it. The
# honest measurement is the delivered file: minimalist_1789502356.mp4 read 119
# words in 54.7s of speech, which is 2.17 words a second.
#
# At 2.17, the old 118-130 band buys 54-60 seconds. That is how a render aimed
# at 62-70s arrived at 56.29 and earned nothing.
# The mean of two delivered renders: 119 words in 54.70s (2.176) and 131 in
# 64.44s (2.033). Spread that wide is what a TTS with no fixed cadence does on
# different sentence shapes, so a single figure is a centre and not a promise
# -- which is why the runtime has PAYOUT_FLOOR_SECONDS under it rather than
# trusting this number to land the video above sixty seconds on its own.
# ...and then the voice was told to hurry. With "a brisk, confident pace" in
# place of "unhurried", the same 133-word script read at 2.32 words a second
# against 1.90. The earlier figures were all taken on the unhurried voice.
#
# Two brisk takes are now measured: 133 words in 57.4s (2.32) and 151 in 59.9s
# (2.52). The centre is their mean. 151 words a minute is also, for once,
# exactly the pace both reviews of the slow renders asked for.
SCENE_WORDS_PER_SECOND = 2.42

# Aimed long enough to clear the payout floor and no longer.
#
# 138-150 was set when the tail after the narration was 1.6 seconds. It is
# CLOSING_SECONDS + 0.4 now -- six -- because the closing card was starting
# before the argument finished, and that tail is runtime the narration does not
# have to fill. A 156-word render came out at 74.3s and read at 125 words a
# minute, which is slow for this format.
#
# 130-142 words is 60-65s spoken at the measured 2.17 w/s, plus six seconds of
# card: 66-71s delivered. Over the sixty-second floor with margin, and
# PAYOUT_FLOOR_SECONDS still catches anything that lands short.
#
# 124-132 now, because the card is no longer silent. The payoff is read aloud
# over it (about 2.5s) and the voice is followed by SPOKEN_CLOSE_HOLD rather
# than six seconds of nothing, so the same delivered length needs fewer body
# words. At the fastest measured rate, 2.176, 124 words is 57.0s + 2.5 + 1.6 =
# 61.1s; at the slowest, 2.033, 132 words is 64.9 + 4.1 = 69.0s. Both are
# paid, and the review that called 72 seconds "a lot" gets its few seconds
# back without anyone getting a sixty-second floor taken away.
# How much the rate varies around that centre, as a fraction.
#
# Three takes on the unhurried instruction measured 1.90, 2.03 and 2.18 words a
# second -- a centre of 2.04, none further than 7% from it. The two brisk takes
# sit at 2.32 and 2.52, which is only +-4% of 2.42, and it was tempting to
# narrow this to match. Two samples do not measure a spread: the evidence that
# this voice wanders by 7% stands until there are enough brisk takes to say
# otherwise, and every widening of this number so far has come from a render
# that surprised me.
SCENE_RATE_SPREAD = 0.07


def scene_rate_bounds() -> tuple[float, float]:
    """(slowest, fastest) words a second the narration plausibly reads at."""
    return (SCENE_WORDS_PER_SECOND * (1.0 - SCENE_RATE_SPREAD),
            SCENE_WORDS_PER_SECOND * (1.0 + SCENE_RATE_SPREAD))


# The budget, derived from that range rather than from the centre.
#
# Two limits, pulling opposite ways. The minimum has to clear the TikTok payout
# floor even when the voice reads FAST, because a render that lands at 59
# seconds earns nothing. The maximum has to stay watchable even when it reads
# SLOW -- a 74-second video was called long by both reviews of one.
#
# Two limits at 2.25-2.59 w/s, with a five-word payoff read aloud and
# SPOKEN_CLOSE_HOLD after it, pulling against each other:
#
#   the ceiling  157 words at the slow end is 72.7s. Both reviews of a
#                74.3-second render called it long, so 73s is the cap.
#   the floor    pinning the minimum so that even a FAST read needs no silent
#                padding would want 153 words, and 153-157 is a four-word band.
#                Every budget so far has been overshot -- 138-150 produced 164,
#                124-132 produced 133, 140-148 produced 151 -- so a four-word
#                band is not a spec, it is a coin toss.
#
# So the minimum gives way instead, by the smallest amount that leaves a band
# wide enough to write inside: at 148 words a fast read lands at 60.7s and
# PAYOUT_FLOOR_SECONDS holds the card 2.3s longer. That is a beat on a card
# that would be held anyway, not the six silent seconds a review called out.
SCENE_MIN_WORDS, SCENE_MAX_WORDS = 148, 155


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
DOMAIN: {band}

Plan the video as {act_count} ACTS. Each act gets its own metaphor, because one
piece of geometry on screen for a whole minute is the thing that loses the
viewer -- and because an act that runs longer than about twelve seconds has
finished animating and is holding a still picture for the rest of it.

The acts are one argument, and each one is a different move in it:

{beats}

Choose each act's metaphor from this catalogue by which geometry actually
argues that act. NEVER use the same one twice in one video:
{catalogue}

For this domain, reach first for: {preferred}
Use others when the argument genuinely wants them -- but a video whose shapes
all come from outside its domain's list usually means the argument drifted.

THE RULE THAT MATTERS MOST -- every act's narration must carry something
checkable. Writing that could survive having its subject swapped is writing
that says nothing.

  For this domain, that means: {evidence}

  BAD:  "Delay is borrowing against tomorrow. You accumulate a silent
         psychological debt and quiet shame."
  GOOD: "Procrastination is not a time-management problem. Fuschia Sirois
         found it tracks mood repair -- people delay to escape a feeling, not
         a task, and the delay costs them roughly a fifth of the working day."

If you do not know a real figure, use a real mechanism or a named effect
instead. Never invent a statistic, a study or a person. A concrete mechanism
beats a fabricated number every time.

The theses are read aloud as one continuous narration and their combined
length sets the length of the video. Together they must total
{words_low}-{words_high} words -- split roughly evenly, about {words_per_act}
words each. The payoff is read aloud after the last act, over the closing card,
so write it to be said as well as seen.

ONE SET OF NUMBERS. Decide the assumptions once -- one fee, one rate of return,
one time horizon, one starting amount -- and make every figure in every thesis,
label and payoff follow from them. A render once showed "1.5% FEE" on screen
while the narration said "a one percent fee"; a viewer who knows the subject
reads that as a video that does not.

WORK THE NUMBERS OUT. Say the figure the assumptions actually produce, not a
rounder or more dramatic one. "Half" means close to 50%. At 7% returns a 1%
fee costs about a quarter of the gains over twenty-five years, not half --
the render that said "half" was wrong by two to one in its first sentence.

NO ABSOLUTES unless they are literally true: not "zero", "never", "always",
"guaranteed", "everyone", "no one". "Past winners rarely stay winners" is
true; "past performance carries zero predictive value" is not, and it is the
kind of line that gets a channel corrected in its own comments.

OPEN ON THE FIGURE. The first sentence carries the single most concrete number
in the video -- a dollar amount beats a percentage, a percentage beats an
adjective -- because it is on screen in the first second and it is what stops
the scroll.

PAY OFF THE TITLE. If the concept promises a specific quantity -- "the year
your fees overtake your returns", "the exact moment willpower fails", "how many
hours" -- then one act must put that number on screen in its labels and say it
in its thesis. A video titled "the year X happens" that never names a year has
broken its own promise, and the viewer who came for the number leaves without
it. Decide the figure first, then build the act that lands it.

HOW THE SENTENCES HAVE TO READ. Three rules, each from a delivered render.

  LENGTH. The first sentence is at most 12 words and opens ON the figure. One
  render began "Forty-three percent of daily human actions occur without
  conscious decision, executing as automatic neural chunks that bypass
  executive deliberation entirely once an environmental cue appears" -- 25
  words, and the scroll decision is made in three. "Forty-three percent of
  your day is automatic." is the same claim in eight.

  Every other sentence is at most 16 words, and long ones carry a comma. The
  captions break at punctuation, so a sentence with none in it breaks in the
  middle of a thought: that same render produced "43% OF DAILY" and "INTO THE
  BASAL" on screen.

  PLAIN WORDS. No term a fifteen-year-old would not know, unless the same
  sentence defines it. "Prefrontal control", "executive deliberation" and
  "automatic neural chunks" all reached a delivered video.

  NO INVENTED AGGREGATES. "Approximately one hundred fifty thousand repeated
  choices over thirty years" has no source and sounds like it has one. If you
  did not read a figure somewhere, do not state one -- and never compute a
  figure yourself, because arithmetic done in a sentence comes out wrong: one
  render claimed a 2% fee overtakes returns in "year twenty-three" when it is
  year 31, and that it destroys "three hundred sixty thousand dollars" of a
  million-dollar portfolio when it destroys 3.46 million.

  NO CONTESTED FINDINGS AS FACT. Ego depletion -- willpower as a resource that
  fatigue uses up -- has largely failed to replicate, and a render asserted it
  flatly. The same goes for power posing, the Stanford prison experiment,
  priming effects on behaviour, and learning styles. Where a finding is
  disputed, either say so or choose a different mechanism.

LABEL THE RANKED TEMPLATES IN THE RIGHT ORDER. comparison_split fills only
its bottom bar, so tier3 is the row that WINS. A fees video labelled it
"SPIVA 92% LAG" and the picture then showed a failure statistic winning:

  BAD:  tier1 "ACTIVE 8%", tier2 "BENCHMARK 100%", tier3 "SPIVA 92% LAG"
  GOOD: tier1 "ADVISOR 1.5%", tier2 "ACTIVE FUND 0.6%", tier3 "INDEX 0.03%"

The same applies to balance_scale (the right pan wins), two_doors (the right
door opens) and split_path (the bright path ends high, the flat grey one
crashes). Put what the video RECOMMENDS on the winning side, every time.

THE PICTURE MUST AGREE WITH THE WORDS. A rendered act once argued "how small
fees compound into major losses" over a vessel filling up and a counter
climbing to 9.75x, which told the viewer the opposite of the narration. So:

  "direction": "fill" when the quantity in that act GROWS, "drain" when it is
     LOST, spent, eroded or paid away. Loss is never drawn as growth.
  "axis_max" and "axis_suffix": the scale the argument is actually made on --
     {{"axis_max": 30, "axis_suffix": "y"}} for something that plays out over
     thirty years, {{"axis_max": 90, "axis_suffix": "d"}} for ninety days.
     Compounding in money is a decades-long argument.
  "end_multiple": for compounding_jar ONLY, and only when the narration states
     the multiple out loud -- 2.4 for "your money two and a half times over".
     Leave it out otherwise: the counter then shows how full the vessel is
     rather than a figure nobody claimed. It used to print 37.78x on every
     filling act, which is 1.01^365, and on a habits video that was a
     statistic from nowhere in 66px type.
  "end_value": 0.0-1.0, where the quantity ENDS as a share of where it began.
     A 1% annual fee over thirty years leaves about 0.75, so write 0.75 -- not
     0, which would say the fee took everything. Overstating it is as wrong as
     drawing it backwards.

Return ONE JSON object and nothing else:

{{"acts": [
   {{"template": "<catalogue key>",
     "beat": "<the beat name for this act>",
     "title": "2-5 words, uppercase, this act's idea",
     "subtitle": "one short line under the title",
     "labels": {{}},
     "direction": "fill" | "drain",
     "axis_max": 30, "axis_suffix": "y", "end_value": 0.75,
     "end_multiple": 0,
     "thesis": "this act's narration"}},
   ... exactly {act_count} ...
 ],
 "payoff": "the closing line of the whole video, 3-8 words",
 "cta": "what the last card asks for, 2-4 words, e.g. Follow for more",
 "publish": {{"title": "...", "description": "...", "hashtags": ["..."]}}}}

"labels" are the words stamped onto that act's geometry -- one or two words
each, uppercase, 16 characters at most, with figures written as figures:
"88%", "$200K", "YEAR 25", "3 BPS" -- never "EIGHTY-EIGHT PERCENT". A longer
label is shortened on screen, and a shortened label usually loses the word
that mattered. Which slots exist depends on the metaphor; use the ones that
suit it and leave the rest out:
{label_slots}

FILL THE LABELS FOR EVERY ACT. They are not decoration -- they are the only
words the viewer reads on the geometry itself. A template left unlabelled falls
back to placeholder copy, and a video about active management came out stamped
"5 YEARS / 50 YEARS / COMFORT NOW", which is about nothing at all. Write them
for THIS concept.

COPY RULES for every title, subtitle and label:
- Subject and verb must agree. "why decisions fail", never "why decision
  fails". A plural subject takes a plural verb.
- No trailing full stop on a title or a label. Subtitles may have one.
- Sentence case for subtitles, and the title in the words you would say aloud.
  Do not write in all capitals; the renderer sets the case itself."""


def default_act_count(duration: float = 0.0) -> int:
    """
    How many acts a video of this length should be planned as.

    Delegated to the renderer's own cap rather than fixed here, because the
    number that matters is how long one template can hold the screen before it
    has run out of animation -- and that is a property of the templates.
    """
    return plan_act_count(float(duration) or (SCENE_MIN_SECONDS + 4.0))


def build_scene_plan_prompt(concept: str, acts: int = 0,
                            duration: float = 0.0) -> str:
    """The multi-act prompt, with the catalogue and the domain spliced in."""
    count = max(2, min(8, int(acts))) if acts else default_act_count(duration)
    guide = domain_guide(concept)

    catalogue = "\n".join(
        f'    "{key}": {spec["suits"]}' for key, spec in SCENE_METAPHORS.items())
    # From LABEL_SLOTS_FOR, and showing the default alongside each slot as
    # an example of the KIND of phrase it takes -- not as something to copy.
    # The defaults are written for the metaphor rather than for any subject,
    # which is exactly why an act that fails to override them looks wrong.
    slots = "\n".join(
        f'    {key:<19} ' + ", ".join(
            f'"{slot}" (e.g. "{example}")' for slot, example in sorted(wanted.items()))
        for key, wanted in ((key, LABEL_SLOTS_FOR.get(key) or {})
                            for key in METAPHOR_TYPES) if wanted)
    beats = "\n".join(
        f"  Act {i} -- {name}: {description}"
        for i, (name, description) in enumerate(act_beats(count), start=1))

    return SCENE_PLAN_PROMPT.format(
        concept=str(concept).strip(),
        band=str(guide["band"]).replace("_", " "),
        act_count=count,
        beats=beats,
        catalogue=catalogue,
        preferred=", ".join(guide["metaphors"]),
        evidence=guide["evidence"],
        label_slots=slots or "    (none)",
        words_low=SCENE_MIN_WORDS,
        words_high=SCENE_MAX_WORDS,
        words_per_act=int(round((SCENE_MIN_WORDS + SCENE_MAX_WORDS) / 2 / count)),
    )


def relabel_note(titles: Sequence[str]) -> str:
    """The complaint appended to a retry when acts came back unlabelled."""
    named = ", ".join(f'"{title}"' for title in titles if title) or "some acts"
    return (
        "\n\n---\nYOUR LAST ANSWER LEFT THE LABELS EMPTY ON: " + named + ".\n"
        "An act with no labels is rendered with placeholder copy written for "
        "the metaphor rather than for this concept, which puts words about a "
        "completely different subject on screen. Return the whole plan again "
        "with every act's \"labels\" filled in for THIS concept, using the slot "
        "names listed for the metaphor that act uses."
    )


def _positive_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def parse_scene_plan(raw: str, acts: int = 0, concept: str = "") -> dict[str, Any]:
    """
    Validates a multi-act plan into something normalise_spec can take.

    Returns {} rather than a partial plan when there are fewer than two usable
    acts: one act is what this was built to replace, so falling back to the
    single-metaphor path is the honest outcome.
    """
    parsed = parse_scene_response(raw)
    if not isinstance(parsed, dict):
        return {}

    limit = max(2, min(8, int(acts))) if acts else 8
    planned: list[dict[str, Any]] = []
    for entry in (parsed.get("acts") or [])[:limit]:
        if not isinstance(entry, dict):
            continue
        thesis = str(entry.get("thesis") or "").strip()
        if not thesis:
            continue
        template = str(entry.get("template") or entry.get("metaphor_type")
                       or "").strip().lower()
        try:
            axis_max = max(1, int(entry.get("axis_max") or 365))
        except (TypeError, ValueError):
            axis_max = 365
        try:
            end_value = min(1.0, max(0.0, float(entry.get("end_value") or 0.0)))
        except (TypeError, ValueError):
            end_value = 0.0
        planned.append({
            "template": template if template in SCENE_TEMPLATE_KEYS else "",
            "title": str(entry.get("title") or "").strip()[:40],
            "subtitle": str(entry.get("subtitle") or "").strip()[:90],
            "labels": entry.get("labels") if isinstance(entry.get("labels"), dict) else {},
            "planned": True,
            "direction": str(entry.get("direction") or "fill").strip().lower(),
            "axis_max": axis_max,
            "axis_suffix": str(entry.get("axis_suffix") or "d").strip()[:3],
            "end_value": end_value,
            "end_multiple": _positive_float(entry.get("end_multiple")),
            "thesis": thesis,
        })

    if len(planned) < 2:
        return {}

    # A repeated metaphor defeats the point of acts. The replacement is drawn
    # from the concept's own domain pool first, so a finance video that asked
    # for the same vessel twice gets another finance shape rather than
    # whatever happens to sit next in the catalogue.
    preferred = list(domain_guide(concept)["metaphors"]) if concept else []
    spare = preferred + [key for key in SCENE_TEMPLATE_KEYS
                         if key not in ("auto", "custom") and key not in preferred]
    seen: set[str] = set()
    for act in planned:
        if not act["template"] or act["template"] in seen:
            act["template"] = next(
                (key for key in spare if key not in seen), act["template"] or spare[0])
        seen.add(act["template"])

    # Which acts came back with nothing stamped on their geometry. An act with
    # no labels at all is the one that used to render the template's own
    # placeholder copy, so this is the signal generate_scene_plan retries on.
    unlabelled = [act["title"] or act["template"] for act in planned
                  if LABEL_SLOTS_FOR.get(act["template"])
                  and len(missing_label_slots(act["template"], act["labels"]))
                  == len(LABEL_SLOTS_FOR[act["template"]])]

    return {
        "acts": planned,
        "unlabelled": unlabelled,
        "payoff": str(parsed.get("payoff") or "").strip()[:60],
        "cta": str(parsed.get("cta") or "").strip()[:40],
        "thesis": " ".join(act["thesis"] for act in planned),
        "publish": parsed.get("publish") if isinstance(parsed.get("publish"), dict) else {},
        "source": "gemini-plan",
    }


def generate_scene_plan(concept: str,
                        progress: ProgressFn | None = None,
                        acts: int = 0,
                        duration: float = 0.0) -> dict[str, Any]:
    """
    A multi-act plan for a concept.

    `acts` defaults to what the runtime can hold without any one metaphor
    sitting still -- six at the usual length, where it used to be three at
    twenty-two seconds each.

    Raises GeminiError when no model returns a usable plan; the caller falls
    back to the single-metaphor path, which still renders a video.
    """
    count = max(2, min(8, int(acts))) if acts else default_act_count(duration)
    client = get_client()
    prompt = build_scene_plan_prompt(concept, acts=count, duration=duration)
    last: Exception | None = None

    # Two rounds of the candidates, not one.
    #
    # The caller's fallback is a single metaphor held for the whole video --
    # much worse than any plan -- so another round of asking is cheap by
    # comparison. A delivered render lost its six acts to a failure that the
    # very next attempt on the same concept did not reproduce.
    for model in MODEL_CANDIDATES * 2:
        try:
            if progress:
                progress(f"Asking {model} for a {count}-act plan...")
            response = generate_with_retry(client, model, prompt, progress=progress)
            plan = parse_scene_plan(getattr(response, "text", "") or "",
                                    acts=count, concept=concept)

            # An act with nothing stamped on its geometry used to fall through
            # to the template's placeholder copy, which is how a video about
            # fund fees ended up captioned WHO YOU ARE / WHO YOU THINK. One
            # retry, naming the acts, is cheap and usually enough.
            if plan and plan.get("unlabelled"):
                if progress:
                    progress(f"{len(plan['unlabelled'])} acts came back "
                             "unlabelled; asking again...")
                retry = generate_with_retry(
                    client, model,
                    prompt + relabel_note(plan["unlabelled"]), progress=progress)
                second = parse_scene_plan(getattr(retry, "text", "") or "",
                                          acts=count, concept=concept)
                # Keep whichever plan labelled more of its geometry.
                if second and len(second.get("unlabelled") or []) < len(plan["unlabelled"]):
                    plan = second

            if plan:
                plan["concept"] = str(concept).strip()
                plan["model"] = model
                return plan
        except Exception as exc:                              # noqa: BLE001
            last = exc

    raise GeminiError(
        f"No model returned a usable {count}-act plan for {concept!r}"
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
MEASURED: {words} words, {sentences} sentences, {wpm:.0f} words per minute.

Those three figures are counted, not estimated. If you cite a word count, a
sentence count or a pace anywhere in your answer, use exactly these numbers.
Do not produce your own -- a card that says "147 words" in one note and "156
words" in another is read, correctly, as a card that measured neither.

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

    import re as _re

    body = str(script).strip()
    seconds = float(entry.get("duration") or 0.0)
    word_count = len([w for w in body.split() if w])
    sentence_count = len([s for s in _re.split(r"(?<=[.!?])\s+", body)
                          if len(s.split()) >= 3]) or 1

    prompt = VIRAL_PROMPT.format(
        script=body,
        duration=seconds,
        words=word_count,
        sentences=sentence_count,
        wpm=(word_count / seconds * 60.0) if seconds > 0 else 0.0,
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
