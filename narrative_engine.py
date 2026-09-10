# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""
Narrative Studio -- episodic storytelling from a one-line premise.

The pipeline is four stages, and each one is a pure function of the stage
before it, so any stage can be re-run without redoing the others:

  1. script      Gemini writes narration in short beats, and in the same pass
                 extracts the protagonist, the environments and the recurring
                 objects.
  2. storyboard  the beats become timed segments, each carrying a camera
                 framing and the entity tags it belongs to.
  3. assets      narration audio, and one still per segment.
  4. assembly    stills become Ken Burns moves, cut to the voice, dissolved
                 together, subtitled and mixed under an ambient bed.

The entity block in stage 1 is the reason the thing holds together. An image
model asked for "the man" twelve times draws twelve different men; asked for
the same forty-word description twelve times, it draws something close enough
to read as one person. That description is written once and pasted into every
prompt.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Callable, Sequence

ProgressFn = Callable[[str], None]

# ---------------------------------------------------------------------------
# Channel style
# ---------------------------------------------------------------------------

VISUAL_AESTHETICS: dict[str, dict[str, str]] = {
    "vector_comic": {
        "label": "Desaturated Vector Comic",
        "prompt": "desaturated vector comic illustration, limited muted palette, "
                  "flat graphic shapes with heavy ink outlines, halftone grain, "
                  "restrained colour, printed-page texture",
        "palette": "muted slate, bone, oxblood",
    },
    "lofi_anime": {
        "label": "Cinematic Lofi Anime",
        "prompt": "cinematic lofi anime still, soft cel shading, warm hazy light, "
                  "gentle film grain, shallow depth of field, nostalgic and quiet, "
                  "anime background art in the style of a quiet slice-of-life film",
        "palette": "dusty amber, teal shadow, cream",
    },
    "nordic_noir": {
        "label": "Nordic Noir",
        "prompt": "nordic noir cinematography, cold desaturated palette, heavy "
                  "shadow, practical light sources only, sparse composition, rain "
                  "and window reflections, quiet dread",
        "palette": "steel blue, charcoal, sodium orange",
    },
    "flat_minimal": {
        "label": "Flat Graphic Minimalist",
        "prompt": "flat graphic minimalist illustration, large simple shapes, two "
                  "or three flat colours plus paper white, no gradients, generous "
                  "negative space, editorial poster feel",
        "palette": "ink black, warm paper, single accent",
    },
}

NARRATIVE_TONES: dict[str, dict[str, str]] = {
    "reflection": {
        "label": "Quiet Self-Reflection",
        "prompt": "Calm, first person, past tense looking back. Short plain "
                  "sentences. No hype, no exclamation. The feeling arrives from "
                  "the specifics, never from adjectives.",
    },
    "stoic": {
        "label": "Stoic Motivation",
        "prompt": "Second person, measured and unsentimental. States hard things "
                  "plainly and does not console. Short declaratives. No slogans.",
    },
    "suspense": {
        "label": "Suspense Investigation",
        "prompt": "Third person, present tense, withholding. Each beat reveals one "
                  "fact and raises one question. Cold and procedural.",
    },
}

DURATION_FORMATS: dict[str, dict[str, Any]] = {
    "short": {
        "label": "60s Short (9:16)",
        "seconds": 60,
        "aspect": (1080, 1920),
        "segments": (10, 16),
        "note": "One idea, one turn, one landing.",
    },
    "longform": {
        "label": "3-5 min Long-form (9:16)",
        "seconds": 240,
        "aspect": (1080, 1920),
        "segments": (34, 52),
        "note": "Room for a full arc: setup, drift, turn, consequence, landing.",
    },
    "longform_wide": {
        "label": "3-5 min Long-form (16:9)",
        "seconds": 240,
        "aspect": (1920, 1080),
        "segments": (34, 52),
        "note": "Same arc, framed for a desktop feed.",
    },
}

DEFAULT_AESTHETIC = "vector_comic"
DEFAULT_TONE = "reflection"
DEFAULT_FORMAT = "short"

# Narration pace. Measured against Gemini TTS at its default rate, which is a
# little slower than edge-tts with the +12% boost this project uses elsewhere.
WORDS_PER_SECOND = 2.6
MIN_SEGMENT_SECONDS = 1.6
MAX_SEGMENT_SECONDS = 9.0


# ---------------------------------------------------------------------------
# Stage 1 -- script and entities
# ---------------------------------------------------------------------------

def build_narrative_prompt(topic: str, aesthetic: str = DEFAULT_AESTHETIC,
                           tone: str = DEFAULT_TONE,
                           duration: str = DEFAULT_FORMAT) -> str:
    """
    Asks for the script and the world in one call.

    One call rather than two because the entities have to describe the people
    and places this particular script uses. Extracted afterwards from finished
    prose, they come back generic -- "a man", "a room" -- which is exactly what
    makes a sequence of stills look like twelve unrelated pictures.
    """
    look = VISUAL_AESTHETICS.get(aesthetic, VISUAL_AESTHETICS[DEFAULT_AESTHETIC])
    voice = NARRATIVE_TONES.get(tone, NARRATIVE_TONES[DEFAULT_TONE])
    fmt = DURATION_FORMATS.get(duration, DURATION_FORMATS[DEFAULT_FORMAT])
    low, high = fmt["segments"]
    seconds = int(fmt["seconds"])
    words = int(seconds * WORDS_PER_SECOND)

    return f"""You write narration for an episodic storytelling channel. Return ONE JSON object and nothing else.

PREMISE: {topic.strip()}

TONE: {voice["prompt"]}

LENGTH: about {seconds} seconds of narration, which is roughly {words} words read aloud. Break it into {low}-{high} beats. A beat is one or two short sentences -- what a narrator says over a single shot.

Fields:

"title": 4-8 words, the episode title.

"logline": one sentence describing the episode.

"protagonist": {{
    "name": a first name or a role, used as the reference tag,
    "age": approximate age,
    "appearance": 25-40 words. Build, hair, face, posture. Concrete and repeatable -- this exact text is pasted into every image prompt, so it must describe someone specific enough to be recognised twice,
    "clothing": 15-25 words, one consistent outfit for the whole episode,
    "demeanour": 10-20 words, how they carry themselves
}}

"environments": 2-5 entries, each {{
    "tag": short lowercase id, e.g. "suburban_home",
    "name": "Suburban Home Interior",
    "description": 25-40 words of concrete visual detail -- light, materials, clutter, time of day
}}

"objects": 1-4 recurring props, each {{
    "tag": short lowercase id,
    "name": "Old Sedan",
    "description": 15-30 words, specific enough to be drawn the same way twice
}}

"beats": an array of {low}-{high} entries, each {{
    "line": the narration for this beat, spoken exactly as written,
    "environment": one of the environment tags,
    "objects": array of object tags visible in this shot, may be empty,
    "framing": a camera instruction naming the shot size and the subject's action, e.g. "Eye-level medium close-up of Daniel looking out of a rain-streaked window, neutral domestic backdrop behind him",
    "beat_type": one of "setup", "drift", "turn", "consequence", "landing"
}}

Rules. Every beat's "line" is spoken narration only -- no stage directions, no timestamps, no speaker labels. The beats read as one continuous piece when concatenated. Vary the shot sizes; do not frame every beat as a medium close-up. The visual style is "{look["prompt"]}" -- do not restate it in the framings, it is applied separately."""


def parse_narrative(raw: str) -> dict[str, Any]:
    """
    Pulls the JSON object out of a model response.

    Forgiving on purpose: models fence their JSON, prepend a sentence, or
    return it with a trailing comma, and a strict parse throws away an answer
    that is 99% usable.
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
    # Trailing commas before a closing brace or bracket are the single most
    # common way a long generated object fails to parse.
    candidates += [re.sub(r",(\s*[}\]])", r"\1", c) for c in list(candidates)]

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _clean_line(value: Any, limit: int = 400) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    # Models occasionally leak a speaker label or a stage direction into a beat.
    text = re.sub(r"^(NARRATOR|VO|VOICEOVER)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\[[^\]]{0,60}\]\s*", "", text)
    return text[:limit]


def _tag(value: Any, fallback: str = "scene") -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    return (slug or fallback)[:32]


def extract_entities(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Normalises the protagonist, environments and objects into reference cards.

    Every field is defaulted and every list is bounded, because this is model
    output feeding directly into image prompts: a missing "appearance" must
    degrade to a plainer picture, never to a crash forty seconds into a render.
    """
    protagonist_raw = raw.get("protagonist")
    if not isinstance(protagonist_raw, dict):
        protagonist_raw = {}

    protagonist = {
        "name": _clean_line(protagonist_raw.get("name") or "The narrator", 40),
        "age": _clean_line(protagonist_raw.get("age") or "late twenties", 30),
        "appearance": _clean_line(protagonist_raw.get("appearance")
                                  or "average build, short dark hair, tired eyes, quiet posture", 320),
        "clothing": _clean_line(protagonist_raw.get("clothing")
                                or "plain dark hoodie and worn jeans", 200),
        "demeanour": _clean_line(protagonist_raw.get("demeanour")
                                 or "watchful, holds still, speaks rarely", 200),
    }

    environments: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in (raw.get("environments") or [])[:6]:
        if not isinstance(item, dict):
            continue
        tag = _tag(item.get("tag") or item.get("name"), f"env{len(environments)}")
        if tag in seen:
            continue
        seen.add(tag)
        environments.append({
            "tag": tag,
            "name": _clean_line(item.get("name") or tag.replace("_", " ").title(), 80),
            "description": _clean_line(item.get("description") or "", 320),
        })
    if not environments:
        environments = [{"tag": "primary", "name": "Primary Setting",
                         "description": "a plain interior, natural light, few objects"}]

    objects: list[dict[str, str]] = []
    seen = set()
    for item in (raw.get("objects") or [])[:5]:
        if not isinstance(item, dict):
            continue
        tag = _tag(item.get("tag") or item.get("name"), f"obj{len(objects)}")
        if tag in seen:
            continue
        seen.add(tag)
        objects.append({
            "tag": tag,
            "name": _clean_line(item.get("name") or tag.replace("_", " ").title(), 80),
            "description": _clean_line(item.get("description") or "", 240),
        })

    return {"protagonist": protagonist, "environments": environments, "objects": objects}


def character_reference(entities: dict[str, Any]) -> str:
    """The protagonist as one repeatable sentence, for pasting into prompts."""
    p = entities.get("protagonist") or {}
    parts = [str(p.get("name") or "the narrator"), f"aged {p.get('age')}" if p.get("age") else "",
             str(p.get("appearance") or ""), f"wearing {p.get('clothing')}" if p.get("clothing") else "",
             str(p.get("demeanour") or "")]
    return ", ".join(part for part in parts if part).strip(", ")


# ---------------------------------------------------------------------------
# Stage 2 -- storyboard
# ---------------------------------------------------------------------------

def estimate_seconds(text: str, words_per_second: float = WORDS_PER_SECOND) -> float:
    """
    How long a line takes to read, before the voice is synthesized.

    Word count plus a beat for the punctuation: a line with two full stops in
    it is read slower than a line of the same length without them, and getting
    that wrong is what makes an unmeasured storyboard drift.
    """
    words = len([w for w in re.split(r"\s+", str(text or "").strip()) if w])
    if not words:
        return 0.0
    pauses = len(re.findall(r"[.!?,;:]", str(text)))
    return round(words / max(words_per_second, 0.5) + pauses * 0.16, 2)


def build_storyboard(raw: dict[str, Any], entities: dict[str, Any],
                     duration: str = DEFAULT_FORMAT) -> list[dict[str, Any]]:
    """
    Turns beats into numbered, timed segments with a framing and entity tags.

    Timings here are estimates. They are replaced by the real ones once the
    voice exists -- see `retime_storyboard` -- but the storyboard has to be
    reviewable before anything is synthesized.
    """
    env_tags = {env["tag"] for env in entities["environments"]}
    default_env = entities["environments"][0]["tag"]
    obj_tags = {obj["tag"] for obj in entities["objects"]}

    segments: list[dict[str, Any]] = []
    cursor = 0.0
    for item in (raw.get("beats") or []):
        if isinstance(item, str):
            item = {"line": item}
        if not isinstance(item, dict):
            continue
        line = _clean_line(item.get("line") or item.get("text") or item.get("voiceover"))
        if not line:
            continue

        seconds = min(MAX_SEGMENT_SECONDS, max(MIN_SEGMENT_SECONDS, estimate_seconds(line)))
        env = _tag(item.get("environment"), default_env)
        if env not in env_tags:
            env = default_env
        objects = [_tag(o) for o in (item.get("objects") or []) if isinstance(o, str)]
        objects = [o for o in objects if o in obj_tags][:3]

        segments.append({
            "index": len(segments) + 1,
            "line": line,
            "words": len(line.split()),
            "start": round(cursor, 2),
            "duration": seconds,
            "end": round(cursor + seconds, 2),
            "framing": _clean_line(item.get("framing") or "", 260)
                       or f"Medium shot of {entities['protagonist']['name']}",
            "environment": env,
            "objects": objects,
            "beat_type": _clean_line(item.get("beat_type") or "", 24).lower() or "drift",
            "character": entities["protagonist"]["name"],
        })
        cursor += seconds

    return segments


def retime_storyboard(segments: Sequence[dict[str, Any]],
                      words: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Re-cuts the storyboard against the narration's real word timings.

    Estimated durations are always a little wrong, and the error accumulates:
    by segment thirty a picture can be two seconds off the line being spoken.
    Walking the word list and cutting where each segment's last word actually
    ends puts every cut back on the voice.

    Falls back to proportional scaling when the provider returned no timings,
    which is what Gemini TTS does.
    """
    segments = [dict(s) for s in segments]
    if not segments:
        return []

    usable = [w for w in words if isinstance(w, dict) and "end" in w]
    if not usable:
        return segments

    cursor_word = 0
    cursor_time = 0.0
    for segment in segments:
        take = max(1, int(segment["words"]))
        cursor_word += take
        index = min(cursor_word, len(usable)) - 1
        end = float(usable[index].get("end") or 0.0)
        if end <= cursor_time:
            end = cursor_time + MIN_SEGMENT_SECONDS
        segment["start"] = round(cursor_time, 2)
        segment["duration"] = round(max(MIN_SEGMENT_SECONDS, end - cursor_time), 2)
        segment["end"] = round(segment["start"] + segment["duration"], 2)
        cursor_time = segment["end"]

    return segments


def cover_narration(segments: Sequence[dict[str, Any]], speech: float,
                    tail: float = 0.35) -> list[dict[str, Any]]:
    """
    Stretches the last segment so the picture outlasts the voice.

    Retiming cuts each segment at the end of its last word, and the storyboard
    word count never matches the provider's word list exactly -- contractions,
    punctuation and numbers all split differently. The residue is small, but it
    lands entirely on the final segment, and because the mux trims to the
    shorter stream it clips the closing words.

    Measured on a 12-segment episode before this existed: narration 46.27s,
    picture 45.50s, and the last 1.2 seconds of the file were three times
    louder than the middle -- the narrator was still talking when it ended.
    """
    segments = [dict(s) for s in segments]
    if not segments or speech <= 0:
        return segments

    runtime = storyboard_runtime(segments)
    shortfall = speech + tail - runtime
    if shortfall <= 0:
        return segments

    last = segments[-1]
    last["duration"] = round(float(last["duration"]) + shortfall, 2)
    last["end"] = round(float(last["start"]) + last["duration"], 2)
    return segments


def storyboard_runtime(segments: Sequence[dict[str, Any]]) -> float:
    return round(sum(float(s.get("duration") or 0.0) for s in segments), 2)


def image_prompt(segment: dict[str, Any], entities: dict[str, Any],
                 aesthetic: str = DEFAULT_AESTHETIC,
                 portrait: bool = True) -> str:
    """
    The prompt for one segment's still.

    Order matters: style, then the framing, then the character card, then the
    environment. Image models weight the front of a prompt most heavily, and
    the style has to survive twelve variations of subject matter or the
    episode stops looking like one thing.
    """
    look = VISUAL_AESTHETICS.get(aesthetic, VISUAL_AESTHETICS[DEFAULT_AESTHETIC])
    env = next((e for e in entities["environments"]
                if e["tag"] == segment.get("environment")), None)
    props = [o for o in entities["objects"] if o["tag"] in (segment.get("objects") or [])]

    parts = [
        look["prompt"] + ".",
        f"Palette: {look['palette']}.",
        str(segment.get("framing") or "").rstrip(".") + ".",
        f"The recurring character: {character_reference(entities)}.",
    ]
    if env:
        parts.append(f"Setting -- {env['name']}: {env['description']}.")
    for prop in props:
        parts.append(f"Visible: {prop['name']} -- {prop['description']}.")
    parts.append("Vertical 9:16 composition." if portrait else "Wide 16:9 composition.")
    parts.append("Cinematic still frame. No text, no lettering, no watermark, no logo.")
    return " ".join(part for part in parts if part.strip(" ."))


# ---------------------------------------------------------------------------
# Fallback narrative, for no key and no network
# ---------------------------------------------------------------------------

def fallback_narrative(topic: str, aesthetic: str = DEFAULT_AESTHETIC,
                       tone: str = DEFAULT_TONE,
                       duration: str = DEFAULT_FORMAT) -> dict[str, Any]:
    """
    A complete, renderable episode with no API call at all.

    Deliberately skeletal. It exists so the wizard can be walked end to end
    with the network down, not so it can write for you.
    """
    fmt = DURATION_FORMATS.get(duration, DURATION_FORMATS[DEFAULT_FORMAT])
    beats_wanted = max(8, min(fmt["segments"][0], 14))
    premise = (topic or "a life measured in small rooms").strip().rstrip(".")

    skeleton = [
        ("setup", f"This is about {premise}."),
        ("setup", "It did not look like a beginning at the time."),
        ("drift", "The days were the same shape as each other."),
        ("drift", "Nobody was watching. That turned out to matter."),
        ("drift", "The work was quiet and it was mostly boring."),
        ("turn", "Then one ordinary week, something moved."),
        ("turn", "Not much. Enough to notice."),
        ("consequence", "The same hours started returning something."),
        ("consequence", "It compounded while I was not looking."),
        ("consequence", "The room got bigger. Then it got left behind."),
        ("landing", "None of it arrived on the day I expected."),
        ("landing", "It arrived because the boring weeks did."),
        ("landing", "That is the whole story."),
        ("landing", "It is probably yours too."),
    ][:beats_wanted]

    return {
        "title": " ".join(premise.split()[:6]).upper() or "AN ORDINARY EPISODE",
        "logline": f"An episode about {premise}.",
        "protagonist": {
            "name": "The narrator", "age": "late twenties",
            "appearance": "average build, short dark hair, unshaven, tired eyes, "
                          "shoulders slightly forward, watchful face",
            "clothing": "plain grey hoodie, dark jeans, worn trainers",
            "demeanour": "still, unhurried, watches more than he speaks",
        },
        "environments": [
            {"tag": "home", "name": "Small Suburban Interior",
             "description": "a narrow bedroom, low evening light through thin curtains, "
                            "single lamp, stacked boxes, carpet worn at the door"},
            {"tag": "outside", "name": "Residential Street at Dusk",
             "description": "parked cars, sodium streetlights just coming on, wet tarmac, "
                            "low houses, nobody about"},
        ],
        "objects": [
            {"tag": "desk", "name": "Worn Desk",
             "description": "a scratched pine desk under the window, one lamp, a closed laptop"},
        ],
        "beats": [{"line": line, "environment": "home" if i % 3 else "outside",
                   "objects": ["desk"] if i % 4 == 0 else [],
                   "framing": _FALLBACK_FRAMINGS[i % len(_FALLBACK_FRAMINGS)],
                   "beat_type": kind}
                  for i, (kind, line) in enumerate(skeleton)],
        "source": "fallback",
    }


_FALLBACK_FRAMINGS = (
    "Wide establishing shot of the room, the character small in frame",
    "Eye-level medium close-up of the character, looking off camera",
    "Over-the-shoulder shot toward a window",
    "Low-angle wide shot of the street",
    "Close-up of hands at rest on a desk",
    "Medium shot, character seen from behind, facing away",
)


def normalise_narrative(raw: dict[str, Any] | None, topic: str = "",
                        aesthetic: str = DEFAULT_AESTHETIC,
                        tone: str = DEFAULT_TONE,
                        duration: str = DEFAULT_FORMAT) -> dict[str, Any]:
    """Coerces any response -- or none -- into a complete, renderable episode."""
    raw = dict(raw or {})
    entities = extract_entities(raw)
    segments = build_storyboard(raw, entities, duration)

    if not segments:
        raw = fallback_narrative(topic, aesthetic, tone, duration)
        entities = extract_entities(raw)
        segments = build_storyboard(raw, entities, duration)

    return {
        "topic": topic.strip(),
        "title": _clean_line(raw.get("title") or (topic[:48].upper() or "UNTITLED"), 80),
        "logline": _clean_line(raw.get("logline") or "", 220),
        "aesthetic": aesthetic if aesthetic in VISUAL_AESTHETICS else DEFAULT_AESTHETIC,
        "tone": tone if tone in NARRATIVE_TONES else DEFAULT_TONE,
        "format": duration if duration in DURATION_FORMATS else DEFAULT_FORMAT,
        "entities": entities,
        "segments": segments,
        "script": " ".join(str(s["line"]) for s in segments),
        "runtime": storyboard_runtime(segments),
        "source": str(raw.get("source") or "gemini"),
    }


def generate_narrative(topic: str, aesthetic: str = DEFAULT_AESTHETIC,
                       tone: str = DEFAULT_TONE, duration: str = DEFAULT_FORMAT,
                       progress: ProgressFn | None = None) -> dict[str, Any]:
    """
    Asks Gemini for the script and the world, and normalises what comes back.

    Never raises for a bad response: a model that returns unusable JSON falls
    through to the skeleton episode, because a wizard that dead-ends on step
    one is worse than one that starts you from a plain draft.
    """
    from gemini_engine import MODEL_CANDIDATES, generate_with_retry, get_client

    prompt = build_narrative_prompt(topic, aesthetic, tone, duration)
    last: Exception | None = None

    for model in MODEL_CANDIDATES:
        try:
            if progress:
                progress(f"{model} is writing the script and casting the world...")
            client = get_client()
            response = generate_with_retry(client, model, prompt, progress=None)
            parsed = parse_narrative(getattr(response, "text", "") or "")
            if not parsed.get("beats"):
                raise ValueError(f"{model} returned no beats")
            parsed["source"] = model
            return normalise_narrative(parsed, topic, aesthetic, tone, duration)
        except Exception as exc:
            last = exc
            if progress:
                progress(f"{model} could not write it ({exc}); trying the next model...")
            continue

    if progress:
        progress(f"No model could write the episode ({last}). Using the skeleton draft.")
    return normalise_narrative(fallback_narrative(topic, aesthetic, tone, duration),
                               topic, aesthetic, tone, duration)


# ---------------------------------------------------------------------------
# Stage 3 -- visuals
#
# Three providers, tried in order and each one a real fallback rather than an
# error path:
#
#   gemini      the only one that can draw a named character the same way
#               twice, which is the whole reason the entity block exists.
#               Needs billing enabled -- image generation is not on the free
#               tier, and a free-tier key returns 429 with "limit: 0".
#   pexels      real photography, CC0, no character consistency. Good for
#               environments and objects, wrong for the protagonist.
#   procedural  drawn here from the aesthetic's palette. Always available,
#               never blocks a render, and honest about being a backdrop.
# ---------------------------------------------------------------------------

IMAGE_PROVIDERS: tuple[str, ...] = ("gemini", "pexels", "procedural")

GEMINI_IMAGE_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-image",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image",
)

# Aesthetic -> (background, mid, accent) as RGB, for the procedural backdrop.
_AESTHETIC_PALETTE: dict[str, tuple[tuple[int, int, int], ...]] = {
    "vector_comic": ((28, 30, 34), (78, 74, 70), (176, 92, 74)),
    "lofi_anime": ((32, 30, 38), (92, 78, 96), (214, 168, 116)),
    "nordic_noir": ((14, 18, 24), (44, 62, 78), (196, 122, 54)),
    "flat_minimal": ((242, 238, 230), (30, 30, 32), (198, 84, 62)),
}


# Colour grades, applied to stock photography so it sits inside the chosen
# aesthetic. Only the Gemini provider can draw *in* a style; a stock photo
# arrives as whatever it was shot as, and six bright suburban interiors under a
# Nordic Noir premise look like a slideshow of estate agent listings. The grade
# will not turn a photograph into an illustration, but it does put every shot
# in one world.
#
# (saturation, contrast, brightness, shadow tint RGB, highlight tint RGB, grain)
_GRADES: dict[str, tuple[float, float, float, tuple[int, int, int], tuple[int, int, int], float]] = {
    "vector_comic": (0.42, 1.18, 0.94, (28, 30, 38), (232, 226, 214), 7.0),
    "lofi_anime": (0.78, 0.94, 1.06, (38, 32, 52), (255, 236, 206), 5.0),
    "nordic_noir": (0.34, 1.30, 0.82, (10, 20, 34), (236, 214, 178), 6.0),
    "flat_minimal": (0.22, 1.55, 1.04, (18, 18, 20), (250, 246, 238), 2.0),
}


def apply_grade(image: Any, aesthetic: str = DEFAULT_AESTHETIC) -> Any:
    """
    Pushes a photograph toward the episode's palette.

    A split tone rather than a global tint: shadows toward one colour and
    highlights toward another is what film grading actually does, and it is
    what stops the result looking like a photo with a filter on it.
    """
    import numpy as _np
    from PIL import Image as _Image
    from PIL import ImageEnhance

    saturation, contrast, brightness, shadow, highlight, grain = _GRADES.get(
        aesthetic, _GRADES[DEFAULT_AESTHETIC])

    graded = image.convert("RGB")
    graded = ImageEnhance.Color(graded).enhance(saturation)
    graded = ImageEnhance.Contrast(graded).enhance(contrast)
    graded = ImageEnhance.Brightness(graded).enhance(brightness)

    arr = _np.asarray(graded, dtype=_np.float32) / 255.0
    luma = arr.mean(axis=2, keepdims=True)
    shadow_rgb = _np.asarray(shadow, dtype=_np.float32) / 255.0
    highlight_rgb = _np.asarray(highlight, dtype=_np.float32) / 255.0
    # Weight each tint by how dark or bright the pixel is, then blend gently.
    tinted = shadow_rgb * (1.0 - luma) + highlight_rgb * luma
    arr = arr * 0.74 + tinted * 0.26

    if grain > 0:
        rng = _np.random.default_rng(11)
        arr += rng.normal(0.0, grain / 255.0, arr.shape[:2] + (1,))

    return _Image.fromarray(_np.clip(arr * 255.0, 0, 255).astype("uint8"))


class ImageUnavailable(RuntimeError):
    """No provider could supply a still for this segment."""


def _still_path(directory: str, index: int, suffix: str = "png") -> str:
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"still_{index:03d}_{int(time.time() * 1000) % 100000}.{suffix}")


def generate_image_gemini(prompt: str, output_path: str,
                          progress: ProgressFn | None = None) -> str:
    """
    Asks a Gemini image model for one still.

    Raises rather than returning a placeholder: the caller's fallback chain is
    what decides the substitute, and swallowing the reason here would hide a
    billing problem behind a stock photo.
    """
    from gemini_engine import get_client

    client = get_client()
    last: Exception | None = None
    for model in GEMINI_IMAGE_MODELS:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            for candidate in (response.candidates or []):
                content = getattr(candidate, "content", None)
                for part in (getattr(content, "parts", None) or []):
                    blob = getattr(part, "inline_data", None)
                    data = getattr(blob, "data", None) if blob else None
                    if data:
                        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                        with open(output_path, "wb") as handle:
                            handle.write(data)
                        return output_path
            raise ValueError(f"{model} returned no image data")
        except Exception as exc:
            last = exc
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                # Out of quota on this model is out of quota on all of them.
                raise ImageUnavailable(
                    "Gemini image generation is not enabled for this key. Image models "
                    "are not on the free tier -- enable billing on the Google AI Studio "
                    "project, or use the Pexels/procedural providers."
                ) from exc
            continue
    raise ImageUnavailable(f"No Gemini image model produced a still: {last}")


def build_photo_pool(segments: Sequence[dict[str, Any]], entities: dict[str, Any],
                     progress: ProgressFn | None = None) -> dict[str, list[Any]]:
    """
    Fetches a set of distinct photos per environment, up front.

    Searching per segment returns the same photograph every time, because the
    query is the environment and the environment does not change between
    segments -- six shots of the basement come back as six copies of one
    basement. Pulling a set per environment and handing out a different one to
    each segment is the difference between a sequence and a held frame.
    """
    from demo_data import fetch_photo_set

    wanted: dict[str, int] = {}
    for segment in segments:
        tag = str(segment.get("environment") or "")
        wanted[tag] = wanted.get(tag, 0) + 1

    pool: dict[str, list[Any]] = {}
    for env in entities["environments"]:
        tag = env["tag"]
        count = wanted.get(tag, 0)
        if count <= 0:
            continue
        if progress:
            progress(f"Sourcing {min(count, 6)} photos for {env['name']}...")
        try:
            pool[tag] = [image for image, _credit in fetch_photo_set(env["name"], min(count, 6))]
        except Exception:
            pool[tag] = []
    return pool


def generate_image_pexels(segment: dict[str, Any], entities: dict[str, Any],
                          output_path: str, size: tuple[int, int],
                          pool: dict[str, list[Any]] | None = None,
                          occurrence: int = 0,
                          aesthetic: str = DEFAULT_AESTHETIC) -> str:
    """
    Real photography for the segment's environment.

    Searched on the environment rather than the framing: "Eye-level medium
    close-up of Daniel" matches nothing on a stock library, while "suburban
    home interior dusk" matches plenty. `pool` supplies the pre-fetched set so
    consecutive segments in one environment get different frames.
    """
    from demo_data import PhotoLookupError, fetch_photo
    from video_engine import fit_image_to_aspect

    env = next((e for e in entities["environments"]
                if e["tag"] == segment.get("environment")), None)
    query = (env["name"] if env else "quiet interior").strip()
    tag = str(segment.get("environment") or "")

    image = None
    options = (pool or {}).get(tag) or []
    if options:
        image = options[occurrence % len(options)]
    else:
        try:
            image, _credit = fetch_photo(query, provider="auto")
        except (PhotoLookupError, Exception) as exc:
            raise ImageUnavailable(f"No stock photo for {query!r}: {exc}") from exc

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    framed = fit_image_to_aspect(image, size, "crop_fill")
    apply_grade(framed, aesthetic).save(output_path)
    return output_path


def generate_image_procedural(segment: dict[str, Any], entities: dict[str, Any],
                              output_path: str, size: tuple[int, int],
                              aesthetic: str = DEFAULT_AESTHETIC) -> str:
    """
    A styled backdrop drawn from the aesthetic's palette.

    Not a placeholder square: a graded ground, a horizon, a light source and a
    silhouette block, varied by the segment index so consecutive shots differ.
    It reads as art direction rather than as a failure.
    """
    from PIL import Image, ImageDraw, ImageFilter

    width, height = size
    base, mid, accent = _AESTHETIC_PALETTE.get(aesthetic,
                                               _AESTHETIC_PALETTE[DEFAULT_AESTHETIC])
    index = int(segment.get("index") or 1)

    canvas = Image.new("RGB", (width, height), base)
    draw = ImageDraw.Draw(canvas)

    # A vertical grade, warmer toward the light source.
    for y in range(0, height, 4):
        share = y / max(height - 1, 1)
        blend = tuple(int(base[c] + (mid[c] - base[c]) * (share ** 1.4)) for c in range(3))
        draw.rectangle((0, y, width, y + 4), fill=blend)

    # A light: low and off to one side, alternating sides down the episode.
    side = 0.26 if index % 2 else 0.74
    glow = Image.new("RGB", (width, height), (0, 0, 0))
    ImageDraw.Draw(glow).ellipse(
        (width * side - width * 0.42, height * 0.20 - width * 0.42,
         width * side + width * 0.42, height * 0.20 + width * 0.42),
        fill=tuple(int(c * 0.55) for c in accent),
    )
    canvas = Image.blend(canvas, glow.filter(ImageFilter.GaussianBlur(width // 12)), 0.55)
    draw = ImageDraw.Draw(canvas)

    # A horizon and a couple of blocks: enough structure for a pan to bite on.
    horizon = int(height * (0.60 + 0.06 * math.sin(index)))
    draw.rectangle((0, horizon, width, height), fill=tuple(int(c * 0.55) for c in base))
    draw.line((0, horizon, width, horizon), fill=mid, width=max(2, height // 540))

    for block in range(3):
        seed = (index * 7919 + block * 104729) % 1000 / 1000.0
        block_w = width * (0.12 + 0.22 * seed)
        left = width * (-0.1 + 1.15 * ((seed * 3.7) % 1.0))
        top = horizon - height * (0.05 + 0.18 * ((seed * 5.3) % 1.0))
        draw.rectangle((left, top, left + block_w, horizon),
                       fill=tuple(int(c * 0.72) for c in base))

    # Grain, and not only because three of the four aesthetics ask for it: a
    # Ken Burns move over a smooth gradient is almost invisible. Measured on a
    # hard-edged pattern a 16% zoom moves 26 mean-absolute; over the ungrained
    # gradient it moved 1.4, which on screen is a still photograph.
    import numpy as _np

    noise = _np.random.default_rng(index * 7919).normal(0.0, 7.0, (height, width, 1))
    texture = _np.clip(_np.asarray(canvas, dtype=_np.float32) + noise, 0, 255)
    canvas = Image.fromarray(texture.astype("uint8"))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    canvas.save(output_path)
    return output_path


def segment_still(segment: dict[str, Any], entities: dict[str, Any],
                  directory: str, size: tuple[int, int],
                  aesthetic: str = DEFAULT_AESTHETIC,
                  providers: Sequence[str] = IMAGE_PROVIDERS,
                  progress: ProgressFn | None = None,
                  pool: dict[str, list[Any]] | None = None,
                  occurrence: int = 0) -> dict[str, Any]:
    """
    One still for one segment, from the first provider that delivers.

    Returns {"path", "provider", "notes"} -- `notes` carries why the earlier
    providers were skipped, so the UI can say "Gemini needs billing" once
    instead of failing silently to stock.
    """
    path = _still_path(directory, int(segment.get("index") or 0))
    prompt = image_prompt(segment, entities, aesthetic,
                          portrait=size[1] >= size[0])
    notes: list[str] = []

    for provider in providers:
        try:
            if provider == "gemini":
                generate_image_gemini(prompt, path, progress)
            elif provider == "pexels":
                generate_image_pexels(segment, entities, path, size, pool,
                                      occurrence, aesthetic)
            elif provider == "procedural":
                generate_image_procedural(segment, entities, path, size, aesthetic)
            else:
                continue
            return {"path": path, "provider": provider, "notes": notes, "prompt": prompt}
        except Exception as exc:
            notes.append(f"{provider}: {exc}")
            continue

    raise ImageUnavailable("; ".join(notes) or "no providers configured")


# ---------------------------------------------------------------------------
# Stage 4 -- Ken Burns, assembly, subtitles
# ---------------------------------------------------------------------------

KEN_BURNS_ZOOM = 0.16           # how far in or out over a whole segment
_MOVES = ("in_center", "out_center", "in_left", "in_right", "pan_left", "pan_right")


def ken_burns_clip(still_path: str, output_path: str, duration: float,
                   size: tuple[int, int], fps: int = 30, move: str = "in_center",
                   ffmpeg: str | None = None) -> str:
    """
    Turns a still into a moving shot.

    zoompan works on the *input* frame, so the still is scaled up first: zoom
    on a source at output resolution resamples from fewer and fewer pixels and
    the shot goes soft exactly as it gets closest. Scaling to 2x first means
    every zoom level still has real pixels behind it.
    """
    import subprocess

    import imageio_ffmpeg

    ffmpeg = ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()
    width, height = size
    frames = max(2, int(round(duration * fps)))
    big_w, big_h = width * 2, height * 2

    zoom_in = not move.startswith("out")
    # zoompan's `zoom` is evaluated per output frame; stepping it by a constant
    # gives a linear move, which is what reads as deliberate rather than eased.
    step = KEN_BURNS_ZOOM / frames
    if zoom_in:
        zoom_expr = f"min(zoom+{step:.6f},{1 + KEN_BURNS_ZOOM:.4f})"
        start_zoom = 1.0
    else:
        zoom_expr = f"max({1 + KEN_BURNS_ZOOM:.4f}-{step:.6f}*on,1.0)"
        start_zoom = 1 + KEN_BURNS_ZOOM

    centre_x = "iw/2-(iw/zoom/2)"
    centre_y = "ih/2-(ih/zoom/2)"
    if move.endswith("left"):
        x_expr = f"(iw-iw/zoom)*(on/{frames})" if "pan" in move else "0"
    elif move.endswith("right"):
        x_expr = f"(iw-iw/zoom)*(1-on/{frames})" if "pan" in move else "iw-iw/zoom"
    else:
        x_expr = centre_x

    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-loop", "1", "-i", os.path.abspath(still_path),
        "-vf", (f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase,"
                f"crop={big_w}:{big_h},"
                f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{centre_y}':"
                f"d={frames}:s={width}x{height}:fps={fps}"),
        "-frames:v", str(frames), "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", os.path.abspath(output_path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"Ken Burns pass failed: {(proc.stderr or '')[:300]}")
    _ = start_zoom
    return output_path


def concat_segments(clips: Sequence[str], output_path: str,
                    durations: Sequence[float], transition: str = "dissolve",
                    dissolve: float = 0.45, fps: int = 30,
                    ffmpeg: str | None = None) -> str:
    """
    Stitches the segment clips into one picture track.

    Hard cuts go through the concat demuxer: a stream copy, near instant.

    Dissolves need an xfade chain, and the arithmetic is the part worth
    stating, because getting it wrong silently slides the picture off the
    voice. xfade's output runs `offset + len(second input)`, so with every
    clip but the last padded by `dissolve`:

        offset[k] = sum(durations[:k])          -- the plain cumulative start
        length after step k = sum(durations[:k+1]) + dissolve
        length after the final, unpadded step   = sum(durations)

    Each transition therefore consumes exactly the padding that was added for
    it, the visible content of segment k lasts its full duration before any
    blending begins, and the finished track is sum(durations) long.

    Subtracting `dissolve` from the offset as well as padding the clip double
    counts it and the track comes out one dissolve short -- 7.38s against a
    7.80s voice on the four-segment case in the tests.
    """
    import subprocess
    import tempfile as _tempfile

    import imageio_ffmpeg

    ffmpeg = ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()
    clips = [os.path.abspath(c) for c in clips]
    if not clips:
        raise ValueError("nothing to concatenate")
    if len(clips) == 1:
        import shutil
        shutil.copyfile(clips[0], output_path)
        return output_path

    if transition != "dissolve" or dissolve <= 0.01:
        listing = os.path.join(_tempfile.gettempdir(), f"rf_concat_{int(time.time() * 1000)}.txt")
        with open(listing, "w", encoding="utf-8") as handle:
            for clip in clips:
                handle.write(f"file '{clip}'\n")
        command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                   "-f", "concat", "-safe", "0", "-i", listing,
                   "-c", "copy", os.path.abspath(output_path)]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=900)
        try:
            os.remove(listing)
        except OSError:
            pass
        if proc.returncode != 0:
            raise RuntimeError(f"concat failed: {(proc.stderr or '')[:300]}")
        return output_path

    inputs: list[str] = []
    for clip in clips:
        inputs += ["-i", clip]

    steps: list[str] = []
    label = "0:v"
    offset = 0.0
    for index in range(1, len(clips)):
        offset += float(durations[index - 1])
        out = f"x{index}"
        steps.append(f"[{label}][{index}:v]xfade=transition=fade:"
                     f"duration={dissolve:.3f}:offset={offset:.3f}[{out}]")
        label = out

    command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *inputs,
               "-filter_complex", ";".join(steps), "-map", f"[{label}]",
               "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "20", "-pix_fmt", "yuv420p", os.path.abspath(output_path)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"dissolve chain failed: {(proc.stderr or '')[:400]}")
    return output_path


def narrative_subtitles(segments: Sequence[dict[str, Any]], output_path: str,
                        size: tuple[int, int]) -> str:
    """
    One clean subtitle cue per segment, timed to that segment.

    Deliberately not the kinetic word-by-word style the Commentary Machine
    burns: this is narration over a picture, and a caption that jumps per word
    fights the read.
    """
    from video_engine import _ass_escape, _ass_time, ass_font_name

    width, height = size
    font_size = max(30, int(width * 0.042))
    margin_v = int(height * 0.10)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Narrative,{ass_font_name()},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,"
        f"&H64000000,0,0,0,0,100,100,0.6,0,1,{max(2, font_size // 14)},0,2,"
        f"{int(width * 0.10)},{int(width * 0.10)},{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = []
    for segment in segments:
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start + 1.0)
        text = _ass_escape(str(segment.get("line") or ""))
        if not text:
            continue
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Narrative,,0,0,0,,"
                     f"{{\\fad(140,140)}}{text}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(header + "\n".join(lines) + "\n")
    return output_path


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------

AMBIENT_VOLUME = 0.085          # under a narrator, not beside them
DISSOLVE_SECONDS = 0.45


def fit_storyboard_to(segments: Sequence[dict[str, Any]], total: float) -> list[dict[str, Any]]:
    """
    Scales an estimated storyboard onto a known runtime.

    The fallback when the voice provider returns no word timings, which is what
    Gemini TTS does. Without it a board estimated at 60s against a voice that
    actually reads in 52 leaves eight seconds of picture over silence.
    """
    segments = [dict(s) for s in segments]
    estimated = storyboard_runtime(segments)
    if estimated <= 0 or total <= 0:
        return segments

    scale = total / estimated
    cursor = 0.0
    for segment in segments:
        segment["duration"] = round(max(MIN_SEGMENT_SECONDS,
                                        float(segment["duration"]) * scale), 2)
        segment["start"] = round(cursor, 2)
        segment["end"] = round(cursor + segment["duration"], 2)
        cursor = segment["end"]
    return segments


def _move_for(segment: dict[str, Any]) -> str:
    """
    Which Ken Burns move this segment gets.

    Keyed off the beat type and the index rather than chosen at random: a
    "turn" pushing in and a "landing" pulling out is grammar, and an episode
    where consecutive shots always move the same way reads as a slideshow.
    """
    beat = str(segment.get("beat_type") or "drift")
    index = int(segment.get("index") or 0)
    if beat == "turn":
        return "in_center"
    if beat == "landing":
        return "out_center"
    if beat == "setup":
        return "pan_left" if index % 2 else "pan_right"
    return _MOVES[index % len(_MOVES)]


def produce_episode(
    episode: dict[str, Any],
    output_path: str,
    workspace: str,
    fps: int = 30,
    voice_provider: str = "gemini",
    voice: str = "Charon",
    transition: str = "dissolve",
    subtitles: bool = True,
    ambient: bool = True,
    ambient_volume: float = AMBIENT_VOLUME,
    image_providers: Sequence[str] = IMAGE_PROVIDERS,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Stages 3 and 4: assets, then assembly.

    The order is dictated by one fact -- the voice decides the timing. Stills
    are only worth generating once the segments know how long they actually
    are, so narration comes first and the storyboard is re-cut against it
    before a single image is requested.
    """
    from moviepy import AudioFileClip, CompositeAudioClip

    from audio_engine import build_ducked_bed, synthesize_narration
    from video_engine import burn_ass_subtitles, source_duration

    def say(message: str) -> None:
        if progress:
            progress(message)

    os.makedirs(workspace, exist_ok=True)
    size = tuple(DURATION_FORMATS.get(str(episode.get("format")),
                                      DURATION_FORMATS[DEFAULT_FORMAT])["aspect"])
    segments = [dict(s) for s in episode["segments"]]
    scratch: list[str] = []

    # --- narration ---------------------------------------------------------
    say(f"Narrating {len(segments)} segments...")
    narration = synthesize_narration(
        episode["script"], provider=voice_provider, voice=voice,
        output_path=os.path.join(workspace, "narration.wav" if voice_provider == "gemini"
                                 else "narration.mp3"),
        style="Measured, unhurried, close-mic. Let the sentences land.",
    )
    speech = float(narration["duration"])
    words = list(narration.get("words") or [])

    # --- re-cut the board onto the real voice ------------------------------
    if words:
        segments = retime_storyboard(segments, words)
        say(f"Re-cut {len(segments)} segments against {len(words)} word timings.")
    else:
        segments = fit_storyboard_to(segments, speech)
        say(f"No word timings from {voice_provider}; scaled the board onto {speech:.1f}s.")

    # Whichever path ran, the picture has to outlast the voice: the mux trims
    # to the shorter stream, so a picture that ends first clips the last words.
    segments = cover_narration(segments, speech)
    runtime = storyboard_runtime(segments)

    # --- stills ------------------------------------------------------------
    pool: dict[str, list[Any]] = {}
    if "pexels" in image_providers:
        pool = build_photo_pool(segments, episode["entities"], progress)

    stills: list[dict[str, Any]] = []
    seen_env: dict[str, int] = {}
    for segment in segments:
        say(f"Still {segment['index']}/{len(segments)}...")
        tag = str(segment.get("environment") or "")
        occurrence = seen_env.get(tag, 0)
        seen_env[tag] = occurrence + 1
        try:
            still = segment_still(segment, episode["entities"],
                                  os.path.join(workspace, "stills"), size,
                                  str(episode.get("aesthetic") or DEFAULT_AESTHETIC),
                                  image_providers, progress=None,
                                  pool=pool, occurrence=occurrence)
        except ImageUnavailable as exc:
            raise RuntimeError(f"Segment {segment['index']}: {exc}") from exc
        stills.append(still)
        segment["still"] = still["path"]
        segment["image_provider"] = still["provider"]

    providers_used = sorted({s["provider"] for s in stills})
    notes = sorted({note for s in stills for note in s["notes"]})

    # --- Ken Burns ---------------------------------------------------------
    clips: list[str] = []
    dissolve = DISSOLVE_SECONDS if transition == "dissolve" else 0.0
    for position, segment in enumerate(segments):
        say(f"Move {segment['index']}/{len(segments)}...")
        # Every clip but the last carries the transition's worth of extra tail.
        pad = dissolve if position < len(segments) - 1 else 0.0
        clip = os.path.join(workspace, f"clip_{segment['index']:03d}.mp4")
        ken_burns_clip(str(segment["still"]), clip,
                       float(segment["duration"]) + pad, size, fps, _move_for(segment))
        clips.append(clip)
        scratch.append(clip)

    say("Stitching the timeline...")
    picture = os.path.join(workspace, "picture.mp4")
    concat_segments(clips, picture, [float(s["duration"]) for s in segments],
                    transition=transition, dissolve=dissolve, fps=fps)
    scratch.append(picture)

    # --- audio -------------------------------------------------------------
    say("Mixing narration and ambience...")
    tracks: list[Any] = []
    open_clips: list[Any] = []
    voice_clip = AudioFileClip(str(narration["path"]))
    open_clips.append(voice_clip)
    tracks.append(voice_clip.with_start(0.0))

    if ambient:
        bed_path = build_ducked_bed(
            runtime + 0.3, style="suspense", narration_path=str(narration["path"]),
            output_path=os.path.join(workspace, "ambient.wav"), volume=float(ambient_volume),
        )
        bed = AudioFileClip(bed_path)
        open_clips.append(bed)
        if bed.duration > runtime:
            bed = bed.subclipped(0, runtime)
        tracks.append(bed)
        scratch.append(bed_path)

    mix = CompositeAudioClip(tracks)
    mix.duration = runtime
    mixed = os.path.join(workspace, "mix.wav")
    mix.write_audiofile(mixed, fps=44100, logger=None)
    open_clips.append(mix)
    scratch.append(mixed)
    for item in open_clips:
        try:
            item.close()
        except Exception:
            pass

    say("Muxing...")
    muxed = os.path.join(workspace, "muxed.mp4")
    _mux(picture, mixed, muxed)
    scratch.append(muxed)

    # --- subtitles ---------------------------------------------------------
    final = muxed
    subtitle_path = ""
    if subtitles:
        say("Burning subtitles...")
        subtitle_path = narrative_subtitles(segments, os.path.join(workspace, "subs.ass"), size)
        burn_ass_subtitles(muxed, subtitle_path, output_path)
        final = output_path
    else:
        import shutil
        shutil.copyfile(muxed, output_path)
        final = output_path

    for path in scratch:
        try:
            os.remove(path)
        except OSError:
            pass

    actual = source_duration(final)
    return {
        "output_path": final,
        "duration": actual,
        "planned_runtime": runtime,
        "narration_seconds": speech,
        "segments": segments,
        "fps": fps,
        "size": size,
        "image_providers": providers_used,
        "provider_notes": notes,
        "tts_provider": str(narration.get("provider") or voice_provider),
        "voice": str(narration.get("voice") or voice),
        "timings_exact": bool(words),
        "transition": transition,
        "subtitles": subtitle_path,
        "ambient": bool(ambient),
    }


def _mux(picture: str, audio: str, output_path: str,
         ffmpeg: str | None = None) -> str:
    """Marries the picture track to the mixed audio without re-encoding video."""
    import subprocess

    import imageio_ffmpeg

    ffmpeg = ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", os.path.abspath(picture), "-i", os.path.abspath(audio),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        # The picture is cut to the voice, so trimming to the shorter of the
        # two only ever removes a rounding frame.
        "-shortest", os.path.abspath(output_path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"mux failed: {(proc.stderr or '')[:300]}")
    return output_path


def metadata_pack(episode: dict[str, Any], result: dict[str, Any]) -> str:
    """The episode's paper trail: script, cast, storyboard and how it was made."""
    look = VISUAL_AESTHETICS.get(str(episode.get("aesthetic")), {})
    tone = NARRATIVE_TONES.get(str(episode.get("tone")), {})
    fmt = DURATION_FORMATS.get(str(episode.get("format")), {})
    ents = episode["entities"]
    p = ents["protagonist"]

    lines = [
        "REELFORGE - NARRATIVE STUDIO EPISODE PACK",
        "=" * 60, "",
        f"TITLE     {episode.get('title')}",
        f"LOGLINE   {episode.get('logline')}",
        f"PREMISE   {episode.get('topic')}",
        "",
        f"Aesthetic {look.get('label', '')}",
        f"Tone      {tone.get('label', '')}",
        f"Format    {fmt.get('label', '')}",
        f"Runtime   {float(result.get('duration') or 0):.1f}s "
        f"({len(result.get('segments') or [])} segments)",
        f"Voice     {result.get('tts_provider')} / {result.get('voice')}"
        + ("  (exact word timings)" if result.get("timings_exact")
           else "  (no word timings - board scaled to the read)"),
        f"Stills    {', '.join(result.get('image_providers') or [])}",
        "",
        "-" * 60, "CAST", "",
        f"{p['name']}, {p['age']}",
        f"  Appearance: {p['appearance']}",
        f"  Clothing  : {p['clothing']}",
        f"  Demeanour : {p['demeanour']}",
        "",
        "ENVIRONMENTS",
    ]
    for env in ents["environments"]:
        lines.append(f"  [{env['tag']}] {env['name']} - {env['description']}")
    if ents["objects"]:
        lines.append("")
        lines.append("RECURRING OBJECTS")
        for obj in ents["objects"]:
            lines.append(f"  [{obj['tag']}] {obj['name']} - {obj['description']}")

    lines += ["", "-" * 60, "STORYBOARD", ""]
    for segment in (result.get("segments") or []):
        lines.append(
            f"  {segment['index']:>3}. {segment['start']:>6.2f}-{segment['end']:>6.2f}s "
            f"[{segment.get('beat_type', '')}/{segment.get('environment', '')}] "
            f"({segment.get('image_provider', '')})"
        )
        lines.append(f"       VO    : {segment['line']}")
        lines.append(f"       Camera: {segment.get('framing', '')}")

    lines += ["", "-" * 60, "FULL NARRATION", "", str(episode.get("script") or "")]
    if result.get("provider_notes"):
        lines += ["", "-" * 60, "PROVIDER NOTES", ""]
        lines += [f"  {note}" for note in result["provider_notes"]]
    return "\n".join(lines)
