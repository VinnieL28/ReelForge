# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false, reportUnknownLambdaType=false
"""
Monetization compliance for Reelforge.

Three jobs:
  1. Describe what each footage licence actually permits.
  2. Keep a provenance ledger -- what footage went into which render, under
     what licence, with what evidence. This is the artefact that answers a
     copyright claim or a monetization appeal months later.
  3. Block a render from being marked publish-ready until the paperwork holds
     up: commercial rights, attribution, licensed TTS, AI disclosure.

None of this is legal advice; it encodes the platform rules and licence terms
that were current when it was written, and both change. Verify before relying
on it for a monetized channel.
"""

from __future__ import annotations

import os
import re
import json
import time
from typing import Any, Sequence

LEDGER_NAME = "provenance.json"


# ---------------------------------------------------------------------------
# Licences
# ---------------------------------------------------------------------------

class Licence:
    """What a given footage licence lets you actually do."""

    def __init__(
        self,
        key: str,
        label: str,
        commercial: bool,
        attribution: bool,
        share_alike: bool = False,
        needs_evidence: bool = False,
        note: str = "",
    ) -> None:
        self.key = key
        self.label = label
        self.commercial = commercial          # may be monetized
        self.attribution = attribution        # credit must be shown
        self.share_alike = share_alike        # derivative inherits the licence
        self.needs_evidence = needs_evidence  # a reference must be recorded
        self.note = note


LICENCES: dict[str, Licence] = {
    "own": Licence(
        "own", "I filmed / created this", commercial=True, attribution=False,
        note="Cleanest option. You hold the rights outright.",
    ),
    "permission": Licence(
        "permission", "Written permission from the creator", commercial=True,
        attribution=False, needs_evidence=True,
        note="Keep the message or agreement. Record where it lives.",
    ),
    "stock": Licence(
        "stock", "Purchased stock licence", commercial=True, attribution=False,
        needs_evidence=True,
        note="Storyblocks, Artgrid, Envato, Pexels. Record the licence/order id.",
    ),
    "pexels": Licence(
        "pexels", "Pexels License (free commercial, no credit required)",
        commercial=True, attribution=False,
        note="Free for commercial use with no attribution required. It is NOT CC0: "
             "you may not sell unaltered copies, imply endorsement by people or "
             "brands shown, or portray identifiable people in a bad light. "
             "Crediting the photographer is appreciated but optional.",
    ),
    "cc0": Licence(
        "cc0", "CC0 / Public Domain", commercial=True, attribution=False,
        note="No credit required, though it is still polite.",
    ),
    "cc-by": Licence(
        "cc-by", "CC BY (credit required)", commercial=True, attribution=True,
        note="Credit the creator on screen or in the description.",
    ),
    "cc-by-sa": Licence(
        "cc-by-sa", "CC BY-SA (credit + share-alike)", commercial=True, attribution=True,
        share_alike=True,
        note="Credit required, and the finished video inherits the same licence. "
             "That is compatible with monetizing, but you cannot claim it as all rights reserved.",
    ),
    "fair_use": Licence(
        "fair_use", "Fair use / transformative commentary (your assertion)",
        commercial=True, attribution=True,
        note="This is a claim you are making, not a licence anyone granted you. "
             "Commentary and criticism are recognised fair-use purposes, but fair "
             "use is a defence decided case by case -- it does not prevent a copyright "
             "claim being filed, and platforms can still remove or demonetize the "
             "video. Keep the commentary substantial and credit the source.",
    ),
    "unverified": Licence(
        "unverified", "Unverified / downloaded from a feed", commercial=False,
        attribution=True,
        note="Someone else's copyrighted work. Not cleared for monetized publishing.",
    ),
}

DEFAULT_LICENCE = "unverified"


# Commons hands back licence names as free text; map the common ones.
_COMMONS_LICENCE_MAP = {
    "cc0": "cc0", "public domain": "cc0", "pd": "cc0", "cc pd": "cc0",
    "cc by": "cc-by", "cc-by": "cc-by",
    "cc by-sa": "cc-by-sa", "cc-by-sa": "cc-by-sa",
    # Providers that state their own licence rather than a Creative Commons one.
    "pexels": "pexels",
}


def normalise_licence(raw: str) -> str:
    """Maps a provider's licence string onto a key in LICENCES."""
    probe = (raw or "").strip().lower()
    if not probe:
        return DEFAULT_LICENCE
    for needle, key in sorted(_COMMONS_LICENCE_MAP.items(), key=lambda kv: -len(kv[0])):
        if probe.startswith(needle):
            return key
    return DEFAULT_LICENCE


# ---------------------------------------------------------------------------
# Text-to-speech licensing
#
# The distinction that catches people out: edge-tts reaches Microsoft Edge's
# read-aloud endpoint, which is not licensed for redistributing the audio in
# monetized content. Fine for drafts; not for a channel that earns.
# ---------------------------------------------------------------------------

TTS_PROVIDERS: dict[str, dict[str, Any]] = {
    "none": {
        "label": "No spoken narration",
        "commercial": True,
        "synthetic": False,
        "note": "A silent cut. Nothing to license and no synthetic speech to "
                "declare, though any AI-written on-screen copy is still yours "
                "to stand behind.",
    },
    "gemini": {
        "label": "Gemini TTS (licensed for commercial use)",
        "commercial": True,
        "synthetic": True,
        "note": "Uses your Gemini API key. Google's terms permit commercial use "
                "of generated output. No word-level timings, so caption timing "
                "is estimated from the audio length.",
    },
    "edge": {
        "label": "edge-tts (DRAFT ONLY - not licensed for monetized publishing)",
        "commercial": False,
        "synthetic": True,
        "note": "Free and gives exact word timings, which makes it ideal for "
                "drafting and for tuning caption timing. Swap to a licensed "
                "voice before publishing anything monetized.",
    },
}


# Providers that actually speak, for narration-engine pickers. "none" is a
# state a render ends up in, not something a user picks from a menu.
SPEAKING_PROVIDERS: tuple[str, ...] = ("gemini", "edge")


# ---------------------------------------------------------------------------
# AI disclosure
# ---------------------------------------------------------------------------

AI_DISCLOSURE_LINE = (
    "This video uses AI-generated narration and an AI-assisted script. "
    "The footage is real and used under the licence credited above."
)

PLATFORM_DISCLOSURE_STEPS = {
    "YouTube": "In YouTube Studio, tick 'Altered or synthetic content' when uploading "
               "(Details -> Show more -> Altered content).",
    "TikTok": "Turn on the 'AI-generated content' label in the post screen "
              "(More options -> AI-generated content).",
    "Instagram": "Use the 'AI info' label when posting a Reel with synthetic audio.",
}

# Verify these before relying on them; platforms move the goalposts.
MONETIZATION_NOTES = {
    "YouTube": "Partner Programme: 1,000 subscribers plus either 4,000 valid public "
               "watch hours in 12 months, or 10 million valid public Shorts views in "
               "90 days. Channels built on other people's content are refused under "
               "the reused/inauthentic content rules.",
    "TikTok": "Creator Rewards: roughly 10,000 followers and 100,000 views in 30 days, "
              "and videos must run LONGER THAN ONE MINUTE to qualify at all.",
}

TIKTOK_REWARDS_MIN_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Provenance ledger
# ---------------------------------------------------------------------------

def ledger_path(exports_dir: str) -> str:
    return os.path.join(exports_dir, LEDGER_NAME)


def load_ledger(exports_dir: str) -> list[dict[str, Any]]:
    """Reads the provenance ledger, tolerating a missing or corrupt file."""
    path = ledger_path(exports_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (ValueError, OSError):
        return []


def append_ledger(exports_dir: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Appends one render's provenance and returns the stored entry."""
    os.makedirs(exports_dir, exist_ok=True)
    entries = load_ledger(exports_dir)
    entry = dict(entry)
    entry.setdefault("recorded_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    entry.setdefault("id", f"{int(time.time())}-{len(entries) + 1}")
    entries.append(entry)
    with open(ledger_path(exports_dir), "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, ensure_ascii=False)
    return entry


def find_entry(exports_dir: str, video_name: str) -> dict[str, Any] | None:
    """Finds the newest ledger entry for a rendered file."""
    for entry in reversed(load_ledger(exports_dir)):
        if str(entry.get("video_name")) == video_name:
            return entry
    return None


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

def attribution_line(entry: dict[str, Any]) -> str:
    """The credit string to burn on screen or paste into the description."""
    licence = LICENCES.get(str(entry.get("licence") or DEFAULT_LICENCE), LICENCES[DEFAULT_LICENCE])
    if not licence.attribution:
        return ""

    author = str(entry.get("source_author") or "").strip() or "Unknown creator"
    title = str(entry.get("source_title") or "").strip()
    url = str(entry.get("source_url") or "").strip()

    parts = [f'"{title}"' if title else "Source footage", f"by {author}", f"({licence.label})"]
    if url:
        parts.append(f"- {url}")
    return " ".join(parts)


def publish_readiness(entry: dict[str, Any]) -> dict[str, Any]:
    """
    Decides whether a render is safe to publish on a monetized channel.

    Returns {"ready": bool, "blockers": [...], "warnings": [...], "passed": [...]}.
    Blockers are things that would get the video claimed or the channel
    demonetized; warnings are things worth knowing.
    """
    blockers: list[str] = []
    warnings: list[str] = []
    passed: list[str] = []

    licence = LICENCES.get(str(entry.get("licence") or DEFAULT_LICENCE), LICENCES[DEFAULT_LICENCE])

    # --- footage rights ----------------------------------------------------
    if not licence.commercial:
        blockers.append(
            f"Footage licence is '{licence.label}'. {licence.note} "
            "Replace it with licensed footage, or record the permission you hold."
        )
    elif licence.key == "fair_use":
        # Worded deliberately as a claim. The ledger must never read as though
        # a rights holder granted something they did not.
        when = str(entry.get("fair_use_asserted_at") or "an unrecorded date")
        passed.append(f"Publishing under your own fair-use assertion (made {when}).")
        warnings.append(
            "Fair use is a defence, not a permission. A copyright claim can still be filed "
            "against this video, and YouTube's reused-content review is a separate judgement "
            "again. It rests on your commentary being substantial and transformative."
        )
    else:
        passed.append(f"Footage cleared for commercial use ({licence.label}).")

    if licence.needs_evidence and not str(entry.get("licence_reference") or "").strip():
        blockers.append(
            f"'{licence.label}' needs a reference recorded (order id, licence number, "
            "or where the permission message is stored)."
        )
    elif licence.needs_evidence:
        passed.append("Licence reference on file.")

    if licence.attribution:
        if attribution_line(entry):
            passed.append("Attribution line generated for the description.")
        else:
            blockers.append(
                f"'{licence.label}' requires credit, but no creator is recorded for this footage."
            )

    if licence.share_alike:
        warnings.append(
            "Share-alike licence: the finished video inherits CC BY-SA. You can still "
            "monetize it, but you cannot claim it as all rights reserved, and others "
            "may reuse it under the same terms."
        )

    # --- voice rights ------------------------------------------------------
    provider = str(entry.get("tts_provider") or "edge")
    spec = TTS_PROVIDERS.get(provider, TTS_PROVIDERS["edge"])
    if not spec["commercial"]:
        blockers.append(
            f"Narration was produced with {spec['label']}. Re-voice with a licensed "
            "provider before publishing to a monetized channel."
        )
    elif spec.get("synthetic", True):
        passed.append(f"Narration voice is licensed for commercial use ({provider}).")
    else:
        passed.append(f"{spec['label']} - nothing to license on the voice track.")

    # --- disclosure --------------------------------------------------------
    # Only synthetic speech forces the label. A silent vector animation has no
    # synthetic media in it, and blocking it with a message about narration it
    # does not have would be both wrong and untrustworthy.
    if entry.get("ai_disclosed"):
        passed.append("AI disclosure acknowledged.")
    elif spec.get("synthetic", True):
        blockers.append(
            "AI narration must be disclosed. Tick the disclosure box, and set the "
            "synthetic-content label when you upload."
        )
    else:
        passed.append("No synthetic voice in this render, so the synthetic-media "
                      "label is not required.")
        warnings.append(
            "The on-screen copy was AI-assisted even though no synthetic voice was "
            "used. Neither platform requires a label for that, but saying so in the "
            "description costs nothing."
        )

    # --- platform fit ------------------------------------------------------
    duration = float(entry.get("duration") or 0.0)
    if duration and duration < TIKTOK_REWARDS_MIN_SECONDS:
        warnings.append(
            f"Runs {duration:.0f}s. TikTok Creator Rewards only counts videos over "
            f"{TIKTOK_REWARDS_MIN_SECONDS:.0f}s, so this earns nothing there. "
            "YouTube Shorts monetization has no such floor."
        )
    elif duration:
        passed.append(f"Runs {duration:.0f}s - long enough for TikTok Creator Rewards.")

    warnings.append(
        "Narration and captions make the edit transformative, which is what YouTube's "
        "reused-content review looks for. Keep the commentary substantive rather than "
        "a few words over an untouched clip."
    )

    passed.append("Music and sound effects are synthesized in-app - nothing to clear.")

    return {
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "passed": passed,
    }


def build_publish_pack(entry: dict[str, Any], script: str = "") -> str:
    """
    Assembles the text you actually need at upload time: description with
    credits, the AI disclosure, per-platform label steps, and the licence trail.
    """
    licence = LICENCES.get(str(entry.get("licence") or DEFAULT_LICENCE), LICENCES[DEFAULT_LICENCE])
    credit = attribution_line(entry)
    lines: list[str] = []

    lines.append("=" * 66)
    lines.append("PUBLISH PACK - " + str(entry.get("video_name") or "render"))
    lines.append("=" * 66)

    lines.append("\n--- DESCRIPTION (paste into YouTube / TikTok) ---\n")
    if credit:
        lines.append(f"Footage: {credit}")
    lines.append(AI_DISCLOSURE_LINE)

    lines.append("\n--- REQUIRED LABELS AT UPLOAD ---\n")
    for platform, step in PLATFORM_DISCLOSURE_STEPS.items():
        lines.append(f"[{platform}] {step}")

    lines.append("\n--- LICENCE TRAIL (keep this) ---\n")
    lines.append(f"Licence          : {licence.label}")
    if licence.key == "fair_use":
        lines.append("Fair use claimed : yes, asserted by the uploader on "
                     f"{entry.get('fair_use_asserted_at', 'an unrecorded date')}")
        lines.append("                   (an assertion, not a granted licence)")
    if entry.get("licence_reference"):
        lines.append(f"Reference        : {entry['licence_reference']}")
    for field, caption in (
        ("source_title", "Source title"), ("source_author", "Source author"),
        ("source_url", "Source URL"), ("source_provider", "Source provider"),
    ):
        if entry.get(field):
            lines.append(f"{caption:17}: {entry[field]}")
    lines.append(f"Narration voice  : {entry.get('tts_provider', '?')} / {entry.get('voice', '?')}")
    lines.append(f"Script model     : {entry.get('script_model', '?')}")
    lines.append(f"Rendered         : {entry.get('recorded_at', '?')}")
    lines.append(f"Duration         : {float(entry.get('duration') or 0):.1f}s")

    if script.strip():
        lines.append("\n--- SPOKEN SCRIPT ---\n")
        lines.append(script.strip())

    lines.append("\n--- MONETIZATION NOTES (verify, these change) ---\n")
    for platform, note in MONETIZATION_NOTES.items():
        lines.append(f"[{platform}] {note}")

    return "\n".join(lines) + "\n"


def summarise_ledger(exports_dir: str) -> dict[str, Any]:
    """Counts of what is publish-ready versus blocked, for the dashboard."""
    entries = load_ledger(exports_dir)
    ready = sum(1 for e in entries if publish_readiness(e)["ready"])
    return {"total": len(entries), "ready": ready, "blocked": len(entries) - ready}


# ---------------------------------------------------------------------------
# Viral scorecard
#
# Three axes, scored 1-10, deliberately measurable rather than vibes:
#
#   Hook intrigue       -- what the first three seconds do. A hook either opens
#                          a loop the viewer needs closed, or it announces a
#                          topic and lets them scroll.
#   Information density -- how much of the script is load-bearing. The failure
#                          mode of a templated script is not being wrong, it is
#                          saying nothing: "this costs more than you think"
#                          carries no number, no name and no mechanism.
#   Monetization safety -- whether the video survives YouTube's reused-content
#                          review and TikTok's originality rules.
#
# Gemini scores the same three axes when a key is configured
# (gemini_engine.score_virality). This module is the floor: it always runs, it
# needs no network, and it is what the tests assert against.
# ---------------------------------------------------------------------------

VIRAL_TARGET_SCORE = 8.0
HOOK_SECONDS = 3.0
# At roughly 2.75 words per second, three seconds of speech is about nine words.
HOOK_WORDS = 9

# An opening with none of the measured signals in it. Not a midpoint and not a
# passing mark: it is what "states the topic and nothing more" is worth in a
# feed where the scroll is decided in three seconds.
HOOK_FLOOR = 2.0

# A first sentence this long or shorter has finished speaking inside the first
# three seconds at a normal narration pace.
HOOK_SENTENCE_WORDS = 12

# Above this the card reads as workable rather than broken. Distinct from
# VIRAL_TARGET_SCORE, which is the bar for publishing.
WORKABLE_SCORE = 6.5

# Below this the monetization axis is not a weakness, it is a blocker: a
# licence that is not cleared, or narration too thin to read as transformative.
MONETIZATION_FLOOR = 6.0

# Openings that state a subject instead of opening a loop. Every one of these
# is a sentence the viewer has heard before, which is the problem.
_WEAK_HOOK_OPENERS = (
    "in this video", "today i want to", "today we are", "today we're",
    "let me tell you", "have you ever wondered", "welcome back",
    "hey guys", "what's up guys", "here are", "here is", "this is a video",
    "i'm going to show you", "we are going to talk", "let's talk about",
    "did you know that", "so basically",
)

# Phrases that fill runtime without adding information. Counted, not banned --
# one is a figure of speech, six is a script with nothing in it.
_FILLER_PHRASES = (
    "more than you think", "you won't believe", "it's crazy", "it's insane",
    "game changer", "next level", "at the end of the day", "the truth is",
    "let that sink in", "trust me", "literally everything", "change your life",
    "the secret is", "nobody talks about", "most people don't know",
    "this one simple", "is key", "is everything", "think about it",
    "believe it or not", "no cap", "mind blowing",
)

# Openings that do open a loop: a number, a contradiction, a stake, a refusal.
_STRONG_HOOK_SIGNALS = (
    "stop", "never", "everyone", "nobody", "wrong", "mistake", "actually",
    "cost me", "lost", "why", "how i", "the reason", "until i", "before you",
)


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", str(text or "").strip()) if w]


# List scaffolding: "Number 3", "Mistake #2:", "Tip 4", "5 things", "part two".
# These digits are structure, not information, and counting them as facts rates
# a pure-template script the same as a researched one -- measured: the old Reel
# Studio listicle scored 8.7 "facts per 100 words" on nothing but its own
# numbering.
_LIST_MARKER_RE = re.compile(
    r"\b(?:number|mistake|tip|secret|reason|step|rule|habit|point|way|fact|thing|part|lesson)s?"
    r"\s*#?\s*(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE)
_LEADING_COUNT_RE = re.compile(
    r"\b\d+\s+(?:things?|ways?|reasons?|facts?|secrets?|tips?|mistakes?|habits?|rules?|steps?|lessons?)\b",
    re.IGNORECASE)


def _strip_list_markers(text: str) -> str:
    """Removes listicle numbering so it cannot be mistaken for a real figure."""
    return _LEADING_COUNT_RE.sub(" ", _LIST_MARKER_RE.sub(" ", str(text or "")))


# "twenty-three", "forty seven", "a hundred and twelve". Deliberately excludes
# bare "one" through "ten": "one of the reasons" and "no second chances" are
# not figures, and counting them would let vague copy score as specific.
_SPELLED_NUMBER = (
    r"\b(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?"
    r"|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
    r"|hundred|thousand|million|billion|trillion)\b"
)


def _concrete_tokens(text: str) -> dict[str, int]:
    """
    Counts the things that make a sentence checkable.

    Numbers, units, proper nouns and years are what separate "it costs more
    than you think" from "it costs $1,299, which is $100 over the S24 Ultra".
    """
    body = str(text or "")
    numbers = re.findall(r"(?<![\w.])\d[\d,]*\.?\d*(?![\w])", body)
    # Spelled-out numbers count too. These scripts are *spoken*, so a writer
    # aiming at TTS legitimately writes "twenty-three minutes" rather than "23
    # minutes" -- and a metric that only sees digits rates a well-researched
    # narration script as empty. Measured on a live Gemini research script:
    # 60% sentence coverage with digits only, 100% once words are counted.
    numbers += re.findall(_SPELLED_NUMBER, body, re.IGNORECASE)
    units = re.findall(
        r"\b\d[\d,.]*\s?(?:%|percent|dollars?|usd|hours?|hrs?|minutes?|mins?|"
        r"seconds?|secs?|days?|weeks?|months?|years?|kg|lbs?|km|miles?|mph|hp|"
        r"gb|mb|tb|x|times|k|m|bn|billion|million|thousand)\b", body, re.IGNORECASE)
    years = re.findall(r"\b(?:19|20)\d{2}\b", body)
    # Capitalised words that are not sentence openers: names, brands, places.
    propers = re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-zA-Z]{2,}\b", body)

    return {
        "numbers": len(numbers),
        "units": len(units),
        "years": len(years),
        "propers": len(set(propers)),
    }


def score_hook(script: str) -> dict[str, Any]:
    """Scores the first three seconds of speech, 1-10, and says why."""
    words = _words(script)
    if not words:
        return {"score": 1.0, "opening": "",
                "notes": ["There is no script to hook anyone with."]}

    opening = " ".join(words[:HOOK_WORDS])
    low = opening.lower()
    notes: list[str] = []

    # Built from evidence rather than adjusted away from a neutral midpoint.
    # The midpoint start was the bug: a hook with nothing measurable in it came
    # back a 5, which reads as "average" when what it actually means is "this
    # opening gives no one a reason to stay". An opening earns its score.
    score = HOOK_FLOOR
    diagnosis: list[str] = []

    weak = next((p for p in _WEAK_HOOK_OPENERS if low.startswith(p) or f" {p}" in f" {low}"), "")
    if weak:
        score -= 2.0
        notes.append(f'Opens with "{weak}" -- that announces a topic instead of opening a loop.')
        diagnosis.append("weak_opener")

    tokens = _concrete_tokens(opening)
    if tokens["numbers"] or tokens["units"]:
        score += 2.0
        notes.append("Leads with a specific figure, which is a reason to keep watching.")
    else:
        diagnosis.append("no_figure")

    if any(sig in low for sig in _STRONG_HOOK_SIGNALS):
        score += 2.2
        notes.append("Opens on a contradiction or a stake rather than a subject.")
    else:
        diagnosis.append("no_stake")

    if "?" in opening:
        score += 0.8
        notes.append("Poses a question in the first breath.")
    if tokens["propers"]:
        score += 0.6
        notes.append("Names something specific in the opening line.")

    # A hook that takes twenty words to arrive has already lost the scroll.
    first_sentence = re.split(r"(?<=[.!?])\s", str(script).strip())[0]
    opening_words = len(_words(first_sentence))
    if opening_words <= HOOK_SENTENCE_WORDS:
        score += 1.8
        notes.append(f"The first sentence lands in {opening_words} words, inside the "
                     "three seconds the scroll is decided in.")
    elif opening_words > 18:
        score -= 1.0
        notes.append(f"The first sentence runs {opening_words} words -- "
                     "it lands after the scroll decision is made.")
        diagnosis.append("slow_open")

    return {"score": max(1.0, min(10.0, score)), "opening": opening, "notes": notes,
            "diagnosis": diagnosis, "opening_words": opening_words}


def score_density(script: str) -> dict[str, Any]:
    """Scores how much of the script is load-bearing, 1-10."""
    words = _words(script)
    body = str(script or "")
    low = body.lower()
    notes: list[str] = []

    if len(words) < 12:
        return {"score": 1.0, "filler_hits": [], "per_100": 0.0, "covered": 0.0,
                "notes": ["Too short to carry any information."]}

    stripped = _strip_list_markers(body)
    tokens = _concrete_tokens(stripped)
    facts = tokens["numbers"] + tokens["units"] + tokens["years"] + tokens["propers"]
    per_100 = facts / (len(words) / 100.0)

    # Density is two questions, not one: how much is in the script, and how
    # evenly it is spread. A script can hit a good per-100 rate from one
    # fact-stuffed sentence while the other five say nothing -- and those five
    # are where the viewer leaves. `covered` is the fraction of sentences
    # carrying at least one checkable token.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", stripped.strip()) if len(_words(s)) >= 3]
    with_fact = sum(1 for s in sentences if sum(_concrete_tokens(s).values()) > 0)
    covered = with_fact / len(sentences) if sentences else 0.0

    # Calibrated against this project's own output: the Reel Studio template
    # scores 0 per 100 once its own numbering is discounted, the offline fact
    # bank lands near 20, and a researched script near 33.
    score = 2.0 + min(4.0, per_100 * 0.30) + 4.0 * covered
    notes.append(f"{facts} checkable details in {len(words)} words ({per_100:.1f} per 100); "
                 f"{with_fact} of {len(sentences)} sentences carry one.")

    filler_hits = [p for p in _FILLER_PHRASES if p in low]
    if filler_hits:
        score -= min(4.0, 0.9 * len(filler_hits))
        shown = ", ".join(f'"{p}"' for p in filler_hits[:4])
        notes.append(f"{len(filler_hits)} filler phrase(s) doing no work: {shown}.")

    # "Mistake 1:" / "Tip 3:" numbering with nothing behind it is the tell of a
    # template. The numbering is fine; the numbering *plus* no facts is not.
    if re.search(r"\b(?:mistake|tip|reason|secret|step|rule)\s*#?\s*\d", low) and per_100 < 3.0:
        score -= 1.5
        notes.append("Numbered list structure with almost no specifics behind it -- "
                     "that is the shape of a template, not a script.")

    if len(set(w.lower() for w in words)) / len(words) < 0.45:
        score -= 1.0
        notes.append("Heavily repetitive vocabulary.")

    return {"score": max(1.0, min(10.0, score)), "filler_hits": filler_hits,
            "per_100": per_100, "covered": covered, "notes": notes}


def score_monetization(entry: dict[str, Any], script: str = "") -> dict[str, Any]:
    """Scores survival odds against reused-content and originality review."""
    notes: list[str] = []
    score = 8.0

    licence_key = str(entry.get("licence") or DEFAULT_LICENCE)
    licence = LICENCES.get(licence_key, LICENCES[DEFAULT_LICENCE])
    duration = float(entry.get("duration") or 0.0)
    words = len(_words(script))

    if not licence.commercial:
        score -= 5.0
        notes.append(f"Footage licence '{licence.label}' is not cleared for a monetized upload.")
    elif licence_key == "fair_use":
        score -= 1.5
        notes.append("Rests on a fair-use assertion, which reused-content review judges separately.")
    else:
        notes.append(f"Footage cleared ({licence.label}).")

    # YouTube's reused-content rule turns on how much of the video is *yours*.
    # Commentary over someone else's clip with twenty words of narration is the
    # exact shape that gets refused.
    if duration > 0 and words:
        wps = words / duration
        if wps < 1.2:
            score -= 2.0
            notes.append(f"Only {words} words across {duration:.0f}s ({wps:.1f} words/s) -- "
                         "too little original commentary to read as transformative.")
        else:
            notes.append(f"{words} words of original narration over {duration:.0f}s.")

    provider = TTS_PROVIDERS.get(str(entry.get("tts_provider") or ""), {})
    if provider.get("synthetic") and not entry.get("ai_disclosed"):
        score -= 1.5
        notes.append("Synthetic narration is not disclosed -- both platforms require the label.")

    if 0 < duration < TIKTOK_REWARDS_MIN_SECONDS:
        notes.append(f"Under {TIKTOK_REWARDS_MIN_SECONDS:.0f}s, so it earns nothing from "
                     "TikTok Creator Rewards (YouTube Shorts is unaffected).")

    return {"score": max(1.0, min(10.0, score)), "notes": notes}


# What each axis being the weakest actually means, in the words the verdict
# uses. Kept beside the verdict so the two cannot drift apart.
_AXIS_FAULTS = {
    "hook": "the first three seconds do not open a loop",
    "density": "the script has too little in it",
    "monetization": "it will not survive monetization review",
}


def weakest_axis(card: dict[str, Any]) -> str:
    """The axis dragging the card down, by name."""
    return min(_AXIS_FAULTS, key=lambda axis: float(
        (card.get(axis) or {}).get("score", 10.0)))


def verdict_for(overall: float, card: dict[str, Any]) -> str:
    """
    One line on what the number means.

    Shared by the offline card and the model-led one so a score of 7.4 never
    reads as "workable" on one path and "not ready" on the other.
    """
    axis = weakest_axis(card)
    fault = _AXIS_FAULTS[axis]

    if overall >= 8.5:
        return "Strong on all three axes."
    if overall >= VIRAL_TARGET_SCORE:
        return "Good enough to publish."
    if overall >= WORKABLE_SCORE:
        return f"Workable, but {fault}."
    return f"Not ready: {fault}."


def retention_tip(hook: dict[str, Any], density: dict[str, Any],
                  money: dict[str, Any]) -> str:
    """
    One sentence naming the single change that buys the most retention.

    Derived from the same measurements the axes are scored on, so it can point
    at the actual phrase or the actual missing figure rather than saying "add
    more detail". Ordered by what costs the most viewers: the first three
    seconds outrank everything, because nothing later in the video recovers a
    scroll that already happened.
    """
    marks = set(hook.get("diagnosis") or ())
    opening = str(hook.get("opening") or "").strip()

    if "weak_opener" in marks:
        return (f'Delete the throat-clearing from "{opening[:48]}" and open on the '
                f'hardest fact in the script -- the viewer decides before you '
                f'finish announcing the topic.')

    if "slow_open" in marks:
        return (f'Your first sentence runs {hook.get("opening_words", 0)} words; cut it '
                f'to twelve or fewer so the hook finishes speaking before the three '
                f'second scroll decision.')

    if "no_figure" in marks and "no_stake" in marks:
        return ("Put a number, a date or a named thing in the first nine words -- the "
                "opening currently states a subject and gives no reason to stay for "
                "the second sentence.")

    if "no_stake" in marks:
        return ("Reframe the opening as a contradiction of what the viewer already "
                "believes rather than a statement of the topic, so the first three "
                "seconds open a loop instead of closing one.")

    filler = list(density.get("filler_hits") or ())
    if filler:
        shown = ", ".join(f'"{p}"' for p in filler[:2])
        return (f"Cut {shown} and spend the seconds on a checkable figure instead; "
                f"filler in the first third is where the retention curve bends.")

    covered = float(density.get("covered") or 0.0)
    if covered < 0.6:
        return (f"Only {covered * 100:.0f}% of your sentences carry something checkable "
                f"-- give the two emptiest sentences a figure each, or cut them and "
                f"reach the payoff sooner.")

    if float(money.get("score") or 0.0) < 6.0:
        return ("Fix the monetization flag before retention: a video that cannot be "
                "monetized does not benefit from a better hook.")

    return (f'Hold "{opening[:40]}" as the opening and cut any sentence before the '
            f'first figure -- the hook is working, so the only thing left to buy is '
            f'getting to the payoff faster.')


def viral_scorecard(script: str, entry: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    The full 1-10 scorecard.

    Returns {"overall", "hook", "density", "monetization", "verdict",
    "needs_rewrite", "source"}. `needs_rewrite` is what the UI hangs the
    "Rewrite for High Retention" button on.

    The overall is weighted toward the hook because retention is decided in the
    first three seconds and nothing later in the video recovers a scroll.
    """
    entry = entry or {}
    hook = score_hook(script)
    density = score_density(script)
    money = score_monetization(entry, script)

    overall = round(hook["score"] * 0.45 + density["score"] * 0.35 + money["score"] * 0.20, 1)

    # A monetization failure is categorical, not a weighted contribution. A
    # researched script over uncleared footage used to average out to exactly
    # 8.0 and report "good enough to publish" -- of something that cannot be
    # monetized at all. The ceiling is derived from the failing axis rather
    # than being a fixed number, so the card still moves as the problem is
    # fixed instead of parking on one value.
    if money["score"] < MONETIZATION_FLOOR:
        overall = round(min(overall, money["score"] + 2.0), 1)

    weakest = weakest_axis({"hook": hook, "density": density, "monetization": money})
    verdict = verdict_for(overall, {"hook": hook, "density": density,
                                    "monetization": money})

    return {
        "overall": overall,
        "hook": hook,
        "density": density,
        "monetization": money,
        "verdict": verdict,
        "weakest": weakest,
        "retention_tip": retention_tip(hook, density, money),
        "needs_rewrite": overall < VIRAL_TARGET_SCORE,
        "source": "heuristic",
    }
