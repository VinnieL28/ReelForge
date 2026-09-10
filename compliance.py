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
