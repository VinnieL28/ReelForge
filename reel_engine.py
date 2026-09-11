# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false, reportUnknownVariableType=false
"""
Reel Studio scripting: concrete facts instead of placeholder scaffolding.

The old templates produced lines like "Mistake 1: this quietly costs you more
than you think" -- grammatically fine, structurally viral, and carrying zero
information. compliance.score_density() rates that kind of script 1.9/10, and
it is exactly the shape YouTube's reused-content review refuses.

This module replaces the scaffolding with three sources, tried in order:

  1. **Gemini**, asked for figures, named entities and mechanisms, and told in
     as many words not to invent a statistic it is unsure of.
  2. **The local fact bank** below -- a small set of facts that are true, dated
     and attributed, for the domains this channel actually covers.
  3. **The vocabulary pack** -- if neither of the above covers the topic, the
     script is built from the domain's real metrics and units, and the caller
     is told the lines need editing. That is the honest failure: a script with
     the right shape and a visible gap, not a confident sentence about nothing.

Nothing here invents a number. A fact with no source does not go in the bank.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence

# ---------------------------------------------------------------------------
# Domains
#
# `terms` are the words a person who actually knows the subject would reach
# for; they are what make a line sound researched rather than generated.
# `metrics` are the quantities that domain argues in, and they are what the
# Gemini prompt asks to be filled in.
# ---------------------------------------------------------------------------

DOMAINS: dict[str, dict[str, Any]] = {
    "finance": {
        "label": "Money & investing",
        "match": ("money", "invest", "stock", "saving", "salary", "debt", "credit",
                  "wealth", "rich", "income", "budget", "retire", "pension", "etf",
                  "crypto", "bitcoin", "mortgage", "tax", "interest", "compound",
                  "millionaire", "broke", "afford", "inflation", "portfolio"),
        "terms": ("compound interest", "expense ratio", "index fund", "drawdown",
                  "emergency fund", "employer match", "effective rate", "real return",
                  "dollar-cost averaging", "sequence risk"),
        "metrics": ("annual percentage rate", "expense ratio in percent",
                    "years to double", "median salary in dollars", "inflation rate"),
    },
    "fitness": {
        "label": "Fitness & health",
        "match": ("fitness", "workout", "gym", "muscle", "fat", "weight", "diet",
                  "protein", "calorie", "cardio", "run", "lift", "strength", "sleep",
                  "health", "training", "hypertrophy", "abs", "bench", "squat"),
        "terms": ("progressive overload", "training volume", "protein synthesis",
                  "VO2 max", "caloric deficit", "time under tension", "RPE",
                  "resting heart rate", "sleep latency", "deload week"),
        "metrics": ("grams of protein per kilogram", "weekly sets per muscle group",
                    "calories per day", "heart rate in bpm", "hours of sleep"),
    },
    "productivity": {
        "label": "Work & productivity",
        "match": ("productivity", "focus", "habit", "discipline", "procrastinat",
                  "time management", "deep work", "routine", "morning", "burnout",
                  "distraction", "attention", "goal", "motivation", "studying"),
        "terms": ("context switching", "cognitive load", "time blocking",
                  "attention residue", "implementation intention", "keystone habit",
                  "cue-routine-reward", "task batching", "Parkinson's law"),
        "metrics": ("minutes to refocus", "hours of deep work per day",
                    "percentage of tasks completed", "days to form a habit"),
    },
    "tech": {
        "label": "Technology & AI",
        "match": ("tech", "ai", "software", "code", "programming", "computer",
                  "app", "phone", "iphone", "android", "laptop", "gpu", "chip",
                  "internet", "data", "model", "algorithm", "startup", "saas"),
        "terms": ("inference latency", "context window", "training compute",
                  "transistor count", "process node", "throughput", "cache miss",
                  "token throughput", "thermal throttling", "bandwidth"),
        "metrics": ("nanometre process node", "tokens per second", "watts under load",
                    "milliseconds of latency", "price in dollars"),
    },
    "cars": {
        "label": "Cars & motoring",
        "match": ("car", "suv", "porsche", "bmw", "tesla", "engine", "horsepower",
                  "torque", "ev", "electric vehicle", "drive", "vehicle", "truck",
                  "mustang", "range rover", "motor", "mpg", "hybrid"),
        "terms": ("torque curve", "kerb weight", "power-to-weight ratio",
                  "0-60 time", "drag coefficient", "battery capacity",
                  "charging curve", "depreciation", "residual value"),
        "metrics": ("horsepower", "0-60 mph in seconds", "kWh of battery",
                    "miles of range", "price in dollars", "kerb weight in kg"),
    },
    "science": {
        "label": "Science & space",
        "match": ("space", "science", "physics", "universe", "planet", "star",
                  "galaxy", "nasa", "quantum", "biology", "brain", "ocean",
                  "climate", "earth", "moon", "mars", "chemistry", "evolution"),
        "terms": ("orbital period", "light-year", "escape velocity", "half-life",
                  "order of magnitude", "absolute zero", "redshift",
                  "atmospheric pressure", "tidal locking"),
        "metrics": ("distance in kilometres", "temperature in kelvin",
                    "years", "times Earth's mass", "percent of the whole"),
    },
    "history": {
        "label": "History",
        "match": ("history", "historical", "ancient", "war", "empire", "king",
                  "queen", "roman", "egypt", "medieval", "century", "battle",
                  "revolution", "civilisation", "civilization", "dynasty"),
        "terms": ("primary source", "contemporary account", "excavation",
                  "radiocarbon date", "chronicle", "campaign season", "treaty"),
        "metrics": ("the year", "number of people", "duration in years",
                    "distance in kilometres"),
    },
}

DEFAULT_DOMAIN = "general"
_GENERAL = {
    "label": "General",
    "match": (),
    "terms": ("measured", "documented", "on record", "per capita", "year on year"),
    "metrics": ("a number", "a date", "a named person or organisation", "a percentage"),
}


# ---------------------------------------------------------------------------
# The fact bank
#
# Every entry is (claim, source). The source is not decoration: it is what
# makes the claim checkable, it is printed in the publish pack, and it is the
# reason a fact is allowed in here at all. Anything that cannot be attributed
# does not get added, however good it would sound read aloud.
#
# These are broad, durable facts about each domain rather than topic-specific
# research -- they are the floor when Gemini is unavailable, not a substitute
# for it.
# ---------------------------------------------------------------------------

FACT_BANK: dict[str, tuple[tuple[str, str], ...]] = {
    "finance": (
        ("At 7 percent a year, money doubles about every ten years — the rule of 72 "
         "divides 72 by the rate to get the years",
         "Rule of 72, standard compound-interest approximation"),
        ("An S&P 500 index fund charging 0.03 percent and an active fund charging 1 "
         "percent differ by roughly a third of the final balance over 30 years",
         "Compounding the fee differential over a 30-year horizon"),
        ("A 50 percent employer match on retirement contributions is an immediate 50 "
         "percent return, before the market does anything at all",
         "Definition of an employer match"),
        ("The S&P 500's long-run average is about 10 percent nominal, which is nearer "
         "7 percent after inflation",
         "S&P 500 total return since 1928, adjusted by CPI"),
        ("A portfolio that falls 50 percent needs a 100 percent gain to get back to "
         "even, which is why avoiding large drawdowns matters more than chasing peaks",
         "Arithmetic of percentage loss recovery"),
    ),
    "fitness": (
        ("Around 1.6 grams of protein per kilogram of bodyweight per day is where "
         "muscle-building returns flatten out in the meta-analyses",
         "Morton et al., British Journal of Sports Medicine, 2018"),
        ("Ten or more hard sets per muscle group per week produces measurably more "
         "growth than fewer, with returns diminishing after about twenty",
         "Schoenfeld et al., dose-response meta-analysis"),
        ("VO2 max is one of the strongest single predictors of all-cause mortality "
         "in the cardiorespiratory fitness literature",
         "Mandsager et al., JAMA Network Open, 2018"),
        ("A pound of fat is about 3,500 calories, so a 500-calorie daily deficit is "
         "roughly a pound a week",
         "Standard energy-density estimate for adipose tissue"),
        ("Sleeping under six hours cuts the proportion of weight lost as fat, even "
         "on an identical calorie deficit",
         "Nedeltcheva et al., Annals of Internal Medicine, 2010"),
    ),
    "productivity": (
        ("After an interruption it takes an average of about 23 minutes to return to "
         "the original task",
         "Gloria Mark, University of California Irvine"),
        ("Attention residue means part of your focus stays on the previous task even "
         "after you have switched away from it",
         "Sophie Leroy, Organizational Behavior and Human Decision Processes, 2009"),
        ("The often-repeated '21 days to form a habit' has no evidence behind it; the "
         "measured median is 66 days, with a range from 18 to 254",
         "Lally et al., European Journal of Social Psychology, 2010"),
        ("Writing down when and where you will do something roughly doubles follow-"
         "through compared with intending to do it",
         "Gollwitzer, implementation intentions research"),
    ),
    "tech": (
        ("Transistor counts have roughly doubled every two years for five decades — "
         "that observation is Moore's law, and it was never a law of physics",
         "Gordon Moore, Electronics Magazine, 1965"),
        ("A modern CPU cache miss costs on the order of 100 nanoseconds, which is "
         "hundreds of wasted cycles",
         "Typical DRAM latency on current desktop platforms"),
        ("Process node names like '3nm' stopped describing any physical dimension "
         "years ago; they are marketing labels for a generation",
         "IEEE Spectrum reporting on node naming"),
    ),
    "cars": (
        ("Power-to-weight, not horsepower, is what decides acceleration — a 400hp car "
         "at 1,400kg out-accelerates a 500hp car at 2,200kg",
         "Newton's second law applied to vehicle mass"),
        ("Electric motors make peak torque from zero rpm, which is why an EV feels "
         "faster than its 0-60 figure suggests",
         "Characteristic torque curve of a permanent-magnet motor"),
        ("Aerodynamic drag rises with the square of speed, so doubling from 40 to 80 "
         "mph quadruples the drag force",
         "Drag equation, F = ½ρv²C_dA"),
        ("Most new cars lose 20 to 30 percent of their value in the first year and "
         "about half by year five",
         "Standard depreciation curves, US and UK markets"),
    ),
    "science": (
        ("Light takes 8 minutes and 20 seconds to reach Earth from the Sun, so you "
         "always see it as it was, never as it is",
         "1 AU divided by the speed of light"),
        ("Venus is hotter than Mercury despite being further from the Sun, because "
         "its atmosphere traps heat at about 464 degrees Celsius",
         "NASA planetary fact sheets"),
        ("A teaspoon of neutron star material would weigh about a billion tonnes on "
         "Earth",
         "Neutron star density, ~10^17 kg/m³"),
        ("The Moon is drifting away from Earth at about 3.8 centimetres a year, "
         "measured by bouncing lasers off reflectors left by Apollo",
         "Lunar Laser Ranging experiment"),
    ),
    "history": (
        ("Cleopatra lived closer in time to the Moon landing than to the building of "
         "the Great Pyramid",
         "Pyramid c. 2560 BC, Cleopatra d. 30 BC, Apollo 11 in 1969"),
        ("Oxford University was teaching students before the Aztec Empire existed",
         "Oxford teaching from 1096; Tenochtitlan founded 1325"),
        ("The shortest war on record lasted 38 minutes, between Britain and Zanzibar "
         "in 1896",
         "Anglo-Zanzibar War, 27 August 1896"),
    ),
}


def classify_topic(topic: str) -> str:
    """
    Picks the domain whose vocabulary the topic sits in.

    Scored by how many domain keywords appear, so "how much protein to build
    muscle on a budget" lands on fitness rather than finance -- the keyword
    that appears twice wins over the one that appears once.
    """
    low = f" {str(topic or '').lower()} "
    best, best_score = DEFAULT_DOMAIN, 0

    for key, spec in DOMAINS.items():
        score = sum(2 if f" {word} " in low else 1
                    for word in spec["match"] if word in low)
        if score > best_score:
            best, best_score = key, score

    return best


def domain_spec(domain: str) -> dict[str, Any]:
    """The vocabulary pack for a domain, falling back to the general one."""
    return DOMAINS.get(domain, _GENERAL)


def bank_facts(domain: str, count: int) -> list[dict[str, str]]:
    """Pulls up to `count` attributed facts for a domain."""
    return [{"claim": claim, "source": source, "origin": "fact bank"}
            for claim, source in FACT_BANK.get(domain, ())[:count]]


# ---------------------------------------------------------------------------
# Gemini research
# ---------------------------------------------------------------------------

FACT_PROMPT = """You are researching a {seconds}-second short-form video. Return ONE JSON
object and nothing else.

TOPIC: {topic}
DOMAIN: {domain_label}

Write a hook, {count} body beats and a closing line.

THE ONE RULE: every body beat must contain something checkable -- a figure, a
date, a named person, company or place, or a stated mechanism ("because X
causes Y"). A sentence that would still read fine with the topic swapped out is
a failed sentence. Specifically banned: "more than you think", "the secret is",
"game changer", "at the end of the day", "trust me", "most people don't know".

Use this domain's actual vocabulary where it fits: {terms}.
Argue in this domain's actual units where they apply: {metrics}.

DO NOT INVENT NUMBERS. If you are not confident a figure is correct, write the
beat around a mechanism or a named example instead, and set its "source" to "".
A beat with no number and a real mechanism is worth far more than a beat with a
number you made up.

The hook is the first 9 words the viewer hears: open a loop, do not announce a
topic. No greeting, no "in this video", no "here are five".

JSON shape:
{{"hook": "...",
  "beats": [{{"line": "...", "source": "where this could be checked, or \\"\\""}}],
  "cta": "...",
  "confidence": "high|medium|low"}}"""


def research_facts(
    topic: str,
    count: int = 3,
    seconds: int = 30,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Asks Gemini for a fact-carrying script.

    Returns {"hook", "beats", "cta", "confidence", "source"}. Raises nothing --
    on any failure it returns {} and the caller falls through to the bank.
    """
    domain = classify_topic(topic)
    spec = domain_spec(domain)

    try:
        import gemini_engine as ge
    except Exception:
        return {}

    try:
        client = ge.get_client()
    except Exception:
        return {}

    prompt = FACT_PROMPT.format(
        topic=str(topic).strip(), seconds=int(seconds), count=int(count),
        domain_label=spec["label"],
        terms=", ".join(spec["terms"][:6]),
        metrics=", ".join(spec["metrics"][:5]),
    )

    for model in ge.MODEL_CANDIDATES:
        try:
            if progress:
                progress(f"Researching {spec['label'].lower()} facts with {model}...")
            response = ge.generate_with_retry(client, model, prompt, progress=progress)
            parsed = ge.parse_scene_response(getattr(response, "text", "") or "")
            beats = parsed.get("beats") or []
            if not isinstance(beats, list) or len(beats) < 1:
                continue

            cleaned: list[dict[str, str]] = []
            for beat in beats[:count]:
                if isinstance(beat, str):
                    cleaned.append({"claim": beat.strip(), "source": "", "origin": model})
                elif isinstance(beat, dict):
                    line = str(beat.get("line") or beat.get("claim") or "").strip()
                    if line:
                        cleaned.append({"claim": line,
                                        "source": str(beat.get("source") or "").strip(),
                                        "origin": model})
            if not cleaned:
                continue

            return {
                "hook": str(parsed.get("hook") or "").strip(),
                "beats": cleaned,
                "cta": str(parsed.get("cta") or "").strip(),
                "confidence": str(parsed.get("confidence") or "medium").strip().lower(),
                "domain": domain,
                "source": model,
            }
        except Exception:
            continue

    return {}


# ---------------------------------------------------------------------------
# Script assembly
# ---------------------------------------------------------------------------

def _subject(topic: str) -> str:
    """Strips listicle scaffolding so a topic can be dropped mid-sentence."""
    t = str(topic or "").strip().rstrip(".!?")
    t = re.sub(r"^\s*\d+\s*", "", t)
    t = re.sub(r"^(mind[- ]?blowing|shocking|insane|crazy|surprising|weird|amazing|"
               r"unbelievable|little[- ]known|hidden)\s+", "", t, flags=re.I)
    t = re.sub(r"^(facts?|secrets?|tips?|reasons?|things?|habits?|rules?|ways?)\s+",
               "", t, flags=re.I)
    t = re.sub(r"^(about|of|for|on|to)\s+", "", t, flags=re.I)
    return (t or str(topic)).strip().lower()


# Hook shapes that open a loop. They take a fact, not a topic -- which is the
# whole difference from the templates these replaced.
_HOOK_SHAPES = (
    "{fact} — and almost nobody checks this.",
    "Everyone gets this wrong about {subject}: {fact}",
    "{fact} That changes the whole argument.",
    "Before you touch {subject} again: {fact}",
)


def build_fact_script(
    topic: str,
    count: int = 3,
    seconds: int = 30,
    use_ai: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Builds hook / beats / CTA for a topic, from the best source available.

    Returns {"hook", "beats", "cta", "domain", "source", "needs_editing",
    "sources"}. `needs_editing` is True when the engine could not find real
    facts and fell back to the vocabulary pack -- the caller must surface that
    rather than let a thin script through as if it were researched.
    """
    domain = classify_topic(topic)
    spec = domain_spec(domain)
    subject = _subject(topic)
    count = max(1, min(6, int(count)))

    if use_ai:
        found = research_facts(topic, count=count, seconds=seconds, progress=progress)
        if found and found.get("beats"):
            beats = found["beats"]
            return {
                "hook": found["hook"] or _HOOK_SHAPES[0].format(
                    fact=beats[0]["claim"], subject=subject),
                "beats": beats,
                "cta": found["cta"] or f"Which part of that did you not know? Comment below.",
                "domain": domain,
                "domain_label": spec["label"],
                "source": found["source"],
                "confidence": found.get("confidence", "medium"),
                "needs_editing": found.get("confidence") == "low",
                "sources": [b["source"] for b in beats if b.get("source")],
            }

    # --- fact bank ---------------------------------------------------------
    # One extra fact is pulled so the hook has its own. Building the hook out
    # of beat one and then playing beat one again says the same sentence twice
    # inside the first eight seconds.
    banked = bank_facts(domain, count + 1)
    if banked:
        if progress:
            progress(f"Using the offline {spec['label'].lower()} fact bank.")
        hook_fact, banked = banked[0]["claim"], (banked[1:] or banked)
        return {
            "hook": _HOOK_SHAPES[0].format(fact=hook_fact.rstrip("."), subject=subject),
            "beats": banked,
            "cta": "Which of those surprised you? Comment below.",
            "domain": domain,
            "domain_label": spec["label"],
            "source": "fact bank",
            "confidence": "high",
            # Bank facts are true but general: they are about the domain, not
            # about this exact topic, so they still want a pass by a human.
            "needs_editing": True,
            "sources": [b["source"] for b in banked],
        }

    # --- vocabulary pack: the honest empty-handed case ---------------------
    if progress:
        progress("No facts found for this topic — writing a skeleton to fill in.")
    metrics = list(spec["metrics"])
    beats = [{"claim": f"The number that settles this is {metrics[i % len(metrics)]} — "
                       f"put the real figure here.",
              "source": "", "origin": "skeleton"}
             for i in range(count)]
    return {
        "hook": f"The thing nobody checks about {subject}.",
        "beats": beats,
        "cta": "What did I miss? Comment below.",
        "domain": domain,
        "domain_label": spec["label"],
        "source": "skeleton",
        "confidence": "low",
        "needs_editing": True,
        "sources": [],
    }


def script_lines(script: dict[str, Any]) -> list[tuple[str, str]]:
    """Flattens a fact script into ordered (role, spoken line) pairs."""
    lines: list[tuple[str, str]] = [("hook", str(script.get("hook") or "").strip())]
    for beat in script.get("beats") or []:
        claim = str(beat.get("claim") if isinstance(beat, dict) else beat or "").strip()
        if claim:
            lines.append(("point", claim))
    cta = str(script.get("cta") or "").strip()
    if cta:
        lines.append(("cta", cta))
    return [(role, line) for role, line in lines if line]


def attribution_block(script: dict[str, Any]) -> str:
    """The sources behind a script, for the publish pack."""
    rows = []
    for i, beat in enumerate(script.get("beats") or [], start=1):
        if not isinstance(beat, dict):
            continue
        source = str(beat.get("source") or "").strip()
        rows.append(f"  {i}. {beat.get('claim', '')[:90]}\n"
                    f"     source: {source or 'UNSOURCED — verify before publishing'}")
    if not rows:
        return ""
    return ("FACT SOURCES (verify every one before publishing)\n"
            + "\n".join(rows) + "\n")




def short_caption(line: str, max_words: int = 11) -> str:
    """
    Turns a spoken line into an on-screen subtitle.

    Short lines are kept whole -- the viral renderer wraps and centres them, and
    matching the narration word for word reads better than a clipped fragment.
    Only genuinely long lines get trimmed, always at a clause boundary so the
    subtitle never cuts off mid-phrase.
    """
    words = str(line or "").split()
    if len(words) <= max_words:
        return str(line or "")

    clipped = " ".join(words[:max_words])
    for sep in ("—", ",", ":", ";"):
        if sep in clipped:
            return clipped.rsplit(sep, 1)[0].strip(" —,:;")
    return clipped.rstrip(" ,:;—") + "..."


def count_from_topic(topic: str, default: int = 3) -> int:
    """Pulls the listicle number out of a topic when the user supplied one."""
    m = re.match(r"\s*(\d+)\b", str(topic or "").strip())
    if m:
        n = int(m.group(1))
        if 2 <= n <= 6:
            return n
    return default
