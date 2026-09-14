# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
"""
Minimalist Motion -- a standalone 2D vector animation engine.

Draws native 1080x1920 shorts from code alone: no source footage, no stock, no
model-generated imagery. White strokes on pure black, the look those faceless
philosophy/discipline channels run on.

Why PIL and not manim: manim needs a working LaTeX install and a Cairo build,
neither of which is here, and a renderer that only works on a machine with the
right system binaries is not a renderer. Everything below runs on Pillow and
numpy, which the app already depends on. Anti-aliasing comes from drawing at
2x and box-filtering down; the glow comes from an additive blur of the frame,
which is exactly right when the background is pure black.

The whole surface is deterministic and offline -- Gemini only ever supplies
numbers, and `fallback_scene_spec` covers the case where it supplies nothing.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

import vector_rig as rig
from paths import resolve_font
from video_engine import (_SCRATCH_RENDERS, purge_scratch_renders,
                          video_encoder, write_clip)

# ---------------------------------------------------------------------------
# Look
# ---------------------------------------------------------------------------

CANVAS: tuple[int, int] = (1080, 1920)
SUPERSAMPLE = 2                       # draw at 2x, box-filter down: free AA

# The bottom band every vertical platform covers with its own chrome: the
# caption and @handle on TikTok, the description and audio row on Reels, the
# title and progress scrubber on Shorts. 200px of a 1920 frame is a little over
# 10%, which clears all three.
#
# Enforced in Frame.text and Frame.wrapped rather than left to each template to
# remember, because it was not remembered: measured across all fifteen
# templates before this existed, every single one put type inside the band --
# the payoff line sat at y=1720 and the progress hairline at y=1866, which is
# underneath TikTok's caption.
#
# Geometry is deliberately *not* clamped. A staircase reaching the floor is the
# picture; a sentence hidden behind a caption is a bug.
# The action rail on Shorts and TikTok -- like, comment, share, sound -- plus
# the caption and the progress scrubber. Measured against the taller of the
# two: 220px of a 1920 frame is 11.5%, which clears the rail on a 20:9 phone
# rather than only on a 16:9 one.
SAFE_BOTTOM = 220
SAFE_Y: float = CANVAS[1] - SAFE_BOTTOM

# Type stops 30px higher again, which is the room the progress hairline needs.
TEXT_SAFE_Y: float = SAFE_Y - 30.0

# How far ink actually falls below a centred text anchor, as a fraction of the
# nominal size. The clamps used 0.62, which is roughly a baseline-to-descender
# figure and not what these anchors measure: text is drawn centred, so the ink
# reaches about a line-height's half plus the descender. Measured on the
# rendered frames, the payoff line at size 52 put ink 36px below the clamped
# block bottom against the 32px the old figure reserved, and six of those
# pixels landed inside the action rail. 0.85 leaves margin at every size the
# templates use.
_INK_BELOW_ANCHOR = 0.85

BLACK: tuple[int, int, int] = (0, 0, 0)
WHITE: tuple[int, int, int] = (255, 255, 255)
GREY: tuple[int, int, int] = (136, 136, 136)          # #888888
DIM: tuple[int, int, int] = (44, 44, 44)              # unreached path, ticks
# The glow is three blurs, not one.
#
# A single Gaussian is either a tight rim or a wide wash and cannot be both,
# which is what makes one-pass bloom look like a filter. Real light falls off
# on more than one scale: a bright core within a few pixels of the stroke, a
# soft body out to ~40px, and a faint atmosphere well beyond it. Summing three
# passes at different radii gives that falloff, and because each is computed at
# a fraction of the frame size the three together cost less than the one
# full-resolution blur did.
#
# (radius in output pixels, weight, resolution divisor)
GLOW_PASSES: tuple[tuple[float, float, int], ...] = (
    (10.0, 0.85, 2),      # core: tight and bright, hugs the line
    (30.0, 0.55, 4),      # body: the halo you actually read as glow
    (90.0, 0.30, 8),      # atmosphere: lifts the black around a lit region
)
GLOW_STRENGTH = 1.0       # master multiplier over the three passes above
_GLOW_LUTS = [[min(255, int(v * weight)) for v in range(256)] * 3
              for _, weight, _ in GLOW_PASSES]

# One reusable drawing surface per thread. Clearing a 2160x3840 buffer costs
# 2.4ms against 14ms to allocate a fresh one, and a render draws hundreds.
_SURFACES = threading.local()


def _surface(width: int, height: int) -> Image.Image:
    """A cleared drawing surface. Only one may be live per thread at a time."""
    existing = getattr(_SURFACES, "image", None)
    if existing is not None and existing.size == (width, height):
        existing.paste(BLACK, (0, 0, width, height))
        return existing
    fresh = Image.new("RGB", (width, height), BLACK)
    _SURFACES.image = fresh
    return fresh

# Clean geometric sans, heaviest weight first: the faces those channels use,
# then the Windows defaults, then what a Debian container actually ships.
# Resolved through paths.resolve_font, which searches assets/fonts/ first --
# drop a .ttf in there and it wins on every platform.
_BOLD_FACES = (
    "Montserrat-ExtraBold.ttf", "Montserrat-Bold.ttf",
    "Inter-Bold.ttf", "Inter_18pt-Bold.ttf",
    "seguibl.ttf", "arialbd.ttf", "segoeuib.ttf",
    "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.ttf",
)
_LIGHT_FACES = (
    "Montserrat-Regular.ttf", "Montserrat-Medium.ttf",
    "Inter-Regular.ttf", "Inter_18pt-Regular.ttf",
    "segoeui.ttf", "arial.ttf",
    "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "FreeSans.ttf",
)

_FONT_CACHE: dict[tuple[str, int], Any] = {}


def load_face(size: int, weight: str = "bold") -> Any:
    """Loads (and caches) a display face. Cached because it is hit per frame."""
    key = (weight, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    path = resolve_font(_BOLD_FACES if weight == "bold" else _LIGHT_FACES)
    if path:
        try:
            font = ImageFont.truetype(path, size)
            _FONT_CACHE[key] = font
            return font
        except (IOError, OSError):
            pass
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def active_face_name() -> str:
    """Which face the engine actually found, for the UI to report honestly."""
    path = resolve_font(_BOLD_FACES)
    return os.path.basename(path) if path else "PIL default (no TTF found)"


# ---------------------------------------------------------------------------
# Easing and geometry
# ---------------------------------------------------------------------------

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def ease_in_out(t: float) -> float:
    t = clamp(t)
    return 3 * t * t - 2 * t * t * t


def ease_out(t: float) -> float:
    t = clamp(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in(t: float) -> float:
    t = clamp(t)
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """Arrives fast, settles slowly. The default for anything that travels."""
    t = clamp(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in_cubic(t: float) -> float:
    """Creeps, then goes. Gravity, and anything being pulled in."""
    t = clamp(t)
    return t ** 3


def rush_into(t: float, bite: float = 4.5) -> float:
    """
    Hangs back, then covers the last third almost at once.

    Sharper than ease_in_cubic: for the beat before an impact, where the
    picture should feel like it is being yanked rather than falling.
    """
    t = clamp(t)
    return (math.exp(bite * t) - 1.0) / (math.exp(bite) - 1.0)


def overshoot(t: float, amount: float = 1.7) -> float:
    """Goes past the mark and comes back -- weight, for anything that lands."""
    t = clamp(t)
    p = t - 1.0
    return p * p * ((amount + 1) * p + amount) + 1.0


def fade(t: float, start: float, attack: float = 0.45, hold: float = 1e9,
         release: float = 0.45) -> float:
    """Opacity envelope for an element that appears at `start` and leaves after `hold`."""
    if t < start:
        return 0.0
    up = ease_out((t - start) / max(attack, 1e-6))
    if t < start + hold:
        return up
    return up * (1.0 - ease_in((t - start - hold) / max(release, 1e-6)))


def catmull_rom(points: Sequence[tuple[float, float]], per_span: int = 24) -> list[tuple[float, float]]:
    """
    Smooths a control polygon into a curve that passes through every point.

    Bezier control points would need the caller to supply handles; Catmull-Rom
    only needs the points themselves, which is what Gemini can reliably emit.
    """
    pts: list[tuple[float, float]] = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 3:
        return pts

    padded = [pts[0]] + pts + [pts[-1]]
    out: list[tuple[float, float]] = []
    for i in range(len(padded) - 3):
        p0, p1, p2, p3 = padded[i:i + 4]
        for step in range(per_span):
            u = step / per_span
            u2, u3 = u * u, u * u * u
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * u
                       + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * u2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * u3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * u
                       + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * u2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * u3)
            out.append((x, y))
    out.append(pts[-1])
    return out


def arc_lengths(points: Sequence[tuple[float, float]]) -> list[float]:
    """Cumulative distance along a polyline, normalised to 0..1."""
    if len(points) < 2:
        return [0.0] * len(points)
    run = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        run.append(run[-1] + math.hypot(dx, dy))
    total = run[-1] or 1.0
    return [d / total for d in run]


def point_at(points: Sequence[tuple[float, float]], lengths: Sequence[float],
             u: float) -> tuple[float, float]:
    """Position at fraction `u` of the path, by arc length so speed stays even."""
    u = clamp(u)
    for i in range(1, len(lengths)):
        if lengths[i] >= u:
            span = lengths[i] - lengths[i - 1] or 1.0
            local = (u - lengths[i - 1]) / span
            x = points[i - 1][0] + (points[i][0] - points[i - 1][0]) * local
            y = points[i - 1][1] + (points[i][1] - points[i - 1][1]) * local
            return (x, y)
    last = points[-1]
    return (float(last[0]), float(last[1]))


def slice_to(points: Sequence[tuple[float, float]], lengths: Sequence[float],
             u: float) -> list[tuple[float, float]]:
    """The travelled part of a path, for progressive reveals."""
    u = clamp(u)
    out = [points[0]]
    for i in range(1, len(lengths)):
        if lengths[i] >= u:
            out.append(point_at(points, lengths, u))
            break
        out.append(points[i])
    return out


def mix(colour: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    """Scales a colour toward black -- the only fade that works on this palette."""
    a = clamp(amount)
    return (int(colour[0] * a), int(colour[1] * a), int(colour[2] * a))


# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------

class Frame:
    """
    One supersampled drawing surface, in 1080x1920 coordinates.

    Every method takes canvas-space numbers; the 2x scaling is internal, so a
    template never has to think about it.
    """

    def __init__(self, size: tuple[int, int] = CANVAS, ss: int = SUPERSAMPLE,
                 reuse: bool = False) -> None:
        self.w, self.h = size
        self.ss = ss
        # `reuse` recycles the thread's drawing buffer. Only the render loop
        # sets it, because it provably holds one frame at a time; anything
        # holding two Frames at once must allocate.
        self.image = (_surface(self.w * ss, self.h * ss) if reuse
                      else Image.new("RGB", (self.w * ss, self.h * ss), BLACK))
        self.draw = ImageDraw.Draw(self.image)

    # -- primitives --------------------------------------------------------
    def polyline(self, points: Sequence[tuple[float, float]],
                 colour: tuple[int, int, int] = WHITE, width: float = 6.0) -> None:
        if len(points) < 2:
            return
        s = self.ss
        self.draw.line([(p[0] * s, p[1] * s) for p in points],
                       fill=colour, width=max(1, int(width * s)), joint="curve")

    def line(self, a: tuple[float, float], b: tuple[float, float],
             colour: tuple[int, int, int] = WHITE, width: float = 6.0) -> None:
        self.polyline([a, b], colour, width)

    def circle(self, centre: tuple[float, float], radius: float,
               colour: tuple[int, int, int] = WHITE, width: float = 0.0) -> None:
        s = self.ss
        cx, cy, r = centre[0] * s, centre[1] * s, radius * s
        box = (cx - r, cy - r, cx + r, cy + r)
        if width <= 0:
            self.draw.ellipse(box, fill=colour)
        else:
            self.draw.ellipse(box, outline=colour, width=max(1, int(width * s)))

    def ellipse(self, centre: tuple[float, float], rx: float, ry: float,
                colour: tuple[int, int, int] = WHITE, width: float = 0.0) -> None:
        """An axis-aligned ellipse -- the funnel rings are circles in perspective."""
        s = self.ss
        cx, cy = centre[0] * s, centre[1] * s
        box = (cx - rx * s, cy - ry * s, cx + rx * s, cy + ry * s)
        if width <= 0:
            self.draw.ellipse(box, fill=colour)
        else:
            self.draw.ellipse(box, outline=colour, width=max(1, int(width * s)))

    def rect(self, box: tuple[float, float, float, float],
             colour: tuple[int, int, int] = WHITE, width: float = 0.0,
             radius: float = 0.0) -> None:
        s = self.ss
        scaled = (box[0] * s, box[1] * s, box[2] * s, box[3] * s)
        if radius > 0:
            if width <= 0:
                self.draw.rounded_rectangle(scaled, radius=radius * s, fill=colour)
            else:
                self.draw.rounded_rectangle(scaled, radius=radius * s, outline=colour,
                                            width=max(1, int(width * s)))
        elif width <= 0:
            self.draw.rectangle(scaled, fill=colour)
        else:
            self.draw.rectangle(scaled, outline=colour, width=max(1, int(width * s)))

    def polygon(self, points: Sequence[tuple[float, float]],
                colour: tuple[int, int, int] = WHITE) -> None:
        s = self.ss
        self.draw.polygon([(p[0] * s, p[1] * s) for p in points], fill=colour)

    def safe_y(self, y: float, size: float) -> float:
        """
        Lifts a text baseline out of the platform's bottom chrome.

        0.62 of the point size is the descender-inclusive half-height for a
        vertically centred line, which is the anchor every caller here uses.
        """
        return min(float(y), TEXT_SAFE_Y - size * _INK_BELOW_ANCHOR)

    def text(self, body: str, centre: tuple[float, float], size: int = 56,
             colour: tuple[int, int, int] = WHITE, weight: str = "bold",
             anchor: str = "mm", tracking: float = 0.0, clamp_safe: bool = True) -> None:
        """Draws one line. `tracking` spaces the letters, which reads as premium."""
        if not body:
            return
        s = self.ss
        font = load_face(int(size * s), weight)
        y = self.safe_y(centre[1], size) if clamp_safe else centre[1]

        if tracking <= 0:
            self.draw.text((centre[0] * s, y * s), body, font=font,
                           fill=colour, anchor=anchor)
            return

        gap = tracking * s
        widths = [self.draw.textlength(ch, font=font) for ch in body]
        total = sum(widths) + gap * (len(body) - 1)
        x = centre[0] * s - total / 2
        for ch, w in zip(body, widths):
            self.draw.text((x, y * s), ch, font=font, fill=colour, anchor="lm")
            x += w + gap

    def wrapped(self, body: str, centre: tuple[float, float], size: int = 44,
                colour: tuple[int, int, int] = WHITE, weight: str = "light",
                max_width: float = 860.0, leading: float = 1.28) -> None:
        """Centre-wraps a sentence around `centre`."""
        if not body:
            return
        font = load_face(int(size * self.ss), weight)
        limit = max_width * self.ss

        lines: list[str] = []
        current = ""
        for word in body.split():
            trial = f"{current} {word}".strip()
            if self.draw.textlength(trial, font=font) <= limit or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

        step = size * leading
        top = centre[1] - step * (len(lines) - 1) / 2
        # Clamp the block, not each line: lifting only the last line would
        # close the leading and the paragraph would read as a typo.
        overflow = (top + step * (len(lines) - 1)
                    + size * _INK_BELOW_ANCHOR) - TEXT_SAFE_Y
        if overflow > 0:
            top -= overflow
        for i, line in enumerate(lines):
            self.text(line, (centre[0], top + i * step), size, colour, weight,
                      clamp_safe=False)

    # -- output ------------------------------------------------------------
    def finish(self, glow: float = GLOW_STRENGTH) -> np.ndarray:
        """Box-filters the supersampled canvas down and adds the multi-pass bloom."""
        # reduce() is a dedicated box reduction: 14ms against 48ms for a
        # general resize at the same quality for an integer factor.
        sharp = self.image.reduce(self.ss) if self.ss > 1 else self.image
        if glow <= 0:
            return np.asarray(sharp, dtype=np.uint8)

        # The three passes are summed at half resolution and upscaled once.
        # Doing the LUT and the add at full size per pass costs three 6.2M-pixel
        # upscales, three full-res LUTs and three full-res adds -- 137ms a frame
        # against 47ms this way, for a halo that is soft enough that half
        # resolution is indistinguishable.
        half = (self.w // 2, self.h // 2)
        halo: Image.Image | None = None
        for index, (radius, weight, divisor) in enumerate(GLOW_PASSES):
            blurred = (sharp.reduce(divisor)
                       .filter(ImageFilter.GaussianBlur(radius=radius / divisor)))
            if blurred.size != half:
                blurred = blurred.resize(half, Image.Resampling.BILINEAR)
            lut = (_GLOW_LUTS[index] if glow == GLOW_STRENGTH
                   else [min(255, int(v * weight * glow)) for v in range(256)] * 3)
            scaled = blurred.point(lut)
            # Saturating uint8 add in C beats the same sum in float32 numpy.
            halo = scaled if halo is None else ImageChops.add(halo, scaled)

        if halo is None:
            return np.asarray(sharp, dtype=np.uint8)
        full = halo.resize((self.w, self.h), Image.Resampling.BILINEAR)
        return np.asarray(ImageChops.add(sharp, full), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Animation phases
#
# Every template runs the same three-beat structure, so the sound design and
# the picture cannot drift apart: the drop is placed at `impact`, and `impact`
# is where the picture pays off.
#
#   1. draw    the geometry writes itself on -- a trim-path from 0% to 100%
#   2. travel  the object moves along it, heavy cubic easing
#   3. impact  the milestone: the ball clears, the jar fills, the wall falls
#   4. settle  the closing line lands and the frame holds
# ---------------------------------------------------------------------------

LEAD_IN = 1.1               # titles breathe before anything moves
SETTLE_OUT = 2.2            # the closing beat


class Phases:
    """Where each beat starts and ends, in seconds, for one scene."""

    __slots__ = ("lead", "draw_end", "impact", "end", "duration")

    def __init__(self, duration: float, climax: float,
                 draw_end: float | None = None) -> None:
        self.duration = max(duration, 1.0)
        self.lead = min(LEAD_IN, self.duration * 0.12)
        self.end = max(self.lead + 0.5, self.duration - SETTLE_OUT)
        self.impact = min(max(climax, self.lead + 0.4), self.duration - 0.3)
        # The path finishes drawing before the object gets there, so the
        # object always has somewhere left to travel to. A model-supplied
        # draw_end is honoured, but never past the impact -- geometry still
        # being drawn when the beat lands reads as a stall.
        default = self.lead + (self.impact - self.lead) * 0.55
        self.draw_end = (min(max(draw_end, self.lead + 0.2), self.impact - 0.15)
                         if draw_end else default)

    def draw(self, t: float) -> float:
        """0..1 across the trim-path beat."""
        return ease_in_out(clamp((t - self.lead) / max(self.draw_end - self.lead, 1e-6)))

    def travel(self, t: float, easing: Callable[[float], float] | None = None) -> float:
        """0..1 from the lead-in to the impact, eased hard by default."""
        raw = clamp((t - self.lead) / max(self.impact - self.lead, 1e-6))
        return (easing or ease_out_cubic)(raw)

    def after(self, t: float, span: float = 1.0) -> float:
        """0..1 in the `span` seconds following the impact."""
        return clamp((t - self.impact) / max(span, 1e-6))

    def tail(self, t: float) -> float:
        """0..1 across the settle beat."""
        return clamp((t - self.end) / max(self.duration - self.end, 1e-6))


def phases_of(spec: dict[str, Any], duration: float) -> Phases:
    return Phases(duration, float(spec.get("climax") or duration * 0.75),
                  float(spec.get("draw_end") or 0.0) or None)


# ---------------------------------------------------------------------------
# Ambient layer
#
# A frame of pure geometry on pure black reads as a diagram. What makes these
# channels feel alive is what is happening *behind* the subject: a grid that
# breathes, dust drifting through the light. It costs a few milliseconds and
# it is most of the difference between a chart and a film.
# ---------------------------------------------------------------------------

GRID_SPACING = 135.0
GRID_COLOUR = (17, 17, 17)
_DRIFT_SEED = 20260910


def draw_grid(frame: Frame, t: float, phases: Phases, strength: float = 1.0) -> None:
    """A faint lattice that pulses outward from the impact."""
    if strength <= 0:
        return

    # The pulse: a ring of brightness expanding from the centre when the beat
    # lands, so the drop is felt in the background as well as heard.
    since = t - phases.impact
    pulse = math.exp(-2.6 * since) if 0.0 <= since < 1.6 else 0.0
    breathe = 0.72 + 0.28 * math.sin(2 * math.pi * 0.09 * t)
    cx, cy = frame.w / 2, frame.h * 0.52

    columns = int(frame.w / GRID_SPACING) + 2
    rows = int(frame.h / GRID_SPACING) + 2

    for i in range(columns):
        x = i * GRID_SPACING
        lift = pulse * math.exp(-((x - cx) / 420.0) ** 2) if pulse else 0.0
        level = strength * breathe * (1.0 + 5.0 * lift)
        frame.line((x, 0), (x, frame.h), mix(GRID_COLOUR, level), 2)
    for j in range(rows):
        y = j * GRID_SPACING
        lift = pulse * math.exp(-((y - cy) / 520.0) ** 2) if pulse else 0.0
        level = strength * breathe * (1.0 + 5.0 * lift)
        frame.line((0, y), (frame.w, y), mix(GRID_COLOUR, level), 2)


def draw_drift(frame: Frame, t: float, count: int = 26, strength: float = 1.0) -> None:
    """
    Slow motes rising through the frame.

    Positions come from a fixed hash of the index rather than an RNG, so frame
    N looks the same however it was reached -- a renderer that draws frames out
    of order must not produce a different film.
    """
    if strength <= 0:
        return
    for i in range(count):
        seed = (i * 2654435761 + _DRIFT_SEED) & 0xFFFFFFFF
        x = (seed % 10007) / 10007.0
        speed = 0.018 + ((seed >> 8) % 100) / 3400.0
        phase = ((seed >> 16) % 1000) / 1000.0
        size = 2.0 + ((seed >> 24) % 5) * 0.8

        y = ((phase + t * speed) % 1.0)
        # Fade in and out at the edges so nothing pops into existence.
        alpha = math.sin(math.pi * y) ** 0.7
        wobble = math.sin(t * 0.5 + i) * 14.0
        frame.circle((x * frame.w + wobble, (1.0 - y) * frame.h), size,
                     mix((92, 92, 92), alpha * 0.55 * strength))


# The words each template writes onto its own geometry. Hardcoding these was
# the other half of the sameness problem: the shape adapted to the topic and
# then labelled itself "5 YEARS / 50 YEARS" regardless. Gemini fills the slots;
# the defaults here are what a template says when nobody supplied anything.
LABEL_SLOTS: dict[str, dict[str, str]] = {
    "split_path": {"near": "5 YEARS", "far": "50 YEARS", "easy": "COMFORT NOW"},
    "compounding_jar": {"unit": "DAY", "meter": "x"},
    "staircase_progress": {"before": "RESISTANCE", "after": "LEVERAGE",
                           "left": "EFFORT", "right": "REWARD"},
    "balance_scale": {"left": "NOW", "right": "LATER"},
    "gravity_funnel": {"pull": "PULL"},
    "domino_chain": {"first": "ONE PUSH", "last": "EVERYTHING"},
    "comparison_split": {"tier1": "BASIC", "tier2": "HARD", "tier3": "SMART"},
    "steep_staircase": {"stage1": "DAY 1", "stage2": "WEEK 1", "stage3": "MONTH 1",
                        "stage4": "YEAR 1", "stage5": "YEAR 5", "summit": "MASTERY"},
    "delusion_mirror": {"real": "WHO YOU ARE", "imagined": "WHO YOU THINK",
                        "gap": "THE GAP"},
    "chain_anchor": {"anchor1": "EXCUSES", "anchor2": "FEAR", "freed": "FREE"},
    "growth_consistency": {"input": "SAME EFFORT", "output": "COMPOUNDED"},
    "sisyphus_boulder": {"mark1": "WEEK 1", "mark2": "MONTH 1", "mark3": "YEAR 1",
                         "mark4": "YEAR 3", "slope": "THE GRIND", "summit": "THE TOP"},
    "discipline_iceberg": {"above": "THE RESULT", "below": "THE WORK NOBODY SAW"},
    "two_doors": {"left": "COMFORT", "right": "THE HARD ONE", "through": "GO"},
    "custom": {},
}

# Long labels break the composition they sit in, so they are capped rather
# than wrapped -- these are stamps on a diagram, not sentences. A few slots are
# drawn through `wrapped()` and can take a short phrase, so they pass wrap=True
# and get a longer allowance.
LABEL_MAX_CHARS = 16
LABEL_WRAP_CHARS = 44


def label(spec: dict[str, Any], slot: str, fallback: str = "",
          wrap: bool = False) -> str:
    """The text for one labelled slot, from the spec or the template default."""
    limit = LABEL_WRAP_CHARS if wrap else LABEL_MAX_CHARS
    supplied = spec.get("labels")
    if isinstance(supplied, dict):
        value = str(supplied.get(slot) or "").strip()
        if value:
            return value.upper()[:limit]
    defaults = LABEL_SLOTS.get(str(spec.get("template") or ""), {})
    return (fallback or defaults.get(slot, "")).upper()[:limit]


def draw_ambient(frame: Frame, t: float, phases: Phases, spec: dict[str, Any]) -> None:
    """The background pass every template runs before its own geometry."""
    level = float(spec.get("ambient", 1.0) or 0.0)
    if level <= 0:
        return
    draw_grid(frame, t, phases, level)
    draw_drift(frame, t, strength=level)


# ---------------------------------------------------------------------------
# Shared furniture
# ---------------------------------------------------------------------------

def draw_titles(frame: Frame, spec: dict[str, Any], t: float) -> None:
    """Title and subtitle, fading in over the first beat."""
    alpha = fade(t, 0.15, attack=0.7)
    if alpha <= 0.01:
        return
    title = str(spec.get("title") or "").upper()
    subtitle = str(spec.get("subtitle") or "")
    frame.wrapped(title, (frame.w / 2, 300), size=72, colour=mix(WHITE, alpha),
                  weight="bold", max_width=900, leading=1.15)
    if subtitle:
        frame.wrapped(subtitle, (frame.w / 2, 430), size=38,
                      colour=mix(GREY, fade(t, 0.55, attack=0.7)), weight="light",
                      max_width=820)


def draw_footer(frame: Frame, spec: dict[str, Any], t: float, duration: float) -> None:
    """Closing line, held for the last beat so the point lands."""
    line = str(spec.get("payoff") or "").upper()
    if not line:
        return
    start = max(0.0, duration - 4.2)
    alpha = fade(t, start, attack=0.8)
    if alpha > 0.01:
        # Anchored to the safe line rather than to a fixed y. It used to ask
        # for a literal y and let the clamp bottom-align it, which worked while
        # SAFE_BOTTOM happened to be 200: raising the margin to 220 moved the
        # safe line up and left that literal below it, so the payoff bled six
        # pixels into the action rail. Measured across all fourteen
        # templates, bottom-aligning here leaves 3.8% ink behind the text --
        # a deliberate move further up collided with each template's own axis
        # labels at 28%.
        frame.wrapped(line, (frame.w / 2, TEXT_SAFE_Y), size=52,
                      colour=mix(WHITE, alpha), weight="bold",
                      max_width=900, leading=1.18)


# The hairline sits just inside the safe area, not at the foot of the frame,
# and below TEXT_SAFE_Y so a payoff line can never land on top of it.
PROGRESS_Y: float = SAFE_Y - 8.0


def draw_progress(frame: Frame, t: float, duration: float) -> None:
    """A hairline time bar. Retention furniture: it promises the video is short."""
    frame.line((90, PROGRESS_Y), (990, PROGRESS_Y), DIM, 4)
    done = clamp(t / max(duration, 1e-6))
    if done > 0:
        frame.line((90, PROGRESS_Y), (90 + 900 * done, PROGRESS_Y), GREY, 4)


# ---------------------------------------------------------------------------
# Template 1 -- The Curve of Pain vs. Comfort
# ---------------------------------------------------------------------------

_PAIN_CONTROL = [(120, 880), (250, 1120), (400, 1430), (560, 1330),
                 (700, 960), (850, 800), (990, 780)]
_EASY_CONTROL = [(120, 1020), (330, 1010), (520, 1015), (680, 1040),
                 (820, 1280), (900, 1480), (960, 1510)]


def _spikes(frame: Frame, x0: float, x1: float, base: float, height: float,
            count: int, colour: tuple[int, int, int]) -> None:
    step = (x1 - x0) / max(count, 1)
    for i in range(count):
        left = x0 + i * step
        frame.polygon([(left, base), (left + step / 2, base - height), (left + step, base)],
                      colour)


def scene_curve(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """A ball takes the steep drop and ends high; the flat path ends on spikes."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    pain = catmull_rom(_PAIN_CONTROL)
    easy = catmull_rom(_EASY_CONTROL)
    pain_len, easy_len = arc_lengths(pain), arc_lengths(easy)

    lead = 1.1                                  # titles breathe first
    span = max(duration - lead - 2.2, 1.0)
    u = ease_in_out(clamp((t - lead) / span))

    # Both routes are visible from the start, unlit -- the shape is the message.
    frame.polyline(pain, DIM, 7)
    frame.polyline(easy, mix(GREY, 0.30), 7)

    # The easy route runs ahead early, then falls off the end.
    easy_u = clamp(u * 1.22)
    frame.polyline(slice_to(easy, easy_len, easy_u), GREY, 7)
    ex, ey = point_at(easy, easy_len, easy_u)
    frame.circle((ex, ey), 20, GREY)

    _spikes(frame, 800, 1000, 1570, 90, 5, mix(GREY, 0.55 + 0.45 * clamp((u - 0.7) / 0.3)))

    # The hard route: lit as it is travelled.
    frame.polyline(slice_to(pain, pain_len, u), WHITE, 8)
    px, py = point_at(pain, pain_len, u)
    frame.circle((px, py), 26, WHITE)
    frame.circle((px, py), 44, mix(WHITE, 0.35), 3)

    # Labels, tied to where the ball actually is rather than to the clock.
    if u > 0.16:
        frame.text(label(spec, "near"), (430, 1500), 44,
                   mix(GREY, fade(t, lead + span * 0.16, 0.5)), tracking=3)
    if u > 0.66:
        frame.text(label(spec, "far"), (760, 690), 46,
                   mix(WHITE, fade(t, lead + span * 0.66, 0.5)), tracking=3)
    if easy_u > 0.78:
        frame.text(label(spec, "easy"), (312, 906), 36,
                   mix(GREY, fade(t, lead + span * 0.64, 0.5)), tracking=3)


# ---------------------------------------------------------------------------
# Template 2 -- The 1% Daily Vessel
# ---------------------------------------------------------------------------

_JAR = (330.0, 720.0, 750.0, 1440.0)         # left, top, right, bottom
_JAR_DAYS = 365


def scene_vessel(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """A jar filling on a 1.01^n curve: nothing, nothing, nothing, then everything."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    left, top, right, bottom = _JAR
    lead = 1.1
    span = max(duration - lead - 2.2, 1.0)
    u = clamp((t - lead) / span)

    day = int(u * _JAR_DAYS)
    growth = 1.01 ** day
    ceiling = 1.01 ** _JAR_DAYS
    level = clamp((growth - 1.0) / (ceiling - 1.0))

    # Fluid first, so the outline sits on top of it.
    if level > 0.002:
        surface = bottom - (bottom - top) * level
        body: list[tuple[float, float]] = []
        for i in range(41):
            x = left + (right - left) * (i / 40)
            wave = math.sin(i / 40 * math.pi * 3 + t * 2.4) * 6 * (0.4 + level)
            body.append((x, surface + wave))
        body += [(right, bottom), (left, bottom)]
        frame.polygon(body, WHITE)

    # Vessel outline and lip.
    frame.rect((left, top, right, bottom), WHITE, width=7, radius=34)
    frame.line((left - 26, top), (right + 26, top), WHITE, 7)

    # Tick marks: quarters of a year up the side.
    for i in range(1, 5):
        y = bottom - (bottom - top) * (i / 4)
        frame.line((right + 14, y), (right + 46, y), DIM if level < i / 4 else GREY, 4)
        frame.text(f"{i * 90}d", (right + 96, y), 26,
                   DIM if level < i / 4 else GREY, weight="light")

    # Compounding particles falling in, one per simulated fortnight.
    for k in range(14):
        phase = (t * 0.9 + k * 0.37) % 1.0
        if phase > 0.62:
            continue
        x = left + 46 + (k * 97) % max(1.0, (right - left - 92))
        y = top - 130 + phase / 0.62 * 130
        frame.circle((x, y), 7, mix(WHITE, 1.0 - phase / 0.62))

    frame.text(f'{label(spec, "unit")} {day}', (frame.w / 2, 1478), 38, GREY, tracking=5)
    frame.text(f"{growth:.2f}x", (frame.w / 2, 1552), 62,
               WHITE if level > 0.5 else GREY, tracking=2)


# ---------------------------------------------------------------------------
# Template 3 -- The Resistance Staircase / Leverage
# ---------------------------------------------------------------------------

_STEPS = 6
_STAIR_BOX = (110.0, 780.0, 900.0, 1420.0)


def _stair_points() -> list[tuple[float, float]]:
    left, top, right, bottom = _STAIR_BOX
    width = (right - left) / _STEPS
    rise = (bottom - top) / _STEPS
    pts: list[tuple[float, float]] = [(left, bottom)]
    for i in range(_STEPS):
        x = left + i * width
        y = bottom - i * rise
        pts.append((x, y))
        pts.append((x + width, y))
        pts.append((x + width, y - rise))
    return pts


def _stick_figure(frame: Frame, foot: tuple[float, float], lean: float,
                  colour: tuple[int, int, int], scale: float = 1.0) -> None:
    """A pushing figure: head, spine, braced legs, both arms forward."""
    x, y = foot
    h = 150 * scale
    hip = (x, y - h * 0.52)
    shoulder = (x + lean * 26, y - h * 0.92)
    frame.circle((shoulder[0] + lean * 8, shoulder[1] - h * 0.16), 20 * scale, colour)
    frame.line(hip, shoulder, colour, 7 * scale)
    frame.line(hip, (x - lean * 34, y), colour, 7 * scale)
    frame.line(hip, (x + lean * 12, y), colour, 7 * scale)
    frame.line(shoulder, (shoulder[0] + lean * 60, shoulder[1] + h * 0.10), colour, 7 * scale)


def scene_staircase(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """Linear effort, exponential reward -- and the moment the slope flips."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    stairs = _stair_points()
    lengths = arc_lengths(stairs)
    lead = 1.1
    span = max(duration - lead - 2.2, 1.0)
    u = clamp((t - lead) / span)

    climb = clamp(u / 0.78)                     # the last fifth is the payoff
    frame.polyline(stairs, DIM, 7)
    frame.polyline(slice_to(stairs, lengths, climb), WHITE, 8)

    # The phase caption steps aside once the closing line arrives.
    caption = fade(t, lead + span * 0.1, 0.6) * (1.0 - clamp((t - (duration - 4.6)) / 0.6))

    if u <= 0.80:
        bx, by = point_at(stairs, lengths, climb)
        frame.circle((bx, by - 34), 32, WHITE)
        _stick_figure(frame, (bx - 74, by), 1.0, GREY)
        frame.text(label(spec, "before"), (frame.w / 2, 700), 40, mix(GREY, caption), tracking=4)
    else:
        # Over the top: the sphere runs away downhill and the figure stands up.
        roll = ease_in(clamp((u - 0.80) / 0.20))
        top_x, top_y = stairs[-1]
        bx = top_x + roll * 90
        by = top_y - 34 + roll * roll * 320
        frame.line((top_x, top_y), (top_x + 130, top_y + 360), GREY, 6)
        frame.circle((bx, by), 32 + roll * 8, WHITE)
        _stick_figure(frame, (top_x - 96, top_y), 1.0, WHITE)
        frame.text(label(spec, "after"), (frame.w / 2, 700), 46,
                   mix(WHITE, fade(t, lead + span * 0.82, 0.5)), tracking=6)

    # Effort is linear; reward is not. The crossing is the whole argument.
    # The base sits 40px higher than it reads like it should: the payoff line
    # now bottom-aligns inside the safe area at ~1626, and these captions used
    # to be drawn straight through it.
    base, height = 1560.0, 140.0
    for i, (caption, value, colour) in enumerate((
        (label(spec, "left"), clamp(u), GREY),
        (label(spec, "right"), clamp(u ** 3.2), WHITE),
    )):
        x = 220 + i * 640
        frame.line((x, base), (x, base - height), DIM, 5)
        frame.line((x, base), (x, base - height * value), colour, 11)
        frame.text(caption, (x, base + 44), 26, colour, weight="light", tracking=3)


# ---------------------------------------------------------------------------
# Template 4 -- Custom vector, driven by Gemini's JSON
# ---------------------------------------------------------------------------

_COLOURS = {"white": WHITE, "grey": GREY, "gray": GREY, "dim": DIM}


def _colour_of(name: Any, alpha: float = 1.0) -> tuple[int, int, int]:
    return mix(_COLOURS.get(str(name or "white").lower(), WHITE), alpha)


def _denorm(point: Sequence[float], size: tuple[int, int]) -> tuple[float, float]:
    """Gemini works in 0..1; the canvas does not."""
    return (float(point[0]) * size[0], float(point[1]) * size[1])


def scene_custom(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """
    Renders an element list supplied as JSON.

    Every field is optional and every value is clamped, because this is model
    output: a missing "points" key must produce a plainer video, never a crash
    two hundred frames into a render.
    """
    draw_ambient(frame, t, phases_of(spec, duration), spec)

    size = (frame.w, frame.h)
    paths: dict[str, tuple[list[tuple[float, float]], list[float]]] = {}

    elements = spec.get("elements")
    if not isinstance(elements, list):
        elements = []

    # Paths first so a dot can follow one declared later in the list.
    for element in elements:
        if not isinstance(element, dict) or str(element.get("type")) != "path":
            continue
        raw = element.get("points")
        if not isinstance(raw, list) or len(raw) < 2:
            continue
        try:
            pts = [_denorm(p, size) for p in raw if isinstance(p, (list, tuple)) and len(p) >= 2]
        except (TypeError, ValueError):
            continue
        if len(pts) < 2:
            continue
        if element.get("curve", True) and len(pts) >= 3:
            pts = catmull_rom(pts)
        paths[str(element.get("id") or f"p{len(paths)}")] = (pts, arc_lengths(pts))

    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue
        kind = str(element.get("type") or "")
        start = float(element.get("from", 0.0) or 0.0)
        end = float(element.get("to", duration) or duration)
        if t < start:
            continue
        alpha = fade(t, start, attack=0.4, hold=max(0.0, end - start), release=0.5)
        if alpha <= 0.01:
            continue
        progress = clamp((t - start) / max(end - start, 1e-6))

        if kind == "path":
            key = str(element.get("id") or f"p{index}")
            if key not in paths:
                continue
            pts, lengths = paths[key]
            width = float(element.get("width", 7) or 7)
            frame.polyline(pts, _colour_of("dim", alpha), width)
            frame.polyline(slice_to(pts, lengths, ease_in_out(progress)),
                           _colour_of(element.get("colour"), alpha), width)

        elif kind == "dot":
            follows = str(element.get("follows") or "")
            radius = float(element.get("radius", 24) or 24)
            if follows in paths:
                pts, lengths = paths[follows]
                x, y = point_at(pts, lengths, ease_in_out(progress))
            else:
                x, y = _denorm(element.get("at") or (0.5, 0.5), size)
            frame.circle((x, y), radius, _colour_of(element.get("colour"), alpha))
            if element.get("halo", True):
                frame.circle((x, y), radius * 1.7, _colour_of(element.get("colour"), alpha * 0.35), 3)

        elif kind == "text":
            x, y = _denorm(element.get("at") or (0.5, 0.5), size)
            frame.wrapped(str(element.get("text") or ""), (x, y),
                          int(element.get("size", 46) or 46),
                          _colour_of(element.get("colour"), alpha),
                          weight=str(element.get("weight") or "bold"),
                          max_width=float(element.get("max_width", 0.82) or 0.82) * frame.w)

        elif kind == "circle":
            x, y = _denorm(element.get("at") or (0.5, 0.5), size)
            radius = float(element.get("radius", 0.15) or 0.15) * frame.w
            if element.get("grow", False):
                radius *= ease_out(progress)
            frame.circle((x, y), radius, _colour_of(element.get("colour"), alpha),
                         0.0 if element.get("fill", False) else float(element.get("width", 6) or 6))

        elif kind == "bar":
            x, y = _denorm(element.get("at") or (0.5, 0.5), size)
            width = float(element.get("width", 0.06) or 0.06) * frame.w
            height = float(element.get("height", 0.18) or 0.18) * frame.h
            level = ease_out(progress) if element.get("grow", True) else 1.0
            frame.rect((x - width / 2, y - height, x + width / 2, y), _colour_of("dim", alpha), 4)
            frame.rect((x - width / 2, y - height * level, x + width / 2, y),
                       _colour_of(element.get("colour"), alpha))


# ---------------------------------------------------------------------------
# Template -- The Balance Scale
#
# Immediate gratification against delayed payoff. The left pan loads fast and
# stops; the right pan loads slowly and never stops. The beam tips the other
# way at the impact, which is the whole argument in one movement.
# ---------------------------------------------------------------------------

_SCALE_PIVOT = (540.0, 880.0)
_SCALE_ARM = 372.0
_SCALE_MAX_TILT = 0.30              # radians
_SCALE_STAND = 500.0                # pivot to floor
_WEIGHT_W, _WEIGHT_H = 128.0, 34.0
_WEIGHT_SLOTS = 5


def _pan(frame: Frame, anchor: tuple[float, float], colour: tuple[int, int, int],
         load: float) -> None:
    """
    A plate hanging on two cords, with weights stacked on top of it.

    Stacked *on* the plate rather than drawn inside a trapezoid: a trapezoid
    with rungs in it reads as a lampshade, and the whole point of the shot is
    that one side is visibly carrying more than the other.
    """
    x, y = anchor
    drop = 132.0
    plate = y + drop
    half = _WEIGHT_W * 0.72

    frame.line((x - 6, y), (x - half * 0.8, plate), mix(colour, 0.5), 3)
    frame.line((x + 6, y), (x + half * 0.8, plate), mix(colour, 0.5), 3)
    frame.line((x - half, plate), (x + half, plate), colour, 8)

    whole = int(load * _WEIGHT_SLOTS)
    for i in range(min(whole, _WEIGHT_SLOTS)):
        top = plate - (i + 1) * (_WEIGHT_H + 6)
        frame.rect((x - _WEIGHT_W / 2, top, x + _WEIGHT_W / 2, top + _WEIGHT_H),
                   colour, width=5, radius=5)
    partial = load * _WEIGHT_SLOTS - whole
    if 0.06 < partial and whole < _WEIGHT_SLOTS:
        height = _WEIGHT_H * partial
        top = plate - whole * (_WEIGHT_H + 6) - height
        frame.rect((x - _WEIGHT_W / 2, top, x + _WEIGHT_W / 2, top + height),
                   mix(colour, 0.45), width=4, radius=4)


def scene_balance(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """A scale that tips the other way once the slow side finally lands."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    px, py = _SCALE_PIVOT
    floor = py + _SCALE_STAND

    # Left: instant, then flat. Right: slow, then unstoppable.
    u = ph.travel(t, ease_in_out)
    now_load = ease_out(clamp(u / 0.26))
    later_load = clamp(u ** 2.7) * 1.3

    diff = clamp((later_load - now_load) * 0.95, -1.0, 1.0)
    tilt = _SCALE_MAX_TILT * (ease_out_cubic(abs(diff)) * (1 if diff >= 0 else -1))
    # A short wobble after the flip so it settles like a real beam.
    since = t - ph.impact
    if 0 <= since < 1.5:
        tilt += math.sin(since * 10.0) * 0.038 * math.exp(-2.8 * since)

    # Stand: an outline, not a filled wedge.
    frame.line((px, py), (px - 92, floor), WHITE, 6)
    frame.line((px, py), (px + 92, floor), WHITE, 6)
    frame.line((px - 150, floor), (px + 150, floor), WHITE, 8)

    # Beam.
    dx, dy = math.cos(tilt) * _SCALE_ARM, math.sin(tilt) * _SCALE_ARM
    left = (px - dx, py - dy)
    right = (px + dx, py + dy)
    frame.line(left, right, WHITE, 10)
    frame.circle((px, py), 20, WHITE)
    frame.circle((px, py), 34, mix(WHITE, 0.30), 3)

    _pan(frame, left, GREY, now_load)
    _pan(frame, right, WHITE, min(1.0, later_load))

    won = ph.after(t, 0.7)
    frame.text(label(spec, "left"), (left[0], left[1] + 132 + 54), 40,
               mix(GREY, fade(t, ph.lead + 0.2, 0.5)), tracking=5)
    frame.text(label(spec, "right"), (right[0], right[1] + 132 + 54), 44,
               mix(WHITE, fade(t, ph.lead + 0.6, 0.5) * (0.5 + 0.5 * won)), tracking=5)


# ---------------------------------------------------------------------------
# Template -- The Gravity Funnel
#
# What pulls you in gets faster the closer you get. A mote circles a throat,
# the orbit tightens, and past the point of no return it is gone in three
# frames. Reads as a distraction sink or as compounding pull, depending on
# the copy over it.
# ---------------------------------------------------------------------------

_FUNNEL_CENTRE = (540.0, 1030.0)
_FUNNEL_TOP_R = 400.0
_FUNNEL_RINGS = 9


def scene_funnel(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """Concentric rings narrowing to a throat, with a mote spiralling in."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    cx, cy = _FUNNEL_CENTRE
    reveal = ph.draw(t)

    # The funnel: ellipses shrinking and sinking, drawn from the rim inward as
    # the trim-path beat runs.
    for i in range(_FUNNEL_RINGS):
        share = i / (_FUNNEL_RINGS - 1)
        if reveal < share * 0.85:
            break
        radius = _FUNNEL_TOP_R * (1.0 - share) ** 1.55 + 16
        depth = cy + share * 330
        lit = 0.30 + 0.70 * share
        frame.ellipse((cx, depth), radius, radius * 0.32, mix(WHITE, lit), 4)

    # Walls, to read it as a solid rather than a stack of hoops.
    for side in (-1, 1):
        frame.line((cx + side * _FUNNEL_TOP_R, cy),
                   (cx + side * 16, cy + 330), mix(WHITE, 0.45 * reveal), 4)

    # The mote: angle accelerates and radius collapses as it falls in.
    u = ph.travel(t, ease_in_cubic)
    if u > 0.001:
        turns = 3.4
        angle = 2 * math.pi * turns * (u ** 1.9)
        radius = _FUNNEL_TOP_R * (1.0 - u) ** 1.55 + 16
        depth = cy + u * 330
        mx = cx + math.cos(angle) * radius
        my = depth + math.sin(angle) * radius * 0.32

        # Trail: a few positions behind, fading.
        for k in range(1, 7):
            back = max(0.0, u - k * 0.022)
            a2 = 2 * math.pi * turns * (back ** 1.9)
            r2 = _FUNNEL_TOP_R * (1.0 - back) ** 1.55 + 16
            d2 = cy + back * 330
            frame.circle((cx + math.cos(a2) * r2, d2 + math.sin(a2) * r2 * 0.32),
                         16 - k * 1.6, mix(WHITE, 0.30 * (1.0 - k / 7)))
        frame.circle((mx, my), 20, WHITE)

    # Past the throat: a burst, then nothing.
    burst = ph.after(t, 0.75)
    if 0 < burst < 1:
        for i in range(12):
            angle = 2 * math.pi * i / 12
            reach = 40 + burst * 250
            frame.circle((cx + math.cos(angle) * reach, cy + 330 + math.sin(angle) * reach * 0.35),
                         7 * (1 - burst), mix(WHITE, 1 - burst))

    frame.text(label(spec, "pull"), (cx, cy - _FUNNEL_TOP_R * 0.34 - 90), 38,
               mix(GREY, fade(t, ph.lead + 0.3, 0.6)), tracking=6)


# ---------------------------------------------------------------------------
# Template -- The Domino Chain
#
# One small push topples something it could never have moved directly. Each
# domino is 1.35x its neighbour, so the last is roughly eleven times the first
# and the scale is the point.
# ---------------------------------------------------------------------------

_DOMINO_COUNT = 6
_DOMINO_GROWTH = 1.46               # last tile is ~6.6x the first
_DOMINO_GAP = 0.55                  # gap as a fraction of the tile that pushes
_DOMINO_BOX = (90.0, 990.0)         # left, right
_DOMINO_FLOOR = 1330.0
_DOMINO_MAX_H = 520.0


def _domino_layout() -> list[tuple[float, float, float]]:
    """
    (x, height, width) per tile, solved to fit the frame.

    The sizes cannot be hard-coded: at 1.46x growth the eighth tile is taller
    than the canvas and the last two fall clean off the right edge. So the
    proportions are fixed and the whole run is scaled to the box, including
    the room the final tile needs to lie down in.
    """
    raw_heights = [_DOMINO_GROWTH ** i for i in range(_DOMINO_COUNT)]
    raw_gaps = [h * _DOMINO_GAP for h in raw_heights[:-1]]
    # The last tile lands flat, so the run needs its full height to the right.
    raw_extent = sum(raw_gaps) + raw_heights[-1]

    left, right = _DOMINO_BOX
    scale = min((right - left) / raw_extent, _DOMINO_MAX_H / raw_heights[-1])

    tiles: list[tuple[float, float, float]] = []
    x = left
    for i, raw in enumerate(raw_heights):
        height = raw * scale
        tiles.append((x, height, max(10.0, height * 0.19)))
        if i < len(raw_gaps):
            x += raw_gaps[i] * scale
    return tiles


def scene_dominoes(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """A cascade where every tile is bigger than the one that knocked it."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    tiles = _domino_layout()
    span = max(ph.impact - ph.lead, 0.4)
    # Each tile starts later than the last and the gaps shorten, so the chain
    # accelerates the way a real one does.
    starts = [ph.lead + span * (i / _DOMINO_COUNT) ** 1.4 for i in range(_DOMINO_COUNT)]
    fall_time = span / _DOMINO_COUNT * 1.6

    frame.line((_DOMINO_BOX[0] - 30, _DOMINO_FLOOR), (_DOMINO_BOX[1] + 30, _DOMINO_FLOOR),
               mix(WHITE, 0.55), 5)

    for i, (x, height, width) in enumerate(tiles):
        progress = clamp((t - starts[i]) / fall_time)
        # Gravity, not a linear tip: slow off the vertical, then it goes.
        angle = (math.pi / 2) * ease_in_cubic(progress) * 0.94
        standing = 1.0 - progress
        lit = 0.42 + 0.58 * clamp(progress * 2.0)

        sin_a, cos_a = math.sin(angle), math.cos(angle)
        base = (x, _DOMINO_FLOOR)
        corners = [
            base,
            (base[0] + width * cos_a, base[1] - width * sin_a),
            (base[0] + width * cos_a + height * sin_a, base[1] - width * sin_a - height * cos_a),
            (base[0] + height * sin_a, base[1] - height * cos_a),
        ]
        # A filled body so a fallen tile still reads as an object rather than
        # as one more line in a pile of lines.
        frame.polygon(corners, mix(WHITE, 0.06 + 0.10 * standing))
        frame.polyline(corners + [corners[0]], mix(WHITE, lit), 5)

    # The finger that starts it.
    nudge = clamp((t - ph.lead) / 0.55)
    if nudge < 1.0:
        first_x = tiles[0][0]
        tip = first_x - 26
        frame.line((tip - 78 + nudge * 56, _DOMINO_FLOOR - tiles[0][1] * 0.7),
                   (tip - 12 + nudge * 56, _DOMINO_FLOOR - tiles[0][1] * 0.7),
                   mix(GREY, 1.0 - nudge), 7)

    frame.text(label(spec, "first"), (tiles[0][0] + 60, _DOMINO_FLOOR + 74), 32,
               mix(GREY, fade(t, ph.lead, 0.5)), tracking=4)
    last_x, last_h, _ = tiles[-1]
    frame.text(label(spec, "last"), (min(last_x + 40, frame.w - 170), _DOMINO_FLOOR - last_h - 60), 38,
               mix(WHITE, fade(t, ph.impact - 0.5, 0.6)), tracking=4)


# ---------------------------------------------------------------------------
# Character metaphors
#
# The five below put the stick-figure rig on screen. The difference from the
# geometric templates is not decoration: a figure straining against a chain
# argues something an abstract curve cannot, because the viewer reads effort
# from a body before they read it from a slope.
# ---------------------------------------------------------------------------

def _track(frame: Frame, x0: float, x1: float, y: float, fill: float,
           colour: tuple[int, int, int], height: float = 26.0) -> None:
    """A horizontal progress rail with a filled portion."""
    frame.rect((x0, y - height / 2, x1, y + height / 2), DIM, width=3, radius=height / 2)
    if fill > 0.004:
        frame.rect((x0, y - height / 2, x0 + (x1 - x0) * clamp(fill), y + height / 2),
                   colour, radius=height / 2)


# --- 1. Comparison Split ----------------------------------------------------

_TIERS = (
    # (floor y, pose, label slot, share of the rail it reaches, facing)
    (760.0, "idle", "tier1", 0.22),
    (1090.0, "pushing", "tier2", 0.55),
    (1420.0, "flexing", "tier3", 1.00),
)
_TIER_FIGURE_H = 200.0


def scene_comparison(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """Three tiers doing the same work at three different rates."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    for index, (floor, pose, slot, reach) in enumerate(_TIERS):
        # Tiers arrive one after another so the comparison builds rather than
        # appearing all at once.
        appear = ph.lead + index * 0.55
        alpha = fade(t, appear, 0.6)
        if alpha <= 0.02:
            continue

        best = index == len(_TIERS) - 1
        colour = mix(WHITE if best else GREY, alpha)

        frame.line((90, floor), (990, floor), mix(DIM, alpha), 3)
        rig.draw_figure(frame, pose, phase=(t * 0.9 + index * 0.3) % 1.0,
                        anchor=(210, floor), height=_TIER_FIGURE_H,
                        colour=colour, weight=5.0)

        # Each tier moves at its own rate, and only the last one finishes.
        progress = clamp((t - appear - 0.3) / max(ph.impact - appear - 0.3, 0.4))
        fill = reach * (ease_out_cubic(progress) if best else ease_in_out(progress))
        _track(frame, 360, 950, floor - 46, fill, colour)

        frame.text(label(spec, slot), (380, floor - 122), 32,
                   mix(WHITE if best else GREY, alpha), tracking=4)
        if best and ph.after(t, 0.5) > 0:
            frame.text("100%", (930, floor - 122), 30,
                       mix(WHITE, ph.after(t, 0.5)), tracking=3)


# --- 2. The Steep Staircase -------------------------------------------------

_CLIMB_STEPS = 5
_CLIMB_BOX = (150.0, 940.0, 640.0, 1420.0)      # left, right, top, floor


def _climb_points() -> list[tuple[float, float]]:
    left, right, top, floor = _CLIMB_BOX
    run = (right - left) / _CLIMB_STEPS
    rise = (floor - top) / _CLIMB_STEPS
    points = [(left - 60, floor)]
    for i in range(_CLIMB_STEPS):
        x = left + i * run
        y = floor - i * rise
        points += [(x, y), (x, y - rise), (x + run, y - rise)]
    return points


def _trophy(frame: Frame, centre: tuple[float, float], size: float,
            colour: tuple[int, int, int], glow: float = 0.0) -> None:
    """A cup on a plinth -- what the last step is for."""
    x, y = centre
    bowl_w, bowl_h = size * 0.62, size * 0.52

    if glow > 0.02:
        for ring in range(3):
            frame.circle((x, y - size * 0.30), size * (0.62 + ring * 0.34),
                         mix(colour, 0.20 * glow / (ring + 1)), 3)

    frame.polyline([(x - bowl_w / 2, y - size * 0.62),
                    (x + bowl_w / 2, y - size * 0.62),
                    (x + bowl_w * 0.30, y - size * 0.62 + bowl_h),
                    (x - bowl_w * 0.30, y - size * 0.62 + bowl_h),
                    (x - bowl_w / 2, y - size * 0.62)], colour, 6)
    for side in (-1, 1):
        frame.ellipse((x + side * bowl_w * 0.60, y - size * 0.48),
                      size * 0.16, size * 0.20, colour, 5)
    frame.line((x, y - size * 0.10), (x, y - size * 0.28), colour, 6)
    frame.line((x - size * 0.26, y - size * 0.06), (x + size * 0.26, y - size * 0.06), colour, 7)
    frame.rect((x - size * 0.34, y - size * 0.06, x + size * 0.34, y), colour, width=5)


def scene_climb(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """A figure walking up labelled stages, one step at a time."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    left, right, top, floor = _CLIMB_BOX
    run = (right - left) / _CLIMB_STEPS
    rise = (floor - top) / _CLIMB_STEPS
    steps = _climb_points()

    reveal = ph.draw(t)
    lengths = arc_lengths(steps)
    frame.polyline(steps, DIM, 6)
    frame.polyline(slice_to(steps, lengths, reveal), WHITE, 7)

    # Stage labels on each tread, lighting as the figure passes them.
    walk = ph.travel(t, ease_in_out)
    reached = walk * _CLIMB_STEPS
    slots = ("stage1", "stage2", "stage3", "stage4", "stage5")
    for i in range(_CLIMB_STEPS):
        text = label(spec, slots[i])
        if not text:
            continue
        lit = clamp(reached - i)
        frame.text(text, (left + i * run + run / 2, floor - i * rise - 42),
                   28, mix(WHITE if lit > 0.5 else DIM, 0.35 + 0.65 * lit), tracking=3)

    # The prize on the top tread, lighting as the climb closes on it.
    _trophy(frame, (right + 40, top), 86,
            mix(WHITE, 0.35 + 0.65 * clamp(walk * 1.2)), glow=clamp((walk - 0.6) / 0.4))

    # The figure climbs tread by tread rather than gliding up the diagonal.
    step_index = min(_CLIMB_STEPS - 1, int(reached))
    within = reached - step_index
    x = left + step_index * run + run * (0.25 + 0.5 * within)
    y = floor - (step_index + 1) * rise
    rig.draw_figure(frame, "climbing_stairs", phase=(t * 1.5) % 1.0, anchor=(x, y),
                    height=210, colour=WHITE, weight=6.0)

    # At the top: the pose changes to say arrival.
    if ph.after(t, 0.5) > 0.4:
        # The upper-left is the empty quadrant of a rising staircase, and the
        # figure finishes on the top tread at the right -- so the summit label
        # goes left, not above it.
        frame.text(label(spec, "summit"), (360, top + 90), 42,
                   mix(WHITE, ph.after(t, 1.0)), tracking=6)


# --- 3. The Delusion Mirror -------------------------------------------------

_MIRROR_BOX = (560.0, 660.0, 960.0, 1400.0)     # left, top, right, bottom


def scene_mirror(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """What the figure is, beside what the figure believes it is."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    left, top, right, bottom = _MIRROR_BOX
    grow = ph.travel(t, ease_out_cubic)

    # The real figure, unchanged throughout.
    rig.draw_figure(frame, "idle", phase=(t * 0.35) % 1.0, anchor=(300, 1400),
                    height=330, colour=GREY, weight=6.0, facing=1)
    frame.text(label(spec, "real"), (300, 1470), 34,
               mix(GREY, fade(t, ph.lead + 0.3, 0.6)), tracking=4)

    # The mirror.
    frame.rect((left, top, right, bottom), mix(WHITE, 0.75), width=7, radius=14)
    frame.rect((left + 16, top + 16, right - 16, bottom - 16), mix(WHITE, 0.18), width=3, radius=8)

    # The reflection: bigger, brighter and flexing, and it keeps growing.
    reflection_h = 300 + 190 * grow
    glow_level = 0.45 + 0.55 * grow
    skeleton = rig.draw_figure(
        frame, "flexing", phase=(t * 0.8) % 1.0,
        anchor=((left + right) / 2, bottom - 60), height=reflection_h,
        colour=mix(WHITE, glow_level), weight=6.0 + 3.0 * grow, facing=-1,
    )
    # A halo that reads as self-flattery rather than as light.
    for ring in range(3):
        frame.circle(skeleton.head, skeleton.head_radius * (1.7 + ring * 0.85),
                     mix(WHITE, glow_level * 0.30 / (ring + 1)), 3)

    frame.text(label(spec, "imagined"), ((left + right) / 2, bottom + 48), 36,
               mix(WHITE, fade(t, ph.lead + 0.8, 0.6) * glow_level), tracking=4)

    # The gap between them, stated at the beat.
    gap = ph.after(t, 0.8)
    if gap > 0:
        frame.line((380, 1160), (left - 30, 1160), mix(WHITE, gap * 0.7), 4)
        frame.text(label(spec, "gap"), ((380 + left) / 2, 1120), 30,
                   mix(WHITE, gap), tracking=4)


# --- 4. The Chain & Anchor --------------------------------------------------

_ANCHOR_SLOTS = ("anchor1", "anchor2")


def _chain(frame: Frame, start: tuple[float, float], end: tuple[float, float],
           links: int, colour: tuple[int, int, int], sag: float, width: float = 4.0) -> None:
    """A run of links between two points, sagging when slack."""
    for i in range(links):
        u = (i + 0.5) / links
        x = start[0] + (end[0] - start[0]) * u
        y = start[1] + (end[1] - start[1]) * u + math.sin(math.pi * u) * sag
        frame.ellipse((x, y), 15, 9, colour, width)


def _anchor_block(frame: Frame, centre: tuple[float, float], size: float,
                  colour: tuple[int, int, int], text: str, alpha: float,
                  text_drop: float = 0.0) -> None:
    """A dead weight with its name on it."""
    x, y = centre
    half = size / 2
    top, base = half * 0.72, half
    frame.polygon([(x - top, y - half * 0.7), (x + top, y - half * 0.7),
                   (x + base, y + half * 0.7), (x - base, y + half * 0.7)],
                  mix(WHITE, 0.10 * alpha))
    frame.polyline([(x - top, y - half * 0.7), (x + top, y - half * 0.7),
                    (x + base, y + half * 0.7), (x - base, y + half * 0.7),
                    (x - top, y - half * 0.7)], colour, 6)
    # A shackle on top and hatching inside: an outline alone reads as an empty
    # crate, and the whole point is that the thing is heavy.
    frame.circle((x, y - half * 0.7 - 22), 18, colour, 5)
    frame.line((x - 18, y - half * 0.7 - 22), (x + 18, y - half * 0.7 - 22), colour, 4)
    for k in range(1, 4):
        share = k / 4
        band_y = y - half * 0.7 + half * 1.4 * share
        inset = top + (base - top) * share
        frame.line((x - inset, band_y), (x + inset, band_y), mix(colour, 0.42), 3)
    if text:
        frame.text(text, (x, y + half * 0.7 + 52 + text_drop), 26, colour, tracking=2)


def scene_chains(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """Hauling named dead weight, until it lets go."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    floor = 1300.0
    frame.line((60, floor), (1020, floor), mix(WHITE, 0.45), 6)

    strain = ph.travel(t, ease_in_out)
    snapped = ph.after(t, 1.1)

    # The figure inches forward while it drags, then straightens once free.
    x = 660 + strain * 120 + snapped * 130
    pose = "struggling_chained" if snapped <= 0 else "walking"
    skeleton = rig.draw_figure(
        frame, pose, phase=(t * 1.3) % 1.0, anchor=(x, floor), height=460,
        colour=WHITE, weight=8.0,
    )

    grip = min(skeleton.hand_l, skeleton.hand_r, key=lambda p: p[0])
    anchors = [(300.0, floor - 96, 220.0), (120.0, floor - 74, 176.0)]

    for index, (ax, ay, size) in enumerate(anchors):
        alpha = fade(t, ph.lead + 0.3 + index * 0.25, 0.6)
        if alpha <= 0.02:
            continue
        # After the snap the weights stay put and the links scatter.
        colour = mix(GREY if snapped <= 0 else DIM, alpha)
        _anchor_block(frame, (ax, ay), size, colour,
                      label(spec, _ANCHOR_SLOTS[index]), alpha,
                      text_drop=index * 46.0)

        if snapped <= 0:
            # Taut as the strain builds: the sag is what shows the effort, and
            # it sags upward-bounded so a link never drops through the floor.
            _chain(frame, (ax + size / 2, ay - size * 0.55), grip, 8 + index,
                   mix(WHITE, 0.55 + 0.45 * strain), sag=42 * (1.0 - strain))
        else:
            for k in range(6):
                spread = snapped * (90 + k * 26)
                frame.ellipse((ax + size / 2 + spread, ay - size * 0.55 - spread * 0.55),
                              15, 9, mix(GREY, (1.0 - snapped) * 0.8), 4)

    if snapped > 0.25:
        frame.text(label(spec, "freed"), (x + 40, floor - 540), 42,
                   mix(WHITE, clamp((snapped - 0.25) / 0.5)), tracking=5)


# --- 5. Growth & Consistency ------------------------------------------------

_PLANT_ROOT = (700.0, 1380.0)


def _tree(frame: Frame, root: tuple[float, float], growth: float,
          colour: tuple[int, int, int]) -> None:
    """
    Seed to sapling to canopy, driven by one 0..1 parameter.

    Branches are placed from a fixed table rather than a random walk: the same
    growth value must draw the same tree on every frame, or it flickers.
    """
    x, y = root
    if growth < 0.06:
        frame.circle((x, y - 10), 12 * (0.4 + growth * 8), colour)
        return

    trunk = 60 + 430 * ease_out(growth)
    frame.line((x, y), (x, y - trunk), colour, 7 + 5 * growth)

    stages = (
        # (height up the trunk, angle from vertical, length share, when it appears)
        (0.42, -0.72, 0.40, 0.22), (0.42, 0.72, 0.40, 0.26),
        (0.66, -0.60, 0.34, 0.42), (0.66, 0.60, 0.34, 0.46),
        (0.86, -0.48, 0.26, 0.62), (0.86, 0.48, 0.26, 0.66),
    )
    for height, angle, share, appears in stages:
        if growth < appears:
            continue
        local = clamp((growth - appears) / 0.25)
        base = (x, y - trunk * height)
        reach = trunk * share * local
        tip = (base[0] + math.sin(angle) * reach, base[1] - math.cos(angle) * reach)
        frame.line(base, tip, colour, 5 + 3 * local)
        if local > 0.6:
            frame.circle(tip, 16 * (local - 0.6) / 0.4, mix(colour, 0.8))

    # The canopy arrives last and is what makes the payoff visible.
    if growth > 0.72:
        bloom = clamp((growth - 0.72) / 0.28)
        for ring in range(3):
            frame.ellipse((x, y - trunk * 0.86), (150 + ring * 46) * bloom,
                          (110 + ring * 34) * bloom,
                          mix(colour, 0.42 * bloom / (ring + 1)), 4)


def _watering_can(frame: Frame, hand: tuple[float, float], tilt: float,
                  colour: tuple[int, int, int]) -> tuple[float, float]:
    """A small can hanging off the hand; returns where the spout points."""
    x, y = hand
    w, h = 54.0, 40.0
    lean = 0.35 * tilt
    dx, dy = math.cos(lean), math.sin(lean)
    corners = [
        (x - w * 0.2 * dx, y - h * 0.2),
        (x + w * 0.8 * dx, y - h * 0.2 + w * 0.8 * dy),
        (x + w * 0.8 * dx, y + h * 0.8 + w * 0.8 * dy),
        (x - w * 0.2 * dx, y + h * 0.8),
    ]
    frame.polyline(corners + [corners[0]], colour, 4)
    spout = (corners[1][0] + 34 * dx, corners[1][1] + 22 + 34 * dy)
    frame.line(corners[1], spout, colour, 4)
    return spout


def scene_growth(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """A seed watered every day until it is a tree."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    floor = _PLANT_ROOT[1]
    frame.line((80, floor), (1000, floor), mix(WHITE, 0.45), 5)

    growth = ph.travel(t, ease_in_out)
    _tree(frame, _PLANT_ROOT, growth, WHITE)

    # The figure keeps watering at the same rate the whole way through, which
    # is the argument: the input never changed, only what it was compounding.
    pour = (t * 1.1) % 1.0
    skeleton = rig.draw_figure(frame, "watering_plant", phase=pour, anchor=(300, floor),
                               height=340, colour=GREY, weight=6.0)
    spout = _watering_can(frame, skeleton.hand_r, 0.5 + 0.5 * math.sin(pour * 2 * math.pi), GREY)

    # Droplets, arcing from the spout toward the root.
    for k in range(5):
        drop = ((t * 1.6) + k * 0.2) % 1.0
        dx = spout[0] + (_PLANT_ROOT[0] - 120 - spout[0]) * drop
        dy = spout[1] + 240 * drop * drop
        if dy < floor:
            frame.circle((dx, dy), 6, mix(GREY, 1.0 - drop * 0.6))

    frame.text(label(spec, "input"), (300, floor + 62), 30,
               mix(GREY, fade(t, ph.lead + 0.4, 0.6)), tracking=4)
    if growth > 0.7:
        frame.text(label(spec, "output"), (_PLANT_ROOT[0], floor + 62), 34,
                   mix(WHITE, clamp((growth - 0.7) / 0.25)), tracking=4)


# ---------------------------------------------------------------------------
# Template -- Sisyphus Boulder
#
# A steep slope with checkpoints on it. The figure pushes a textured stone up
# past each one, and the stone slips back a little between pushes -- because a
# boulder that only ever moves forward is a conveyor belt, not a struggle.
# ---------------------------------------------------------------------------

_SLOPE = (110.0, 1430.0, 960.0, 700.0)      # base x/y, summit x/y
_BOULDER_R = 78.0
_PUSHER_H = 300.0
_CHECKPOINTS = 4


def _boulder(frame: Frame, centre: tuple[float, float], radius: float,
             spin: float, colour: tuple[int, int, int]) -> None:
    """
    A stone with facets, rotating as it rolls.

    The facets are the point: a plain circle rolling up a line reads as a dot
    sliding, and the whole shot depends on seeing the thing turn.

    Filled with black before it is stroked, so it occludes the pusher behind
    it. On a steep slope a figure with its shoulder against a boulder will
    always overlap it -- that is what pushing looks like -- and two outlines
    crossing read as a tangle of lines. One of them has to be solid.
    """
    frame.circle(centre, radius, BLACK)
    frame.circle(centre, radius, colour, 7)
    for face in range(5):
        angle = spin + face * (2 * math.pi / 5)
        inner = radius * (0.34 + 0.16 * math.sin(face * 2.1))
        a = (centre[0] + math.sin(angle) * radius * 0.92,
             centre[1] + math.cos(angle) * radius * 0.92)
        b = (centre[0] + math.sin(angle + 1.1) * inner,
             centre[1] + math.cos(angle + 1.1) * inner)
        frame.line(a, b, mix(colour, 0.45), 4)
    frame.circle(centre, radius * 0.30, mix(colour, 0.30), 4)


def scene_sisyphus(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """The slope, the stone, and the ground it loses between pushes."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    bx, by, sx, sy = _SLOPE
    reveal = ph.draw(t)
    slope_angle = math.atan2(by - sy, sx - bx)

    # The slope draws itself in, then the checkpoints appear along it.
    frame.line((bx, by), (bx + (sx - bx) * reveal, by - (by - sy) * reveal), WHITE, 8)
    frame.line((60, by), (bx, by), mix(WHITE, 0.4 * reveal), 6)

    progress = ph.travel(t, ease_out_cubic)
    # Two steps forward, a little back: the slip is what makes it a grind.
    slip = 0.05 * math.sin(progress * math.pi * _CHECKPOINTS * 2.0) * (1.0 - progress)
    position = clamp(progress + slip)

    # Everything rides the slope, so both the stone and the pusher are placed
    # from a point on the line rather than from an x coordinate. Offsetting the
    # stone by its radius along the surface *normal* is what makes it sit on
    # the hill instead of floating beside it.
    run, rise = sx - bx, sy - by
    length = math.hypot(run, rise) or 1.0
    along = (run / length, rise / length)
    # Named `outward` rather than `normal`: the checkpoint loop below already
    # uses `normal` for an angle, and shadowing it made this a float.
    outward = (rise / length, -run / length)     # points away from the surface

    def on_slope(u: float, lift: float = 0.0) -> tuple[float, float]:
        u = clamp(u)
        return (bx + run * u + outward[0] * lift, by + rise * u + outward[1] * lift)

    for i in range(1, _CHECKPOINTS + 1):
        # Spaced so the last mark stops short of the summit -- a checkpoint at
        # share 1.0 sits underneath the summit label.
        share = i / (_CHECKPOINTS + 1)
        if reveal < share * 0.9:
            continue
        cx = bx + (sx - bx) * share
        cy = by - (by - sy) * share
        passed = position >= share
        # A notch cut perpendicular to the slope, so it sits on the hill.
        notch = slope_angle + math.pi / 2
        frame.line((cx - math.cos(notch) * 22, cy + math.sin(notch) * 22),
                   (cx + math.cos(notch) * 22, cy - math.sin(notch) * 22),
                   WHITE if passed else DIM, 6)
        text = label(spec, f"mark{i}")
        if text:
            frame.text(text, (cx + 30, cy - 54), 26,
                       mix(WHITE if passed else DIM, 0.4 + 0.6 * clamp(position - share + 1)),
                       tracking=2)

    # Stone and pusher.
    #
    # The figure is placed from the stone, not from the slope: solve for the
    # feet that put the pushing hand on the stone's lower-left surface, and the
    # feet land within a pixel of the slope anyway. Placing it by slope
    # parameter instead left the hands 140px short of the rock, pushing air.
    stone = on_slope(position, _BOULDER_R)
    contact_angle = math.radians(168)
    contact = (stone[0] + math.cos(contact_angle) * _BOULDER_R,
               stone[1] - math.sin(contact_angle) * _BOULDER_R)
    reach = rig.build_skeleton("pushing_heavy_load", 0.0, (0.0, 0.0), _PUSHER_H).hand_r
    feet = (contact[0] - reach[0], contact[1] - reach[1])

    # Figure first, stone second: the stone is filled black and covers it.
    rig.draw_figure(frame, "pushing_heavy_load", phase=(t * 1.4) % 1.0,
                    anchor=feet, height=_PUSHER_H, colour=WHITE, weight=7.0)
    _boulder(frame, stone, _BOULDER_R, -position * 9.0, WHITE)
    _ = along

    frame.text(label(spec, "slope"), (330, by + 74), 32,
               mix(GREY, fade(t, ph.lead + 0.3, 0.6)), tracking=4)
    if ph.after(t, 0.6) > 0.3:
        # Upper-left is the empty quadrant of a rising slope, and the stone
        # finishes at the top right.
        frame.text(label(spec, "summit"), (330, sy - 30), 40,
                   mix(WHITE, ph.after(t, 1.0)), tracking=5)


# ---------------------------------------------------------------------------
# Template -- The Discipline Iceberg
#
# The classic: a small lit peak and the mass under the water nobody sees. The
# waterline is the only horizontal in the frame, which is what makes the scale
# below it land.
# ---------------------------------------------------------------------------

_WATERLINE = 860.0
_BERG_X = 540.0


def scene_iceberg(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """A visible tip, and the volume of work holding it up."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    reveal = ph.draw(t)
    sink = ph.travel(t, ease_out_cubic)

    # The waterline, drawn first and always.
    frame.line((60, _WATERLINE), (1020, _WATERLINE), mix(WHITE, 0.55 * reveal), 5)
    for k in range(7):
        x = 90 + k * 145
        wobble = math.sin(t * 1.1 + k) * 6
        frame.line((x, _WATERLINE + 16 + wobble), (x + 76, _WATERLINE + 16 + wobble),
                   mix(WHITE, 0.16 * reveal), 3)

    # The tip: small, bright, above the line.
    tip = [(_BERG_X, _WATERLINE - 250), (_BERG_X + 132, _WATERLINE),
           (_BERG_X - 132, _WATERLINE)]
    frame.polygon(tip, mix(WHITE, 0.20))
    frame.polyline(tip + [tip[0]], WHITE, 7)
    frame.line((_BERG_X, _WATERLINE - 250), (_BERG_X - 54, _WATERLINE),
               mix(WHITE, 0.5), 4)

    # The mass: revealed downward as the beat approaches.
    depth = 720.0 * sink
    if depth > 12:
        body = [
            (_BERG_X - 132, _WATERLINE),
            (_BERG_X - 300 * sink, _WATERLINE + depth * 0.34),
            (_BERG_X - 236 * sink, _WATERLINE + depth * 0.78),
            (_BERG_X - 60 * sink, _WATERLINE + depth),
            (_BERG_X + 120 * sink, _WATERLINE + depth * 0.86),
            (_BERG_X + 318 * sink, _WATERLINE + depth * 0.40),
            (_BERG_X + 132, _WATERLINE),
        ]
        frame.polygon(body, mix(WHITE, 0.09))
        frame.polyline(body, mix(WHITE, 0.55 + 0.30 * sink), 6)
        # Internal facets, so the underside has volume rather than being a blob.
        for k in range(1, 4):
            share = k / 4
            frame.line((_BERG_X - 132 + 40 * k, _WATERLINE),
                       (_BERG_X - 90 * sink + 70 * k, _WATERLINE + depth * (0.45 + 0.15 * share)),
                       mix(WHITE, 0.20 * sink), 3)

    above = label(spec, "above")
    below = label(spec, "below", wrap=True)
    if above:
        frame.text(above, (_BERG_X, _WATERLINE - 320), 40,
                   mix(WHITE, fade(t, ph.lead + 0.2, 0.6)), tracking=4)
    if below and sink > 0.35:
        frame.wrapped(below, (_BERG_X, _WATERLINE + 300), 36,
                      mix(GREY, clamp((sink - 0.35) / 0.4)), weight="bold", max_width=560)

    # A figure standing on the tip, for scale.
    if reveal > 0.6:
        rig.draw_figure(frame, "reflective", phase=(t * 0.4) % 1.0,
                        anchor=(_BERG_X, _WATERLINE - 244), height=150,
                        colour=mix(WHITE, clamp((reveal - 0.6) / 0.3)), weight=4.0)


# ---------------------------------------------------------------------------
# Template -- The Divergent Path / Two Doors
#
# One door dark, one lit. The figure walks to the threshold and chooses. The
# unchosen door dims further rather than disappearing, because the point is
# that it stays available and stays wrong.
# ---------------------------------------------------------------------------

_DOOR_W, _DOOR_H = 300.0, 470.0
_DOOR_Y = 1180.0
_DOOR_LEFT_X, _DOOR_RIGHT_X = 250.0, 830.0


def _door(frame: Frame, centre_x: float, glow: float, colour: tuple[int, int, int],
          open_amount: float = 0.0) -> None:
    """A doorway, lit from within by `glow`."""
    left = centre_x - _DOOR_W / 2
    top = _DOOR_Y - _DOOR_H

    if glow > 0.02:
        # Light spilling out across the floor.
        for ring in range(4):
            spread = (1 + ring) * 34 * glow
            frame.polygon([(left - spread * 0.5, _DOOR_Y),
                           (left + _DOOR_W + spread * 0.5, _DOOR_Y),
                           (left + _DOOR_W + spread * 1.6, _DOOR_Y + 150 * glow),
                           (left - spread * 1.6, _DOOR_Y + 150 * glow)],
                          mix(WHITE, 0.05 * glow / (ring + 1)))
        frame.rect((left + 14, top + 14, left + _DOOR_W - 14, _DOOR_Y),
                   mix(WHITE, 0.30 * glow))

    frame.rect((left, top, left + _DOOR_W, _DOOR_Y), colour, width=8, radius=6)
    frame.line((left, _DOOR_Y), (left + _DOOR_W, _DOOR_Y), colour, 8)
    # The handle, and the leaf swinging in when it opens.
    frame.circle((left + _DOOR_W - 46, _DOOR_Y - _DOOR_H / 2), 11, colour)
    if open_amount > 0.02:
        swing = _DOOR_W * 0.55 * open_amount
        frame.polyline([(left, top), (left + swing, top + 40 * open_amount),
                        (left + swing, _DOOR_Y - 40 * open_amount), (left, _DOOR_Y)],
                       mix(colour, 0.6), 5)


def scene_doors(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """Two doors, one walk, one choice."""
    ph = phases_of(spec, duration)
    draw_ambient(frame, t, ph, spec)

    floor = _DOOR_Y
    reveal = ph.draw(t)
    frame.line((40, floor), (1040, floor), mix(WHITE, 0.45 * reveal), 6)

    walk = ph.travel(t, ease_out_cubic)
    chosen = ph.after(t, 1.0)

    # The dark door dims further once the choice is made.
    _door(frame, _DOOR_LEFT_X, 0.0, mix(GREY, (0.55 - 0.30 * chosen) * reveal))
    _door(frame, _DOOR_RIGHT_X, (0.25 + 0.75 * walk) * reveal,
          mix(WHITE, reveal), open_amount=chosen)

    frame.text(label(spec, "left"), (_DOOR_LEFT_X, floor + 70), 34,
               mix(GREY, fade(t, ph.lead + 0.3, 0.6) * (1.0 - 0.4 * chosen)), tracking=4)
    frame.text(label(spec, "right"), (_DOOR_RIGHT_X, floor + 70), 38,
               mix(WHITE, fade(t, ph.lead + 0.5, 0.6)), tracking=4)

    # The figure starts between the doors and walks to the lit one.
    start_x, end_x = 540.0, _DOOR_RIGHT_X
    x = start_x + (end_x - start_x) * walk
    pose = "walking" if chosen <= 0 else "reaching_upward"
    height = 300 - 90 * chosen        # walking away, into the doorway
    rig.draw_figure(frame, pose, phase=(t * 1.5) % 1.0, anchor=(x, floor),
                    height=height, colour=WHITE, weight=7.0 - 2.0 * chosen)

    if chosen > 0.4:
        frame.text(label(spec, "through"), (frame.w / 2, floor - _DOOR_H - 110), 40,
                   mix(WHITE, clamp((chosen - 0.4) / 0.4)), tracking=5)


SceneFn = Callable[[Frame, float, dict[str, Any], float], None]

# The metaphor library.
#
# Keys are the metaphor_type names Gemini chooses from, so the model's answer
# is the registry key with no translation layer in between. Each entry carries
# its own copy, so a template is a complete short on its own even when the
# model is unreachable.
TEMPLATES: dict[str, dict[str, Any]] = {
    "split_path": {
        "label": "① Steep vs. Shallow Path",
        "blurb": "Two routes race: the steep one dips hard and ends high, the flat "
                 "one cruises and ends on spikes.",
        "fn": scene_curve,
        "title": "5 YEARS OF PAIN",
        "subtitle": "buys fifty years of comfort",
        "payoff": "Choose your hard.",
        "climax": 0.72,
        "suits": "trade-offs, delayed reward, two ways to spend the same decade",
    },
    "compounding_jar": {
        "label": "② Compounding Skill Jar",
        "blurb": "An outlined vessel filling on a 1.01^n curve with a live day "
                 "counter: nothing, nothing, then everything.",
        "fn": scene_vessel,
        "title": "1% BETTER",
        "subtitle": "every single day for a year",
        "payoff": "37x. That is the whole secret.",
        "climax": 0.88,
        "suits": "compounding, habits, why early progress is invisible",
    },
    "staircase_progress": {
        "label": "③ Exponential Staircase",
        "blurb": "A figure pushing up consistent steps while effort stays linear "
                 "and reward goes vertical.",
        "fn": scene_staircase,
        "title": "PUSH LONG ENOUGH",
        "subtitle": "and the hill starts pushing back",
        "payoff": "Resistance becomes leverage.",
        "climax": 0.80,
        "suits": "discipline, systems over motivation, slow steady effort",
    },
    "balance_scale": {
        "label": "④ Balance Scale",
        "blurb": "One pan loads instantly and stops; the other loads slowly and "
                 "never stops. The beam flips on the beat.",
        "fn": scene_balance,
        "title": "NOW OR LATER",
        "subtitle": "one of them keeps paying",
        "payoff": "The slow pan always wins.",
        "climax": 0.74,
        "suits": "instant gratification, patience, choosing between two payoffs",
    },
    "gravity_funnel": {
        "label": "⑤ Gravity Funnel",
        "blurb": "A mote circling a throat, the orbit tightening until it is gone "
                 "in three frames.",
        "fn": scene_funnel,
        "title": "THE PULL",
        "subtitle": "gets stronger the closer you get",
        "payoff": "Leave early. There is no late.",
        "climax": 0.82,
        "suits": "distraction, addiction, momentum you cannot escape",
    },
    "domino_chain": {
        "label": "⑥ Domino Chain",
        "blurb": "Eight tiles, each 1.35x the last, falling faster as they go. The "
                 "final one is eleven times the first.",
        "fn": scene_dominoes,
        "title": "ONE SMALL PUSH",
        "subtitle": "topples what you could never lift",
        "payoff": "Start with the tile you can move.",
        "climax": 0.86,
        "suits": "leverage, small starts, cascading consequences",
    },
    "comparison_split": {
        "label": "⑦ Comparison Split",
        "blurb": "Three tiers doing the same work at three different rates. Only "
                 "the bottom one finishes.",
        "fn": scene_comparison,
        "title": "THREE WAYS TO WORK",
        "subtitle": "same hours, three outcomes",
        "payoff": "Effort is not the variable.",
        "climax": 0.80,
        "suits": "comparing approaches, working hard versus working well, tiers of skill",
    },
    "steep_staircase": {
        "label": "⑧ The Steep Staircase",
        "blurb": "A figure walking up labelled stages one tread at a time, each "
                 "lighting as it is passed.",
        "fn": scene_climb,
        "title": "ONE STEP AT A TIME",
        "subtitle": "the stages nobody skips",
        "payoff": "There is no lift.",
        "climax": 0.84,
        "suits": "progression, stages of mastery, the long route with no shortcut",
    },
    "delusion_mirror": {
        "label": "⑨ The Delusion Mirror",
        "blurb": "A plain figure beside the glowing, flexing version it sees in "
                 "the mirror -- and the gap between them.",
        "fn": scene_mirror,
        "title": "THE GAP",
        "subtitle": "between who you are and who you think you are",
        "payoff": "Close it with reps, not with belief.",
        "climax": 0.78,
        "suits": "self-image, delusion, the difference between confidence and competence",
    },
    "chain_anchor": {
        "label": "⑩ The Chain & Anchor",
        "blurb": "A figure hauling named dead weight, until the chain lets go and "
                 "it straightens up and walks.",
        "fn": scene_chains,
        "title": "WHAT YOU DRAG",
        "subtitle": "you chose to pick up",
        "payoff": "Put it down. Then move.",
        "climax": 0.76,
        "suits": "excuses, fear, baggage, the things slowing someone down",
    },
    "growth_consistency": {
        "label": "⑪ Growth & Consistency",
        "blurb": "A figure watering at the same rate all the way through while a "
                 "seed becomes a canopy.",
        "fn": scene_growth,
        "title": "SAME WATER",
        "subtitle": "every single day",
        "payoff": "The input never changed.",
        "climax": 0.86,
        "suits": "patience, consistency, tending something before it shows anything",
    },
    "sisyphus_boulder": {
        "label": "⑫ Sisyphus Boulder",
        "blurb": "A steep slope with checkpoints, a faceted stone that turns as it "
                 "rolls, and ground lost between pushes.",
        "fn": scene_sisyphus,
        "title": "THE SAME HILL",
        "subtitle": "every single morning",
        "payoff": "The hill is the point.",
        "climax": 0.84,
        "suits": "perseverance, grinding, work that resets, effort without applause",
    },
    "discipline_iceberg": {
        "label": "⑬ The Discipline Iceberg",
        "blurb": "A small lit tip above the waterline and the mass underneath it, "
                 "revealed downward as the beat lands.",
        "fn": scene_iceberg,
        "title": "WHAT THEY SEE",
        "subtitle": "is the part above the water",
        "payoff": "The rest is why it floats.",
        "climax": 0.80,
        "suits": "hidden work, overnight success, the unseen cost of a visible result",
    },
    "two_doors": {
        "label": "⑭ The Divergent Path",
        "blurb": "One dark door, one lit. The figure walks to the threshold, and the "
                 "unchosen door dims rather than vanishing.",
        "fn": scene_doors,
        "title": "TWO DOORS",
        "subtitle": "both of them stay open",
        "payoff": "Walk through one of them.",
        "climax": 0.78,
        "suits": "a decision, two futures, the cost of not choosing",
    },
    "custom": {
        "label": "⑮ Dynamic AI Scene",
        "blurb": "Gemini writes the geometry from scratch: paths, followed dots, "
                 "bars and text placed for your concept alone.",
        "fn": scene_custom,
        "title": "",
        "subtitle": "",
        "payoff": "",
        "climax": 0.75,
        "suits": "anything the six fixed metaphors do not fit",
    },
}

# The six named metaphors Gemini picks between. "custom" is not in the list --
# the model reaches it by being asked for one, not by choosing it.
METAPHOR_TYPES: tuple[str, ...] = (
    "staircase_progress", "compounding_jar", "balance_scale",
    "split_path", "gravity_funnel", "domino_chain",
    # Character metaphors -- these put the rig on screen.
    "comparison_split", "steep_staircase", "delusion_mirror",
    "chain_anchor", "growth_consistency", "sisyphus_boulder",
    "discipline_iceberg", "two_doors",
)

# The eight that put the stick-figure rig on screen.
CHARACTER_TYPES: tuple[str, ...] = (
    "comparison_split", "steep_staircase", "delusion_mirror",
    "chain_anchor", "growth_consistency", "sisyphus_boulder",
    "discipline_iceberg", "two_doors",
)

# The old keys, kept so ledger entries and saved session state from before the
# rename still resolve instead of silently falling back to the default.
TEMPLATE_ALIASES: dict[str, str] = {
    "curve": "split_path",
    "vessel": "compounding_jar",
    "staircase": "staircase_progress",
    "scale": "balance_scale",
    "funnel": "gravity_funnel",
    "dominoes": "domino_chain",
}

# "auto" is a UI value, not a template: it means "let the model choose".
AUTO_TEMPLATE = "auto"
DEFAULT_TEMPLATE = "split_path"
# The ceiling was 60.0, which is precisely the number TikTok Creator Rewards
# pays nothing at or below -- so the cap was silently clipping every scene to
# just short of monetizable.
#
# It is now well clear of the 62-75s target band rather than just above it. At
# 80.0 a thesis that ran long landed on the ceiling exactly, and hitting this
# cap is not harmless: the video is truncated to it and the last seconds of
# narration are cut off mid-sentence. A ceiling should be a guard against a
# runaway, not a value normal output lands on.
MIN_DURATION, MAX_DURATION = 8.0, 95.0


def resolve_template(name: Any) -> str:
    """A template key from anything -- a new name, an old name, or nonsense."""
    key = str(name or "").strip().lower()
    key = TEMPLATE_ALIASES.get(key, key)
    return key if key in TEMPLATES else DEFAULT_TEMPLATE


# ---------------------------------------------------------------------------
# Scene specs
# ---------------------------------------------------------------------------

def normalise_spec(raw: dict[str, Any] | None) -> dict[str, Any]:
    """
    Coerces anything -- a model response included -- into a renderable spec.

    Nothing downstream may assume a key exists or that a number is sane, so
    every field is defaulted and every range is clamped here, once.
    """
    raw = dict(raw or {})
    template = resolve_template(raw.get("template") or raw.get("metaphor_type"))
    preset = TEMPLATES[template]

    try:
        duration = float(raw.get("duration") or 18.0)
    except (TypeError, ValueError):
        duration = 18.0
    duration = max(MIN_DURATION, min(MAX_DURATION, duration))

    # animation_phases is the model's blueprint for the three beats; `climax`
    # is the older flat field. Either may be missing, so both are optional and
    # both get clamped into the duration.
    blueprint = raw.get("animation_phases") or raw.get("phases") or {}
    if not isinstance(blueprint, dict):
        blueprint = {}

    def _seconds(*keys: str) -> float:
        for key in keys:
            for source in (blueprint, raw):
                if key in source:
                    try:
                        value = float(source[key] or 0.0)
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        return value
        return 0.0

    climax = _seconds("impact", "climax") or duration * float(preset["climax"])
    climax = max(0.5, min(duration - 0.4, climax))
    draw_end = _seconds("draw_end", "draw")
    draw_end = min(draw_end, climax - 0.15) if draw_end else 0.0

    spec: dict[str, Any] = {
        "template": template,
        "title": str(raw.get("title") or preset["title"] or "").strip(),
        "subtitle": str(raw.get("subtitle") or preset["subtitle"] or "").strip(),
        "payoff": str(raw.get("payoff") or preset["payoff"] or "").strip(),
        "thesis": str(raw.get("thesis") or "").strip(),
        "duration": duration,
        "climax": climax,
        "draw_end": max(0.0, draw_end),
        "concept": str(raw.get("concept") or "").strip(),
        "source": str(raw.get("source") or "preset"),
        "elements": raw.get("elements") if isinstance(raw.get("elements"), list) else [],
        # 0 turns the grid and the drifting motes off, for a clean diagram look.
        "ambient": max(0.0, min(1.5, float(raw.get("ambient", 1.0) or 0.0))),
        "labels": raw.get("labels") if isinstance(raw.get("labels"), dict) else {},
        "publish": raw.get("publish") if isinstance(raw.get("publish"), dict) else {},
    }

    # A scene may be told in several acts, each with its own metaphor. One
    # metaphor holds attention for about twenty seconds; the runtime is now
    # over a minute, and stretching a single piece of geometry across it gives
    # a visual event roughly every twenty-three seconds. Measured on the first
    # 68-second render: the picture changed by 0.3% per second and 19 of 68
    # seconds were completely still.
    #
    # Acts are normalised recursively but carry no duration of their own until
    # the narration is synthesized -- see allocate_acts.
    acts = raw.get("acts")
    if isinstance(acts, list):
        built = [normalise_act(act) for act in acts if isinstance(act, dict)]
        if len(built) >= 2:
            spec["acts"] = built

    return spec


def _positive(value: Any) -> float:
    """A non-negative float, or 0.0 for anything unusable."""
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0


def normalise_act(raw: dict[str, Any]) -> dict[str, Any]:
    """
    One act of a multi-act scene.

    Deliberately not a full spec: an act has no duration, because its share of
    the runtime is decided by how much of the narration it carries. Its beat is
    stored as a fraction and turned into seconds at allocation time.
    """
    template = resolve_template(raw.get("template") or raw.get("metaphor_type"))
    preset = TEMPLATES[template]

    try:
        fraction = float(raw.get("climax_fraction") or preset["climax"])
    except (TypeError, ValueError):
        fraction = float(preset["climax"])

    return {
        "template": template,
        "title": str(raw.get("title") or preset["title"] or "").strip(),
        "subtitle": str(raw.get("subtitle") or preset["subtitle"] or "").strip(),
        "payoff": "",                      # the closing line belongs to the whole
        "thesis": str(raw.get("thesis") or "").strip(),
        "labels": raw.get("labels") if isinstance(raw.get("labels"), dict) else {},
        "elements": raw.get("elements") if isinstance(raw.get("elements"), list) else [],
        "ambient": max(0.0, min(1.5, float(raw.get("ambient", 1.0) or 0.0))),
        "climax_fraction": max(0.15, min(0.9, fraction)),
        "concept": str(raw.get("concept") or "").strip(),
        # Filled by allocate_acts -- and carried through if it already ran.
        #
        # This has to be idempotent. render_animation normalises the spec a
        # second time, after build_minimalist_video has allocated the act
        # timings, and zeroing them here silently collapsed every act to
        # start=0 seconds=0. act_at then fell past all of them to the last one
        # and drew its end state for the whole video: three acts planned, one
        # frozen frame rendered.
        "seconds": _positive(raw.get("seconds")),
        "start": max(0.0, _positive(raw.get("start"))),
        "climax": _positive(raw.get("climax")),
        "draw_end": _positive(raw.get("draw_end")),
    }


# The shortest an act can run and still register as its own idea rather than a
# flicker. Below this the cut reads as a glitch.
MIN_ACT_SECONDS = 8.0


def allocate_acts(spec: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    """
    Gives each act its share of the runtime, proportional to the words it
    carries.

    Proportional to narration rather than equal thirds: an act whose thesis is
    two sentences should not hold the screen as long as one with five, or the
    picture and the voice drift apart over the course of the video.
    """
    acts = [dict(act) for act in (spec.get("acts") or [])]
    if not acts:
        return []

    weights = [max(1, len(str(act.get("thesis") or "").split())) for act in acts]
    total_weight = float(sum(weights))

    # Floor first, then share what is left over by weight, so a short act still
    # gets long enough to read.
    floor = min(MIN_ACT_SECONDS, duration / len(acts))
    spare = max(0.0, duration - floor * len(acts))

    start = 0.0
    for act, weight in zip(acts, weights):
        seconds = floor + spare * (weight / total_weight)
        act["seconds"] = seconds
        act["start"] = start
        act["climax"] = max(0.5, min(seconds - 0.4,
                                     seconds * float(act["climax_fraction"])))
        act["draw_end"] = 0.0
        start += seconds

    # Absorb rounding into the last act so the acts sum to exactly `duration`.
    acts[-1]["seconds"] = max(0.1, duration - acts[-1]["start"])
    return acts


def act_at(spec: dict[str, Any], t: float,
           duration: float) -> tuple[dict[str, Any], float, float]:
    """
    (act, time within that act, that act's length) for the moment `t`.

    Falls back to the spec itself when there are no acts, which is what keeps
    every single-metaphor caller -- the presets, the offline fallback, every
    existing test -- working unchanged.
    """
    acts = spec.get("acts") or []
    if not acts:
        return spec, t, duration

    for act in acts:
        if t < act["start"] + act["seconds"]:
            return act, max(0.0, t - act["start"]), max(0.1, act["seconds"])

    last = acts[-1]
    return last, max(0.0, t - last["start"]), max(0.1, last["seconds"])


# A watchable default for the custom template when no model supplied geometry.
_CUSTOM_FALLBACK_ELEMENTS: list[dict[str, Any]] = [
    {"type": "path", "id": "rise", "colour": "white", "from": 1.0, "to": 13.0,
     "points": [[0.14, 0.72], [0.34, 0.68], [0.56, 0.58], [0.76, 0.42], [0.90, 0.33]]},
    {"type": "dot", "follows": "rise", "radius": 24, "colour": "white",
     "from": 1.0, "to": 13.0},
    {"type": "path", "id": "flat", "colour": "grey", "from": 1.0, "to": 13.0,
     "points": [[0.14, 0.72], [0.40, 0.73], [0.66, 0.745], [0.90, 0.76]]},
    {"type": "dot", "follows": "flat", "radius": 18, "colour": "grey",
     "from": 1.0, "to": 13.0},
    {"type": "bar", "at": [0.22, 0.88], "width": 0.055, "height": 0.10,
     "grow": True, "colour": "grey", "from": 2.0, "to": 13.0},
    {"type": "bar", "at": [0.78, 0.88], "width": 0.055, "height": 0.20,
     "grow": True, "colour": "white", "from": 2.0, "to": 13.0},
]


def fallback_scene_spec(concept: str, template: str = DEFAULT_TEMPLATE,
                        duration: float = 18.0) -> dict[str, Any]:
    """
    A complete spec with no API call at all.

    The mode has to work with the network down or the key missing, so the
    templates carry their own copy and this just fills in the concept.
    """
    resolved = resolve_template(template)
    preset = TEMPLATES[resolved]
    concept = (concept or "").strip()
    return normalise_spec({
        "template": resolve_template(template),
        "title": preset["title"] or (concept[:40].upper() if concept else "THINK LONGER"),
        # A typed concept short enough to read goes on screen as the subtitle --
        # otherwise the no-AI path would render a video that says nothing about
        # what the user actually asked for. Long preset concepts keep the
        # crafted line instead of a truncated sentence.
        "subtitle": concept if 0 < len(concept) <= 70 else preset["subtitle"],
        "payoff": preset["payoff"],
        "thesis": (f"{concept}. " if concept else "") + str(preset["blurb"]),
        "duration": duration,
        "concept": concept,
        "source": "preset",
        # The custom template has no built-in geometry, so with no model behind
        # it there is nothing to draw and it degrades to a title card -- ink
        # measured flat at 2.4% across the whole runtime. This gives it a real
        # scene to fall back to: one path that climbs, one that flattens.
        "elements": _CUSTOM_FALLBACK_ELEMENTS if resolved == "custom" else [],
    })


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

ProgressFn = Callable[[int, int, str], None]


def make_scene_frame(spec: dict[str, Any], t: float, duration: float) -> np.ndarray:
    """
    One finished RGB frame at time `t`. Public so tests can inspect a frame.

    With acts, the geometry and the title come from whichever act `t` falls in
    -- each gets its own local clock, so every act draws itself from the start
    rather than joining halfway through someone else's animation.

    Two things stay on the whole video's clock. The progress hairline is
    retention furniture and must promise the real remaining time, not the act's.
    And the payoff is the closing line of the argument, so it belongs to the
    end of the video rather than to the end of every act.
    """
    frame = Frame(reuse=True)
    act, local_t, act_seconds = act_at(spec, t, duration)

    scene: SceneFn = TEMPLATES[str(act["template"])]["fn"]
    scene(frame, local_t, act, act_seconds)
    draw_titles(frame, act, local_t)
    draw_footer(frame, spec, t, duration)
    draw_progress(frame, t, duration)
    return frame.finish()


def render_animation(
    spec: dict[str, Any],
    output_path: str,
    fps: int = 30,
    audio_path: str | None = None,
    progress_callback: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Renders the scene to an MP4, with an optional pre-mixed audio track.

    Audio is passed in already mixed rather than assembled here: the mix has to
    know the narration length before the duration is final, and that decision
    belongs one level up in `build_minimalist_video`.
    """
    from moviepy import AudioFileClip, VideoClip

    spec = normalise_spec(spec)
    duration = float(spec["duration"])
    total = max(1, int(duration * fps))
    drawn = {"n": 0}

    def make_frame(t: float) -> np.ndarray:
        drawn["n"] += 1
        if progress_callback and drawn["n"] % max(1, total // 12) == 0:
            progress_callback(min(drawn["n"], total), total,
                              f"Drawing frame {min(drawn['n'], total)}/{total}...")
        return make_scene_frame(spec, float(t), duration)

    clip = VideoClip(make_frame, duration=duration)
    open_clips: list[Any] = [clip]

    if audio_path and os.path.exists(audio_path):
        bed = AudioFileClip(audio_path)
        open_clips.append(bed)
        if bed.duration > duration:
            bed = bed.subclipped(0, duration)
        clip = clip.with_audio(bed)
        open_clips.append(clip)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if progress_callback:
        progress_callback(total, total, "Encoding MP4...")

    # Through the shared writer rather than straight to write_videofile. Two
    # things come with it, and this engine needs both more than any other
    # because it is the one people render most:
    #
    #   * A CPU retry. video_encoder() probes NVENC with a single 256x256
    #     frame, which proves very little: the driver can still give up on a
    #     1080x1920 stream when another process holds the encoder session, and
    #     consumer cards cap concurrent sessions. That failure lands at the end
    #     of a render that is now over a minute long, and throwing it away is
    #     the worst possible moment to find out.
    #   * Forced yuv420p. libx264 picks yuv444p for some inputs and the file
    #     then plays as a green screen on iOS -- which, for a TikTok-first
    #     engine, is the whole audience.
    write_clip(
        clip, output_path, fps=fps,
        with_audio=bool(audio_path and os.path.exists(audio_path)),
        progress_callback=(
            (lambda message: progress_callback(total, total, message))
            if progress_callback else None),
    )

    # Re-read after the write: a fallback poisons the probe, so asking now is
    # what actually happened rather than what was planned.
    enc = video_encoder()

    for item in open_clips:
        try:
            item.close()
        except Exception:
            pass

    return {
        "output_path": output_path,
        "duration": duration,
        "fps": fps,
        "frames": total,
        "template": spec["template"],
        "climax": spec["climax"],
        "size": CANVAS,
        "gpu": bool(enc.get("gpu")),
    }


def estimate_render_seconds(duration: float, fps: int = 30) -> float:
    """
    Rough wall-clock estimate, so the UI can warn before a long render.

    Measured on this machine at about 9.5 drawn frames per second.
    """
    return max(2.0, (duration * fps) / 9.5)


def scratch_wav(prefix: str) -> str:
    """
    A scratch audio path, registered for cleanup.

    Registering here rather than at the call sites means one
    `purge_scratch_renders()` clears the bed, the mix and the voice take along
    with the video intermediates -- there is no second cleanup path to forget.
    """
    import tempfile

    path = os.path.join(tempfile.gettempdir(), f"rf_{prefix}_{int(time.time() * 1000)}.wav")
    _SCRATCH_RENDERS.append(path)
    return path


# ---------------------------------------------------------------------------
# Publish metadata
# ---------------------------------------------------------------------------

BASE_HASHTAGS = ("#Mindset", "#Discipline", "#Psychology", "#Shorts")


def normalise_publish(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Coerces the publish block into something the UI can render safely.

    The four base tags are always present -- they are what the audience for
    this format actually searches -- with the model's suggestions appended and
    de-duplicated case-insensitively.
    """
    supplied_block = spec.get("publish")
    raw: dict[str, Any] = supplied_block if isinstance(supplied_block, dict) else {}
    title = str(raw.get("title") or spec.get("title") or "").strip()
    description = str(raw.get("description") or "").strip()

    tags: list[str] = []
    seen: set[str] = set()
    supplied = raw.get("hashtags")
    offered: list[Any] = supplied if isinstance(supplied, list) else []
    for tag in list(BASE_HASHTAGS) + offered:
        text = str(tag).strip()
        if not text:
            continue
        if not text.startswith("#"):
            text = "#" + text.lstrip("#")
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        tags.append(text)

    if not description:
        description = "\n\n".join(part for part in (
            str(spec.get("payoff") or "").strip(),
            str(spec.get("thesis") or "").strip(),
            "Animated from scratch - no stock footage, no filler.",
        ) if part)

    return {
        "title": (title or "The rule nobody tells you")[:100],
        "description": description,
        "hashtags": tags[:10],
    }


# ---------------------------------------------------------------------------
# Full build: audio mix + animation
# ---------------------------------------------------------------------------

def build_minimalist_video(
    spec: dict[str, Any],
    output_path: str,
    fps: int = 30,
    bgm: bool = True,
    bgm_volume: float = 0.30,
    sfx: bool = True,
    sfx_volume: float = 0.80,
    narrate: bool = False,
    voice: str = "Charon",
    tts_provider: str = "gemini",
    tts_style: str = "Calm, certain, unhurried. Almost cold.",
    progress_callback: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Renders the animation with its bed, its drop and (optionally) narration.

    Order matters: the voice is synthesized first because its length decides
    the length of everything else. A 22-word thesis that reads in 9 seconds
    should not stretch an 18-second animation, but a 31-word one that reads in
    20 must -- otherwise the last sentence is cut off mid-word.
    """
    from moviepy import AudioFileClip, CompositeAudioClip

    from audio_engine import (
        SFX_DROP, SUB_DROP_IMPACT, build_ducked_bed, get_sfx_path,
        synthesize_narration, trim_leading_silence,
    )

    spec = normalise_spec(spec)
    duration = float(spec["duration"])
    steps, step = 5, {"n": 0}

    def stage(message: str) -> None:
        step["n"] += 1
        if progress_callback:
            progress_callback(step["n"], steps, message)

    # --- 1. narration decides the runtime ---------------------------------
    narration_path = ""
    narration_trimmed = 0.0

    # With acts, the narration is the acts read end to end. It is synthesized
    # as one take rather than three: three takes joined leave an audible seam
    # at each boundary, and the whole point of allocating act time by word
    # count is that one continuous voice stays glued to the picture.
    thesis = str(spec.get("thesis") or "").strip()
    if spec.get("acts"):
        spoken_parts = [str(act.get("thesis") or "").strip()
                        for act in spec["acts"]]
        joined = " ".join(part for part in spoken_parts if part)
        if joined:
            thesis = joined
            spec["thesis"] = joined

    if narrate and thesis:
        stage("Stage 1/5 - Narration: synthesizing the thesis...")
        take = synthesize_narration(
            thesis, provider=tts_provider, voice=voice,
            output_path=scratch_wav("mmvoice"), style=tts_style,
        )
        # No leading dead air: the voice starts on the first frame, with the
        # animation, or the opening beat is wasted.
        cleaned = trim_leading_silence(str(take["path"]))
        narration_path = str(cleaned["path"])
        narration_trimmed = float(cleaned["trimmed"])
        if narration_path != str(take["path"]):
            _SCRATCH_RENDERS.append(narration_path)

        spoken = float(cleaned["duration"])
        if spoken + 1.6 > duration:
            grown = min(MAX_DURATION, spoken + 1.6)
            # Keep the climax at the same point in the story, not the clock.
            spec["climax"] = float(spec["climax"]) * (grown / duration)
            duration = grown
            spec["duration"] = duration
    else:
        stage("Stage 1/5 - Narration: skipped (silent cut).")

    climax = float(spec["climax"])

    # The runtime is final now, so the acts can be given their share of it.
    if spec.get("acts"):
        spec["acts"] = allocate_acts(spec, duration)

    # --- 2. bed ------------------------------------------------------------
    tracks: list[Any] = []
    open_clips: list[Any] = []

    if narration_path:
        voice_clip = AudioFileClip(narration_path)
        open_clips.append(voice_clip)
        tracks.append(voice_clip.with_start(0.0))

    if bgm:
        stage("Stage 2/5 - Audio: synthesizing the phonk bed...")
        bed_path = build_ducked_bed(
            duration + 0.2, style="phonk",
            narration_path=narration_path or None,
            output_path=scratch_wav("mmbed"),
            volume=float(bgm_volume),
        )
        bed = AudioFileClip(bed_path)
        open_clips.append(bed)
        if bed.duration > duration:
            bed = bed.subclipped(0, duration)
        tracks.append(bed)
    else:
        stage("Stage 2/5 - Audio: bed off.")

    # --- 3. the drop, landing on the climax frame --------------------------
    if sfx:
        stage("Stage 3/5 - Audio: placing the sub drop on the climax...")
        hit = AudioFileClip(get_sfx_path(SFX_DROP))
        open_clips.append(hit)
        start = max(0.0, climax - SUB_DROP_IMPACT)
        if start + hit.duration > duration:
            hit = hit.subclipped(0, max(0.05, duration - start))
        tracks.append(hit.with_volume_scaled(float(sfx_volume)).with_start(start))
    else:
        stage("Stage 3/5 - Audio: SFX off.")

    mixed_path = ""
    if tracks:
        stage("Stage 4/5 - Audio: mixing...")
        mix = CompositeAudioClip(tracks)
        mix.duration = duration
        mixed_path = scratch_wav("mmmix")
        mix.write_audiofile(mixed_path, fps=44100, logger=None)
        open_clips.append(mix)
    else:
        stage("Stage 4/5 - Audio: silent.")

    for item in open_clips:
        try:
            item.close()
        except Exception:
            pass

    # --- 4. draw and encode ------------------------------------------------
    stage("Stage 5/5 - Drawing frames...")

    def frame_progress(done: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(steps, steps, message)

    result = render_animation(
        spec, output_path, fps=fps,
        audio_path=mixed_path or None,
        progress_callback=frame_progress,
    )

    # The bed, the mix and the voice take have all been consumed by the encoder
    # by now. Purging here rather than only in the UI means every caller --
    # tests included -- leaves the temp directory as it found it.
    purge_scratch_renders()

    result.update({
        "narration": bool(narration_path),
        "narration_trimmed": narration_trimmed,
        "tts_provider": tts_provider if narration_path else "none",
        "voice": voice if narration_path else "",
        "bgm": bool(bgm),
        "sfx": bool(sfx),
        "thesis": thesis,
        "spec": spec,
        "publish": normalise_publish(spec),
    })
    return result
