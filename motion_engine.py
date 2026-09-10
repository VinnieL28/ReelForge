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

from paths import resolve_font
from video_engine import _SCRATCH_RENDERS, purge_scratch_renders, video_encoder

# ---------------------------------------------------------------------------
# Look
# ---------------------------------------------------------------------------

CANVAS: tuple[int, int] = (1080, 1920)
SUPERSAMPLE = 2                       # draw at 2x, box-filter down: free AA

BLACK: tuple[int, int, int] = (0, 0, 0)
WHITE: tuple[int, int, int] = (255, 255, 255)
GREY: tuple[int, int, int] = (136, 136, 136)          # #888888
DIM: tuple[int, int, int] = (44, 44, 44)              # unreached path, ticks
# Tuned by measuring the halo across a stroke: this gives a rim of ~37/255 at
# 6px that is gone by 90px. Tighter settings were invisible on a 1080-wide
# frame; wider ones stopped being an edge glow and became a bloom wash.
GLOW_STRENGTH = 1.2
GLOW_RADIUS = 24
# The halo is computed at a quarter resolution and scaled back up. It is a soft
# blur -- there is nothing in it above a quarter-res Nyquist -- and doing it at
# full size cost 83ms a frame against 26ms this way.
_GLOW_DIVISOR = 4
_GLOW_LUT = [min(255, int(v * GLOW_STRENGTH)) for v in range(256)] * 3

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

    def text(self, body: str, centre: tuple[float, float], size: int = 56,
             colour: tuple[int, int, int] = WHITE, weight: str = "bold",
             anchor: str = "mm", tracking: float = 0.0) -> None:
        """Draws one line. `tracking` spaces the letters, which reads as premium."""
        if not body:
            return
        s = self.ss
        font = load_face(int(size * s), weight)
        if tracking <= 0:
            self.draw.text((centre[0] * s, centre[1] * s), body, font=font,
                           fill=colour, anchor=anchor)
            return

        gap = tracking * s
        widths = [self.draw.textlength(ch, font=font) for ch in body]
        total = sum(widths) + gap * (len(body) - 1)
        x = centre[0] * s - total / 2
        for ch, w in zip(body, widths):
            self.draw.text((x, centre[1] * s), ch, font=font, fill=colour, anchor="lm")
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
        for i, line in enumerate(lines):
            self.text(line, (centre[0], top + i * step), size, colour, weight)

    # -- output ------------------------------------------------------------
    def finish(self, glow: float = GLOW_STRENGTH) -> np.ndarray:
        """Box-filters the supersampled canvas down and adds the bloom."""
        # reduce() is a dedicated box reduction: 14ms against 48ms for a
        # general resize at the same quality for an integer factor.
        sharp = self.image.reduce(self.ss) if self.ss > 1 else self.image
        if glow <= 0:
            return np.asarray(sharp, dtype=np.uint8)

        halo = (sharp.reduce(_GLOW_DIVISOR)
                .filter(ImageFilter.GaussianBlur(radius=GLOW_RADIUS / _GLOW_DIVISOR))
                .resize((self.w, self.h), Image.Resampling.BILINEAR))
        lut = _GLOW_LUT if glow == GLOW_STRENGTH else [min(255, int(v * glow)) for v in range(256)] * 3
        # Saturating uint8 add in C beats the same sum in float32 numpy.
        return np.asarray(ImageChops.add(sharp, halo.point(lut)), dtype=np.uint8)


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
        frame.wrapped(line, (frame.w / 2, 1720), size=52, colour=mix(WHITE, alpha),
                      weight="bold", max_width=900, leading=1.18)


def draw_progress(frame: Frame, t: float, duration: float) -> None:
    """A hairline time bar. Retention furniture: it promises the video is short."""
    y = 1866.0
    frame.line((90, y), (990, y), DIM, 4)
    done = clamp(t / max(duration, 1e-6))
    if done > 0:
        frame.line((90, y), (90 + 900 * done, y), GREY, 4)


# ---------------------------------------------------------------------------
# Template 1 -- The Curve of Pain vs. Comfort
# ---------------------------------------------------------------------------

_PAIN_CONTROL = [(120, 880), (250, 1120), (400, 1430), (560, 1330),
                 (700, 960), (850, 800), (990, 780)]
_EASY_CONTROL = [(120, 1020), (330, 1010), (520, 1015), (680, 1040),
                 (820, 1300), (900, 1560), (960, 1600)]


def _spikes(frame: Frame, x0: float, x1: float, base: float, height: float,
            count: int, colour: tuple[int, int, int]) -> None:
    step = (x1 - x0) / max(count, 1)
    for i in range(count):
        left = x0 + i * step
        frame.polygon([(left, base), (left + step / 2, base - height), (left + step, base)],
                      colour)


def scene_curve(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """A ball takes the steep drop and ends high; the flat path ends on spikes."""
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

    _spikes(frame, 800, 1000, 1660, 90, 5, mix(GREY, 0.55 + 0.45 * clamp((u - 0.7) / 0.3)))

    # The hard route: lit as it is travelled.
    frame.polyline(slice_to(pain, pain_len, u), WHITE, 8)
    px, py = point_at(pain, pain_len, u)
    frame.circle((px, py), 26, WHITE)
    frame.circle((px, py), 44, mix(WHITE, 0.35), 3)

    # Labels, tied to where the ball actually is rather than to the clock.
    if u > 0.16:
        frame.text("5 YEARS", (430, 1580), 44,
                   mix(GREY, fade(t, lead + span * 0.16, 0.5)), tracking=3)
    if u > 0.66:
        frame.text("50 YEARS", (760, 690), 46,
                   mix(WHITE, fade(t, lead + span * 0.66, 0.5)), tracking=3)
    if easy_u > 0.78:
        frame.text("COMFORT NOW", (312, 906), 36,
                   mix(GREY, fade(t, lead + span * 0.64, 0.5)), tracking=3)


# ---------------------------------------------------------------------------
# Template 2 -- The 1% Daily Vessel
# ---------------------------------------------------------------------------

_JAR = (330.0, 720.0, 750.0, 1440.0)         # left, top, right, bottom
_JAR_DAYS = 365


def scene_vessel(frame: Frame, t: float, spec: dict[str, Any], duration: float) -> None:
    """A jar filling on a 1.01^n curve: nothing, nothing, nothing, then everything."""
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

    frame.text(f"DAY {day}", (frame.w / 2, 1524), 38, GREY, tracking=5)
    frame.text(f"{growth:.2f}x", (frame.w / 2, 1600), 62,
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
        frame.text("RESISTANCE", (frame.w / 2, 700), 40, mix(GREY, caption), tracking=4)
    else:
        # Over the top: the sphere runs away downhill and the figure stands up.
        roll = ease_in(clamp((u - 0.80) / 0.20))
        top_x, top_y = stairs[-1]
        bx = top_x + roll * 90
        by = top_y - 34 + roll * roll * 320
        frame.line((top_x, top_y), (top_x + 130, top_y + 360), GREY, 6)
        frame.circle((bx, by), 32 + roll * 8, WHITE)
        _stick_figure(frame, (top_x - 96, top_y), 1.0, WHITE)
        frame.text("LEVERAGE", (frame.w / 2, 700), 46,
                   mix(WHITE, fade(t, lead + span * 0.82, 0.5)), tracking=6)

    # Effort is linear; reward is not. The crossing is the whole argument.
    base, height = 1600.0, 140.0
    for i, (label, value, colour) in enumerate((
        ("EFFORT", clamp(u), GREY),
        ("REWARD", clamp(u ** 3.2), WHITE),
    )):
        x = 220 + i * 640
        frame.line((x, base), (x, base - height), DIM, 5)
        frame.line((x, base), (x, base - height * value), colour, 11)
        frame.text(label, (x, base + 44), 26, colour, weight="light", tracking=3)


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
# Templates registry
# ---------------------------------------------------------------------------

SceneFn = Callable[[Frame, float, dict[str, Any], float], None]

TEMPLATES: dict[str, dict[str, Any]] = {
    "curve": {
        "label": "The Curve of Pain vs. Comfort",
        "blurb": "A ball takes the steep drop and ends high; the flat road ends on spikes.",
        "fn": scene_curve,
        "title": "5 YEARS OF PAIN",
        "subtitle": "buys fifty years of comfort",
        "payoff": "Choose your hard.",
        "climax": 0.72,
    },
    "vessel": {
        "label": "The 1% Daily Vessel",
        "blurb": "A jar filling on a 1.01^n curve: nothing, nothing, then everything.",
        "fn": scene_vessel,
        "title": "1% BETTER",
        "subtitle": "every single day for a year",
        "payoff": "37x. That is the whole secret.",
        "climax": 0.88,
    },
    "staircase": {
        "label": "The Resistance Staircase",
        "blurb": "Linear effort, exponential reward, and the moment the slope flips.",
        "fn": scene_staircase,
        "title": "PUSH LONG ENOUGH",
        "subtitle": "and the hill starts pushing back",
        "payoff": "Resistance becomes leverage.",
        "climax": 0.80,
    },
    "custom": {
        "label": "Custom Gemini Vector",
        "blurb": "Renders coordinates and commands the model writes for your concept.",
        "fn": scene_custom,
        "title": "",
        "subtitle": "",
        "payoff": "",
        "climax": 0.75,
    },
}

DEFAULT_TEMPLATE = "curve"
MIN_DURATION, MAX_DURATION = 8.0, 60.0


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
    template = str(raw.get("template") or DEFAULT_TEMPLATE).strip().lower()
    if template not in TEMPLATES:
        template = DEFAULT_TEMPLATE
    preset = TEMPLATES[template]

    try:
        duration = float(raw.get("duration") or 18.0)
    except (TypeError, ValueError):
        duration = 18.0
    duration = max(MIN_DURATION, min(MAX_DURATION, duration))

    try:
        climax = float(raw.get("climax") or 0.0) or duration * float(preset["climax"])
    except (TypeError, ValueError):
        climax = duration * float(preset["climax"])
    climax = max(0.5, min(duration - 0.4, climax))

    spec: dict[str, Any] = {
        "template": template,
        "title": str(raw.get("title") or preset["title"] or "").strip(),
        "subtitle": str(raw.get("subtitle") or preset["subtitle"] or "").strip(),
        "payoff": str(raw.get("payoff") or preset["payoff"] or "").strip(),
        "thesis": str(raw.get("thesis") or "").strip(),
        "duration": duration,
        "climax": climax,
        "concept": str(raw.get("concept") or "").strip(),
        "source": str(raw.get("source") or "preset"),
        "elements": raw.get("elements") if isinstance(raw.get("elements"), list) else [],
        "publish": raw.get("publish") if isinstance(raw.get("publish"), dict) else {},
    }
    return spec


def fallback_scene_spec(concept: str, template: str = DEFAULT_TEMPLATE,
                        duration: float = 18.0) -> dict[str, Any]:
    """
    A complete spec with no API call at all.

    The mode has to work with the network down or the key missing, so the
    templates carry their own copy and this just fills in the concept.
    """
    preset = TEMPLATES.get(template, TEMPLATES[DEFAULT_TEMPLATE])
    concept = (concept or "").strip()
    return normalise_spec({
        "template": template,
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
    })


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

ProgressFn = Callable[[int, int, str], None]


def make_scene_frame(spec: dict[str, Any], t: float, duration: float) -> np.ndarray:
    """One finished RGB frame at time `t`. Public so tests can inspect a frame."""
    frame = Frame(reuse=True)
    scene: SceneFn = TEMPLATES[str(spec["template"])]["fn"]
    scene(frame, t, spec, duration)
    draw_titles(frame, spec, t)
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
    enc = video_encoder()
    if progress_callback:
        progress_callback(total, total, "Encoding MP4...")

    clip.write_videofile(
        output_path, fps=fps, codec=enc["codec"],
        audio=bool(audio_path and os.path.exists(audio_path)),
        audio_codec="aac" if audio_path else None,
        preset=enc["preset"], ffmpeg_params=list(enc["ffmpeg_params"]),
    )

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
    thesis = str(spec.get("thesis") or "").strip()

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
