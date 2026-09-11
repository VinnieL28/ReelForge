# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false, reportUnknownLambdaType=false
"""
Versus Duel engine: split-screen comparison rounds, animated stat cards and a
single winner reveal.

Split out of video_engine so the duel has one home. video_engine imports this
lazily (inside build_reel_video) rather than at module scope, so the dependency
runs one way only and there is no import cycle.

Three things this module is careful about, each of which was a visible bug:

* **The title slot is shared.** The intro's hook, every round's metric header
  and the winner card all want the top of the frame. Reels are cut with a
  crossfade, which blends whole frames -- so two titles in the same slot ghost
  through each other for the length of the transition. Every clip therefore
  *retires* its title before its own tail (`title_gate`), and the slot is empty
  on both sides of the cut.

* **One WINNER, once.** The round card's WINNER tag is retired the same way,
  so the last round cannot dissolve into the winner card and show the word
  twice in two places.

* **Static layers are drawn once.** The panel scrims and the centre divider do
  not depend on `t`, but were rebuilt for every frame -- 1920 Python-level
  line draws per panel per frame. They are now baked into one cached overlay
  pasted with a mask, which is most of the 2.3x speedup on frame generation.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from moviepy import VideoClip

from video_engine import RGB, _load_bold_font, make_watermark_tile

DUEL_ACCENT_A: RGB = (34, 211, 238)     # cyan
DUEL_ACCENT_B: RGB = (251, 191, 36)     # amber
DUEL_GOLD: RGB = (255, 199, 61)

_GLASS_FILL = (14, 17, 24, 176)         # translucent card body
_GLASS_EDGE = (255, 255, 255, 46)
_GLASS_HILITE = (255, 255, 255, 92)     # top bevel that sells the glass

# How long before a clip ends the shared title slot is vacated, and how long
# after it starts the title takes to arrive. Both must exceed the longest
# transition the UI offers (1.5s) / 2 for the slot to be genuinely empty at the
# midpoint of the blend; 0.85 covers the 0.5s default with room to spare.
TITLE_OUT = 0.85
TITLE_IN = 0.28

# The top strip the shared title slot occupies. Fades are composited through a
# tile this tall rather than the whole frame, which keeps the cost of a fade to
# a tenth of a full-canvas allocation.
TITLE_BAND_H = 340

# Ken Burns source headroom. 1.34 meant every frame resampled a 1447px-wide
# intermediate down to 1080; 1.18 keeps the same 12% move with ~30% fewer
# pixels through the resampler, which is real time on a 600-frame render.
PANEL_HEADROOM = 1.18


def _ease_out_back(t: float, overshoot: float = 1.70158) -> float:
    """Pop easing: overshoots slightly then settles, so cards 'snap' in."""
    t = min(1.0, max(0.0, t)) - 1.0
    return t * t * ((overshoot + 1.0) * t + overshoot) + 1.0


def _ease_out_cubic(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def title_gate(t: float, duration: float,
               lead: float = TITLE_IN, tail: float = TITLE_OUT) -> float:
    """
    Opacity of anything occupying the shared title slot, 0..1.

    Zero at both ends of the clip. The reason is the crossfade: MoviePy blends
    entire frames, so a title that is still lit on the outgoing clip is drawn
    on top of the incoming clip's title for the whole transition. Clearing the
    slot first is the only fix that does not require the clip to know how it
    will be cut.
    """
    if duration <= 0:
        return 0.0
    if t <= 0.0:
        return 0.0

    rise = min(1.0, t / lead) if lead > 0 else 1.0
    remaining = duration - t
    fall = min(1.0, remaining / tail) if tail > 0 else 1.0
    return max(0.0, min(1.0, min(rise, fall)))


def fade_tile(size: tuple[int, int], alpha: float,
              paint: Any) -> tuple[Image.Image, Image.Image] | None:
    """
    Draws through `paint` onto a transparent tile and returns it at `alpha`.

    This exists because of a PIL behaviour that is easy to miss and silently
    defeats every fade: **ImageDraw.text() ignores the alpha channel of its
    fill**. Shapes honour it -- a rounded_rectangle filled (255,255,255,9) on an
    "RGBA"-mode draw really is nearly invisible -- but text uses the glyph mask
    as the paste mask and throws the ink's alpha away, so the same 9/255 white
    renders at full brightness. Measured: max pixel 255, with and without a
    stroke.

    So anything that has to fade gets drawn opaque onto its own tile, and the
    tile's whole alpha channel is scaled. Returns (rgb, mask) to paste, or None
    when there is nothing to show.
    """
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 0.004:
        return None

    tile = Image.new("RGBA", size, (0, 0, 0, 0))
    paint(ImageDraw.Draw(tile, "RGBA"))

    mask = tile.getchannel("A")
    if alpha < 0.999:
        mask = mask.point(lambda v: int(v * alpha))
    return tile.convert("RGB"), mask


def paint_faded(frame: Image.Image, alpha: float, paint: Any) -> None:
    """
    Draws `paint` onto `frame` at `alpha`, from the frame's own origin.

    The tile in `fade_tile` is only needed while something is actually fading.
    Routing every frame through it costs a full-canvas RGBA allocation, a draw,
    a convert and a masked paste -- measured at 188ms/frame across a finished
    duel against 55ms for a round clip, because the winner card was rebuilding
    a 1080x1920 tile for all 108 of its frames to fade during the first 13.

    So: opaque draws straight onto the frame, and only a genuine fade pays for
    the tile.
    """
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 0.004:
        return
    if alpha >= 0.999:
        paint(ImageDraw.Draw(frame, "RGBA"))
        return

    tile = fade_tile(frame.size, alpha, paint)
    if tile is not None:
        frame.paste(tile[0], (0, 0), tile[1])


def crossfade_sequence(clips: list[Any], transition: float) -> tuple[Any, list[float]]:
    """
    Lays clips end to end with overlapping crossfades, as one VideoClip.

    Returns (clip, start times). The start times are the plain cumulative
    starts -- `sum(durations[:k]) - k * transition` -- and are what the audio
    layer anchors narration and SFX to, so they must not drift.

    Why not CompositeVideoClip: `vfx.CrossFadeIn` attaches a mask to a clip,
    and MoviePy keeps that mask for the clip's entire duration rather than only
    across the transition. Every frame therefore takes the masked path in
    compose_on -- a PIL conversion, a full-canvas alpha_composite, four
    uint8/float casts and a convert back. Measured on a 1080x1920 duel: 54ms to
    draw a round frame, 131ms to get that same frame out of the composite, and
    230ms inside a transition.

    Here a frame outside a transition is the slide's own frame, returned
    untouched, and a frame inside one is a single numpy lerp of two.
    """
    if not clips:
        raise ValueError("crossfade_sequence needs at least one clip")

    transition = max(0.0, float(transition))
    if len(clips) > 1:
        transition = min(transition, min(float(c.duration) for c in clips) * 0.4)

    starts: list[float] = []
    cursor = 0.0
    for clip in clips:
        starts.append(cursor)
        cursor += float(clip.duration) - transition
    total = cursor + transition

    spans = [(starts[i], starts[i] + float(c.duration)) for i, c in enumerate(clips)]

    def frame_at(index: int, t: float):
        local = min(max(0.0, t - starts[index]), float(clips[index].duration) - 1e-4)
        return clips[index].get_frame(local)

    def make_frame(t: float):
        t = min(max(0.0, float(t)), total - 1e-4)

        # The later clip wins when two overlap, so find the last one that has
        # started; the one before it is the only possible partner.
        index = 0
        for i, (start, end) in enumerate(spans):
            if start <= t < end:
                index = i
        top = frame_at(index, t)

        if index == 0 or transition <= 0:
            return top

        into = t - starts[index]
        if into >= transition:
            return top

        under = frame_at(index - 1, t)
        blend = into / transition
        # float32 rather than float64: same result to well under one code
        # value, at half the memory traffic on a 6MB frame.
        return (under.astype(np.float32) * (1.0 - blend)
                + top.astype(np.float32) * blend).astype(np.uint8)

    return VideoClip(make_frame, duration=total), starts


def format_stat(value: float, unit: str = "", decimals: int | None = None) -> str:
    """
    Formats a duel score: '$1,299', '23h', '3.8s', '94'.

    `decimals` should be pinned to the *final* score's precision while a counter
    animates -- inferring it per frame makes an integer target flicker through
    values like '$323.04' on the way up.
    """
    if decimals is None:
        decimals = 0 if float(value).is_integer() else 1
    body = f"{value:,.{decimals}f}"
    return f"${body}" if unit == "$" else f"{body}{unit}"


def duel_round_beats(duration: float) -> dict[str, float]:
    """
    The animation clock for one round, shared by the renderer and the SFX
    scheduler so hits land exactly on the frame the card appears.
    """
    return {
        "a_in": 0.12 * duration,
        "b_in": 0.42 * duration,
        "count": max(0.45, 0.22 * duration),
        "reveal": 0.72 * duration,
    }


def _draw_trophy(draw: ImageDraw.ImageDraw, xy: tuple[float, float], size: float, color: RGB) -> None:
    """Draws a trophy glyph -- the display fonts carry no emoji coverage."""
    x, y = xy
    cup_w, cup_h = size * 0.62, size * 0.52
    fill = color + (255,)

    draw.rounded_rectangle([x, y, x + cup_w, y + cup_h], radius=int(size * 0.10), fill=fill)
    draw.pieslice([x + cup_w * 0.10, y + cup_h * 0.45, x + cup_w * 0.90, y + cup_h * 1.25], 0, 180, fill=fill)
    for side in (0, 1):
        hx = x - size * 0.16 if side == 0 else x + cup_w
        draw.arc([hx, y + cup_h * 0.05, hx + size * 0.16, y + cup_h * 0.62],
                 start=(90 if side == 0 else 270), end=(270 if side == 0 else 90),
                 fill=fill, width=max(2, int(size * 0.07)))
    draw.rectangle([x + cup_w * 0.42, y + cup_h * 0.95, x + cup_w * 0.58, y + size * 0.80], fill=fill)
    draw.rounded_rectangle([x + cup_w * 0.18, y + size * 0.78, x + cup_w * 0.82, y + size * 0.94],
                           radius=int(size * 0.05), fill=fill)


def _wrap_to_width(draw: ImageDraw.ImageDraw, text: str, font: Any, max_w: int) -> list[str]:
    """Greedy word wrap against a pixel width."""
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for word in words:
        trial = " ".join(cur + [word])
        if cur and draw.textlength(trial, font=font) > max_w:
            lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return lines or [text]


def _panel_bounds(size: tuple[int, int], layout: str) -> tuple[list[int], list[int], list[int]]:
    """
    Full-bleed panel boxes plus the divider band between them.

    Panels butt directly against the divider -- edge-to-edge photography reads
    far more premium than a photo floating inside a card.
    """
    w, h = size
    if layout == "side_by_side":
        band = int(w * 0.018)
        mid = w // 2
        return ([0, 0, mid - band // 2, h],
                [mid + band // 2, 0, w, h],
                [mid - band // 2, 0, mid + band // 2, h])

    band = int(h * 0.016)
    mid = int(h * 0.49)
    return ([0, 0, w, mid - band // 2],
            [0, mid + band // 2, w, h],
            [0, mid - band // 2, w, mid + band // 2])


def _prep_panel_source(img: Image.Image, box: list[int], headroom: float = PANEL_HEADROOM) -> Image.Image:
    """
    Pre-scales a source photo to just above panel size.

    Ken Burns then crops a window out of this each frame; resizing from a
    modest intermediate rather than a 1920px original is what keeps the render
    fast enough to be usable.
    """
    pw, ph = box[2] - box[0], box[3] - box[1]
    tw, th = int(pw * headroom), int(ph * headroom)

    src = img.convert("RGB")
    sw, sh = src.size
    scale = max(tw / sw, th / sh)
    src = src.resize((max(1, int(sw * scale)), max(1, int(sh * scale))), Image.Resampling.LANCZOS)

    sw, sh = src.size
    left, top = (sw - tw) // 2, (sh - th) // 2
    return src.crop((left, top, left + tw, top + th))


def _ken_burns_panel(source: Image.Image, box: list[int], progress: float, zoom_in: bool) -> Image.Image:
    """Crops a slowly drifting window out of the prepped source for one frame."""
    pw, ph = box[2] - box[0], box[3] - box[1]
    sw, sh = source.size

    # 1.0 -> full source (widest); larger zoom = tighter crop.
    span = 0.86 if zoom_in else 0.98
    drift = 0.12 * (progress if zoom_in else (1.0 - progress))
    factor = span - drift

    cw, ch = max(2, int(sw * factor)), max(2, int(sh * factor))
    x = (sw - cw) // 2
    y = (sh - ch) // 2
    return source.crop((x, y, x + cw, y + ch)).resize((pw, ph), Image.Resampling.BILINEAR)


def _panel_scrim(size: tuple[int, int], accent: RGB) -> Image.Image:
    """
    Gradient scrim laid over a panel photo.

    Darkens BOTH ends and leaves the middle clear: the name plate sits at the
    top of a panel and the stat card at the bottom, so a single-ended ramp
    always left one of them stranded on bright photography.
    """
    pw, ph = size
    grad = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    for i in range(ph):
        p = i / max(1, ph - 1)
        # Distance from the clear middle band, 0 at centre -> 1 at either edge.
        d = min(1.0, abs(p - 0.5) * 2.0)
        shade = int(214 * (d ** 1.9))
        tint = d ** 2.4
        gd.line([(0, i), (pw, i)], fill=(
            int(accent[0] * 0.16 * tint), int(accent[1] * 0.16 * tint), int(accent[2] * 0.18 * tint), shade,
        ))
    return grad


def _divider_glow(size: tuple[int, int], divider: tuple[int, ...], layout: str,
                  accent_a: RGB, accent_b: RGB) -> Image.Image:
    """High-gloss centre divider with cyan/amber neon bleed into both panels."""
    w, h = size
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)

    if layout == "side_by_side":
        cx = (divider[0] + divider[2]) // 2
        reach = int(w * 0.055)
        for i in range(reach):
            k = (1.0 - i / reach) ** 2.3
            gd.line([(cx - i, 0), (cx - i, h)], fill=accent_a + (int(120 * k),))
            gd.line([(cx + i, 0), (cx + i, h)], fill=accent_b + (int(120 * k),))
        gd.line([(cx, 0), (cx, h)], fill=(255, 255, 255, 226), width=max(2, int(w * 0.004)))
    else:
        cy = (divider[1] + divider[3]) // 2
        reach = int(h * 0.045)
        for i in range(reach):
            k = (1.0 - i / reach) ** 2.3
            gd.line([(0, cy - i), (w, cy - i)], fill=accent_a + (int(120 * k),))
            gd.line([(0, cy + i), (w, cy + i)], fill=accent_b + (int(120 * k),))
        gd.line([(0, cy), (w, cy)], fill=(255, 255, 255, 226), width=max(2, int(h * 0.0022)))

    return glow


@lru_cache(maxsize=8)
def _static_layer(
    size: tuple[int, int], layout: str, accent_a: RGB, accent_b: RGB,
) -> tuple[Image.Image, Image.Image]:
    """
    The whole non-animating overlay -- both panel scrims and the divider glow --
    baked once into an RGB image plus its alpha mask.

    Nothing here depends on `t`. Rebuilding it per frame cost ~48ms of a 100ms
    frame (3 full-canvas alpha composites, 4 RGBA conversions and 3,845
    Python-level line draws), which is what made a duel freeze at Stage 4/4.
    Pasting a cached RGB tile through a cached mask is one blit.
    """
    box_a, box_b, divider = _panel_bounds(size, layout)

    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    for box, accent in ((box_a, accent_a), (box_b, accent_b)):
        pw, ph = box[2] - box[0], box[3] - box[1]
        layer.paste(_panel_scrim((pw, ph), accent), (box[0], box[1]))

    # `over` is associative, so compositing the divider onto the scrim layer
    # here is identical to drawing it onto the finished frame afterwards.
    layer = Image.alpha_composite(
        layer, _divider_glow(size, tuple(divider), layout, accent_a, accent_b))

    return layer.convert("RGB"), layer.getchannel("A")


def _glass_card(
    draw: ImageDraw.ImageDraw,
    box: list[int],
    accent: RGB,
    alpha: float = 1.0,
    glow: float = 0.0,
) -> None:
    """Translucent glass panel: dark body, bright top bevel, accent edge."""
    radius = int(min(box[2] - box[0], box[3] - box[1]) * 0.22)
    radius = max(10, min(radius, 30))
    a = max(0.0, min(1.0, alpha))

    if glow > 0:
        for i, spread in enumerate((12, 7, 3)):
            draw.rounded_rectangle(
                [box[0] - spread, box[1] - spread, box[2] + spread, box[3] + spread],
                radius=radius + spread, outline=accent + (int(150 * glow / (i + 1.5)),), width=3,
            )

    body = (_GLASS_FILL[0], _GLASS_FILL[1], _GLASS_FILL[2], int(_GLASS_FILL[3] * a))
    draw.rounded_rectangle(box, radius=radius, fill=body,
                           outline=(_GLASS_EDGE[0], _GLASS_EDGE[1], _GLASS_EDGE[2], int(_GLASS_EDGE[3] * a)),
                           width=2)
    draw.rounded_rectangle(box, radius=radius,
                           outline=accent + (int((90 + 150 * glow) * a),), width=int(2 + 2 * glow))
    # Top bevel highlight -- the cue that reads as "glass" at a glance.
    draw.line([(box[0] + radius, box[1] + 2), (box[2] - radius, box[1] + 2)],
              fill=(_GLASS_HILITE[0], _GLASS_HILITE[1], _GLASS_HILITE[2], int(_GLASS_HILITE[3] * a)), width=2)


def _progress_bar(
    draw: ImageDraw.ImageDraw,
    box: list[int],
    fraction: float,
    accent: RGB,
    alpha: float = 1.0,
) -> None:
    """Animated magnitude bar: dark track with an accent fill and a hot tip."""
    x0, y0, x1, y1 = box
    h = y1 - y0
    r = max(2, h // 2)
    a = max(0.0, min(1.0, alpha))

    draw.rounded_rectangle(box, radius=r, fill=(255, 255, 255, int(26 * a)))
    frac = max(0.0, min(1.0, fraction))
    if frac <= 0.001:
        return

    fill_w = max(h, int((x1 - x0) * frac))
    draw.rounded_rectangle([x0, y0, x0 + fill_w, y1], radius=r, fill=accent + (int(232 * a),))
    # Bright leading edge so the fill reads as energy, not a static block.
    tip = min(x0 + fill_w, x1)
    draw.ellipse([tip - r - 1, y0 - 1, tip + r + 1, y1 + 1], fill=(255, 255, 255, int(200 * a)))


def _draw_name_plate(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    width: int,
    side: str,
    name: str,
    hook: str,
    accent: RGB,
    canvas_w: int,
) -> int:
    """Side tag + product name + hook. Returns the y below the block."""
    x, y = origin

    tag_font = _load_bold_font(max(16, int(canvas_w * 0.024)))
    tag_h = int(canvas_w * 0.040)
    tag_w = int(canvas_w * 0.052)
    draw.rounded_rectangle([x, y, x + tag_w, y + tag_h], radius=int(tag_h * 0.30), fill=accent + (240,))
    tb = draw.textbbox((0, 0), side.upper(), font=tag_font)
    draw.text((x + (tag_w - (tb[2] - tb[0])) / 2 - tb[0], y + (tag_h - (tb[3] - tb[1])) / 2 - tb[1]),
              side.upper(), font=tag_font, fill=(6, 8, 14))

    name_font = _load_bold_font(int(canvas_w * 0.050))
    line_step = int(canvas_w * 0.060)
    ny = y + tag_h + int(canvas_w * 0.018)
    lines = _wrap_to_width(draw, name, name_font, width)[:2]
    for i, line in enumerate(lines):
        draw.text((x, ny + i * line_step), line, font=name_font,
                  fill=(255, 255, 255), stroke_width=max(2, int(canvas_w * 0.004)), stroke_fill=(0, 0, 0, 220))
    # Clear the full height of the name block before the hook goes under it.
    ny += len(lines) * line_step + int(canvas_w * 0.006)

    if hook:
        hook_font = _load_bold_font(int(canvas_w * 0.027))
        draw.text((x, ny), hook, font=hook_font, fill=(212, 220, 236),
                  stroke_width=2, stroke_fill=(0, 0, 0, 205))
        ny += int(canvas_w * 0.038)

    return ny


def _draw_vs_medallion(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, scale: float = 1.0) -> None:
    """Chromed VS medallion that sits on the divider."""
    cx, cy = center
    r = max(8, int(radius * scale))

    for i, spread in enumerate((16, 9, 4)):
        draw.ellipse([cx - r - spread, cy - r - spread, cx + r + spread, cy + r + spread],
                     outline=(255, 255, 255, int(66 / (i + 1.3))), width=3)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(9, 11, 17, 248), outline=(255, 255, 255, 132), width=3)
    draw.arc([cx - r, cy - r, cx + r, cy + r], start=200, end=340, fill=(255, 255, 255, 190), width=3)

    f = _load_bold_font(max(12, int(r * 1.02)))
    b = draw.textbbox((0, 0), "VS", font=f)
    draw.text((cx - (b[2] - b[0]) / 2 - b[0], cy - (b[3] - b[1]) / 2 - b[1]), "VS", font=f, fill=(255, 255, 255))


def _compose_duel_frame(
    size: tuple[int, int],
    sources: dict[str, Image.Image],
    boxes: dict[str, list[int]],
    layout: str,
    progress: float,
    accent_a: RGB,
    accent_b: RGB,
) -> Image.Image:
    """Builds the photographic base for one frame: both panels + the static overlay."""
    w, h = size
    frame = Image.new("RGB", (w, h), (8, 9, 13))

    for key, zoom_in in (("a", True), ("b", False)):
        box = boxes[key]
        frame.paste(_ken_burns_panel(sources[key], box, progress, zoom_in), (box[0], box[1]))

    overlay_rgb, overlay_alpha = _static_layer(size, layout, accent_a, accent_b)
    frame.paste(overlay_rgb, (0, 0), overlay_alpha)
    return frame


def _duel_sources(
    size: tuple[int, int],
    item_a: dict[str, Any],
    item_b: dict[str, Any],
    layout: str,
) -> tuple[dict[str, Image.Image], dict[str, list[int]]]:
    """Prepares panel boxes and pre-scaled Ken Burns sources for both sides."""
    box_a, box_b, divider = _panel_bounds(size, layout)
    boxes = {"a": box_a, "b": box_b, "divider": divider}

    sources: dict[str, Image.Image] = {}
    for key, item, box in (("a", item_a, box_a), ("b", item_b, box_b)):
        img = item.get("image")
        if not isinstance(img, Image.Image):
            pw, ph = box[2] - box[0], box[3] - box[1]
            img = Image.new("RGB", (max(2, pw), max(2, ph)), (16, 18, 26))
        sources[key] = _prep_panel_source(img, box)

    return sources, boxes


def create_duel_intro_clip(
    size: tuple[int, int],
    item_a: dict[str, Any],
    item_b: dict[str, Any],
    headline: str = "",
    duration: float = 4.0,
    layout: str = "stacked",
    watermark_text: str = "",
    accent_a: RGB = DUEL_ACCENT_A,
    accent_b: RGB = DUEL_ACCENT_B,
) -> VideoClip:
    """Opening card: both contenders full-bleed, the hook line, and a VS slam."""
    w, h = size
    sources, boxes = _duel_sources(size, item_a, item_b, layout)
    watermark = make_watermark_tile((w, h), watermark_text, position="bottom_right")

    vx = (boxes["divider"][0] + boxes["divider"][2]) // 2
    vy = (boxes["divider"][1] + boxes["divider"][3]) // 2
    r0 = int(min(w, h) * 0.056)

    name_a = str(item_a.get("name", "Item A"))
    name_b = str(item_b.get("name", "Item B"))
    hook_a = str(item_a.get("hook", "") or "")
    hook_b = str(item_b.get("hook", "") or "")

    def make_frame(t: float):
        p = min(1.0, t / duration)
        frame = _compose_duel_frame(size, sources, boxes, layout, p, accent_a, accent_b)
        draw = ImageDraw.Draw(frame, "RGBA")

        margin = int(w * 0.06)
        plate_w = int(w * (0.40 if layout == "side_by_side" else 0.62))
        # Clear the headline banner that sits over panel A. The reservation is
        # unconditional: the headline fades out at the tail, and a name plate
        # that slid up to fill the gap would read as a layout glitch.
        head_clear = int(h * 0.115) if (headline and layout != "side_by_side") else int(h * 0.055)
        _draw_name_plate(draw, (boxes["a"][0] + margin, boxes["a"][1] + head_clear),
                         plate_w, "a", name_a, hook_a, accent_a, w)
        _draw_name_plate(draw, (boxes["b"][0] + margin, boxes["b"][1] + int(h * 0.045)),
                         plate_w, "b", name_b, hook_b, accent_b, w)

        # The hook owns the title slot, and hands it back before the cut so the
        # first round's metric header does not dissolve through it. The band
        # starts at (0, 0), so tile coordinates and frame coordinates agree.
        gate = title_gate(t, duration)
        if headline:
            def paint_headline(td: ImageDraw.ImageDraw) -> None:
                f = _load_bold_font(int(w * 0.050))
                lines = _wrap_to_width(td, headline.upper(), f, int(w * 0.86))[:2]
                # Lifts away as it goes, so the exit reads as a move rather than
                # a dip in exposure.
                dy = int((1.0 - gate) * h * 0.030)
                for i, line in enumerate(lines):
                    lw = td.textlength(line, font=f)
                    td.text(((w - lw) / 2, int(h * 0.022) - dy + i * int(w * 0.058)),
                            line, font=f, fill=(255, 255, 255),
                            stroke_width=max(3, int(w * 0.006)), stroke_fill=(0, 0, 0, 238))

            # The band starts at the frame origin, so the opaque fast path can
            # draw the painter straight onto the frame.
            if gate >= 0.999:
                paint_headline(draw)
            else:
                tile = fade_tile((w, TITLE_BAND_H), gate, paint_headline)
                if tile is not None:
                    frame.paste(tile[0], (0, 0), tile[1])

        slam = _ease_out_back(min(1.0, t / 0.5))
        scale = (2.3 - 1.3 * slam) if t < 0.5 else 1.0 + 0.035 * math.sin(t * 3.4)
        _draw_vs_medallion(draw, (vx, vy), r0, scale)

        if watermark is not None:
            tile, pos = watermark
            frame.paste(tile, pos, tile)
        return np.array(frame)

    return VideoClip(make_frame, duration=duration)


def create_duel_round_clip(
    size: tuple[int, int],
    item_a: dict[str, Any],
    item_b: dict[str, Any],
    round_info: dict[str, Any],
    duration: float = 5.0,
    layout: str = "stacked",
    watermark_text: str = "",
    accent_a: RGB = DUEL_ACCENT_A,
    accent_b: RGB = DUEL_ACCENT_B,
) -> VideoClip:
    """
    One comparison round over live photography: glass stat cards snap in one
    after the other, their numbers count up, magnitude bars fill, then the
    winning side is revealed with a glow and WINNER tag.

    The metric header and the WINNER tag both live on `title_gate`, so this
    clip arrives with an empty title slot and leaves with one -- whatever it is
    cut against.
    """
    w, h = size
    sources, boxes = _duel_sources(size, item_a, item_b, layout)
    watermark = make_watermark_tile((w, h), watermark_text, position="bottom_right")

    metric = str(round_info.get("metric", "Round"))
    note = str(round_info.get("note", "") or "")
    unit = str(round_info.get("unit", "") or "")
    a_score = float(round_info.get("a_score", 0) or 0)
    b_score = float(round_info.get("b_score", 0) or 0)
    winner = str(round_info.get("winner", "") or "").upper()

    # Pin counter precision to the final values so integers never flicker decimals.
    a_dec = 0 if a_score.is_integer() else 1
    b_dec = 0 if b_score.is_integer() else 1

    # Bars show "how well this side did", not raw magnitude. On a lower-is-better
    # metric like price, a pure magnitude bar gives the WINNER the shorter bar,
    # which reads as a contradiction. The winner flag tells us the direction.
    lo, hi = min(abs(a_score), abs(b_score)), max(abs(a_score), abs(b_score))
    winner_score = a_score if winner == "A" else (b_score if winner == "B" else None)
    lower_is_better = winner_score is not None and hi > lo and abs(winner_score) == lo

    def bar_fraction(value: float) -> float:
        v = abs(value)
        if lower_is_better:
            return min(1.0, lo / v) if v > 1e-9 else 1.0
        return min(1.0, v / hi) if hi > 1e-9 else 0.0

    name_a = str(item_a.get("name", "Item A"))
    name_b = str(item_b.get("name", "Item B"))
    hook_a = str(item_a.get("hook", "") or "")
    hook_b = str(item_b.get("hook", "") or "")

    beats = duel_round_beats(duration)
    vx = (boxes["divider"][0] + boxes["divider"][2]) // 2
    vy = (boxes["divider"][1] + boxes["divider"][3]) // 2

    def make_frame(t: float):
        p = min(1.0, t / duration)
        frame = _compose_duel_frame(size, sources, boxes, layout, p, accent_a, accent_b)
        draw = ImageDraw.Draw(frame, "RGBA")
        gate = title_gate(t, duration)

        # --- metric header pill -------------------------------------------
        def paint_header(td: ImageDraw.ImageDraw) -> None:
            title_font = _load_bold_font(int(w * 0.058))
            tw = td.textlength(metric.upper(), font=title_font)
            pill_w, pill_h = int(tw + w * 0.14), int(w * 0.105)
            dy = int((1.0 - gate) * h * 0.026)
            pill_x, pill_y = int((w - pill_w) / 2), int(h * 0.022) - dy
            _glass_card(td, [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
                        (255, 255, 255), alpha=0.92)
            td.text(((w - tw) / 2, pill_y + pill_h * 0.16), metric.upper(), font=title_font,
                    fill=(255, 255, 255),
                    stroke_width=max(2, int(w * 0.004)), stroke_fill=(0, 0, 0, 220))
            if note:
                nf = _load_bold_font(int(w * 0.024))
                nw = td.textlength(note.upper(), font=nf)
                td.text(((w - nw) / 2, pill_y + pill_h + int(h * 0.008)), note.upper(), font=nf,
                        fill=(178, 188, 208), stroke_width=2, stroke_fill=(0, 0, 0, 200))

        if gate >= 0.999:
            paint_header(draw)
        else:
            header = fade_tile((w, TITLE_BAND_H), gate, paint_header)
            if header is not None:
                frame.paste(header[0], (0, 0), header[1])

        show_winner = _ease_out_cubic((t - beats["reveal"]) / max(0.25, duration - beats["reveal"])) \
            if t >= beats["reveal"] else 0.0

        # --- per-side name plate + glass stat card -------------------------
        margin = int(w * 0.06)
        plate_w = int(w * (0.36 if layout == "side_by_side" else 0.56))

        for key, name, hook, score, dec, accent in (
            ("a", name_a, hook_a, a_score, a_dec, accent_a),
            ("b", name_b, hook_b, b_score, b_dec, accent_b),
        ):
            box = boxes[key]
            is_winner = winner == key.upper()

            # Panel A's plate must clear the metric header pill above it;
            # stacked layouts put that pill directly over panel A.
            head_clear = int(h * 0.125) if (key == "a" and layout != "side_by_side") else int(h * 0.035)
            _draw_name_plate(draw, (box[0] + margin, box[1] + head_clear),
                             plate_w, key, name, hook, accent, w)

            start = beats["a_in"] if key == "a" else beats["b_in"]
            if t < start:
                continue

            pop = _ease_out_back(min(1.0, (t - start) / 0.45))
            fade = min(1.0, (t - start) / 0.30)
            counted = score * _ease_out_cubic((t - start) / beats["count"]) \
                if t < start + beats["count"] else score

            card_w = int(w * (0.40 if layout == "side_by_side" else 0.60) * (0.94 + 0.06 * pop))
            card_h = int(w * 0.185 * (0.94 + 0.06 * pop))
            card_x = box[0] + margin
            card_y = box[3] - card_h - int(h * (0.045 if key == "a" else 0.075))
            card = [card_x, card_y, card_x + card_w, card_y + card_h]

            # The winning card's glow retires with the tag. Leaving it lit
            # would keep half the "this side won" treatment burning through the
            # dissolve into the winner card, which is the same ghosting in a
            # quieter form.
            _glass_card(draw, card, accent, alpha=fade,
                        glow=(show_winner * gate) if is_winner else 0.0)

            val_font = _load_bold_font(max(22, int(card_h * 0.46)))
            draw.text((card_x + int(card_h * 0.22), card_y + int(card_h * 0.12)),
                      format_stat(counted, unit, dec), font=val_font, fill=accent + (int(255 * fade),),
                      stroke_width=max(2, int(card_h * 0.030)), stroke_fill=(0, 0, 0, int(230 * fade)))

            lab_font = _load_bold_font(max(13, int(card_h * 0.155)))
            draw.text((card_x + int(card_h * 0.23), card_y + int(card_h * 0.60)), metric.upper(),
                      font=lab_font, fill=(176, 186, 206, int(255 * fade)))

            bar_y = card_y + int(card_h * 0.80)
            _progress_bar(draw, [card_x + int(card_h * 0.22), bar_y,
                                 card_x + card_w - int(card_h * 0.22), bar_y + max(6, int(card_h * 0.10))],
                          bar_fraction(counted), accent, alpha=fade)

            # The round's WINNER tag is gated too. Without this the final round
            # dissolves into the winner card and the word appears twice, in two
            # places, for the length of the transition.
            tag_lit = show_winner * gate
            if is_winner and tag_lit > 0.02:
                tag_font = _load_bold_font(max(15, int(card_h * 0.20)))
                tw2 = draw.textlength("WINNER", font=tag_font)
                tag_w, tag_h = int(tw2 + card_h * 0.36), int(card_h * 0.34)
                tx, ty = card[2] - tag_w - int(card_h * 0.16), card_y - tag_h // 2

                def paint_tag(td: ImageDraw.ImageDraw, _w=tag_w, _h=tag_h,
                              _f=tag_font, _a=accent) -> None:
                    td.rounded_rectangle([0, 0, _w, _h], radius=int(_h * 0.42),
                                         fill=_a + (246,))
                    td.text((int(card_h * 0.18), _h * 0.18), "WINNER", font=_f, fill=(6, 8, 14))

                tag_tile = fade_tile((tag_w + 2, tag_h + 2), tag_lit, paint_tag)
                if tag_tile is not None:
                    frame.paste(tag_tile[0], (tx, ty), tag_tile[1])

        _draw_vs_medallion(draw, (vx, vy), int(min(w, h) * 0.050))

        if watermark is not None:
            tile, pos = watermark
            frame.paste(tile, pos, tile)
        return np.array(frame)

    return VideoClip(make_frame, duration=duration)


def create_winner_clip(
    size: tuple[int, int],
    winner_item: dict[str, Any],
    tally: tuple[int, int],
    duration: float = 4.0,
    is_a: bool = True,
    headline: str = "",
    watermark_text: str = "",
    accent_a: RGB = DUEL_ACCENT_A,
    accent_b: RGB = DUEL_ACCENT_B,
) -> VideoClip:
    """
    The end card: the winning photograph and ONE victory block.

    Everything -- trophy, wordmark, name, score, vote prompt -- is a single
    vertically centred stack that animates in as one unit after the cut. The
    old card spread the same elements from 23% to 87% of frame height with a
    dead third in the middle, and began at full opacity on frame one, so the
    last round's own WINNER tag ghosted straight through it.
    """
    w, h = size
    side_accent = accent_a if is_a else accent_b

    photo = winner_item.get("image")
    if not isinstance(photo, Image.Image):
        photo = Image.new("RGB", (w, h), (16, 18, 26))
    full_box = [0, 0, w, h]
    source = _prep_panel_source(photo, full_box, headroom=1.16)

    name = str(winner_item.get("name", "Winner"))
    cta = (headline or "WHICH ONE WOULD YOU PICK?").upper()
    tally_text = f"{tally[0]} — {tally[1]}  ROUNDS"
    watermark = make_watermark_tile((w, h), watermark_text, position="bottom_right")

    # The veil is static -- built once, pasted through its own alpha.
    veil = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    vd = ImageDraw.Draw(veil)
    for i in range(h):
        k = abs((i / max(1, h - 1)) - 0.46)
        # Keep the winning product readable -- the veil is for type contrast,
        # not for hiding the photo the whole card is celebrating.
        vd.line([(0, i), (w, i)], fill=(6, 7, 11, int(96 + 130 * min(1.0, k * 2.0))))
    veil_rgb, veil_alpha = veil.convert("RGB"), veil.getchannel("A")

    def make_frame(t: float):
        p = min(1.0, t / duration)

        # Slow push-in on the winner's photo, heavily darkened for the card.
        frame = _ken_burns_panel(source, full_box, p, zoom_in=True)
        frame.paste(veil_rgb, (0, 0), veil_alpha)
        draw = ImageDraw.Draw(frame, "RGBA")

        # Gold vignette frame.
        inset = int(w * 0.045)
        draw.rounded_rectangle([inset, inset, w - inset, h - inset], radius=int(w * 0.045),
                               outline=DUEL_GOLD + (150,), width=3)

        # One gate for the whole stack: it arrives together after the cut and
        # holds. TITLE_IN is doubled here because a victory card that snaps to
        # full opacity in a quarter second reads as a jump cut.
        gate = title_gate(t, duration, lead=TITLE_IN * 2.0, tail=0.0)

        def paint_stack(td: ImageDraw.ImageDraw) -> None:
            # --- measure the stack, then centre it --------------------------
            nf = _load_bold_font(int(w * 0.078))
            name_lines = _wrap_to_width(td, name, nf, int(w * 0.84))[:2]
            name_step = int(w * 0.086)

            badge_r = int(w * 0.105)
            gap = int(w * 0.030)
            wf = _load_bold_font(int(w * 0.066))
            tf = _load_bold_font(int(w * 0.044))
            cf = _load_bold_font(int(w * 0.036))
            cta_lines = _wrap_to_width(td, cta, cf, int(w * 0.80))[:2]

            wordmark_h = int(w * 0.082)
            chip_h = int(w * 0.086)
            cta_step = int(w * 0.050)
            stack_h = (badge_r * 2 + gap + wordmark_h + gap
                       + len(name_lines) * name_step + gap
                       + chip_h + int(gap * 1.4) + len(cta_lines) * cta_step)

            # Rises the last few pixels into place as it fades up.
            y = int((h - stack_h) / 2) + int((1.0 - gate) * h * 0.035)

            # --- trophy badge --------------------------------------------------
            pop = _ease_out_back(min(1.0, t / 0.55))
            scale = pop * (1.0 + 0.022 * math.sin(t * 4.0))
            r = int(badge_r * scale)
            bx, by = w // 2, y + badge_r
            for i, spread in enumerate((22, 13, 6)):
                td.ellipse([bx - r - spread, by - r - spread, bx + r + spread, by + r + spread],
                           outline=DUEL_GOLD + (int(120 / (i + 1.2)),), width=3)
            td.ellipse([bx - r, by - r, bx + r, by + r],
                       fill=(10, 12, 18, 242), outline=DUEL_GOLD + (245,), width=4)
            _draw_trophy(td, (bx - r * 0.34, by - r * 0.44), r * 1.05, DUEL_GOLD)
            y += badge_r * 2 + gap

            # --- the one and only WINNER wordmark ---------------------------------
            lw = td.textlength("WINNER", font=wf)
            td.text(((w - lw) / 2, y), "WINNER", font=wf, fill=DUEL_GOLD,
                    stroke_width=max(3, int(w * 0.006)), stroke_fill=(0, 0, 0, 240))
            y += wordmark_h + gap

            # --- winner name -------------------------------------------------------
            for i, line in enumerate(name_lines):
                nw2 = td.textlength(line, font=nf)
                td.text(((w - nw2) / 2, y + i * name_step), line, font=nf, fill=(255, 255, 255),
                        stroke_width=max(3, int(w * 0.007)), stroke_fill=(0, 0, 0, 240))
            y += len(name_lines) * name_step + gap

            # --- final round score --------------------------------------------------
            tw2 = td.textlength(tally_text, font=tf)
            chip_w = int(tw2 + w * 0.10)
            cx0 = int((w - chip_w) / 2)
            _glass_card(td, [cx0, y, cx0 + chip_w, y + chip_h], side_accent, alpha=1.0)
            td.text(((w - tw2) / 2, y + chip_h * 0.20), tally_text, font=tf, fill=(255, 255, 255))
            y += chip_h + int(gap * 1.4)

            # --- comment vote prompt -------------------------------------------------
            for i, line in enumerate(cta_lines):
                cw2 = td.textlength(line, font=cf)
                td.text(((w - cw2) / 2, y + i * cta_step), line, font=cf,
                        fill=(226, 232, 240), stroke_width=2, stroke_fill=(0, 0, 0, 210))

        paint_faded(frame, gate, paint_stack)

        if watermark is not None:
            tile, pos = watermark
            frame.paste(tile, pos, tile)
        return np.array(frame)

    return VideoClip(make_frame, duration=duration)
