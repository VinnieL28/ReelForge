# pyright: reportMissingImports=false
"""
A modular stick-figure rig.

Pure vector: a circle head and straight limbs, posed by forward kinematics from
a table of joint angles. Nothing here knows about the animation engine -- it
draws onto any surface exposing `line`, `circle`, `polyline` and `polygon`, so
`minimalist_engine.Frame` satisfies it by duck typing and there is no import cycle
between the rig and the engine that uses it.

Two design decisions worth stating.

**Angles, not coordinates.** A pose is a dict of joint angles and the positions
are solved from them. Hard-coded positions cannot be scaled, mirrored or
blended; angles can, which is what makes a walk cycle a one-line interpolation
rather than six keyframes.

**Named joints on the way out.** A scene needs to hang a watering can off a
hand or a chain off both fists, so the skeleton exposes every joint by name
rather than only drawing itself.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

# Angles are measured from straight down, positive swinging toward +x (screen
# right). 0 = hanging, pi/2 = horizontal right, -pi/2 = horizontal left,
# pi = straight up. Screen y grows downward, which is why cos is the y term.
DOWN = 0.0
RIGHT = math.pi / 2
LEFT = -math.pi / 2
UP = math.pi


Point = tuple[float, float]


class Surface(Protocol):
    """The drawing calls the rig needs. `minimalist_engine.Frame` satisfies this."""

    def line(self, a: tuple[float, float], b: tuple[float, float],
             colour: tuple[int, int, int] = ..., width: float = ...) -> None: ...

    def circle(self, centre: tuple[float, float], radius: float,
               colour: tuple[int, int, int] = ..., width: float = ...) -> None: ...

    def polyline(self, points: Any, colour: tuple[int, int, int] = ...,
                 width: float = ...) -> None: ...


# ---------------------------------------------------------------------------
# Proportions
#
# As fractions of total standing height, roughly seven-and-a-half heads, which
# is the figure-drawing convention and reads as an adult rather than a child.
# ---------------------------------------------------------------------------

HEAD_RADIUS = 0.088
HEAD_CENTRE = 0.912          # from the feet, upward
NECK = 0.800
CHEST = 0.640                # the torso hinge, between hip and neck
HIP = 0.470
# Half the shoulder span, as a fraction of height.
#
# Arms used to start at the neck -- one point, for both of them, sitting inside
# the head's glow. A flexing pose then drew as two chevrons hanging beside the
# head with nothing joining them to the body, which is what the frame-by-frame
# report saw as "disconnected limbs". 0.090 puts each shoulder clear of a head
# whose radius is 0.088, so the join is always visible.
SHOULDER_HALF = 0.090
UPPER_ARM = 0.175
FOREARM = 0.165
THIGH = 0.245
SHIN = 0.225

POSES: tuple[str, ...] = (
    "idle_standing", "reflective", "walking", "climbing_stairs",
    "pushing_heavy_load", "walking_on_beam", "flexing_arms", "reaching_upward",
    "watering_can_pouring", "struggling_chained",
)

# The names the scenes used before this rig grew. Kept so a saved spec or an
# older scene keeps working instead of silently falling back to idle.
POSE_ALIASES: dict[str, str] = {
    "idle": "idle_standing",
    "pushing": "pushing_heavy_load",
    "flexing": "flexing_arms",
    "watering_plant": "watering_can_pouring",
    "stairs": "climbing_stairs",
    "beam": "walking_on_beam",
    "reaching": "reaching_upward",
}


def resolve_pose(name: str) -> str:
    """A pose key from a new name, an old name, or nonsense."""
    key = str(name or "").strip().lower()
    key = POSE_ALIASES.get(key, key)
    return key if key in POSES else "idle_standing"


def _tip(origin: tuple[float, float], angle: float, length: float) -> tuple[float, float]:
    """Where a limb of `length` ends, swung `angle` from straight down."""
    return (origin[0] + math.sin(angle) * length,
            origin[1] + math.cos(angle) * length)


class Skeleton:
    """Resolved joint positions in canvas space, plus the props' anchor points."""

    __slots__ = ("head", "neck", "chest", "hip", "shoulder_l", "elbow_l", "hand_l",
                 "shoulder_r", "elbow_r", "hand_r", "knee_l", "foot_l",
                 "knee_r", "foot_r", "height", "head_radius", "lean")

    # Annotated as well as slotted so callers get real types: without these a
    # checker sees `object` and every `hand_l[0]` is an error.
    head: Point
    neck: Point
    chest: Point
    hip: Point
    shoulder_l: Point
    elbow_l: Point
    hand_l: Point
    shoulder_r: Point
    elbow_r: Point
    hand_r: Point
    knee_l: Point
    foot_l: Point
    knee_r: Point
    foot_r: Point
    height: float
    head_radius: float
    lean: float

    def __init__(self, **joints: Any) -> None:
        for name, value in joints.items():
            setattr(self, name, value)

    @property
    def hands(self) -> tuple[Point, Point]:
        return (self.hand_l, self.hand_r)

    @property
    def reach(self) -> Point:
        """The forward hand -- where a prop goes."""
        return self.hand_r if self.hand_r[0] >= self.hand_l[0] else self.hand_l


# ---------------------------------------------------------------------------
# Poses
#
# Each returns the joint angles for a moment in a pose. `phase` runs 0..1 and
# is what animates: a walk cycle, a push straining and easing, a breath.
# ---------------------------------------------------------------------------

def _pose_angles(pose: str, phase: float) -> dict[str, float]:
    swing = math.sin(phase * 2 * math.pi)
    counter = math.sin(phase * 2 * math.pi + math.pi)

    if pose == "walking":
        return {
            "lean": 0.06,
            # Arms counter-swing the legs, which is what stops a walk cycle
            # reading as a shuffle.
            "shoulder_l": 0.34 * counter, "elbow_l": 0.30 + 0.22 * abs(counter),
            "shoulder_r": 0.34 * swing, "elbow_r": 0.30 + 0.22 * abs(swing),
            "hip_l": 0.42 * swing, "knee_l": 0.32 * max(0.0, -swing) + 0.06,
            "hip_r": 0.42 * counter, "knee_r": 0.32 * max(0.0, -counter) + 0.06,
        }

    if pose == "climbing_stairs":
        # A stair climb is a walk with the knee driven much higher and the
        # torso pitched forward over the leading foot.
        return {
            "lean": 0.20, "curve": 0.10,
            "shoulder_l": 0.42 * counter, "elbow_l": 0.46 + 0.24 * abs(counter),
            "shoulder_r": 0.42 * swing, "elbow_r": 0.46 + 0.24 * abs(swing),
            "hip_l": 0.66 * swing - 0.10, "knee_l": 0.62 * max(0.0, swing) + 0.08,
            "hip_r": 0.66 * counter - 0.10, "knee_r": 0.62 * max(0.0, counter) + 0.08,
        }

    if pose == "walking_on_beam":
        # Arms out for balance, feet close to the centre line, small sway.
        sway = math.sin(phase * 2 * math.pi)
        return {
            "lean": 0.03 * sway, "curve": -0.04,
            "shoulder_l": -1.46 - 0.10 * sway, "elbow_l": -1.52,
            "shoulder_r": 1.46 - 0.10 * sway, "elbow_r": 1.52,
            "hip_l": 0.20 * swing, "knee_l": 0.16 * max(0.0, swing) + 0.04,
            "hip_r": 0.20 * counter, "knee_r": 0.16 * max(0.0, counter) + 0.04,
        }

    if pose == "reaching_upward":
        stretch = 0.5 + 0.5 * math.sin(phase * 2 * math.pi)
        return {
            # Reaching lengthens the spine: the curve goes negative, arching back.
            "lean": -0.06, "curve": -0.12 - 0.05 * stretch,
            # Solved rather than guessed: at 2.86/3.05 the hands land at x=0.063
            # of body height, inside the head's own 0.088 radius, so the arms
            # close into a narrow V over the skull. 2.58/2.78 puts them at
            # x=0.153 -- clear of the head and reading as a reach.
            "shoulder_l": -2.58 - 0.09 * stretch, "elbow_l": -2.78,
            "shoulder_r": 2.58 + 0.09 * stretch, "elbow_r": 2.78,
            "hip_l": -0.13, "knee_l": -0.03,
            "hip_r": 0.13, "knee_r": 0.03,
        }

    if pose == "reflective":
        # Standing still, weight on one side, head tipped down a little.
        breath = math.sin(phase * 2 * math.pi)
        return {
            "lean": 0.05, "curve": 0.13 + 0.02 * breath,
            "shoulder_l": -0.30, "elbow_l": -0.62,
            "shoulder_r": 0.22 + 0.02 * breath, "elbow_r": 0.14,
            "hip_l": -0.16, "knee_l": -0.05,
            "hip_r": 0.13, "knee_r": 0.10,
        }

    if pose == "pushing_heavy_load":
        effort = 0.5 + 0.5 * math.sin(phase * 2 * math.pi)
        return {
            # The hunch is the tell: shoulders round over the hands under load.
            "lean": 0.40 + 0.06 * effort, "curve": 0.22 + 0.05 * effort,
            # Both arms forward and level: the classic shoulder-into-it shape.
            "shoulder_l": 1.28, "elbow_l": 1.42,
            "shoulder_r": 1.38, "elbow_r": 1.50,
            # Back leg driving, front leg braced.
            "hip_l": -0.52 - 0.08 * effort, "knee_l": -0.10,
            "hip_r": 0.30, "knee_r": 0.46 + 0.10 * effort,
        }

    if pose == "flexing_arms":
        pump = 0.5 + 0.5 * math.sin(phase * 2 * math.pi)
        return {
            "lean": 0.0,
            # Upper arms out past horizontal, forearms folded back up: a
            # double-biceps pose, which is the silhouette that reads as strong.
            "shoulder_l": -1.42 - 0.05 * pump, "elbow_l": -2.68 - 0.10 * pump,
            "shoulder_r": 1.42 + 0.05 * pump, "elbow_r": 2.68 + 0.10 * pump,
            "hip_l": -0.22, "knee_l": -0.05,
            "hip_r": 0.22, "knee_r": 0.05,
        }

    if pose == "struggling_chained":
        strain = 0.5 + 0.5 * math.sin(phase * 2 * math.pi)
        return {
            # Low and far forward: hauling, not walking.
            "lean": 0.44 + 0.05 * strain, "curve": 0.26 + 0.06 * strain,
            # Arms trailing back and down: the chain is behind, being dragged.
            "shoulder_l": -0.92, "elbow_l": -0.34,
            "shoulder_r": -0.78, "elbow_r": -0.26,
            "hip_l": -0.54 - 0.08 * strain, "knee_l": -0.12,
            "hip_r": 0.34, "knee_r": 0.40 + 0.10 * strain,
        }

    if pose == "watering_can_pouring":
        tip = 0.5 + 0.5 * math.sin(phase * 2 * math.pi)
        return {
            "lean": 0.22,
            # Near arm hangs, far arm extends forward and down over the plant.
            "shoulder_l": -0.16, "elbow_l": -0.10,
            "shoulder_r": 1.34 + 0.08 * tip, "elbow_r": 0.62 + 0.18 * tip,
            "hip_l": -0.18, "knee_l": -0.04,
            "hip_r": 0.16, "knee_r": 0.06,
        }

    # idle: a quiet stance with a breath in it
    breath = math.sin(phase * 2 * math.pi)
    return {
        "lean": 0.02 + 0.012 * breath,
        "shoulder_l": -0.25 - 0.02 * breath, "elbow_l": -0.16,
        "shoulder_r": 0.25 + 0.02 * breath, "elbow_r": 0.16,
        "hip_l": -0.11, "knee_l": -0.03,
        "hip_r": 0.11, "knee_r": 0.03,
    }


def build_skeleton(pose: str = "idle", phase: float = 0.0,
                   anchor: tuple[float, float] = (540.0, 1400.0),
                   height: float = 420.0, facing: int = 1) -> Skeleton:
    """
    Solves a pose into canvas positions.

    `anchor` is the point between the feet, so a figure stands *on* a floor
    line rather than being centred on one. `facing` of -1 mirrors it.
    """
    angles = _pose_angles(resolve_pose(pose), phase)
    lean = float(angles.get("lean", 0.0)) * facing

    ax, ay = anchor
    hip = (ax + math.sin(lean) * height * HIP * 0.35, ay - height * HIP)

    # The torso is two segments hinged at the chest, not one straight spine.
    # A single line can only tilt; two can curve, which is what lets a figure
    # hunch into a push or straighten up when it is released -- and the change
    # between those two shapes is most of what reads as effort.
    #
    # `pi - lean`, not `pi + lean`: straight up is pi, and positive angles swing
    # toward +x, so subtracting tilts the torso forward. Adding tilted it
    # backward while the hip offset and the limb angles moved forward, which is
    # why the pushing figure leaned away from its own hands.
    curve = float(angles.get("curve", 0.0)) * facing
    chest = _tip(hip, math.pi - lean, height * (CHEST - HIP))
    neck = _tip(chest, math.pi - lean - curve, height * (NECK - CHEST))
    head = _tip(chest, math.pi - lean - curve, height * (HEAD_CENTRE - CHEST))

    # The shoulders sit either side of the neck, square to the spine, so they
    # lean and hunch with the torso instead of staying level while it curves.
    spine = math.pi - lean - curve
    shoulder_r = _tip(neck, spine - math.pi / 2 * facing, height * SHOULDER_HALF)
    shoulder_l = _tip(neck, spine + math.pi / 2 * facing, height * SHOULDER_HALF)
    sockets = {"l": shoulder_l, "r": shoulder_r}

    def arm(side: str) -> tuple[tuple[float, float], tuple[float, float]]:
        shoulder = float(angles[f"shoulder_{side}"]) * facing + lean + curve
        elbow_angle = float(angles[f"elbow_{side}"]) * facing + lean + curve
        elbow = _tip(sockets[side], shoulder, height * UPPER_ARM)
        return elbow, _tip(elbow, elbow_angle, height * FOREARM)

    def leg(side: str) -> tuple[tuple[float, float], tuple[float, float]]:
        hip_angle = float(angles[f"hip_{side}"]) * facing
        knee_angle = float(angles[f"knee_{side}"]) * facing
        knee = _tip(hip, hip_angle, height * THIGH)
        return knee, _tip(knee, hip_angle + knee_angle, height * SHIN)

    elbow_l, hand_l = arm("l")
    elbow_r, hand_r = arm("r")
    knee_l, foot_l = leg("l")
    knee_r, foot_r = leg("r")

    return Skeleton(
        head=head, neck=neck, chest=chest, hip=hip,
        shoulder_l=shoulder_l, elbow_l=elbow_l, hand_l=hand_l,
        shoulder_r=shoulder_r, elbow_r=elbow_r, hand_r=hand_r,
        knee_l=knee_l, foot_l=foot_l, knee_r=knee_r, foot_r=foot_r,
        height=height, head_radius=height * HEAD_RADIUS, lean=lean,
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_skeleton(surface: Surface, skeleton: Skeleton,
                  colour: tuple[int, int, int] = (255, 255, 255),
                  weight: float | None = None, filled_head: bool = False) -> None:
    """
    Strokes a solved skeleton.

    The far-side limbs go down first at a lighter weight so the figure reads
    with a front and a back instead of as a flat tangle of lines.
    """
    stroke = weight if weight is not None else max(3.0, skeleton.height * 0.019)

    # 0.78, not 0.62. The far side has to read as *behind* the near side, not
    # as absent: at 0.62 a GREY figure's back limbs come out at 84/255 and
    # vanish into the glow wherever they cross the front ones, which the
    # frame-by-frame report saw as a figure drawn "with one arm/one leg".
    back = tuple(int(c * 0.78) for c in colour)

    surface.polyline([skeleton.shoulder_l, skeleton.elbow_l, skeleton.hand_l], back, stroke * 0.88)  # type: ignore[arg-type]
    surface.polyline([skeleton.hip, skeleton.knee_l, skeleton.foot_l], back, stroke * 0.88)  # type: ignore[arg-type]

    surface.polyline([skeleton.hip, skeleton.chest, skeleton.neck], colour, stroke)  # type: ignore[arg-type]
    # The clavicle. Without it the two arms are joined to the spine only by
    # being near it.
    surface.polyline([skeleton.shoulder_l, skeleton.neck, skeleton.shoulder_r], colour, stroke * 0.9)  # type: ignore[arg-type]
    surface.polyline([skeleton.hip, skeleton.knee_r, skeleton.foot_r], colour, stroke)  # type: ignore[arg-type]
    surface.polyline([skeleton.shoulder_r, skeleton.elbow_r, skeleton.hand_r], colour, stroke)  # type: ignore[arg-type]

    surface.circle(skeleton.head, skeleton.head_radius, colour,
                   0.0 if filled_head else stroke)


def draw_figure(surface: Surface, pose: str = "idle", phase: float = 0.0,
                anchor: tuple[float, float] = (540.0, 1400.0),
                height: float = 420.0, facing: int = 1,
                colour: tuple[int, int, int] = (255, 255, 255),
                weight: float | None = None,
                filled_head: bool = False) -> Skeleton:
    """Poses and strokes in one call; returns the skeleton for hanging props."""
    skeleton = build_skeleton(pose, phase, anchor, height, facing)
    draw_skeleton(surface, skeleton, colour, weight, filled_head)
    # A surface that keeps a record of what it drew gets told about the
    # figure, so layout checks can measure labels against it.
    recorder = getattr(surface, "record_figure", None)
    if callable(recorder):
        recorder(skeleton)
    return skeleton
