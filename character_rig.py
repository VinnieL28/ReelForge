# pyright: reportMissingImports=false
"""
A modular stick-figure rig.

Pure vector: a circle head and straight limbs, posed by forward kinematics from
a table of joint angles. Nothing here knows about the animation engine -- it
draws onto any surface exposing `line`, `circle`, `polyline` and `polygon`, so
`motion_engine.Frame` satisfies it by duck typing and there is no import cycle
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
    """The drawing calls the rig needs. `motion_engine.Frame` satisfies this."""

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
HIP = 0.470
UPPER_ARM = 0.175
FOREARM = 0.165
THIGH = 0.245
SHIN = 0.225

POSES: tuple[str, ...] = (
    "idle", "walking", "pushing", "flexing", "struggling_chained", "watering_plant",
)


def _tip(origin: tuple[float, float], angle: float, length: float) -> tuple[float, float]:
    """Where a limb of `length` ends, swung `angle` from straight down."""
    return (origin[0] + math.sin(angle) * length,
            origin[1] + math.cos(angle) * length)


class Skeleton:
    """Resolved joint positions in canvas space, plus the props' anchor points."""

    __slots__ = ("head", "neck", "hip", "shoulder_l", "elbow_l", "hand_l",
                 "shoulder_r", "elbow_r", "hand_r", "knee_l", "foot_l",
                 "knee_r", "foot_r", "height", "head_radius", "lean")

    # Annotated as well as slotted so callers get real types: without these a
    # checker sees `object` and every `hand_l[0]` is an error.
    head: Point
    neck: Point
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

    if pose == "pushing":
        effort = 0.5 + 0.5 * math.sin(phase * 2 * math.pi)
        return {
            "lean": 0.40 + 0.06 * effort,
            # Both arms forward and level: the classic shoulder-into-it shape.
            "shoulder_l": 1.28, "elbow_l": 1.42,
            "shoulder_r": 1.38, "elbow_r": 1.50,
            # Back leg driving, front leg braced.
            "hip_l": -0.52 - 0.08 * effort, "knee_l": -0.10,
            "hip_r": 0.30, "knee_r": 0.46 + 0.10 * effort,
        }

    if pose == "flexing":
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
            "lean": 0.44 + 0.05 * strain,
            # Arms trailing back and down: the chain is behind, being dragged.
            "shoulder_l": -0.92, "elbow_l": -0.34,
            "shoulder_r": -0.78, "elbow_r": -0.26,
            "hip_l": -0.54 - 0.08 * strain, "knee_l": -0.12,
            "hip_r": 0.34, "knee_r": 0.40 + 0.10 * strain,
        }

    if pose == "watering_plant":
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
    angles = _pose_angles(pose if pose in POSES else "idle", phase)
    lean = float(angles.get("lean", 0.0)) * facing

    ax, ay = anchor
    hip = (ax + math.sin(lean) * height * HIP * 0.35, ay - height * HIP)
    # The torso leans as one piece from the hip, so the head travels further
    # than the chest -- which is what makes a lean read as effort.
    #
    # `pi - lean`, not `pi + lean`: straight up is pi, and positive angles swing
    # toward +x, so subtracting tilts the torso forward. Adding tilted it
    # backward while the hip offset and the limb angles moved forward, which is
    # why the pushing figure leaned away from its own hands.
    neck = _tip(hip, math.pi - lean, height * (NECK - HIP))
    head = _tip(hip, math.pi - lean, height * (HEAD_CENTRE - HIP))

    def arm(side: str) -> tuple[tuple[float, float], tuple[float, float]]:
        shoulder = float(angles[f"shoulder_{side}"]) * facing + lean
        elbow_angle = float(angles[f"elbow_{side}"]) * facing + lean
        elbow = _tip(neck, shoulder, height * UPPER_ARM)
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
        head=head, neck=neck, hip=hip,
        shoulder_l=neck, elbow_l=elbow_l, hand_l=hand_l,
        shoulder_r=neck, elbow_r=elbow_r, hand_r=hand_r,
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
    back = tuple(int(c * 0.62) for c in colour)

    surface.polyline([skeleton.neck, skeleton.elbow_l, skeleton.hand_l], back, stroke * 0.86)  # type: ignore[arg-type]
    surface.polyline([skeleton.hip, skeleton.knee_l, skeleton.foot_l], back, stroke * 0.86)  # type: ignore[arg-type]

    surface.line(skeleton.neck, skeleton.hip, colour, stroke)
    surface.polyline([skeleton.hip, skeleton.knee_r, skeleton.foot_r], colour, stroke)  # type: ignore[arg-type]
    surface.polyline([skeleton.neck, skeleton.elbow_r, skeleton.hand_r], colour, stroke)  # type: ignore[arg-type]

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
    return skeleton
