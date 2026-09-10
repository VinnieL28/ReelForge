"""The rig and every scene: poses solve, scenes draw, nothing is static.

No network. Run with:
    .venv\\Scripts\\python tests\\test_vector_scenes.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import minimalist_engine as me
import vector_rig as rig

# ---- 1. the rig ------------------------------------------------------------
print(f"1. rig: {len(rig.POSES)} poses")
for pose in rig.POSES:
    skeleton = rig.build_skeleton(pose, 0.25, (0.0, 0.0), 1.0)
    # every joint solved and finite
    for joint in ("head", "neck", "chest", "hip", "hand_l", "hand_r",
                  "foot_l", "foot_r", "knee_l", "knee_r"):
        point = getattr(skeleton, joint)
        assert len(point) == 2 and all(math.isfinite(v) for v in point), f"{pose}.{joint}"
    # the figure stands on the anchor, not through it
    assert -skeleton.foot_l[1] < 0.30 and -skeleton.foot_r[1] < 0.30, f"{pose}: feet float"
    # and fits inside a sane bounding box
    xs = [getattr(skeleton, j)[0] for j in ("head", "hand_l", "hand_r", "foot_l", "foot_r")]
    assert max(abs(x) for x in xs) < 0.85, f"{pose}: limbs way out of frame"
    print(f"   {pose:22} head=({skeleton.head[0]:+.2f},{-skeleton.head[1]:.2f}) "
          f"hands=({skeleton.hand_l[0]:+.2f},{skeleton.hand_r[0]:+.2f})")

# the torso is two hinged segments, so the chest is not on the hip-neck line
bent = rig.build_skeleton("pushing_heavy_load", 0.25, (0.0, 0.0), 1.0)
hx, hy = bent.hip
nx, ny = bent.neck
cx, cy = bent.chest
span = math.hypot(nx - hx, ny - hy) or 1.0
offset = abs((nx - hx) * (hy - cy) - (hx - cx) * (ny - hy)) / span
print(f"\n   chest sits {offset:.3f} of body height off the hip-neck line")
assert offset > 0.004, "the torso is still one straight segment"

straight = rig.build_skeleton("walking_on_beam", 0.25, (0.0, 0.0), 1.0)
hx, hy = straight.hip
nx, ny = straight.neck
cx, cy = straight.chest
span = math.hypot(nx - hx, ny - hy) or 1.0
flat = abs((nx - hx) * (hy - cy) - (hx - cx) * (ny - hy)) / span
print(f"   and only {flat:.3f} when the pose is upright")
assert flat < offset, "the spine curves the same amount whatever the pose"

# aliases keep older scene code working
print(f"\n   aliases: {[f'{k}->{rig.resolve_pose(k)}' for k in ('idle', 'pushing', 'flexing')]}")
for old, new in (("idle", "idle_standing"), ("pushing", "pushing_heavy_load"),
                 ("flexing", "flexing_arms"), ("watering_plant", "watering_can_pouring")):
    assert rig.resolve_pose(old) == new, old
assert rig.resolve_pose("nonsense") == "idle_standing"

# ---- 2. every scene draws, and moves ---------------------------------------
print(f"\n2. scenes: {len(me.TEMPLATES)} templates, {len(me.METAPHOR_TYPES)} selectable, "
      f"{len(me.CHARACTER_TYPES)} with a figure")
for name in me.TEMPLATES:
    spec = me.fallback_scene_spec("", name, 18.0)
    duration = float(spec["duration"])
    inks = []
    for frac in (0.25, 0.55, 0.85):
        frame = me.make_scene_frame(spec, duration * frac, duration)
        assert frame.shape == (1920, 1080, 3), f"{name}: {frame.shape}"
        assert frame.dtype.name == "uint8"
        inks.append(float((frame.max(axis=2) > 40).mean()))

    moved = max(inks) - min(inks) > 0.002
    print(f"   {name:22} ink {min(inks) * 100:4.1f}%..{max(inks) * 100:4.1f}%  "
          f"{'animates' if moved else 'STATIC'}")
    assert 0.004 < max(inks) < 0.65, f"{name}: blank or blown out"
    assert moved, f"{name}: nothing changes across the runtime"

# ---- 3. the palette ---------------------------------------------------------
#
# The ground is #000000, but the ambient layer draws a lattice at (17,17,17)
# over it and the glow lifts that to about 22 -- so "pure black" is a claim
# about the canvas, not about every pixel of a finished frame. Check both: the
# modal value with the grid on, and an exact zero with it off.
import numpy as np

spec = me.fallback_scene_spec("", "two_doors", 18.0)
lit = me.make_scene_frame(spec, 4.0, 18.0)

brightness = lit.reshape(-1, 3).max(axis=1)
values, counts = np.unique(brightness, return_counts=True)
modal = int(values[counts.argmax()])
exact = float(counts.max() / counts.sum())
near = float((brightness < 30).mean())
print(f"\n3. ambient grid on : modal pixel {modal}, {exact * 100:.0f}% exactly black, "
      f"{near * 100:.0f}% below 30")
assert modal == 0, "the dominant background value is not black"
# The rest of the "black" is glow spill at 1-3, which is what a wide bloom
# does. What matters is that the frame is dominated by near-black ground.
assert near > 0.80, "the frame is not dominated by near-black ground"

plain = me.make_scene_frame(me.normalise_spec({**spec, "ambient": 0.0}), 4.0, 18.0)
corner = plain[0:40, 0:40]
print(f"   ambient grid off: corner max = {corner.max()} (#000000)")
assert corner.max() == 0, "the canvas is not pure black even with ambient off"

spread = int(abs(lit.astype(int) - lit.astype(int).mean(axis=2, keepdims=True)).max())
print(f"   max channel spread from grey = {spread} (0 = strictly monochrome)")
assert spread <= 2, "the palette drifted off greyscale"

# ---- 4. easing and phases ---------------------------------------------------
print("\n4. easing")
for name in ("ease_out_cubic", "ease_in_cubic", "rush_into", "overshoot", "ease_in_out"):
    fn = getattr(me, name)
    assert abs(fn(0.0)) < 0.02 and abs(fn(1.0) - 1.0) < 0.02, f"{name} is not 0..1"
    print(f"   {name:15} 0.25->{fn(0.25):.3f}  0.5->{fn(0.5):.3f}  0.75->{fn(0.75):.3f}")
# ease_out_cubic must front-load: more than half the distance by the halfway mark
assert me.ease_out_cubic(0.5) > 0.8, "ease_out_cubic is not front-loaded"
assert me.ease_in_cubic(0.5) < 0.2, "ease_in_cubic is not back-loaded"

# phases: the geometry must finish drawing before the beat lands
spec = me.normalise_spec({"metaphor_type": "sisyphus_boulder", "duration": 20,
                          "animation_phases": {"draw_end": 6.5, "impact": 16.0}})
ph = me.phases_of(spec, 20.0)
print(f"\n   phases: lead {ph.lead:.2f} -> draw_end {ph.draw_end:.2f} -> "
      f"impact {ph.impact:.2f} -> end {ph.end:.2f}")
assert ph.lead < ph.draw_end < ph.impact < ph.duration
assert abs(ph.draw(ph.draw_end) - 1.0) < 0.01, "the trim-path does not complete"
assert ph.draw(ph.lead) < 0.01, "the trim-path starts already drawn"

print("\nVECTOR RIG AND SCENES VERIFIED")
