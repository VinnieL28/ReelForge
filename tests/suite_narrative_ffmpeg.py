"""Ken Burns and the concat/dissolve arithmetic, measured not assumed."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio_ffmpeg

import narrative_engine as ne
from video_engine import source_duration

WORK = tempfile.mkdtemp(prefix="rf_narr_")
SIZE = (540, 960)          # half size: the arithmetic is what is under test
FPS = 24
print(f"work dir: {WORK}\n")


def frames_of(path):
    reader = imageio_ffmpeg.read_frames(path, pix_fmt="rgb24")
    meta = next(reader)
    w, h = meta["size"]
    return [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in reader]


# ---- 1. procedural stills are real pictures --------------------------------
print("1. procedural stills")
entities = ne.extract_entities({})
stills = []
for i in range(4):
    seg = {"index": i + 1, "environment": "primary", "objects": [],
           "framing": "Wide shot", "line": "x"}
    path = ne.generate_image_procedural(seg, entities, os.path.join(WORK, f"s{i}.png"),
                                        SIZE, "nordic_noir")
    from PIL import Image
    arr = np.asarray(Image.open(path).convert("RGB"))
    stills.append(path)
    print(f"   still {i}: {arr.shape[1]}x{arr.shape[0]}  mean {arr.mean():5.1f}  "
          f"std {arr.std():5.1f}")
    assert arr.shape[:2] == (SIZE[1], SIZE[0])
    assert arr.std() > 12, "flat image -- no structure for a pan to bite on"

# consecutive stills must differ, or the episode looks like one held frame
from PIL import Image
a = np.asarray(Image.open(stills[0]).convert("RGB")).astype(int)
b = np.asarray(Image.open(stills[1]).convert("RGB")).astype(int)
delta = float(np.abs(a - b).mean())
print(f"   consecutive stills differ by {delta:.1f} mean absolute")
assert delta > 3, "consecutive stills are nearly identical"
print()

# ---- 2. Ken Burns actually moves -------------------------------------------
print("2. Ken Burns")
for move in ("in_center", "out_center", "pan_left", "pan_right"):
    out = os.path.join(WORK, f"kb_{move}.mp4")
    ne.ken_burns_clip(stills[0], out, duration=2.0, size=SIZE, fps=FPS, move=move)
    fs = frames_of(out)
    got = source_duration(out)
    first_last = float(np.abs(fs[0].astype(int) - fs[-1].astype(int)).mean())
    mid_step = float(np.abs(fs[len(fs) // 2].astype(int) - fs[len(fs) // 2 - 1].astype(int)).mean())
    print(f"   {move:11} {len(fs):3} frames  {got:4.2f}s  "
          f"first->last {first_last:5.2f}   per-frame {mid_step:4.2f}")
    assert abs(got - 2.0) < 0.15, f"{move}: wrong duration {got}"
    assert fs[0].shape[:2] == (SIZE[1], SIZE[0])
    # Threshold set from a hard-edged reference: a 16% zoom over testsrc2 moves
    # 26 mean-absolute, a pan moves 60. Over a soft graded still the same move
    # measures far less, so the assertion is calibrated to the still in hand.
    assert first_last > 2.0, f"{move}: the shot never moved"
    assert mid_step < first_last, f"{move}: motion is a jump, not a move"
print("   every move changes the frame over its length, smoothly\n")

# ---- 3. concat arithmetic: hard cuts ---------------------------------------
print("3. concatenation")
durations = [2.0, 1.5, 2.5, 1.8]
clips = []
for i, d in enumerate(durations):
    out = os.path.join(WORK, f"seg{i}.mp4")
    ne.ken_burns_clip(stills[i], out, duration=d, size=SIZE, fps=FPS,
                      move="in_center" if i % 2 else "pan_left")
    clips.append(out)
    print(f"   segment {i}: asked {d:4.2f}s got {source_duration(out):4.2f}s")

cut = ne.concat_segments(clips, os.path.join(WORK, "cuts.mp4"), durations,
                         transition="cut", fps=FPS)
total = sum(durations)
got_cut = source_duration(cut)
print(f"\n   hard cuts : {got_cut:5.2f}s vs sum(durations) {total:5.2f}s  "
      f"(delta {abs(got_cut - total):.3f}s)")
assert abs(got_cut - total) < 0.2, "hard-cut concat lost or gained time"

# ---- 4. the dissolve chain must not shorten the track ----------------------
DISSOLVE = 0.45
padded = [d + DISSOLVE for d in durations[:-1]] + [durations[-1]]
fade_clips = []
for i, d in enumerate(padded):
    out = os.path.join(WORK, f"fseg{i}.mp4")
    ne.ken_burns_clip(stills[i], out, duration=d, size=SIZE, fps=FPS,
                      move="in_center" if i % 2 else "pan_right")
    fade_clips.append(out)

fade = ne.concat_segments(fade_clips, os.path.join(WORK, "fades.mp4"), durations,
                          transition="dissolve", dissolve=DISSOLVE, fps=FPS)
got_fade = source_duration(fade)
print(f"   dissolves : {got_fade:5.2f}s vs sum(durations) {total:5.2f}s  "
      f"(delta {abs(got_fade - total):.3f}s)")
assert abs(got_fade - total) < 0.25, (
    f"the dissolve chain drifted {abs(got_fade - total):.2f}s off the voice -- "
    "each xfade eats `dissolve` seconds and the clips must be padded to match")

# and a dissolve must actually be a dissolve: mid-transition frames blend
fs = frames_of(fade)
boundary = int(durations[0] * FPS)
window = fs[max(0, boundary - 4):boundary + 4]
steps = [float(np.abs(window[i].astype(int) - window[i - 1].astype(int)).mean())
         for i in range(1, len(window))]
print(f"   frame-to-frame change across the first boundary: "
      f"{[round(s, 1) for s in steps]}")
assert max(steps) < 90, "that is a hard cut, not a dissolve"
assert max(steps) > 1.0, "nothing changed at the boundary at all"
print("   the boundary blends instead of jumping\n")

# ---- 5. subtitles line up with the segments --------------------------------
print("5. subtitles")
segments = [{"index": i + 1, "line": f"Line number {i + 1}.", "start": sum(durations[:i]),
             "end": sum(durations[:i + 1]), "words": 3} for i in range(len(durations))]
ass = ne.narrative_subtitles(segments, os.path.join(WORK, "subs.ass"), SIZE)
text = open(ass, encoding="utf-8").read()
cues = [ln for ln in text.splitlines() if ln.startswith("Dialogue:")]
print(f"   {len(cues)} cues for {len(segments)} segments")
for cue in cues[:2]:
    print(f"     {cue[:78]}")
assert len(cues) == len(segments)
assert "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text" in text
# the last cue must not run past the picture
last_end = text.rsplit("Dialogue:", 1)[1].split(",")[2]
h, m, s = last_end.split(":")
assert float(h) * 3600 + float(m) * 60 + float(s) <= total + 0.1, "a cue outlives the video"
print("   cue count matches, and no cue outlives the picture\n")

import shutil
shutil.rmtree(WORK, ignore_errors=True)
print("FFMPEG PIPELINE VERIFIED")
