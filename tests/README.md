# Tests

```bash
.venv\Scripts\python tests\test_narrative_unit.py     # no network, ~2s
.venv\Scripts\python tests\test_narrative_ffmpeg.py   # needs ffmpeg, ~90s
```

`test_narrative_unit.py` covers entity extraction, storyboard JSON parsing
(fenced, prose-wrapped, trailing commas), segmentation, retiming against word
timings, image-prompt consistency and the offline fallback. No network.

`test_narrative_ffmpeg.py` covers the parts that are only true if you measure
them: that a Ken Burns move actually moves, that hard-cut concatenation
preserves total duration, and that the xfade dissolve chain lands on
`sum(durations)` rather than one dissolve short. It renders real clips.

`test_vector_scenes.py` covers the stick-figure rig and every scene: all ten
poses solve to finite joints inside the frame, the two-segment torso actually
curves under load and straightens when upright, the pose aliases still resolve,
all fifteen templates draw and change across their runtime, the palette stays
monochrome on a black ground, and the easing curves are front- or back-loaded
as intended. No network.
