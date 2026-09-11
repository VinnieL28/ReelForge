# Tests

```bash
.venv\Scripts\python -m pytest tests/                  # everything, ~3 min
.venv\Scripts\python -m pytest tests/ -m "not slow"    # ~25s, no ffmpeg renders
.venv\Scripts\python -m pytest tests/ --network        # also the live download
```

Markers: `slow` runs real ffmpeg renders, `network` hits the live platforms.
Network tests are skipped unless `--network` is passed.

## The pytest modules

`test_regression.py` is the defect suite. Every test in it corresponds to
something that was reproduced first and is now asserted against: the duel's
state bleed between presets, its colliding titles and doubled WINNER, its
per-frame cost, Gemini File API rejections caused by the container rather than
the content, the viral scorecard rating a pure template as highly as a
researched script, the Minimalist Motion safe area, and the Narrative Studio's
Ken Burns move and Shorts cap.

`test_modes.py` checks that every production mode is reachable, has a guidance
card and a role entry, and that each render pipeline comes out with picture and
sound the same length. `ALL_MODES` is hardcoded on purpose — its job is to fail
when a mode is added, so that adding one forces the rest of the registration to
happen with it.

`test_atmosphere.py` covers Atmosphere Studio and the YouTube publisher. The
two things that actually matter in long-form ambient are measured rather than
eyeballed:

- **Seams.** A loop that clicks once every two minutes is worse than no loop,
  because the listener is asleep and the click wakes them. Every join is
  measured against the ordinary sample-to-sample step in the same signal, and
  the suite refuses to trust a seam reading taken from silence — a NaN upstream
  produces a file that is the right length, the right format, and empty.
- **Drift geometry.** The cycle mode is stream-copied for hours, so if it does
  not return to exactly where it started there is a visible jump every cycle.
  Measured by reading the zoom off a grid rather than by comparing pixels: a
  0.04% zoom decorrelates a noisy image completely while the geometry is fine,
  and that difference sent one earlier investigation chasing a bug that was not
  there.

The publisher half needs no network. It covers the credential validation (a
service-account key is the commonest setup mistake), the metadata limits that
would otherwise fail after a multi-gigabyte upload, and the scheduling rule
that `publishAt` is only honoured on a `private` video — send it with `public`
and YouTube ignores the date and publishes immediately.

`test_suites.py` runs the standalone `suite_*.py` scripts as subprocesses.

## The standalone suites

These are scripts, not pytest modules: they print a readable account of what
they measured, which is what you want while working on the engine they cover.
They are named `suite_` rather than `test_` because pytest imports every
`test_*.py` at collection time — under the old names their whole bodies ran
during `--collect-only`, which made collection take 30 seconds and fired a live
TikTok download before a single test executed.

```bash
.venv\Scripts\python tests\suite_narrative_unit.py    # no network, ~2s
.venv\Scripts\python tests\suite_narrative_ffmpeg.py  # needs ffmpeg, ~90s
.venv\Scripts\python tests\suite_vector_scenes.py     # no network, ~30s
.venv\Scripts\python tests\suite_url_ingest.py        # needs network, ~20s
```

`suite_narrative_unit.py` covers entity extraction, storyboard JSON parsing
(fenced, prose-wrapped, trailing commas), segmentation, retiming against word
timings, image-prompt consistency and the offline fallback.

`suite_narrative_ffmpeg.py` covers the parts that are only true if you measure
them: that a Ken Burns move actually moves, that hard-cut concatenation
preserves total duration, and that the xfade dissolve chain lands on
`sum(durations)` rather than one dissolve short. It renders real clips.

`suite_vector_scenes.py` covers the stick-figure rig and every scene: all ten
poses solve to finite joints inside the frame, the two-segment torso actually
curves under load and straightens when upright, the pose aliases still resolve,
all fifteen templates draw and change across their runtime, the palette stays
monochrome on a black ground, and the easing curves are front- or back-loaded
as intended.

`suite_url_ingest.py` covers URL sanitising and the TikTok ingest fix: tracking
parameters stripped, YouTube's `?v=` preserved (stripping everything after `?`
would break it), idempotence, the five failure-message shapes, and a live
TikTok download. Set `SKIP_NETWORK=1` to skip the download.
