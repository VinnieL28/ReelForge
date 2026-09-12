"""
Every production mode renders, and picture and sound come out the same length.

An A/V offset is the failure this project keeps rediscovering in new forms --
narration clipped at the tail, a picture track that ends before the last word,
a boomerang whose loop drifts off the beat. The mux trims to the shorter
stream, so the symptom is always the same: the video is fine until the moment
it is not, and nothing in the render log says so.

These tests render genuinely short outputs (2-8 seconds) with procedural audio,
so they need no API key and no network. The engines that cannot run offline --
Gemini scripting, Gemini/edge narration -- are covered by asserting the
contract their callers rely on instead.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

import minimalist_engine as me
import video_engine as ve
from audio_engine import generate_synth_music, save_wav_to_file

# The tolerance. One frame at 24fps is 41.7ms, so anything at or under 50ms is
# sub-frame and cannot be seen or heard. The engines aim for 0.00s.
SYNC_TOLERANCE = 0.05

# Every mode the app offers. This list is deliberately hardcoded rather than
# read from app.MODE_LABELS: its job is to fail when a mode is added, so that
# adding one forces a guidance card and a role entry to be added with it.
ALL_MODES = ("dashboard", "commentary", "minimalist", "narrative", "batch", "reel",
             "duel", "atmosphere", "scout", "library", "admin")

# Modes that do not make anything. They get no guidance card, because "best
# for" and "~90s per render" are answers to questions they do not raise.
NON_PRODUCTION_MODES = ("admin", "library", "scout", "dashboard")


def av_offset(path: str) -> tuple[float, float, float]:
    """(video seconds, audio seconds, |difference|) for a rendered file."""
    from moviepy import AudioFileClip, VideoFileClip

    with VideoFileClip(path) as clip:
        video = float(clip.duration or 0.0)
    with AudioFileClip(path) as audio:
        sound = float(audio.duration or 0.0)
    return video, sound, abs(video - sound)


@pytest.fixture(scope="module")
def bed(tmp_path_factory) -> str:
    """A ten-second procedural music bed -- no network, no API key."""
    path = str(tmp_path_factory.mktemp("audio") / "bed.wav")
    stereo, rate = generate_synth_music(style="lofi", duration=10.0)
    return save_wav_to_file(stereo, rate, path)


def _photo(tone: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (1400, 1400), tone)


# ---------------------------------------------------------------------------
# Mode coverage
# ---------------------------------------------------------------------------

def test_every_mode_is_reachable_and_has_a_guide():
    """A mode with no guidance card is a mode someone forgot to finish."""
    import app

    assert set(app.MODE_LABELS) == set(ALL_MODES)
    for mode in ALL_MODES:
        if mode in NON_PRODUCTION_MODES:
            assert mode not in app.MODE_GUIDES, (
                f"{mode} makes nothing, so it should not carry a production card")
            continue
        assert mode in app.MODE_GUIDES, f"{mode} has no guidance card"
        guide = app.MODE_GUIDES[mode]
        assert guide["badges"] and guide["niche"] and guide["eta"]


def test_roles_gate_the_mode_list():
    import auth

    admin = set(auth.allowed_modes("admin"))
    creator = set(auth.allowed_modes("creator"))
    assert admin >= creator
    assert "admin" in admin and "admin" not in creator
    assert admin <= set(ALL_MODES)
    # The library is scoped to the signed-in account's own folder, so a creator
    # seeing it is a creator seeing their own work and nobody else's.
    assert "library" in creator
    # Research feeds the production modes a creator already has, so it is not
    # an admin-only tool.
    assert "scout" in creator
    # The landing page is the default view, so every role must be able to
    # reach it or sign-in lands on an access error.
    assert "dashboard" in creator and "dashboard" in admin


# ---------------------------------------------------------------------------
# A/V sync, per engine
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_reel_render_is_in_sync(workdir, bed):
    """Reel Studio: still slides, Ken Burns, captions, music bed."""
    slides = [
        {"kind": "image", "image": _photo((40, 48, 70)), "caption": "First beat",
         "duration": 2.0, "motion": "zoom_in", "role": "hook"},
        {"kind": "image", "image": _photo((70, 48, 40)), "caption": "Second beat",
         "duration": 2.0, "motion": "pan_right", "role": "point"},
    ]
    out = os.path.join(workdir, "reel.mp4")
    result = ve.build_reel_video(
        slides, fit_mode="crop_fill", transition_type="crossfade", transition_dur=0.4,
        audio_path=bed, fps=24, output_path=out, sfx_enabled=False,
    )
    assert os.path.exists(out) and os.path.getsize(out) > 10_000

    video, audio, offset = av_offset(out)
    print(f"reel: video {video:.3f}s, audio {audio:.3f}s, offset {offset:.3f}s")
    assert offset <= SYNC_TOLERANCE, f"{offset:.3f}s out of sync"
    assert abs(video - result["duration"]) <= SYNC_TOLERANCE


@pytest.mark.slow
def test_duel_render_is_in_sync(workdir, bed):
    """Versus Duel: intro, one round, winner card -- the full three-act shape."""
    item_a = {"name": "Range Rover Vogue", "hook": "British flagship", "image": _photo((64, 72, 86))}
    item_b = {"name": "Porsche Cayenne", "hook": "The driver's SUV", "image": _photo((86, 70, 54))}
    rnd = {"metric": "Horsepower", "a_score": 395, "b_score": 468, "unit": "hp",
           "winner": "B", "note": "Peak output"}

    slides = [
        {"kind": "duel_intro", "item_a": item_a, "item_b": item_b, "layout": "stacked",
         "caption": "Range Rover vs Cayenne", "duration": 2.0},
        {"kind": "duel_round", "item_a": item_a, "item_b": item_b, "round": rnd,
         "layout": "stacked", "caption": "Horsepower", "duration": 2.5},
        {"kind": "duel_winner", "winner_item": item_b, "winner_is_a": False,
         "tally": (1, 0), "layout": "stacked", "caption": "Which would you pick?",
         "duration": 2.0},
    ]
    out = os.path.join(workdir, "duel.mp4")
    result = ve.build_reel_video(
        slides, transition_type="crossfade", transition_dur=0.4,
        audio_path=bed, fps=24, output_path=out, sfx_enabled=True,
    )
    assert os.path.exists(out) and os.path.getsize(out) > 10_000

    video, audio, offset = av_offset(out)
    print(f"duel: video {video:.3f}s, audio {audio:.3f}s, offset {offset:.3f}s")
    assert offset <= SYNC_TOLERANCE, f"{offset:.3f}s out of sync"
    assert abs(video - result["duration"]) <= SYNC_TOLERANCE


@pytest.mark.slow
def test_minimalist_render_is_in_sync(workdir):
    """Minimalist Motion: drawn frames, synthesized bed, the sub-bass drop."""
    spec = me.normalise_spec({
        **me.fallback_scene_spec("consistency beats intensity", "compounding_jar", 4.0),
        "duration": 4.0,
    })
    out = os.path.join(workdir, "minimalist.mp4")
    result = me.build_minimalist_video(
        spec, out, fps=24, bgm=True, bgm_volume=0.3, sfx=True, narrate=False,
    )
    assert os.path.exists(out) and os.path.getsize(out) > 10_000

    video, audio, offset = av_offset(out)
    print(f"minimalist: video {video:.3f}s, audio {audio:.3f}s, offset {offset:.3f}s")
    assert offset <= SYNC_TOLERANCE, f"{offset:.3f}s out of sync"
    assert abs(video - float(result["duration"])) <= SYNC_TOLERANCE


@pytest.mark.slow
def test_the_drop_lands_on_the_impact_frame(workdir):
    """Minimalist Motion's whole design is that the sub-bass hits the frame the
    milestone lands on. A sync offset here is inaudible as an offset and just
    makes the video feel wrong."""
    from moviepy import AudioFileClip

    spec = me.normalise_spec({
        **me.fallback_scene_spec("compounding", "compounding_jar", 8.0),
        "animation_phases": {"draw_end": 2.0, "impact": 4.0},
    })
    out = os.path.join(workdir, "drop.mp4")
    result = me.build_minimalist_video(spec, out, fps=24, bgm=False, sfx=True, narrate=False)

    # 22.05kHz, not 8kHz: the drop is a sub-80Hz event and decimating to 8kHz
    # through moviepy's resampler loses most of it. Measured RMS at 8kHz was
    # 0.000 across every window, which read as "the drop is missing" when it
    # was in fact landing exactly on the beat.
    with AudioFileClip(out) as audio:
        samples = audio.to_soundarray(fps=22050)
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples

    window = 22050 // 4                     # 0.25s
    usable = mono[: len(mono) // window * window].reshape(-1, window)
    rms = np.sqrt((usable ** 2).mean(axis=1))
    loudest = float(np.argmax(rms)) * 0.25

    phases = me.phases_of(spec, float(result["duration"]))
    print(f"impact at {phases.impact:.2f}s, loudest 0.25s window at {loudest:.2f}s")
    assert rms.max() > 0.05, "there is no drop in the mix at all"
    assert abs(loudest - phases.impact) <= 0.3, (
        f"the drop is {abs(loudest - phases.impact):.2f}s off the impact frame")


@pytest.mark.slow
def test_narrative_concat_is_in_sync(workdir, ffmpeg, bed):
    """Narrative Studio: Ken Burns segments dissolved together must total
    exactly the sum of their durations, or the voice slides off the picture."""
    import narrative_engine as ne

    still = os.path.join(workdir, "still.png")
    Image.new("RGB", (1080, 1920), (30, 34, 44)).save(still)

    durations = [1.6, 1.8, 1.5]
    dissolve = 0.4
    clips = []
    for i, seconds in enumerate(durations):
        clip = os.path.join(workdir, f"seg{i}.mp4")
        # Every clip but the last carries the transition's worth of extra tail.
        pad = dissolve if i < len(durations) - 1 else 0.0
        ne.ken_burns_clip(still, clip, seconds + pad, (360, 640), fps=24,
                          move=ne._MOVES[i % len(ne._MOVES)], ffmpeg=ffmpeg)
        clips.append(clip)

    out = os.path.join(workdir, "episode.mp4")
    ne.concat_segments(clips, out, durations, transition="dissolve",
                       dissolve=dissolve, fps=24, ffmpeg=ffmpeg)

    from moviepy import VideoFileClip

    with VideoFileClip(out) as clip:
        total = float(clip.duration)

    expected = sum(durations)
    print(f"narrative: {total:.3f}s against {expected:.3f}s planned")
    assert abs(total - expected) <= 0.12, (
        f"the dissolve chain drifted {abs(total - expected):.3f}s — "
        "padding the clips AND subtracting the dissolve from the offset "
        "double-counts it")


@pytest.mark.slow
def test_commentary_mux_is_in_sync(workdir, bed, ffmpeg):
    """Commentary Machine: the narration decides the length, and the source
    clip is looped or trimmed to it. Both streams must come out equal."""
    import subprocess

    source = os.path.join(workdir, "source.mp4")
    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=540x960:rate=30:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", source],
                   capture_output=True, check=True)

    out = os.path.join(workdir, "commentary.mp4")
    result = ve.render_commentary_video(
        source_video=source, narration_path=bed, script="A short test line.",
        output_path=out, target_size=(540, 960), fps=24,
        loop_mode="loop", original_volume=0.0, burn_captions=False,
    )
    assert os.path.exists(out)

    video, audio, offset = av_offset(out)
    print(f"commentary: video {video:.3f}s, audio {audio:.3f}s, offset {offset:.3f}s "
          f"(narration {result['duration']:.3f}s)")
    assert offset <= SYNC_TOLERANCE, f"{offset:.3f}s out of sync"
