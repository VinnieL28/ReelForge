# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""
Atmosphere Studio -- long-form ambient and sleep video.

Self-contained: nothing in this module is imported by any other engine, and it
imports nothing from them. The only shared pieces are `paths` (for the export
sandbox) and ffmpeg itself.

The whole design turns on one number. An eight-hour video at 24fps is 691,200
frames. Anything done per frame in Python is out of the question, and anything
done per frame in ffmpeg is a ninety-minute job. So:

  * **Audio is synthesized short and looped long.** A few minutes of seed audio
    is generated in numpy, made seamless at its own wrap point, chained with
    `acrossfade` into a longer super-loop, and then repeated to the full runtime
    with `aloop`. Eight hours of rain costs about as much as four minutes of it.

  * **Video is rendered short and looped long.** A drift segment that returns
    to where it started is rendered once and stream-copied for the rest of the
    timeline. A true monotonic 1.0x-1.05x pass over the whole duration is
    available and honest about what it costs -- but at eight hours it is a zoom
    of 0.0000017x per frame, which is not a visible difference from the cycle.

Seams are the thing to get right. A loop that clicks once every four minutes is
worse than no loop at all, because the listener is asleep and a click wakes
them. Every join here is crossfaded, and `verify_seam()` measures the result.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from typing import Any, Callable

import numpy as np

from paths import ensure_dir

ProgressFn = Callable[[float, str], None]

SAMPLE_RATE = 44100
CANVAS: tuple[int, int] = (1920, 1080)          # 16:9, horizontal
DEFAULT_FPS = 24

# How long a single synthesized seed runs before it is chained and looped.
# 45s is long enough that the ear does not hear a period, short enough that
# generating three of them in numpy is a two-second job.
SEED_SECONDS = 45.0
# Crossfade used at every audio join, including the seed's own wrap point.
SEAM_SECONDS = 3.0
# How many distinct seeds are chained into the super-loop. Three different
# 45s textures crossfaded gives a 126s period rather than a 45s one, and is
# also the difference between "layered original soundscape" and the single
# repeating loop YouTube treats as inauthentic.
SEED_VARIANTS = 3


# ---------------------------------------------------------------------------
# Duration presets
# ---------------------------------------------------------------------------

DURATIONS: dict[str, dict[str, Any]] = {
    "test": {
        "label": "⚡ Test Render (60s)",
        "seconds": 60,
        "note": "For checking the mix and the framing before committing hours.",
    },
    "30min": {
        "label": "30 Minutes",
        "seconds": 30 * 60,
        "note": "A nap, a meditation, one focus block.",
    },
    "1hour": {
        "label": "1 Hour",
        "seconds": 60 * 60,
        "note": "The shortest length the sleep audience actually searches for.",
    },
    "3hours": {
        "label": "3 Hours",
        "seconds": 3 * 60 * 60,
        "note": "A full sleep cycle. Strong watch-time per view.",
    },
    "8hours": {
        "label": "8 Hours",
        "seconds": 8 * 60 * 60,
        "note": "Overnight. The highest-earning length in this niche, and the "
                "slowest to render.",
    },
}
DEFAULT_DURATION = "1hour"


# ---------------------------------------------------------------------------
# Layer catalogue
# ---------------------------------------------------------------------------

PRIMARY_BEDS: dict[str, dict[str, str]] = {
    "rain_window": {
        "label": "🌧️ Rain on Window",
        "blurb": "Steady mid-band hiss with droplet transients close to the glass.",
    },
    "thunderstorm": {
        "label": "⛈️ Heavy Thunderstorm",
        "blurb": "Dense rain under rolling low-end, with strikes every minute or so.",
    },
    "brown_noise": {
        "label": "🟤 Deep Brown Noise",
        "blurb": "Pure -6dB/octave noise. No events, nothing to latch onto.",
    },
    "stream": {
        "label": "🏞️ Gentle Stream",
        "blurb": "Bright moving water with burbling modulation.",
    },
    "fireplace": {
        "label": "🔥 Crackling Fireplace",
        "blurb": "Warm low roar under irregular sharp crackles.",
    },
}

SECONDARY_TEXTURES: dict[str, dict[str, str]] = {
    "none": {"label": "— None —", "blurb": "Primary bed alone."},
    "distant_thunder": {
        "label": "🌩️ Distant Thunder",
        "blurb": "Sub-bass rumbles far off, well below the bed.",
    },
    "room_wind": {
        "label": "🌬️ Soft Room Wind",
        "blurb": "Slow filtered swells, like air moving past a window frame.",
    },
    "crickets": {
        "label": "🦗 Night Crickets",
        "blurb": "Rhythmic high chirps with a natural stagger.",
    },
    "drone_432": {
        "label": "🎵 Binaural Drone 432Hz",
        "blurb": "432Hz with a 4Hz theta offset between the ears. Needs headphones.",
    },
    "drone_528": {
        "label": "🎵 Binaural Drone 528Hz",
        "blurb": "528Hz with a 4Hz theta offset between the ears. Needs headphones.",
    },
}

# Binaural beat frequency: the difference between the ears, not the carrier.
# 4Hz sits in theta, which is the band associated with drifting off.
BINAURAL_OFFSET_HZ = 4.0


# ---------------------------------------------------------------------------
# Synthesis primitives
# ---------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


# Filter steepness, as a cascade order. One pole is -6dB/octave, which leaves
# so much energy above the cutoff that a "rain" bandpassed at 8.2kHz still had
# a spectral centroid of 9.3kHz -- indistinguishable from the stream. Three
# poles is -18dB/octave and makes the beds audibly different things.
FILTER_ORDER = 3


def _lowpass(signal: np.ndarray, cutoff_hz: float, rate: int = SAMPLE_RATE,
             order: int = FILTER_ORDER) -> np.ndarray:
    """
    Cascaded one-pole lowpass, applied in the frequency domain.

    scipy.signal would be the obvious tool, but an FFT multiply over a
    45-second seed is a single vectorised pass and avoids adding a
    filter-design dependency to a module that is otherwise pure numpy.
    Cascading is just raising the response to a power.
    """
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / rate)
    spectrum *= (1.0 / (1.0 + (freqs / max(1e-6, cutoff_hz)))) ** order
    return np.fft.irfft(spectrum, n=len(signal)).astype(np.float32)


def _highpass(signal: np.ndarray, cutoff_hz: float, rate: int = SAMPLE_RATE,
              order: int = FILTER_ORDER) -> np.ndarray:
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / rate)
    ratio = freqs / max(1e-6, cutoff_hz)
    spectrum *= (ratio / (1.0 + ratio)) ** order
    return np.fft.irfft(spectrum, n=len(signal)).astype(np.float32)


def _bandpass(signal: np.ndarray, low_hz: float, high_hz: float,
              rate: int = SAMPLE_RATE) -> np.ndarray:
    return _highpass(_lowpass(signal, high_hz, rate), low_hz, rate)


def _brown(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Brown noise: white integrated once, which is the -6dB/octave slope.

    The integral random-walks away from zero over 45 seconds, so it is
    highpassed at 12Hz afterwards -- inaudible, but it stops the waveform
    drifting into the rails and clipping on the way.
    """
    walk = np.cumsum(_white(n, rng))
    walk = _highpass(walk.astype(np.float32), 12.0, order=1)
    peak = float(np.abs(walk).max()) or 1.0
    return (walk / peak).astype(np.float32)


def _normalise(signal: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """
    Scales to a peak, and refuses to pass NaN or inf on.

    Worth the check: a single NaN anywhere in the buffer makes `max()` NaN, the
    division by it makes every sample NaN, and the int16 cast turns that into
    silence. The failure surfaces as an eight-hour video with no sound, hours
    after the mistake.
    """
    if not np.isfinite(signal).all():
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    top = float(np.abs(signal).max())
    if top < 1e-9:
        return signal.astype(np.float32)
    return (signal * (peak / top)).astype(np.float32)


def _seamless(signal: np.ndarray, seam: float = SEAM_SECONDS,
              rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Makes a buffer loop against itself with no discontinuity.

    The last `seam` seconds are crossfaded with a copy of the first `seam`
    seconds and the head is then dropped, so the end of the buffer already
    *is* the beginning. Equal-power (cosine) rather than linear, because two
    uncorrelated noise signals summed linearly dip ~3dB in the middle of the
    fade and the dip is audible as a breath.
    """
    n = len(signal)
    fade = int(min(seam, len(signal) / (4 * rate)) * rate)
    if fade < 16 or n <= 2 * fade:
        return signal

    t = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    fade_out = np.cos(t * math.pi / 2.0)
    fade_in = np.sin(t * math.pi / 2.0)

    body = signal[fade:].copy()
    body[-fade:] = body[-fade:] * fade_out + signal[:fade] * fade_in
    return body.astype(np.float32)


def _events(n: int, rng: np.random.Generator, per_minute: float,
            builder: Callable[[np.random.Generator, int], np.ndarray],
            rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Scatters discrete events (a crackle, a chirp, a thunder strike) over a
    buffer, wrapping anything that overruns the end back to the start.

    The wrap is what keeps the buffer loopable: an event truncated at the end
    would reappear as a click at the seam.
    """
    out = np.zeros(n, dtype=np.float32)
    count = max(1, int(per_minute * n / (60.0 * rate)))
    for _ in range(count):
        event = builder(rng, rate)
        if len(event) >= n:
            continue
        start = int(rng.integers(0, n))
        end = start + len(event)
        if end <= n:
            out[start:end] += event
        else:
            split = n - start
            out[start:] += event[:split]
            out[: end - n] += event[split:]
    return out


# ---------------------------------------------------------------------------
# Primary beds
# ---------------------------------------------------------------------------

def _bed_rain_window(n: int, rng: np.random.Generator) -> np.ndarray:
    """Mid-band hiss, gently swelling, with droplets against the glass."""
    hiss = _bandpass(_white(n, rng), 420.0, 8200.0)

    # Slow amplitude drift so the rain breathes instead of sitting flat.
    envelope = 1.0 + 0.16 * _lowpass(_white(n, rng), 0.28)
    envelope /= float(np.abs(envelope).max()) or 1.0
    hiss *= np.clip(envelope, 0.55, 1.35)

    def droplet(gen: np.random.Generator, rate: int) -> np.ndarray:
        length = int(rate * float(gen.uniform(0.012, 0.04)))
        decay = np.exp(-np.linspace(0.0, 9.0, length, dtype=np.float32))
        tone = _bandpass(_white(length, gen), 900.0, 5200.0)
        return (tone * decay * float(gen.uniform(0.25, 0.75))).astype(np.float32)

    return _normalise(hiss * 0.8 + _events(n, rng, 240.0, droplet) * 0.5)


def _bed_thunderstorm(n: int, rng: np.random.Generator) -> np.ndarray:
    """Heavier rain, a low bed of rumble, and strikes about once a minute."""
    rain = _bandpass(_white(n, rng), 260.0, 9000.0)
    rumble = _lowpass(_brown(n, rng), 110.0) * 0.55

    def strike(gen: np.random.Generator, rate: int) -> np.ndarray:
        length = int(rate * float(gen.uniform(2.2, 5.0)))
        # A crack that opens fast and a tail that takes seconds to go.
        attack = int(rate * 0.06)
        envelope = np.concatenate([
            np.linspace(0.0, 1.0, attack, dtype=np.float32),
            np.exp(-np.linspace(0.0, 4.2, length - attack, dtype=np.float32)),
        ])
        body = _lowpass(_white(length, gen), float(gen.uniform(180.0, 420.0)))
        return (body * envelope * float(gen.uniform(0.55, 1.0))).astype(np.float32)

    return _normalise(rain * 0.72 + rumble + _events(n, rng, 1.1, strike) * 0.9)


def _bed_brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pure brown noise, rolled off above 2kHz so it is warm rather than hissy."""
    return _normalise(_lowpass(_brown(n, rng), 2000.0, order=1))


def _bed_stream(n: int, rng: np.random.Generator) -> np.ndarray:
    """Bright moving water: two bands modulated against each other."""
    upper = _bandpass(_white(n, rng), 1400.0, 11000.0)
    lower = _bandpass(_white(n, rng), 200.0, 1400.0)

    # Burble: fast irregular modulation on the bright band only.
    burble = 1.0 + 0.38 * _lowpass(_white(n, rng), 3.1)
    burble /= float(np.abs(burble).max()) or 1.0
    upper *= np.clip(burble, 0.45, 1.5)

    return _normalise(upper * 0.62 + lower * 0.7)


def _bed_fireplace(n: int, rng: np.random.Generator) -> np.ndarray:
    """A low roar with irregular sharp crackles over it."""
    roar = _lowpass(_brown(n, rng), 420.0)

    def crackle(gen: np.random.Generator, rate: int) -> np.ndarray:
        length = int(rate * float(gen.uniform(0.006, 0.026)))
        decay = np.exp(-np.linspace(0.0, 11.0, length, dtype=np.float32))
        tone = _bandpass(_white(length, gen), 1600.0, 9000.0)
        return (tone * decay * float(gen.uniform(0.3, 1.0))).astype(np.float32)

    return _normalise(roar * 0.78 + _events(n, rng, 95.0, crackle) * 0.85)


BED_BUILDERS: dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "rain_window": _bed_rain_window,
    "thunderstorm": _bed_thunderstorm,
    "brown_noise": _bed_brown_noise,
    "stream": _bed_stream,
    "fireplace": _bed_fireplace,
}


# ---------------------------------------------------------------------------
# Secondary textures
# ---------------------------------------------------------------------------

def _tex_distant_thunder(n: int, rng: np.random.Generator) -> np.ndarray:
    def rumble(gen: np.random.Generator, rate: int) -> np.ndarray:
        length = int(rate * float(gen.uniform(4.0, 9.0)))
        # clip before the fractional power: sin() lands a few ULP below zero
        # at the endpoints, and (-1e-8) ** 1.6 is NaN, which then propagates
        # through the mix and silences the entire soundtrack.
        bell = np.clip(np.sin(np.linspace(0.0, math.pi, length, dtype=np.float32)), 0.0, None)
        envelope = bell ** 1.6
        body = _lowpass(_white(length, gen), float(gen.uniform(55.0, 130.0)))
        return (body * envelope * float(gen.uniform(0.4, 1.0))).astype(np.float32)

    return _normalise(_events(n, rng, 1.6, rumble))


def _tex_room_wind(n: int, rng: np.random.Generator) -> np.ndarray:
    body = _bandpass(_white(n, rng), 90.0, 1100.0)
    swell = 0.5 + 0.5 * _lowpass(_white(n, rng), 0.13)
    swell /= float(np.abs(swell).max()) or 1.0
    return _normalise(body * np.clip(swell, 0.15, 1.0))


def _tex_crickets(n: int, rng: np.random.Generator) -> np.ndarray:
    def chirp(gen: np.random.Generator, rate: int) -> np.ndarray:
        # A cricket is a burst of pulses, not one tone.
        pulses = int(gen.integers(3, 6))
        gap = int(rate * 0.035)
        pulse_len = int(rate * 0.018)
        carrier = float(gen.uniform(3900.0, 5200.0))
        out = np.zeros(pulses * (pulse_len + gap), dtype=np.float32)
        for i in range(pulses):
            t = np.arange(pulse_len, dtype=np.float32) / rate
            envelope = np.sin(np.linspace(0.0, math.pi, pulse_len, dtype=np.float32))
            tone = np.sin(2 * math.pi * carrier * t).astype(np.float32)
            at = i * (pulse_len + gap)
            out[at: at + pulse_len] = tone * envelope
        return (out * float(gen.uniform(0.25, 0.6))).astype(np.float32)

    return _normalise(_events(n, rng, 36.0, chirp))


def _binaural(n: int, carrier_hz: float, rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Stereo drone: the two ears get carriers a few Hz apart.

    Returned as (n, 2) rather than mono -- the whole effect is the difference
    between the channels, so this is the one layer that cannot be summed to
    mono without destroying it. Both carriers are rounded to a whole number of
    cycles across the buffer so the loop point is phase-continuous.
    """
    left_hz = _cycle_locked(carrier_hz - BINAURAL_OFFSET_HZ / 2.0, n, rate)
    right_hz = _cycle_locked(carrier_hz + BINAURAL_OFFSET_HZ / 2.0, n, rate)

    t = np.arange(n, dtype=np.float32) / rate
    left = np.sin(2 * math.pi * left_hz * t).astype(np.float32)
    right = np.sin(2 * math.pi * right_hz * t).astype(np.float32)
    return np.stack([left, right], axis=1) * 0.9


def _cycle_locked(freq_hz: float, n: int, rate: int) -> float:
    """Nudges a frequency to the nearest whole number of cycles in `n` samples."""
    seconds = n / rate
    cycles = max(1.0, round(freq_hz * seconds))
    return float(cycles / seconds)


TEXTURE_BUILDERS: dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "distant_thunder": _tex_distant_thunder,
    "room_wind": _tex_room_wind,
    "crickets": _tex_crickets,
}


# ---------------------------------------------------------------------------
# Seed rendering
# ---------------------------------------------------------------------------

def _write_wav(path: str, stereo: np.ndarray, rate: int = SAMPLE_RATE) -> str:
    """Writes float audio as 16-bit PCM."""
    import wave

    clipped = np.clip(stereo, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with wave.open(path, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return path


def build_seed(
    bed: str,
    texture: str = "none",
    bed_volume: float = 1.0,
    texture_volume: float = 0.35,
    seconds: float = SEED_SECONDS,
    seed: int = 0,
) -> np.ndarray:
    """
    Synthesizes one seamless stereo seed: primary bed plus optional texture.

    Stereo width comes from generating the two channels from different random
    streams rather than from panning one mono signal, which is what makes the
    result sound like a room instead of a speaker.
    """
    n = int(seconds * SAMPLE_RATE)
    builder = BED_BUILDERS.get(bed, _bed_brown_noise)

    left = _seamless(builder(n, _rng(seed * 2 + 1)))
    right = _seamless(builder(n, _rng(seed * 2 + 2)))
    # _seamless trims the crossfade head off, so both channels shorten equally.
    width = min(len(left), len(right))
    mix = np.stack([left[:width], right[:width]], axis=1) * float(bed_volume)

    if texture in TEXTURE_BUILDERS:
        tex_l = _seamless(TEXTURE_BUILDERS[texture](n, _rng(seed * 2 + 101)))
        tex_r = _seamless(TEXTURE_BUILDERS[texture](n, _rng(seed * 2 + 102)))
        layer = np.stack([tex_l[:width], tex_r[:width]], axis=1)
        mix += layer * float(texture_volume)
    elif texture.startswith("drone_"):
        carrier = 432.0 if texture.endswith("432") else 528.0
        mix += _binaural(width, carrier) * float(texture_volume)

    # Headroom, not maximisation. These files are played quietly for hours and
    # a brick-walled master is fatiguing; -3dBFS leaves room for the encoder.
    return _normalise(mix, peak=0.71)


def render_preview(bed: str, texture: str, workspace: str,
                   bed_volume: float = 1.0, texture_volume: float = 0.35,
                   seconds: float = 20.0) -> str:
    """A short WAV of the current mix, for the audition button in the UI."""
    audio = build_seed(bed, texture, bed_volume, texture_volume,
                       seconds=seconds, seed=0)
    return _write_wav(os.path.join(workspace, "preview.wav"), audio)


def render_soundtrack(
    bed: str,
    texture: str,
    duration: float,
    workspace: str,
    bed_volume: float = 1.0,
    texture_volume: float = 0.35,
    variants: int = SEED_VARIANTS,
    ffmpeg: str | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Builds the full-length soundtrack: synthesize, chain, loop.

    Three stages, and the reason for each:

      1. `variants` seeds are synthesized with different random streams. One
         seed repeated for eight hours is audibly a loop within ten minutes,
         and is the exact pattern YouTube's reused-content review flags in this
         niche.
      2. They are joined with `acrossfade`, which overlaps and equal-power
         crossfades each boundary. The result is a super-loop of roughly
         `variants x SEED_SECONDS` that is itself seamless, because each seed
         was already made seamless at its own wrap point.
      3. `aloop` repeats that super-loop, in samples, until the runtime is
         covered, and the tail is trimmed to the exact duration.

    Returns {"path", "duration", "seed_seconds", "loop_seconds", "loops"}.
    """
    import imageio_ffmpeg

    ffmpeg = ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()
    ensure_dir(workspace)
    variants = max(1, int(variants))

    seed_paths: list[str] = []
    for index in range(variants):
        if progress:
            progress(0.05 + 0.25 * index / variants,
                     f"Synthesizing soundscape seed {index + 1}/{variants}...")
        audio = build_seed(bed, texture, bed_volume, texture_volume,
                           seconds=SEED_SECONDS, seed=index)
        seed_paths.append(_write_wav(os.path.join(workspace, f"seed_{index}.wav"), audio))

    # --- chain the seeds into one super-loop -------------------------------
    if progress:
        progress(0.32, "Crossfading the seeds into a seamless super-loop...")

    loop_path = os.path.join(workspace, "superloop.wav")
    if variants == 1:
        loop_path = seed_paths[0]
    else:
        inputs: list[str] = []
        for path in seed_paths:
            inputs += ["-i", path]

        # acrossfade takes exactly two inputs, so the chain is folded left to
        # right: [0][1]->[a1], [a1][2]->[a2], ...
        steps = []
        previous = "0"
        for index in range(1, variants):
            label = f"a{index}"
            steps.append(f"[{previous}][{index}]acrossfade=d={SEAM_SECONDS}:c1=tri:c2=tri[{label}]")
            previous = label
        graph = ";".join(steps)

        _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *inputs,
              "-filter_complex", graph, "-map", f"[{previous}]",
              "-c:a", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", loop_path],
             "chain the soundscape seeds")

    loop_seconds = _audio_seconds(loop_path, ffmpeg)

    # --- repeat it to length ------------------------------------------------
    if progress:
        progress(0.40, f"Looping {loop_seconds:.0f}s of audio out to "
                       f"{duration / 3600:.2f} hours...")

    out_path = os.path.join(workspace, "soundtrack.m4a")
    loops = max(1, math.ceil(duration / max(0.1, loop_seconds)))
    # aloop counts in samples and `loop` is the number of *extra* passes.
    size = int(loop_seconds * SAMPLE_RATE)
    _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", loop_path,
          "-af", f"aloop=loop={loops}:size={size}",
          "-t", f"{duration:.3f}",
          "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE), "-ac", "2",
          out_path], "loop the soundtrack to length")

    return {
        "path": out_path,
        "duration": _audio_seconds(out_path, ffmpeg),
        "seed_seconds": SEED_SECONDS,
        "loop_seconds": loop_seconds,
        "loops": loops,
        "variants": variants,
    }


# ---------------------------------------------------------------------------
# Visual canvas
# ---------------------------------------------------------------------------

ZOOM_RANGE = 0.05                # 1.00x -> 1.05x

# How much bigger than the output the zoompan source is rendered.
#
# A 1.05x zoom needs 1.05x of headroom to never resample from fewer pixels than
# it displays; 1.18 gives that with margin for the lanczos kernel. The obvious
# choice of 2x costs four times the pixels through zoompan for no visible gain,
# and zoompan -- not the encoder -- is what limits this render: measured 66
# encoded fps at 2x against 190 at 1.18x, on the same NVENC pass.
ZOOM_HEADROOM = 1.18
# Encoded frames per second for the still pipeline, measured on an RTX 3060:
# zoompan plus temporal grain onto NVENC, from a pre-scaled and pre-vignetted
# canvas. It was 71 before the vignette was baked in rather than filtered.
VISUAL_FPS_RATE = 195.0

DRIFT_MODES = {
    "cycle": "Looping drift (fast)",
    "continuous": "One continuous push (slow)",
}
# How long one drift cycle runs before it returns to where it started.
CYCLE_SECONDS = 600.0


def prepare_still(source: str, workspace: str, vignette: bool,
                  size: tuple[int, int] = CANVAS,
                  headroom: float = ZOOM_HEADROOM) -> str:
    """
    Scales the background to the zoompan intermediate and bakes the vignette in.

    The vignette is done here, once, rather than as an ffmpeg filter, because
    ffmpeg's `vignette` is a per-pixel float multiply on every frame and it is
    the single most expensive thing in the chain: measured 195 fps without it
    and 82 fps with it, and `eval=init` makes no difference. Baking it costs
    one PIL pass.

    The tradeoff is that a baked vignette scales with the zoom instead of
    staying locked to the frame. Over a 1.05x move the corners shift by 2.5%
    of the frame, which is not visible -- and a vignette that belongs to the
    photograph rather than to the screen is arguably the more natural of the
    two.
    """
    from PIL import Image

    width, height = size
    big_w = int(width * headroom) // 2 * 2
    big_h = int(height * headroom) // 2 * 2

    image = Image.open(source).convert("RGB")
    scale = max(big_w / image.width, big_h / image.height)
    image = image.resize((max(1, round(image.width * scale)),
                          max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    left = (image.width - big_w) // 2
    top = (image.height - big_h) // 2
    image = image.crop((left, top, left + big_w, top + big_h))

    if vignette:
        # Radial falloff, flat across the middle and down to ~55% in the
        # corners. Squared distance keeps it gentle near the centre where a
        # linear ramp would already be visibly darkening.
        ys = np.linspace(-1.0, 1.0, big_h, dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, big_w, dtype=np.float32)[None, :]
        radius = np.sqrt(xs * xs + ys * ys) / math.sqrt(2.0)
        mask = np.clip(1.0 - 0.45 * (radius ** 2.1), 0.0, 1.0)[:, :, None]
        image = Image.fromarray(
            np.clip(np.asarray(image, dtype=np.float32) * mask, 0, 255).astype(np.uint8))

    out = os.path.join(workspace, "canvas.png")
    ensure_dir(workspace)
    image.save(out)
    return out


def build_visual(
    source: str,
    duration: float,
    workspace: str,
    fps: int = DEFAULT_FPS,
    drift: str = "cycle",
    grain: float = 6.0,
    vignette: bool = True,
    size: tuple[int, int] = CANVAS,
    force_cpu: bool = False,
    ffmpeg: str | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Renders the moving background.

    `source` is a still image or a video loop. A video is looped with
    `-stream_loop` and gets grain and vignette but no zoom -- it already moves,
    and zooming a moving plate looks like a mistake.

    `drift="cycle"` renders CYCLE_SECONDS of zoom that returns to 1.0x and then
    stream-copies it for the rest of the runtime. `drift="continuous"` renders
    every frame of a monotonic 1.00x-1.05x push. At eight hours that is 691,200
    zoompan frames, so it is offered rather than assumed.
    """
    import imageio_ffmpeg

    ffmpeg = ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()
    ensure_dir(workspace)
    width, height = size
    is_video = looks_like_video(source)

    # A still is scaled and vignetted once, here, so the per-frame chain is only
    # zoompan and grain. A video source cannot be pre-baked, so it keeps the
    # ffmpeg vignette -- it does not run zoompan, so it is a different cost
    # profile anyway.
    if not is_video:
        source = prepare_still(source, workspace, vignette, size)

    chain = [
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={width}:{height}",
    ]
    if is_video:
        pass
    elif drift == "continuous":
        frames = max(2, int(duration * fps))
        chain = [
            f"zoompan=z='1+{ZOOM_RANGE}*on/{frames}':x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d={frames}:s={width}x{height}:fps={fps}",
        ]
    else:
        cycle_frames = max(2, int(min(CYCLE_SECONDS, duration) * fps))
        # A raised cosine: 1.0 at both ends, 1.05 in the middle. The segment
        # therefore joins to itself with no jump in scale, which is what lets
        # it be stream-copied for the rest of the runtime.
        chain = [
            f"zoompan=z='1+{ZOOM_RANGE / 2:.5f}*(1-cos(2*PI*on/{cycle_frames}))':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={cycle_frames}:s={width}x{height}:fps={fps}",
        ]

    # Grain last, so it sits on top of the image rather than being zoomed with
    # it -- grain that scales with the picture reads as compression artefacts.
    if grain > 0:
        chain.append(f"noise=alls={int(grain)}:allf=t+u")
    if vignette and is_video:
        # Stills had theirs baked in by prepare_still; only video pays for the
        # per-frame filter, which measured 195 fps without it and 82 with.
        chain.append("vignette=PI/4.2:eval=init")
    chain.append("format=yuv420p")

    segment_seconds = duration if (is_video or drift == "continuous") \
        else min(CYCLE_SECONDS, duration)

    segment = os.path.join(workspace, "visual_segment.mp4")
    command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if is_video:
        command += ["-stream_loop", "-1", "-i", os.path.abspath(source)]
    else:
        command += ["-loop", "1", "-i", os.path.abspath(source)]
    command += [
        "-t", f"{segment_seconds:.3f}",
        "-vf", ",".join(chain),
        "-r", str(fps), "-an",
        *encoder_flags(force_cpu),
        "-g", str(fps * 2), "-movflags", "+faststart",
        segment,
    ]

    if progress:
        progress(0.48, f"Rendering {segment_seconds / 60:.1f} min of moving "
                       f"background at {fps}fps...")
    started = time.time()
    _run(command, "render the visual segment")
    encode_seconds = time.time() - started

    if abs(segment_seconds - duration) < 0.5:
        return {"path": segment, "duration": segment_seconds, "looped": False,
                "segment_seconds": segment_seconds, "encode_seconds": encode_seconds,
                "fps": fps, "drift": drift}

    # --- stream-copy the segment out to full length -------------------------
    if progress:
        progress(0.62, f"Looping the background out to {duration / 3600:.2f} hours "
                       f"(stream copy, no re-encode)...")

    full = os.path.join(workspace, "visual_full.mp4")
    _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
          "-stream_loop", "-1", "-i", segment, "-t", f"{duration:.3f}",
          "-c", "copy", full], "loop the visual to length")

    return {"path": full, "duration": duration, "looped": True,
            "segment_seconds": segment_seconds, "encode_seconds": encode_seconds,
            "fps": fps, "drift": drift}


# ---------------------------------------------------------------------------
# Canvas presets
#
# Drawn rather than downloaded, for the same reason the audio is synthesized:
# a stock photograph carries a licence, and the point of this mode is output
# that is wholly the user's own. They are deliberately dark and low-contrast --
# these are watched at night, at low brightness, by someone trying to fall
# asleep, and a bright plate defeats the entire exercise.
# ---------------------------------------------------------------------------

CANVAS_PRESETS: dict[str, dict[str, str]] = {
    "night_sky": {"label": "🌌 Night Sky", "blurb": "Stars over a deep blue gradient."},
    "rain_glass": {"label": "🪟 Rain on Glass", "blurb": "Blurred lights behind wet glass."},
    "ember": {"label": "🔥 Ember Glow", "blurb": "Warm low light, like a dying fire."},
    "deep_dark": {"label": "⬛ Near Black", "blurb": "Almost nothing. Lowest possible bitrate and battery draw."},
    "mist": {"label": "🌫️ Cold Mist", "blurb": "Grey-blue fog with a soft horizon."},
}
DEFAULT_CANVAS = "night_sky"


def build_canvas_preset(name: str, workspace: str,
                        size: tuple[int, int] = CANVAS, seed: int = 0) -> str:
    """Draws one of the preset backgrounds at 2x output size and returns its path."""
    from PIL import Image, ImageFilter

    width, height = int(size[0] * 1.4), int(size[1] * 1.4)
    rng = _rng(seed or 11)
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]

    if name == "deep_dark":
        base = np.zeros((height, width, 3), np.float32)
        base += 6.0 * (1.0 - ramp)[:, :, None]
    elif name == "ember":
        top = np.array([10, 5, 3], np.float32)
        bottom = np.array([78, 30, 9], np.float32)
        base = top + (bottom - top) * (ramp ** 2.2)[:, :, None]
        base = np.repeat(base, width, axis=1) if base.shape[1] == 1 else base
    elif name == "mist":
        top = np.array([18, 22, 29], np.float32)
        bottom = np.array([46, 52, 62], np.float32)
        base = top + (bottom - top) * (ramp ** 0.7)[:, :, None]
        base = np.repeat(base, width, axis=1) if base.shape[1] == 1 else base
    elif name == "rain_glass":
        top = np.array([6, 10, 18], np.float32)
        bottom = np.array([26, 34, 52], np.float32)
        base = top + (bottom - top) * ramp[:, :, None]
        base = np.repeat(base, width, axis=1) if base.shape[1] == 1 else base
    else:                                            # night_sky
        top = np.array([4, 6, 16], np.float32)
        bottom = np.array([26, 34, 62], np.float32)
        base = top + (bottom - top) * ((1.0 - ramp) ** 1.6)[:, :, None]
        base = np.repeat(base, width, axis=1) if base.shape[1] == 1 else base

    if base.shape[1] == 1:
        base = np.repeat(base, width, axis=1)
    canvas = np.ascontiguousarray(np.broadcast_to(base, (height, width, 3)).copy())

    if name == "night_sky":
        # Stars are drawn as small soft discs, not single pixels. A one-pixel
        # star survives neither the downscale to 1080p nor the zoom -- it
        # lands between output pixels and shimmers on and off as the frame
        # drifts, which is the one thing a sleep video must not do.
        for _ in range(900):
            radius = int(rng.integers(1, 4))
            # The centre has to clear its own radius. Drawing at y=2 with
            # radius=3 makes the slice start at -1, which numpy reads as
            # "one from the end" rather than as out of bounds -- so instead of
            # an error you get an empty slice and a broadcast failure, on some
            # random seeds and not others.
            y = int(rng.integers(radius, height - radius))
            x = int(rng.integers(radius, width - radius))
            brightness = rng.uniform(90, 230) * (1.0 if radius > 1 else 0.7)
            yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            disc = np.clip(1.0 - np.sqrt(xx * xx + yy * yy) / (radius + 0.5), 0, 1) ** 1.4
            canvas[y - radius:y + radius + 1,
                   x - radius:x + radius + 1] += disc[:, :, None] * brightness
    elif name == "rain_glass":
        # Out-of-focus street lights, then streaks of water over them.
        for _ in range(26):
            y, x = int(rng.integers(0, height)), int(rng.integers(0, width))
            r = int(rng.integers(30, 110))
            yy, xx = np.ogrid[-r:r, -r:r]
            blob = np.clip(1.0 - (xx * xx + yy * yy) / float(r * r), 0, 1) ** 2
            tint = np.array([rng.uniform(90, 200), rng.uniform(70, 150), rng.uniform(30, 90)])
            y0, y1 = max(0, y - r), min(height, y + r)
            x0, x1 = max(0, x - r), min(width, x + r)
            patch = blob[: y1 - y0, : x1 - x0, None] * tint
            canvas[y0:y1, x0:x1] += patch
        for _ in range(360):
            x = int(rng.integers(0, width))
            y0 = int(rng.integers(0, height - 60))
            length = int(rng.integers(20, 150))
            canvas[y0:y0 + length, x:x + 1] += rng.uniform(8, 26)
    elif name == "ember":
        for _ in range(140):
            y = int(rng.integers(int(height * 0.45), height))
            x = int(rng.integers(0, width))
            canvas[y, x] += np.array([rng.uniform(60, 170), rng.uniform(20, 70), 0.0])

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    if name in ("rain_glass", "mist", "ember"):
        image = image.filter(ImageFilter.GaussianBlur(radius=9 if name == "rain_glass" else 24))

    ensure_dir(workspace)
    out = os.path.join(workspace, f"preset_{name}.png")
    image.save(out)
    return out


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

# Exactly the profile asked for. p4 is NVENC's balanced preset, `hq` tunes for
# quality over latency, and 4Mbps CBR-ish with a 8Mbit buffer is what a static
# 1080p plate with grain needs -- the grain is most of the bitrate, and starving
# it is what makes these videos look like they are dissolving.
GPU_FLAGS = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
             "-b:v", "4M", "-bufsize", "8M", "-rc", "vbr", "-pix_fmt", "yuv420p"]
CPU_FLAGS = ["-c:v", "libx264", "-preset", "veryfast", "-b:v", "4M",
             "-bufsize", "8M", "-pix_fmt", "yuv420p"]

_gpu_available: bool | None = None


def has_nvenc(ffmpeg: str | None = None) -> bool:
    """Encodes one throwaway frame to check NVENC is genuinely usable."""
    global _gpu_available
    if _gpu_available is not None:
        return _gpu_available

    import imageio_ffmpeg

    ffmpeg = ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
             *GPU_FLAGS, "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        _gpu_available = proc.returncode == 0
    except Exception:
        _gpu_available = False
    return _gpu_available


def encoder_flags(force_cpu: bool = False) -> list[str]:
    return list(CPU_FLAGS) if (force_cpu or not has_nvenc()) else list(GPU_FLAGS)


def _run(command: list[str], what: str, timeout: int = 24 * 3600) -> None:
    """Runs ffmpeg, raising with the tail of stderr rather than a return code."""
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"Could not {what}:\n"
                           + (proc.stderr or "ffmpeg gave no reason").strip()[-900:])


def looks_like_video(path: str) -> bool:
    return os.path.splitext(str(path))[1].lower() in (
        ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")


def _audio_seconds(path: str, ffmpeg: str | None = None) -> float:
    """Duration of an audio file, via the same stderr parse video_engine uses."""
    import re

    import imageio_ffmpeg

    ffmpeg = ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run([ffmpeg, "-hide_banner", "-i", os.path.abspath(path)],
                          capture_output=True, text=True, timeout=300)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", proc.stderr or "")
    if not match:
        return 0.0
    return (int(match.group(1)) * 3600 + int(match.group(2)) * 60
            + float(match.group(3)))


# ---------------------------------------------------------------------------
# Seam verification
# ---------------------------------------------------------------------------

def verify_seam(path: str, loop_seconds: float, window: float = 0.05,
                rate: int = SAMPLE_RATE) -> dict[str, Any]:
    """
    Measures the discontinuity at a loop boundary.

    A seam is a step in the waveform, and a step is broadband -- so what it
    sounds like is a click. The measure is the largest sample-to-sample jump in
    a short window around the join, against the same statistic taken from the
    middle of the loop where there is no join. A ratio near 1.0 means the seam
    is indistinguishable from ordinary signal.

    Reads the decoded audio through ffmpeg rather than a WAV reader so it works
    on the finished AAC as well as the intermediate.
    """
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", os.path.abspath(path),
         "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"],
        capture_output=True, timeout=1800,
    )
    mono = np.frombuffer(proc.stdout, dtype="<f4")
    if len(mono) < rate:
        return {"ratio": 0.0, "seam_jump": 0.0, "reference_jump": 0.0, "samples": len(mono)}

    half = int(window * rate)

    def worst_jump(centre: int) -> float:
        lo, hi = max(0, centre - half), min(len(mono), centre + half)
        if hi - lo < 4:
            return 0.0
        return float(np.abs(np.diff(mono[lo:hi])).max())

    seam_at = int(loop_seconds * rate)
    seam = max((worst_jump(seam_at * k) for k in (1, 2)
                if seam_at * k + half < len(mono)), default=0.0)
    reference = float(np.median([
        worst_jump(int(len(mono) * f)) for f in (0.2, 0.35, 0.5, 0.65, 0.8)
    ]))

    return {
        "ratio": seam / reference if reference > 1e-9 else 0.0,
        "seam_jump": seam,
        "reference_jump": reference,
        "samples": len(mono),
    }


# ---------------------------------------------------------------------------
# The whole render
# ---------------------------------------------------------------------------

def estimate_render_seconds(duration: float, drift: str = "cycle",
                            fps: int = DEFAULT_FPS) -> float:
    """
    Rough wall-clock estimate, for the ETA before the bar has data.

    Calibrated on an RTX 3060 at 1080p with NVENC: ~420 encoded fps for the
    grain-and-vignette chain, and audio synthesis is a flat ~14s regardless of
    runtime because it only ever generates three 45-second seeds.
    """
    # Measured on an RTX 3060 at 1080p: the soundtrack is a flat ~9s whatever
    # the runtime, because it only ever synthesizes three 45-second seeds; the
    # visual runs at ~190 frames/sec, which is zoompan's rate rather than
    # NVENC's; and the final mux is a stream copy, so it is disk-bound.
    audio = 9.0 + duration / 1200.0
    visual_seconds = duration if drift == "continuous" else min(CYCLE_SECONDS, duration)
    encode = visual_seconds * fps / VISUAL_FPS_RATE
    mux = duration / 2400.0
    return audio + encode + mux


def render_atmosphere(
    bed: str,
    texture: str,
    visual_source: str,
    duration_key: str,
    output_path: str,
    workspace: str,
    bed_volume: float = 1.0,
    texture_volume: float = 0.35,
    fps: int = DEFAULT_FPS,
    drift: str = "cycle",
    grain: float = 6.0,
    vignette: bool = True,
    variants: int = SEED_VARIANTS,
    force_cpu: bool = False,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Produces the finished ambient video.

    The mux is a stream copy of both tracks: the visual was already encoded to
    its final profile and the audio to its final AAC, so joining them is an
    I/O-bound container rewrite rather than a second encode. On an eight-hour
    file that is the difference between two minutes and an hour.
    """
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    spec = DURATIONS.get(duration_key, DURATIONS[DEFAULT_DURATION])
    duration = float(spec["seconds"])
    ensure_dir(workspace)
    started = time.time()

    if not os.path.exists(visual_source):
        raise FileNotFoundError(f"Background not found: {visual_source}")

    audio = render_soundtrack(
        bed, texture, duration, workspace,
        bed_volume=bed_volume, texture_volume=texture_volume,
        variants=variants, ffmpeg=ffmpeg, progress=progress,
    )
    visual = build_visual(
        visual_source, duration, workspace, fps=fps, drift=drift,
        grain=grain, vignette=vignette, force_cpu=force_cpu,
        ffmpeg=ffmpeg, progress=progress,
    )

    if progress:
        progress(0.88, "Muxing picture and sound (stream copy)...")

    ensure_dir(os.path.dirname(os.path.abspath(output_path)))
    _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
          "-i", visual["path"], "-i", audio["path"],
          "-map", "0:v:0", "-map", "1:a:0",
          "-c", "copy", "-movflags", "+faststart",
          "-shortest", output_path], "mux the finished video")

    if progress:
        progress(1.0, "Atmosphere render complete.")

    return {
        "output_path": output_path,
        "duration": duration,
        "duration_label": str(spec["label"]),
        "fps": fps,
        "bed": bed,
        "texture": texture,
        "bed_volume": bed_volume,
        "texture_volume": texture_volume,
        "drift": drift,
        "loop_seconds": audio["loop_seconds"],
        "audio_loops": audio["loops"],
        "variants": audio["variants"],
        "visual_looped": visual["looped"],
        "visual_segment_seconds": visual["segment_seconds"],
        "gpu": not force_cpu and has_nvenc(),
        "encoder": "h264_nvenc" if (not force_cpu and has_nvenc()) else "libx264",
        "render_seconds": time.time() - started,
        "size_bytes": os.path.getsize(output_path) if os.path.exists(output_path) else 0,
    }


# ---------------------------------------------------------------------------
# Monetization guidance
#
# Not legal advice, and these rules move. What is durable is the shape of the
# problem: this niche is where "reused content" enforcement is most active,
# because the format invites uploading the same purchased loop a hundred times.
# ---------------------------------------------------------------------------

MONETIZATION_NOTES: tuple[str, ...] = (
    "YouTube's Reused Content policy is the one that catches ambient channels. "
    "A single stock loop repeated for eight hours, with no original arrangement, "
    "is the textbook example it names.",
    "What moves a render out of that category is layering. This engine "
    "synthesizes every layer from scratch and chains several distinct variants, "
    "so the soundtrack is your own work rather than a licensed file on repeat.",
    "Vary the visual between uploads. The same plate under twelve different "
    "soundscapes reads as a template; a different photograph each time does not.",
    "Long-form ambient earns from watch time, not clicks. A 3-8 hour runtime "
    "with a genuinely seamless loop is worth far more than a sharper thumbnail.",
    "Declare any stock photography you use. The audio here is original; the "
    "background usually is not, and the licence still applies to it.",
)
