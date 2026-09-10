# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false, reportUnknownLambdaType=false
"""
Audio Generation Engine for Reels & Shorts
Generates procedural royalty-free music tracks (Lofi, Ambient, Upbeat) using NumPy & SciPy,
synthesizes ultra-realistic AI voiceovers via edge-tts, or processes uploaded custom audio.
"""

from __future__ import annotations

import os
import asyncio
import tempfile
from typing import Any, Callable, Coroutine, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

import edge_tts

_T = TypeVar("_T")
ProgressFn = Callable[[int, int, str], None]

# Narrator voices that perform well on short-form feeds. Keys are the
# Microsoft Edge neural voice IDs passed straight to edge_tts.Communicate.
VIRAL_VOICES = {
    "en-US-ChristopherNeural": "🎙️ Christopher — Deep US Male (Documentary / MrBeast energy)",
    "en-US-GuyNeural": "🔥 Guy — Punchy US Male (High-energy hype narration)",
    "en-US-JennyNeural": "✨ Jenny — Warm US Female (Lifestyle & storytelling)",
    "en-US-AriaNeural": "💫 Aria — Bright US Female (Upbeat listicles)",
    "en-US-EricNeural": "🧠 Eric — Calm US Male (Facts & explainers)",
    "en-US-MichelleNeural": "🌙 Michelle — Soft US Female (ASMR / calm luxury)",
    "en-GB-RyanNeural": "🇬🇧 Ryan — British Male (Authoritative luxury)",
    "en-GB-SoniaNeural": "🇬🇧 Sonia — British Female (Elegant narration)",
    "en-AU-NatashaNeural": "🇦🇺 Natasha — Australian Female (Friendly vlog)",
}

DEFAULT_VOICE = "en-US-ChristopherNeural"


def _run_async(coro: Coroutine[Any, Any, _T]) -> _T:
    """
    Runs an async coroutine from Streamlit's synchronous script thread.

    Streamlit executes the script body without a running event loop, so
    asyncio.run() is normally fine -- but a stale loop can be left bound to the
    thread by other libraries, so fall back to an explicit fresh loop.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def synthesize_with_word_timings(
    text: str,
    voice: str = DEFAULT_VOICE,
    output_path: str | None = None,
    rate: str = "+12%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
) -> dict[str, Any]:
    """
    Synthesizes narration AND captures per-word timing boundaries.

    Streams the synthesis rather than using Communicate.save() so the
    WordBoundary events can be collected alongside the audio. Note that
    edge-tts defaults to `boundary="SentenceBoundary"` -- without the explicit
    opt-in below you get one cue for a whole sentence and no word timings at
    all, which is useless for kinetic captions.

    Returns {"path", "duration", "words": [{"start", "end", "text"}, ...]}.
    """
    if not text or not text.strip():
        raise RuntimeError("Cannot synthesize an empty script.")

    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name

    submaker = edge_tts.SubMaker()

    async def _stream() -> None:
        communicate = edge_tts.Communicate(
            text.strip(), voice,
            rate=rate, pitch=pitch, volume=volume,
            boundary="WordBoundary",
        )
        with open(str(output_path), "wb") as handle:
            async for chunk in communicate.stream():
                # Test the discriminator inline: binding it to a local loses
                # the TypedDict narrowing that makes chunk["data"] valid.
                if chunk["type"] == "audio":
                    # "data" is optional on edge-tts's TTSChunk, so read it
                    # defensively rather than subscripting.
                    payload = chunk.get("data")
                    if payload:
                        handle.write(payload)
                elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                    submaker.feed(chunk)

    try:
        _run_async(_stream())
    except Exception as exc:
        raise RuntimeError(
            f"Voiceover synthesis failed for voice '{voice}': {exc}. "
            "edge-tts streams from Microsoft's servers, so an internet connection is required."
        ) from exc

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(
            "Voiceover synthesis produced an empty file. The text may contain no speakable characters."
        )

    words = [
        {
            "start": float(cue.start.total_seconds()),
            "end": float(cue.end.total_seconds()),
            "text": str(cue.content),
        }
        for cue in submaker.cues
        if str(cue.content).strip()
    ]

    return {
        "path": output_path,
        "duration": get_audio_duration(output_path),
        "words": words,
    }


def generate_voiceover(
    text: str,
    voice: str = DEFAULT_VOICE,
    output_path: str | None = None,
    rate: str = "+12%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
) -> str | None:
    """
    Synthesizes an ultra-realistic AI voiceover with edge-tts (free, no API key).

    rate/pitch/volume use edge-tts signed-percentage syntax, e.g. '+12%', '-5%',
    '+0Hz'. A slight rate boost is the default because short-form narration
    lands harder when it is a touch faster than natural speech.

    Returns the path to the written MP3, or None when `text` is empty.
    Raises RuntimeError if synthesis fails (e.g. no network connection).
    """
    if not text or not text.strip():
        return None

    result = synthesize_with_word_timings(
        text, voice=voice, output_path=output_path,
        rate=rate, pitch=pitch, volume=volume,
    )
    return str(result["path"])


def get_audio_duration(filepath: str) -> float:
    """Returns the duration in seconds of any audio file readable by ffmpeg."""
    from moviepy import AudioFileClip

    clip = AudioFileClip(filepath)
    try:
        return float(clip.duration)
    finally:
        clip.close()


def generate_slide_voiceovers(
    slides: Sequence[dict[str, Any]],
    voice: str = DEFAULT_VOICE,
    rate: str = "+12%",
    pitch: str = "+0Hz",
    lead_in: float = 0.35,
    tail_out: float = 0.85,
    min_duration: float = 1.5,
    progress_callback: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Synthesizes a voiceover for every slide and syncs slide durations to the
    spoken audio so no narration is ever cut off mid-sentence.

    Each slide's script is read from its 'voiceover' key, falling back to
    'caption'. Slides are mutated in place with:
      - 'voice_path':     path to the synthesized MP3
      - 'voice_duration': raw spoken length in seconds
      - 'voice_offset':   lead-in silence before narration starts
      - 'duration':       lead_in + spoken length + tail_out (min `min_duration`)

    Returns a summary dict with the total synced runtime and any per-slide errors.
    """
    errors: list[str] = []
    total_spoken = 0.0

    for idx, slide in enumerate(slides):
        script = str(slide.get("voiceover") or slide.get("caption") or "").strip()

        if progress_callback:
            progress_callback(idx, len(slides), f"Synthesizing voiceover {idx + 1}/{len(slides)}...")

        if not script:
            slide["voice_path"] = None
            slide["voice_duration"] = 0.0
            slide["voice_offset"] = 0.0
            continue

        # Slides may ask for their own pacing -- duel rounds, for instance, need
        # extra tail time so the winner reveal lands after the narration ends.
        slide_lead = float(slide.get("lead_in", lead_in))
        slide_tail = float(slide.get("tail_out", tail_out))
        slide_min = float(slide.get("min_duration", min_duration))

        try:
            path = generate_voiceover(script, voice=voice, rate=rate, pitch=pitch)
            if not path:
                raise RuntimeError("no speakable text")
            spoken = get_audio_duration(path)

            slide["voice_path"] = path
            slide["voice_duration"] = spoken
            slide["voice_offset"] = slide_lead
            slide["duration"] = round(max(slide_min, slide_lead + spoken + slide_tail), 2)
            total_spoken += spoken
        except Exception as exc:
            slide["voice_path"] = None
            slide["voice_duration"] = 0.0
            slide["voice_offset"] = 0.0
            errors.append(f"Slide {idx + 1}: {exc}")

    if progress_callback:
        progress_callback(len(slides), len(slides), "Voiceover synthesis complete.")

    return {
        "total_spoken": total_spoken,
        "total_runtime": sum(s.get("duration", 0.0) for s in slides),
        "errors": errors,
        "voiced_slides": sum(1 for s in slides if s.get("voice_path")),
    }


def generate_synth_music(
    style: str = "lofi",
    duration: float = 10.0,
    sample_rate: int = 44100,
) -> tuple[NDArray[np.float32], int]:
    """
    Procedurally synthesizes a background music track based on selected style.
    Returns: numpy array of stereo float32 audio (-1.0 to 1.0), sample_rate.
    """
    total_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    
    left = np.zeros(total_samples, dtype=np.float32)
    right = np.zeros(total_samples, dtype=np.float32)
    
    if style == "ambient":
        bpm = 72
        beat_dur = 60.0 / bpm
        chord_dur = beat_dur * 4
        
        chords = [
            [146.83, 220.00, 277.18, 369.99, 554.37],  # Dmaj7
            [123.47, 185.00, 246.94, 293.66, 440.00],  # Bm7
            [98.00, 196.00, 246.94, 293.66, 392.00],   # Gmaj7
            [110.00, 164.81, 220.00, 277.18, 440.00]   # A7
        ]
        
        for idx, chord in enumerate(chords):
            c_start = idx * chord_dur
            while c_start < duration:
                c_end = min(c_start + chord_dur, duration)
                mask = (t >= c_start) & (t < c_end)
                if not np.any(mask):
                    break
                t_sub = t[mask] - c_start
                env = np.sin(np.pi * (t_sub / (chord_dur + 0.1))) ** 1.5
                
                for freq in chord:
                    osc1 = np.sin(2 * np.pi * freq * t[mask])
                    osc2 = np.sin(2 * np.pi * (freq * 1.004) * t[mask])
                    osc3 = np.sin(2 * np.pi * (freq * 0.996) * t[mask])
                    osc4 = 0.5 * np.sin(2 * np.pi * (freq * 2) * t[mask])
                    
                    tone_l = (osc1 + osc2 + osc4) * 0.08 * env
                    tone_r = (osc1 + osc3 + osc4) * 0.08 * env
                    left[mask] += tone_l
                    right[mask] += tone_r
                
                c_start += len(chords) * chord_dur

    elif style == "upbeat":
        bpm = 124
        beat_dur = 60.0 / bpm
        num_beats = int(duration / beat_dur) + 1
        
        for b in range(num_beats):
            b_time = b * beat_dur
            k_mask = (t >= b_time) & (t < b_time + 0.35)
            if np.any(k_mask):
                t_k = t[k_mask] - b_time
                k_pitch = 45 + 110 * np.exp(-25 * t_k)
                kick = np.sin(2 * np.pi * k_pitch * t_k) * np.exp(-14 * t_k) * 0.35
                left[k_mask] += kick
                right[k_mask] += kick
            
            hh_time = b_time + (beat_dur / 2.0)
            hh_mask = (t >= hh_time) & (t < hh_time + 0.08)
            if np.any(hh_mask):
                t_hh = t[hh_mask] - hh_time
                noise = np.random.uniform(-1, 1, len(t_hh))
                hat = noise * np.exp(-50 * t_hh) * 0.09
                left[hh_mask] += hat * 0.8
                right[hh_mask] += hat * 1.1

            if b % 2 == 1:
                sn_mask = (t >= b_time) & (t < b_time + 0.22)
                if np.any(sn_mask):
                    t_sn = t[sn_mask] - b_time
                    sn_body = np.sin(2 * np.pi * 180 * t_sn) * np.exp(-20 * t_sn)
                    sn_noise = np.random.uniform(-1, 1, len(t_sn)) * np.exp(-16 * t_sn)
                    snare = (sn_body * 0.3 + sn_noise * 0.7) * 0.25
                    left[sn_mask] += snare
                    right[sn_mask] += snare
        
        # Plucky eighth-note arpeggio over the drums -- this loop was previously
        # left unfinished, so the "upbeat" style shipped with drums but no melody.
        arp_notes = [349.23, 415.30, 523.25, 622.25, 698.46, 523.25, 415.30, 349.23]
        arp_step = beat_dur / 2.0
        num_steps = int(duration / arp_step) + 1
        for s in range(num_steps):
            s_time = s * arp_step
            a_mask = (t >= s_time) & (t < s_time + arp_step * 0.95)
            if not np.any(a_mask):
                continue
            t_a = t[a_mask] - s_time
            freq = arp_notes[s % len(arp_notes)]
            env = np.exp(-6.0 * t_a) * (1.0 - np.exp(-260.0 * t_a))
            tone = (
                np.sin(2 * np.pi * freq * t_a) * 0.6
                + np.sin(4 * np.pi * freq * t_a) * 0.25
                + np.sin(6 * np.pi * freq * t_a) * 0.10
            ) * env * 0.13
            # Alternate the stereo placement so the arp shimmers across the field.
            pan = 0.62 if s % 2 == 0 else 1.0
            left[a_mask] += tone * pan
            right[a_mask] += tone * (1.62 - pan)
    else:
        # "Lofi Chill" - Warm 85 BPM relaxed beat with electric piano chords
        bpm = 85
        beat_dur = 60.0 / bpm
        bar_dur = beat_dur * 4
        
        lofi_chords = [
            [174.61, 220.00, 261.63, 329.63], # Fmaj7
            [164.81, 196.00, 246.94, 293.66], # Em7
            [146.83, 174.61, 220.00, 261.63], # Dm7
            [130.81, 164.81, 196.00, 246.94], # Cmaj7
        ]
        
        num_bars = int(duration / bar_dur) + 1
        for bar in range(num_bars):
            chord = lofi_chords[bar % len(lofi_chords)]
            b_start = bar * bar_dur
            b_mask = (t >= b_start) & (t < b_start + bar_dur)
            if np.any(b_mask):
                t_b = t[b_mask] - b_start
                flutter = 1.0 + 0.004 * np.sin(2 * np.pi * 3.5 * t_b)
                tremolo = 0.85 + 0.15 * np.sin(2 * np.pi * 5.0 * t_b)
                env = np.exp(-0.7 * t_b) * tremolo
                
                for freq in chord:
                    w_freq = freq * flutter
                    c_left = (
                        np.sin(2 * np.pi * w_freq * t[b_mask]) * 0.7 +
                        np.sin(4 * np.pi * w_freq * t[b_mask]) * 0.2
                    ) * env * 0.12
                    c_right = (
                        np.sin(2 * np.pi * (w_freq * 1.002) * t[b_mask]) * 0.7 +
                        np.sin(4 * np.pi * (w_freq * 1.002) * t[b_mask]) * 0.2
                    ) * env * 0.12
                    left[b_mask] += c_left
                    right[b_mask] += c_right

        # Sub-bass
        for bar in range(num_bars):
            root_freq = lofi_chords[bar % len(lofi_chords)][0] / 2.0
            b_start = bar * bar_dur
            b_mask = (t >= b_start) & (t < b_start + bar_dur)
            if np.any(b_mask):
                t_b = t[b_mask] - b_start
                bass_env = np.exp(-0.5 * t_b)
                sub = np.sin(2 * np.pi * root_freq * t[b_mask]) * bass_env * 0.22
                left[b_mask] += sub
                right[b_mask] += sub

        # Drum beats
        num_beats = int(duration / beat_dur) + 1
        for b in range(num_beats):
            b_time = b * beat_dur
            if b % 4 in [0, 2]:
                k_mask = (t >= b_time) & (t < b_time + 0.3)
                if np.any(k_mask):
                    t_k = t[k_mask] - b_time
                    k_pitch = 48 + 90 * np.exp(-35 * t_k)
                    kick = np.sin(2 * np.pi * k_pitch * t_k) * np.exp(-18 * t_k) * 0.28
                    left[k_mask] += kick
                    right[k_mask] += kick
            
            if b % 4 in [1, 3]:
                sn_mask = (t >= b_time) & (t < b_time + 0.18)
                if np.any(sn_mask):
                    t_sn = t[sn_mask] - b_time
                    sn_tone = np.sin(2 * np.pi * 210 * t_sn) * np.exp(-25 * t_sn) * 0.2
                    sn_noise = np.random.uniform(-1, 1, len(t_sn)) * np.exp(-22 * t_sn) * 0.15
                    sn = (sn_tone + sn_noise) * 0.25
                    left[sn_mask] += sn
                    right[sn_mask] += sn
            
            for offset in [0.0, beat_dur / 2.0]:
                hh_t = b_time + offset
                hh_mask = (t >= hh_t) & (t < hh_t + 0.06)
                if np.any(hh_mask):
                    t_hh = t[hh_mask] - hh_t
                    hh = np.random.uniform(-1, 1, len(t_hh)) * np.exp(-60 * t_hh) * 0.05
                    left[hh_mask] += hh
                    right[hh_mask] += hh
                    
        crackle = np.random.uniform(-0.015, 0.015, total_samples).astype(np.float32)
        left += crackle
        right += crackle

    fade_len = int(1.5 * sample_rate)
    if fade_len > 0 and total_samples > fade_len:
        fade_curve = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        left[-fade_len:] *= fade_curve
        right[-fade_len:] *= fade_curve

    max_amp = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-6)
    target_peak = 0.85
    gain = target_peak / max_amp
    left = np.clip(left * gain, -1.0, 1.0)
    right = np.clip(right * gain, -1.0, 1.0)
    
    stereo = np.column_stack((left, right))
    return stereo, sample_rate


# ---------------------------------------------------------------------------
# Kinetic sound design
#
# Short-form comparison content lives on its edit rhythm: a riser into each cut,
# a mechanical hit as a stat lands, a chime on the verdict. These are
# synthesized rather than sampled so the app ships with no audio assets and no
# licensing questions.
# ---------------------------------------------------------------------------

SFX_WHOOSH = "whoosh"
SFX_IMPACT = "impact"
SFX_CHIME = "chime"

_sfx_cache: dict[tuple[str, int], str] = {}


def _stereo(
    left: NDArray[np.floating[Any]],
    right: NDArray[np.floating[Any]],
    peak: float,
) -> NDArray[np.float32]:
    """Normalizes a pair of channels to `peak` and stacks them."""
    hottest = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-6)
    gain = peak / hottest
    return np.column_stack((
        np.clip(left * gain, -1.0, 1.0),
        np.clip(right * gain, -1.0, 1.0),
    )).astype(np.float32)


def make_whoosh(duration: float = 1.10, sample_rate: int = 44100) -> NDArray[np.float32]:
    """
    Heavy sub riser + air sweep for transitions.

    Builds tension across the whole length, then drops sharply at the cut point
    so the following slide lands in the gap the riser leaves behind.
    """
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)
    p = (t / duration).astype(np.float32)

    # Sub sweep 38 -> 96 Hz, phase-integrated so the pitch glides smoothly.
    sub_f = 38.0 + 58.0 * p ** 2
    sub = np.sin(2 * np.pi * np.cumsum(sub_f) / sample_rate).astype(np.float32)
    sub *= (p ** 1.5) * 0.85

    # Air: white noise pushed through a rising one-pole high-pass, which turns
    # a flat hiss into a sweep without needing an FFT filter bank.
    noise = np.random.uniform(-1, 1, n).astype(np.float32)
    air = np.empty(n, dtype=np.float32)
    prev_in = prev_out = np.float32(0.0)
    for i in range(n):
        # Cutoff climbs with the riser; coefficient approaches 1 = more highs.
        a = 0.55 + 0.44 * float(p[i])
        prev_out = np.float32(a * (prev_out + noise[i] - prev_in))
        prev_in = noise[i]
        air[i] = prev_out
    air *= (p ** 2.2) * 0.55

    # Fast release at the very end so the riser clears the downbeat.
    tail = np.ones(n, dtype=np.float32)
    rel = max(1, int(0.06 * sample_rate))
    tail[-rel:] = np.linspace(1.0, 0.0, rel, dtype=np.float32)

    body = (sub + air) * tail
    # Slight stereo divergence on the air layer keeps it wide but centred low.
    return _stereo(body + air * 0.10, body - air * 0.10, peak=0.80)


def make_impact(duration: float = 0.30, sample_rate: int = 44100) -> NDArray[np.float32]:
    """Crisp mechanical click + weighted thump for stat badges landing."""
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)

    # Transient click: very fast noise burst.
    click = np.random.uniform(-1, 1, n).astype(np.float32) * np.exp(-260.0 * t)

    # Mechanical body: two detuned high partials, snappy decay.
    body = (
        np.sin(2 * np.pi * 1850 * t) * 0.6
        + np.sin(2 * np.pi * 2760 * t) * 0.3
    ).astype(np.float32) * np.exp(-58.0 * t)

    # Low thump gives the hit weight on phone speakers.
    thump_f = 150.0 * np.exp(-26.0 * t) + 52.0
    thump = np.sin(2 * np.pi * np.cumsum(thump_f) / sample_rate).astype(np.float32)
    thump *= np.exp(-30.0 * t) * 0.75

    mix = click * 0.55 + body * 0.5 + thump
    return _stereo(mix, mix, peak=0.72)


def make_chime(duration: float = 1.30, sample_rate: int = 44100) -> NDArray[np.float32]:
    """Bright bell/ding for the winner award -- a major triad struck once."""
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)

    # C6 major triad plus an octave partial; bell-like inharmonic decay rates.
    partials = ((1046.50, 1.00, 3.2), (1318.51, 0.55, 4.1), (1567.98, 0.42, 4.8), (2093.00, 0.30, 6.0))
    tone = np.zeros(n, dtype=np.float32)
    for freq, amp, decay in partials:
        tone += (np.sin(2 * np.pi * freq * t) * amp * np.exp(-decay * t)).astype(np.float32)

    # Strike transient so it reads as struck, not faded in.
    tone += np.random.uniform(-1, 1, n).astype(np.float32) * np.exp(-420.0 * t) * 0.22

    # Gentle shimmer, opposed per channel for width.
    shimmer = 1.0 + 0.03 * np.sin(2 * np.pi * 5.5 * t)
    return _stereo(tone * shimmer, tone * (2.0 - shimmer), peak=0.68)


def make_hook_impact(duration: float = 0.50, sample_rate: int = 44100) -> NDArray[np.float32]:
    """
    The opening anchor: a short sub-bass drop with an air whoosh over it.

    Sits at 0.0s underneath the first words of narration to stop the scroll.
    Kept to half a second and weighted low so it lands as a physical thump on
    a phone speaker without masking the voice above it.
    """
    n = max(1, int(duration * sample_rate))
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)
    p = (t / duration).astype(np.float32)

    # Sub drop: 120 Hz falling to 38 Hz, phase-integrated so the pitch glides.
    sub_f = 120.0 * np.exp(-4.2 * t) + 38.0
    sub = np.sin(2 * np.pi * np.cumsum(sub_f) / sample_rate).astype(np.float32)
    sub *= np.exp(-5.0 * t) * 0.95

    # Transient click so the hit has a defined front edge.
    click = (np.random.uniform(-1, 1, n).astype(np.float32) * np.exp(-90.0 * t) * 0.30)

    # Air: noise sweeping downward, opposite the riser used at transitions.
    noise = np.random.uniform(-1, 1, n).astype(np.float32)
    air = np.empty(n, dtype=np.float32)
    prev_in = prev_out = np.float32(0.0)
    for i in range(n):
        a = 0.94 - 0.40 * float(p[i])     # closes over time = darkening sweep
        prev_out = np.float32(a * (prev_out + noise[i] - prev_in))
        prev_in = noise[i]
        air[i] = prev_out
    air *= np.exp(-7.0 * t) * 0.32

    body = sub + click + air
    # Slight width on the air only; the sub stays mono and centred.
    return _stereo(body + air * 0.12, body - air * 0.12, peak=0.88)


SFX_HOOK = "hook"

_SFX_BUILDERS = {
    SFX_WHOOSH: make_whoosh,
    SFX_IMPACT: make_impact,
    SFX_CHIME: make_chime,
    SFX_HOOK: make_hook_impact,
}


def get_sfx_path(name: str, sample_rate: int = 44100) -> str:
    """
    Renders a named SFX to a cached WAV and returns its path.

    Cached per process: a duel schedules dozens of hits, and re-synthesizing
    each one would dominate render time.
    """
    key = (name, sample_rate)
    cached = _sfx_cache.get(key)
    if cached and os.path.exists(cached):
        return cached

    builder = _SFX_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown SFX '{name}'. Choose from {sorted(_SFX_BUILDERS)}.")

    audio = builder(sample_rate=sample_rate)
    path = os.path.join(tempfile.gettempdir(), f"reelforge_sfx_{name}_{sample_rate}.wav")
    save_wav_to_file(audio, sample_rate, path)
    _sfx_cache[key] = path
    return path


def save_wav_to_file(stereo_audio: NDArray[np.float32], sample_rate: int, filepath: str) -> str:
    """
    Saves stereo float32 audio as 16-bit PCM WAV file.
    """
    audio_int16 = (stereo_audio * 32767).astype(np.int16)
    wavfile.write(filepath, sample_rate, audio_int16)
    return filepath


# ---------------------------------------------------------------------------
# Licensed narration (Gemini TTS)
#
# edge-tts is excellent for drafting -- it is free and returns exact word
# boundaries -- but its endpoint is not licensed for redistributing audio in
# monetized content. Gemini TTS runs on the API key this project already uses
# and Google's terms permit commercial use of the output.
#
# The trade: Gemini returns no word timings, so kinetic captions fall back to
# the estimator below rather than exact boundaries.
# ---------------------------------------------------------------------------

# A curated slice of Gemini's prebuilt voices that suit short-form narration.
GEMINI_VOICES: dict[str, str] = {
    "Charon": "🎙️ Charon — Deep, informative (documentary narration)",
    "Kore": "🔥 Kore — Firm and punchy (high-energy hype)",
    "Fenrir": "⚡ Fenrir — Excitable (fast reaction energy)",
    "Puck": "✨ Puck — Upbeat and bright (light commentary)",
    "Aoede": "💫 Aoede — Breezy (lifestyle storytelling)",
    "Leda": "🌙 Leda — Youthful and warm (soft narration)",
    "Orus": "🧠 Orus — Steady and clear (facts & explainers)",
}

GEMINI_TTS_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
)

GEMINI_TTS_RATE = 24000          # Gemini returns 24 kHz mono signed 16-bit PCM


def _pcm16_to_wav(pcm: bytes, output_path: str, rate: int = GEMINI_TTS_RATE) -> str:
    """Wraps raw signed 16-bit mono PCM in a WAV container."""
    import wave

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with wave.open(output_path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return output_path


def _speech_regions(
    samples: NDArray[np.float32],
    sample_rate: int,
    frame_seconds: float = 0.02,
    min_gap: float = 0.12,
) -> list[tuple[float, float]]:
    """
    Finds the stretches of a narration track that actually contain speech.

    Word timings are then laid out across these regions instead of across the
    whole file, so pauses between sentences do not drag the captions out of
    sync with the voice.
    """
    frame = max(1, int(frame_seconds * sample_rate))
    usable = (len(samples) // frame) * frame
    if usable <= 0:
        return []

    energy = np.sqrt((samples[:usable].reshape(-1, frame) ** 2).mean(axis=1))
    if not np.any(energy > 0):
        return []

    # Threshold relative to the track's own loudness, so it adapts to level.
    loud = float(np.percentile(energy[energy > 0], 90))
    voiced = energy > max(loud * 0.12, 1e-4)

    regions: list[tuple[float, float]] = []
    start: int | None = None
    for index, is_voiced in enumerate(voiced):
        if is_voiced and start is None:
            start = index
        elif not is_voiced and start is not None:
            regions.append((start * frame_seconds, index * frame_seconds))
            start = None
    if start is not None:
        regions.append((start * frame_seconds, len(voiced) * frame_seconds))

    # Bridge micro-gaps; a pause inside a word is not a sentence break.
    merged: list[tuple[float, float]] = []
    for region in regions:
        if merged and region[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], region[1])
        else:
            merged.append(region)
    return merged


def estimate_word_timings(audio_path: str, text: str) -> list[dict[str, Any]]:
    """
    Approximates per-word timings for a narration track that has none.

    Words are weighted by length and laid out across the detected speech
    regions only -- pauses are skipped, so a sentence break does not drag the
    captions out of sync.

    This is an estimate, not forced alignment. Measured against edge-tts's real
    boundaries on the same audio: mean error 0.31s, median 0.22s, worst 0.90s
    over an 8s take, with the error accumulating across a long sentence.
    Syllable-based weighting was tried and scored no better. Use the edge-tts
    provider when frame-accurate captions matter more than the licence.
    """
    words = [w for w in (text or "").split() if w.strip()]
    if not words:
        return []

    samples = _load_mono(audio_path, 16000)
    total = len(samples) / 16000.0
    regions = _speech_regions(samples, 16000) or [(0.0, total)]
    speech_total = sum(end - start for start, end in regions) or total

    weights = [float(len(w.strip(".,!?:;\"'")) + 1) for w in words]
    weight_total = sum(weights) or 1.0

    def at(position: float) -> float:
        """Maps a position along concatenated speech onto real wall-clock time."""
        remaining = position
        for start, end in regions:
            span = end - start
            if remaining <= span:
                return start + remaining
            remaining -= span
        return regions[-1][1]

    timings: list[dict[str, Any]] = []
    cursor = 0.0
    for word, weight in zip(words, weights):
        share = speech_total * (weight / weight_total)
        timings.append({
            "start": round(at(cursor), 3),
            "end": round(at(min(cursor + share, speech_total)), 3),
            "text": word.strip(".,!?:;\"'") or word,
        })
        cursor += share
    return timings


def synthesize_gemini(
    text: str,
    voice: str = "Charon",
    output_path: str | None = None,
    style: str = "",
    models: Sequence[str] = GEMINI_TTS_MODELS,
) -> dict[str, Any]:
    """
    Synthesizes narration with Gemini TTS, licensed for commercial use.

    `style` is a plain-language delivery note ("fast, punchy") -- Gemini takes
    direction in the prompt rather than through rate/pitch parameters.
    """
    if not text or not text.strip():
        raise RuntimeError("Cannot synthesize an empty script.")

    from google.genai import types

    from gemini_engine import generate_with_retry, get_client

    client = get_client()
    prompt = f"{style.strip()}: {text.strip()}" if style.strip() else text.strip()

    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    errors: list[str] = []
    for model in models:
        try:
            response = generate_with_retry(
                client, model, prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                        )
                    ),
                ),
            )
            # Every hop here is optional in the SDK's typing, and a safety
            # block genuinely does return a candidate with no parts.
            candidates = getattr(response, "candidates", None) or []
            parts = getattr(getattr(candidates[0], "content", None), "parts", None) if candidates else None
            inline = getattr(parts[0], "inline_data", None) if parts else None
            pcm = getattr(inline, "data", None) if inline else None
            if not pcm:
                errors.append(f"{model}: returned no audio (likely blocked or empty response)")
                continue

            _pcm16_to_wav(pcm, output_path)
            return {
                "path": output_path,
                "duration": len(pcm) / 2.0 / GEMINI_TTS_RATE,
                "words": estimate_word_timings(output_path, text),
                "provider": "gemini",
                "voice": voice,
                "model": model,
                "timings_exact": False,
            }
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:140]}")

    raise RuntimeError(
        "Gemini TTS failed on every model:\n" + "\n".join(f"• {e}" for e in errors)
    )


def synthesize_narration(
    text: str,
    provider: str = "gemini",
    voice: str = "",
    output_path: str | None = None,
    rate: str = "+12%",
    pitch: str = "+0Hz",
    style: str = "",
) -> dict[str, Any]:
    """
    One entry point for narration, whichever provider is selected.

    Always returns {"path", "duration", "words", "provider", "voice",
    "timings_exact"} so callers do not care which engine produced it.
    """
    if provider == "gemini":
        return synthesize_gemini(
            text, voice=voice or "Charon", output_path=output_path, style=style,
        )

    result = synthesize_with_word_timings(
        text, voice=voice or DEFAULT_VOICE, output_path=output_path,
        rate=rate, pitch=pitch,
    )
    return {
        "path": result["path"],
        "duration": result["duration"],
        "words": result["words"],
        "provider": "edge",
        "voice": voice or DEFAULT_VOICE,
        "model": "edge-tts",
        "timings_exact": True,
    }


# ---------------------------------------------------------------------------
# Suspense background music + auto-ducking
#
# The BGM is synthesized rather than shipped as an asset: no file to license,
# no file to lose, and it can be generated to the exact length of the narration
# so there is never a loop seam.
# ---------------------------------------------------------------------------

BGM_DEFAULT_VOLUME = 0.12
BGM_DUCK_DEPTH = 0.65        # how far the bed drops while the voice is talking


def make_suspense_bgm(duration: float, sample_rate: int = 44100) -> NDArray[np.float32]:
    """
    Synthesizes a suspenseful ambient bed: sub-bass drone, minor-triad pad,
    slow breathing swell and a distant pulse.

    Generated to the exact requested length, so it never has to loop.
    """
    n = max(1, int(duration * sample_rate))
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)

    # Very slow "breathing" so the bed feels alive rather than static.
    breath = (0.72 + 0.28 * np.sin(2 * np.pi * 0.055 * t - np.pi / 2)).astype(np.float32)

    # Sub-bass root (A1) with a touch of drift for weight without mud.
    sub = np.sin(2 * np.pi * 55.0 * t + 0.35 * np.sin(2 * np.pi * 0.07 * t)).astype(np.float32) * 0.55
    sub += np.sin(2 * np.pi * 82.41 * t).astype(np.float32) * 0.16      # E2 fifth

    # A-minor pad, slightly detuned per voice so it shimmers.
    pad = np.zeros(n, dtype=np.float32)
    for freq, amp, detune in ((110.00, 0.20, 0.6), (130.81, 0.16, -0.5), (164.81, 0.13, 0.4)):
        pad += (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)
        pad += (np.sin(2 * np.pi * (freq + detune) * t) * amp * 0.7).astype(np.float32)

    # Tension shimmer: a quiet high partial that wavers.
    shimmer = (np.sin(2 * np.pi * 1318.5 * t) *
               (0.012 + 0.010 * np.sin(2 * np.pi * 0.13 * t))).astype(np.float32)

    # Distant pulse every 4 seconds -- the "something is coming" heartbeat.
    pulse = np.zeros(n, dtype=np.float32)
    period = 4.0
    for k in range(int(duration / period) + 1):
        start = k * period
        idx = int(start * sample_rate)
        span = min(int(1.2 * sample_rate), n - idx)
        if span <= 0:
            continue
        local = t[:span] - t[0]
        env = np.exp(-3.2 * local).astype(np.float32)
        pulse[idx:idx + span] += (np.sin(2 * np.pi * 41.2 * local) * env * 0.22).astype(np.float32)

    body = (sub * breath + pad * 0.55 * breath + shimmer + pulse).astype(np.float32)

    # Gentle stereo width: the pad leans, the sub stays centred.
    left = body + pad * 0.05
    right = body - pad * 0.05

    # Fade the very start and end so it never clicks in or out.
    fade = max(1, int(0.6 * sample_rate))
    if n > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        left[:fade] *= ramp
        right[:fade] *= ramp
        left[-fade:] *= ramp[::-1]
        right[-fade:] *= ramp[::-1]

    return _stereo(left, right, peak=0.55)


def _load_mono(path: str, sample_rate: int) -> NDArray[np.float32]:
    """Decodes any audio file to a mono float array at `sample_rate`."""
    from moviepy import AudioFileClip

    clip = AudioFileClip(path)
    try:
        samples = np.asarray(clip.to_soundarray(fps=sample_rate), dtype=np.float32)
    finally:
        clip.close()

    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return samples.astype(np.float32)


def speech_duck_curve(
    narration_path: str,
    duration: float,
    sample_rate: int = 44100,
    duck_depth: float = BGM_DUCK_DEPTH,
    control_rate: int = 200,
    attack: float = 0.05,
    release: float = 0.40,
) -> NDArray[np.float32]:
    """
    Builds a per-sample gain curve that pulls the music down while the narrator
    speaks -- a sidechain compressor done in numpy.

    Gain drops fast (attack) when speech starts so the first syllable is never
    buried, and recovers slowly (release) so the bed doesn't pump between words.
    """
    speech = _load_mono(narration_path, sample_rate)
    total = max(1, int(duration * sample_rate))

    block = max(1, sample_rate // control_rate)
    blocks = max(1, total // block + 1)

    # Peak level per control block, padded out to the full video duration.
    padded = np.zeros(blocks * block, dtype=np.float32)
    usable = min(len(speech), len(padded))
    padded[:usable] = np.abs(speech[:usable])
    env = padded.reshape(blocks, block).max(axis=1)

    # Reference level from the loud part of the speech, so the depth of the
    # duck doesn't depend on how hot the TTS happened to render.
    loud = float(np.percentile(env[env > 1e-4], 85)) if np.any(env > 1e-4) else 1.0
    presence = np.clip(env / max(loud * 0.35, 1e-4), 0.0, 1.0).astype(np.float32)

    # One-pole smoothing with separate attack/release coefficients.
    a_coef = float(np.exp(-1.0 / max(attack * control_rate, 1.0)))
    r_coef = float(np.exp(-1.0 / max(release * control_rate, 1.0)))
    smoothed = np.empty_like(presence)
    running = 0.0
    for i, value in enumerate(presence):
        coef = a_coef if value > running else r_coef
        running = coef * running + (1.0 - coef) * float(value)
        smoothed[i] = running

    gain_ctrl = (1.0 - float(duck_depth) * smoothed).astype(np.float32)

    # Upsample the control-rate curve to audio rate.
    ctrl_t = np.arange(blocks, dtype=np.float32) * (block / sample_rate)
    audio_t = np.arange(total, dtype=np.float32) / sample_rate
    return np.interp(audio_t, ctrl_t, gain_ctrl).astype(np.float32)


def build_ducked_bgm(
    duration: float,
    narration_path: str | None = None,
    output_path: str | None = None,
    sample_rate: int = 44100,
    volume: float = BGM_DEFAULT_VOLUME,
    duck_depth: float = BGM_DUCK_DEPTH,
) -> str:
    """
    Renders the suspense bed at `volume`, auto-ducked against the narration,
    and writes it to a WAV ready to drop into the render's audio mix.
    """
    bed = make_suspense_bgm(duration, sample_rate)

    if narration_path and os.path.exists(narration_path):
        gain = speech_duck_curve(
            narration_path, duration, sample_rate=sample_rate, duck_depth=duck_depth,
        )
        span = min(len(gain), bed.shape[0])
        bed = bed[:span] * gain[:span, None]

    bed = np.clip(bed * float(volume), -1.0, 1.0).astype(np.float32)

    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    save_wav_to_file(bed, sample_rate, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Minimalist Motion: atmospheric phonk bed, climax drop, silence trimming
#
# Same reasoning as the suspense bed -- synthesized to the exact length asked
# for, so there is no loop seam to hide and no sample to license.
# ---------------------------------------------------------------------------

PHONK_BPM = 62.0                 # slowed: the whole point of the genre
PHONK_REVERB_SECONDS = 1.9


def _reverb(signal: NDArray[np.float32], sample_rate: int,
            decay: float = PHONK_REVERB_SECONDS, wet: float = 0.38) -> NDArray[np.float32]:
    """
    Convolution reverb against a decaying-noise impulse.

    A synthetic impulse rather than a recorded one: it costs nothing, ships
    with no licence, and for a wash this far back in the mix the difference
    from a real room is inaudible.
    """
    from scipy.signal import fftconvolve

    n = max(16, int(decay * sample_rate))
    rng = np.random.default_rng(7)
    envelope = np.exp(-4.2 * np.linspace(0.0, 1.0, n, dtype=np.float32))
    impulse = (rng.standard_normal(n).astype(np.float32) * envelope)
    impulse[0] = 1.0
    impulse /= np.abs(impulse).sum() or 1.0

    tail = fftconvolve(signal, impulse, mode="full")[:len(signal)].astype(np.float32)
    return ((1.0 - wet) * signal + wet * tail).astype(np.float32)


def make_phonk_bed(duration: float, sample_rate: int = 44100) -> NDArray[np.float32]:
    """
    A slowed, reverbed atmospheric bed: 808 sub, minor pad, muted cowbell,
    tape noise.

    Written to the exact duration requested, so it never needs looping.
    """
    n = max(1, int(duration * sample_rate))
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)
    beat = 60.0 / PHONK_BPM

    # --- 808 sub, pitch-glided down on each hit ----------------------------
    sub = np.zeros(n, dtype=np.float32)
    hits = [k * beat for k in range(int(duration / beat) + 2)
            if (k % 4) in (0, 3)]                       # on 1 and the "and" of 3
    for start in hits:
        idx = int(start * sample_rate)
        span = min(int(beat * 1.6 * sample_rate), n - idx)
        if span <= 0:
            continue
        local = np.arange(span, dtype=np.float32) / sample_rate
        glide = 58.0 * np.exp(-1.5 * local) + 34.0      # 92 Hz down to a 34 Hz floor
        phase = 2 * np.pi * np.cumsum(glide) / sample_rate
        env = np.exp(-2.1 * local).astype(np.float32)
        sub[idx:idx + span] += (np.sin(phase) * env * 0.85).astype(np.float32)

    # --- detuned minor pad, breathing slowly -------------------------------
    breath = (0.70 + 0.30 * np.sin(2 * np.pi * 0.045 * t - np.pi / 2)).astype(np.float32)
    pad = np.zeros(n, dtype=np.float32)
    for freq, amp in ((110.00, 0.22), (130.81, 0.17), (164.81, 0.14), (220.00, 0.07)):
        pad += (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)
        pad += (np.sin(2 * np.pi * (freq * 1.004) * t) * amp * 0.8).astype(np.float32)
    pad *= breath

    # --- muted cowbell, the genre signature, kept well back ----------------
    bell = np.zeros(n, dtype=np.float32)
    pattern = (0.0, 1.5, 2.0, 3.5)
    bars = int(duration / (beat * 4)) + 1
    for bar in range(bars):
        for offset in pattern:
            start = (bar * 4 + offset) * beat
            idx = int(start * sample_rate)
            span = min(int(0.16 * sample_rate), n - idx)
            if span <= 0:
                continue
            local = np.arange(span, dtype=np.float32) / sample_rate
            env = np.exp(-26.0 * local).astype(np.float32)
            tone = (np.sin(2 * np.pi * 540.0 * local)
                    + 0.6 * np.sin(2 * np.pi * 800.0 * local)).astype(np.float32)
            bell[idx:idx + span] += (tone * env * 0.10).astype(np.float32)

    # --- tape noise floor --------------------------------------------------
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(n).astype(np.float32)
    noise = np.convolve(noise, np.ones(48, dtype=np.float32) / 48, mode="same").astype(np.float32)
    noise *= 0.020

    body = (sub + pad * 0.42 + bell + noise).astype(np.float32)
    body = _reverb(body, sample_rate)

    left = body + pad * 0.05
    right = body - pad * 0.05

    fade = max(1, int(0.75 * sample_rate))
    if n > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        left[:fade] *= ramp
        right[:fade] *= ramp
        left[-fade:] *= ramp[::-1]
        right[-fade:] *= ramp[::-1]

    return _stereo(left, right, peak=0.62)


# Where the hit actually lands inside make_sub_drop -- start the cue this many
# seconds before the visual climax and the impact falls on the exact frame.
SUB_DROP_IMPACT = 1.25


def make_sub_drop(duration: float = 2.60, sample_rate: int = 44100) -> NDArray[np.float32]:
    """
    A riser into a sub-bass drop, for the moment the object clears the obstacle.

    The riser is the useful half: an impact with no approach is a thud, an
    impact with 1.25s of rising noise under it is an event.
    """
    n = max(1, int(duration * sample_rate))
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)

    rise_n = min(n, int(SUB_DROP_IMPACT * sample_rate))
    local = t[:rise_n] / max(SUB_DROP_IMPACT, 1e-6)

    rng = np.random.default_rng(23)
    noise = rng.standard_normal(rise_n).astype(np.float32)
    # Narrowing moving average = a filter sweeping upward as the riser builds.
    swept = np.empty(rise_n, dtype=np.float32)
    window = 1
    step = max(1, rise_n // 24)
    for start in range(0, rise_n, step):
        width = max(1, int(60 * (1.0 - local[min(start, rise_n - 1)]) ** 2) + 1)
        if width != window:
            window = width
        chunk = noise[start:start + step]
        if window > 1 and len(chunk) > 1:
            chunk = np.convolve(chunk, np.ones(window, dtype=np.float32) / window,
                                mode="same").astype(np.float32)
        swept[start:start + len(chunk)] = chunk
    out[:rise_n] += swept * (local ** 2.4) * 0.42

    sweep_hz = 180.0 * (1.0 + 9.0 * local ** 2.2)
    phase = 2 * np.pi * np.cumsum(sweep_hz) / sample_rate
    out[:rise_n] += (np.sin(phase) * (local ** 3) * 0.20).astype(np.float32)

    # The drop itself.
    idx = rise_n
    span = n - idx
    if span > 0:
        dl = np.arange(span, dtype=np.float32) / sample_rate
        glide = 74.0 * np.exp(-2.6 * dl) + 30.0
        dphase = 2 * np.pi * np.cumsum(glide) / sample_rate
        env = np.exp(-1.7 * dl).astype(np.float32)
        out[idx:] += (np.sin(dphase) * env * 0.95).astype(np.float32)
        click = min(span, int(0.010 * sample_rate))
        out[idx:idx + click] += np.linspace(0.55, 0.0, click, dtype=np.float32)

    return _stereo(out, out, peak=0.92)


SFX_DROP = "drop"
_SFX_BUILDERS[SFX_DROP] = make_sub_drop


_BED_BUILDERS = {
    "suspense": make_suspense_bgm,
    "phonk": make_phonk_bed,
}


def build_ducked_bed(
    duration: float,
    style: str = "phonk",
    narration_path: str | None = None,
    output_path: str | None = None,
    sample_rate: int = 44100,
    volume: float = BGM_DEFAULT_VOLUME,
    duck_depth: float = BGM_DUCK_DEPTH,
) -> str:
    """
    Renders a named bed at `volume`, ducked against the narration if there is one.

    `build_ducked_bgm` is the suspense-only shorthand this generalises; both
    stay because the commentary machine calls the old name in three places.
    """
    builder = _BED_BUILDERS.get(style, make_phonk_bed)
    bed = builder(duration, sample_rate)

    if narration_path and os.path.exists(narration_path):
        gain = speech_duck_curve(
            narration_path, duration, sample_rate=sample_rate, duck_depth=duck_depth,
        )
        span = min(len(gain), bed.shape[0])
        bed = bed[:span] * gain[:span, None]

    bed = np.clip(bed * float(volume), -1.0, 1.0).astype(np.float32)

    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    save_wav_to_file(bed, sample_rate, output_path)
    return output_path


def trim_leading_silence(
    audio_path: str,
    output_path: str | None = None,
    keep: float = 0.04,
    threshold: float = 0.02,
) -> dict[str, Any]:
    """
    Removes dead air from the front of a TTS take.

    Gemini TTS routinely returns 200-400ms of silence before the first word.
    On a 15-second short that is a wasted opening beat, and it puts the voice
    out of step with a visual cue that was placed at 0.0s.

    Returns {"path", "trimmed", "duration"}; `path` is the original file when
    there was nothing worth trimming.
    """
    from moviepy import AudioFileClip

    try:
        samples = _load_mono(audio_path, 16000)
    except Exception:
        return {"path": audio_path, "trimmed": 0.0, "duration": get_audio_duration(audio_path)}

    if samples.size == 0:
        return {"path": audio_path, "trimmed": 0.0, "duration": 0.0}

    loud = np.abs(samples) > float(threshold)
    if not loud.any():
        return {"path": audio_path, "trimmed": 0.0, "duration": len(samples) / 16000.0}

    onset = max(0.0, float(np.argmax(loud)) / 16000.0 - float(keep))
    if onset < 0.05:
        return {"path": audio_path, "trimmed": 0.0, "duration": len(samples) / 16000.0}

    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    clip = AudioFileClip(audio_path)
    try:
        trimmed = clip.subclipped(onset, clip.duration)
        trimmed.write_audiofile(output_path, fps=44100, logger=None)
        duration = float(trimmed.duration)
    finally:
        clip.close()

    return {"path": output_path, "trimmed": onset, "duration": duration}
