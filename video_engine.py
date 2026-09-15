# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false, reportUnknownLambdaType=false
"""
Video Generation Engine for AI Reels & Shorts
Handles image canvas fitting, Ken Burns motion effects, PIL typography overlays,
transitions, audio blending, and MP4 rendering using MoviePy.

Also hosts the Versus Duel engine: split-screen comparison rounds with animated
stat badges and a final winner reveal card.
"""

from __future__ import annotations

import os
import re
import math
import time
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import VideoClip, CompositeVideoClip, AudioFileClip, CompositeAudioClip, vfx

from paths import installed_families, resolve_font

RGB = tuple[int, int, int]
ProgressFn = Callable[[int, int, str], None]

# --- reframing ------------------------------------------------------------
# The sidebar "Image fit" options, in display order. Kept here so the label and
# the filter graph that implements it cannot drift apart.
FIT_MODES = {
    "blur_pad": "Blur fill",
    "crop_fill": "Crop",
    "color_pad": "Pad",
}
DEFAULT_FIT = "blur_pad"

# Letterbox colour for the 'Pad' fit mode, as an ffmpeg colour and as RGB.
PAD_COLOUR = "black"
PAD_RGB = (0, 0, 0)
# Background blur for 'Blur fill' -- ffmpeg boxblur luma:chroma radius.
BLUR_STRENGTH = "25:5"
# Reframe/boomerang intermediates written to the system temp directory, kept
# so they can be deleted once the render that needed them has finished.
_SCRATCH_RENDERS: list[str] = []


ASPECT_RATIOS = {
    "9:16 (Vertical Reels / Shorts / TikTok)": (1080, 1920),
    "1:1 (Square Instagram Post)": (1080, 1080),
    "16:9 (Landscape YouTube)": (1920, 1080),
    "4:5 (Portrait Feed Post)": (1080, 1350)
}


# ---------------------------------------------------------------------------
# Encoder selection
#
# The bundled ffmpeg carries NVENC, but "listed in -encoders" does not mean the
# driver will accept a frame, so the encoder is probed once at runtime and the
# result cached. Anything unexpected falls back to libx264.
#
# Measured on this project's 1080x1920 output (RTX 3060, 20s clip):
#   libx264 -preset medium -crf 20   6.18s   18.77 MB
#   h264_nvenc -preset p5 -cq 25     1.95s   18.77 MB   <- 3.2x, same size
# cq 25 was chosen because it lands on the same file size as the CPU encoder;
# lower cq is bigger, higher is smaller and softer.
# ---------------------------------------------------------------------------

# Every ffmpeg invocation gets a ceiling. A finishing pass that wedges -- a
# malformed subtitle file, a codec negotiation that never returns -- otherwise
# blocks its caller forever, and when the caller is a Streamlit script run that
# is a browser tab that never comes back and a user who cannot tell a hang from
# a slow render. Five minutes is far above the measured worst case for these
# passes (a 5-minute 1080x1920 burn-in runs about 40s on the CPU path).
FFMPEG_TIMEOUT = 300

GPU_CODEC = "h264_nvenc"
GPU_PRESET = "p5"
# Constant-quality target. 25 produced a 609 kb/s stream for 1080x1920 line
# art over glows and gradients -- which banded visibly before the platform had
# even re-encoded it. 21 lands the same content around 3-4 Mb/s.
GPU_CQ = "21"
CPU_CODEC = "libx264"
# veryfast, not medium. Measured on this project's 1080x1920 output at CRF 20:
# 2.18x faster and 12% smaller, because a faster preset spends fewer bits
# chasing the same quality target. On a machine with no usable NVENC -- which
# includes the Docker image, since there is no GPU in the container -- this is
# the difference between a render that finishes and one someone gives up on.
CPU_PRESET = "veryfast"
CPU_CRF = "18"

_encoder_cache: dict[str, Any] | None = None


def probe_gpu_encoder() -> bool:
    """Encodes one throwaway frame to check NVENC is genuinely usable."""
    import subprocess

    import imageio_ffmpeg

    try:
        proc = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
                "-c:v", GPU_CODEC, "-preset", GPU_PRESET, "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=45,
        )
        return proc.returncode == 0
    except Exception:
        return False


def video_encoder(force_cpu: bool = False) -> dict[str, Any]:
    """
    Returns the encoder settings to use, probing the GPU once per process.

    {"codec", "preset", "ffmpeg_params", "cli", "gpu"} -- `ffmpeg_params` suits
    MoviePy's write_videofile, `cli` suits a raw ffmpeg command line.
    """
    global _encoder_cache

    if force_cpu:
        return {
            "codec": CPU_CODEC, "preset": CPU_PRESET,
            "ffmpeg_params": ["-crf", CPU_CRF],
            "cli": ["-c:v", CPU_CODEC, "-preset", CPU_PRESET, "-crf", CPU_CRF],
            "gpu": False,
        }

    if _encoder_cache is None:
        if probe_gpu_encoder():
            _encoder_cache = {
                "codec": GPU_CODEC, "preset": GPU_PRESET,
                # -spatial-aq redistributes bits toward flat, low-detail
                # regions, which is precisely where this engine banded: a glow
                # falling off across a black background is a wide, smooth
                # gradient and the default rate control starves it.
                "ffmpeg_params": ["-rc", "vbr", "-cq", GPU_CQ, "-b:v", "0",
                                  "-spatial-aq", "1", "-aq-strength", "8"],
                "cli": ["-c:v", GPU_CODEC, "-preset", GPU_PRESET,
                        "-rc", "vbr", "-cq", GPU_CQ, "-b:v", "0",
                        "-spatial-aq", "1", "-aq-strength", "8"],
                "gpu": True,
            }
        else:
            _encoder_cache = {
                "codec": CPU_CODEC, "preset": CPU_PRESET,
                "ffmpeg_params": ["-crf", CPU_CRF],
                "cli": ["-c:v", CPU_CODEC, "-preset", CPU_PRESET, "-crf", CPU_CRF],
                "gpu": False,
            }
    return _encoder_cache


def write_clip(
    clip: Any,
    output_path: str,
    fps: int = 24,
    with_audio: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """
    Encodes a MoviePy clip, falling back to CPU if the GPU encoder gives up
    mid-render.

    `video_encoder()` probes NVENC with one 256x256 frame, which is a weak
    promise: the driver can still fail on a 1080x1920 stream once another
    process is holding the encoder session, and on consumer cards there is a
    hard cap on concurrent NVENC sessions. When that happens the failure lands
    at the very end of a long render, so it is caught here and re-encoded on
    libx264 -preset veryfast rather than thrown away.

    yuv420p is forced on the fallback path because it is the only pixel format
    every phone decoder accepts; libx264 will otherwise pick yuv444p for some
    inputs and the file plays as a green screen on iOS.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    enc = video_encoder()

    try:
        clip.write_videofile(
            output_path,
            fps=fps,
            codec=enc["codec"],
            audio_codec="aac" if with_audio else None,
            preset=enc["preset"],
            audio_fps=DELIVERY_SAMPLE_RATE if with_audio else None,
            ffmpeg_params=list(enc["ffmpeg_params"]) + [
                "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        return output_path
    except Exception as exc:
        if not enc["gpu"]:
            raise
        message = f"{type(exc).__name__}: {exc}"

    # One retry, on the CPU, and the probe result is poisoned so the rest of
    # the session does not walk into the same wall.
    global _encoder_cache
    _encoder_cache = video_encoder(force_cpu=True)
    if progress_callback:
        progress_callback(f"GPU encode failed ({message[:90]}) — re-encoding on the CPU.")
    print(f"Warning: {GPU_CODEC} failed, falling back to {CPU_CODEC}: {message}")

    clip.write_videofile(
        output_path,
        fps=fps,
        codec=CPU_CODEC,
        audio_codec="aac" if with_audio else None,
        preset="veryfast",
        audio_fps=DELIVERY_SAMPLE_RATE if with_audio else None,
        ffmpeg_params=["-crf", CPU_CRF, "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart"],
    )
    return output_path


# What every platform re-encodes audio to. Writing anything higher is size
# with no listener on the other end of it; the delivered file carried 96 kHz.
DELIVERY_SAMPLE_RATE = 48000


# ---------------------------------------------------------------------------
# Loudness
#
# A finished render measured -22.8 LUFS integrated with a true peak of
# -2.9 dBFS. Social platforms normalise to about -14, so everything this made
# played roughly nine decibels quieter than the video before it in the feed --
# which reads as amateur before a word is heard.
#
# Nine decibels of flat gain is not available: it would put the true peak at
# +5.9 dBFS and clip. Reaching the target needs the dynamic range brought in as
# well as the level brought up, which is what ffmpeg's loudnorm does. It is a
# separate pass over the finished file, with the video stream copied, so it
# costs an I/O-bound rewrite rather than a second encode.
# ---------------------------------------------------------------------------

# What the platforms normalise to. Anything quieter is turned up by the player,
# anything louder is turned down; matching it means the mix is heard as mixed.
TARGET_LUFS = -14.0

# Ceiling for the true peak, in dBFS. -1.5 leaves room for the intersample
# peaks a lossy re-encode introduces downstream.
TARGET_TRUE_PEAK_DB = -1.5

# What loudnorm's limiter is actually asked for, which has to sit below the
# ceiling. Measured: requesting exactly -1.5 produced a finished file at
# -1.2 dBFS, because the limiter runs before the AAC encode and the encode adds
# its own intersample peaks. Half a decibel of headroom covers that.
_TRUE_PEAK_HEADROOM_DB = 0.5

# Loudness range. 11 LU keeps the sub-bass drop feeling like a drop instead of
# flattening it into the bed.
TARGET_LRA = 11.0


def measure_loudness(path: str, ffmpeg: str = "") -> dict[str, float]:
    """
    Integrated loudness and true peak of a file, via ffmpeg's ebur128.

    Returns {"lufs", "true_peak", "lra"}; any value that could not be parsed
    comes back as 0.0, which the caller must read as unknown.
    """
    import re
    import subprocess

    import imageio_ffmpeg

    ffmpeg = ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostats", "-i", os.path.abspath(path),
             "-af", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    except Exception:
        return {"lufs": 0.0, "true_peak": 0.0, "lra": 0.0}

    text = proc.stderr or ""
    # The summary block at the end is the integrated figure; the per-frame
    # lines above it are not.
    tail = text[text.rfind("Integrated loudness"):] if "Integrated loudness" in text else ""

    def grab(pattern: str, source: str) -> float:
        found = re.findall(pattern, source)
        try:
            return float(found[-1])
        except (IndexError, TypeError, ValueError):
            return 0.0

    return {
        "lufs": grab(r"I:\s*(-?\d+\.?\d*)\s*LUFS", tail),
        "true_peak": grab(r"Peak:\s*(-?\d+\.?\d*)\s*dBFS", tail),
        "lra": grab(r"LRA:\s*(-?\d+\.?\d*)\s*LU", tail),
    }


def _loudnorm_measure(path: str, target_lufs: float, true_peak_db: float,
                      lra: float, ffmpeg: str) -> dict[str, str]:
    """
    loudnorm's own analysis pass, as the JSON it prints to stderr.

    Returns {} when anything about the parse fails, which makes the caller fall
    back to the single-pass filter -- less accurate, but still an improvement
    on leaving the mix nine decibels quiet.
    """
    import json
    import subprocess

    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostats", "-i", os.path.abspath(path),
             "-af", (f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA={lra}"
                     ":print_format=json"),
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    except Exception:
        return {}

    text = proc.stderr or ""
    start = text.rfind("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}

    try:
        parsed = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return {}

    wanted = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    if not all(key in parsed for key in wanted):
        return {}
    # Refuse the degenerate analysis of a silent track: -inf breaks the filter.
    if any("inf" in str(parsed[key]).lower() for key in wanted):
        return {}
    return {key: str(parsed[key]) for key in wanted}

def normalise_loudness(path: str, target_lufs: float = TARGET_LUFS,
                       true_peak_db: float = TARGET_TRUE_PEAK_DB,
                       lra: float = TARGET_LRA,
                       progress_callback: Callable[[str], None] | None = None,
                       ffmpeg: str = "") -> dict[str, Any]:
    """
    Brings a finished render up to the platform loudness target, in place.

    The video stream is copied, so only the audio is re-encoded. Returns
    {"before", "after", "applied"} -- and `applied` is False with the file
    untouched whenever the measurement or the pass fails, because a video that
    is too quiet is a worse video and a video that is gone is a lost render.
    """
    import os as _os
    import shutil
    import subprocess
    import tempfile as _tempfile

    import imageio_ffmpeg

    ffmpeg = ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()
    source = _os.path.abspath(path)
    before = measure_loudness(source, ffmpeg)

    if not before["lufs"]:
        if progress_callback:
            progress_callback("Could not measure loudness; leaving the mix alone.")
        return {"before": before, "after": before, "applied": False}

    if progress_callback:
        progress_callback(f"Loudness {before['lufs']:.1f} LUFS -> "
                          f"{target_lufs:.0f} LUFS...")

    # The limiter is asked for a little below the ceiling; see the constant.
    limiter_tp = float(true_peak_db) - _TRUE_PEAK_HEADROOM_DB

    work = _tempfile.mkdtemp(prefix="rf_loud_")
    staged = _os.path.join(work, "normalised.mp4")
    try:
        # Two passes, not one. Single-pass loudnorm works from a running
        # estimate and lands near the target rather than on it: measured, it
        # produced -15.6 LUFS against a -14 request and a true peak of
        # -0.8 dBFS against a -1.5 ceiling -- over the limit it was given. The
        # first pass measures the file, the second applies the correction with
        # those figures supplied, which is what makes the limiter exact.
        stats = _loudnorm_measure(source, target_lufs, limiter_tp, lra, ffmpeg)
        measured = ""
        if stats:
            measured = (f":measured_I={stats['input_i']}"
                        f":measured_TP={stats['input_tp']}"
                        f":measured_LRA={stats['input_lra']}"
                        f":measured_thresh={stats['input_thresh']}"
                        f":offset={stats['target_offset']}:linear=true")

        proc = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", source,
             "-af", (f"loudnorm=I={target_lufs}:TP={limiter_tp}:LRA={lra}"
                     f"{measured}:print_format=summary"),
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             # The delivery rate has to be set HERE, not on the encoder that
             # wrote the file: this pass runs after it and re-encodes the
             # audio. loudnorm works internally at 192 kHz, so with no -ar the
             # stream inherits that and AAC ships it at 96 -- which is what the
             # delivered file carried even after write_clip started asking for
             # 48.
             "-ar", str(DELIVERY_SAMPLE_RATE),
             # AAC frames are 1024 samples, so the re-encode pads the tail and
             # the audio comes out up to ~0.1s longer than the picture. This
             # project holds A/V sync at 0.000s across every mode, and -shortest
             # is what keeps the loudness pass from being the thing that breaks
             # it: the padding is trimmed rather than muxed in.
             "-shortest",
             "-movflags", "+faststart", staged],
            capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)

        if proc.returncode != 0 or not _os.path.exists(staged):
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-6:])
            if progress_callback:
                progress_callback(f"Loudness pass failed, keeping the original. {tail[:120]}")
            return {"before": before, "after": before, "applied": False}

        shutil.move(staged, source)
    except Exception as exc:                                  # noqa: BLE001
        if progress_callback:
            progress_callback(f"Loudness pass failed ({type(exc).__name__}); "
                              "keeping the original.")
        return {"before": before, "after": before, "applied": False}
    finally:
        shutil.rmtree(work, ignore_errors=True)

    after = measure_loudness(source, ffmpeg)
    return {"before": before, "after": after, "applied": True}


# ---------------------------------------------------------------------------
# Gemini File API pre-flight
#
# The File API rejects a clip by accepting the upload and then parking it in
# FAILED, which surfaces as "Gemini could not process this clip". Every case
# seen here came from the container, not the content: HEVC/H.265 from an
# iPhone, VP9 from a YouTube download, variable frame rate from a screen
# recorder, a rotation matrix in the metadata, or an HDR (bt2020/arib-std-b67)
# colour profile.
#
# Rather than guess which of those the API will tolerate, every ingested clip
# is normalised to the one profile that has never been refused: H.264 High,
# yuv420p, constant 30fps, AAC-LC 128k at 44.1kHz, no side data.
# ---------------------------------------------------------------------------

GEMINI_SAFE_FPS = 30
GEMINI_SAFE_HEIGHT = 720          # the model reads it at low resolution anyway
GEMINI_SAFE_AUDIO_RATE = 44100
GEMINI_SAFE_AUDIO_BITRATE = "128k"


_VIDEO_STREAM_RE = re.compile(
    r"Stream #\d+:\d+.*?: Video: (?P<codec>[\w.]+)[^,]*,\s*"
    r"(?P<pix>[\w]+)(?:\((?P<colour>[^)]*)\))?", re.IGNORECASE)
_AUDIO_STREAM_RE = re.compile(
    r"Stream #\d+:\d+.*?: Audio: (?P<codec>[\w.]+)[^,]*,\s*(?P<rate>\d+) Hz", re.IGNORECASE)
_SIZE_RE = re.compile(r",\s*(?P<w>\d{2,5})x(?P<h>\d{2,5})[\s,\[]")
_FPS_RE = re.compile(r",\s*(?P<fps>[\d.]+) fps")
_ROTATE_RE = re.compile(r"(displaymatrix:\s*rotation|rotate\s*:)", re.IGNORECASE)
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)")

# Rates a constant-frame-rate file actually reports. A source that lands
# somewhere else (29.83, 47.95) was assembled from variable timestamps.
_CFR_RATES = (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0)


def probe_stream_info(video_path: str) -> dict[str, Any]:
    """
    Reads codec, pixel format, colour transfer and frame rate out of a clip.

    ffprobe would be the obvious tool and is used when it is on PATH, but
    imageio-ffmpeg ships ffmpeg *without* ffprobe, so on a default install of
    this project there is none. The fallback parses `ffmpeg -i`, whose stream
    lines carry everything needed:

        Stream #0:0: Video: hevc (Main), yuv420p10le(tv, bt2020nc/bt2020/smpte2084), 1080x1920, 30 fps
        Stream #0:1: Audio: aac (LC), 44100 Hz, stereo, fltp, 128 kb/s

    Returns {} only if ffmpeg itself could not be run.
    """
    import json
    import shutil
    import subprocess

    import imageio_ffmpeg

    path = os.path.abspath(video_path)

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        guess = os.path.join(os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()), "ffprobe.exe")
        ffprobe = guess if os.path.exists(guess) else None

    if ffprobe:
        try:
            proc = subprocess.run(
                [ffprobe, "-v", "error", "-print_format", "json",
                 "-show_streams", "-show_format", path],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0:
                return _info_from_ffprobe(json.loads(proc.stdout or "{}"))
        except Exception:
            pass

    # `ffmpeg -i` with no output file exits 1 and writes the stream table to
    # stderr. That exit code is expected, not an error.
    try:
        proc = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", path],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return {}

    return _info_from_ffmpeg_stderr(proc.stderr or "")


def _info_from_ffprobe(data: dict[str, Any]) -> dict[str, Any]:
    """Normalises ffprobe's JSON into the shape needs_gemini_normalise wants."""
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})

    def _rate(value: Any) -> float:
        try:
            num, _, den = str(value).partition("/")
            return float(num) / float(den or 1)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0

    return {
        "codec": str(video.get("codec_name") or ""),
        "pix_fmt": str(video.get("pix_fmt") or ""),
        "transfer": str(video.get("color_transfer") or ""),
        "primaries": str(video.get("color_primaries") or ""),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        # r_frame_rate is the container's nominal rate; avg_frame_rate is what
        # the frames actually came out at. They diverge on VFR sources, which
        # is exactly the case that has to be caught.
        "fps": _rate(video.get("r_frame_rate", "0/1")),
        "avg_fps": _rate(video.get("avg_frame_rate", "0/1")),
        "rotated": bool(video.get("side_data_list")),
        "audio_codec": str(audio.get("codec_name") or ""),
        "audio_rate": int(audio.get("sample_rate") or 0),
        "has_audio": bool(audio),
        "duration": float(data.get("format", {}).get("duration") or 0.0),
        "source": "ffprobe",
    }


def _info_from_ffmpeg_stderr(text: str) -> dict[str, Any]:
    """Parses the stream table `ffmpeg -i` prints when given no output file."""
    video_line = ""
    audio_line = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not video_line and ": Video:" in stripped:
            video_line = stripped
        elif not audio_line and ": Audio:" in stripped:
            audio_line = stripped

    if not video_line:
        return {}

    vm = _VIDEO_STREAM_RE.search(video_line)
    am = _AUDIO_STREAM_RE.search(audio_line) if audio_line else None
    size = _SIZE_RE.search(video_line)
    fps_m = _FPS_RE.search(video_line)

    # "yuv420p10le(tv, bt2020nc/bt2020/smpte2084)" -- the transfer is last.
    colour = (vm.group("colour") or "") if vm else ""
    transfer = ""
    for chunk in colour.replace(",", "/").split("/"):
        token = chunk.strip()
        if token in _GEMINI_HDR_TRANSFERS or token in ("bt709", "bt470bg", "smpte170m"):
            transfer = token

    duration = 0.0
    dm = _DURATION_RE.search(text)
    if dm:
        duration = int(dm.group(1)) * 3600 + int(dm.group(2)) * 60 + float(dm.group(3))

    fps = float(fps_m.group("fps")) if fps_m else 0.0
    return {
        "codec": (vm.group("codec") if vm else "").lower(),
        "pix_fmt": (vm.group("pix") if vm else "").lower(),
        "transfer": transfer,
        "primaries": "bt2020" if "bt2020" in colour else "",
        "width": int(size.group("w")) if size else 0,
        "height": int(size.group("h")) if size else 0,
        "fps": fps,
        # `ffmpeg -i` reports one averaged rate, so there is no second number
        # to compare against. `needs_gemini_normalise` falls back to checking
        # that the rate is one a CFR encoder would actually produce.
        "avg_fps": fps,
        "rotated": bool(_ROTATE_RE.search(text)),
        "audio_codec": (am.group("codec") if am else "").lower(),
        "audio_rate": int(am.group("rate")) if am else 0,
        "has_audio": bool(audio_line),
        "duration": duration,
        "source": "ffmpeg",
    }


# Everything outside this set has been observed to fail, be re-wrapped, or be
# silently degraded by the File API.
_GEMINI_OK_CODECS = frozenset({"h264"})
_GEMINI_OK_PIX_FMTS = frozenset({"yuv420p", "yuvj420p"})
_GEMINI_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})


def needs_gemini_normalise(info: dict[str, Any]) -> tuple[bool, str]:
    """
    Decides whether a clip has to be transcoded, and says why.

    An empty `info` (no ffprobe) means transcode: the cost of a needless
    ~4s transcode is far below the cost of a failed upload after a 40MB push.
    """
    if not info:
        return True, "could not probe the container"

    if info.get("codec") not in _GEMINI_OK_CODECS:
        return True, f"{info.get('codec') or 'unknown'} video (needs H.264)"
    if info.get("pix_fmt") not in _GEMINI_OK_PIX_FMTS:
        return True, f"{info.get('pix_fmt') or 'unknown'} pixels (needs yuv420p)"
    if info.get("transfer") in _GEMINI_HDR_TRANSFERS:
        return True, f"HDR transfer ({info.get('transfer')})"
    if info.get("rotated"):
        return True, "a rotation matrix in the metadata"

    fps, avg = float(info.get("fps") or 0.0), float(info.get("avg_fps") or 0.0)
    if fps <= 0:
        return True, "no declared frame rate"
    if avg > 0 and abs(fps - avg) > 0.75:
        return True, f"variable frame rate ({avg:.1f} avg vs {fps:.1f} nominal)"
    if fps > GEMINI_SAFE_FPS + 1:
        return True, f"{fps:.0f}fps (needs {GEMINI_SAFE_FPS} CFR)"
    # Without ffprobe there is only one rate to look at, so VFR is inferred
    # from the rate being one no CFR encoder emits -- a screen recording that
    # averages 29.83fps is variable whatever the container claims.
    if not any(abs(fps - rate) < 0.05 for rate in _CFR_RATES):
        return True, f"{fps:.2f}fps is not a constant rate"

    if info.get("has_audio") and info.get("audio_codec") not in ("aac",):
        return True, f"{info.get('audio_codec')} audio (needs AAC)"

    return False, "already a baseline H.264 MP4"


def normalize_for_gemini(
    video_path: str,
    dest_dir: str | None = None,
    progress: Callable[[str], None] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Returns a clip the Gemini File API will accept, transcoding only if needed.

    {"path", "transcoded", "reason", "info"}. `path` is the original file when
    no work was required, so nothing is copied for a clip that is already fine.

    The output is deliberately small: 720p is above what the model samples, and
    a 40MB TikTok rip becomes ~6MB, which is most of the upload wait.
    """
    import subprocess

    import imageio_ffmpeg

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    info = probe_stream_info(video_path)
    needed, reason = needs_gemini_normalise(info)
    if not needed and not force:
        return {"path": video_path, "transcoded": False, "reason": reason, "info": info}

    dest_dir = dest_dir or os.path.dirname(os.path.abspath(video_path))
    os.makedirs(dest_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(dest_dir, f"{stem}_gemini.mp4")

    if progress:
        progress(f"Normalising for Gemini — {reason}...")

    # scale keeps the aspect and forces even dimensions: H.264 cannot encode an
    # odd height in yuv420p and ffmpeg fails outright rather than rounding.
    vf = (f"scale=-2:'min({GEMINI_SAFE_HEIGHT},ih)':flags=bicubic,"
          f"format=yuv420p,fps={GEMINI_SAFE_FPS}")

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        # ffmpeg auto-rotates by default, so a phone clip's rotation matrix is
        # baked into the pixels here and the re-encode emits none of its own.
        "-i", os.path.abspath(video_path),
        "-map_metadata", "-1", "-map_chapters", "-1",
        "-vf", vf,
        "-c:v", "libx264", "-profile:v", "high", "-level", "4.0",
        "-preset", "veryfast", "-crf", "24",
        # Constant frame rate, stated three ways: the filter above sets the
        # rate, -vsync cfr stops ffmpeg dropping or duplicating around it, and
        # -r pins the container. A VFR screen recording needs all three.
        "-vsync", "cfr", "-r", str(GEMINI_SAFE_FPS),
        "-g", str(GEMINI_SAFE_FPS * 2), "-pix_fmt", "yuv420p",
        # Strip HDR / non-standard primaries by asserting plain bt709.
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart",
    ]
    if info.get("has_audio", True):
        command += ["-c:a", "aac", "-b:a", GEMINI_SAFE_AUDIO_BITRATE,
                    "-ar", str(GEMINI_SAFE_AUDIO_RATE), "-ac", "2"]
    else:
        command += ["-an"]
    command.append(out_path)

    proc = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(
            "Could not normalise the clip for Gemini:\n"
            + (proc.stderr or "ffmpeg gave no reason").strip()[-700:]
        )

    _SCRATCH_RENDERS.append(out_path)
    if progress:
        before = os.path.getsize(video_path) / 1_048_576
        after = os.path.getsize(out_path) / 1_048_576
        progress(f"Normalised: {before:.1f} MB → {after:.1f} MB, H.264 / {GEMINI_SAFE_FPS} CFR.")

    return {"path": out_path, "transcoded": True, "reason": reason,
            "info": info, "output_info": probe_stream_info(out_path)}


def fit_image_to_aspect(
    image_input: Image.Image | str,
    target_size: tuple[int, int] = (1080, 1920),
    fit_mode: str = "blur_pad",
) -> Image.Image:
    """
    Fits an input image (PIL Image or file path) into target canvas dimensions.
    fit_mode options:
      - 'blur_pad': Scales image to fit, padded with heavily blurred background of the same image.
      - 'crop_fill': Scales and crops image to fill target canvas completely.
      - 'color_pad': Scales image to fit inside target canvas with dark slate background.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")

    target_w, target_h = target_size
    img_w, img_h = img.size
    img_aspect = img_w / img_h
    target_aspect = target_w / target_h

    canvas = Image.new("RGB", (target_w, target_h), (15, 23, 42))

    if fit_mode == "crop_fill":
        if img_aspect > target_aspect:
            scale = target_h / img_h
            scaled_w = int(img_w * scale)
            scaled_h = target_h
            scaled_img = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
            x_crop = (scaled_w - target_w) // 2
            canvas = scaled_img.crop((x_crop, 0, x_crop + target_w, target_h))
        else:
            scale = target_w / img_w
            scaled_w = target_w
            scaled_h = int(img_h * scale)
            scaled_img = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
            y_crop = (scaled_h - target_h) // 2
            canvas = scaled_img.crop((0, y_crop, target_w, y_crop + target_h))

    elif fit_mode == "blur_pad":
        bg = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(35))
        darkener = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        bg = Image.blend(bg, darkener, 0.25)

        if img_aspect > target_aspect:
            fg_w = target_w
            fg_h = int(target_w / img_aspect)
        else:
            fg_h = target_h
            fg_w = int(target_h * img_aspect)

        fg = img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)
        pos_x = (target_w - fg_w) // 2
        pos_y = (target_h - fg_h) // 2
        bg.paste(fg, (pos_x, pos_y))
        canvas = bg

    else:  # 'color_pad'
        if img_aspect > target_aspect:
            fg_w = target_w
            fg_h = int(target_w / img_aspect)
        else:
            fg_h = target_h
            fg_w = int(target_h * img_aspect)

        fg = img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)
        pos_x = (target_w - fg_w) // 2
        pos_y = (target_h - fg_h) // 2
        canvas.paste(fg, (pos_x, pos_y))

    return canvas


# Heavy display faces, in preference order, for viral-style subtitles.
_BOLD_FONT_CANDIDATES = [
    "Montserrat-ExtraBold.ttf",   # assets/fonts, if the operator supplied it
    "seguibl.ttf",                # Segoe UI Black
    "arialbd.ttf",                # Arial Bold
    "impact.ttf",                 # Impact
    "verdanab.ttf",               # Verdana Bold
    "segoeuib.ttf",               # Segoe UI Bold
    # What a Debian container has once fonts-dejavu-core / fonts-liberation /
    # fonts-freefont-ttf are installed. Without these the caption renderer
    # silently drops to an 11px bitmap face on a Linux server.
    "DejaVuSans-Bold.ttf",
    "LiberationSans-Bold.ttf",
    "FreeSansBold.ttf",
]

VIRAL_HIGHLIGHT_COLOR = (255, 214, 10)   # bright feed-stopping yellow

# Short function words never worth highlighting.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "is", "are", "was", "were", "be", "been", "it", "its", "this",
    "that", "with", "as", "by", "from", "you", "your", "we", "our", "they",
    "them", "he", "she", "his", "her", "not", "can", "will", "just", "so",
    "up", "out", "how", "why", "what", "when", "who", "all", "one", "do",
}


def _load_bold_font(size: int) -> Any:
    """Loads the heaviest available display font at `size`, with a safe fallback."""
    path = resolve_font(_BOLD_FONT_CANDIDATES)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _auto_highlight_words(words):
    """
    Picks the words worth painting yellow when the caller has not chosen any.

    Prioritizes numbers and figures (which drive retention on listicle reels),
    then any word the author already SHOUTED, then the longest content words.
    """
    scored = []
    for i, raw in enumerate(words):
        token = raw.strip(".,!?:;\"'()[]—-").lower()
        if not token or token in _STOPWORDS:
            continue
        if any(ch.isdigit() for ch in token) or "%" in raw or "$" in raw:
            scored.append((0, -len(token), i))
        elif raw.isupper() and len(token) > 1:
            scored.append((1, -len(token), i))
        elif len(token) >= 6:
            scored.append((2, -len(token), i))

    scored.sort()
    # Cap the highlights: everything emphasized means nothing emphasized.
    keep = max(1, min(3, len(words) // 3)) if scored else 0
    return {idx for _, _, idx in scored[:keep]}


def draw_viral_caption(
    img: Image.Image,
    text: str,
    position: str = "center",
    highlight_words: Sequence[str] | None = None,
    accent_color: RGB = VIRAL_HIGHLIGHT_COLOR,
    font_scale: float = 0.078,
    stroke_ratio: float = 0.14,
) -> Image.Image:
    """
    Renders bold, high-contrast viral subtitles tuned for muted mobile feeds:
    large centered uppercase text, white fill, thick black stroke, and bright
    yellow keyword highlights.

    `highlight_words` may be an explicit iterable of words to paint in the accent
    color; when omitted, keywords are chosen automatically (numbers first).
    """
    if not text or not text.strip():
        return img

    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(28, int(w * font_scale))
    font = _load_bold_font(font_size)
    stroke_w = max(3, int(font_size * stroke_ratio))

    display_text = text.strip().upper()
    words = display_text.split()

    if highlight_words:
        wanted = {str(x).strip().strip(".,!?:;").upper() for x in highlight_words}
        highlight_idx = {
            i for i, word in enumerate(words)
            if word.strip(".,!?:;\"'()[]—-") in wanted
        }
    else:
        highlight_idx = _auto_highlight_words(words)

    # Wrap into lines, tracking each word's index so highlights survive wrapping.
    max_text_width = int(w * 0.86)
    space_w = draw.textlength(" ", font=font)

    lines = []
    current = []
    current_w = 0.0
    for i, word in enumerate(words):
        word_w = draw.textlength(word, font=font)
        addition = word_w if not current else space_w + word_w
        if current and current_w + addition > max_text_width:
            lines.append(current)
            current = [(i, word, word_w)]
            current_w = word_w
        else:
            current.append((i, word, word_w))
            current_w += addition
    if current:
        lines.append(current)

    line_h = int(font_size * 1.22)
    total_h = line_h * len(lines)

    if position == "top":
        start_y = int(h * 0.12)
    elif position == "bottom":
        start_y = int(h * 0.78) - total_h
    else:  # center
        start_y = (h - total_h) // 2

    curr_y = start_y
    for line in lines:
        line_w = sum(word_w for _, _, word_w in line) + space_w * (len(line) - 1)
        curr_x = (w - line_w) / 2.0

        for i, word, word_w in line:
            fill = accent_color if i in highlight_idx else (255, 255, 255)
            draw.text(
                (curr_x, curr_y),
                word,
                font=font,
                fill=fill + (255,),
                stroke_width=stroke_w,
                stroke_fill=(0, 0, 0, 255)
            )
            curr_x += word_w + space_w

        curr_y += line_h

    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    return out.convert("RGB")


def draw_caption_overlay(
    img: Image.Image,
    text: str,
    position: str = "bottom",
    style: str = "glass",
    accent_color: RGB = (255, 105, 180),
    max_font_size: int = 42,
    highlight_words: Sequence[str] | None = None,
) -> Image.Image:
    """
    Renders styled typography / caption overlay card on PIL Image canvas.
    Styles: 'viral', 'glass', 'dark', 'neon', 'minimal'
    Positions: 'bottom', 'center', 'top'
    """
    if not text or not text.strip():
        return img

    if style == "viral":
        return draw_viral_caption(
            img,
            text,
            position=position,
            highlight_words=highlight_words
        )

    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(22, min(max_font_size, int(w * 0.042)))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    max_text_width = int(w * 0.82)
    words = text.split()
    lines = []
    curr_line = []

    for word in words:
        test_str = " ".join(curr_line + [word])
        bbox = draw.textbbox((0, 0), test_str, font=font)
        if (bbox[2] - bbox[0]) <= max_text_width:
            curr_line.append(word)
        else:
            if curr_line:
                lines.append(" ".join(curr_line))
            curr_line = [word]
    if curr_line:
        lines.append(" ".join(curr_line))

    line_widths = []
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    total_text_h = sum(line_heights) + (len(lines) - 1) * 8
    max_line_w = max(line_widths) if line_widths else 0

    pad_x = 32
    pad_y = 20
    box_w = max_line_w + pad_x * 2
    box_h = total_text_h + pad_y * 2

    box_x = (w - box_w) // 2
    if position == "top":
        box_y = int(h * 0.10)
    elif position == "center":
        box_y = (h - box_h) // 2
    else:  # bottom
        box_y = int(h * 0.80) - box_h // 2

    rect = [box_x, box_y, box_x + box_w, box_y + box_h]

    if style == "glass":
        draw.rounded_rectangle(rect, radius=20, fill=(15, 23, 42, 210), outline=(255, 255, 255, 70), width=2)
        draw.rounded_rectangle([box_x + 10, box_y + 12, box_x + 16, box_y + box_h - 12], radius=3, fill=accent_color + (255,))
        text_start_x = box_x + 32
    elif style == "neon":
        draw.rounded_rectangle(rect, radius=16, fill=(5, 5, 10, 235), outline=accent_color + (255,), width=3)
        text_start_x = box_x + pad_x
    elif style == "dark":
        draw.rounded_rectangle(rect, radius=16, fill=(0, 0, 0, 220))
        text_start_x = box_x + pad_x
    else:  # minimal
        draw.rounded_rectangle(rect, radius=16, fill=(255, 255, 255, 235), outline=(220, 220, 220, 200), width=2)
        text_start_x = box_x + pad_x

    curr_y = box_y + pad_y
    for i, line in enumerate(lines):
        text_color = (15, 23, 42) if style == "minimal" else (255, 255, 255)
        draw.text((text_start_x, curr_y), line, fill=text_color, font=font)
        curr_y += line_heights[i] + 8

    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    return out.convert("RGB")


def make_watermark_tile(size, watermark_text, position="top_right"):
    """
    Builds the watermark badge as a small RGBA tile plus its paste coordinates.

    Returned separately from the slide art so the badge can be composited
    *after* the Ken Burns transform -- baking it into the source image lets the
    motion crop push the branding off-frame.

    Returns (tile, (x, y)) or None when there is nothing to draw.
    """
    if not watermark_text or not watermark_text.strip():
        return None

    w, h = size
    font_size = max(16, int(w * 0.026))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = measure.textbbox((0, 0), watermark_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # textbbox reports floats; Image.new needs integer dimensions.
    pad_x, pad_y = 16, 10
    card_w = int(tw + pad_x * 2)
    card_h = int(th + pad_y * 2)

    if position == "top_left":
        cx, cy = 30, 40
    elif position == "bottom_left":
        cx, cy = 30, h - card_h - 40
    elif position == "bottom_right":
        cx, cy = w - card_w - 30, h - card_h - 40
    else:  # top_right default
        cx, cy = w - card_w - 30, 40

    tile = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(tile)
    tdraw.rounded_rectangle(
        [0, 0, card_w - 1, card_h - 1],
        radius=12, fill=(0, 0, 0, 160), outline=(255, 255, 255, 50), width=1
    )
    tdraw.text((pad_x, pad_y), watermark_text, fill=(255, 255, 255, 230), font=font)

    return tile, (int(cx), int(cy))


def draw_watermark(img, watermark_text, position="top_right"):
    """Draws brand watermark text badge on top corner."""
    if not watermark_text or not watermark_text.strip():
        return img

    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(16, int(w * 0.026))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), watermark_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # textbbox reports floats; Image.new needs integer dimensions.
    pad_x, pad_y = 16, 10
    card_w = int(tw + pad_x * 2)
    card_h = int(th + pad_y * 2)

    if position == "top_left":
        cx = 30
        cy = 40
    elif position == "bottom_left":
        cx = 30
        cy = h - card_h - 40
    else:  # top_right default
        cx = w - card_w - 30
        cy = 40

    draw.rounded_rectangle([cx, cy, cx + card_w, cy + card_h], radius=12, fill=(0, 0, 0, 160), outline=(255, 255, 255, 50), width=1)
    draw.text((cx + pad_x, cy + pad_y), watermark_text, fill=(255, 255, 255, 230), font=font)

    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    return out.convert("RGB")

def create_slide_clip(
    base_image: Image.Image,
    duration: float = 4.0,
    motion_type: str = "zoom_in",
    caption_text: str = "",
    caption_pos: str = "bottom",
    caption_style: str = "glass",
    watermark_text: str = "",
    accent_color: RGB = (255, 105, 180),
    highlight_words: Sequence[str] | None = None,
) -> VideoClip:
    """
    Creates a VideoClip for a single slide with Ken Burns motion effect and typography overlays.
    """
    processed_img = base_image.copy()
    if caption_text:
        processed_img = draw_caption_overlay(
            processed_img,
            caption_text,
            position=caption_pos,
            style=caption_style,
            accent_color=accent_color,
            highlight_words=highlight_words
        )
    w, h = processed_img.size

    # Composited per-frame after the Ken Burns crop so the branding stays
    # pinned to the frame edge instead of being cropped away by the motion.
    watermark = make_watermark_tile((w, h), watermark_text)

    def make_frame(t):
        if duration <= 0:
            progress = 0.0
        else:
            progress = min(1.0, max(0.0, t / duration))

        if motion_type == "zoom_in":
            scale = 1.0 + 0.12 * progress
            crop_w = int(w / scale)
            crop_h = int(h / scale)
            x1 = (w - crop_w) // 2
            y1 = (h - crop_h) // 2
            frame_img = processed_img.crop((x1, y1, x1 + crop_w, y1 + crop_h)).resize((w, h), Image.Resampling.BILINEAR)

        elif motion_type == "zoom_out":
            scale = 1.12 - 0.12 * progress
            crop_w = int(w / scale)
            crop_h = int(h / scale)
            x1 = (w - crop_w) // 2
            y1 = (h - crop_h) // 2
            frame_img = processed_img.crop((x1, y1, x1 + crop_w, y1 + crop_h)).resize((w, h), Image.Resampling.BILINEAR)

        elif motion_type == "pan_right":
            scale = 1.10
            crop_w = int(w / scale)
            crop_h = int(h / scale)
            max_x = w - crop_w
            x1 = int(max_x * progress)
            y1 = (h - crop_h) // 2
            frame_img = processed_img.crop((x1, y1, x1 + crop_w, y1 + crop_h)).resize((w, h), Image.Resampling.BILINEAR)

        elif motion_type == "pan_left":
            scale = 1.10
            crop_w = int(w / scale)
            crop_h = int(h / scale)
            max_x = w - crop_w
            x1 = int(max_x * (1.0 - progress))
            y1 = (h - crop_h) // 2
            frame_img = processed_img.crop((x1, y1, x1 + crop_w, y1 + crop_h)).resize((w, h), Image.Resampling.BILINEAR)

        elif motion_type == "breathe":
            scale = 1.0 + 0.08 * math.sin(math.pi * progress)
            crop_w = int(w / scale)
            crop_h = int(h / scale)
            x1 = (w - crop_w) // 2
            y1 = (h - crop_h) // 2
            frame_img = processed_img.crop((x1, y1, x1 + crop_w, y1 + crop_h)).resize((w, h), Image.Resampling.BILINEAR)

        else:  # static
            frame_img = processed_img.copy()

        if watermark is not None:
            tile, pos = watermark
            frame_img.paste(tile, pos, tile)

        return np.array(frame_img)

    clip = VideoClip(make_frame, duration=duration)
    return clip


# ---------------------------------------------------------------------------
# Faceless Commentary Engine
#
# Takes a raw viral clip plus an edge-tts narration track and produces a
# vertical short: the clip reframed to 9:16, its own audio ducked under the
# narration, and burned-in captions timed to the spoken script.
# ---------------------------------------------------------------------------


def caption_overlay_rgba(
    size: tuple[int, int],
    text: str,
    position: str = "center",
    font_scale: float = 0.075,
) -> Image.Image:
    """
    Renders caption text onto a fully transparent RGBA canvas.

    draw_viral_caption() flattens onto an opaque base, which is right for a
    still slide but useless over live video -- this returns just the glyphs so
    they can be alpha-composited onto each frame.
    """
    w, h = size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if not text or not text.strip():
        return overlay

    draw = ImageDraw.Draw(overlay)
    font_size = max(26, int(w * font_scale))
    font = _load_bold_font(font_size)
    stroke_w = max(3, int(font_size * 0.14))

    words = text.strip().upper().split()
    highlight = _auto_highlight_words(words)

    space_w = draw.textlength(" ", font=font)
    max_w = int(w * 0.86)

    lines: list[list[tuple[int, str, float]]] = []
    cur: list[tuple[int, str, float]] = []
    cur_w = 0.0
    for i, word in enumerate(words):
        ww = draw.textlength(word, font=font)
        add = ww if not cur else space_w + ww
        if cur and cur_w + add > max_w:
            lines.append(cur)
            cur, cur_w = [(i, word, ww)], ww
        else:
            cur.append((i, word, ww))
            cur_w += add
    if cur:
        lines.append(cur)

    line_h = int(font_size * 1.24)
    total_h = line_h * len(lines)
    if position == "top":
        y = int(h * 0.12)
    elif position == "bottom":
        y = int(h * 0.80) - total_h
    else:
        y = (h - total_h) // 2

    for line in lines:
        line_w = sum(ww for _, _, ww in line) + space_w * (len(line) - 1)
        x = (w - line_w) / 2.0
        for i, word, ww in line:
            fill = VIRAL_HIGHLIGHT_COLOR if i in highlight else (255, 255, 255)
            draw.text((x, y), word, font=font, fill=fill + (255,),
                      stroke_width=stroke_w, stroke_fill=(0, 0, 0, 255))
            x += ww + space_w
        y += line_h

    return overlay


def split_caption_chunks(script: str, words_per_chunk: int = 4) -> list[str]:
    """Breaks a script into short on-screen caption chunks, respecting sentence ends."""
    chunks: list[str] = []
    current: list[str] = []

    for word in script.split():
        current.append(word)
        ends_sentence = word.endswith((".", "!", "?"))
        if len(current) >= words_per_chunk or ends_sentence:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Kinetic word-by-word captions (Advanced SubStation Alpha)
#
# edge-tts hands back per-word timing boundaries; those become an .ass file
# where every word gets its own Dialogue event with the spoken word highlighted.
# libass (compiled into the bundled ffmpeg) burns it in one clean pass.
# ---------------------------------------------------------------------------

# ASS colours are &HAABBGGRR -- note the reversed byte order versus HTML.
ASS_WHITE = "&H00FFFFFF"
ASS_BLACK = "&H00000000"
ASS_YELLOW = "&H0000FFFF"
ASS_LIME = "&H0000FF00"

KINETIC_COLOURS = {
    "yellow": ASS_YELLOW,
    "lime": ASS_LIME,
}

# Display faces in preference order; libass resolves these via fontconfig.
# libass takes a family *name*, not a path, and substitutes silently when the
# family is missing -- which is how a burned caption ends up in a default serif
# on a server. The list is filtered against what is really installed before a
# name is written into the .ass header.
ASS_FONT_CANDIDATES = (
    "Montserrat ExtraBold", "Arial Black", "Impact", "Arial",
    "DejaVu Sans Bold", "Liberation Sans Bold", "FreeSans Bold", "DejaVu Sans",
)


def ass_font_name() -> str:
    """The first caption family that is actually installed on this machine."""
    available = installed_families(ASS_FONT_CANDIDATES)
    return available[0] if available else ASS_FONT_CANDIDATES[0]


def _ass_time(seconds: float) -> str:
    """Formats seconds as an ASS timestamp, H:MM:SS.cc (centisecond precision)."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis == 100:                      # rounding can tip into the next second
        centis, secs = 0, secs + 1
        if secs == 60:
            secs, minutes = 0, minutes + 1
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_escape(text: str) -> str:
    """Neutralises characters libass would read as override syntax."""
    return (text.replace("\\", "/")
                .replace("{", "(")
                .replace("}", ")")
                .replace("\n", " ")
                .strip())


def group_words_into_phrases(
    words: Sequence[dict[str, Any]],
    max_words: int = 4,
    max_gap: float = 0.55,
) -> list[list[dict[str, Any]]]:
    """
    Chunks word timings into short on-screen phrases.

    Splits on a natural pause as well as on length: edge-tts strips punctuation
    from its boundary text, so the silence between words is the only reliable
    signal that a sentence ended.
    """
    phrases: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for word in words:
        if current:
            gap = float(word["start"]) - float(current[-1]["end"])
            if gap > max_gap or len(current) >= max_words:
                phrases.append(current)
                current = []
        current.append(dict(word))

    if current:
        phrases.append(current)
    return phrases


def build_ass_subtitles(
    words: Sequence[dict[str, Any]],
    size: tuple[int, int] = (1080, 1920),
    position: str = "bottom",
    highlight: str = "yellow",
    font_scale: float = 0.082,
    max_words: int = 4,
    uppercase: bool = True,
    pop_scale: int = 112,
) -> str:
    """
    Builds an .ass subtitle script with one Dialogue event per spoken word.

    Each event shows the whole phrase with only the currently-spoken word in the
    accent colour, which is the CapCut / TikTok look. Plain ASS karaoke (\\k) is
    deliberately not used: it progressively fills a line left-to-right rather
    than isolating the single active word.
    """
    w, h = size
    font_size = max(28, int(w * font_scale))
    outline = max(3, int(font_size * 0.09))
    font = ass_font_name()
    accent = KINETIC_COLOURS.get(highlight, ASS_YELLOW)

    if position == "center":
        alignment, margin_v = 5, 40
    elif position == "top":
        alignment, margin_v = 8, int(h * 0.12)
    else:                                   # lower third -- the default
        alignment, margin_v = 2, int(h * 0.22)

    side_margin = int(w * 0.08)
    header = (
        "[Script Info]\n"
        "; Generated by the Reelforge Commentary Machine\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {w}\n"
        f"PlayResY: {h}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.601\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Kinetic,{font},{font_size},{ASS_WHITE},{accent},{ASS_BLACK},"
        f"{ASS_BLACK},-1,0,0,0,100,100,0,0,1,{outline},0,{alignment},"
        f"{side_margin},{side_margin},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    phrases = group_words_into_phrases(words, max_words=max_words)
    lines: list[str] = []

    for p_index, phrase in enumerate(phrases):
        if not phrase:
            continue

        phrase_end = float(phrase[-1]["end"])
        # Hold the final word of a phrase until the next phrase begins so the
        # captions stay on screen through short pauses instead of blinking out.
        if p_index + 1 < len(phrases):
            next_start = float(phrases[p_index + 1][0]["start"])
            phrase_end = min(next_start, phrase_end + 0.45)
        else:
            phrase_end += 0.35

        tokens = [_ass_escape(str(word["text"])) for word in phrase]
        if uppercase:
            tokens = [token.upper() for token in tokens]

        for w_index, word in enumerate(phrase):
            start = float(word["start"])
            end = (float(phrase[w_index + 1]["start"])
                   if w_index + 1 < len(phrase) else phrase_end)
            if end <= start:
                end = start + 0.08

            parts: list[str] = []
            for j, token in enumerate(tokens):
                if j == w_index:
                    parts.append(
                        f"{{\\c{accent}&\\fscx{pop_scale}\\fscy{pop_scale}}}"
                        f"{token}"
                        f"{{\\c{ASS_WHITE}&\\fscx100\\fscy100}}"
                    )
                else:
                    parts.append(token)

            lines.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},"
                f"Kinetic,,0,0,0,,{' '.join(parts)}"
            )

    return header + "\n".join(lines) + "\n"


def write_ass_file(
    words: Sequence[dict[str, Any]],
    output_path: str,
    size: tuple[int, int] = (1080, 1920),
    **kwargs: Any,
) -> str:
    """Writes the kinetic caption script to disk as UTF-8 .ass."""
    content = build_ass_subtitles(words, size=size, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return output_path


# ---------------------------------------------------------------------------
# Focus sticker / callout anchor
#
# A high-contrast ring dropped over the opening seconds, the way hook-driven
# edits point the eye at the thing that is about to happen. Drawn once as an
# RGBA PNG and animated by ffmpeg, so it costs nothing per frame in Python.
# ---------------------------------------------------------------------------

FOCUS_POSITIONS: dict[str, str] = {
    "top": "Top-Center",
    "center": "Center",
    "lower": "Lower-Center",
}

# Vertical anchor per preset, as an overlay y-expression. `h` is the sticker's
# own height, which changes every frame while pulsing, so each anchor subtracts
# half of it to stay centred on the intended point.
_FOCUS_Y = {
    "top": "(H*0.22)-(h/2)",
    "center": "(H-h)/2",
    "lower": "(H*0.70)-(h/2)",
}

FOCUS_COLOUR: RGB = (255, 45, 45)


def build_focus_circle_png(
    output_path: str,
    diameter: int = 420,
    colour: RGB = FOCUS_COLOUR,
    glow_layers: int = 7,
) -> str:
    """
    Draws the focus ring: a hard bright circle wrapped in a soft outer glow.

    The glow is stacked translucent rings rather than a blur, which keeps the
    edge crisp at 1080p while still reading as a halo.
    """
    pad = max(12, int(diameter * 0.16))
    canvas = diameter + pad * 2
    image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Outer glow: widest and faintest first, tightening inward.
    for layer in range(glow_layers, 0, -1):
        spread = int(pad * (layer / glow_layers))
        alpha = int(46 * (1.0 - layer / (glow_layers + 1)) ** 0.7) + 8
        width = max(2, int(diameter * 0.018) + spread // 3)
        draw.ellipse(
            [pad - spread, pad - spread, canvas - pad + spread, canvas - pad + spread],
            outline=colour + (alpha,), width=width,
        )

    # A dark liner under the bright ring keeps it readable on light footage.
    draw.ellipse([pad - 3, pad - 3, canvas - pad + 3, canvas - pad + 3],
                 outline=(0, 0, 0, 150), width=max(3, int(diameter * 0.030)))
    draw.ellipse([pad, pad, canvas - pad, canvas - pad],
                 outline=colour + (255,), width=max(5, int(diameter * 0.036)))
    draw.ellipse([pad + 6, pad + 6, canvas - pad - 6, canvas - pad - 6],
                 outline=(255, 255, 255, 90), width=max(2, int(diameter * 0.010)))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    image.save(output_path)
    return output_path


def _focus_filter_chain(
    focus: dict[str, Any],
    source_label: str,
    out_label: str,
) -> list[str]:
    """Builds the two filter steps that size, pulse and place the sticker."""
    diameter = int(focus.get("diameter_px", 420))
    start = float(focus.get("start", 0.0))
    end = float(focus.get("end", 2.0))
    pulse = bool(focus.get("pulse", True))
    y_expr = _FOCUS_Y.get(str(focus.get("position", "center")), _FOCUS_Y["center"])

    # Soft edges so the sticker eases in and out instead of popping.
    fade_out_at = max(0.0, (end - start) - 0.35)
    fades = (f"fade=t=in:st=0:d=0.25:alpha=1,"
             f"fade=t=out:st={fade_out_at:.3f}:d=0.35:alpha=1")

    if pulse:
        # Size breathes ~9%. An opacity pulse via geq was measured at well
        # under 1 fps on a 1080x1920 timeline -- geq evaluates its expression
        # per pixel per plane -- so the animation is carried by scale alone,
        # which runs at ~50 fps for the same result on screen.
        wave = "(1+0.09*sin(2*PI*1.7*t))"
        sized = f"scale=w='{diameter}*{wave}':h='{diameter}*{wave}':eval=frame,{fades}"
    else:
        sized = f"scale={diameter}:{diameter},{fades}"

    return [
        f"[1:v]format=rgba,{sized}[focus]",
        # shortest=1 is load-bearing: the sticker is a `-loop 1` still image,
        # so it is an INFINITE input. Without it the overlay keeps emitting
        # frames after the video ends -- a 4s clip encoded 49 minutes of video
        # before this was caught. The CLI's -shortest does not help, because it
        # only bounds muxing and the video stream itself never ends.
        f"[{source_label}][focus]overlay=x='(W-w)/2':y='{y_expr}'"
        f":enable='between(t,{start:.3f},{end:.3f})':shortest=1[{out_label}]",
    ]


def apply_finishing_pass(
    video_path: str,
    output_path: str,
    ass_path: str | None = None,
    focus: dict[str, Any] | None = None,
    crf: int = 20,
    preset: str = "medium",
) -> str:
    """
    Single ffmpeg pass that burns kinetic captions and/or the focus sticker.

    Both effects share one encode -- running them as separate passes would cost
    a second full re-encode and another generation of quality loss. Audio is
    stream-copied throughout.

    The subtitle file is copied next to the working directory so the filter
    argument stays a bare filename: a Windows drive letter ("D:") inside a
    filter string is parsed as an option separator.
    """
    import shutil
    import subprocess
    import tempfile as _tempfile

    import imageio_ffmpeg

    if not ass_path and not focus:
        raise ValueError("apply_finishing_pass needs captions, a focus sticker, or both.")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    work = _tempfile.mkdtemp(prefix="rf_finish_")

    try:
        command: list[str] = [ffmpeg, "-y", "-i", os.path.abspath(video_path)]
        chain: list[str] = []
        label = "0:v"

        if focus:
            circle = build_focus_circle_png(
                os.path.join(work, "focus.png"),
                diameter=int(focus.get("diameter_px", 420)),
                colour=tuple(focus.get("colour", FOCUS_COLOUR)),  # type: ignore[arg-type]
            )
            command += ["-loop", "1", "-i", circle]

        if ass_path:
            shutil.copyfile(ass_path, os.path.join(work, "captions.ass"))
            chain.append(f"[{label}]ass=captions.ass[vsub]")
            label = "vsub"

        if focus:
            chain += _focus_filter_chain(focus, label, "vout")
            label = "vout"

        command += [
            "-filter_complex", ";".join(chain),
            "-map", f"[{label}]", "-map", "0:a?",
            *video_encoder()["cli"],
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "copy", "-shortest",
            os.path.abspath(output_path),
        ]

        proc = subprocess.run(command, cwd=work, capture_output=True, text=True,
                              timeout=FFMPEG_TIMEOUT)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if proc.returncode != 0 or not os.path.exists(output_path):
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-14:])
        raise RuntimeError(f"ffmpeg finishing pass failed:\n{tail}")

    return output_path


def burn_ass_subtitles(
    video_path: str,
    ass_path: str,
    output_path: str,
    crf: int = 20,
    preset: str = "medium",
) -> str:
    """
    Burns an .ass file onto a video using ffmpeg's libass filter.

    The filter runs with cwd set to a scratch copy of the subtitle file so the
    path inside the filter string stays a bare filename: a Windows drive letter
    ("D:") inside a filter argument is parsed as an option separator and needs
    fragile escaping otherwise.
    """
    import shutil
    import subprocess
    import tempfile as _tempfile

    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    work = _tempfile.mkdtemp(prefix="rf_ass_")
    shutil.copyfile(ass_path, os.path.join(work, "captions.ass"))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    command = [
        ffmpeg, "-y",
        "-i", os.path.abspath(video_path),
        "-vf", "ass=captions.ass",
        *video_encoder()["cli"],
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "copy",
        os.path.abspath(output_path),
    ]

    try:
        proc = subprocess.run(command, cwd=work, capture_output=True, text=True,
                              timeout=FFMPEG_TIMEOUT)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if proc.returncode != 0 or not os.path.exists(output_path):
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        raise RuntimeError(f"ffmpeg failed to burn subtitles:\n{tail}")

    return output_path


# How a short clip is stretched to cover a longer narration.
LOOP_MODES = {
    "boomerang": "Boomerang — plays forward then reverses, seamless at both ends",
    "loop": "Hard loop — restarts from frame one (visible cut)",
    "hold": "Freeze — holds the final frame",
}


def source_time(t: float, base_duration: float, loop_mode: str = "boomerang") -> float:
    """
    Maps a timeline position onto a position inside the source clip.

    A 9s clip under a 30s voiceover has to come from somewhere. Boomerang
    ping-pongs through the footage so the seam at each end is continuous
    motion rather than a jump cut, which is why it is the default.
    """
    if base_duration <= 0:
        return 0.0
    if t < base_duration:
        return t

    if loop_mode == "hold":
        return base_duration
    if loop_mode == "loop":
        return t % base_duration

    # Boomerang: triangle wave over a 2x-length cycle.
    cycle = 2.0 * base_duration
    pos = t % cycle
    return pos if pos <= base_duration else cycle - pos


def fit_filter_chain(
    target_size: tuple[int, int],
    fit: str = "crop_fill",
    in_label: str = "0:v",
    out_label: str = "fr",
) -> str:
    """
    Builds the ffmpeg filter_complex fragment behind the sidebar's "Image fit".

      crop_fill  Scale up until the frame is covered, then centre-crop. Fills
                 1080x1920 completely -- no letterboxing, at the cost of the
                 edges of a landscape source.
      blur_pad   That same crop-fill pass, blurred, used as a background plate;
                 the untouched source is scaled to fit and centred on top. The
                 frame is full, and every pixel of the original stays in view.
      color_pad  Scale to fit and pad the remainder with flat colour.

    Returns a fragment ending in `[out_label]`, ready to join with ';'.
    """
    tw, th = target_size
    src = f"[{in_label}]"

    if fit == "crop_fill":
        return (f"{src}scale={tw}:{th}:force_original_aspect_ratio=increase,"
                f"crop={tw}:{th}:(in_w-{tw})/2:(in_h-{th})/2[{out_label}]")

    if fit == "blur_pad":
        # Two passes over the same frame: one covering and blurred, one fitted.
        # The split is explicit -- ffmpeg will insert one for a reused label,
        # but naming the branches keeps the graph readable when the boomerang
        # pass appends its own split onto the end of this chain.
        return (
            f"{src}split=2[{out_label}bgsrc][{out_label}fgsrc];"
            f"[{out_label}bgsrc]scale={tw}:{th}:force_original_aspect_ratio=increase,"
            f"crop={tw}:{th},boxblur={BLUR_STRENGTH}[{out_label}bg];"
            f"[{out_label}fgsrc]scale={tw}:{th}:"
            f"force_original_aspect_ratio=decrease[{out_label}fg];"
            f"[{out_label}bg][{out_label}fg]overlay=(W-w)/2:(H-h)/2[{out_label}]"
        )

    # color_pad, and anything unrecognised, lands here.
    return (f"{src}scale={tw}:{th}:force_original_aspect_ratio=decrease,"
            f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color={PAD_COLOUR}[{out_label}]")


def _scratch_path(kind: str) -> str:
    import tempfile as _tempfile

    return os.path.join(_tempfile.gettempdir(), f"rf_{kind}_{int(time.time() * 1000)}.mp4")


def _run_ffmpeg(graph: str, source: str, out: str, extra_in: Sequence[str] = ()) -> str | None:
    """Runs one filter_complex pass; returns the output path, or None on failure."""
    import subprocess

    import imageio_ffmpeg

    enc = video_encoder()
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        *extra_in, "-i", os.path.abspath(source),
        "-filter_complex", graph, "-map", "[v]", "-an",
        *enc["cli"], "-pix_fmt", "yuv420p", out,
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except Exception as exc:
        print(f"Warning: reframe pass failed to start: {exc}")
        return None

    if proc.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        print(f"Warning: reframe pass failed: {(proc.stderr or '').strip()[:300]}")
        return None

    _SCRATCH_RENDERS.append(out)
    return out


def source_duration(video_path: str) -> float:
    """Length of a clip in seconds, or 0.0 if it cannot be read."""
    from moviepy import VideoFileClip

    try:
        probe = VideoFileClip(video_path)
        seconds = float(probe.duration or 0.0)
        probe.close()
        return seconds
    except Exception:
        return 0.0


def reframe_source(
    video_path: str,
    target_size: tuple[int, int],
    fit: str = "crop_fill",
    duration: float | None = None,
    max_seconds: float = 300.0,
) -> str | None:
    """
    Pre-renders a clip at the output size using the chosen fit, once.

    The alternative is resizing every frame in PIL inside the render loop,
    which was the second-worst bottleneck in this pipeline after backward
    seeking. Doing it here also means "Image fit" is produced by exactly the
    same filter graph whichever path a clip takes.

    `duration` trims the pass to what the narration will actually use, so a
    3-minute source under a 40-second voiceover only re-encodes 40 seconds.
    Returns the scratch path, or None so the caller falls back to PIL.
    """
    seconds = duration if duration and duration > 0 else source_duration(video_path)
    if seconds > max_seconds:
        return None

    out = _scratch_path("reframe")
    graph = fit_filter_chain(target_size, fit, in_label="0:v", out_label="v")
    extra: list[str] = []
    if duration and duration > 0:
        # Half a second of slack so the last frame is never short.
        extra = ["-t", f"{duration + 0.5:.3f}"]
    return _run_ffmpeg(graph, video_path, out, extra_in=extra)


def build_boomerang_source(
    video_path: str,
    target_size: tuple[int, int] | None = None,
    fit: str = "crop_fill",
    max_source_seconds: float = 20.0,
) -> str | None:
    """
    Pre-renders one seamless forward+reverse cycle of a clip to a temp file.

    Two reasons this is a file and not a playback trick:

    Speed. Boomerang playback reads the source backwards, and a backward seek
    per frame is brutally slow: measured at 224 ms/frame versus 60 ms
    sequential on a 1080x1920 webm, which turned a 75s render into nine
    minutes. Materialising the cycle once lets the renderer read straight
    through, and folding the reframe into the same pass removes the per-frame
    PIL resize as well.

    Smoothness. A naive split/reverse/concat duplicates a frame at both seams:
    the last frame plays, then plays again as the first frame of the reverse,
    and the cycle ends on the first frame only for the loop to start on it
    again. Verified on a 12-frame source, the order came out 0..11,11..0 --
    two frozen frames per cycle, which at 24fps is the 83ms hitch that reads
    as the picture snapping back. Trimming one frame off the head of each
    branch gives 1..11,10..0, which wraps round to 1: a continuous triangle
    with no repeats and no skips.

    Returns the temp path, or None when it is not worth it (long source, or
    ffmpeg failed) so the caller falls back to seeking.
    """
    duration = source_duration(video_path)

    # The reverse filter buffers the whole clip in memory; keep that bounded.
    if duration <= 0 or duration > max_source_seconds:
        return None

    parts: list[str] = []
    head = "0:v"
    if target_size:
        parts.append(fit_filter_chain(target_size, fit, in_label="0:v", out_label="fr"))
        head = "fr"

    parts.append(
        f"[{head}]split[fwd][rev];"
        "[fwd]trim=start_frame=1,setpts=PTS-STARTPTS[f];"
        "[rev]reverse,trim=start_frame=1,setpts=PTS-STARTPTS[r];"
        "[f][r]concat=n=2:v=1[v]"
    )
    return _run_ffmpeg(";".join(parts), video_path, _scratch_path("boomerang"))


def purge_scratch_renders() -> int:
    """
    Deletes every render intermediate this process created.

    Covers the reframe and boomerang passes here plus the beds, mixes and voice
    takes Minimalist Motion registers through `motion_engine.scratch_wav`. They
    all live in the system temp directory, which the exports/ sweeper never
    looks at, so without this they accumulate at roughly one source-clip-sized
    file per render.
    """
    removed = 0
    while _SCRATCH_RENDERS:
        path = _SCRATCH_RENDERS.pop()
        try:
            if os.path.exists(path):
                os.remove(path)
                removed += 1
        except OSError:
            # Still held open by a reader; it ages out with the temp directory.
            pass
    return removed


def sweep_temp_renders(max_age_seconds: float = 3600.0) -> dict[str, int]:
    """
    Removes stale render intermediates left in the system temp directory.

    `purge_scratch_renders` only knows about files this process created. A run
    that crashed, or was killed mid-render, leaves its scratch behind forever --
    the exports/ sweeper never looks in the temp directory. This is the
    backstop, and it skips anything the current process still has in flight.
    """
    import glob
    import tempfile

    live = {os.path.abspath(path) for path in _SCRATCH_RENDERS}
    cutoff = time.time() - max(0.0, float(max_age_seconds))
    removed = freed = 0

    for path in glob.glob(os.path.join(tempfile.gettempdir(), "rf_*")):
        try:
            if os.path.abspath(path) in live or os.path.getmtime(path) > cutoff:
                continue
            size = os.path.getsize(path)
            os.remove(path)
            removed += 1
            freed += size
        except OSError:
            # Held open by another process; it ages out on the next sweep.
            continue

    return {"removed": removed, "bytes": freed}


def _build_commentary_clip(
    source_video: str,
    duration: float,
    script: str,
    speech_duration: float,
    target_size: tuple[int, int],
    burn_captions: bool,
    caption_position: str,
    watermark_text: str,
    loop_mode: str,
    fit: str = "crop_fill",
) -> tuple[VideoClip, list[Any], Any]:
    """
    Builds the visual layer shared by the full render and the muted export.

    Returns (clip, clips_to_close, source_clip) so callers can attach audio or
    not, and close everything afterwards.
    """
    from moviepy import VideoFileClip

    effective_mode = loop_mode
    source_seconds = source_duration(source_video)
    prescaled = False

    # Boomerang over a short source is the common case (a 9s clip under a 75s
    # narration): materialise one clean forward+reverse cycle so the reader
    # runs sequentially instead of seeking backwards on every frame.
    if loop_mode == "boomerang" and 0 < source_seconds < duration:
        cycle = build_boomerang_source(source_video, target_size=target_size, fit=fit)
        if cycle:
            source_video = cycle
            effective_mode = "loop"     # the cycle already contains the reverse
            prescaled = True            # ...and is already at output size

    if not prescaled:
        # Every other case still needs the fit applied. One ffmpeg pass beats a
        # PIL resize per frame, and keeps the three fit modes pixel-identical
        # to the boomerang path.
        framed = reframe_source(
            source_video, target_size, fit,
            duration=duration if source_seconds >= duration else None,
        )
        if framed:
            source_video = framed
            prescaled = True

    base = VideoFileClip(source_video)
    open_clips: list[Any] = [base]

    tw, th = target_size
    sw, sh = base.size
    compose: Any = None
    crop_x = crop_y = 0

    if prescaled and (sw, sh) == (tw, th):
        # ffmpeg already produced exactly the output frame; resizing again
        # would cost a PIL pass per frame for no visible difference.
        scaled = base
    elif fit in ("blur_pad", "color_pad"):
        # PIL fallback for the padded modes, reached only when the ffmpeg pass
        # above was unavailable: scale to fit, then fill the remainder.
        ratio = min(tw / sw, th / sh)
        fit_w = max(1, int(round(sw * ratio)))
        fit_h = max(1, int(round(sh * ratio)))
        scaled = base.resized((fit_w, fit_h))
        open_clips.append(scaled)
        off_x, off_y = (tw - fit_w) // 2, (th - fit_h) // 2
        small = (max(1, tw // 24), max(1, th // 24))

        def pad_frame(frame: np.ndarray) -> np.ndarray:
            picture = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
            if fit == "blur_pad":
                # Down then up is a box blur for a fraction of the cost of a
                # real one, and the plate is out of focus by design.
                canvas = picture.resize(small, Image.Resampling.BILINEAR)
                canvas = canvas.resize((tw, th), Image.Resampling.BILINEAR)
            else:
                canvas = Image.new("RGB", (tw, th), PAD_RGB)
            canvas.paste(picture, (off_x, off_y))
            return np.asarray(canvas, dtype=np.float32)

        compose = pad_frame
    else:
        ratio = max(tw / sw, th / sh)
        fill_w, fill_h = max(tw, int(sw * ratio)), max(th, int(sh * ratio))
        scaled = base.resized((fill_w, fill_h))
        open_clips.append(scaled)
        crop_x, crop_y = (fill_w - tw) // 2, (fill_h - th) // 2

    # --- caption track, timed across the spoken audio -----------------------
    # Timed off speech_duration, never off the source length, so looping the
    # picture cannot pull the captions out of sync with the voice.
    overlays: list[tuple[float, float, Any, Any]] = []
    if burn_captions and script.strip():
        chunks = split_caption_chunks(script)
        weights = [max(1, len(c.split())) for c in chunks]
        total_w = float(sum(weights))
        cursor = 0.0
        for chunk, weight in zip(chunks, weights):
            span = speech_duration * (weight / total_w)
            rgba = np.array(caption_overlay_rgba((tw, th), chunk, caption_position), dtype=np.float32)
            overlays.append((cursor, cursor + span, rgba[..., :3], rgba[..., 3:4] / 255.0))
            cursor += span

    mark = make_watermark_tile(target_size, watermark_text, position="top_right")
    mark_arr = None
    if mark is not None:
        tile, (mx, my) = mark
        tile_a = np.array(tile, dtype=np.float32)
        mark_arr = (mx, my, tile_a[..., :3], tile_a[..., 3:4] / 255.0)

    base_dur = float(base.duration or 0.0)
    last_frame_t = max(0.0, base_dur - 1e-3)

    def make_frame(t: float):
        src_t = min(source_time(t, base_dur, effective_mode), last_frame_t)
        raw = np.asarray(scaled.get_frame(src_t), dtype=np.float32)
        frame = compose(raw) if compose is not None else raw[crop_y:crop_y + th, crop_x:crop_x + tw]

        for start, end, rgb, alpha in overlays:
            if start <= t < end:
                frame = frame * (1.0 - alpha) + rgb * alpha
                break

        if mark_arr is not None:
            mx, my, m_rgb, m_a = mark_arr
            mh, mw = m_rgb.shape[:2]
            if my + mh <= th and mx + mw <= tw:
                region = frame[my:my + mh, mx:mx + mw]
                frame[my:my + mh, mx:mx + mw] = region * (1.0 - m_a) + m_rgb * m_a

        return np.clip(frame, 0, 255).astype(np.uint8)

    return VideoClip(make_frame, duration=duration), open_clips, base


def _close_all(clips: Sequence[Any]) -> None:
    for clip in clips:
        try:
            clip.close()
        except Exception:
            pass


def render_commentary_video(
    source_video: str,
    narration_path: str,
    script: str,
    output_path: str,
    target_size: tuple[int, int] = (1080, 1920),
    fps: int | None = None,
    original_volume: float = 0.10,
    narration_volume: float = 1.0,
    caption_position: str = "center",
    burn_captions: bool = True,
    watermark_text: str = "",
    loop_mode: str = "boomerang",
    fit: str = "crop_fill",
    bgm_path: str | None = None,
    bgm_volume: float = 1.0,
    ass_path: str | None = None,
    focus: dict[str, Any] | None = None,
    hook_sfx: bool = False,
    hook_volume: float = 0.75,
    tail: float = 0.4,
    progress_callback: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Lays an AI narration over a raw clip and burns viral captions onto it.

    The narration drives the runtime: a short clip is looped (boomerang by
    default) to cover the voiceover and a long one is trimmed to it, so picture
    and speech always end together.
    """
    if progress_callback:
        progress_callback(1, 4, "Stage 2/4 · Visual rendering — loading source clip...")

    narration = AudioFileClip(narration_path)
    speech = float(narration.duration)
    duration = speech + max(0.0, tail)

    if progress_callback:
        progress_callback(2, 4, "Stage 2/4 · Visual rendering — reframing and building captions...")

    video, open_clips, base = _build_commentary_clip(
        source_video, duration, script, speech, target_size,
        burn_captions, caption_position, watermark_text, loop_mode, fit,
    )
    open_clips.append(narration)
    out_fps = int(fps or getattr(base, "fps", None) or 30)
    base_dur = float(base.duration or 0.0)

    # --- audio: ducked source under the narration ---------------------------
    if progress_callback:
        progress_callback(3, 4, "Stage 3/4 · Audio/SFX multiplexing — mixing narration over clip...")

    tracks = []
    if base.audio is not None and original_volume > 0:
        src_audio = base.audio
        if src_audio.duration > duration:
            src_audio = src_audio.subclipped(0, duration)
        tracks.append(src_audio.with_volume_scaled(original_volume))
        open_clips.append(src_audio)

    tracks.append(narration.with_volume_scaled(narration_volume) if narration_volume != 1.0 else narration)

    # Suspense bed. The WAV arrives already ducked against the narration
    # envelope, so it only needs level matching here, not another gain rider.
    if bgm_path and os.path.exists(bgm_path):
        bed = AudioFileClip(bgm_path)
        open_clips.append(bed)
        if bed.duration > duration:
            bed = bed.subclipped(0, duration)
        tracks.append(bed.with_volume_scaled(bgm_volume) if bgm_volume != 1.0 else bed)

    # Opening anchor: a half-second sub-bass hit at 0.0s, sitting underneath the
    # first words rather than in front of them. Paired with the focus circle so
    # the visual and audio hooks land on the same frame.
    if hook_sfx:
        try:
            from audio_engine import SFX_HOOK, get_sfx_path

            hit = AudioFileClip(get_sfx_path(SFX_HOOK))
            open_clips.append(hit)
            if hit.duration > duration:
                hit = hit.subclipped(0, duration)
            tracks.append(hit.with_volume_scaled(hook_volume).with_start(0.0))
        except Exception as exc:
            print(f"Warning: hook SFX could not be mixed: {exc}")

    # Always composite, even for a single track: the clip runs `tail` seconds
    # longer than the narration, and a bare AudioFileClip asked for samples past
    # its end raises OSError, whereas a composite pads the gap with silence.
    audio = CompositeAudioClip(tracks)
    audio.duration = duration
    video = video.with_audio(audio)

    if progress_callback:
        progress_callback(4, 4, "Stage 4/4 · Final export — encoding MP4...")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # With kinetic captions the picture is encoded first, then libass burns the
    # .ass on in a second pass that copies the audio stream untouched.
    needs_finish = bool(ass_path or focus)
    stage_path = output_path
    if needs_finish:
        root, ext = os.path.splitext(output_path)
        stage_path = f"{root}__nosubs{ext or '.mp4'}"

    enc = video_encoder()
    video.write_videofile(
        stage_path, fps=out_fps, codec=enc["codec"], audio_codec="aac",
        preset=enc["preset"], ffmpeg_params=list(enc["ffmpeg_params"]),
    )

    _close_all(open_clips + [video])
    purge_scratch_renders()

    if needs_finish:
        if progress_callback:
            what = " + ".join(
                part for part in ("kinetic captions" if ass_path else "",
                                  "focus sticker" if focus else "") if part
            )
            progress_callback(4, 4, f"Stage 4/4 - Final export: burning {what}...")
        apply_finishing_pass(stage_path, output_path, ass_path=ass_path, focus=focus)
        try:
            os.remove(stage_path)
        except OSError:
            pass

    loops = (duration / base_dur) if base_dur > 0 else 0.0
    return {
        "output_path": output_path,
        "captions_mode": "kinetic" if ass_path else ("chunked" if burn_captions else "none"),
        "ass_path": ass_path or "",
        "bgm": bool(bgm_path),
        "focus": bool(focus),
        "hook_sfx": bool(hook_sfx),
        "duration": duration,
        "fps": out_fps,
        "captions": 0 if not burn_captions else len(split_caption_chunks(script)),
        "source_duration": base_dur,
        "loop_mode": loop_mode,
        "loops": loops,
        "target_size": target_size,
        "fit": fit,
    }


def export_muted_video(
    source_video: str,
    output_path: str,
    duration: float,
    script: str = "",
    target_size: tuple[int, int] = (1080, 1920),
    fps: int | None = None,
    caption_position: str = "center",
    burn_captions: bool = False,
    watermark_text: str = "",
    loop_mode: str = "boomerang",
    fit: str = "crop_fill",
    progress_callback: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Writes the silent visual track on its own, reframed and looped to `duration`.

    This is the CapCut hand-off: drop this and the .mp3 on a timeline and they
    line up frame-for-frame with no manual retiming.
    """
    if progress_callback:
        progress_callback(1, 2, "Building silent video track...")

    video, open_clips, base = _build_commentary_clip(
        source_video, duration, script, duration, target_size,
        burn_captions, caption_position, watermark_text, loop_mode, fit,
    )
    out_fps = int(fps or getattr(base, "fps", None) or 30)
    base_dur = float(base.duration or 0.0)

    if progress_callback:
        progress_callback(2, 2, "Encoding muted MP4...")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    enc = video_encoder()
    video.write_videofile(
        output_path, fps=out_fps, codec=enc["codec"], audio=False,
        preset=enc["preset"], ffmpeg_params=list(enc["ffmpeg_params"]),
    )

    _close_all(open_clips + [video])
    purge_scratch_renders()

    return {
        "output_path": output_path,
        "duration": duration,
        "fps": out_fps,
        "source_duration": base_dur,
        "loop_mode": loop_mode,
        "loops": (duration / base_dur) if base_dur > 0 else 0.0,
        "fit": fit,
    }


# ---------------------------------------------------------------------------
# Versus Duel Engine
#
# Lives in duel_engine.py. It is imported lazily -- inside the functions that
# need it -- rather than here, so the dependency runs one way: duel_engine may
# import this module at its top, and nothing imports back.
#
# The names below are re-exported for callers (and ledger entries) that still
# reach for video_engine.duel_*.
# ---------------------------------------------------------------------------


def __getattr__(name: str):
    """Forwards the duel symbols to duel_engine on first access."""
    if name in _DUEL_EXPORTS:
        import duel_engine

        return getattr(duel_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_DUEL_EXPORTS = frozenset({
    "DUEL_ACCENT_A", "DUEL_ACCENT_B", "DUEL_GOLD",
    "format_stat", "duel_round_beats", "title_gate",
    "create_duel_intro_clip", "create_duel_round_clip", "create_winner_clip",
})


def _schedule_sfx(
    slides: Sequence[dict[str, Any]],
    starts: Sequence[float],
    transition_type: str,
    transition_dur: float,
) -> list[tuple[float, str]]:
    """
    Builds the (time, sfx) cue sheet for a reel.

    Risers are placed *before* each cut so the sweep resolves on the new slide;
    impacts and chimes come off duel_round_beats(), the same clock the renderer
    animates to, which is what keeps picture and sound locked together.
    """
    from audio_engine import SFX_CHIME, SFX_IMPACT, SFX_WHOOSH
    from duel_engine import duel_round_beats

    cues: list[tuple[float, str]] = []
    hard_cut = transition_type == "cut" or transition_dur <= 0

    for idx, slide in enumerate(slides):
        if idx >= len(starts):
            break
        start = float(starts[idx])
        kind = slide.get("kind", "image")

        # Riser leading into every cut except the opening frame.
        if idx > 0:
            lead = 0.62 if hard_cut else max(0.62, float(transition_dur))
            cues.append((start - lead, SFX_WHOOSH))

        if kind == "duel_round":
            beats = duel_round_beats(float(slide.get("duration", 5.0)))
            cues.append((start + beats["a_in"], SFX_IMPACT))
            cues.append((start + beats["b_in"], SFX_IMPACT))
            if str(slide.get("round", {}).get("winner", "")).upper() in ("A", "B"):
                cues.append((start + beats["reveal"], SFX_CHIME))
        elif kind == "duel_winner":
            cues.append((start + 0.30, SFX_CHIME))

    return cues


def build_reel_video(
    slides,
    aspect_ratio_name="9:16 (Vertical Reels / Shorts / TikTok)",
    fit_mode="blur_pad",
    transition_type="crossfade",
    transition_dur=0.5,
    audio_path=None,
    audio_volume=0.8,
    watermark_text="",
    fps=24,
    output_path="output_reel.mp4",
    progress_callback=None,
    voiceover_volume=1.0,
    music_duck=0.28,
    sfx_enabled: bool = True,
    sfx_volume: float = 0.55
):
    """
    Assembles complete multi-slide video reel with motion, transitions, typography & audio.

    When slides carry a 'voice_path' (an AI voiceover synthesized by
    audio_engine), narration is layered on top of the music bed at each slide's
    start time and the music is automatically ducked to `music_duck` of its
    volume so the narration stays intelligible.
    """
    target_size = ASPECT_RATIOS.get(aspect_ratio_name, (1080, 1920))
    total_steps = len(slides) + 3

    if progress_callback:
        progress_callback(1, total_steps, "Stage 2/4 · Visual rendering — preparing slides and motion...")

    from duel_engine import (
        create_duel_intro_clip, create_duel_round_clip, create_winner_clip,
    )

    raw_clips = []
    accent_colors = [(255, 105, 180), (56, 189, 248), (250, 204, 21), (168, 85, 247)]

    for idx, slide_info in enumerate(slides):
        img_raw = slide_info.get("image")
        caption = slide_info.get("caption", "")
        dur = slide_info.get("duration", 4.0)
        motion = slide_info.get("motion", "zoom_in")
        c_pos = slide_info.get("caption_pos", "bottom")
        c_style = slide_info.get("caption_style", "glass")
        accent = accent_colors[idx % len(accent_colors)]

        kind = slide_info.get("kind", "image")

        if kind == "duel_intro":
            slide_clip = create_duel_intro_clip(
                target_size,
                slide_info.get("item_a", {}),
                slide_info.get("item_b", {}),
                headline=slide_info.get("caption", ""),
                duration=dur,
                layout=slide_info.get("layout", "stacked"),
                watermark_text=watermark_text,
            )
        elif kind == "duel_round":
            slide_clip = create_duel_round_clip(
                target_size,
                slide_info.get("item_a", {}),
                slide_info.get("item_b", {}),
                slide_info.get("round", {}),
                duration=dur,
                layout=slide_info.get("layout", "stacked"),
                watermark_text=watermark_text,
            )
        elif kind == "duel_winner":
            slide_clip = create_winner_clip(
                target_size,
                slide_info.get("winner_item", {}),
                tuple(slide_info.get("tally", (0, 0))),
                duration=dur,
                is_a=slide_info.get("winner_is_a", True),
                headline=slide_info.get("caption", ""),
                watermark_text=watermark_text,
            )
        else:
            fitted_img = fit_image_to_aspect(img_raw, target_size=target_size, fit_mode=fit_mode)
            slide_clip = create_slide_clip(
                fitted_img,
                duration=dur,
                motion_type=motion,
                caption_text=caption,
                caption_pos=c_pos,
                caption_style=c_style,
                watermark_text=watermark_text,
                accent_color=accent,
                highlight_words=slide_info.get("highlight_words")
            )

        raw_clips.append(slide_clip)

        if progress_callback:
            progress_callback(2 + idx, total_steps, f"Stage 2/4 · Visual rendering — slide {idx + 1}/{len(slides)}")

    if progress_callback:
        progress_callback(len(slides) + 1, total_steps, f"Stage 2/4 · Visual rendering — applying '{transition_type}' transitions...")

    # Start time of each slide on the finished timeline, so voiceover clips can
    # be anchored to the exact frame their slide appears.
    slide_start_times = []

    if len(raw_clips) == 1 or transition_type == "cut" or transition_dur <= 0.0:
        timeline_clips = []
        curr_time = 0.0
        for c in raw_clips:
            slide_start_times.append(curr_time)
            timeline_clips.append(c.with_start(curr_time))
            curr_time += c.duration
        final_video = CompositeVideoClip(timeline_clips)
    elif transition_type == "crossfade":
        # The common path, and the one worth not going through
        # CompositeVideoClip for. vfx.CrossFadeIn masks the clip for its whole
        # duration, so every frame pays for a full-canvas alpha composite even
        # when nothing is overlapping: 131ms per frame against 54ms to draw it.
        from duel_engine import crossfade_sequence

        final_video, slide_start_times = crossfade_sequence(raw_clips, transition_dur)
    else:
        timeline_clips = []
        curr_time = 0.0
        actual_trans_dur = min(transition_dur, min(c.duration for c in raw_clips) * 0.4)

        for idx, c in enumerate(raw_clips):
            if idx == 0:
                slide_start_times.append(0.0)
                timeline_clips.append(c.with_start(0.0))
                curr_time = c.duration - actual_trans_dur
            else:
                slide_start_times.append(curr_time)

                if transition_type == "crossfade":
                    c_trans = c.with_start(curr_time).with_effects([vfx.CrossFadeIn(actual_trans_dur)])
                    timeline_clips.append(c_trans)

                elif transition_type == "slide_left":
                    def make_pos(t):
                        if t < actual_trans_dur:
                            prog = t / actual_trans_dur
                            x = int(target_size[0] * (1.0 - prog))
                            return (x, 0)
                        return (0, 0)

                    c_trans = c.with_start(curr_time).with_position(make_pos)
                    timeline_clips.append(c_trans)

                elif transition_type == "slide_bottom":
                    def make_pos_b(t):
                        if t < actual_trans_dur:
                            prog = t / actual_trans_dur
                            y = int(target_size[1] * (1.0 - prog))
                            return (0, y)
                        return (0, 0)

                    c_trans = c.with_start(curr_time).with_position(make_pos_b)
                    timeline_clips.append(c_trans)

                elif transition_type == "fade_black":
                    c_trans = c.with_start(curr_time).with_effects([vfx.FadeIn(actual_trans_dur)])
                    timeline_clips.append(c_trans)

                else:
                    timeline_clips.append(c.with_start(curr_time))

                curr_time += c.duration - actual_trans_dur

        final_video = CompositeVideoClip(timeline_clips)

    if progress_callback:
        progress_callback(len(slides) + 2, total_steps, "Stage 3/4 · Audio/SFX multiplexing — mixing narration, music and SFX...")

    audio_clip = None
    open_audio_clips = []
    v_dur = final_video.duration

    # Layer 1: AI voiceover narration, anchored to each slide's start frame.
    narration_clips = []
    for idx, slide_info in enumerate(slides):
        v_path = slide_info.get("voice_path")
        if not v_path or not os.path.exists(v_path):
            continue
        try:
            vo = AudioFileClip(v_path)
            open_audio_clips.append(vo)

            start_at = slide_start_times[idx] + float(slide_info.get("voice_offset", 0.0))
            # Never let narration spill past the end of the reel.
            if start_at >= v_dur:
                continue
            if start_at + vo.duration > v_dur:
                vo = vo.subclipped(0, max(0.05, v_dur - start_at))

            if voiceover_volume != 1.0 and hasattr(vo, "with_volume_scaled"):
                vo = vo.with_volume_scaled(voiceover_volume)

            narration_clips.append(vo.with_start(start_at))
        except Exception as e:
            print(f"Warning: Voiceover for slide {idx + 1} could not be mixed: {e}")

    # Layer 2: music bed, looped to length and ducked under any narration.
    music_clip = None
    if audio_path and os.path.exists(audio_path):
        try:
            a_raw = AudioFileClip(audio_path)
            open_audio_clips.append(a_raw)

            if a_raw.duration < v_dur and a_raw.duration > 0:
                num_loops = math.ceil(v_dur / a_raw.duration)
                from moviepy import concatenate_audioclips
                a_looped = concatenate_audioclips([a_raw] * num_loops)
                open_audio_clips.append(a_looped)
                music_clip = a_looped.subclipped(0, v_dur)
            else:
                music_clip = a_raw.subclipped(0, v_dur)

            music_level = audio_volume * (music_duck if narration_clips else 1.0)
            if hasattr(music_clip, "with_volume_scaled"):
                music_clip = music_clip.with_volume_scaled(music_level)
        except Exception as e:
            print(f"Warning: Audio processing encountered issue: {e}")
            music_clip = None

    # Layer 3: kinetic SFX, scheduled off the same animation clock the duel
    # renderer uses, so every hit lands on the frame its card appears.
    sfx_clips = []
    if sfx_enabled and sfx_volume > 0:
        for when, sfx_name in _schedule_sfx(slides, slide_start_times, transition_type, transition_dur):
            if when < 0 or when >= v_dur:
                continue
            try:
                from audio_engine import get_sfx_path

                clip = AudioFileClip(get_sfx_path(sfx_name))
                open_audio_clips.append(clip)
                if when + clip.duration > v_dur:
                    clip = clip.subclipped(0, max(0.05, v_dur - when))
                if hasattr(clip, "with_volume_scaled"):
                    clip = clip.with_volume_scaled(sfx_volume)
                sfx_clips.append(clip.with_start(when))
            except Exception as e:
                print(f"Warning: SFX '{sfx_name}' could not be mixed: {e}")

    mix = ([music_clip] if music_clip is not None else []) + narration_clips + sfx_clips
    if mix:
        try:
            audio_clip = CompositeAudioClip(mix) if len(mix) > 1 else mix[0]
            audio_clip = audio_clip.with_duration(v_dur)
            final_video = final_video.with_audio(audio_clip)
        except Exception as e:
            print(f"Warning: Audio mixdown failed: {e}")
            audio_clip = None

    if progress_callback:
        progress_callback(total_steps, total_steps, "Stage 4/4 · Final export — encoding MP4 (this is the slow part)...")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    final_duration = final_video.duration
    write_clip(final_video, output_path, fps=fps, with_audio=audio_clip is not None)

    for c in raw_clips:
        c.close()
    for a in open_audio_clips:
        try:
            a.close()
        except Exception:
            pass
    final_video.close()

    return {
        "output_path": output_path,
        "duration": final_duration,
        "fps": fps,
        "num_slides": len(slides),
        "aspect_ratio": aspect_ratio_name,
        "num_voiceovers": len(narration_clips),
        "has_music": music_clip is not None
    }

