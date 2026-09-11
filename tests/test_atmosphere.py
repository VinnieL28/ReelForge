"""
Atmosphere Studio: the soundscape synthesizer, the visual drift, the uploader.

The two things that actually matter in long-form ambient are measured rather
than asserted by eye:

* **Seams.** A loop that clicks once every two minutes is worse than no loop,
  because the listener is asleep and the click wakes them. Every join is
  measured against the ordinary sample-to-sample step in the same signal.
* **Drift geometry.** The cycle mode is stream-copied for hours, so if it does
  not return to exactly where it started there is a visible jump every cycle.
  Measured by reading the zoom off a grid, not by comparing pixels -- a 0.04%
  zoom decorrelates a noisy image completely while the geometry is fine.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest
from PIL import Image

import ambient_engine as ae
import publisher as pub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_image(path: str, size: tuple[int, int] = (1920, 1080), step: int = 40) -> str:
    """A hard-edged grid: the spacing between lines is a direct read of zoom."""
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[::step, :, :] = 255
    img[:, ::step, :] = 255
    Image.fromarray(img).save(path)
    return path


def _zoom_of(frame: np.ndarray) -> float:
    """
    Grid spacing by least-squares over every detected line centroid.

    Averaging the fit over all ~27 lines is what makes this accurate enough to
    resolve a 0.4% difference; counting peak rows quantises to whole pixels and
    an FFT bin is too coarse at this period.
    """
    col = frame.mean(axis=2).mean(axis=1)
    threshold = col.mean() + 0.5 * col.std()
    centres: list[float] = []
    start = None
    for i, above in enumerate(col > threshold):
        if above and start is None:
            start = i
        elif not above and start is not None:
            weights = col[start:i] - threshold
            if weights.sum() > 0:
                centres.append(float((np.arange(start, i) * weights).sum() / weights.sum()))
            start = None
    if len(centres) < 8:
        return 0.0
    return float(np.polyfit(np.arange(len(centres), dtype=np.float64),
                            np.asarray(centres), 1)[0])


def _decode_mono(path: str, rate: int = 22050) -> np.ndarray:
    import imageio_ffmpeg

    proc = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
         "-i", os.path.abspath(path), "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"],
        capture_output=True, timeout=600)
    return np.frombuffer(proc.stdout, dtype="<f4")


# ---------------------------------------------------------------------------
# Soundscape synthesis
# ---------------------------------------------------------------------------

class TestSoundscapes:

    @pytest.mark.parametrize("bed", list(ae.PRIMARY_BEDS))
    def test_every_bed_is_finite_and_in_range(self, bed):
        """A single NaN makes max() NaN, which silences the whole eight-hour
        render -- and it surfaces hours after the mistake."""
        audio = ae.build_seed(bed, "none", seconds=4.0, seed=0)
        assert np.isfinite(audio).all(), f"{bed} produced NaN or inf"
        assert audio.ndim == 2 and audio.shape[1] == 2, "not stereo"
        peak = float(np.abs(audio).max())
        assert 0.5 < peak <= 1.0, f"{bed} peaked at {peak}"

    @pytest.mark.parametrize("texture", [t for t in ae.SECONDARY_TEXTURES if t != "none"])
    def test_every_texture_is_finite(self, texture):
        audio = ae.build_seed("brown_noise", texture, seconds=4.0, seed=0)
        assert np.isfinite(audio).all(), f"{texture} produced NaN or inf"
        assert float(np.abs(audio).max()) > 0.1, f"{texture} is silent"

    def test_the_beds_are_spectrally_distinct(self):
        """Five beds that all sound like hiss are one bed with five names."""
        centroids = {}
        for bed in ae.PRIMARY_BEDS:
            mono = ae.build_seed(bed, "none", seconds=4.0, seed=0).mean(axis=1)
            spec = np.abs(np.fft.rfft(mono))
            freqs = np.fft.rfftfreq(len(mono), 1.0 / ae.SAMPLE_RATE)
            centroids[bed] = float((spec * freqs).sum() / max(1e-9, spec.sum()))

        assert centroids["brown_noise"] == min(centroids.values()), \
            "brown noise is not the darkest bed"
        assert centroids["stream"] > centroids["fireplace"] * 1.5, \
            "the stream and the fire occupy the same band"
        assert max(centroids.values()) > min(centroids.values()) * 3, \
            f"the beds are barely distinguishable: {centroids}"

    def test_the_thunderstorm_has_more_low_end_than_plain_rain(self):
        def sub_energy(bed: str) -> float:
            mono = ae.build_seed(bed, "none", seconds=4.0, seed=0).mean(axis=1)
            power = np.abs(np.fft.rfft(mono)) ** 2
            freqs = np.fft.rfftfreq(len(mono), 1.0 / ae.SAMPLE_RATE)
            return float(power[freqs < 150].sum() / power.sum())

        assert sub_energy("thunderstorm") > sub_energy("rain_window") * 1.5

    @pytest.mark.parametrize("texture,carrier", [("drone_432", 432.0), ("drone_528", 528.0)])
    def test_the_binaural_drone_differs_between_the_ears(self, texture, carrier):
        """The entire effect is the difference between the channels. Summed to
        mono it does not exist, so a mono implementation is a silent failure."""
        audio = ae.build_seed("brown_noise", texture, bed_volume=0.0,
                              texture_volume=1.0, seconds=8.0, seed=0)

        def peak_hz(signal: np.ndarray) -> float:
            spec = np.abs(np.fft.rfft(signal))
            return float(np.fft.rfftfreq(len(signal), 1.0 / ae.SAMPLE_RATE)[np.argmax(spec)])

        left, right = peak_hz(audio[:, 0]), peak_hz(audio[:, 1])
        assert float(np.abs(audio[:, 0] - audio[:, 1]).max()) > 0.1, "identical channels"
        assert abs(abs(right - left) - ae.BINAURAL_OFFSET_HZ) < 1.0, \
            f"beat is {abs(right - left):.2f}Hz, wanted {ae.BINAURAL_OFFSET_HZ}"
        assert abs((left + right) / 2 - carrier) < 2.0

    @pytest.mark.parametrize("bed", list(ae.PRIMARY_BEDS))
    def test_the_seed_loops_against_itself(self, bed):
        """The join from the last sample back to the first must not be the
        biggest step in the buffer -- that step is what a click is."""
        mono = ae.build_seed(bed, "none", seconds=8.0, seed=0).mean(axis=1)
        wrap = float(abs(mono[0] - mono[-1]))
        worst_interior = float(np.abs(np.diff(mono)).max())
        assert wrap < worst_interior, (
            f"{bed}: the wrap jump ({wrap:.5f}) is larger than any interior step "
            f"({worst_interior:.5f})")


class TestSoundtrackChain:

    @pytest.mark.slow
    def test_seeds_are_chained_not_repeated(self, workdir):
        """One seed on repeat for eight hours is audibly a loop within ten
        minutes, and is the exact pattern reused-content review flags here."""
        result = ae.render_soundtrack("rain_window", "distant_thunder", 120.0, workdir)
        assert result["variants"] == ae.SEED_VARIANTS
        assert result["loop_seconds"] > ae.SEED_SECONDS * 1.5, (
            f"the super-loop is {result['loop_seconds']:.0f}s — the seeds were "
            "not chained")

    @pytest.mark.slow
    def test_the_soundtrack_is_the_length_asked_for(self, workdir):
        result = ae.render_soundtrack("brown_noise", "none", 90.0, workdir)
        assert abs(result["duration"] - 90.0) < 1.0, result["duration"]

    @pytest.mark.slow
    def test_the_soundtrack_is_not_silent(self, workdir):
        """The check a seam test cannot do for itself: a NaN upstream produces
        a file that is the right length, the right format, and empty."""
        result = ae.render_soundtrack("thunderstorm", "distant_thunder", 90.0, workdir)
        mono = _decode_mono(result["path"])
        assert len(mono) > 0
        window = 22050
        rms = np.sqrt((mono[: len(mono) // window * window]
                       .reshape(-1, window) ** 2).mean(axis=1))
        assert rms.min() > 0.005, f"there is a silent second in the mix (min {rms.min():.5f})"

    @pytest.mark.slow
    def test_the_loop_point_does_not_click(self, workdir):
        result = ae.render_soundtrack("rain_window", "room_wind", 300.0, workdir)
        seam = ae.verify_seam(result["path"], result["loop_seconds"])

        assert seam["reference_jump"] > 1e-4, (
            "the decoded audio is effectively silent, so the seam measurement "
            "means nothing")
        assert seam["ratio"] < 3.0, (
            f"the loop point is a {seam['ratio']:.1f}x larger step than ordinary "
            "signal — that is an audible click, every loop, all night")


# ---------------------------------------------------------------------------
# Visual canvas
# ---------------------------------------------------------------------------

class TestCanvasPresets:

    @pytest.mark.parametrize("name", list(ae.CANVAS_PRESETS))
    def test_presets_are_dark_enough_to_sleep_through(self, name, workdir):
        path = ae.build_canvas_preset(name, workdir, seed=5)
        arr = np.asarray(Image.open(path).convert("L"), np.float32)
        assert arr.mean() < 90, f"{name} has mean luma {arr.mean():.0f} — too bright"

    @pytest.mark.parametrize("seed", [0, 3, 5, 11, 17, 23, 31])
    def test_presets_draw_on_every_seed(self, seed, workdir):
        """The star loop once indexed `canvas[y - radius:...]` with y as small
        as 2 and radius up to 3. numpy reads a negative start as "from the
        end", so it silently produced an empty slice and a broadcast error --
        on some seeds and not others."""
        for name in ae.CANVAS_PRESETS:
            path = ae.build_canvas_preset(name, workdir, seed=seed)
            assert os.path.exists(path)

    @pytest.mark.parametrize("name", list(ae.CANVAS_PRESETS))
    def test_presets_are_bigger_than_the_output(self, name, workdir):
        """The zoom resamples from this, so it needs more pixels than 1080p."""
        path = ae.build_canvas_preset(name, workdir, seed=5)
        width, height = Image.open(path).size
        assert width > ae.CANVAS[0] and height > ae.CANVAS[1]

    def test_the_night_sky_has_visible_stars(self, workdir):
        """Single-pixel stars land between output pixels and shimmer on and off
        as the frame drifts, which is the one thing a sleep video must not do."""
        path = ae.build_canvas_preset("night_sky", workdir, seed=5)
        arr = np.asarray(Image.open(path).convert("L"), np.float32)
        assert (arr > 120).sum() > 200, "no visible stars"

        # Downscaled to output size they must survive, i.e. be more than 1px.
        small = np.asarray(Image.open(path).convert("L")
                           .resize(ae.CANVAS, Image.Resampling.LANCZOS), np.float32)
        assert (small > 90).sum() > 100, "the stars vanish at output resolution"


class TestVisualDrift:

    @pytest.mark.slow
    def test_the_cycle_returns_to_where_it_started(self, workdir, ffmpeg):
        """The cycle segment is stream-copied for hours. If it does not land
        back on 1.00x there is a visible jump every single cycle."""
        from moviepy import VideoFileClip

        src = _grid_image(os.path.join(workdir, "grid.png"))
        result = ae.build_visual(src, 20.0, workdir, fps=24, drift="cycle",
                                 grain=0.0, vignette=False, ffmpeg=ffmpeg)
        with VideoFileClip(result["path"]) as clip:
            frames = int(round(clip.duration * 24))
            base = _zoom_of(np.asarray(clip.get_frame(0.0), np.float32))
            mid = _zoom_of(np.asarray(clip.get_frame(240 / 24), np.float32)) / base
            end = _zoom_of(np.asarray(clip.get_frame((frames - 1) / 24), np.float32)) / base

        assert mid > 1.045, f"the cycle only reached {mid:.4f}x, not 1.05x"
        assert abs(end - 1.0) < 0.004, f"the cycle ended at {end:.4f}x, not 1.0x"

    @pytest.mark.slow
    def test_continuous_drift_pushes_all_the_way(self, workdir, ffmpeg):
        from moviepy import VideoFileClip

        src = _grid_image(os.path.join(workdir, "grid.png"))
        result = ae.build_visual(src, 20.0, workdir, fps=24, drift="continuous",
                                 grain=0.0, vignette=False, ffmpeg=ffmpeg)
        with VideoFileClip(result["path"]) as clip:
            frames = int(round(clip.duration * 24))
            base = _zoom_of(np.asarray(clip.get_frame(0.0), np.float32))
            end = _zoom_of(np.asarray(clip.get_frame((frames - 1) / 24), np.float32)) / base
        assert end > 1.045, f"continuous drift only reached {end:.4f}x"

    @pytest.mark.slow
    def test_a_long_run_is_looped_rather_than_re_encoded(self, workdir, ffmpeg):
        """Beyond one cycle the segment is stream-copied, which is the whole
        reason an eight-hour render takes minutes instead of hours."""
        src = _grid_image(os.path.join(workdir, "grid.png"))
        result = ae.build_visual(src, ae.CYCLE_SECONDS + 30.0, workdir, fps=24,
                                 drift="cycle", grain=0.0, vignette=False,
                                 ffmpeg=ffmpeg)
        assert result["looped"] is True
        assert result["segment_seconds"] == ae.CYCLE_SECONDS

    def test_the_vignette_is_baked_into_the_canvas(self, workdir):
        """ffmpeg's vignette is a per-frame float multiply and was 60% of the
        render: 195 fps without it, 82 with. Baking it costs one PIL pass."""
        src = _grid_image(os.path.join(workdir, "grid.png"))
        canvas = ae.prepare_still(src, workdir, vignette=True)
        arr = np.asarray(Image.open(canvas).convert("L"), np.float32)
        h, w = arr.shape
        centre = arr[h // 2 - 40:h // 2 + 40, w // 2 - 40:w // 2 + 40].mean()
        corner = arr[:80, :80].mean()
        assert corner < centre * 0.8, "the vignette was not baked in"

        plain = ae.prepare_still(src, workdir, vignette=False)
        flat = np.asarray(Image.open(plain).convert("L"), np.float32)
        assert flat[:80, :80].mean() > corner, "vignette=False still darkened the corners"


class TestEncoder:

    def test_the_gpu_profile_is_the_one_specified(self):
        flags = " ".join(ae.GPU_FLAGS)
        for expected in ("h264_nvenc", "-preset p4", "-tune hq", "-b:v 4M", "-bufsize 8M"):
            assert expected in flags, f"{expected} is missing from {flags}"

    def test_there_is_a_cpu_fallback(self):
        flags = ae.encoder_flags(force_cpu=True)
        assert "libx264" in flags and "h264_nvenc" not in flags
        assert "yuv420p" in flags, "every phone decoder needs yuv420p"


# ---------------------------------------------------------------------------
# The whole render
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_test_render_produces_a_playable_file(workdir):
    """The 60s preset, end to end: 16:9 H.264 with continuous stereo audio."""
    import imageio_ffmpeg

    src = ae.build_canvas_preset("night_sky", workdir, seed=3)
    out = os.path.join(workdir, "atmosphere.mp4")
    result = ae.render_atmosphere(
        bed="rain_window", texture="distant_thunder", visual_source=src,
        duration_key="test", output_path=out, workspace=workdir,
        fps=24, drift="cycle",
    )
    assert os.path.exists(out) and result["size_bytes"] > 1_000_000

    proc = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", out],
                          capture_output=True, text=True)
    assert "Video: h264" in proc.stderr
    assert "1920x1080" in proc.stderr, "not 16:9 1080p"
    assert "Audio: aac" in proc.stderr and "stereo" in proc.stderr

    mono = _decode_mono(out)
    window = 22050
    rms = np.sqrt((mono[: len(mono) // window * window].reshape(-1, window) ** 2).mean(axis=1))
    assert rms.min() > 0.005, "a silent second in the finished render"
    assert abs(result["duration"] - 60.0) < 0.5


def test_the_duration_presets_cover_what_was_asked_for():
    seconds = {k: v["seconds"] for k, v in ae.DURATIONS.items()}
    assert seconds["test"] == 60
    assert seconds["30min"] == 1800
    assert seconds["1hour"] == 3600
    assert seconds["3hours"] == 10800
    assert seconds["8hours"] == 28800


def test_the_estimate_is_roughly_right_for_a_long_render():
    """An eight-hour cycle render must be minutes, not hours -- that is the
    whole point of looping a segment instead of rendering every frame."""
    cycle = ae.estimate_render_seconds(8 * 3600, "cycle")
    continuous = ae.estimate_render_seconds(8 * 3600, "continuous")
    assert cycle < 400, f"{cycle:.0f}s for an 8h cycle render"
    assert continuous > cycle * 20, "continuous should be dramatically slower"


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------

class TestPublisherCredentials:

    def test_a_service_account_key_is_rejected(self, monkeypatch, workdir):
        monkeypatch.setattr(pub, "SECRETS_DIR", workdir)
        monkeypatch.setattr(pub, "CLIENT_SECRETS_PATH",
                            os.path.join(workdir, "client_secrets.json"))
        with pytest.raises(pub.PublishError, match="service account"):
            pub.save_client_secrets('{"type":"service_account","project_id":"x"}')

    def test_malformed_json_is_rejected(self, monkeypatch, workdir):
        monkeypatch.setattr(pub, "SECRETS_DIR", workdir)
        monkeypatch.setattr(pub, "CLIENT_SECRETS_PATH",
                            os.path.join(workdir, "client_secrets.json"))
        with pytest.raises(pub.PublishError, match="valid JSON"):
            pub.save_client_secrets("not json at all")

    def test_a_desktop_client_is_accepted(self, monkeypatch, workdir):
        monkeypatch.setattr(pub, "SECRETS_DIR", workdir)
        monkeypatch.setattr(pub, "CLIENT_SECRETS_PATH",
                            os.path.join(workdir, "client_secrets.json"))
        path = pub.save_client_secrets(
            '{"installed":{"client_id":"x","client_secret":"y",'
            '"auth_uri":"https://a","token_uri":"https://t"}}')
        assert os.path.exists(path)

    def test_status_reports_nothing_when_nothing_is_configured(self, monkeypatch, workdir):
        monkeypatch.setattr(pub, "CLIENT_SECRETS_PATH", os.path.join(workdir, "nope.json"))
        monkeypatch.setattr(pub, "TOKEN_PATH", os.path.join(workdir, "nope_token.json"))
        status = pub.account_status()
        assert status == {"client_secrets": False, "token": False, "connected": False,
                          "channel": "", "channel_id": "", "error": ""}

    def test_the_scope_is_upload_only(self):
        """The full youtube scope would also grant read/write over comments,
        playlists and the channel. This only ever inserts a video."""
        assert pub.SCOPES == ["https://www.googleapis.com/auth/youtube.upload"]


class TestPublisherMetadata:

    @pytest.mark.parametrize("title,description,tags,expect", [
        ("", "", [], "title is required"),
        ("x" * 120, "", [], "is 120 characters"),
        ("Rain <Sounds>", "", [], "Angle brackets"),
        ("Rain", "d" * 6000, [], "description"),
        ("Rain", "", ["tag" * 60] * 5, "Tags total"),
    ])
    def test_bad_metadata_is_caught_before_the_upload(self, title, description, tags, expect):
        problems = pub.validate_metadata(title, description, tags)
        assert problems, f"{expect} was not caught"
        assert any(expect.lower() in p.lower() or expect in p for p in problems), problems

    def test_good_metadata_passes(self):
        assert pub.validate_metadata(
            "Rain Sounds for Sleeping | 8 Hours", "A description.",
            ["rain sounds", "sleep"]) == []

    def test_a_scheduled_upload_is_forced_private(self):
        """publishAt is only honoured when privacyStatus is private. Sent
        alongside 'public', YouTube ignores the date and publishes at once."""
        body = pub.build_body("T", "D", [], privacy="scheduled",
                              publish_at="2026-12-01T20:00:00Z")
        assert body["status"]["privacyStatus"] == "private"
        assert body["status"]["publishAt"] == "2026-12-01T20:00:00Z"

    def test_an_unscheduled_upload_carries_no_publish_at(self):
        for privacy in ("private", "unlisted", "public"):
            body = pub.build_body("T", "D", [], privacy=privacy)
            assert "publishAt" not in body["status"]
            assert body["status"]["privacyStatus"] == privacy

    def test_scheduling_without_a_date_is_refused(self):
        with pytest.raises(pub.PublishError):
            pub.build_body("T", "D", [], privacy="scheduled")

    def test_the_default_privacy_is_private(self):
        """Nothing goes public by accident."""
        assert pub.DEFAULT_PRIVACY == "private"
        assert pub.build_body("T", "D", [])["status"]["privacyStatus"] == "private"

    def test_naive_datetimes_are_converted_from_local_time(self):
        """A naive datetime is what a date widget hands over, and it means
        local time. Stamping it with Z would schedule it in UTC instead."""
        import datetime as dt

        naive = dt.datetime(2026, 12, 1, 20, 0, 0)
        expected = naive.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert pub.to_rfc3339(naive) == expected
        aware = dt.datetime(2026, 12, 1, 20, 0, 0, tzinfo=dt.timezone.utc)
        assert pub.to_rfc3339(aware) == "2026-12-01T20:00:00Z"

    def test_tags_are_deduplicated_case_insensitively(self):
        assert pub.parse_tags("Rain, sleep , SLEEP,, rain\ninsomnia") == \
            ["Rain", "sleep", "insomnia"]

    def test_the_offline_seo_passes_its_own_validator(self):
        for seconds in (1800, 3600, 8 * 3600):
            meta = pub.fallback_seo("Rain on Window", seconds)
            assert pub.validate_metadata(meta["title"], meta["description"],
                                         meta["tags"]) == []

    def test_the_body_is_truncated_to_the_api_limits(self):
        body = pub.build_body("x" * 300, "d" * 9000, [], category_id="27")
        assert len(body["snippet"]["title"]) == pub.TITLE_LIMIT
        assert len(body["snippet"]["description"]) == pub.DESCRIPTION_LIMIT
        assert body["snippet"]["categoryId"] == "27"


class TestPublisherSecrets:

    def test_the_secrets_directory_is_gitignored(self):
        """A refresh token can upload to the channel until it is revoked."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for candidate in (".secrets/client_secrets.json", ".secrets/youtube_token.json"):
            proc = subprocess.run(["git", "check-ignore", "-q", candidate],
                                  cwd=root, capture_output=True)
            assert proc.returncode == 0, f"{candidate} is NOT gitignored"

    def test_nothing_logs_token_material(self):
        """A token in a log or a traceback is a token in a bug report."""
        source = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "publisher.py"), encoding="utf-8").read()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"'):
                continue
            if "print(" in stripped or "progress(" in stripped:
                assert "token" not in stripped.lower() or "token_path" in stripped, \
                    f"possible token leak: {stripped}"
