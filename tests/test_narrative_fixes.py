"""
Narrative Studio: the silence that became runtime, and the loader under it.

An episode rendered at 655 seconds from a 131-word script -- in sync, correctly
muxed, with the first subtitle held for thirty-six of them. Every stage checked
out in isolation: the storyboard came to 63s, ken_burns_clip was exact to the
frame, concat_segments was exact to the frame. The input was wrong. The picture
is cut to cover the voice, and the voice was mostly silence.

Underneath that was a worse one. `_load_mono` asked MoviePy to resample while
decoding, and on a 44.1kHz file with a true peak of 0.85 the 16kHz read came
back peaking at 0.163 -- and on another file, at nothing at all. Everything
that looks for silence compares against an absolute threshold, so the trim that
was supposed to catch this had been finding nothing to trim, in this engine and
in the vector one.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import audio_engine as ae
import narrative_engine as ne


@pytest.fixture()
def padded(tmp_path):
    """Two seconds of real signal inside three and a half of silence each side."""
    stereo, rate = ae.generate_synth_music(style="lofi", duration=2.0)
    pad = np.zeros((int(3.5 * rate), 2), dtype=stereo.dtype)
    path = str(tmp_path / "padded.wav")
    ae.save_wav_to_file(np.concatenate([pad, stereo, pad]), rate, path)
    return path


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------

class TestMonoLoader:

    def test_amplitude_survives_the_resample(self, tmp_path):
        """MoviePy's own resample attenuated a 0.85 peak to 0.163, which reads
        as silence to every threshold downstream."""
        stereo, rate = ae.generate_synth_music(style="lofi", duration=2.0)
        path = str(tmp_path / "plain.wav")
        ae.save_wav_to_file(stereo, rate, path)

        loaded = ae._load_mono(path, 16000)
        assert float(np.abs(loaded).max()) > 0.5, "the signal came back quiet"

    def test_the_timebase_survives_it_too(self, tmp_path):
        """Striding would keep the amplitude and land on 22050 for a 44.1kHz
        file asked for 16000 -- and every caller divides by the rate it asked
        for, so offsets would come out 38% long."""
        stereo, rate = ae.generate_synth_music(style="lofi", duration=2.0)
        path = str(tmp_path / "plain.wav")
        ae.save_wav_to_file(stereo, rate, path)

        loaded = ae._load_mono(path, 16000)
        assert len(loaded) / 16000.0 == pytest.approx(2.0, abs=0.05)

    def test_a_rate_the_file_already_has_is_left_alone(self, tmp_path):
        stereo, rate = ae.generate_synth_music(style="lofi", duration=1.0)
        path = str(tmp_path / "plain.wav")
        ae.save_wav_to_file(stereo, rate, path)
        assert len(ae._load_mono(path, rate)) == pytest.approx(rate, rel=0.02)


# ---------------------------------------------------------------------------
# The trim
# ---------------------------------------------------------------------------

class TestTrimSilence:

    def test_it_removes_silence_from_both_ends(self, padded):
        out = ae.trim_silence(padded, output_path=padded.replace(".wav", "_t.wav"))
        assert out["trimmed_head"] == pytest.approx(3.5, abs=0.3)
        assert out["trimmed_tail"] == pytest.approx(3.5, abs=0.3)
        assert out["duration"] == pytest.approx(2.0, abs=0.3)

    def test_the_tail_is_the_half_that_matters_here(self, padded):
        """Leading silence wastes an opening beat. Trailing silence becomes
        runtime, because the picture is cut to cover the voice."""
        out = ae.trim_silence(padded, output_path=padded.replace(".wav", "_t.wav"))
        assert out["trimmed_tail"] > 1.0

    def test_a_silent_take_is_left_alone(self, tmp_path):
        """Trimming it to nothing would be worse than leaving it: the caller
        can see a zero-length take and say so."""
        path = str(tmp_path / "silent.wav")
        ae.save_wav_to_file(np.zeros((44100 * 2, 2), np.float32), 44100, path)
        out = ae.trim_silence(path, output_path=str(tmp_path / "out.wav"))
        assert out["path"] == path
        assert out["trimmed_head"] == 0.0 and out["trimmed_tail"] == 0.0

    def test_a_take_that_starts_and_ends_on_signal_keeps_its_length(self, tmp_path):
        """Not "is not rewritten": a synth bed fades in and out, so there is
        always a little to take off the ends. What must not happen is losing
        real content."""
        stereo, rate = ae.generate_synth_music(style="lofi", duration=2.0)
        path = str(tmp_path / "tight.wav")
        ae.save_wav_to_file(stereo, rate, path)
        out = ae.trim_silence(path, output_path=str(tmp_path / "out.wav"))
        assert out["duration"] > 1.5, "real content was trimmed away"

    def test_an_unreadable_file_does_not_raise(self, tmp_path):
        path = str(tmp_path / "broken.wav")
        open(path, "wb").write(b"not audio")
        assert ae.trim_silence(path)["path"] == path

    def test_the_leading_trim_works_again_too(self, padded):
        """Same loader, same bug: the vector engine's opening trim had been
        finding nothing either."""
        out = ae.trim_leading_silence(padded, output_path=padded.replace(".wav", "_l.wav"))
        assert out["trimmed"] == pytest.approx(3.5, abs=0.3)


# ---------------------------------------------------------------------------
# The floor under the whole thing
# ---------------------------------------------------------------------------

class TestNarrationPlausibility:

    def test_there_is_a_floor(self):
        assert 0 < ne.MIN_PLAUSIBLE_WORDS_PER_SECOND < ne.WORDS_PER_SECOND

    def test_the_floor_leaves_room_for_a_slow_reading(self):
        """Gemini TTS measures about 2.0 words a second on this material. The
        floor has to be wide enough that only a broken take trips it."""
        assert ne.MIN_PLAUSIBLE_WORDS_PER_SECOND <= 1.2

    def test_the_check_is_wired_into_production(self):
        import inspect

        source = inspect.getsource(ne.produce_episode)
        assert "MIN_PLAUSIBLE_WORDS_PER_SECOND" in source
        assert "trim_silence" in source

    def test_a_655_second_take_would_now_be_refused(self):
        """131 words at the floor is 131 seconds. A 655-second take is not a
        reading of that script, and the storyboard's own timing is used."""
        words = 131
        assert words / ne.MIN_PLAUSIBLE_WORDS_PER_SECOND < 655.0


# ---------------------------------------------------------------------------
# The ending
# ---------------------------------------------------------------------------

class TestEpisodeEnding:

    def test_there_is_an_end_fade(self):
        assert ne.END_FADE_SECONDS > 0

    def test_picture_and_sound_go_out_together(self):
        import inspect

        source = inspect.getsource(ne._fade_tail)
        assert "fade=t=out" in source and "afade=t=out" in source

    def test_a_short_episode_is_not_mostly_fade(self):
        import inspect

        assert "total * 0.25" in inspect.getsource(ne._fade_tail)

    def test_a_failed_fade_leaves_the_episode_alone(self):
        """An episode that ends abruptly is worse. An episode that is gone is
        a lost render."""
        import inspect

        source = inspect.getsource(ne._fade_tail)
        assert "except Exception" in source
        assert "shutil.move" in source

    def test_the_episode_is_levelled_like_every_other_render(self):
        import inspect

        assert "normalise_loudness" in inspect.getsource(ne.produce_episode)


# ---------------------------------------------------------------------------
# The stages that were never broken
# ---------------------------------------------------------------------------

class TestTimingStagesAreExact:
    """
    Pinned because they were each suspected and each cleared. The next person
    reading the 655-second story should not have to re-measure them.
    """

    def test_the_storyboard_lands_inside_its_format(self):
        board = [{"index": i, "line": "a line of narration here", "duration": 5.0,
                  "start": i * 5.0, "words": 5} for i in range(11)]
        assert ne.storyboard_runtime(board) == pytest.approx(55.0)

    def test_ken_burns_is_exact(self, tmp_path):
        from PIL import Image

        import video_engine as ve

        still = str(tmp_path / "s.png")
        Image.new("RGB", (540, 960), (40, 44, 52)).save(still)
        out = str(tmp_path / "k.mp4")
        ne.ken_burns_clip(still, out, 3.0, (540, 960), fps=24, move="in_center")
        assert float(ve.probe_stream_info(out).get("duration") or 0) == pytest.approx(
            3.0, abs=0.12)

    def test_the_mux_trims_to_the_shorter_stream(self):
        import inspect

        assert '"-shortest"' in inspect.getsource(ne._mux)
