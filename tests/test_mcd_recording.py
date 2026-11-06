"""Reading an ``.mcd`` the way the tool reads one, on a file small enough to build here.

The companion to ``test_recording.py``, which does the same for a ``.raw``. Both fixtures hold
the same pattern — channel c, sample s is ``1000*c + s`` — so the two formats can be asserted
to give the same numbers.
"""
import numpy as np
import pytest

from mua_sta import recording as rec

import mcdbuilder

N_CHANNELS, N_SAMPLES, BLOCK = 8, 50, 20       # 50 = 20 + 20 + 10: a short last block


@pytest.fixture
def frames():
    return mcdbuilder.ramp_frames(N_SAMPLES, N_CHANNELS)


@pytest.fixture
def mcd_file(tmp_path, frames):
    return mcdbuilder.write_mcd(tmp_path / "rec.mcd", frames, block_samples=BLOCK)


def open_mcd(path):
    """``open_info`` told what this fixture holds, so only the tests that are about the
    header disagreeing with its caller have to deal with that warning."""
    return rec.open_info(path, n_channels=N_CHANNELS, dtype="uint16", fs=20_000.0)


@pytest.fixture
def raw_file(tmp_path, frames):
    path = tmp_path / "rec.raw"
    path.write_bytes(frames.tobytes())
    return path


def test_the_format_is_judged_by_content_not_by_name(tmp_path, mcd_file, raw_file):
    """A recording is recognised by the ``MCSSTRM`` magic, so a renamed one still is, and a
    ``.raw`` that happens to be called ``.mcd`` is not mistaken for one."""
    assert rec.detect_format(mcd_file) == rec.MCD
    assert rec.detect_format(raw_file) == rec.RAW

    disguised = tmp_path / "recording.bak"
    disguised.write_bytes(mcd_file.read_bytes())
    assert rec.detect_format(disguised) == rec.MCD

    liar = tmp_path / "not-really.mcd"
    liar.write_bytes(raw_file.read_bytes())
    assert rec.detect_format(liar) == rec.RAW


def test_open_info_takes_everything_from_the_header(mcd_file):
    """Nothing about an ``.mcd`` has to be supplied: the header states all of it, and it is
    what gets used even where the caller passed the ``.raw`` defaults."""
    with pytest.warns(UserWarning, match="header says 8"):
        info = rec.open_info(mcd_file)
    assert info.format == rec.MCD
    assert info.n_channels == N_CHANNELS            # not the 256 of the default
    assert info.n_samples == N_SAMPLES
    assert info.fs == 20_000.0
    assert np.dtype(info.dtype) == np.dtype("uint16")
    assert info.uv_per_level == pytest.approx(mcdbuilder.UV_PER_LEVEL)
    assert "mcd" in info.describe() and "8 ch" in info.describe()


def test_open_info_warns_rather_than_fails_when_told_otherwise(mcd_file):
    with pytest.warns(UserWarning, match="channel count 256"):
        info = rec.open_info(mcd_file, n_channels=256, fs=25_000.0)
    assert (info.n_channels, info.fs) == (N_CHANNELS, 20_000.0)


def test_a_header_that_agrees_does_not_warn(mcd_file):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rec.open_info(mcd_file, n_channels=N_CHANNELS, dtype="uint16", fs=20_000.0)


def test_read_channel_gives_the_same_values_as_the_raw_does(mcd_file, raw_file, frames):
    mcd_info = open_mcd(mcd_file)
    raw_info = rec.open_info(raw_file, n_channels=N_CHANNELS)
    for channel in (0, 3, N_CHANNELS - 1):
        from_mcd = rec.read_channel(mcd_info, channel)
        np.testing.assert_array_equal(from_mcd, frames[:, channel])
        np.testing.assert_array_equal(from_mcd, rec.read_channel(raw_info, channel))


def test_read_channel_reports_progress_the_same_way_the_raw_path_does(mcd_file):
    """Reported on the ``chunk_samples`` scale, not once per block: a 36-minute recording has
    21 000 blocks and the GUI is not called 21 000 times."""
    info = open_mcd(mcd_file)
    seen = []
    rec.read_channel(info, 2, progress=lambda *a: seen.append(a), chunk_samples=20)
    assert len(seen) > 1
    assert seen[-1] == (N_SAMPLES, N_SAMPLES, rec.READING_TRIGGER)
    assert all(total == N_SAMPLES for _, total, _ in seen)

    once = []
    rec.read_channel(info, 2, progress=lambda *a: once.append(a))
    assert once == [(N_SAMPLES, N_SAMPLES, rec.READING_TRIGGER)]


def test_read_channel_rejects_a_channel_out_of_range(mcd_file):
    info = open_mcd(mcd_file)
    with pytest.raises(ValueError, match="outside"):
        rec.read_channel(info, N_CHANNELS)


def test_the_electrode_stream_is_chosen_over_the_digital_one(tmp_path, frames):
    """A real recording carries a one-channel ``digi0001`` stream as well, and the electrode
    stream is the one with the channels on it."""
    from mua_sta import mcd
    path = mcdbuilder.write_mcd(tmp_path / "two.mcd", frames, block_samples=BLOCK)
    with mcd.McdFile(path) as m:
        assert [s.chunk_tag for s in m.streams] == [b"digi0001", b"elec0001"]
        assert m.stream().chunk_tag == b"elec0001"
    assert open_mcd(path).n_channels == N_CHANNELS


def test_a_file_without_the_digital_stream_reads_the_same(tmp_path, frames):
    path = mcdbuilder.write_mcd(tmp_path / "solo.mcd", frames, block_samples=BLOCK,
                                with_digital=False)
    info = open_mcd(path)
    assert info.n_channels == N_CHANNELS and info.n_samples == N_SAMPLES
    np.testing.assert_array_equal(rec.read_channel(info, 5), frames[:, 5])


@pytest.mark.parametrize("block_samples", [1, 7, 25, 50, 64])
def test_any_block_size_reads_back_whole(tmp_path, frames, block_samples):
    """Whether the blocks divide the recording evenly or leave a short one at the end."""
    path = mcdbuilder.write_mcd(tmp_path / f"b{block_samples}.mcd", frames,
                                block_samples=block_samples)
    info = open_mcd(path)
    assert info.n_samples == N_SAMPLES
    np.testing.assert_array_equal(rec.read_channel(info, 1), frames[:, 1])


def test_the_mapping_labels_still_line_up_with_the_channel_order(tmp_path):
    """Channel *i* is MCS label *i + 1* in an ``.mcd`` as in a ``.raw``: the labels are in the
    header, in file order, and getting this wrong moves every result one electrode."""
    from mua_sta import mcd
    frames = mcdbuilder.ramp_frames(4, 256)
    path = mcdbuilder.write_mcd(tmp_path / "full.mcd", frames, block_samples=4)
    with mcd.McdFile(path) as m:
        labels = m.stream().labels
    assert labels[0] == "01" and labels[125] == "126" and labels[255] == "256"
    assert [rec.channel_index_to_label(i) for i in (0, 125, 255)] == [1, 126, 256]
    assert rec.label_to_channel_index(127) == rec.DEFAULT_TRIGGER_CHANNEL
