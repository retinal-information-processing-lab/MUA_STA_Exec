"""The ``.mcd`` path against real recordings, with their ``.raw`` exports as ground truth.

``MC_DataTool``'s ``.raw`` is nothing but the concatenation of the ``.mcd``'s electrode
payloads, so for a recording that exists in both forms every number the tool computes must
come out the same whichever one it was given. These two have both, and ``verify_raw()`` says
they are byte-identical, which makes them the best test available for the reader.

Skipped where the recordings are not mounted, so the suite still runs anywhere. The full pass
over the trigger channel reads the whole 19 GB and is kept behind ``MUA_STA_SLOW_TESTS=1``.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from mua_sta import recording as rec

EXPERIMENT = Path("/mnt/10To/B13/20261103_GadB13_ReaChR_bis")
NAMES = ["00_AccCheck_40Hz_16px_42sq_40%35ND", "01_Swn_30Hz_6pxShift_40%35ND"]

pytestmark = pytest.mark.skipif(
    not EXPERIMENT.is_dir(),
    reason=f"{EXPERIMENT} is not mounted: no recording to compare against")


def pair(name):
    mcd_path = EXPERIMENT / "recordings" / f"{name}.mcd"
    raw_path = EXPERIMENT / "RAW_Files" / f"{name}.raw"
    if not (mcd_path.is_file() and raw_path.is_file()):
        pytest.skip(f"{name}: needs both the .mcd and its .raw export")
    return mcd_path, raw_path


@pytest.mark.parametrize("name", NAMES)
def test_the_raw_really_is_a_byte_for_byte_export(name):
    """Checks the premise of every other test here before relying on it."""
    from mua_sta import mcd
    mcd_path, raw_path = pair(name)
    with mcd.McdFile(mcd_path) as m:
        report = m.verify_raw(raw_path, n_checks=8)
    assert report["ok"], report


@pytest.mark.parametrize("name", NAMES)
def test_the_header_agrees_with_the_raw_the_tool_was_told_about(name):
    """The channel count and rate have to be supplied for a ``.raw`` and are read for an
    ``.mcd``; the two must land on the same recording."""
    mcd_path, raw_path = pair(name)
    from_mcd = rec.open_info(mcd_path)
    from_raw = rec.open_info(raw_path)
    assert from_mcd.format == rec.MCD and from_raw.format == rec.RAW
    assert from_mcd.n_channels == from_raw.n_channels == rec.DEFAULT_N_CHANNELS
    assert from_mcd.fs == from_raw.fs == rec.DEFAULT_FS
    assert from_mcd.n_samples == from_raw.n_samples
    assert from_mcd.uv_per_level == pytest.approx(from_raw.uv_per_level)


@pytest.mark.parametrize("name", NAMES)
def test_spikeinterface_reads_the_same_samples_from_either(name):
    mcd_path, raw_path = pair(name)
    from_mcd = rec.as_spikeinterface(rec.open_info(mcd_path))
    from_raw = rec.as_spikeinterface(rec.open_info(raw_path))
    n_samples = from_raw.get_num_samples()
    assert from_mcd.get_num_samples() == n_samples
    rng = np.random.default_rng(0)
    spans = [(0, 20_000), (n_samples - 20_000, n_samples)]
    spans += [(int(s), int(s) + 20_000)
              for s in rng.integers(0, n_samples - 20_000, size=6)]
    for start, stop in spans:
        np.testing.assert_array_equal(
            from_mcd.get_traces(start_frame=start, end_frame=stop),
            from_raw.get_traces(start_frame=start, end_frame=stop))


@pytest.mark.parametrize("name", NAMES)
def test_the_trigger_channel_agrees_on_sampled_spans(name):
    """What the trigger detection sees, spot-checked without reading 19 GB."""
    mcd_path, raw_path = pair(name)
    mcd_info = rec.open_info(mcd_path)
    channel = rec.DEFAULT_TRIGGER_CHANNEL
    mm = np.memmap(raw_path, dtype=mcd_info.dtype, mode="r")
    try:
        frames = mm[:mcd_info.n_samples * mcd_info.n_channels].reshape(
            mcd_info.n_samples, mcd_info.n_channels)
        from mua_sta import mcd
        with mcd.McdFile(mcd_path) as m:
            rng = np.random.default_rng(1)
            for start in [0, mcd_info.n_samples - 50_000] + \
                    [int(s) for s in rng.integers(0, mcd_info.n_samples - 50_000, size=4)]:
                stop = start + 50_000
                np.testing.assert_array_equal(
                    m.read(start, stop, channels=[channel])[:, 0],
                    frames[start:stop, channel])
    finally:
        mm._mmap.close()


@pytest.mark.skipif(not os.environ.get("MUA_STA_SLOW_TESTS"),
                    reason="reads the whole recording twice; set MUA_STA_SLOW_TESTS=1")
@pytest.mark.parametrize("name", NAMES)
def test_the_whole_trigger_channel_and_its_onsets_are_identical(name):
    """The pass the tool actually makes, and the triggers that come out of it."""
    from mua_sta import triggers as tg
    from mua_sta.rigs import rig
    mcd_path, raw_path = pair(name)
    mcd_info, raw_info = rec.open_info(mcd_path), rec.open_info(raw_path)
    from_mcd = rec.read_channel(mcd_info, rec.DEFAULT_TRIGGER_CHANNEL)
    from_raw = rec.read_channel(raw_info, rec.DEFAULT_TRIGGER_CHANNEL)
    np.testing.assert_array_equal(from_mcd, from_raw)

    threshold = rig(3).trigger_threshold
    onsets = [tg.sanity_check(tg.detect_onsets(trace, threshold), mcd_info.fs, 0.25e-3)
              for trace in (from_mcd, from_raw)]
    np.testing.assert_array_equal(onsets[0].onsets, onsets[1].onsets)
    assert onsets[0].count > 0
