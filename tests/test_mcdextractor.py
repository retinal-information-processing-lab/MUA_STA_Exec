"""The ``.mcd`` as spikeinterface sees it, and the detection that runs on top of it.

Two things have to hold, and neither is obvious from reading the code:

* the recording survives being **pickled**, because the worker pool is started with ``spawn``
  on Windows and that is how the recording reaches a worker. The open file handle must not be
  part of what is pickled;
* reading from several **threads** at once gives each its own data, because ``McdFile.read``
  seeks a shared handle and ``get_best_job_kwargs()`` picks threads on Windows.

Then the real question: detection on an ``.mcd`` must find exactly what detection on the
``.raw`` of the same samples finds.
"""
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mua_sta import recording as rec
from mua_sta import spikes as sp

import mcdbuilder

FS, N_CHANNELS, N_SAMPLES = 20_000.0, 8, 40_000


@pytest.fixture
def signal():
    """Noise with spikes planted on two channels, as ``test_spikes.py`` plants them."""
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 20, size=(N_SAMPLES, N_CHANNELS))
    for channel in (2, 5):
        for at in np.arange(500, N_SAMPLES - 500, 400):
            signal[at:at + 6, channel] -= 600 * np.hanning(6)
    return (signal + mcdbuilder.ADC_ZERO).astype("<u2")


@pytest.fixture
def pair(tmp_path, signal):
    """The same samples as an ``.mcd`` and as the ``.raw`` MC_DataTool would export from it."""
    mcd_path = mcdbuilder.write_mcd(tmp_path / "rec.mcd", signal, fs=FS, block_samples=2_000)
    raw_path = tmp_path / "rec.raw"
    raw_path.write_bytes(signal.tobytes())
    return (rec.open_info(mcd_path, n_channels=N_CHANNELS, fs=FS),
            rec.open_info(raw_path, n_channels=N_CHANNELS, fs=FS))


def test_the_two_formats_describe_the_same_recording(pair):
    mcd_info, raw_info = pair
    assert (mcd_info.n_channels, mcd_info.fs, mcd_info.n_samples) == \
           (raw_info.n_channels, raw_info.fs, raw_info.n_samples)
    assert mcd_info.format == rec.MCD and raw_info.format == rec.RAW


def test_as_spikeinterface_matches_the_raw_exactly(pair):
    mcd_info, raw_info = pair
    from_mcd = rec.as_spikeinterface(mcd_info)
    from_raw = rec.as_spikeinterface(raw_info)
    assert from_mcd.get_num_samples() == from_raw.get_num_samples()
    assert from_mcd.get_num_channels() == from_raw.get_num_channels()
    assert from_mcd.get_dtype() == from_raw.get_dtype()          # int16, after the shift
    np.testing.assert_array_equal(from_mcd.channel_ids, from_raw.channel_ids)
    rng = np.random.default_rng(0)
    for _ in range(5):
        start = int(rng.integers(0, N_SAMPLES - 1000))
        np.testing.assert_array_equal(
            from_mcd.get_traces(start_frame=start, end_frame=start + 1000),
            from_raw.get_traces(start_frame=start, end_frame=start + 1000))
    np.testing.assert_array_equal(
        from_mcd.get_traces(start_frame=0, end_frame=100, channel_ids=[0, 3, 7]),
        from_raw.get_traces(start_frame=0, end_frame=100, channel_ids=[0, 3, 7]))


def test_it_survives_pickling(pair):
    """What ``spawn`` does to it on Windows. ``BaseExtractor.__reduce__`` rebuilds from
    ``_kwargs``, so the open reader must not be in there — and ``McdFile`` cannot be pickled
    at all, which is why this matters."""
    mcd_info, _ = pair
    recording = rec.as_spikeinterface(mcd_info)
    assert recording.check_serializability("memory")
    again = pickle.loads(pickle.dumps(recording))
    np.testing.assert_array_equal(again.get_traces(start_frame=0, end_frame=2000),
                                  recording.get_traces(start_frame=0, end_frame=2000))


def test_the_reader_itself_is_not_picklable(pair):
    """Stated as a test because it is the reason the extractor is built the way it is: if this
    ever starts passing, the care taken in ``mcdextractor`` can be reconsidered."""
    from mua_sta import mcd
    mcd_info, _ = pair
    with mcd.McdFile(mcd_info.path) as m:
        with pytest.raises(TypeError, match="pickle"):
            pickle.dumps(m)


def test_several_threads_read_their_own_data(pair):
    """One reader per thread. A shared handle would hand a thread the span another asked for."""
    mcd_info, raw_info = pair
    from_mcd = rec.as_spikeinterface(mcd_info)
    from_raw = rec.as_spikeinterface(raw_info)
    spans = [(0, 4_000), (10_000, 14_000), (20_000, 24_000), (36_000, 40_000)] * 3

    def read(span):
        return from_mcd.get_traces(start_frame=span[0], end_frame=span[1])

    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(read, spans))
    for (start, stop), traces in zip(spans, got):
        np.testing.assert_array_equal(
            traces, from_raw.get_traces(start_frame=start, end_frame=stop))


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_detection_finds_the_same_spikes_in_both_formats(pair, n_jobs):
    """The whole point: the same recording, read either way, gives the same MUA. With
    ``n_jobs=2`` this also runs the extractor through the worker pool."""
    mcd_info, raw_info = pair
    from_mcd = sp.detect_mua(mcd_info, n_jobs=n_jobs)
    from_raw = sp.detect_mua(raw_info, n_jobs=n_jobs)
    np.testing.assert_array_equal(from_mcd.labels, from_raw.labels)
    np.testing.assert_array_equal(from_mcd.offsets, from_raw.offsets)
    np.testing.assert_array_equal(from_mcd.times, from_raw.times)
    assert {3, 6} <= set(from_mcd.labels.tolist())      # channels 2 and 5, planted above
