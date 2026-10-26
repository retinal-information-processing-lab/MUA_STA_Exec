"""Grouping detected peaks into one multi-unit train per electrode."""
import numpy as np
import pytest

from mua_sta import recording as rec
from mua_sta import spikes as sp


def test_peaks_are_grouped_under_the_right_electrode_label():
    """spikeinterface's channel_index is 0-based; electrode labels are 1-based.

    The previous version of this tool used the index as a label, so every spike train was
    attributed to the next electrode along. This is the regression test for that.
    """
    sample_index = np.array([10, 20, 30, 40, 50])
    channel_index = np.array([0, 2, 0, 2, 5])
    trains = sp.from_peaks(sample_index, channel_index, fs=10.0)

    np.testing.assert_array_equal(trains.labels, [1, 3, 6])
    np.testing.assert_allclose(trains.get(1), [1.0, 3.0])
    np.testing.assert_allclose(trains.get(3), [2.0, 4.0])
    np.testing.assert_allclose(trains.get(6), [5.0])
    assert trains.get(2).size == 0                      # channel 1 had no peaks
    assert trains.total == 5


def test_times_stay_in_order_within_an_electrode():
    rng = np.random.default_rng(0)
    sample_index = np.sort(rng.integers(0, 10_000, 500))
    channel_index = rng.integers(0, 8, 500)
    trains = sp.from_peaks(sample_index, channel_index, fs=20_000.0)
    for label in trains.labels:
        times = trains.get(int(label))
        assert (np.diff(times) >= 0).all()
    assert trains.total == 500


def test_membership_and_counts():
    trains = sp.from_peaks(np.array([1, 2, 3]), np.array([0, 0, 4]), fs=1.0)
    assert 1 in trains and 5 in trains and 2 not in trains
    assert trains.counts() == {1: 2, 5: 1}
    assert "3 spikes on 2 electrodes" in trains.describe()


def test_no_peaks_at_all():
    trains = sp.from_peaks(np.array([], dtype=int), np.array([], dtype=int), fs=1.0)
    assert trains.total == 0 and trains.labels.size == 0
    assert trains.get(1).size == 0


def test_every_mapped_label_round_trips_through_a_channel_index():
    labels = rec.electrode_labels(rec.load_mapping())
    indices = np.array([rec.label_to_channel_index(label) for label in labels])
    trains = sp.from_peaks(np.arange(indices.size), indices, fs=1.0)
    np.testing.assert_array_equal(trains.labels, sorted(labels))


@pytest.mark.parametrize("peak_sign", ["neg", "pos"])
def test_detect_mua_on_a_small_recording(tmp_path, peak_sign):
    """The real detector, on a short recording with planted spikes on two channels."""
    fs, n_channels, n_samples = 20_000.0, 8, 40_000
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 20, size=(n_samples, n_channels))
    planted = np.arange(500, n_samples - 500, 400)
    sign = -1 if peak_sign == "neg" else 1
    for channel in (2, 5):
        for at in planted:
            signal[at:at + 6, channel] += sign * 600 * np.hanning(6)
    raw = tmp_path / "rec.raw"
    raw.write_bytes((signal + 32768).astype("<u2").tobytes())

    info = rec.open_info(raw, n_channels=n_channels, fs=fs)
    trains = sp.detect_mua(info, detect_threshold=6.0, peak_sign=peak_sign, n_jobs=1)

    # labels 3 and 6 are channel indices 2 and 5
    assert {3, 6} <= set(trains.labels.tolist())
    for label in (3, 6):
        assert trains.get(label).size >= 0.8 * planted.size
    quiet = [int(label) for label in trains.labels if label not in (3, 6)]
    for label in quiet:
        assert trains.get(label).size < 0.2 * planted.size


def test_detecting_in_pieces_gives_the_same_spikes(tmp_path, monkeypatch):
    """Detection is cut up only so a run can be stopped; it must not change what is found.

    The noise levels are measured once over the whole recording and passed to every piece, so
    the thresholds are identical. What is left is spikeinterface's own chunking, whose
    boundaries move when the recording it is given is shorter: that costs about one spike in a
    few thousand, and none at all on a real 200 s window cut into four.
    """
    fs, n_channels, n_samples = 20_000.0, 6, 300_000
    rng = np.random.default_rng(3)
    signal = rng.normal(0, 20, size=(n_samples, n_channels))
    for channel in (1, 4):
        for at in range(300, n_samples - 300, 250):
            signal[at:at + 6, channel] -= 600 * np.hanning(6)
    raw = tmp_path / "rec.raw"
    raw.write_bytes((signal + 32768).astype("<u2").tobytes())
    info = rec.open_info(raw, n_channels=n_channels, fs=fs)

    monkeypatch.setattr(sp, "SLICE_SECONDS", 1e9)       # one piece
    whole = sp.detect_mua(info, n_jobs=1)
    monkeypatch.setattr(sp, "SLICE_SECONDS", 3.0)       # five pieces
    assert len(sp._slice_bounds(n_samples, fs)) > 1
    pieces = sp.detect_mua(info, n_jobs=1)

    np.testing.assert_array_equal(pieces.labels, whole.labels)
    assert abs(pieces.total - whole.total) <= max(2, 0.001 * whole.total)
    for label in whole.labels:
        a, b = whole.get(int(label)), pieces.get(int(label))
        shared = np.intersect1d(np.round(a, 5), np.round(b, 5)).size
        most = max(a.size, b.size)
        # a spike in a thousand, and at least one, for the channels that barely fire
        assert most - shared <= max(1, 0.001 * most), int(label)


def test_slice_bounds_cover_the_recording_exactly():
    bounds = sp._slice_bounds(36_976_000, 20_000.0)
    assert bounds[0][0] == 0 and bounds[-1][1] == 36_976_000
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))     # no gap, no overlap
    assert 1 < len(bounds) <= sp.MAX_SLICES
    assert sp._slice_bounds(1000, 20_000.0) == [(0, 1000)]           # a short one is not cut
