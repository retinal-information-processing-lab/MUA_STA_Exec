"""Reading a recording, and which channel of it is which electrode."""
import numpy as np
import pytest

from mua_sta import recording as rec


@pytest.fixture
def raw(tmp_path):
    """A small interleaved recording: channel c, sample s holds 1000*c + s."""
    n_channels, n_samples = 8, 50
    frames = (np.arange(n_channels)[None, :] * 1000
              + np.arange(n_samples)[:, None]).astype("<u2")
    path = tmp_path / "rec.raw"
    path.write_bytes(frames.tobytes())
    return path, n_channels, n_samples


def test_open_info(raw):
    path, n_channels, n_samples = raw
    info = rec.open_info(path, n_channels=n_channels, dtype="uint16", fs=20000.0)
    assert info.n_samples == n_samples
    assert info.duration_s == pytest.approx(n_samples / 20000.0)
    assert "8 ch" in info.describe()


def test_open_info_warns_on_the_wrong_channel_count(raw):
    path, n_channels, _ = raw
    with pytest.warns(UserWarning, match="channel count"):
        rec.open_info(path, n_channels=n_channels + 3)


def test_read_channel_is_strided_not_sequential(raw):
    """The trigger channel is read off a memory map; nothing else in the file is touched."""
    path, n_channels, n_samples = raw
    info = rec.open_info(path, n_channels=n_channels)
    for channel in (0, 3, n_channels - 1):
        got = rec.read_channel(info, channel)
        np.testing.assert_array_equal(got, channel * 1000 + np.arange(n_samples))


def test_read_channel_reports_progress_and_chunks_cleanly(raw):
    """Progress is reported the way tqdm counts: how many of how many, and what is running."""
    path, n_channels, n_samples = raw
    info = rec.open_info(path, n_channels=n_channels)
    seen = []
    got = rec.read_channel(info, 2, progress=lambda *a: seen.append(a), chunk_samples=7)
    np.testing.assert_array_equal(got, 2000 + np.arange(n_samples))
    assert len(seen) > 1
    assert seen[-1] == (n_samples, n_samples, "reading the trigger channel")
    assert all(total == n_samples for _, total, _ in seen)


def test_read_channel_rejects_a_channel_out_of_range(raw):
    path, n_channels, _ = raw
    info = rec.open_info(path, n_channels=n_channels)
    with pytest.raises(ValueError, match="outside"):
        rec.read_channel(info, n_channels)


def test_to_microvolts_is_centred_on_the_adc_zero():
    levels = np.array([32768, 32768 + 100], dtype=np.uint16)
    got = rec.to_microvolts(levels, uv_per_level=0.1)
    np.testing.assert_allclose(got, [0.0, 10.0])


def test_mapping_is_1_based_labels_to_0_based_grid():
    mapping = rec.load_mapping()
    assert len(mapping) == rec.GRID_SIZE ** 2
    assert min(mapping) == 1 and max(mapping) == rec.GRID_SIZE ** 2
    rows = {r for r, _ in mapping.values()}
    cols = {c for _, c in mapping.values()}
    assert rows == cols == set(range(rec.GRID_SIZE))
    assert len(set(mapping.values())) == rec.GRID_SIZE ** 2     # one electrode per position


def test_electrode_labels_leave_out_the_reference_and_trigger_channels():
    labels = rec.electrode_labels(rec.load_mapping())
    assert len(labels) == rec.GRID_SIZE ** 2 - len(rec.EXCLUDED_LABELS)
    assert not set(labels) & rec.EXCLUDED_LABELS


def test_label_and_channel_index_differ_by_one():
    """The bug this replaces: the previous tool used the 0-based index as a 1-based label, so
    every result sat one electrode away from where it belonged."""
    assert rec.label_to_channel_index(1) == 0
    assert rec.channel_index_to_label(0) == 1
    assert rec.label_to_channel_index(rec.DEFAULT_TRIGGER_CHANNEL + 1) == rec.DEFAULT_TRIGGER_CHANNEL
    for label in rec.load_mapping():
        assert rec.channel_index_to_label(rec.label_to_channel_index(label)) == label
