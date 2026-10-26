"""The staged cache: what counts as stale, and that an entry survives a round trip."""
import numpy as np
import pytest

from mua_sta import cache as ca


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "rec.raw"
    path.write_bytes(b"\x00" * 64)
    return path


def test_key_ignores_argument_order():
    assert ca.key(a=1, b="x") == ca.key(b="x", a=1)
    assert ca.key(a=1) != ca.key(a=2)


def test_recording_id_separates_same_name_in_different_folders(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    for folder in (first, second):
        folder.mkdir()
        (folder / "rec.raw").write_bytes(b"")
    assert ca.recording_id(first / "rec.raw") != ca.recording_id(second / "rec.raw")


def test_file_identity_changes_when_the_file_does(recording):
    before = ca.file_identity(recording)
    recording.write_bytes(b"\x00" * 128)
    assert ca.file_identity(recording) != before


def test_store_then_load(tmp_path, recording):
    cache = ca.Cache.for_recording(recording, root=tmp_path / "cache")
    k = ca.key(threshold=17714)
    assert cache.status(ca.TRIGGERS, k) is ca.Status.MISSING
    cache.store(ca.TRIGGERS, k, meta={"count": 3}, onsets=np.arange(3))
    assert cache.status(ca.TRIGGERS, k) is ca.Status.CACHED
    loaded = cache.load(ca.TRIGGERS, k)
    np.testing.assert_array_equal(loaded["onsets"], np.arange(3))
    assert loaded["__meta__"] == {"count": 3}
    assert cache.size(ca.TRIGGERS, k) > 0


def test_a_changed_parameter_makes_a_recording_step_stale(tmp_path, recording):
    """The point of the whole design: the GUI can see what has to run again, and what not."""
    cache = ca.Cache.for_recording(recording, root=tmp_path / "cache")
    cache.store(ca.TRIGGERS, ca.key(threshold=17714), onsets=np.arange(3))
    assert cache.status(ca.TRIGGERS, ca.key(threshold=15630)) is ca.Status.STALE
    assert cache.load(ca.TRIGGERS, ca.key(threshold=15630)) is None


def test_a_content_addressed_step_keeps_both_variants(tmp_path, recording):
    """Switching stimulus must not throw away the other one: rebuilding SWN costs minutes."""
    cache = ca.Cache.for_recording(recording, root=tmp_path / "cache")
    checker, swn = ca.key(kind="checkerboard"), ca.key(kind="swn")
    cache.store(ca.STIMULUS, checker, frames=np.zeros((2, 2, 2), dtype=np.uint8))
    cache.store(ca.STIMULUS, swn, frames=np.ones((2, 2, 2), dtype=np.uint8))
    assert cache.status(ca.STIMULUS, checker) is ca.Status.CACHED
    assert cache.status(ca.STIMULUS, swn) is ca.Status.CACHED
    assert cache.load(ca.STIMULUS, checker)["frames"].max() == 0


def test_drop_and_clear(tmp_path, recording):
    cache = ca.Cache.for_recording(recording, root=tmp_path / "cache")
    k = ca.key(a=1)
    cache.store(ca.TRACE, k, levels=np.zeros(4, dtype=np.uint16))
    cache.store(ca.STIMULUS, k, frames=np.zeros((1, 1, 1), dtype=np.uint8))
    cache.drop(ca.TRACE, k)
    assert cache.status(ca.TRACE, k) is ca.Status.MISSING
    assert cache.status(ca.STIMULUS, k) is ca.Status.CACHED   # stimuli are not recording-scoped
    cache.store(ca.TRACE, k, levels=np.zeros(4, dtype=np.uint16))
    cache.clear_recording()
    assert cache.status(ca.TRACE, k) is ca.Status.MISSING
    assert cache.status(ca.STIMULUS, k) is ca.Status.CACHED
    cache.clear_all()
    assert cache.status(ca.STIMULUS, k) is ca.Status.MISSING


def test_a_half_written_entry_is_not_left_behind(tmp_path, recording):
    """Entries are written to a temporary name and moved into place."""
    cache = ca.Cache.for_recording(recording, root=tmp_path / "cache")
    k = ca.key(a=1)
    cache.store(ca.TRACE, k, levels=np.zeros(4, dtype=np.uint16))
    assert not list((tmp_path / "cache").rglob("*.part"))


def test_a_corrupt_entry_reads_as_missing_rather_than_crashing(tmp_path, recording):
    cache = ca.Cache.for_recording(recording, root=tmp_path / "cache")
    k = ca.key(a=1)
    path = cache.store(ca.TRACE, k, levels=np.zeros(4, dtype=np.uint16))
    path.write_bytes(b"not an npz")
    assert cache.status(ca.TRACE, k) is ca.Status.STALE
