"""The vectorised STA against the straightforward loop, and the whitening it enables."""
import numpy as np
import pytest

from mua_sta import sta as sta_mod


def reference_sta(counts, stimulus, temporal_dimension):
    """The algorithm as the standard pipeline writes it: add one slice per spike bin.

    Kept here on purpose. It is the readable statement of what an STA is, and the fast version
    in the package is only worth having if it agrees with it exactly.
    """
    n_sequences, n_frames = counts.shape
    out = np.zeros((temporal_dimension, *stimulus.shape[1:]), dtype=np.float64)
    for sequence in range(n_sequences):
        for frame in range(temporal_dimension, n_frames):
            start = sequence * n_frames + frame - temporal_dimension
            out += counts[sequence, frame] * stimulus[start:start + temporal_dimension]
    total = counts.sum()
    if total > 0 and np.max(np.abs(out)) > 0:
        out /= total
        out -= np.median(out)
        out /= np.max(np.abs(out))
    return out


@pytest.mark.parametrize("temporal_dimension", [1, 4, 9])
def test_matches_the_reference_loop(temporal_dimension):
    rng = np.random.default_rng(0)
    n_elec, n_sequences, n_frames, rows, cols = 5, 3, 20, 4, 6
    stimulus = (rng.random((n_sequences * n_frames, rows, cols)) > 0.5).astype(np.uint8) * 255
    counts = rng.poisson(0.4, size=(n_elec, n_sequences, n_frames)).astype(np.float32)

    fast = sta_mod.compute_sta(counts, stimulus, temporal_dimension)
    for i in range(n_elec):
        slow = reference_sta(counts[i], stimulus.astype(np.float32) / 255.0, temporal_dimension)
        np.testing.assert_allclose(fast[i], slow, atol=2e-5)


def test_silent_electrode_is_zero():
    stimulus = np.zeros((20, 3, 3), dtype=np.uint8)
    counts = np.zeros((2, 2, 10), dtype=np.float32)
    assert not sta_mod.compute_sta(counts, stimulus, 3).any()


def test_lag_order_is_most_recent_last():
    """Lag T-1 is the frame just before the spike; lag 0 is T frames earlier."""
    temporal_dimension, n_frames = 4, 10
    stimulus = np.zeros((n_frames, 2, 2), dtype=np.uint8)
    stimulus[5] = 255                       # one bright frame
    counts = np.zeros((1, 1, n_frames), dtype=np.float32)
    counts[0, 0, 6] = 1.0                   # one spike in the frame right after it
    result = sta_mod.compute_sta(counts, stimulus, temporal_dimension)[0]
    peak_lag = int(np.argmax(np.abs(result).reshape(temporal_dimension, -1).max(axis=1)))
    assert peak_lag == temporal_dimension - 1


def test_peak_components_pick_the_strongest_lag():
    rng = np.random.default_rng(1)
    volume = rng.normal(size=(3, 6, 5, 7)).astype(np.float32) * 0.01
    volume[1, 4, 2, 3] = 1.0
    spatial, temporal, peak = sta_mod.peak_components(volume)
    assert tuple(peak[1]) == (4, 2, 3)
    assert spatial.shape == (3, 5, 7) and temporal.shape == (3, 6)
    np.testing.assert_allclose(np.max(np.abs(spatial), axis=(1, 2)), 1.0)


def test_binning_uses_triggers_as_edges():
    """One bin per displayed frame, and the last frame of a sequence is not dropped."""
    frames_per_sequence, n_sequences = 8, 2
    triggers = np.arange(frames_per_sequence * n_sequences + 1, dtype=float) * 0.1
    edges = sta_mod.sequence_edges(triggers, n_sequences, frames_per_sequence,
                                   sta_mod.REPEATED_HALF)
    assert edges.shape == (n_sequences, frames_per_sequence // 2 + 1)
    # a spike inside the very last frame of the last sequence must be counted
    spikes = np.array([edges[-1, -1] - 0.01])
    assert sta_mod.count_spikes(spikes, edges)[-1, -1] == 1


def test_sequence_edges_extrapolate_a_missing_final_trigger():
    frames_per_sequence = 4
    triggers = np.arange(frames_per_sequence, dtype=float) * 0.1      # exactly one sequence
    edges = sta_mod.sequence_edges(triggers, 1, frames_per_sequence, (0.0, 1.0))
    assert edges.shape == (1, frames_per_sequence + 1)
    np.testing.assert_allclose(edges[0, -1], 0.4)


def shifting_noise(rng, n_checks: int, shifts: int, n_frames: int) -> np.ndarray:
    """A miniature shifting white noise: a coarse random grid offset by whole shift steps.

    This is what makes the covariance the whitening inverts: two cells ``d`` steps apart land
    in the same check with probability ``1 - d/shifts``, so the correlation falls linearly to
    zero at one full check. A stimulus built of fixed blocks instead (no shifting) would have a
    singular covariance — every cell of a block identical — and nothing to recover.
    """
    n_cells = (n_checks - 1) * shifts
    frames = np.empty((n_frames, n_cells, n_cells), dtype=bool)
    coarse = rng.random((n_frames, n_checks + 1, n_checks + 1)) > 0.5
    fine = np.repeat(np.repeat(coarse, shifts, axis=1), shifts, axis=2)
    offsets = rng.integers(0, shifts, size=(n_frames, 2))
    for i, (oy, ox) in enumerate(offsets):
        frames[i] = fine[i, oy:oy + n_cells, ox:ox + n_cells]
    return (frames * 255).astype(np.uint8)


def test_shifting_noise_has_the_expected_tent_correlation():
    """Guards the fixture above: the correlation must fall as 1 - d/shifts, zero at one check."""
    shifts = 4
    stimulus = shifting_noise(np.random.default_rng(3), n_checks=6, shifts=shifts, n_frames=8000)
    x = stimulus.astype(np.float32) / 255.0
    x -= x.mean()
    for lag in range(1, shifts + 1):
        got = float((x[:, :, :-lag] * x[:, :, lag:]).mean() / x.var())
        assert abs(got - (1 - lag / shifts)) < 0.05


def test_whitening_sharpens_a_blurred_receptive_field():
    """A correlated stimulus blurs the STA; the inverse covariance sharpens it back."""
    rng = np.random.default_rng(2)
    shifts, n_frames = 4, 30000
    stimulus = shifting_noise(rng, n_checks=6, shifts=shifts, n_frames=n_frames)
    n_cells = stimulus.shape[1]

    true_rf = np.zeros((n_cells, n_cells), dtype=np.float32)
    true_rf[9, 11] = 1.0
    drive = (stimulus.reshape(n_frames, -1) / 255.0) @ true_rf.ravel()
    counts = np.zeros((1, 1, n_frames), dtype=np.float32)
    counts[0, 0, 1:] = drive[:-1] > 0.5      # the cell drives the NEXT frame's spikes, so
                                             # lag 0 of a one-frame STA is where it shows up
    raw = sta_mod.compute_sta(counts, stimulus, temporal_dimension=1)[0, 0]
    cov = sta_mod.stimulus_covariance(stimulus)
    sharp = sta_mod.whiten(raw[None], cov, sigma_fraction=0.01)[0]

    assert np.unravel_index(np.argmax(np.abs(raw)), raw.shape) == (9, 11)
    assert np.unravel_index(np.argmax(np.abs(sharp)), sharp.shape) == (9, 11)
    # The raw STA spreads over the whole check around the peak; the whitened one does not.
    ring = (slice(9 - shifts + 1, 9 + shifts), slice(11 - shifts + 1, 11 + shifts))
    def spread(image):
        mask = np.ones(image.shape, dtype=bool)
        mask[ring] = False
        neighbourhood = np.abs(image[ring]).sum() - np.abs(image[9, 11])
        return neighbourhood / np.abs(image[9, 11])
    assert spread(raw) > 5 * spread(sharp)


def test_stitch_puts_electrode_row_zero_first_so_plus_y_is_up():
    """The grid and the tiles share one convention: canvas row index increases upwards.

    Drawn with y up (pyqtgraph's default, or origin="lower"), electrode row 0 is at the bottom
    — matching the standard pipeline's receptive-field frame. The previous tool's grid was
    mirrored against it.
    """
    mapping = {1: (0, 0), 2: (15, 15), 3: (0, 15)}
    labels = np.array([1, 2, 3])
    spatial = np.zeros((3, 2, 2), dtype=np.float32)
    spatial[0] = 1.0        # electrode at row 0, col 0
    spatial[1] = -1.0       # electrode at row 15, col 15
    grid = sta_mod.stitch(spatial, labels, mapping, grid_size=16, padding=3)

    assert grid.image.shape == (16 * 2 + 15 * 3, 16 * 2 + 15 * 3)
    assert grid.image[0, 0] == 1.0                      # row 0 is the first block
    assert grid.image[-1, -1] == -1.0                   # row 15 is the last
    assert np.isnan(grid.image[2, 0])                   # the gap between tiles


def test_stitch_locates_the_electrode_under_a_point():
    mapping = {label: (label // 16, label % 16) for label in range(256)}
    spatial = np.zeros((256, 4, 4), dtype=np.float32)
    grid = sta_mod.stitch(spatial, np.arange(256), mapping, padding=3)
    assert grid.electrode_at(0.0, 0.0, mapping) == 0
    assert grid.electrode_at(4 + 3 + 1.0, 0.0, mapping) == 1        # next tile along x
    assert grid.electrode_at(0.0, 4 + 3 + 1.0, mapping) == 16       # next tile up
    assert grid.electrode_at(4.5, 0.0, mapping) is None             # in the gap


# ------------------------------------------------------------------- orientation check
def inverse_registration(name):
    """The op that undoes ``name``, found by trying all eight rather than by composing them."""
    from mua_sta.rigs import ARRAY_REGISTRATIONS
    sample = np.arange(20.0).reshape(4, 5)
    forward = ARRAY_REGISTRATIONS[name](sample)
    for candidate, op in ARRAY_REGISTRATIONS.items():
        undone = op(forward)
        if undone.shape == sample.shape and np.array_equal(undone, sample):
            return ARRAY_REGISTRATIONS[candidate]
    raise AssertionError(f"no inverse for {name}")


def retinotopic_maps(registration, n_cells=40, grid=16, seed=0):
    """Maps in stimulus coordinates such that ``registration`` puts each blob on its electrode.

    This is the situation the real data is in: the receptive fields are laid out like the array,
    but in the stimulus's own axes, which sit on the array some rotation or mirror away.
    """
    rng = np.random.default_rng(seed)
    mapping = {label: (label // grid, label % grid) for label in range(grid * grid)}
    labels = np.array(sorted(mapping))
    scale = n_cells / (2 * grid)
    offset = n_cells / 4
    y, x = np.mgrid[0:n_cells, 0:n_cells]
    in_array_frame = np.empty((labels.size, n_cells, n_cells), dtype=np.float32)
    for i, label in enumerate(labels):
        row, col = mapping[label]
        cy, cx = offset + row * scale, offset + col * scale
        in_array_frame[i] = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 8.0)
    in_array_frame += rng.normal(0, 0.02, in_array_frame.shape)
    undo = inverse_registration(registration)
    return np.ascontiguousarray(undo(in_array_frame)), labels, mapping


@pytest.mark.parametrize("registration", [
    "identity", "flip_y", "flip_x", "flip_xy",
    "transpose", "transpose_flip_y", "transpose_flip_x", "transpose_flip_xy",
])
def test_measure_registration_recovers_the_right_one(registration):
    spatial, labels, mapping = retinotopic_maps(registration)
    scores = sta_mod.measure_registration(spatial, labels, mapping)
    assert scores[0][0] == registration
    assert scores[0][1] > 0.9
    assert scores[0][1] - scores[1][1] > 0.3        # a clear winner, not a coin toss


def test_measure_registration_finds_nothing_in_noise():
    """With no receptive fields there is nothing to measure, and it must not pretend there is."""
    rng = np.random.default_rng(1)
    spatial = rng.normal(size=(60, 20, 20)).astype(np.float32)
    labels = np.arange(60)
    mapping = {label: (label // 16, label % 16) for label in range(60)}
    scores = sta_mod.measure_registration(spatial, labels, mapping)
    assert max(abs(score) for _, score in scores) < 0.5


def test_rf_quality_tells_a_receptive_field_from_noise():
    rng = np.random.default_rng(2)
    y, x = np.mgrid[0:30, 0:30]
    blob = np.exp(-((y - 15) ** 2 + (x - 15) ** 2) / 6.0).astype(np.float32)
    noise = rng.normal(0, 0.3, (30, 30)).astype(np.float32)
    assert sta_mod.rf_quality(blob) > 10 * sta_mod.rf_quality(noise)
    stacked = sta_mod.rf_quality(np.stack([blob, noise]))
    assert stacked.shape == (2,) and stacked[0] > stacked[1]


def test_field_bounds_find_the_window_the_fields_live_in():
    spatial, labels, mapping = retinotopic_maps("identity", n_cells=80)
    bounds = sta_mod.field_bounds(spatial)
    assert bounds is not None
    r0, r1, c0, c1 = bounds
    assert 0 <= r0 < r1 <= 80 and 0 <= c0 < c1 <= 80
    assert (r1 - r0) < 80 and (c1 - c0) < 80            # it really did narrow the window
    # every good electrode's peak is inside the window
    quality = sta_mod.rf_quality(spatial)
    good = quality > np.median(quality)
    flat = np.abs(spatial[good]).reshape(int(good.sum()), -1)
    peak = np.array(np.unravel_index(flat.argmax(axis=1), (80, 80))).T
    assert (peak[:, 0] >= r0).all() and (peak[:, 0] < r1).all()
    assert (peak[:, 1] >= c0).all() and (peak[:, 1] < c1).all()


def test_field_bounds_give_up_on_noise():
    """With peaks scattered everywhere there is no window to crop to, so do not invent one."""
    rng = np.random.default_rng(4)
    assert sta_mod.field_bounds(rng.normal(size=(60, 40, 40)).astype(np.float32)) is None


def test_stitch_crops_every_tile_the_same_way():
    mapping = {1: (0, 0), 2: (1, 1)}
    spatial = np.zeros((2, 10, 10), dtype=np.float32)
    spatial[0, 4, 5] = 1.0
    spatial[1, 5, 6] = 1.0
    grid = sta_mod.stitch(spatial, np.array([1, 2]), mapping, grid_size=2, padding=1,
                          crop=(3, 7, 4, 8))
    assert grid.tile == (4, 4)
    assert grid.image.shape == (2 * 4 + 1, 2 * 4 + 1)
    assert grid.image[1, 1] == 1.0                      # (4,5) is (1,1) inside the crop
    assert grid.electrode_at(0.5, 0.5, mapping) == 1


def test_the_temporal_time_vector_matches_the_pipeline():
    """One bin per displayed frame, ending at the spike — the same values the standard
    pipeline's get_temporal_sta_time_vector produces, written so the count is never off."""
    for depth, frequency in ((30, 30.0), (40, 30.0), (30, 40.0), (17, 59.94)):
        mine = (np.arange(depth) - (depth - 1)) / frequency
        pipeline = np.flip(np.arange(0, -depth * (1 / frequency), -(1 / frequency)))
        assert mine.size == depth
        np.testing.assert_allclose(mine, pipeline[:depth], atol=1e-12)
        assert mine[-1] == 0.0
