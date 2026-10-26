"""Binning spikes against the stimulus, then the spike-triggered average.

Two departures from the scripts this replaces, both of which matter:

* **Binning uses the triggers themselves as bin edges.** The previous code laid 600 bins over
  the span from the first to the last trigger of a half-sequence, which is 599 frame intervals,
  so every bin was 599/600 of a frame wide and bin *k* drifted from frame *k* by up to half a
  frame by the end of the sequence; the last frame's spikes were dropped as well. Counting
  between consecutive triggers is exact and simpler.
* **The STA is a matrix product, not a Python loop.** The old version added a whole
  ``(T, rows, cols)`` slice of the stimulus per spike bin, for each electrode in turn — about
  6 million numpy calls for one recording. Here each sequence contributes one matrix product
  per lag, for all electrodes at once. Same arithmetic (``tests/test_sta.py`` checks it against
  the straightforward loop), a couple of orders of magnitude less overhead.

Shifting white noise also needs **whitening**. Its frames are spatially correlated — with
``s`` shift steps per check, two cells ``d`` steps apart share a check with probability
``1 - d/s`` — so the raw STA is the receptive field blurred by the stimulus's own covariance.
The estimate that removes it is ``C^-1 STA``, with ``C`` the covariance of the stimulus
ensemble. ``C`` is a property of the stimulus, identical for every electrode, so it is
factorised once and reused: exactly the same result as solving per electrode, hundreds of
times faster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve

#: Portions of a sequence: the novel half drives the STA, the repeated half the rasters.
NOVEL_HALF = (0.0, 0.5)
REPEATED_HALF = (0.5, 1.0)

DEFAULT_TEMPORAL_DIMENSION = 30
DEFAULT_FRAMES_PER_SEQUENCE = 1200
#: Regularisation added to the covariance diagonal, as a fraction of the mean pixel variance.
DEFAULT_SIGMA_FRACTION = 0.05


def sequence_edges(
    trigger_times: np.ndarray,
    n_sequences: int,
    frames_per_sequence: int,
    portion: tuple[float, float],
) -> np.ndarray:
    """Bin edges for one portion of every sequence: ``(n_sequences, n_frames + 1)``.

    The edges are trigger times, so a bin is exactly one displayed frame. The final edge of
    the last sequence may fall past the last trigger (a recording stopped right at the end),
    in which case it is extrapolated by one median frame interval.
    """
    first = int(portion[0] * frames_per_sequence)
    last = int(portion[1] * frames_per_sequence)
    n_frames = last - first
    starts = np.arange(n_sequences) * frames_per_sequence + first
    index = starts[:, None] + np.arange(n_frames + 1)[None, :]
    if index.max() >= trigger_times.size:
        frame_s = float(np.median(np.diff(trigger_times)))
        padded = np.append(trigger_times, trigger_times[-1] + frame_s * np.arange(
            1, index.max() - trigger_times.size + 2))
        return padded[index]
    return trigger_times[index]


def count_spikes(spike_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Spikes of one electrode per frame: ``(n_sequences, n_frames)`` from ``(n_sequences, n+1)``."""
    positions = np.searchsorted(spike_times, edges.ravel()).reshape(edges.shape)
    return np.diff(positions, axis=1).astype(np.float32)


def bin_all(
    trains, labels: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Spike counts for every electrode: ``(n_electrodes, n_sequences, n_frames)``."""
    counts = np.empty((labels.size, edges.shape[0], edges.shape[1] - 1), dtype=np.float32)
    for i, label in enumerate(labels):
        counts[i] = count_spikes(trains.get(int(label)), edges)
    return counts


def raster_trains(
    trains, labels: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Spike times of the repeated half, relative to each sequence's start.

    Flat (``times``, ``offsets``) so it caches without pickling: entry ``i * n_sequences + s``
    is electrode ``labels[i]`` on sequence ``s``.
    """
    pieces: list[np.ndarray] = []
    for label in labels:
        spikes = trains.get(int(label))
        for sequence in range(edges.shape[0]):
            start, stop = edges[sequence, 0], edges[sequence, -1]
            lo, hi = np.searchsorted(spikes, (start, stop))
            pieces.append((spikes[lo:hi] - start).astype(np.float32))
    offsets = np.append(0, np.cumsum([p.size for p in pieces])).astype(np.int64)
    times = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
    return times, offsets


# ------------------------------------------------------------------------------ the STA
def compute_sta(
    counts: np.ndarray,
    stimulus: np.ndarray,
    temporal_dimension: int = DEFAULT_TEMPORAL_DIMENSION,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Spike-triggered average of every electrode: ``(n_electrodes, T, rows, cols)``.

    ``counts`` is ``(n_electrodes, n_sequences, n_frames)`` over the novel half of each
    sequence and ``stimulus`` is ``(n_sequences * n_frames, rows, cols)`` uint8, the novel
    frames stacked in the order they were shown. Lag ``T - 1`` is the frame just before the
    spike, lag 0 is ``T`` frames before it.

    The result is normalised per electrode as the standard pipeline does it: divided by the
    electrode's spike count, median-centred over the whole volume (the median is the mean
    luminance, which is what makes the STA signed) and scaled so its largest excursion is 1.
    An electrode with no spikes comes back as zeros.
    """
    n_elec, n_sequences, n_frames = counts.shape
    T = int(temporal_dimension)
    if not 0 < T < n_frames:
        raise ValueError(f"temporal dimension {T} must be between 1 and {n_frames - 1}")
    if stimulus.shape[0] < n_sequences * n_frames:
        raise ValueError(f"the stimulus holds {stimulus.shape[0]} frames but the triggers ask "
                         f"for {n_sequences * n_frames}")
    rows, cols = stimulus.shape[1:]
    n_pixels = rows * cols

    sta = np.zeros((n_elec, T, n_pixels), dtype=np.float32)
    for sequence in range(n_sequences):
        block = stimulus[sequence * n_frames:(sequence + 1) * n_frames]
        frames = block.reshape(n_frames, n_pixels).astype(np.float32) / 255.0
        weights = counts[:, sequence, T:]                       # (n_elec, n_frames - T)
        if not weights.any():
            continue
        for lag in range(T):
            sta[:, lag] += weights @ frames[lag:lag + n_frames - T]
        if progress is not None:
            progress(f"STA: sequence {sequence + 1}/{n_sequences}")

    totals = counts.sum(axis=(1, 2))
    for i, total in enumerate(totals):
        if total <= 0:
            sta[i] = 0.0
            continue
        sta[i] /= total
        sta[i] -= np.median(sta[i])
        peak = np.max(np.abs(sta[i]))
        if peak > 0:
            sta[i] /= peak
    return sta.reshape(n_elec, T, rows, cols)


def peak_components(sta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per electrode: the spatial frame at its strongest lag, its temporal profile, and where.

    Returns ``(spatial (n, rows, cols), temporal (n, T), peak (n, 3))`` with ``peak`` holding
    ``(lag, row, col)``. Each spatial frame is rescaled so its own largest excursion is 1, so
    a weak electrode is still readable on the grid.
    """
    n_elec, T, rows, cols = sta.shape
    flat = np.abs(sta).reshape(n_elec, -1)
    best = np.argmax(flat, axis=1)
    lag, row, col = np.unravel_index(best, (T, rows, cols))
    index = np.arange(n_elec)
    spatial = sta[index, lag].copy()
    temporal = sta[index, :, row, col].copy()
    scale = np.max(np.abs(spatial), axis=(1, 2))
    np.divide(spatial, scale[:, None, None], out=spatial, where=scale[:, None, None] > 0)
    return spatial, temporal, np.stack([lag, row, col], axis=1)


# ------------------------------------------------------------------------- whitening
def stimulus_covariance(
    stimulus: np.ndarray,
    chunk: int = 2048,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Covariance of the stimulus's pixels, unregularised.

    Accumulated in chunks, so the whole ``(n_frames, n_pixels)`` matrix is never materialised
    (0.7 GB for a 27 000-frame 80x80 stimulus). The regularisation is deliberately *not* folded
    in here: it is added at whitening time, so trying another value costs a factorisation
    rather than another pass over the stimulus.
    """
    n_frames = stimulus.shape[0]
    n_pixels = stimulus.shape[1] * stimulus.shape[2]
    gram = np.zeros((n_pixels, n_pixels), dtype=np.float64)
    total = np.zeros(n_pixels, dtype=np.float64)
    for start in range(0, n_frames, chunk):
        block = stimulus[start:start + chunk].reshape(-1, n_pixels).astype(np.float32) / 255.0
        gram += block.T.astype(np.float64) @ block
        total += block.sum(axis=0, dtype=np.float64)
        if progress is not None:
            progress(f"stimulus covariance: {min(start + chunk, n_frames)}/{n_frames} frames")
    mean = total / n_frames
    return gram / n_frames - np.outer(mean, mean)


def whiten(
    spatial: np.ndarray, cov: np.ndarray, sigma_fraction: float = DEFAULT_SIGMA_FRACTION
) -> np.ndarray:
    """``(C + sigma I)^-1`` applied to every electrode's spatial STA, with one factorisation.

    ``spatial`` is ``(n_electrodes, rows, cols)``; the result is renormalised per electrode the
    same way the raw STA is, so the two are directly comparable.

    ``sigma_fraction`` is a fraction of the mean pixel variance, which keeps it meaningful
    whatever the stimulus contrast. It matters more than it looks: a shifting stimulus's
    covariance is strongly banded, so its high spatial frequencies carry very little variance
    and an under-regularised inverse amplifies them — on a multi-unit STA, which is noisy, that
    buries the receptive field instead of sharpening it. The standard pipeline's fixed
    ``sigma = 5.0`` errs the other way: about twenty times the variance of a black-and-white
    stimulus, which leaves the whitening doing nothing at all.
    """
    n_elec, rows, cols = spatial.shape
    regularised = cov.copy()
    sigma = float(sigma_fraction * np.mean(np.diag(cov)))
    regularised[np.diag_indices_from(regularised)] += sigma
    factor = cho_factor(regularised, lower=True, check_finite=False)
    flat = spatial.reshape(n_elec, rows * cols).astype(np.float64)
    live = np.max(np.abs(flat), axis=1) > 0
    out = np.zeros_like(flat)
    if live.any():
        out[live] = cho_solve(factor, flat[live].T, check_finite=False).T
        out[live] -= np.median(out[live], axis=1, keepdims=True)
        scale = np.max(np.abs(out[live]), axis=1, keepdims=True)
        out[live] = np.divide(out[live], scale, out=out[live], where=scale > 0)
    return out.reshape(n_elec, rows, cols).astype(np.float32)


# ------------------------------------------------------------------- orientation check
def rf_quality(spatial: np.ndarray) -> np.ndarray:
    """How much of a receptive field a map has: its peak over the spread of its border.

    A map with a real receptive field has a strong peak somewhere inside and little going on at
    the edges; a map of noise is equally busy everywhere. Takes one map ``(rows, cols)`` or a
    stack ``(n, rows, cols)`` and gives a scalar or one value each.
    """
    maps = np.abs(np.asarray(spatial))
    single = maps.ndim == 2
    if single:
        maps = maps[None]
    border = np.concatenate(
        [maps[:, 0, :], maps[:, -1, :], maps[:, :, 0], maps[:, :, -1]], axis=1)
    out = maps.reshape(maps.shape[0], -1).max(axis=1) / (border.std(axis=1) + 1e-9)
    return out[0] if single else out


def measure_registration(
    spatial: np.ndarray,
    labels: np.ndarray,
    mapping: dict[int, tuple[int, int]],
) -> list[tuple[str, float]]:
    """Which way the stimulus axes sit on the electrode grid, measured from the data.

    An electrode's position on the array has to predict where its receptive field is. That
    holds under exactly one of the eight ways two image frames can be related, so each is tried
    and scored by how well the peak position correlates with the electrode position. Only the
    better half of the electrodes count, by :func:`rf_quality` — a dead electrode's peak is
    wherever the noise happened to be highest.

    Returns ``(name, score)`` best first, the names being keys of
    :data:`mua_sta.rigs.ARRAY_REGISTRATIONS`. A clear winner near 1 means the array's
    orientation is settled; everything near 0 means there are no receptive fields to go on.
    """
    from .rigs import ARRAY_REGISTRATIONS

    quality = rf_quality(spatial)
    good = quality > np.median(quality)
    if good.sum() < 4:
        return [(name, 0.0) for name in ARRAY_REGISTRATIONS]
    electrode = np.array([mapping[int(label)] for label in labels])[good]
    scores = []
    for name, transform in ARRAY_REGISTRATIONS.items():
        maps = transform(spatial[good])
        flat = np.abs(maps).reshape(maps.shape[0], -1)
        peak = np.array(np.unravel_index(flat.argmax(axis=1), maps.shape[1:])).T
        along = [np.corrcoef(electrode[:, axis], peak[:, axis])[0, 1] for axis in (0, 1)]
        scores.append((name, float(np.mean(along))))
    return sorted(scores, key=lambda item: -item[1])


# ----------------------------------------------------------------------------- display
@dataclass(frozen=True)
class StitchedGrid:
    """Every electrode's spatial STA tiled into one image, for the 16x16 overview."""

    image: np.ndarray           # float32, NaN where no electrode sits
    tile: tuple[int, int]       # (rows, cols) of one tile
    padding: int

    def electrode_at(self, x: float, y: float, mapping: dict[int, tuple[int, int]]) -> int | None:
        """Which electrode a point in canvas coordinates falls on, or None between tiles."""
        rows, cols = self.tile
        row, within_row = divmod(int(y), rows + self.padding)
        col, within_col = divmod(int(x), cols + self.padding)
        if within_row >= rows or within_col >= cols:
            return None                              # in the gap between two tiles
        for label, position in mapping.items():
            if position == (row, col):
                return label
        return None


def field_bounds(
    spatial: np.ndarray, margin: float = 0.35, minimum: int = 8
) -> tuple[int, int, int, int] | None:
    """Where the receptive fields are: the box their peaks fall in, with a margin around it.

    The stimulus covers far more of the retina than the array does — 1680 µm against 480 µm on
    MEA3 — so most of each tile of the overview is empty and the retinotopic shift between
    neighbouring electrodes is a few cells in an 80-cell tile. Cropping every tile to this box
    makes that shift plain without hiding anything: the box is measured from the data, and only
    the better half of the electrodes vote, so a noisy one cannot drag it open.

    Returns ``(row0, row1, col0, col1)``, or None when there is nothing to crop to.
    """
    n_elec, rows, cols = spatial.shape
    quality = rf_quality(spatial)
    good = quality > np.median(quality)
    if good.sum() < minimum:
        return None
    flat = np.abs(spatial[good]).reshape(int(good.sum()), -1)
    peak = np.array(np.unravel_index(flat.argmax(axis=1), (rows, cols))).T
    (r0, c0), (r1, c1) = peak.min(axis=0), peak.max(axis=0) + 1
    pad_r = max(3, int(round((r1 - r0) * margin)))
    pad_c = max(3, int(round((c1 - c0) * margin)))
    r0, r1 = max(0, r0 - pad_r), min(rows, r1 + pad_r)
    c0, c1 = max(0, c0 - pad_c), min(cols, c1 + pad_c)
    if (r1 - r0, c1 - c0) == (rows, cols):
        return None
    return int(r0), int(r1), int(c0), int(c1)


def stitch(
    spatial: np.ndarray,
    labels: np.ndarray,
    mapping: dict[int, tuple[int, int]],
    grid_size: int = 16,
    padding: int = 3,
    crop: tuple[int, int, int, int] | None = None,
) -> StitchedGrid:
    """Tile the spatial STAs onto the electrode grid, in a single +y-up convention.

    The canvas is built so that **its row index increases upwards**: electrode row 0 (y = 0 in
    the probe file) is the first block, and within each tile STA row 0 is its first row. Drawn
    with y increasing upwards — pyqtgraph's default, or ``origin="lower"`` in matplotlib — the
    grid and every tile in it then read with +y up, which is the convention the standard
    pipeline uses for receptive-field positions and the one the four-squares orientation check
    assumes ("the cells whose receptive field is at the top respond to the top square"). The
    previous version of this tool tiled row 0 first and drew it with ``origin="upper"``, so its
    grid came out mirrored against those.

    Electrodes with no tile are left as NaN; a blue-white-red colour map renders them the same
    white as a flat STA, as the earlier stitched figure did. ``crop`` narrows every tile to the
    same window of the stimulus, usually :func:`field_bounds`.
    """
    if crop is not None:
        r0, r1, c0, c1 = crop
        spatial = spatial[:, r0:r1, c0:c1]
    rows, cols = spatial.shape[1:]
    height = grid_size * rows + (grid_size - 1) * padding
    width = grid_size * cols + (grid_size - 1) * padding
    canvas = np.full((height, width), np.nan, dtype=np.float32)
    for i, label in enumerate(labels):
        position = mapping.get(int(label))
        if position is None:
            continue
        row, col = position
        y = row * (rows + padding)
        x = col * (cols + padding)
        canvas[y:y + rows, x:x + cols] = spatial[i]
    return StitchedGrid(canvas, (rows, cols), padding)
