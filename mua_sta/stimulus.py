"""Rebuilding the stimulus that was shown, for both kinds this tool supports.

**Checkerboard** — the stimulus software draws its checks straight from a binary source file
(``binarysource1000Mbits``), consuming it bit by bit from bit 0, least significant bit first
inside each byte. There is no ``.vec``: trigger *i* showed frame *i*, and only the novel half
of each sequence comes from this source, so the frames needed are the first
``n_sequences * frames_per_sequence / 2`` of it.

**Shifting white noise (SWN)** — a fixed library of noise frames in a ``.bin``, played in the
order given by a ``.vec``. Each frame is a checkerboard whose grid is offset by a random
whole number of *shift steps*, a shift step being a fraction of a check. Over many frames the
stimulus therefore resolves finer than one check, and the natural analysis grid has one cell
per shift step: ``checks_per_side * shifts_per_check`` cells per side.

Sampling SWN at the **centre of each cell** is exact, not an approximation: because the grid
only ever moves by whole shift steps, every check edge lands on a cell boundary, so a cell is
always entirely inside one check. It also side-steps the interpolated pixels at check edges
that a ``.bin`` rendered at a non-integer scale contains — the MEA3 library is drawn at 672 px
for a 480 px design, so its checks are 67.2 px wide and its edges are soft. Sampling on a
lattice that does *not* line up with the shifts (what ``frame[::6, ::6]`` does on that file:
112 samples of an 80-cell stimulus) mixes those edges in and invents short-range correlations,
which then corrupt the covariance the STA is whitened with.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .binfile import BinFile, read_header
from .rigs import apply_optical_transform, rig

CHECKERBOARD, SWN = "checkerboard", "swn"
KIND_LABELS = {CHECKERBOARD: "Checkerboard", SWN: "Shifting white noise"}

#: Enough of a mismatch between the sampled values and a binary stimulus to be suspicious.
NON_BINARY_WARN = 0.01


# --------------------------------------------------------------------------- vec files
def read_vec(path: os.PathLike | str) -> tuple[np.ndarray, np.ndarray]:
    """A ``.vec`` as ``(header_row, frame_indices)``.

    Five whitespace-separated columns, one row per DMD refresh (so one row per trigger), after
    a header row whose column 1 holds the number of rows in the file. Column 1 of each row is
    the frame index into the ``.bin``.
    """
    data = np.loadtxt(os.fspath(path))
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{Path(path).name}: expected a .vec of at least 2 columns, got "
                         f"shape {data.shape}")
    header, rows = data[0], data[1:]
    declared = int(header[1])
    if declared not in (rows.shape[0], rows.shape[0] + 1):
        warnings.warn(
            f"{Path(path).name}: the header announces {declared} rows but the file holds "
            f"{rows.shape[0]} frame rows; using what is in the file.", stacklevel=2)
    return header, rows[:, 1].astype(np.int64)


def novel_frame_count(frame_indices: np.ndarray) -> int:
    """How many frames of the library are the novel (never-repeated) ones.

    A sequence is a novel half followed by a repeated half of the same length, so half the
    playlist is novel-frame presentations — and since each novel frame is shown exactly once,
    that count is also how many novel frames there are. They are the lowest indices of the
    library, so ``index < novel_frame_count(...)`` is the test for "novel".
    """
    return int(frame_indices.size // 2)


# ------------------------------------------------------------------- the analysis grid
def cell_centres(size: int, n_cells: int) -> np.ndarray:
    """Pixel index of the centre of each of ``n_cells`` cells spanning ``size`` pixels.

    The step may be fractional (8.4 px on MEA3), which is exactly why the centres are rounded
    individually instead of using a constant stride.
    """
    if n_cells < 1 or n_cells > size:
        raise ValueError(f"{n_cells} cells do not fit in {size} pixels")
    step = size / n_cells
    centres = np.round((np.arange(n_cells) + 0.5) * step).astype(np.int64)
    return np.clip(centres, 0, size - 1)


#: How far a detected check edge may sit from its true position, in pixels. Edges come out of
#: a thresholded image, and are soft when the library is rendered at a non-integer scale.
EDGE_TOLERANCE_PX = 0.8

#: Most shift steps per check worth considering.
MAX_SHIFTS_PER_CHECK = 64


def _grid_offset(edges: np.ndarray, pixels_per_check: float) -> float:
    """Where this frame's check grid starts, in [0, pixels_per_check).

    Every edge of a frame is the same offset plus a whole number of checks, so folding them
    all into one check and taking their *circular* mean averages out the per-edge rounding.
    A plain mean would not: an offset near zero folds to values at both ends of the range.
    """
    angles = 2 * np.pi * np.mod(edges, pixels_per_check) / pixels_per_check
    mean = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    return float(np.mod(mean, 2 * np.pi) / (2 * np.pi) * pixels_per_check)


def _shifts_per_check(offsets: np.ndarray, pixels_per_check: float) -> int:
    """The smallest number of shift steps per check that explains every observed offset.

    Any larger count also "fits" (at the limit, one step per pixel fits anything), so the
    smallest one that does is the design. An offset is explained when it is a whole number of
    steps away, to within the edge-detection tolerance.
    """
    for shifts in range(1, min(MAX_SHIFTS_PER_CHECK, int(pixels_per_check)) + 1):
        step = pixels_per_check / shifts
        residual = np.abs(offsets / step - np.round(offsets / step)) * step
        if np.max(residual) <= EDGE_TOLERANCE_PX:
            return shifts
    return 1


def detect_swn_lattice(
    bin_path: os.PathLike | str, mea: int, n_frames: int = 32
) -> tuple[int, int, str]:
    """Measure ``(checks_per_side, shifts_per_check)`` from the frames themselves.

    Reads a few frames, finds where the checks change value along several rows and columns,
    and fits the check width from the first and last edge of each frame (the edges are not at
    integer pixels when the library is rendered at a non-integer scale, and are detected to
    about +-1 px). Each frame then gives one sub-pixel grid offset, and the shift count is the
    *smallest* one that explains every offset as a whole number of shift steps.

    Returns the two counts and a sentence for the log. The result is a suggestion the user
    confirms in the GUI, not an authority: the stimulus design is.
    """
    header = read_header(bin_path)
    with BinFile(bin_path, mea) as stimulus:
        size = min(stimulus.shape)
        spacings: list[float] = []
        edge_sets: list[np.ndarray] = []
        for index in np.linspace(0, len(stimulus) - 1, num=min(n_frames, len(stimulus))).astype(int):
            binary = stimulus.stored(int(index)) > 127
            for axis in (0, 1):
                lines = binary if axis == 0 else binary.T
                edges = set()
                for line in lines[:: max(1, lines.shape[0] // 8)]:
                    edges.update((np.flatnonzero(np.diff(line)) + 1).tolist())
                edges = np.array(sorted(e for e in edges if 0 < e < lines.shape[1]))
                if edges.size >= 3:
                    spacings.append((edges[-1] - edges[0]) / (edges.size - 1))
                    edge_sets.append(edges)
    if not spacings:
        raise ValueError(f"{Path(bin_path).name}: found no check edges; is this an SWN library?")

    spacing = float(np.median(spacings))
    checks = max(1, int(round(size / spacing)))
    pixels_per_check = size / checks
    offsets = np.array([_grid_offset(edges, pixels_per_check) for edges in edge_sets])
    shifts = _shifts_per_check(offsets, pixels_per_check)
    note = (f"{Path(bin_path).name}: {header['xsize']}x{header['ysize']} px, "
            f"checks ~{pixels_per_check:.1f} px -> {checks} per side, "
            f"{shifts} shift(s) per check ({pixels_per_check / shifts:.1f} px) "
            f"-> a {checks * shifts}x{checks * shifts} grid")
    return checks, shifts, note


# ----------------------------------------------------------------------- stimulus specs
@dataclass(frozen=True)
class CheckerboardSpec:
    """Where a classic checkerboard comes from and how coarse it is."""

    source_path: Path
    n_checks_x: int
    n_checks_y: int
    mea: int
    pixels_per_check: int = 16      # how wide a check was drawn on the DMD
    kind: str = CHECKERBOARD

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Shape of one analysis frame, ``(rows, cols)``, after the rig's optical transform."""
        probe = np.empty((self.n_checks_y, self.n_checks_x), dtype=np.uint8)
        return apply_optical_transform(probe, rig(self.mea).optical_transform).shape

    @property
    def cell_size_um(self) -> float | None:
        """Side of one check on the retina, in µm.

        A check's width in DMD pixels is a property of the stimulus, not of the DMD: a 42-check
        board of 16 px checks covers 672 px of a 760 px DMD, not all of it. So it is asked for
        rather than derived, as the standard pipeline's ``nb_pixels_per_check`` is.
        """
        settings = rig(self.mea)
        if settings.pxl_size_dmd is None:
            return None
        return settings.pxl_size_dmd * self.pixels_per_check

    def cache_deps(self, n_frames: int) -> dict:
        # pixels_per_check is deliberately absent: it only scales the µm the result is reported
        # in, and does not change a single reconstructed frame.
        from .cache import file_identity
        return {"kind": self.kind, "source": file_identity(self.source_path), "mea": self.mea,
                "n_checks_x": self.n_checks_x, "n_checks_y": self.n_checks_y,
                "n_frames": n_frames}

    def build(self, n_frames: int, progress: Callable[[str], None] | None = None) -> np.ndarray:
        """``(n_frames, rows, cols)`` uint8 in 0..255, as the retina saw it."""
        rows, cols = self.n_checks_y, self.n_checks_x
        n_bits = n_frames * rows * cols
        n_bytes = (n_bits + 7) // 8
        if progress is not None:
            progress(f"unpacking {n_frames} checkerboard frames from {self.source_path.name}")
        with open(self.source_path, "rb") as f:
            raw = np.frombuffer(f.read(n_bytes), dtype=np.uint8)
        if raw.size < n_bytes:
            raise ValueError(
                f"{self.source_path.name} holds {raw.size * 8} bits but {n_frames} frames of "
                f"{rows}x{cols} checks need {n_bits}. Fewer sequences, or a coarser "
                "checkerboard, or the wrong source file.")
        bits = np.unpackbits(raw, bitorder="little")[:n_bits]
        # Row-major within a frame: bit (frame, row, col) = frame*rows*cols + row*cols + col.
        frames = bits.reshape(n_frames, rows, cols)
        if rig(self.mea).invert_polarity:
            frames = frames ^ 1
        frames = apply_optical_transform(frames, rig(self.mea).optical_transform)
        return np.ascontiguousarray(frames * np.uint8(255))


@dataclass(frozen=True)
class SwnSpec:
    """Where a shifting-white-noise stimulus comes from and how it maps onto a grid."""

    bin_path: Path
    vec_path: Path
    checks_per_side: int
    shifts_per_check: int
    mea: int
    kind: str = SWN

    @property
    def n_cells(self) -> int:
        return self.checks_per_side * self.shifts_per_check

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.n_cells, self.n_cells)

    @property
    def cell_size_um(self) -> float | None:
        settings = rig(self.mea)
        if settings.pxl_size_dmd is None:
            return None
        header = read_header(self.bin_path)
        return settings.pxl_size_dmd * header["xsize"] / self.n_cells

    def cache_deps(self, n_frames: int) -> dict:
        from .cache import file_identity
        return {"kind": self.kind, "bin": file_identity(self.bin_path),
                "vec": file_identity(self.vec_path), "mea": self.mea,
                "checks_per_side": self.checks_per_side,
                "shifts_per_check": self.shifts_per_check, "n_frames": n_frames}

    def build(self, n_frames: int, progress: Callable[[str], None] | None = None) -> np.ndarray:
        """The novel frames in the order they were shown, ``(n_frames, cells, cells)`` uint8.

        Stops as soon as ``n_frames`` have been collected, so a recording that was cut short
        does not pay for the rest of the library.
        """
        _, frame_indices = read_vec(self.vec_path)
        n_novel = novel_frame_count(frame_indices)
        novel = frame_indices[frame_indices < n_novel]
        if novel.size < n_frames:
            raise ValueError(
                f"{self.vec_path.name} plays {novel.size} novel frames but the recording's "
                f"triggers ask for {n_frames}. Is this the .vec that was shown?")
        wanted = novel[:n_frames]

        out = np.empty((n_frames, self.n_cells, self.n_cells), dtype=np.uint8)
        with BinFile(self.bin_path, self.mea) as stimulus:
            transform = rig(self.mea).optical_transform
            probe = apply_optical_transform(stimulus.stored(0), transform)
            rows = cell_centres(probe.shape[0], self.n_cells)
            cols = cell_centres(probe.shape[1], self.n_cells)
            picker = np.ix_(rows, cols)
            step = 0
            for position, index in enumerate(wanted):
                # The optical transform is a view, so only n_cells**2 values are ever gathered.
                view = apply_optical_transform(stimulus.stored(int(index)), transform)
                out[position] = view[picker]
                if progress is not None and position - step >= 2000:
                    step = position
                    progress(f"reading SWN frames: {position}/{n_frames}")
        if rig(self.mea).invert_polarity:
            out = 255 - out
        self._warn_if_lattice_looks_wrong(out)
        return out

    def _warn_if_lattice_looks_wrong(self, frames: np.ndarray):
        """A cell centre should land inside one check, so the samples should be black or white.

        Grey samples mean the lattice does not line up with the shifts — the one mistake that
        silently degrades the whitened STA — so say so rather than carry on quietly.
        """
        grey = np.count_nonzero((frames > 8) & (frames < 247)) / frames.size
        if grey > NON_BINARY_WARN:
            warnings.warn(
                f"{grey:.1%} of the sampled SWN values are neither black nor white. A cell "
                f"centre should sit inside a single check, so {self.checks_per_side} checks x "
                f"{self.shifts_per_check} shifts is probably not this stimulus's design — "
                "run Detect, or check the numbers against the stimulus file name.",
                stacklevel=3)
