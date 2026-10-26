"""Stimulus triggers: the rising edges of the trigger channel.

The trigger is a TTL pulse recorded on one of the analog channels (MCS label 127, index 126
on both MEA2 and MEA3). Thresholding is done on the **raw ADC levels** rather than on a
converted trace: it avoids turning 37 million samples into floats, and it keeps the threshold
in the one unit that is unambiguous. ``Rig.trigger_threshold`` is in levels above
``ADC_ZERO``; the GUI also shows what it is in µV and lets it be set on the trace itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rigs import ADC_ZERO, UV_PER_LEVEL

#: Safety bound on walking an onset back to the foot of its rising edge.
MAX_EDGE_WALK = 1000


@dataclass(frozen=True)
class Triggers:
    """Detected stimulus onsets and how regular they are."""

    onsets: np.ndarray          # sample indices of the rising edges
    fs: float
    modal_interval: int         # the usual gap between onsets, in samples
    error_onsets: np.ndarray    # onsets whose gap deviates by more than the allowed jitter

    @property
    def times(self) -> np.ndarray:
        """Onsets in seconds from the start of the recording."""
        return self.onsets / self.fs

    @property
    def count(self) -> int:
        return int(self.onsets.size)

    @property
    def frequency(self) -> float:
        """Stimulus frequency in Hz, from the mean gap between onsets."""
        if self.onsets.size < 2:
            return 0.0
        return float(self.fs / np.mean(np.diff(self.onsets)))

    def n_sequences(self, frames_per_sequence: int) -> int:
        return self.count // frames_per_sequence

    def describe(self, frames_per_sequence: int | None = None) -> str:
        parts = [f"{self.count} triggers", f"{self.frequency:.2f} Hz"]
        if frames_per_sequence:
            parts.append(f"{self.n_sequences(frames_per_sequence)} sequences")
        parts.append("regular" if self.error_onsets.size == 0
                     else f"{self.error_onsets.size} irregular gaps")
        return " · ".join(parts)


def threshold_to_uv(threshold_levels: float) -> float:
    """A threshold in ADC levels above zero -> µV at the electrode."""
    return threshold_levels * UV_PER_LEVEL


def detect_onsets(levels: np.ndarray, threshold_levels: float) -> np.ndarray:
    """Rising edges of a raw trigger channel, as sample indices.

    ``levels`` is the channel in raw ADC levels (``uint16``, 32768 = 0 V) and
    ``threshold_levels`` is measured from that zero. Each crossing is then walked back down
    its rising edge, so an onset is the last sample still at baseline rather than wherever the
    pulse happened to pass the threshold. That is what keeps frame times comparable between
    rigs whose thresholds differ, and it is the convention the standard pipeline uses, so
    onsets from this tool and from the pipeline agree.
    """
    signal = levels.astype(np.int32) - ADC_ZERO
    threshold = int(round(threshold_levels))
    crossing = (signal[:-1] < threshold) & (signal[1:] >= threshold)
    indices = np.flatnonzero(crossing)
    for _ in range(MAX_EDGE_WALK):
        rising = (indices > 0) & (signal[indices - 1] < signal[indices])
        if not rising.any():
            break
        indices[rising] -= 1
    return indices.astype(np.int64)


def sanity_check(onsets: np.ndarray, fs: float, maximal_jitter: float = 0.25e-3) -> Triggers:
    """Wrap onsets in a :class:`Triggers`, flagging gaps that deviate from the usual one.

    An irregular gap means a missing or a spurious trigger, which shifts every later frame and
    quietly ruins the STA — so it is worth seeing before anything else runs.
    """
    onsets = np.asarray(onsets, dtype=np.int64)
    if onsets.size < 2:
        return Triggers(onsets, float(fs), 0, np.empty(0, dtype=np.int64))
    gaps = np.diff(onsets)
    values, counts = np.unique(gaps, return_counts=True)
    modal = int(values[np.argmax(counts)])
    bad = np.flatnonzero(np.abs(gaps - modal) >= maximal_jitter * fs)
    return Triggers(onsets, float(fs), modal, onsets[bad])


def envelope(levels: np.ndarray, fs: float, n_points: int = 4000) -> tuple[np.ndarray, ...]:
    """A min/max envelope of the trigger channel for plotting, in ADC levels above zero.

    Plain decimation would drop most pulses (a trigger is a few samples wide in a recording
    millions of samples long), so each plotted bin keeps the lowest and highest level it
    contains. Returns ``(t_seconds, low, high)``.
    """
    signal = levels.astype(np.int32) - ADC_ZERO
    if signal.size <= n_points:
        t = np.arange(signal.size) / fs
        return t, signal, signal
    bin_size = signal.size // n_points
    usable = bin_size * n_points
    blocks = signal[:usable].reshape(n_points, bin_size)
    t = (np.arange(n_points) * bin_size + bin_size / 2) / fs
    return t, blocks.min(axis=1), blocks.max(axis=1)
