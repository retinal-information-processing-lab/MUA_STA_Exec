"""Multi-unit spikes: threshold crossings on each channel independently.

This is deliberately *not* spike sorting. The point of this tool is to judge, while the
retina is still alive, whether it responds — so every channel's threshold crossings are
pooled as one multi-unit train per electrode. The standard pipeline does the sorting later.

The one subtlety is which electrode a train belongs to. spikeinterface reports a 0-based
``channel_index`` into the recording, while the electrode mapping is keyed by MCS label,
1-based. The previous version of this tool used the index as if it were a label, so every
result sat one electrode away from where it belonged; here the conversion is explicit and
goes through :func:`mua_sta.recording.channel_index_to_label`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import recording as rec

DEFAULT_DETECT_THRESHOLD = 6.0      # in median absolute deviations
DEFAULT_PEAK_SIGN = "neg"
DEFAULT_FREQ_MIN = 300.0
DEFAULT_FREQ_MAX = 6000.0


@dataclass(frozen=True)
class SpikeTrains:
    """Spike times per electrode, flat so they cache without pickling.

    ``times`` holds every spike time in seconds, electrode after electrode; ``labels`` are the
    MCS labels present, sorted; ``offsets`` slices ``times`` per label.
    """

    labels: np.ndarray          # int32, sorted MCS labels (1-based)
    offsets: np.ndarray         # int64, length len(labels) + 1
    times: np.ndarray           # float32 seconds, sorted within each electrode

    def __contains__(self, label: int) -> bool:
        pos = np.searchsorted(self.labels, label)
        return bool(pos < self.labels.size and self.labels[pos] == label)

    def get(self, label: int) -> np.ndarray:
        """Spike times of one electrode in seconds; empty when it has none."""
        pos = np.searchsorted(self.labels, label)
        if pos >= self.labels.size or self.labels[pos] != label:
            return np.empty(0, dtype=np.float32)
        return self.times[self.offsets[pos]:self.offsets[pos + 1]]

    def counts(self) -> dict[int, int]:
        return {int(label): int(self.offsets[i + 1] - self.offsets[i])
                for i, label in enumerate(self.labels)}

    @property
    def total(self) -> int:
        return int(self.times.size)

    def describe(self) -> str:
        return (f"{self.total:,} spikes on {self.labels.size} electrodes "
                f"(median {int(np.median(np.diff(self.offsets))) if self.labels.size else 0} each)")


def from_peaks(sample_index: np.ndarray, channel_index: np.ndarray, fs: float) -> SpikeTrains:
    """Group detected peaks by electrode. ``channel_index`` is 0-based, labels come out 1-based."""
    order = np.argsort(channel_index, kind="stable")     # stable: keeps each channel in time order
    channels = np.asarray(channel_index)[order]
    times = (np.asarray(sample_index)[order] / fs).astype(np.float32)
    labels, starts = np.unique(channels, return_index=True)
    offsets = np.append(starts, channels.size).astype(np.int64)
    labels = np.array([rec.channel_index_to_label(int(c)) for c in labels], dtype=np.int32)
    return SpikeTrains(labels, offsets, times)


def detect_mua(
    info: rec.RecordingInfo,
    detect_threshold: float = DEFAULT_DETECT_THRESHOLD,
    peak_sign: str = DEFAULT_PEAK_SIGN,
    freq_min: float = DEFAULT_FREQ_MIN,
    freq_max: float = DEFAULT_FREQ_MAX,
    common_reference: bool = True,
    n_jobs: int = 10,
    progress: Callable[[str], None] | None = None,
) -> SpikeTrains:
    """Band-pass, common-reference and detect peaks on every channel of ``info``.

    The long step of the whole tool: a pass over the full recording on every channel. Its
    result is cached on the detection parameters alone, so nothing about the stimulus or the
    analysis can force it to run again.
    """
    import spikeinterface.preprocessing as spre
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks

    if progress is not None:
        progress("opening the recording")
    recording = rec.as_spikeinterface(info)
    recording = spre.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)
    if common_reference:
        recording = spre.common_reference(recording)

    if progress is not None:
        progress(f"detecting peaks on {info.n_channels} channels")
    peaks = detect_peaks(
        recording,
        method="by_channel",
        method_kwargs={"peak_sign": peak_sign, "detect_threshold": detect_threshold},
        job_kwargs={"n_jobs": n_jobs, "progress_bar": False},
    )
    return from_peaks(peaks["sample_index"], peaks["channel_index"], info.fs)
