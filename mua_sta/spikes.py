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

import contextlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import recording as rec

DEFAULT_DETECT_THRESHOLD = 6.0      # in median absolute deviations
DEFAULT_PEAK_SIGN = "neg"
DEFAULT_FREQ_MIN = 300.0
DEFAULT_FREQ_MAX = 6000.0

#: Roughly how much recording to detect in one go, in seconds. Detection is handed to a pool
#: of processes that cannot be interrupted once it is running, so the recording is cut into a
#: few pieces: stopping then takes at most one piece instead of the whole recording. The noise
#: levels are measured once over everything and passed to each piece, so every piece uses the
#: same thresholds — detecting in pieces then finds the same spikes as detecting in one go,
#: give or take a waveform lying across a cut, which can cross the threshold on both sides.
#: With eight pieces that is a handful of spikes in millions, and the receptive fields, their
#: strength and their layout come out the same (checked on a 30-minute recording).
SLICE_SECONDS = 240.0
MAX_SLICES = 16


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


class _Place:
    """Which piece of the recording is being detected, so progress stays monotonic."""

    index = 0


def _slice_bounds(n_samples: int, fs: float) -> list[tuple[int, int]]:
    """Cut the recording into a few pieces of about :data:`SLICE_SECONDS` each."""
    duration = n_samples / fs
    count = max(1, min(MAX_SLICES, round(duration / SLICE_SECONDS)))
    edges = np.linspace(0, n_samples, count + 1).astype(np.int64)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


@contextlib.contextmanager
def _reporting_through(progress: Callable[[int, int, str], None] | None,
                       place: "_Place | None" = None, n_slices: int = 1):
    """Let spikeinterface's own progress bars drive ours while this block runs.

    spikeinterface measures its passes with ``tqdm`` and offers no callback, so its progress
    bar class is swapped for one that reports instead of drawing. It wraps only the counting,
    never the work, and is put back whatever happens — if a future version stops using tqdm
    there, detection still runs and only the percentage is lost.
    """
    if progress is None:
        yield
        return
    from spikeinterface.core import job_tools
    original = getattr(job_tools, "tqdm", None)
    if original is None:
        yield
        return

    class Reporting(original):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = False       # we need the counting, not the drawing
            super().__init__(*args, **kwargs)

        def _report(self):
            if not self.total:
                return
            name = (self.desc or "detecting").split(" (")[0]
            if place is None or not name.startswith("detect"):
                progress(self.n, self.total, name)
                return
            # map this piece's own bar onto the whole recording, so the percentage and the
            # time left run once from 0 to 100 rather than restarting at every piece
            inside = 100 * self.n / self.total
            progress(int(place.index * 100 + inside), n_slices * 100, "detecting spikes")

        def display(self, *args, **kwargs):
            self._report()
            return True                     # drawn by us, not by tqdm

        def update(self, n=1):
            outcome = super().update(n)
            self._report()
            return outcome

    job_tools.tqdm = Reporting
    try:
        yield
    finally:
        job_tools.tqdm = original


def detect_mua(
    info: rec.RecordingInfo,
    detect_threshold: float = DEFAULT_DETECT_THRESHOLD,
    peak_sign: str = DEFAULT_PEAK_SIGN,
    freq_min: float = DEFAULT_FREQ_MIN,
    freq_max: float = DEFAULT_FREQ_MAX,
    common_reference: bool = True,
    n_jobs: int = 10,
    progress: Callable[[int, int, str], None] | None = None,
) -> SpikeTrains:
    """Band-pass, common-reference and detect peaks on every channel of ``info``.

    The long step of the whole tool: a pass over the full recording on every channel. Its
    result is cached on the detection parameters alone, so nothing about the stimulus or the
    analysis can force it to run again.
    """
    import spikeinterface.core as sc
    import spikeinterface.preprocessing as spre
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks

    if progress is not None:
        progress(0, 0, "opening the recording")
    recording = rec.as_spikeinterface(info)
    recording = spre.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)
    if common_reference:
        recording = spre.common_reference(recording)

    if progress is not None:
        progress(0, 0, "measuring the noise")
    noise_levels = sc.get_noise_levels(recording, return_in_uV=False)

    bounds = _slice_bounds(recording.get_num_samples(), info.fs)
    method_kwargs = {"peak_sign": peak_sign, "detect_threshold": detect_threshold,
                     "noise_levels": noise_levels}
    samples, channels = [], []
    place = _Place()
    with _reporting_through(progress, place, len(bounds)):
        for index, (start, stop) in enumerate(bounds):
            place.index = index
            if progress is not None:        # a stopping point between pieces
                progress(index * 100, len(bounds) * 100, "detecting spikes")
            found = detect_peaks(
                recording.frame_slice(start, stop),
                method="by_channel",
                method_kwargs=method_kwargs,
                job_kwargs={"n_jobs": n_jobs, "progress_bar": progress is not None},
            )
            samples.append(found["sample_index"] + start)
            channels.append(found["channel_index"])
    if progress is not None:
        progress(len(bounds) * 100, len(bounds) * 100, "detecting spikes")
    peaks = {"sample_index": np.concatenate(samples) if samples else np.empty(0, dtype=np.int64),
             "channel_index": np.concatenate(channels) if channels else np.empty(0, dtype=np.int64)}
    return from_peaks(peaks["sample_index"], peaks["channel_index"], info.fs)
