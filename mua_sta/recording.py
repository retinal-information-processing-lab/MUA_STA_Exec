"""Opening a recording, and which channel of it is which electrode.

This is the **only** module that knows the recording file format, and it accepts two:

``.mcd`` is what MC_Rack writes: a chunked container whose header states the channel count,
the sampling rate and the µV per ADC level, and whose electrode payloads are byte-identical
to the ``.raw`` MC_DataTool exports from it (verified across 25.36 GB). It is read directly
by :mod:`mua_sta.mcd`, so no conversion step and no second copy on disk are needed.

``.raw`` is what MC_DataTool writes out of a ``.mcd``: headerless, channel-interleaved
``uint16`` samples (sample 0 of every channel, then sample 1 of every channel, ...), with
32768 meaning 0 V. Having no header, it has to be told its channel count and sampling rate,
which is what the defaults below are for.

The channel order is the MCS channel order in both, so the channel at index ``i`` is MCS
label ``i + 1`` — which is why the electrode mapping below is 1-based while spikeinterface's
``channel_index`` is 0-based. Getting that wrong shifts every result by one electrode, as the
previous version of this tool did.

Everything downstream goes through :class:`RecordingInfo`, :func:`read_channel` and
:func:`as_spikeinterface`, so nothing else in the package knows which of the two it has.
"""
from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import mcd
from .rigs import ADC_ZERO, UV_PER_LEVEL

DEFAULT_N_CHANNELS = 256
DEFAULT_DTYPE = "uint16"
DEFAULT_FS = 20_000.0
DEFAULT_TRIGGER_CHANNEL = 126        # 0-based index; MCS label 127

#: The two formats a recording can be in.
RAW, MCD = "raw", "mcd"

#: An ``.mcd`` always holds 16-bit unsigned samples; its header says so, and this is it.
MCD_DTYPE = "uint16"

#: What :func:`read_channel` calls the pass it makes over the file.
READING_TRIGGER = "reading the trigger channel"

#: MCS labels that carry no electrode (references and the trigger/aux inputs).
EXCLUDED_LABELS = frozenset({127, 128, 255, 256})

#: Side of the electrode grid; the MEA is GRID_SIZE x GRID_SIZE.
GRID_SIZE = 16

MAPPING_FILENAME = "electrodes_mapping_MEA_MCS_256.pkl"


def detect_format(path: os.PathLike | str) -> str:
    """``MCD`` or ``RAW``, judged by the file's **content**, not its name.

    An ``.mcd`` starts with the ``MCSSTRM `` magic, so a recording is still recognised when
    it is called ``.MCD``, ``recording.mcd.bak`` or has no extension at all — the same test
    :mod:`mua_sta.mcd` uses to make sure it never overwrites a recording. Anything else is
    taken to be a headerless ``.raw``.
    """
    return MCD if mcd.is_mcs_recording(path) else RAW


@dataclass(frozen=True)
class RecordingInfo:
    """What the tool needs to know about one recording."""

    path: Path
    n_channels: int
    dtype: str
    fs: float
    n_samples: int                   # samples per channel
    format: str = RAW
    uv_per_level: float = UV_PER_LEVEL

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def describe(self) -> str:
        gb = self.size_bytes / 1e9
        mins, secs = divmod(self.duration_s, 60)
        return (f"{self.format} · {gb:.1f} GB · {self.n_channels} ch · {self.fs / 1000:g} kHz · "
                f"{int(mins)} min {secs:04.1f} s")


def open_info(
    path: os.PathLike | str,
    n_channels: int = DEFAULT_N_CHANNELS,
    dtype: str = DEFAULT_DTYPE,
    fs: float = DEFAULT_FS,
) -> RecordingInfo:
    """Describe a recording, whichever of the two formats it is in.

    For a ``.raw`` nothing is read: the sample count comes from the file size, and the
    channel count, sample format and sampling rate are the caller's to supply. For a ``.mcd``
    all four come from the header, and an argument that disagrees with it only warns.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if detect_format(path) == MCD:
        return _mcd_info(path, n_channels, dtype, fs)
    item_size = np.dtype(dtype).itemsize
    n_values = path.stat().st_size // item_size
    if n_values % n_channels:
        warnings.warn(
            f"{path.name}: {n_values} samples is not a whole number of {n_channels}-channel "
            f"frames ({n_values % n_channels} left over). Is the channel count right?",
            stacklevel=2,
        )
    return RecordingInfo(path, n_channels, dtype, float(fs), n_values // n_channels, RAW)


def _mcd_info(path: Path, n_channels: int, dtype: str, fs: float) -> RecordingInfo:
    """Describe a ``.mcd`` from its header. The header wins; a disagreement warns."""
    with mcd.McdFile(path) as m:
        stream = m.stream()         # the electrode stream: the one with the most channels
        info = RecordingInfo(path, stream.n_channels, MCD_DTYPE, float(m.sample_rate),
                             int(m.n_samples), MCD, float(stream.conv_uv_per_lsb))
    disagreements = []
    if n_channels != info.n_channels:
        disagreements.append(f"channel count {n_channels} (header says {info.n_channels})")
    if np.dtype(dtype) != np.dtype(info.dtype):
        disagreements.append(f"sample format {dtype} (header says {info.dtype})")
    if float(fs) != info.fs:
        disagreements.append(f"sampling rate {fs:g} Hz (header says {info.fs:g} Hz)")
    if disagreements:
        warnings.warn(
            f"{path.name}: ignoring " + ", ".join(disagreements)
            + ". An .mcd states all three, so the header is used.",
            stacklevel=3,
        )
    return info


def read_channel(
    info: RecordingInfo,
    channel_index: int,
    progress: Callable[[int, int, str], None] | None = None,
    chunk_samples: int = 4_000_000,
) -> np.ndarray:
    """One channel of the recording, in raw ADC levels (``uint16``, 32768 = 0 V).

    A pass across the whole file either way, reported to ``progress`` roughly every
    ``chunk_samples`` samples: strided off a memory map for a ``.raw``, block by block with a
    prefetch thread for a ``.mcd``.
    """
    if not 0 <= channel_index < info.n_channels:
        raise ValueError(f"channel {channel_index} outside 0..{info.n_channels - 1}")
    if info.format == MCD:
        return _read_channel_mcd(info, channel_index, progress, chunk_samples)
    mm = np.memmap(info.path, dtype=info.dtype, mode="r")
    try:
        usable = info.n_samples * info.n_channels
        frames = mm[:usable].reshape(info.n_samples, info.n_channels)
        out = np.empty(info.n_samples, dtype=info.dtype)
        for start in range(0, info.n_samples, chunk_samples):
            stop = min(start + chunk_samples, info.n_samples)
            out[start:stop] = frames[start:stop, channel_index]
            if progress is not None:
                progress(stop, info.n_samples, READING_TRIGGER)
    finally:
        mm._mmap.close()
    return out


def _read_channel_mcd(
    info: RecordingInfo,
    channel_index: int,
    progress: Callable[[int, int, str], None] | None,
    chunk_samples: int,
) -> np.ndarray:
    """One channel of a ``.mcd``, streamed block by block.

    ``copy=False`` hands out a view into the reader's own buffer, which is safe here because
    the column is copied into ``out`` before the next block is asked for. Progress is reported
    on the same scale as the ``.raw`` path rather than once per 0.1 s block, which on a
    36-minute recording would be 21 000 callbacks into the GUI.
    """
    out = np.empty(info.n_samples, dtype=info.dtype)
    done = reported = 0
    with mcd.McdFile(info.path) as m:
        for start, _timestamp_ns, frames in m.blocks(prefetch=True, copy=False):
            if start >= info.n_samples:
                break
            stop = min(start + frames.shape[0], info.n_samples)
            out[start:stop] = frames[:stop - start, channel_index]
            done = stop
            if progress is not None and done - reported >= chunk_samples:
                progress(done, info.n_samples, READING_TRIGGER)
                reported = done
    if done != info.n_samples:
        raise EOFError(f"{info.path.name}: read {done} of {info.n_samples} samples")
    if progress is not None and reported != done:
        progress(done, info.n_samples, READING_TRIGGER)
    return out


def to_microvolts(levels: np.ndarray, uv_per_level: float) -> np.ndarray:
    """Raw ADC levels -> µV at the electrode."""
    return (levels.astype(np.float32) - ADC_ZERO) * uv_per_level


def as_spikeinterface(info: RecordingInfo):
    """The recording as a spikeinterface object, signed and ready to filter.

    Unsigned samples have to be made signed *before* filtering, otherwise the 32768 offset
    swamps the signal.
    """
    import spikeinterface.preprocessing as spre

    if info.format == MCD:
        from .mcdextractor import McdRecording
        recording = McdRecording(info.path)
    else:
        import spikeinterface.core as sc
        recording = sc.read_binary(
            file_paths=str(info.path),
            sampling_frequency=info.fs,
            num_channels=info.n_channels,
            dtype=info.dtype,
        )
    if np.dtype(info.dtype).kind == "u":
        recording = spre.unsigned_to_signed(recording)
    return recording


# --------------------------------------------------------------------------- electrodes
def mapping_path() -> Path:
    """The electrode mapping shipped next to the package."""
    return Path(__file__).resolve().parent.parent / MAPPING_FILENAME


def load_mapping(path: os.PathLike | str | None = None) -> dict[int, tuple[int, int]]:
    """MCS channel label (1-based, 1..256) -> its ``(row, col)`` on the 16x16 grid (0-based).

    Row 0 is the lowest row of the array (y = 0 in the probe file), so the grid is drawn with
    +y upwards — the convention the standard pipeline uses for receptive-field positions and
    the one the four-squares orientation check assumes.
    """
    path = Path(path) if path is not None else mapping_path()
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {int(label): (int(rc[0]), int(rc[1])) for label, rc in raw.items()}


def electrode_labels(mapping: dict[int, tuple[int, int]]) -> list[int]:
    """The labels that carry an electrode, in increasing order."""
    return sorted(label for label in mapping if label not in EXCLUDED_LABELS)


def label_to_channel_index(label: int) -> int:
    """MCS label (1-based) -> the channel index spikeinterface uses (0-based)."""
    return label - 1


def channel_index_to_label(channel_index: int) -> int:
    return channel_index + 1
