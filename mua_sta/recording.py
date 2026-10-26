"""Opening a recording, and which channel of it is which electrode.

This is the **only** module that knows the recording file format. Goal 3 replaces the ``.raw``
reader here with a direct ``.mcd`` reader and nothing else in the package changes.

``.raw`` is what MC_DataTool writes out of a ``.mcd``: headerless, channel-interleaved
``uint16`` samples (sample 0 of every channel, then sample 1 of every channel, ...), with
32768 meaning 0 V. The channel order is the MCS channel order, so the channel at index ``i``
is MCS label ``i + 1`` — which is why the electrode mapping below is 1-based while
spikeinterface's ``channel_index`` is 0-based. Getting that wrong shifts every result by one
electrode, as the previous version of this tool did.
"""
from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .rigs import ADC_ZERO

DEFAULT_N_CHANNELS = 256
DEFAULT_DTYPE = "uint16"
DEFAULT_FS = 20_000.0
DEFAULT_TRIGGER_CHANNEL = 126        # 0-based index; MCS label 127

#: MCS labels that carry no electrode (references and the trigger/aux inputs).
EXCLUDED_LABELS = frozenset({127, 128, 255, 256})

#: Side of the electrode grid; the MEA is GRID_SIZE x GRID_SIZE.
GRID_SIZE = 16

MAPPING_FILENAME = "electrodes_mapping_MEA_MCS_256.pkl"


@dataclass(frozen=True)
class RecordingInfo:
    """What the tool needs to know about one recording."""

    path: Path
    n_channels: int
    dtype: str
    fs: float
    n_samples: int                   # samples per channel

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def describe(self) -> str:
        gb = self.size_bytes / 1e9
        mins, secs = divmod(self.duration_s, 60)
        return (f"{gb:.1f} GB · {self.n_channels} ch · {self.fs / 1000:g} kHz · "
                f"{int(mins)} min {secs:04.1f} s")


def open_info(
    path: os.PathLike | str,
    n_channels: int = DEFAULT_N_CHANNELS,
    dtype: str = DEFAULT_DTYPE,
    fs: float = DEFAULT_FS,
) -> RecordingInfo:
    """Describe a recording. Nothing is read: the sample count comes from the file size."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    item_size = np.dtype(dtype).itemsize
    n_values = path.stat().st_size // item_size
    if n_values % n_channels:
        warnings.warn(
            f"{path.name}: {n_values} samples is not a whole number of {n_channels}-channel "
            f"frames ({n_values % n_channels} left over). Is the channel count right?",
            stacklevel=2,
        )
    return RecordingInfo(path, n_channels, dtype, float(fs), n_values // n_channels)


def read_channel(
    info: RecordingInfo,
    channel_index: int,
    progress: Callable[[float], None] | None = None,
    chunk_samples: int = 4_000_000,
) -> np.ndarray:
    """One channel of the recording, in raw ADC levels (``uint16``, 32768 = 0 V).

    Read strided straight off a memory map rather than sample by sample, in chunks so the
    caller can show progress over what is still a pass across the whole file.
    """
    if not 0 <= channel_index < info.n_channels:
        raise ValueError(f"channel {channel_index} outside 0..{info.n_channels - 1}")
    mm = np.memmap(info.path, dtype=info.dtype, mode="r")
    try:
        usable = info.n_samples * info.n_channels
        frames = mm[:usable].reshape(info.n_samples, info.n_channels)
        out = np.empty(info.n_samples, dtype=info.dtype)
        for start in range(0, info.n_samples, chunk_samples):
            stop = min(start + chunk_samples, info.n_samples)
            out[start:stop] = frames[start:stop, channel_index]
            if progress is not None:
                progress(stop / info.n_samples)
    finally:
        mm._mmap.close()
    return out


def to_microvolts(levels: np.ndarray, uv_per_level: float) -> np.ndarray:
    """Raw ADC levels -> µV at the electrode."""
    return (levels.astype(np.float32) - ADC_ZERO) * uv_per_level


def as_spikeinterface(info: RecordingInfo):
    """The recording as a spikeinterface object, signed and ready to filter.

    Unsigned samples have to be made signed *before* filtering, otherwise the 32768 offset
    swamps the signal.
    """
    import spikeinterface.core as sc
    import spikeinterface.preprocessing as spre

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
