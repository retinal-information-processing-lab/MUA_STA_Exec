"""An ``.mcd`` as a spikeinterface recording, for the MUA detection pass.

Only :func:`mua_sta.recording.as_spikeinterface` imports this, and it does so lazily, so
choosing a file in the GUI still costs nothing: importing spikeinterface is seconds of work.

Why this is a module of its own rather than a class inside that function: spikeinterface runs
peak detection in a pool of worker processes, and on Windows that pool is started with
``spawn``, which **pickles** the recording. ``BaseExtractor.__reduce__`` rebuilds it from
``self._kwargs`` (``base.py:819``), so the pickled form is just the file path, and the class has
to be importable by name for that to work — a class defined inside a function is not.

The open file handle must therefore never be part of the object's state. :class:`McdFile` holds
a ``BufferedReader`` and cannot be pickled at all; here each segment opens its own, lazily,
one per process **and** per thread, because ``McdFile.read`` seeks a shared handle and two
threads sharing one would silently read each other's data. spikeinterface's default pool is
processes, but ``get_best_job_kwargs()`` picks threads on Windows, so both have to be right.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment

from . import mcd
from .recording import MCD_DTYPE


class McdRecording(BaseRecording):
    """The electrode stream of an ``.mcd``, in raw ADC levels (``uint16``, 32768 = 0 V).

    Deliberately as bare as :func:`spikeinterface.core.read_binary` on the matching ``.raw``:
    same dtype, same channel order, same integer channel ids, and no gains or offsets set, so
    the two paths hand exactly the same numbers to the filtering and detection that follow.
    Channel *i* is MCS label *i + 1*; :func:`mua_sta.recording.channel_index_to_label` is the
    only place that conversion is written down.
    """

    def __init__(self, file_path: os.PathLike | str):
        path = Path(file_path)
        with mcd.McdFile(path) as m:
            stream = m.stream()     # the electrode stream: the one with the most channels
            n_channels = stream.n_channels
            sampling_frequency = float(m.sample_rate)
            n_samples = int(m.n_samples)
        BaseRecording.__init__(self, sampling_frequency=sampling_frequency,
                               channel_ids=np.arange(n_channels), dtype=MCD_DTYPE)
        self.add_recording_segment(
            McdRecordingSegment(path, sampling_frequency, n_samples))
        # what __reduce__ sends to a worker process, and all it sends
        self._kwargs = {"file_path": str(path)}


class McdRecordingSegment(BaseRecordingSegment):
    """One sweep of the recording. Opens the file on first use in each process and thread."""

    def __init__(self, path: Path, sampling_frequency: float, n_samples: int):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self.file_path = str(path)
        self.n_samples = n_samples
        self._readers: dict[tuple[int, int], mcd.McdFile] = {}

    def get_num_samples(self) -> int:
        return self.n_samples

    def _reader(self) -> mcd.McdFile:
        """This process and thread's own reader. Opening costs the index scan, ~1.5 s on
        19 GB, once per worker — the pool reuses its workers, so it is paid once."""
        key = (os.getpid(), threading.get_ident())
        reader = self._readers.get(key)
        if reader is None:
            reader = self._readers[key] = mcd.McdFile(self.file_path)
        return reader

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None) -> np.ndarray:
        start = 0 if start_frame is None else int(start_frame)
        stop = self.n_samples if end_frame is None else int(end_frame)
        traces = self._reader().read(start, stop)
        if channel_indices is not None:
            traces = traces[:, channel_indices]
        return traces
