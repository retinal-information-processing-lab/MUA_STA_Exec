"""
Direct reader for Multi Channel Systems MC_Rack ``.mcd`` files.

Reverse-engineered from matched ``.mcd`` / ``mc_DataTool`` ``.raw`` pairs
(USB-MEA256, 20 kHz, 256 channels).  Pure Python + numpy, no MCS DLL,
no mc_DataTool, no Windows.

FILE FORMAT
===========

An ``.mcd`` is a RIFF-like chunked container.  Every chunk is::

    offset  size  content
    0       8     tag, 8 ASCII bytes, space padded   e.g. b"MCSHDR  "
    8       8     uint64 payload size (NOT counting these 16 header bytes)
    16      ...   payload

so a chunk occupies ``16 + size`` bytes.  ``MCSSTRM `` and any ``LIST*``
chunk are containers: their payload is a sequence of further chunks.

Top level layout::

    MCSSTRM             size = 0xFFFFFFFFFFFFFFFF  ("until EOF")
      LISThdr                                      the whole file header
        MCSHDR                                     global / hardware info
        STRMHDR                                    stream 1 descriptor  (digital)
        STRMFMT                                    stream 1 sample format
        STRMHDR                                    stream 2 descriptor  (electrodes)
        STRMFMT                                    stream 2 sample format
      IDX1POS           uint64                     file offset of the trailing index
      digi0001          \\  one "block group" per 0.1 s of recording,
      elec0001          /   one chunk per stream, repeated N times
      digi0001
      elec0001
      ...
      IDX1MCS                                      original path + sweep timestamps
      IDXIDX                                       block table: (uint64 ts, uint64 offset) * N
      IDXSWEEP                                     (uint64 start_ts, uint64 end_ts) per sweep

Data chunk payload::

    offset  size          content
    0       8             uint64 start timestamp of this block, ns
    8       8             uint64 end timestamp of this block, ns
    16      nch*ns*2      uint16 little endian samples, FRAME-MAJOR
                          i.e. sample0_ch0, sample0_ch1, ... sample0_ch255,
                               sample1_ch0, ...

That trailing payload is **exactly, byte for byte, what mc_DataTool writes
into the ``.raw`` file**.  A ``.raw`` export is nothing but the concatenation
of every ``elec0001`` chunk's data region - no conversion, no reordering, no
scaling.  So reading the ``.mcd`` directly is bit-identical to reading the
``.raw``, and :meth:`McdFile.to_raw` reproduces mc_DataTool's output.

Key parameters for this rig (all read from the file, nothing hard-coded):

* sample rate        ``MCSHDR[804]`` uint32, in milli-samples/s (20000000 -> 20 kHz)
* gain               ``MCSHDR[812]`` float32 (1000.0)
* channel count      ``MCSHDR[828]`` uint32 (256)
* block duration     ``MCSHDR[868]`` uint32 ns (1e8 -> 0.1 s -> 2000 samples)
* ADC zero           ``STRMFMT`` rec1 uint16 @6 (32768)
* V per ADC step     ``STRMFMT`` rec1 float64 @8 (1.0416667e-07 -> 0.1041667 uV/LSB)
* timestamps         nanoseconds since 1601-01-01 UTC (Windows FILETIME epoch)

Voltage is ``(sample - adc_zero) * conv_v_per_lsb``.

USAGE
=====

    from mcd import McdFile

    m = McdFile("02_Chirp_50hz_35ND40%.mcd")
    print(m)                                  # summary
    x = m.read(0, 20000)                      # (20000, 256) uint16, first second
    uv = m.read(0, 20000, as_uv=True)         # same, float32 microvolts
    trig = m.read(channels=[126])             # the trigger, whole recording

    data = m.data                             # lazy (n_samples, 256) raw-like view
    chunk = data[1000:2000, [0, 1, 2]]        # slices like a numpy memmap

CLI
===

    python mcd.py info   FILE.mcd
    python mcd.py toraw  FILE.mcd [OUT.raw]
    python mcd.py verify FILE.mcd FILE.raw     # byte-exactness check vs mc_DataTool

PROVENANCE
==========

This file is maintained in ``/mnt/Extra/mcd_format_test/``, next to
``MCD_FORMAT.md`` (the byte-level spec) and ``test_safety.py``, and is vendored
verbatim into MUA_STA as ``mua_sta/mcd.py`` so that a plain ``git clone`` of
that tool can read a recording.  The two copies are meant to stay
**identical**::

    diff /mnt/Extra/mcd_format_test/mcd.py mua_sta/mcd.py     # prints nothing

Change the upstream copy, run ``python test_safety.py FILE.mcd`` there, then
copy it over.
"""

from __future__ import annotations

import datetime as _dt
import os
import struct
import sys
from dataclasses import dataclass, field

import numpy as np

__all__ = ["McdFile", "Stream", "Channel", "MCD_EPOCH", "NS_PER_SECOND",
           "is_mcs_recording"]

#: MCS timestamps are nanoseconds since this instant (Windows FILETIME epoch).
MCD_EPOCH = _dt.datetime(1601, 1, 1, tzinfo=_dt.timezone.utc)
NS_PER_SECOND = 1_000_000_000

_CHUNK_HEADER = 16
_STREAM_FIXED = 804      # bytes of STRMHDR payload before the channel table
_CHANNEL_STRIDE = 1104   # bytes per channel entry in STRMHDR
_SIZE_UNTIL_EOF = 0xFFFFFFFFFFFFFFFF
_SYNC_FILE_RANGE_WRITE = 2

try:                                     # Linux only; optional
    import ctypes as _ctypes
    _libc = _ctypes.CDLL("libc.so.6", use_errno=True)
    _libc.sync_file_range.argtypes = [_ctypes.c_int, _ctypes.c_int64,
                                      _ctypes.c_int64, _ctypes.c_uint]
except Exception:                                            # noqa: BLE001
    _libc = None


# --------------------------------------------------------------------------- #
# low level chunk helpers
# --------------------------------------------------------------------------- #

def _read_chunk_header(fh, offset):
    """Return ``(tag, payload_size, payload_offset)`` for the chunk at *offset*."""
    fh.seek(offset)
    head = fh.read(_CHUNK_HEADER)
    if len(head) < _CHUNK_HEADER:
        raise EOFError(f"truncated chunk header at {offset}")
    tag = head[:8]
    size = struct.unpack_from("<Q", head, 8)[0]
    return tag, size, offset + _CHUNK_HEADER


def _open_readonly(path):
    """Open *path* for reading and nothing else.

    The ``.mcd`` files are the rawest copy of irreplaceable recordings, so the
    reader must never hold a writable handle to one.  ``os.O_RDONLY`` makes that
    an operating-system guarantee rather than a convention: any write through
    this descriptor fails with EBADF, whatever the code above it tries to do.

    ``O_BINARY`` is a no-op on POSIX and settles the mode of the descriptor on
    Windows, where this reader also runs: an ``os.open`` fd left in text mode
    would translate CRLF and stop at a 0x1A byte, and a recording is binary
    from end to end.
    """
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    return os.fdopen(fd, "rb", closefd=True)


def is_mcs_recording(path):
    """True if *path* is an MCS recording, judged by its content.

    Checks for the ``MCSSTRM `` magic rather than the file name, so a recording
    is still recognised when it is called ``.MCD``, ``recording.mcd.bak`` or has
    no extension at all.  Used to make sure this package never overwrites one.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == b"MCSSTRM "
    except OSError:
        return False


def _refuse_if_source(out_path, src_path=None):
    """Raise before writing if *out_path* would clobber a recording.

    MC_Rack writes ``.mcd`` files read/write and leaves them that way, so file
    permissions protect nothing: this package has to refuse on its own. Blocks
    any ``.mcd`` name, any existing file that *contains* an MCS recording
    whatever it is called, the source path itself, and symlinks or hard links
    to it (same device plus inode).
    """
    if os.path.splitext(out_path)[1].lower() == ".mcd":
        raise ValueError(
            f"refusing to write to a .mcd path: {out_path!r} "
            "(.mcd files are source recordings and are only ever read)")
    if os.path.exists(out_path) and is_mcs_recording(out_path):
        raise ValueError(
            f"refusing to overwrite {out_path!r}: it contains an MCS recording "
            "(MCSSTRM magic) even though it is not named .mcd")
    if src_path is not None:
        try:
            if os.path.realpath(out_path) == os.path.realpath(src_path):
                raise ValueError(
                    f"refusing to overwrite the source recording: {out_path!r}")
            if os.path.exists(out_path):
                a, b = os.stat(out_path), os.stat(src_path)
                if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
                    raise ValueError(
                        f"refusing to write to a hard link of the source: {out_path!r}")
        except OSError:
            pass
    return out_path


def _start_writeback(fd, offset, nbytes):
    """Ask the kernel to start writing this range out, then drop it from cache.

    Writeback otherwise only begins at ``dirty_background_ratio`` (~12 GB on a
    125 GB machine), so a 50 GB conversion buffers the whole output in RAM,
    ends with a multi-second fsync stall and evicts everything else from the
    page cache.  Hygiene only -- measured throughput is unchanged.
    Best effort; silently skipped where unsupported.
    """
    if _libc is not None:
        try:
            _libc.sync_file_range(fd, offset, nbytes, _SYNC_FILE_RANGE_WRITE)
        except Exception:                                    # noqa: BLE001
            pass
    _drop_cache(fd, offset, nbytes)


def _drop_cache(fd, offset, nbytes):
    """Drop an already-consumed byte range from the page cache."""
    try:
        os.posix_fadvise(fd, offset, nbytes, os.POSIX_FADV_DONTNEED)
    except Exception:                                        # noqa: BLE001
        pass


def _cstr(buf, offset, limit):
    """Decode the NUL-terminated string at *offset* within *buf*."""
    end = buf.find(b"\0", offset, offset + limit)
    if end < 0:
        end = offset + limit
    return buf[offset:end].decode("latin-1")


# --------------------------------------------------------------------------- #
# descriptors
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Channel:
    """One channel of a stream."""

    index: int       #: position inside the interleaved frame (0-based)
    hw_id: int       #: hardware channel id reported by the amplifier
    label: str       #: MC_Rack label, e.g. "01" ... "256"

    def __repr__(self):
        return f"Channel({self.index}, label={self.label!r})"


@dataclass
class Stream:
    """One data stream (one chunk tag) inside the file."""

    chunk_tag: bytes          #: e.g. ``b"elec0001"``
    kind: str                 #: ``"analog"`` / ``"digital"``
    name: str                 #: e.g. ``"Electrode Raw Data"``
    channels: list = field(default_factory=list)
    bits: int = 16
    adc_zero: int = 32768
    conv_v_per_lsb: float = 0.0
    block_samples: int = 0

    @property
    def n_channels(self):
        return len(self.channels)

    @property
    def labels(self):
        return [c.label for c in self.channels]

    @property
    def conv_uv_per_lsb(self):
        return self.conv_v_per_lsb * 1e6

    @property
    def bytes_per_frame(self):
        return self.n_channels * 2

    def to_uv(self, samples):
        """Convert raw uint16 ADC values to microvolts (float32)."""
        return (np.asarray(samples, dtype=np.float32) - self.adc_zero) * np.float32(
            self.conv_uv_per_lsb
        )

    def __repr__(self):
        return (
            f"Stream({self.chunk_tag.decode().strip()!r}, {self.name!r}, "
            f"{self.n_channels} ch, {self.block_samples} samples/block)"
        )


class _DataView:
    """Lazy ``(n_samples, n_channels)`` view over a stream, sliceable like a memmap."""

    def __init__(self, mcd, stream):
        self._mcd = mcd
        self._stream = stream

    @property
    def shape(self):
        return (self._mcd.n_samples, self._stream.n_channels)

    @property
    def dtype(self):
        return np.dtype("<u2")

    def __len__(self):
        return self._mcd.n_samples

    def __array__(self, dtype=None):
        out = self._mcd.read(stream=self._stream)
        return out if dtype is None else out.astype(dtype)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        tkey = key[0] if key else slice(None)
        ckey = key[1] if len(key) > 1 else slice(None)

        if isinstance(tkey, slice):
            start, stop, step = tkey.indices(self._mcd.n_samples)
        else:
            start, stop, step = int(tkey), int(tkey) + 1, 1
        if step != 1:
            block = self._mcd.read(start, stop, stream=self._stream)[::step]
        else:
            block = self._mcd.read(start, stop, stream=self._stream)

        block = block[:, ckey]
        if not isinstance(tkey, slice):
            block = block[0]
        return block


# --------------------------------------------------------------------------- #
# main reader
# --------------------------------------------------------------------------- #

class McdFile:
    """Random-access reader for an MC_Rack ``.mcd`` file."""

    def __init__(self, path, strict_map=False):
        self.strict_map = bool(strict_map)
        self.path = str(path)
        self.file_size = os.path.getsize(self.path)
        self._fh = _open_readonly(self.path)

        self.streams: list[Stream] = []
        self.source_path = ""
        self.hardware = ""
        self.serial = ""
        self.gain = 0.0
        self.sample_rate = 0.0
        self.block_duration_ns = 0
        self.sweeps: list[tuple[int, int]] = []
        self._block_offsets = np.empty(0, dtype=np.int64)
        self._block_timestamps = np.empty(0, dtype=np.uint64)
        self._maps: dict[bytes, tuple] = {}

        self._parse_header()
        self._parse_index()

    # -- context manager ---------------------------------------------------- #

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- header ------------------------------------------------------------- #

    def _parse_header(self):
        fh = self._fh
        tag, size, _ = _read_chunk_header(fh, 0)
        if tag != b"MCSSTRM ":
            raise ValueError(f"{self.path}: not an MCS stream file (tag={tag!r})")

        tag, size, payload = _read_chunk_header(fh, _CHUNK_HEADER)
        if not tag.startswith(b"LIST"):
            raise ValueError(f"{self.path}: expected LIST header chunk, got {tag!r}")
        self._header_end = payload + size

        offset = payload
        pending_stream = None
        while offset < self._header_end:
            tag, size, body = _read_chunk_header(fh, offset)
            fh.seek(body)
            buf = fh.read(size)
            if tag == b"MCSHDR  ":
                self._parse_mcshdr(buf)
            elif tag == b"STRMHDR ":
                pending_stream = self._parse_strmhdr(buf)
                self.streams.append(pending_stream)
            elif tag == b"STRMFMT ":
                if pending_stream is None:
                    raise ValueError("STRMFMT without preceding STRMHDR")
                self._parse_strmfmt(buf, pending_stream)
                pending_stream = None
            offset = body + size

        if not self.streams:
            raise ValueError(f"{self.path}: no streams found in header")

    def _parse_mcshdr(self, buf):
        self.source_path = _cstr(buf, 12, 512)
        rate_millisamples = struct.unpack_from("<I", buf, 804)[0]
        self.sample_rate = rate_millisamples / 1000.0
        self.gain = struct.unpack_from("<f", buf, 812)[0]
        self.block_duration_ns = struct.unpack_from("<I", buf, 868)[0]
        self._hdr_start_ts = struct.unpack_from("<Q", buf, 872)[0]
        self._hdr_end_ts = struct.unpack_from("<Q", buf, 848)[0]
        self.hardware = _cstr(buf, 2208, 64)
        self.serial = _cstr(buf, 2288, 32)

    def _parse_strmhdr(self, buf):
        kind = _cstr(buf, 2, 32)
        name = _cstr(buf, 11, 64)
        chunk_tag = buf[792:800]
        n_channels, rem = divmod(len(buf) - _STREAM_FIXED, _CHANNEL_STRIDE)
        if rem:
            raise ValueError(
                f"STRMHDR payload {len(buf)} is not {_STREAM_FIXED} + n*{_CHANNEL_STRIDE}"
            )
        channels = []
        for k in range(n_channels):
            base = _STREAM_FIXED + k * _CHANNEL_STRIDE
            entry = buf[base : base + _CHANNEL_STRIDE]
            index = struct.unpack_from("<I", entry, 4)[0]
            hw_id = struct.unpack_from("<I", entry, 8)[0]
            label = _cstr(entry, 12, 64)
            channels.append(Channel(index=index, hw_id=hw_id, label=label))
        return Stream(chunk_tag=chunk_tag, kind=kind, name=name, channels=channels)

    @staticmethod
    def _parse_strmfmt(buf, stream):
        # two 16-byte records; rec1 = ADC format, rec2 = block format
        _, _, bits, adc_zero = struct.unpack_from("<4H", buf, 0)
        conv = struct.unpack_from("<d", buf, 8)[0]
        block_samples = struct.unpack_from("<H", buf, 16 + 2)[0]
        stream.bits = bits
        stream.adc_zero = adc_zero
        stream.conv_v_per_lsb = conv
        stream.block_samples = block_samples

    # -- index -------------------------------------------------------------- #

    def _parse_index(self):
        fh = self._fh
        tag, size, body = _read_chunk_header(fh, self._header_end)
        if tag != b"IDX1POS ":
            raise ValueError(f"expected IDX1POS after header, got {tag!r}")
        fh.seek(body)
        index_pos = struct.unpack("<Q", fh.read(8))[0]
        self.data_start = body + size

        offset = index_pos
        while offset + _CHUNK_HEADER <= self.file_size:
            tag, size, payload = _read_chunk_header(fh, offset)
            fh.seek(payload)
            buf = fh.read(size)
            if tag == b"IDXIDX  ":
                table = np.frombuffer(buf, dtype="<u8").reshape(-1, 2)
                self._block_timestamps = table[:, 0].copy()
                self._block_offsets = table[:, 1].astype(np.int64)
            elif tag == b"IDXSWEEP":
                sw = np.frombuffer(buf, dtype="<u8").reshape(-1, 2)
                self.sweeps = [(int(a), int(b)) for a, b in sw]
            offset = payload + size

        if self._block_offsets.size == 0:
            raise ValueError(f"{self.path}: no IDXIDX block table found")

    # -- block maps --------------------------------------------------------- #

    def _scan_group(self, group_start, group_end, key, bpf):
        """Find *key*'s data region inside one block group.

        Returns ``(data_offset, n_samples)`` or ``None`` if the stream has no
        chunk in this group.
        """
        offset = int(group_start)
        end = int(group_end)
        while offset + _CHUNK_HEADER <= end:
            tag, size, payload = _read_chunk_header(self._fh, offset)
            if tag == key:
                return payload + 16, (size - 16) // bpf   # skip the 2 timestamps
            offset = payload + size
        return None

    def _map(self, stream):
        """Build/return ``(data_offsets, n_samples_per_block, sample_starts)``.

        A full scan costs two syscalls per block group, which on a 55 GB / 54k
        block recording means ~110k seeks just to open the file (~45 s cold).
        MC_Rack writes fixed-size blocks, so when the IDXIDX offsets are evenly
        spaced the whole map can be computed arithmetically after probing a
        sample of groups.  Pass ``strict_map=True`` to McdFile to force the
        exhaustive scan instead.
        """
        key = stream.chunk_tag
        cached = self._maps.get(key)
        if cached is not None:
            return cached

        starts = self._block_offsets
        n_blocks = starts.size
        bpf = stream.bytes_per_frame

        group_ends = np.empty(n_blocks, dtype=np.int64)
        group_ends[:-1] = starts[1:]
        group_ends[-1] = self._first_trailing_offset()

        offsets = counts = None

        # ---- fast path: evenly spaced groups with a verified common layout --
        if not self.strict_map and n_blocks >= 2:
            diffs = np.diff(starts)
            stride = int(diffs[0])
            if np.all(diffs == stride):
                # probe both ends plus a spread of groups through the file
                probes = set(int(i) for i in
                             np.linspace(0, n_blocks - 1, min(n_blocks, 256)))
                probes.update({0, 1, n_blocks - 2, n_blocks - 1})
                probes = sorted(i for i in probes if 0 <= i < n_blocks)
                layouts = {}
                for i in probes:
                    r = self._scan_group(starts[i], group_ends[i], key, bpf)
                    if r is None:
                        layouts = None
                        break
                    layouts[i] = (int(r[0] - starts[i]), int(r[1]))
                # every probed group but the last must share one layout
                if layouts:
                    common = {v for i, v in layouts.items() if i != n_blocks - 1}
                    if len(common) == 1:
                        data_off, cnt = common.pop()
                        offsets = starts.astype(np.int64) + data_off
                        counts = np.full(n_blocks, cnt, dtype=np.int64)
                        # the final block is allowed to be short
                        last_off, last_cnt = layouts[n_blocks - 1]
                        offsets[-1] = int(starts[-1]) + last_off
                        counts[-1] = last_cnt

        # ---- exhaustive scan ------------------------------------------------
        if offsets is None:
            offsets = np.empty(n_blocks, dtype=np.int64)
            counts = np.empty(n_blocks, dtype=np.int64)
            for i in range(n_blocks):
                r = self._scan_group(starts[i], group_ends[i], key, bpf)
                if r is None:
                    offsets[i], counts[i] = -1, 0
                else:
                    offsets[i], counts[i] = r

        sample_starts = np.zeros(n_blocks + 1, dtype=np.int64)
        np.cumsum(counts, out=sample_starts[1:])
        cached = (offsets, counts, sample_starts)
        self._maps[key] = cached
        return cached

    def _first_trailing_offset(self):
        """Offset just past the last data block (where the index chunks begin)."""
        fh = self._fh
        tag, size, body = _read_chunk_header(fh, self._header_end)
        fh.seek(body)
        return struct.unpack("<Q", fh.read(8))[0]

    # -- public metadata ---------------------------------------------------- #

    def stream(self, which=None):
        """Return a :class:`Stream` by tag, name, kind or index.

        With *which* omitted, returns the stream with the most channels
        (the electrode stream on a normal MEA recording).
        """
        if which is None:
            return max(self.streams, key=lambda s: s.n_channels)
        if isinstance(which, Stream):
            return which
        if isinstance(which, int):
            return self.streams[which]
        if isinstance(which, bytes):
            which = which.decode()
        key = which.strip().lower()
        for s in self.streams:
            if s.chunk_tag.decode().strip().lower() == key:
                return s
        for s in self.streams:
            if s.name.lower() == key or s.kind.lower() == key:
                return s
        raise KeyError(f"no stream matching {which!r}; have {[s.chunk_tag for s in self.streams]}")

    @property
    def n_blocks(self):
        return int(self._block_offsets.size)

    @property
    def n_samples(self):
        """Total number of samples (frames) of the main stream."""
        return int(self._map(self.stream())[2][-1])

    @property
    def duration_s(self):
        return self.n_samples / self.sample_rate

    @property
    def start_timestamp_ns(self):
        return int(self.sweeps[0][0]) if self.sweeps else int(self._hdr_start_ts)

    @property
    def start_datetime(self):
        """Recording start as a timezone-aware UTC ``datetime``."""
        return MCD_EPOCH + _dt.timedelta(
            microseconds=self.start_timestamp_ns / 1000.0
        )

    def sample_times(self, start=0, stop=None):
        """Seconds from recording start for samples ``[start, stop)``."""
        stop = self.n_samples if stop is None else stop
        return np.arange(start, stop, dtype=np.float64) / self.sample_rate

    @property
    def data(self):
        """Lazy ``(n_samples, n_channels)`` view of the main stream."""
        return _DataView(self, self.stream())

    def view(self, which=None):
        """Lazy ``(n_samples, n_channels)`` view of any stream."""
        return _DataView(self, self.stream(which))

    # -- reading ------------------------------------------------------------ #

    def read(self, start=0, stop=None, channels=None, stream=None, as_uv=False):
        """Read samples ``[start, stop)`` as a ``(n, n_channels)`` array.

        Parameters
        ----------
        start, stop : int
            Sample range, like a Python slice.  ``stop=None`` reads to the end.
        channels : sequence of int or str, optional
            Channel indices (0-based) or MC_Rack labels.  Defaults to all.
        stream : str / bytes / int / Stream, optional
            Which stream to read.  Defaults to the electrode stream.
        as_uv : bool
            Return float32 microvolts instead of raw uint16 ADC values.
        """
        st = self.stream(stream)
        offsets, counts, sample_starts = self._map(st)
        total = int(sample_starts[-1])

        stop = total if stop is None else min(int(stop), total)
        start = max(int(start), 0)
        if stop <= start:
            shape = (0, st.n_channels if channels is None else len(channels))
            return np.empty(shape, dtype=np.float32 if as_uv else np.uint16)

        cols = self._resolve_channels(st, channels)
        nch = st.n_channels
        # dtype is explicitly little-endian so readinto() below stays correct
        # on a big-endian host as well.
        out = np.empty((stop - start, nch if cols is None else len(cols)), dtype="<u2")

        first = int(np.searchsorted(sample_starts, start, side="right") - 1)
        last = int(np.searchsorted(sample_starts, stop, side="left"))
        fh = self._fh
        written = 0
        for b in range(first, last):
            if offsets[b] < 0 or counts[b] == 0:
                continue
            b0 = int(sample_starts[b])
            lo = max(start, b0)
            hi = min(stop, int(sample_starts[b + 1]))
            if hi <= lo:
                continue
            fh.seek(int(offsets[b]) + (lo - b0) * st.bytes_per_frame)
            if cols is None:
                # read straight into the output buffer, no intermediate copy
                dst = out[written : written + (hi - lo)]
                if fh.readinto(memoryview(dst).cast("B")) != dst.nbytes:
                    raise EOFError(f"short read in block {b} of {self.path}")
            else:
                buf = fh.read((hi - lo) * st.bytes_per_frame)
                frames = np.frombuffer(buf, dtype="<u2").reshape(hi - lo, nch)
                out[written : written + (hi - lo)] = frames[:, cols]
            written += hi - lo

        out = out[:written]
        return st.to_uv(out) if as_uv else out

    @staticmethod
    def _resolve_channels(stream, channels):
        if channels is None:
            return None
        by_label = {c.label: c.index for c in stream.channels}
        cols = []
        for ch in np.atleast_1d(channels):
            if isinstance(ch, (str, bytes)):
                key = ch.decode() if isinstance(ch, bytes) else ch
                if key not in by_label:
                    raise KeyError(f"no channel labelled {key!r}")
                cols.append(by_label[key])
            else:
                cols.append(int(ch))
        return np.asarray(cols, dtype=np.intp)

    def blocks(self, stream=None, start_block=0, stop_block=None,
               span_blocks=16, prefetch=True, copy=True):
        """Iterate ``(sample_index, timestamp_ns, frames)`` block by block.

        The memory-safe way to process a 19 GB recording: only a few blocks are
        ever resident.  A reader thread fetches the next span while you work on
        the current one, so filtering overlaps with I/O instead of alternating
        with it.

        Parameters
        ----------
        span_blocks : int
            How many block groups to fetch per read.  Bigger spans mean fewer,
            larger reads; 16-32 is the sweet spot on NVMe.
        prefetch : bool
            Run the reads in a background thread.  Turn off for deterministic
            single-threaded behaviour.
        copy : bool
            ``True`` (default) yields an independent array per block, safe to
            keep.  ``False`` yields a view into the shared read buffer, which
            is faster but **is overwritten a few iterations later** -- only use
            it when each block is consumed before the next is requested (e.g.
            you immediately ``astype`` or accumulate it).
        """
        st = self.stream(stream)
        offsets, counts, sample_starts = self._map(st)
        n = self.n_blocks
        stop_block = n if stop_block is None else min(int(stop_block), n)
        start_block = max(int(start_block), 0)
        layout = self._uniform_layout(st)
        nch = st.n_channels

        if layout is None or stop_block - start_block < 2:   # simple path
            fh = self._fh
            for b in range(start_block, stop_block):
                if offsets[b] < 0 or counts[b] == 0:
                    continue
                fh.seek(int(offsets[b]))
                buf = fh.read(int(counts[b]) * st.bytes_per_frame)
                yield (int(sample_starts[b]), int(self._block_timestamps[b]),
                       np.frombuffer(buf, dtype="<u2").reshape(int(counts[b]), nch))
            return

        stride, data_off, data_len = layout
        span_blocks = max(1, int(span_blocks))
        span = span_blocks * stride
        index_start = self._first_trailing_offset()
        depth = 3 if prefetch else 1
        bufs = [bytearray(span) for _ in range(depth)]
        views = [memoryview(b) for b in bufs]

        def read_span(i, mv):
            j = min(i + span_blocks, stop_block)
            begin = int(self._block_offsets[i])
            want = min((j - i) * stride, index_start - begin)
            fh.seek(begin)
            got = 0
            while got < want:
                k = fh.readinto(mv[got:want])
                if not k:
                    break
                got += k
            return j, got

        def unpack(mv, i, got):
            """Yield each block held in the span buffer."""
            for q in range((got + stride - 1) // stride):
                b = i + q
                if b >= stop_block:
                    return
                base = q * stride + data_off
                avail = min(data_len, max(0, got - (q * stride + data_off)))
                if avail < data_len:
                    return
                frames = np.frombuffer(mv, dtype="<u2", count=data_len // 2,
                                       offset=base).reshape(int(counts[b]), nch)
                yield (int(sample_starts[b]), int(self._block_timestamps[b]),
                       frames.copy() if copy else frames)

        fh = _open_readonly(self.path)
        try:
            if not prefetch:
                i = start_block
                while i < stop_block:
                    nxt, got = read_span(i, views[0])
                    yield from unpack(views[0], i, got)
                    i = nxt
                return

            import queue as _queue
            import threading as _threading

            free_q, full_q = _queue.Queue(), _queue.Queue(maxsize=depth)
            for k in range(depth):
                free_q.put(k)
            err = []

            def reader():
                try:
                    i = start_block
                    while i < stop_block:
                        k = free_q.get()
                        if k is None:
                            return
                        i, got = read_span(i, views[k])
                        full_q.put((k, i, got))
                except Exception as exc:                     # noqa: BLE001
                    err.append(exc)
                finally:
                    full_q.put(None)

            t = _threading.Thread(target=reader, daemon=True)
            t.start()
            try:
                i = start_block
                while True:
                    item = full_q.get()
                    if item is None:
                        break
                    k, nxt, got = item
                    yield from unpack(views[k], i, got)
                    i = nxt
                    free_q.put(k)
                if err:
                    raise err[0]
            finally:
                # unblock the reader if the consumer abandons the generator
                free_q.put(None)
        finally:
            fh.close()

    # -- triggers ----------------------------------------------------------- #

    def channel_levels(self, channel, stream=None, bins=32, step=1):
        """Summarise the voltage levels present on one channel.

        Returns ``(levels, counts)`` for a coarse histogram, which is the quick
        way to choose a trigger threshold by eye.  *step* > 1 subsamples the
        channel, which is plenty for seeing the levels of a TTL-like line.
        """
        v = self.read(channels=[channel], stream=stream).ravel()
        if step > 1:
            v = v[::step]
        counts, edges = np.histogram(v, bins=bins)
        return edges[:-1], counts

    def auto_threshold(self, channel, stream=None, step=7):
        """Guess a trigger threshold for a mostly-idle, pulsing channel.

        Two-stage heuristic: take the median as the idle level, discard far-away
        plateaus (such as a stimulus-off baseline), then put the threshold half
        way between the idle level and the nearest excursion level.

        This is a guess.  Check it against :meth:`channel_levels` and pass an
        explicit ``threshold`` to :meth:`trigger_edges` for anything that
        matters -- a channel with three or more distinct levels is genuinely
        ambiguous and no heuristic can resolve it for you.
        """
        v = self.read(channels=[channel], stream=stream).ravel()[::step].astype(np.float64)
        baseline = np.median(v)
        far = np.percentile(v, 0.01)
        if baseline == far:
            return float(baseline)
        # stage 1: ignore anything beyond half way to the most extreme level
        midpoint = (baseline + far) / 2.0
        active = v[v >= midpoint] if far < baseline else v[v <= midpoint]
        if active.size == 0:
            return float(midpoint)
        # stage 2: half way to the nearest excursion level
        near = np.percentile(active, 0.01 if far < baseline else 99.99)
        return float((baseline + near) / 2.0)

    def trigger_edges(self, channel, threshold=None, edge="falling", stream=None):
        """Return sample indices where *channel* crosses *threshold*.

        Parameters
        ----------
        channel : int or str
            Channel index or MC_Rack label.  On this rig the trigger line is
            index 126 (label ``"127"``).
        threshold : float, optional
            Crossing level.  ``None`` calls :meth:`auto_threshold` -- always
            sanity-check the resulting rate before trusting it.
        edge : {"falling", "rising", "both"}
            Which crossings to report.  The stimulus monitor on this rig idles
            high and dips, so ``"falling"`` marks the stimulus events.
        """
        if threshold is None:
            threshold = self.auto_threshold(channel, stream=stream)
        v = self.read(channels=[channel], stream=stream).ravel().astype(np.float64)
        above = v >= threshold
        falling = np.flatnonzero(above[:-1] & ~above[1:]) + 1
        rising = np.flatnonzero(~above[:-1] & above[1:]) + 1
        if edge == "falling":
            out = falling
        elif edge == "rising":
            out = rising
        elif edge == "both":
            out = np.sort(np.concatenate([rising, falling]))
        else:
            raise ValueError("edge must be 'falling', 'rising' or 'both'")
        return out

    # -- export ------------------------------------------------------------- #

    def _uniform_layout(self, stream):
        """Return (stride, data_offset, data_len) if every block group is
        identical in shape, else None.  Lets to_raw() take the fast path."""
        offsets, counts, _ = self._map(stream)
        starts = self._block_offsets
        if len(starts) < 2 or np.any(offsets < 0):
            return None
        stride = int(starts[1] - starts[0])
        data_off = int(offsets[0] - starts[0])
        data_len = int(counts[0]) * stream.bytes_per_frame
        if (np.all(np.diff(starts) == stride)
                and np.all(offsets - starts == data_off)
                and np.all(counts == counts[0])):
            return stride, data_off, data_len
        return None

    def to_raw(self, out_path, stream=None, progress=None, span_blocks=32,
               threaded=True, fsync=True):
        """Write the mc_DataTool-compatible headerless ``.raw`` file.

        The output is byte-for-byte what ``mc_DataTool`` produces: interleaved
        little-endian uint16, frame-major, no header.

        Reads whole spans of ``span_blocks`` block groups at a time into a
        reused buffer and gathers the sample regions out of it.  The reuse
        matters far more than it looks: allocating a fresh buffer per span and
        calling ``ndarray.tobytes()`` on the strided view runs at ~740 MB/s
        because of page-fault churn, while copying into a preallocated
        destination runs at ~26 GB/s.  With *threaded*, a reader thread keeps
        the drive busy while the main thread gathers and writes.
        """
        _refuse_if_source(out_path, self.path)
        st = self.stream(stream)
        offsets, counts, sample_starts = self._map(st)
        layout = self._uniform_layout(st)
        n = self.n_blocks

        if layout is None:                       # mixed block sizes: simple path
            written = 0
            with open(out_path, "wb", buffering=1024 * 1024) as out:
                for b in range(n):
                    if offsets[b] < 0 or counts[b] == 0:
                        continue
                    self._fh.seek(int(offsets[b]))
                    out.write(self._fh.read(int(counts[b]) * st.bytes_per_frame))
                    written += int(counts[b])
                    if progress is not None and b % 500 == 0:
                        progress(b, n)
                out.flush()
                if fsync:
                    os.fsync(out.fileno())
            if progress is not None:
                progress(n, n)
            return written

        stride, data_off, data_len = layout
        span_blocks = max(1, int(span_blocks))
        span = span_blocks * stride
        index_start = self._first_trailing_offset()

        depth = 3 if threaded else 1
        bufs = [bytearray(span) for _ in range(depth)]
        views = [memoryview(b) for b in bufs]
        dst = np.empty((span_blocks, data_len), dtype=np.uint8)
        dst_mv = memoryview(dst.reshape(-1))

        fh = _open_readonly(self.path)
        out = open(out_path, "wb", buffering=0)

        def read_span(i, mv):
            j = min(i + span_blocks, n)
            begin = int(self._block_offsets[i])
            want = min((j - i) * stride, index_start - begin)
            fh.seek(begin)
            got = 0
            while got < want:
                k = fh.readinto(mv[got:want])
                if not k:
                    break
                got += k
            # a one-pass stream: do not let 50 GB of input sit in the cache
            _drop_cache(fh.fileno(), begin, got)
            return j, got

        state = {"written": 0, "synced": 0}
        flush_every = 64 << 20

        def emit(mv, got):
            full = got // stride
            if full:
                flat = np.frombuffer(mv, dtype=np.uint8, count=full * stride)
                np.copyto(dst[:full],
                          flat.reshape(full, stride)[:, data_off:data_off + data_len])
                out.write(dst_mv[:full * data_len])
                state["written"] += full * data_len
            tail = got - full * stride
            if tail:
                base = full * stride + data_off
                out.write(mv[base:base + data_len])
                state["written"] += data_len
            if state["written"] - state["synced"] >= flush_every:
                _start_writeback(out.fileno(), state["synced"],
                                 state["written"] - state["synced"])
                state["synced"] = state["written"]
            return full + (1 if tail else 0)

        done = 0
        try:
            if not threaded:
                i = 0
                while i < n:
                    i, got = read_span(i, views[0])
                    done += emit(views[0], got)
                    if progress is not None:
                        progress(done, n)
            else:
                import queue as _queue
                import threading as _threading

                free_q, full_q = _queue.Queue(), _queue.Queue(maxsize=depth)
                for k in range(depth):
                    free_q.put(k)
                err = []

                def reader():
                    try:
                        i = 0
                        while i < n:
                            k = free_q.get()
                            i, got = read_span(i, views[k])
                            full_q.put((k, got))
                    except Exception as exc:                  # noqa: BLE001
                        err.append(exc)
                    finally:
                        full_q.put(None)

                t = _threading.Thread(target=reader, daemon=True)
                t.start()
                while True:
                    item = full_q.get()
                    if item is None:
                        break
                    k, got = item
                    done += emit(views[k], got)
                    free_q.put(k)
                    if progress is not None:
                        progress(done, n)
                t.join()
                if err:
                    raise err[0]
            out.flush()
            if fsync:
                os.fsync(out.fileno())
        finally:
            fh.close()
            out.close()

        if progress is not None:
            progress(n, n)
        return int(sample_starts[-1])

    def verify_raw(self, raw_path, stream=None, n_checks=40, rng=None):
        """Check that an existing ``.raw`` matches this ``.mcd`` byte for byte.

        Compares the total size plus *n_checks* random 2000-frame windows
        (always including the first and last).  Returns a dict of results.
        """
        st = self.stream(stream)
        total = self.n_samples
        expected = total * st.bytes_per_frame
        actual = os.path.getsize(raw_path)
        result = {
            "raw_size": actual,
            "expected_size": expected,
            "size_ok": actual == expected,
            "n_windows": 0,
            "mismatched_windows": [],
            "ok": False,
        }
        if not result["size_ok"]:
            return result

        rng = np.random.default_rng(0 if rng is None else rng)
        win = min(2000, total)
        picks = {0, max(0, total - win)}
        while len(picks) < min(n_checks, max(1, total // win)):
            picks.add(int(rng.integers(0, total - win + 1)))
        for s in sorted(picks):
            mine = self.read(s, s + win, stream=st)
            theirs = np.fromfile(
                raw_path, dtype="<u2", count=win * st.n_channels,
                offset=s * st.bytes_per_frame,
            ).reshape(win, st.n_channels)
            result["n_windows"] += 1
            if not np.array_equal(mine, theirs):
                result["mismatched_windows"].append(s)
        result["ok"] = not result["mismatched_windows"]
        return result

    # -- repr --------------------------------------------------------------- #

    def describe(self):
        lines = [
            f"file           {self.path}",
            f"size           {self.file_size:,} bytes",
            f"recorded by    {self.hardware} (serial {self.serial})",
            f"source path    {self.source_path}",
            f"sample rate    {self.sample_rate:g} Hz",
            f"gain           {self.gain:g}",
            f"start (UTC)    {self.start_datetime:%Y-%m-%d %H:%M:%S}",
            f"blocks         {self.n_blocks:,} x {self.block_duration_ns / 1e6:g} ms",
            f"samples        {self.n_samples:,}  ({self.duration_s:.3f} s)",
            f"sweeps         {len(self.sweeps)}",
            "streams:",
        ]
        for s in self.streams:
            offsets, counts, sample_starts = self._map(s)
            lines.append(
                f"  {s.chunk_tag.decode().strip():10s} {s.kind:8s} {s.name!r}"
            )
            lines.append(
                f"    {s.n_channels} channels, {s.bits} bit, ADC zero {s.adc_zero}, "
                f"{s.conv_uv_per_lsb:.6g} uV/LSB"
            )
            lines.append(
                f"    {int(sample_starts[-1]):,} samples, labels "
                f"{s.labels[0]!r}..{s.labels[-1]!r}"
            )
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"<McdFile {os.path.basename(self.path)!r} "
            f"{self.sample_rate:g} Hz, {self.stream().n_channels} ch, "
            f"{self.n_samples:,} samples>"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cli(argv):
    if len(argv) < 3:
        print(__doc__.strip().split("CLI")[-1])
        return 2
    cmd, path = argv[1], argv[2]
    with McdFile(path) as m:
        if cmd == "info":
            print(m.describe())
        elif cmd == "toraw":
            out = argv[3] if len(argv) > 3 else os.path.splitext(path)[0] + ".raw"

            def progress(b, n):
                pct = 100.0 * b / max(n, 1)
                print(f"\r  {pct:5.1f}%  block {b:,}/{n:,}", end="", flush=True)

            n = m.to_raw(out, progress=progress)
            print(f"\nwrote {out}: {n:,} samples x {m.stream().n_channels} channels")
        elif cmd == "verify":
            if len(argv) < 4:
                print("usage: python mcd.py verify FILE.mcd FILE.raw")
                return 2
            r = m.verify_raw(argv[3])
            print(f"raw size      {r['raw_size']:,}")
            print(f"expected      {r['expected_size']:,}")
            print(f"size match    {r['size_ok']}")
            print(f"windows       {r['n_windows']} checked")
            print(f"mismatches    {r['mismatched_windows'] or 'none'}")
            print("RESULT        " + ("BIT-IDENTICAL" if r["ok"] else "DIFFERENT"))
            return 0 if r["ok"] else 1
        else:
            print(f"unknown command {cmd!r}")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
