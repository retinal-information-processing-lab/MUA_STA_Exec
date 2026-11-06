"""Write a minimal but valid ``.mcd``, so the reader can be tested without a 19 GB recording.

Built straight from ``MCD_FORMAT.md`` (``/mnt/Extra/mcd_format_test/``), which makes this the
only place in the repo that writes the format rather than reading it: a test that passes here
says the spec and the reader agree on where every field lives.

What a reader needs to find, and therefore what this writes:

* ``MCSSTRM `` with the "until EOF" size, then a ``LISThdr`` holding ``MCSHDR`` (sampling
  rate, gain, block duration, device) and one ``STRMHDR``/``STRMFMT`` pair per stream;
* ``IDX1POS``, a single ``uint64`` saying where the trailing index starts — the reader uses it
  both to find the index and to know where the last block group ends;
* one **block group** per block, holding one chunk per stream, each chunk being two
  timestamps then frame-major ``uint16`` samples;
* ``IDXIDX`` at the end: ``(timestamp, group offset)`` for every group.

A digital stream is written alongside the electrode one, with a single channel, because a real
recording has one and the reader has to pick the electrode stream out of the two.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

CHUNK_HEADER = 16
STREAM_FIXED = 804              # bytes of STRMHDR payload before the channel table
CHANNEL_STRIDE = 1104           # bytes per channel entry in STRMHDR
SIZE_UNTIL_EOF = 0xFFFFFFFFFFFFFFFF
MCSHDR_SIZE = 2400              # enough to hold the serial at offset 2288

ADC_ZERO = 32768
UV_PER_LEVEL = 0.10416666666666667
V_PER_LEVEL = UV_PER_LEVEL * 1e-6

#: 1601-01-01 is the epoch, and this is an arbitrary but plausible start time.
START_TS = 13_409_038_000_000_000_00


def _chunk(tag: bytes, payload: bytes) -> bytes:
    assert len(tag) == 8, tag
    return tag + struct.pack("<Q", len(payload)) + payload


def _put_cstr(buf: bytearray, offset: int, text: str, limit: int):
    """NUL-terminated ASCII, written only as far as it goes — several of these fields
    overlap the *limit* of the one before, exactly as they do in a real header."""
    data = text.encode("ascii")
    assert len(data) < limit, f"{text!r} does not fit in {limit} bytes"
    buf[offset:offset + len(data) + 1] = data + b"\0"


def _mcshdr(source_path: str, fs: float, gain: float, block_duration_ns: int,
            start_ts: int, end_ts: int, hardware: str, serial: str) -> bytes:
    buf = bytearray(MCSHDR_SIZE)
    _put_cstr(buf, 12, source_path, 512)
    struct.pack_into("<I", buf, 804, round(fs * 1000))      # milli-samples per second
    struct.pack_into("<f", buf, 812, gain)
    struct.pack_into("<Q", buf, 848, end_ts)
    struct.pack_into("<I", buf, 868, block_duration_ns)
    struct.pack_into("<Q", buf, 872, start_ts)
    _put_cstr(buf, 2208, hardware, 64)
    _put_cstr(buf, 2288, serial, 32)
    return _chunk(b"MCSHDR  ", bytes(buf))


def _strmhdr(kind: str, name: str, chunk_tag: bytes, labels: list[str]) -> bytes:
    buf = bytearray(STREAM_FIXED + len(labels) * CHANNEL_STRIDE)
    _put_cstr(buf, 2, kind, 9)          # the name field starts at 11, so this must be short
    _put_cstr(buf, 11, name, 64)
    buf[792:800] = chunk_tag
    for index, label in enumerate(labels):
        base = STREAM_FIXED + index * CHANNEL_STRIDE
        struct.pack_into("<I", buf, base + 4, index)
        struct.pack_into("<I", buf, base + 8, index + 1)        # hardware id
        _put_cstr(buf, base + 12, label, 64)
    return _chunk(b"STRMHDR ", bytes(buf))


def _strmfmt(bits: int, adc_zero: int, v_per_level: float, block_samples: int) -> bytes:
    buf = bytearray(32)                 # two 16-byte records: ADC format, then block format
    struct.pack_into("<4H", buf, 0, 0, 0, bits, adc_zero)
    struct.pack_into("<d", buf, 8, v_per_level)
    struct.pack_into("<H", buf, 18, block_samples)
    return _chunk(b"STRMFMT ", bytes(buf))


def write_mcd(
    path: Path | str,
    frames: np.ndarray,
    *,
    fs: float = 20_000.0,
    block_samples: int = 25,
    labels: list[str] | None = None,
    v_per_level: float = V_PER_LEVEL,
    adc_zero: int = ADC_ZERO,
    gain: float = 1000.0,
    with_digital: bool = True,
    source_path: str = r"D:\synthetic\test.mcd",
    hardware: str = "USB-MEA256",
    serial: str = "00043",
    start_ts: int = START_TS,
) -> Path:
    """Write ``frames`` — ``(n_samples, n_channels)`` ADC levels — as an ``.mcd``.

    The last block is short when ``n_samples`` is not a multiple of ``block_samples``, which
    is what a real recording looks like and what the reader's fast block map has to allow for.
    """
    path = Path(path)
    frames = np.ascontiguousarray(frames, dtype="<u2")
    n_samples, n_channels = frames.shape
    if labels is None:
        labels = [f"{i + 1:02d}" for i in range(n_channels)]
    assert len(labels) == n_channels
    block_duration_ns = round(block_samples / fs * 1e9)
    end_ts = start_ts + round(n_samples / fs * 1e9)

    header_chunks = [_mcshdr(source_path, fs, gain, block_duration_ns,
                             start_ts, end_ts, hardware, serial)]
    if with_digital:
        header_chunks += [_strmhdr("digital", "Digital Data", b"digi0001", ["D1"]),
                          _strmfmt(16, adc_zero, v_per_level, block_samples)]
    header_chunks += [_strmhdr("analog", "Electrode Raw Data", b"elec0001", labels),
                      _strmfmt(16, adc_zero, v_per_level, block_samples)]
    header = _chunk(b"LISThdr ", b"".join(header_chunks))

    # IDX1POS sits between the header and the data, and is 16 + 8 bytes long
    offset = CHUNK_HEADER + len(header) + CHUNK_HEADER + 8
    groups, table = [], []
    for start in range(0, n_samples, block_samples):
        stop = min(start + block_samples, n_samples)
        stamps = struct.pack("<QQ", start_ts + round(start / fs * 1e9),
                             start_ts + round(stop / fs * 1e9))
        group = b""
        if with_digital:
            silence = np.zeros((stop - start, 1), dtype="<u2").tobytes()
            group += _chunk(b"digi0001", stamps + silence)
        group += _chunk(b"elec0001", stamps + frames[start:stop].tobytes())
        table.append((start_ts + round(start / fs * 1e9), offset))
        groups.append(group)
        offset += len(group)

    index_pos = offset
    index = (_chunk(b"IDXIDX  ", np.array(table, dtype="<u8").tobytes())
             + _chunk(b"IDXSWEEP", struct.pack("<QQ", start_ts, end_ts)))
    path.write_bytes(b"MCSSTRM " + struct.pack("<Q", SIZE_UNTIL_EOF)
                     + header
                     + _chunk(b"IDX1POS ", struct.pack("<Q", index_pos))
                     + b"".join(groups)
                     + index)
    return path


def ramp_frames(n_samples: int = 50, n_channels: int = 8) -> np.ndarray:
    """The same pattern the ``.raw`` fixtures use: channel c, sample s holds 1000*c + s."""
    return (np.arange(n_channels)[None, :] * 1000
            + np.arange(n_samples)[:, None]).astype("<u2")
