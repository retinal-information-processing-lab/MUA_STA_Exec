"""Reader for the DMD stimulus ``.bin`` files.

Format: a header of 4 little-endian ``uint16`` (xsize, ysize, nb_images, nb_bits=8),
then ``nb_images`` raw ``uint8`` frames of ``xsize * ysize`` bytes each.

This replaces the lab's ``binfile.py`` (kept at the repo root for reference only). Besides
dropping the numpy APIs that no longer exist (``np.fromstring``, ``np.float``), it fixes two
things that are invisible for square frames and wrong for any other:

* **the shape of a stored frame depends on the rig.** The lab's convention, set by
  ``BinFile.append``, is that a *logical* frame is ``(ysize, xsize)`` — rows, columns, x
  horizontal. What lands in the file is that frame after ``undo_optical_transform``, and on a
  rig whose transform is ``rot90_flipud`` that is a transpose, so the bytes are
  ``(xsize, ysize)``. The old ``read_frame`` always reshaped to ``(xsize, ysize)`` and the
  newer ``read_frame_uint8`` always to ``(ysize, xsize)``; both are right for one rig only.
* **the frame count** is a 2-byte field and wraps silently past 65535, so it is derived from
  the file size and a disagreement warns.

Frames are memory-mapped, so reading 27 000 frames in ``.vec`` order (what a shifting-white-
noise stimulus needs) is random access into the page cache rather than 27 000 seek+read calls.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from .rigs import apply_optical_transform, rig, undo_optical_transform

HEADER_BYTES = 2 * 4


def read_header(path: os.PathLike | str) -> dict:
    """The four header fields, plus ``nb_images_from_size`` derived from the file length."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"{path}: too short to hold a bin header")
    xsize, ysize, nb_images, nb_bits = (int(v) for v in np.frombuffer(raw, dtype="<u2"))
    if xsize == 0 or ysize == 0:
        raise ValueError(f"{path}: bin header says {xsize}x{ysize} pixels")
    if nb_bits != 8:
        raise ValueError(f"{path}: only 8-bit bin files are supported (header says {nb_bits})")
    frame_bytes = xsize * ysize
    from_size = (path.stat().st_size - HEADER_BYTES) // frame_bytes
    return {
        "xsize": xsize,
        "ysize": ysize,
        "nb_images": nb_images,
        "nb_bits": nb_bits,
        "nb_images_from_size": int(from_size),
        "frame_bytes": frame_bytes,
    }


def stored_shape(xsize: int, ysize: int, transform: str | None) -> tuple[int, int]:
    """Shape of a frame *as stored in the file*, for a rig using ``transform``.

    A logical frame is ``(ysize, xsize)``; the file holds it after ``undo_optical_transform``,
    which transposes it when the transform is ``rot90_flipud``.
    """
    rows, cols = ysize, xsize
    probe = undo_optical_transform(np.empty((rows, cols), dtype=np.uint8), transform)
    return probe.shape


class BinFile:
    """Read-only, memory-mapped view of a stimulus ``.bin`` for one rig.

    ``frame(i)`` gives frame ``i`` as the retina saw it: float32 in [0, 1], shape
    ``(rows, cols)``, with the rig's polarity and optical path undone.
    """

    def __init__(self, path: os.PathLike | str, mea: int):
        self.path = Path(path)
        self.mea = mea
        self.rig = rig(mea)
        header = read_header(self.path)
        self.xsize, self.ysize = header["xsize"], header["ysize"]
        self.nb_frames = header["nb_images_from_size"]
        if header["nb_images"] not in (self.nb_frames, self.nb_frames % 65536):
            warnings.warn(
                f"{self.path.name}: header says {header['nb_images']} frames but the file holds "
                f"{self.nb_frames} (the header field is 2 bytes and wraps past 65535); "
                "using the count from the file size.",
                stacklevel=2,
            )
        self._check_frame_size()
        self._stored_shape = stored_shape(self.xsize, self.ysize, self.rig.optical_transform)
        self._frames = np.memmap(
            self.path, dtype=np.uint8, mode="r",
            offset=HEADER_BYTES, shape=(self.nb_frames, *self._stored_shape),
        )

    def _check_frame_size(self):
        limit = self.rig.max_frame_size
        if limit is None:
            return
        max_x, max_y = limit
        if self.xsize > max_x or self.ysize > max_y:
            raise ValueError(
                f"{self.path.name}: {self.xsize}x{self.ysize} px is too big for "
                f"{self.rig.label}, which displays at most {max_x}x{max_y} px"
            )

    def __len__(self) -> int:
        return self.nb_frames

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of a logical frame, ``(rows, cols)``."""
        return (self.ysize, self.xsize)

    def stored(self, index: int) -> np.ndarray:
        """Frame ``index`` exactly as stored, uint8, without any rig correction."""
        if not 0 <= index < self.nb_frames:
            raise IndexError(f"frame {index} outside 0..{self.nb_frames - 1}")
        return self._frames[index]

    def frame(self, index: int) -> np.ndarray:
        """Frame ``index`` as the retina saw it: float32 in [0, 1], shape ``(rows, cols)``."""
        stored = self.stored(index).astype(np.float32) / 255.0
        if self.rig.invert_polarity:
            stored = 1.0 - stored
        return apply_optical_transform(stored, self.rig.optical_transform)

    def close(self):
        frames, self._frames = getattr(self, "_frames", None), None
        if frames is not None:
            frames._mmap.close()

    def __enter__(self) -> "BinFile":
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def write_bin(path: os.PathLike | str, frames: np.ndarray, mea: int) -> Path:
    """Write ``frames`` as a stimulus ``.bin`` for ``mea``, the exact inverse of :meth:`BinFile.frame`.

    ``frames`` is ``(n, rows, cols)`` with values in [0, 1], in retina orientation. Used by the
    tests and to make an orientation test pattern; the tool itself only ever reads.
    """
    path = Path(path)
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 3:
        raise ValueError(f"expected (n, rows, cols), got shape {frames.shape}")
    n_frames, rows, cols = frames.shape
    settings = rig(mea)
    if settings.invert_polarity:
        frames = 1.0 - frames
    stored = undo_optical_transform(frames, settings.optical_transform)
    header = np.array([cols, rows, n_frames % 65536, 8], dtype="<u2")   # xsize = cols, ysize = rows
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.round(stored * 255).astype(np.uint8).tobytes())
    return path
