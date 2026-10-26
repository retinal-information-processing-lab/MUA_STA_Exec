"""Per-rig (MEA setup) hardware settings: the single source of truth.

Everything that depends on which setup a recording comes from lives in ``RIGS`` and
nowhere else, so fixing or adding a rig is done here only. The values are ported from
the standard pipeline's ``params.rig_params``, with two differences:

* the trigger threshold is stored in **ADC levels above the zero level** (the raw
  ``uint16`` scale the file is in), instead of the pipeline's "µV" which are derived from
  an amplifier gain of 1 and come out 1200x off. A ``.mcd`` header states the real scale
  (0.10417 µV per level, i.e. gain 1200), which is what ``UV_PER_LEVEL`` holds;
* ``optical_transform`` and ``invert_polarity`` are applied by one function, used by both
  the checkerboard and the shifting-white-noise path, so the two cannot disagree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ADC_N_BITS = 16
ADC_ZERO = 1 << (ADC_N_BITS - 1)        # 32768: the uint16 level that means 0 V
UV_PER_LEVEL = 0.10416666666666667      # µV per ADC level, as stated by the .mcd header


@dataclass(frozen=True)
class Rig:
    """Hardware settings of one MEA setup."""

    mea: int
    name: str
    trigger_threshold: int              # ADC levels above ADC_ZERO (positive-going pulse)
    size_dmd: tuple[int, int] | None    # dimensions of the DMD, in pixels (x, y)
    pxl_size_dmd: float | None          # size of one DMD pixel on the sample, in µm
    max_frame_size: tuple[int, int] | None   # largest stimulus image the rig can display
    invert_polarity: bool               # the rig displays inverted, so frames are stored inverted
    optical_transform: str | None       # "rot90_flipud", "fliplr" or None
    display_ready: bool                 # stimulus display settings implemented AND tested
    array_registration: str = "identity"    # how stimulus axes sit on the electrode grid
    registration_measured: bool = False     # ... and whether that was checked against data

    @property
    def label(self) -> str:
        return f"MEA{self.mea}" if self.name == "" else f"MEA{self.mea} ({self.name})"


# Only MEA 2 and 3 are implemented and tested for stimulus display. The others are kept so a
# recording can still be opened, but reconstructing their stimulus warns (see `rig`).
RIGS: dict[int, Rig] = {
    1: Rig(1, "", 28134, (760, 1020), 3.5, (1024, 768), False, "rot90_flipud", False),
    2: Rig(2, "", 15630, (864, 864), 3.5, (1920, 1080), False, "rot90_flipud", True),
    # MEA3's registration was measured from 20251013_PupilSize/01_Swn (see "Orientation" in the
    # README): electrode position predicts receptive-field position with r = +0.98 under it,
    # and r = 0 under the identity.
    3: Rig(3, "2p room", 17714, (760, 1020), 2.5, (1024, 768), True, "fliplr", True,
           "transpose_flip_xy", True),
}

DEFAULT_MEA = 3


def rig(mea: int) -> Rig:
    """Settings of one rig. Unknown rigs raise: their values have to be filled in here."""
    try:
        return RIGS[mea]
    except KeyError:
        raise ValueError(
            f"MEA {mea} is not defined in mua_sta.rigs.RIGS (known rigs: {sorted(RIGS)}). "
            "Add its settings there."
        ) from None


def to_sample_frame(frame: np.ndarray, mea: int) -> np.ndarray:
    """A stimulus frame as stored for the DMD -> as the retina saw it.

    Undoes this rig's display polarity, then its optical path, in that order — the same table
    and the same order as the lab's StimulusDisplayer, so what this tool shows and analyses is
    what was projected. ``frame`` holds values in [0, 1] and may be a stack of frames: only
    the last two axes (rows, columns) are touched.
    """
    settings = rig(mea)
    if settings.invert_polarity:
        frame = 1.0 - frame
    return apply_optical_transform(frame, settings.optical_transform)


def apply_optical_transform(frame: np.ndarray, transform: str | None) -> np.ndarray:
    """Correct a frame READ from a .bin for the optical path of the rig.

    Acts on the last two axes, so a stack of frames works. Both transforms are involutions
    (``rot90_flipud`` is the transpose of the two axes, ``fliplr`` reverses the columns), which
    is why writing uses the same operations — see :func:`undo_optical_transform`.
    """
    if transform is None:
        return frame
    if transform == "rot90_flipud":            # == np.flipud(np.rot90(frame)) == frame.T
        return frame.swapaxes(-2, -1)
    if transform == "fliplr":
        return frame[..., ::-1]
    raise ValueError(f"unknown optical_transform {transform!r}")


def undo_optical_transform(frame: np.ndarray, transform: str | None) -> np.ndarray:
    """Pre-compensate a frame about to be WRITTEN to a .bin.

    The same operations as :func:`apply_optical_transform` — both transforms are their own
    inverse — but kept as a separate name so a call site says which direction it means.
    """
    return apply_optical_transform(frame, transform)


# --------------------------------------------------------------- array registration
# How the stimulus's axes sit on the electrode grid. This is a *different* question from
# `optical_transform`, which compensates the DMD-to-sample optics so a frame looks on the
# sample as it was drawn — that one is verified against the lab's StimulusDisplayer. This one
# says how the MEA itself is oriented in that frame, which no stimulus file can tell you: it is
# measured from the data, by checking that an electrode's position predicts where its receptive
# field is (see `sta.measure_registration`). The previous version of this tool assumed the
# identity, which on MEA3 is a transpose and two flips away from the truth.
ARRAY_REGISTRATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda a: a,
    "flip_y": lambda a: a[..., ::-1, :],
    "flip_x": lambda a: a[..., :, ::-1],
    "flip_xy": lambda a: a[..., ::-1, ::-1],
    "transpose": lambda a: a.swapaxes(-2, -1),
    "transpose_flip_y": lambda a: a.swapaxes(-2, -1)[..., ::-1, :],
    "transpose_flip_x": lambda a: a.swapaxes(-2, -1)[..., :, ::-1],
    "transpose_flip_xy": lambda a: a.swapaxes(-2, -1)[..., ::-1, ::-1],
}


def to_array_frame(image: np.ndarray, mea: int, registration: str | None = None) -> np.ndarray:
    """A stimulus-space map -> the electrode grid's axes, for drawing it on the array.

    Acts on the last two axes, so a stack of maps works. ``registration`` overrides the rig's.
    """
    name = registration or rig(mea).array_registration
    try:
        return np.ascontiguousarray(ARRAY_REGISTRATIONS[name](image))
    except KeyError:
        raise ValueError(f"unknown array_registration {name!r} "
                         f"(known: {sorted(ARRAY_REGISTRATIONS)})") from None
