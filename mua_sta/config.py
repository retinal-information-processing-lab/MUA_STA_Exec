"""Everything the GUI remembers between sessions, in ``~/.mua_sta/config.json``."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from . import recording as rec
from . import sta, stimulus
from .rigs import DEFAULT_MEA, rig

CONFIG_DIR = Path.home() / ".mua_sta"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass
class Config:
    """Chosen files, the setup, and every parameter of the four analysis steps."""

    # --- what is being analysed
    mea: int = DEFAULT_MEA
    kind: str = stimulus.SWN
    recording_path: str = ""
    last_dir: str = ""

    # --- the recording. Only the trigger channel is ever shown: this rig always records 256
    # uint16 channels at 20 kHz through the same amplifier, and the trigger threshold is a
    # property of the setup (rigs.py), not something anyone tunes at the bench. They stay
    # fields rather than becoming constants so they are still part of the cache keys.
    n_channels: int = rec.DEFAULT_N_CHANNELS
    dtype: str = rec.DEFAULT_DTYPE
    fs: float = rec.DEFAULT_FS
    trigger_channel: int = rec.DEFAULT_TRIGGER_CHANNEL
    trigger_threshold: int | None = None        # None: this rig's default
    maximal_jitter: float = 0.25e-3

    # --- MUA detection
    detect_threshold: float = 6.0                # in median absolute deviations
    peak_sign: str = "neg"
    freq_min: float = 300.0
    freq_max: float = 6000.0
    common_reference: bool = True
    n_jobs: int = 10

    # --- the stimulus
    checkerboard_source: str = ""
    n_checks: int = 42                           # squares per side; the board is square
    pixels_per_check: int = 16                   # only scales the µm a result is reported in
    swn_bin: str = ""
    swn_vec: str = ""
    swn_lattice_auto: bool = True                # measure the two below from the frames
    checks_per_side: int = 10
    shifts_per_check: int = 8

    # --- the analysis
    frames_per_sequence: int = sta.DEFAULT_FRAMES_PER_SEQUENCE
    temporal_dimension: int = sta.DEFAULT_TEMPORAL_DIMENSION
    whiten: bool = True
    sigma_fraction: float = sta.DEFAULT_SIGMA_FRACTION

    # --- display
    flip_grid_y: bool = False                    # row 0 at the top instead of the bottom
    array_registration: str = ""                 # "" : this rig's measured default
    crop_tiles: bool = True                      # narrow each tile to where the fields are

    path: str = field(default="", compare=False)        # where this was loaded from
    load_error: str = field(default="", compare=False)  # why a saved config could not be read

    # ------------------------------------------------------------------ derived
    @property
    def threshold(self) -> int:
        """The trigger threshold in use: whatever was typed, else this rig's default."""
        return rig(self.mea).trigger_threshold if self.trigger_threshold is None \
            else int(self.trigger_threshold)

    @property
    def registration(self) -> str:
        """How stimulus axes sit on the electrode grid: whatever was chosen, else the rig's."""
        return self.array_registration or rig(self.mea).array_registration

    def stimulus_spec(self):
        """A :class:`CheckerboardSpec` or :class:`SwnSpec` for the chosen stimulus kind."""
        if self.kind == stimulus.CHECKERBOARD:
            if not self.checkerboard_source:
                raise ValueError("no binary source file chosen for the checkerboard")
            return stimulus.CheckerboardSpec(
                Path(self.checkerboard_source), self.n_checks, self.n_checks, self.mea,
                self.pixels_per_check)
        if self.kind == stimulus.SWN:
            if not (self.swn_bin and self.swn_vec):
                raise ValueError("shifting white noise needs both a .bin and a .vec file")
            return stimulus.SwnSpec(
                Path(self.swn_bin), Path(self.swn_vec),
                self.checks_per_side, self.shifts_per_check, self.mea)
        raise ValueError(f"unknown stimulus kind {self.kind!r}")

    # --------------------------------------------------------------- read/write
    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("path", "load_error"):
            data.pop(key)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        known = {f.name for f in fields(cls)} - {"path", "load_error"}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: os.PathLike | str | None = None) -> Path:
        path = Path(path or self.path or DEFAULT_CONFIG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        self.path = str(path)
        return path

    @classmethod
    def load(cls, path: os.PathLike | str | None = None) -> "Config":
        """Load a config. An unreadable one gives the defaults, kept aside as ``<name>.bad``."""
        path = Path(path or DEFAULT_CONFIG_PATH)
        if not path.exists():
            return cls(path=str(path))
        try:
            with open(path) as f:
                config = cls.from_dict(json.load(f))
        except (OSError, ValueError, TypeError) as e:
            bad = path.with_name(path.name + ".bad")
            try:
                os.replace(path, bad)
            except OSError:
                pass
            config = cls(load_error=f"settings file {path} unreadable ({e}): using the defaults "
                                    f"(the file is kept as {bad.name})")
        config.path = str(path)
        return config
