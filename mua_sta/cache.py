"""Staged, key-addressed cache under ``~/.mua_sta/cache``.

Each expensive step writes one cache entry carrying the **key** of the inputs and parameters
it was computed from, so the GUI can tell, per step, whether it is ``CACHED`` (key matches),
``STALE`` (an entry exists but something it depends on changed) or ``MISSING``. Running the
pipeline then only redoes the steps that are not ``CACHED``. That is what stops a mistyped
check count from costing another pass over a 19 GB recording.

Two kinds of entry, because not every step depends on the recording:

* **recording-scoped** (``trace``, ``triggers``, ``spikes``, ``analysis``) — one file per
  recording and step, at ``cache/rec/<rec_id>/<step>.npz``. A new key overwrites the old
  entry, so these never pile up.
* **content-addressed** (``stimulus``, ``cov``) — the key is in the file name, at
  ``cache/stim/<step>-<key>.npz``. A reconstructed stimulus does not depend on the recording,
  so switching between checkerboard and shifting white noise, or between two stimulus files,
  keeps both instead of rebuilding one of them each time.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

CACHE_ROOT = Path.home() / ".mua_sta" / "cache"

TRACE, TRIGGERS, SPIKES, ANALYSIS = "trace", "triggers", "spikes", "analysis"
STIMULUS, COV = "stimulus", "cov"
RECORDING_STEPS = (TRACE, TRIGGERS, SPIKES, ANALYSIS)
CONTENT_STEPS = (STIMULUS, COV)
STEPS = RECORDING_STEPS[:3] + CONTENT_STEPS + (ANALYSIS,)   # the order they run in


class Status(Enum):
    MISSING = "missing"     # never computed
    STALE = "stale"         # computed, but from different inputs or parameters
    CACHED = "cached"       # usable as is

    def __str__(self) -> str:
        return self.value


def file_identity(path: os.PathLike | str) -> dict:
    """What identifies a file for caching: its path, size and modification time."""
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def key(**deps) -> str:
    """A short hash of everything a step depends on. Order of the keyword arguments is free."""
    canonical = json.dumps(deps, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(canonical.encode()).hexdigest()[:16]


def recording_id(path: os.PathLike | str) -> str:
    """Folder name for one recording: its stem plus a hash of the full path, so two
    recordings with the same name in different folders do not collide."""
    path = Path(path)
    return f"{path.stem}-{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]}"


@dataclass
class Cache:
    """The cache for one recording. ``rec_id`` is None until a recording is chosen."""

    rec_id: str | None = None
    root: Path = CACHE_ROOT

    @classmethod
    def for_recording(cls, path: os.PathLike | str | None, root: Path = CACHE_ROOT) -> "Cache":
        return cls(rec_id=recording_id(path) if path else None, root=root)

    # ------------------------------------------------------------------ paths
    def path(self, step: str, step_key: str) -> Path:
        if step in CONTENT_STEPS:
            return self.root / "stim" / f"{step}-{step_key}.npz"
        if self.rec_id is None:
            raise ValueError(f"step {step!r} is recording-scoped but no recording is set")
        return self.root / "rec" / self.rec_id / f"{step}.npz"

    # ----------------------------------------------------------------- status
    def status(self, step: str, step_key: str) -> Status:
        if not self.path(step, step_key).exists():
            # A content-addressed entry has the key in its name, so a changed key lands on a
            # path of its own: those steps report MISSING where a recording-scoped one would
            # report STALE. Either way the step has to run, and the GUI wording follows.
            return Status.MISSING
        return Status.CACHED if self.stored_key(step, step_key) == step_key else Status.STALE

    def stored_key(self, step: str, step_key: str) -> str | None:
        """The key the existing entry was computed from, or None if there is no entry."""
        path = self.path(step, step_key)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                return str(data["__key__"].item())
        except (OSError, ValueError, KeyError):
            return None

    def size(self, step: str, step_key: str) -> int:
        path = self.path(step, step_key)
        return path.stat().st_size if path.exists() else 0

    # --------------------------------------------------------------- contents
    def load(self, step: str, step_key: str) -> dict | None:
        """The cached arrays and metadata, or None when the entry is missing or stale."""
        if self.status(step, step_key) is not Status.CACHED:
            return None
        with np.load(self.path(step, step_key), allow_pickle=False) as data:
            out = {k: data[k] for k in data.files if not k.startswith("__")}
            out["__meta__"] = json.loads(str(data["__meta__"].item())) if "__meta__" in data else {}
        return out

    def store(self, step: str, step_key: str, meta: dict | None = None, **arrays) -> Path:
        """Write one entry. ``arrays`` are saved as they are; ``meta`` must be JSON-able."""
        path = self.path(step, step_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(arrays)
        payload["__key__"] = np.array(step_key)
        payload["__meta__"] = np.array(json.dumps(meta or {}, default=str))
        tmp = path.with_suffix(".npz.part")
        with open(tmp, "wb") as f:
            np.savez(f, **payload)
        os.replace(tmp, path)
        return path

    # ---------------------------------------------------------------- removal
    def drop(self, step: str, step_key: str):
        """Remove one entry (used by the bin button next to each step in the GUI)."""
        self.path(step, step_key).unlink(missing_ok=True)

    def clear_recording(self):
        """Remove every recording-scoped entry of this recording."""
        if self.rec_id is not None:
            shutil.rmtree(self.root / "rec" / self.rec_id, ignore_errors=True)

    def clear_all(self):
        """Remove the whole cache, recordings and stimuli alike."""
        shutil.rmtree(self.root, ignore_errors=True)

    def total_size(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*.npz") if p.is_file())
