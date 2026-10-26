"""The stage graph: which steps have to run, in what order, and what is already cached.

Each step declares what it depends on as a cache key, so the GUI can show, before anything
runs, whether a step is usable as it stands or has to be redone. Running a step first makes
sure its own dependencies are there, which is what makes "change one number and press RUN"
cost only the work that number actually affects.

The split between the steps is chosen around the two slow ones. Reading the trigger channel is
a pass over the whole recording, so it is its own step and does **not** depend on the
threshold: setting the threshold again only re-runs the edge detection, which is instant.
Detecting MUA spikes is the long step, and its key holds only the detection parameters, so
nothing about the stimulus or the analysis can force it to run again.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import cache as ca
from . import recording as rec
from . import spikes as sp
from . import sta as sta_mod
from . import stimulus as st
from . import triggers as tg
from .config import Config

STEP_LABELS = {
    ca.TRACE: "Trigger channel",
    ca.TRIGGERS: "Triggers",
    ca.SPIKES: "MUA spikes",
    ca.STIMULUS: "Stimulus",
    ca.COV: "Stimulus covariance",
    ca.ANALYSIS: "STA and rasters",
}

#: The three lines the window shows, each standing for the steps underneath it. Collapsing
#: them is only presentation: keeping the steps separate is what lets a changed setup re-detect
#: the triggers without reading the recording again.
GROUPS = {
    "recording": (ca.TRACE, ca.TRIGGERS, ca.SPIKES),
    "stimulus": (ca.STIMULUS, ca.COV),
    "sta": (ca.ANALYSIS,),
}
GROUP_LABELS = {
    "recording": "Triggers and spikes",
    "stimulus": "Stimulus",
    "sta": "STA",
}

#: What each step needs before it can run.
DEPENDENCIES = {
    ca.TRACE: (),
    ca.TRIGGERS: (ca.TRACE,),
    ca.SPIKES: (),
    ca.STIMULUS: (ca.TRIGGERS,),
    ca.COV: (ca.STIMULUS,),
    ca.ANALYSIS: (ca.TRIGGERS, ca.SPIKES, ca.STIMULUS),
}

Progress = Callable[[str], None]
Log = Callable[..., None]


@dataclass
class Results:
    """What the analysis step produces, per electrode and in electrode-label order."""

    labels: np.ndarray              # MCS labels, 1-based
    sta_3d: np.ndarray              # (n, T, rows, cols)
    spatial: np.ndarray             # (n, rows, cols) — whitened when asked for
    spatial_raw: np.ndarray         # (n, rows, cols) — never whitened
    temporal: np.ndarray            # (n, T)
    peak: np.ndarray                # (n, 3): lag, row, col
    raster_times: np.ndarray        # flat, see sta.raster_trains
    raster_offsets: np.ndarray
    meta: dict

    @property
    def n_sequences(self) -> int:
        return int(self.meta["n_sequences"])

    def raster(self, label: int) -> list[np.ndarray]:
        """The repeated-half spike trains of one electrode, one array per sequence."""
        position = int(np.searchsorted(self.labels, label))
        if position >= self.labels.size or self.labels[position] != label:
            return []
        base = position * self.n_sequences
        return [self.raster_times[self.raster_offsets[base + s]:self.raster_offsets[base + s + 1]]
                for s in range(self.n_sequences)]

    def index_of(self, label: int) -> int | None:
        position = int(np.searchsorted(self.labels, label))
        if position < self.labels.size and self.labels[position] == label:
            return position
        return None


class Pipeline:
    """Holds the loaded artefacts of one recording and runs the steps that are not cached."""

    def __init__(self, config: Config, log: Log | None = None):
        self.config = config
        self.log = log or (lambda *a, **k: None)
        self.cache = ca.Cache.for_recording(config.recording_path or None)
        self.info: rec.RecordingInfo | None = None
        self.trace: np.ndarray | None = None
        self.triggers: tg.Triggers | None = None
        self.stimulus: np.ndarray | None = None
        self.stimulus_meta: dict = {}
        self.covariance: np.ndarray | None = None
        self.spikes: sp.SpikeTrains | None = None
        self.results: Results | None = None
        self._loaded: dict[str, str] = {}       # step -> the key currently in memory

    # ------------------------------------------------------------------ recording
    def reset(self):
        """Forget everything loaded: the recording or the setup changed."""
        self.cache = ca.Cache.for_recording(self.config.recording_path or None)
        self.info = self.trace = self.triggers = None
        self.stimulus = self.covariance = self.spikes = self.results = None
        self.stimulus_meta = {}
        self._loaded.clear()

    def ensure_info(self) -> rec.RecordingInfo:
        if self.info is None:
            if not self.config.recording_path:
                raise ValueError("no recording chosen")
            self.info = rec.open_info(self.config.recording_path, self.config.n_channels,
                                      self.config.dtype, self.config.fs)
        return self.info

    # ----------------------------------------------------------------------- keys
    def step_key(self, step: str) -> str | None:
        """The key of one step, or None when it cannot be known yet.

        The stimulus needs to know how many sequences were shown, which comes from the
        triggers — so until those exist its key, and its status, are simply unknown.
        """
        config = self.config
        if not config.recording_path and step != ca.STIMULUS:
            return None
        try:
            if step == ca.TRACE:
                return ca.key(file=ca.file_identity(config.recording_path),
                              n_channels=config.n_channels, dtype=config.dtype,
                              trigger_channel=config.trigger_channel)
            if step == ca.TRIGGERS:
                return ca.key(trace=self.step_key(ca.TRACE), threshold=config.threshold,
                              jitter=config.maximal_jitter, fs=config.fs)
            if step == ca.SPIKES:
                return ca.key(file=ca.file_identity(config.recording_path),
                              n_channels=config.n_channels, dtype=config.dtype, fs=config.fs,
                              detect_threshold=config.detect_threshold,
                              peak_sign=config.peak_sign, freq_min=config.freq_min,
                              freq_max=config.freq_max,
                              common_reference=config.common_reference)
            if step == ca.STIMULUS:
                n_frames = self.stimulus_frame_count()
                if n_frames is None:
                    return None
                return ca.key(**config.stimulus_spec().cache_deps(n_frames))
            if step == ca.COV:
                stimulus_key = self.step_key(ca.STIMULUS)
                if stimulus_key is None:
                    return None
                return ca.key(stimulus=stimulus_key)
            if step == ca.ANALYSIS:
                parts = {name: self.step_key(name)
                         for name in (ca.TRIGGERS, ca.SPIKES, ca.STIMULUS)}
                if any(value is None for value in parts.values()):
                    return None
                if config.whiten:
                    parts["cov"] = self.step_key(ca.COV)
                    parts["sigma_fraction"] = config.sigma_fraction
                return ca.key(frames_per_sequence=config.frames_per_sequence,
                              temporal_dimension=config.temporal_dimension,
                              whiten=config.whiten, **parts)
        except (OSError, ValueError):
            return None
        raise ValueError(f"unknown step {step!r}")

    def steps(self) -> tuple[str, ...]:
        """The steps that apply, in running order (the covariance only when whitening)."""
        order = [ca.TRACE, ca.TRIGGERS, ca.SPIKES, ca.STIMULUS]
        if self.config.whiten and self.config.kind == st.SWN:
            order.append(ca.COV)
        order.append(ca.ANALYSIS)
        return tuple(order)

    def status(self, step: str) -> ca.Status | None:
        """``None`` when the step's key is not knowable yet."""
        step_key = self.step_key(step)
        if step_key is None:
            return None
        return self.cache.status(step, step_key)

    def group_steps(self, group: str) -> tuple[str, ...]:
        """The steps of one line, leaving out any that do not apply (the covariance when the
        stimulus is a checkerboard, or when whitening is off)."""
        applicable = set(self.steps())
        return tuple(step for step in GROUPS[group] if step in applicable)

    def group_status(self, group: str) -> ca.Status:
        """One dot for the whole line: red if anything has to be redone, green if all of it is
        usable, grey otherwise (never done, half done, or not knowable yet)."""
        statuses = [self.status(step) for step in self.group_steps(group)]
        if any(status is ca.Status.STALE for status in statuses):
            return ca.Status.STALE
        if statuses and all(status is ca.Status.CACHED for status in statuses):
            return ca.Status.CACHED
        return ca.Status.MISSING

    def run_group(self, group: str, progress: Progress | None = None, force: bool = False):
        """Run one line. Pressing play on a line that is already green means "do it again", so
        a green group is forced; a red or grey one only does what is missing."""
        steps = self.group_steps(group)
        force = force or self.group_status(group) is ca.Status.CACHED
        for step in steps:
            self.run(step, progress, force=force)

    def drop_group(self, group: str):
        """Forget every cached result of one line."""
        for step in GROUPS[group]:
            step_key = self.step_key(step)
            if step_key is not None:
                self.cache.drop(step, step_key)
            self._loaded.pop(step, None)

    def anything_done(self) -> bool:
        """True when any step has already been computed, whatever its settings were.

        Asked per step rather than per line: a line reads grey while only part of it is done,
        but that part is still work the Run button would be throwing away.
        """
        return any(self.status(step) in (ca.Status.CACHED, ca.Status.STALE)
                   for step in self.steps())

    def stimulus_frame_count(self) -> int | None:
        """How many novel frames the stimulus has to supply, or None if not known yet."""
        n_sequences = self.n_sequences()
        if n_sequences is None:
            return None
        return n_sequences * (self.config.frames_per_sequence // 2)

    def n_sequences(self) -> int | None:
        """Sequences shown, from the triggers in memory or in the cache."""
        if self.triggers is not None:
            return self.triggers.n_sequences(self.config.frames_per_sequence)
        step_key = self.step_key(ca.TRIGGERS)
        if step_key is None:
            return None
        cached = self.cache.load(ca.TRIGGERS, step_key)
        if cached is None:
            return None
        return int(cached["onsets"].size // self.config.frames_per_sequence)

    # ----------------------------------------------------------------- the steps
    def run(self, step: str, progress: Progress | None = None, force: bool = False):
        """Make ``step`` available, running whatever it needs that is not cached."""
        for dependency in DEPENDENCIES[step]:
            self.run(dependency, progress)
        step_key = self.step_key(step)
        if step_key is None:
            raise ValueError(f"{STEP_LABELS[step]}: not enough is known yet to run it")
        if force:
            self.cache.drop(step, step_key)
        if self._loaded.get(step) == step_key and not force:
            return
        cached = None if force else self.cache.load(step, step_key)
        if cached is not None:
            self._adopt(step, cached)
            self.log(f"{STEP_LABELS[step]}: loaded from the cache", "info")
        else:
            self._compute(step, step_key, progress)
        self._loaded[step] = step_key

    def run_all(self, progress: Progress | None = None, force: bool = False):
        """Run everything. ``force`` recomputes each step even when it is already cached."""
        for step in self.steps():
            self.run(step, progress, force=force)

    def _adopt(self, step: str, cached: dict):
        """Take a cached entry as this step's result."""
        if step == ca.TRACE:
            self.trace = cached["levels"]
        elif step == ca.TRIGGERS:
            self.triggers = tg.Triggers(cached["onsets"], self.config.fs,
                                        int(cached["modal_interval"]), cached["error_onsets"])
        elif step == ca.SPIKES:
            self.spikes = sp.SpikeTrains(cached["labels"], cached["offsets"], cached["times"])
        elif step == ca.STIMULUS:
            self.stimulus = cached["frames"]
            self.stimulus_meta = cached["__meta__"]
        elif step == ca.COV:
            self.covariance = cached["covariance"]
        elif step == ca.ANALYSIS:
            self.results = Results(
                cached["labels"], cached["sta_3d"], cached["spatial"], cached["spatial_raw"],
                cached["temporal"], cached["peak"], cached["raster_times"],
                cached["raster_offsets"], cached["__meta__"])

    def _compute(self, step: str, step_key: str, progress: Progress | None):
        getattr(self, f"_compute_{step}")(step_key, progress)

    def _compute_trace(self, step_key: str, progress: Progress | None):
        info = self.ensure_info()
        self.log(f"Reading channel {self.config.trigger_channel} of {info.path.name} "
                 f"({info.describe()})", "info")
        fraction = (lambda done: progress(f"trigger channel: {done:.0%}")) if progress else None
        self.trace = rec.read_channel(info, self.config.trigger_channel, progress=fraction)
        self.cache.store(ca.TRACE, step_key, meta={"n_samples": int(self.trace.size)},
                         levels=self.trace)

    def _compute_triggers(self, step_key: str, progress: Progress | None):
        threshold = self.config.threshold
        self.log(f"Detecting trigger onsets above {threshold} ADC levels "
                 f"({tg.threshold_to_uv(threshold):.0f} µV)", "info")
        onsets = tg.detect_onsets(self.trace, threshold)
        self.triggers = tg.sanity_check(onsets, self.config.fs, self.config.maximal_jitter)
        level = "warn" if self.triggers.error_onsets.size else "info"
        self.log(self.triggers.describe(self.config.frames_per_sequence), level)
        self.cache.store(ca.TRIGGERS, step_key,
                         meta={"frequency": self.triggers.frequency},
                         onsets=self.triggers.onsets,
                         modal_interval=np.array(self.triggers.modal_interval),
                         error_onsets=self.triggers.error_onsets)

    def _compute_spikes(self, step_key: str, progress: Progress | None):
        info = self.ensure_info()
        self.log(f"Detecting MUA peaks at {self.config.detect_threshold} MAD "
                 f"({self.config.peak_sign}) on {info.n_channels} channels — the long step",
                 "info")
        self.spikes = sp.detect_mua(
            info, detect_threshold=self.config.detect_threshold,
            peak_sign=self.config.peak_sign, freq_min=self.config.freq_min,
            freq_max=self.config.freq_max, common_reference=self.config.common_reference,
            n_jobs=self.config.n_jobs, progress=progress)
        self.log(self.spikes.describe(), "info")
        self.cache.store(ca.SPIKES, step_key, meta={"total": self.spikes.total},
                         labels=self.spikes.labels, offsets=self.spikes.offsets,
                         times=self.spikes.times)

    def _compute_stimulus(self, step_key: str, progress: Progress | None):
        spec = self.config.stimulus_spec()
        if hasattr(spec, "resolved"):
            # reading the frames to measure the lattice happens here and nowhere earlier
            spec = spec.resolved()
        n_frames = self.stimulus_frame_count()
        self.log(f"Rebuilding {st.KIND_LABELS[spec.kind]}: {n_frames} novel frames on a "
                 f"{spec.grid_shape[0]}x{spec.grid_shape[1]} grid"
                 + (f", {spec.cell_size_um:.1f} µm per cell" if spec.cell_size_um else ""),
                 "info")
        self.stimulus = spec.build(n_frames, progress=progress)
        self.stimulus_meta = {"kind": spec.kind, "cell_size_um": spec.cell_size_um,
                              "grid_shape": list(spec.grid_shape)}
        self.cache.store(ca.STIMULUS, step_key, meta=self.stimulus_meta, frames=self.stimulus)

    def _compute_cov(self, step_key: str, progress: Progress | None):
        n_pixels = self.stimulus.shape[1] * self.stimulus.shape[2]
        self.log(f"Stimulus covariance over {n_pixels} cells", "info")
        self.covariance = sta_mod.stimulus_covariance(self.stimulus, progress=progress)
        self.cache.store(ca.COV, step_key, meta={"n_pixels": n_pixels},
                         covariance=self.covariance)

    def _warn_if_whitening_hurt(self, raw: np.ndarray, whitened: np.ndarray,
                                ratio: float = 0.6):
        """Say so when whitening has buried the receptive fields instead of sharpening them.

        A shifting stimulus's covariance carries very little variance at high spatial
        frequencies, so its inverse amplifies them — and a multi-unit STA, pooled over whatever
        the electrode picked up, is noisy there. Whether that helps or hurts depends on the
        recording, so it is measured rather than assumed, and the comparison is cheap: the raw
        STA is computed either way.
        """
        before = float(np.median(sta_mod.rf_quality(raw)))
        after = float(np.median(sta_mod.rf_quality(whitened)))
        if after < ratio * before:
            self.log(
                f"Whitening made the receptive fields weaker, not sharper (quality {before:.0f} "
                f"-> {after:.0f}). On multi-unit data it usually does: untick it, or raise the "
                "regularisation. The unwhitened STA is kept alongside.", "warn")
        else:
            self.log(f"Whitening: receptive-field quality {before:.0f} -> {after:.0f}", "info")

    def _compute_analysis(self, step_key: str, progress: Progress | None):
        config = self.config
        mapping = rec.load_mapping()
        labels = np.array(rec.electrode_labels(mapping), dtype=np.int32)
        n_sequences = self.triggers.n_sequences(config.frames_per_sequence)
        novel = sta_mod.sequence_edges(self.triggers.times, n_sequences,
                                       config.frames_per_sequence, sta_mod.NOVEL_HALF)
        repeated = sta_mod.sequence_edges(self.triggers.times, n_sequences,
                                          config.frames_per_sequence, sta_mod.REPEATED_HALF)
        self.log(f"Binning {labels.size} electrodes over {n_sequences} sequences", "info")
        counts = sta_mod.bin_all(self.spikes, labels, novel)
        sta_3d = sta_mod.compute_sta(counts, self.stimulus, config.temporal_dimension,
                                     progress=progress)
        spatial_raw, temporal, peak = sta_mod.peak_components(sta_3d)
        spatial = spatial_raw
        if config.whiten and config.kind == st.SWN:
            self.run(ca.COV, progress)
            self.log(f"Whitening the spatial STAs (one factorisation for all electrodes, "
                     f"sigma = {config.sigma_fraction:.1%} of the pixel variance)", "info")
            spatial = sta_mod.whiten(spatial_raw, self.covariance, config.sigma_fraction)
            self._warn_if_whitening_hurt(spatial_raw, spatial)
        raster_times, raster_offsets = sta_mod.raster_trains(self.spikes, labels, repeated)
        self.results = Results(
            labels, sta_3d, spatial, spatial_raw, temporal, peak, raster_times, raster_offsets,
            {"n_sequences": n_sequences, "frequency": self.triggers.frequency,
             "whiten": bool(config.whiten and config.kind == st.SWN),
             "kind": config.kind,
             "cell_size_um": self.stimulus_meta.get("cell_size_um"),
             "temporal_dimension": config.temporal_dimension, "mea": config.mea})
        self.cache.store(ca.ANALYSIS, step_key, meta=self.results.meta, labels=labels,
                         sta_3d=sta_3d, spatial=spatial, spatial_raw=spatial_raw,
                         temporal=temporal, peak=peak, raster_times=raster_times,
                         raster_offsets=raster_offsets)
        self.log(f"STA done: {labels.size} electrodes, {sta_3d.shape[1]} lags, "
                 f"{sta_3d.shape[2]}x{sta_3d.shape[3]} cells", "info")
