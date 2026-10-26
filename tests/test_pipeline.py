"""The stage graph end to end, and that changing a parameter only invalidates what it touches.

The MUA detection step is the one thing not exercised here: it is a pass over the whole
recording through spikeinterface. Instead a known spike train is written into the cache under
the key the pipeline computes for it — which also checks that the keys are what the pipeline
thinks they are. ``test_spikes.py`` covers the detection itself.
"""
import numpy as np
import pytest

from mua_sta import cache as ca
from mua_sta import recording as rec
from mua_sta import spikes as sp
from mua_sta import stimulus as st
from mua_sta.config import Config
from mua_sta.pipeline import Pipeline
from mua_sta.rigs import ADC_ZERO
from tests.test_stimulus import novel_then_repeated, write_swn_bin, write_vec

FRAMES_PER_SEQUENCE = 20
N_SEQUENCES = 3
FRAME_SAMPLES = 100                  # 200 Hz at 20 kHz
FS = 20_000.0


@pytest.fixture
def setup(tmp_path):
    """A tiny but complete experiment: a recording with triggers, and an SWN stimulus."""
    n_triggers = FRAMES_PER_SEQUENCE * N_SEQUENCES
    n_samples = n_triggers * FRAME_SAMPLES + FRAME_SAMPLES
    frames = np.full((n_samples, rec.DEFAULT_N_CHANNELS), ADC_ZERO, dtype="<u2")
    for i in range(n_triggers):
        start = i * FRAME_SAMPLES + 5
        frames[start:start + FRAME_SAMPLES // 2, rec.DEFAULT_TRIGGER_CHANNEL] = ADC_ZERO + 19000
    raw = tmp_path / "rec.raw"
    raw.write_bytes(frames.tobytes())

    n_novel = N_SEQUENCES * (FRAMES_PER_SEQUENCE // 2)
    bin_path = write_swn_bin(tmp_path / "swn.bin", 3, n_checks=5, shifts=2,
                             pixels_per_check=10, n_frames=2 * n_novel + 4)
    vec_path = write_vec(tmp_path / "swn.vec", novel_then_repeated(n_novel))

    config = Config(
        mea=3, kind=st.SWN, recording_path=str(raw), fs=FS,
        swn_bin=str(bin_path), swn_vec=str(vec_path), checks_per_side=5, shifts_per_check=2,
        frames_per_sequence=FRAMES_PER_SEQUENCE, temporal_dimension=3,
        whiten=False,
    )
    return config, tmp_path


def seed_spikes(pipeline, rng=None):
    """Write a known MUA train into the cache, under the key the pipeline asks for."""
    rng = rng or np.random.default_rng(0)
    labels = np.array(rec.electrode_labels(rec.load_mapping())[:6], dtype=np.int32)
    per_electrode = 400
    duration = FRAMES_PER_SEQUENCE * N_SEQUENCES * FRAME_SAMPLES / FS
    times = np.sort(rng.random((labels.size, per_electrode)).astype(np.float32) * duration, axis=1)
    offsets = (np.arange(labels.size + 1) * per_electrode).astype(np.int64)
    trains = sp.SpikeTrains(labels, offsets, times.ravel())
    pipeline.cache.store(ca.SPIKES, pipeline.step_key(ca.SPIKES), meta={"total": trains.total},
                         labels=trains.labels, offsets=trains.offsets, times=trains.times)
    return trains


def build(config, root):
    pipeline = Pipeline(config)
    pipeline.cache = ca.Cache.for_recording(config.recording_path, root=root / "cache")
    return pipeline


def test_runs_end_to_end(setup):
    config, tmp_path = setup
    pipeline = build(config, tmp_path)
    seed_spikes(pipeline)
    pipeline.run_all()

    assert pipeline.triggers.count == FRAMES_PER_SEQUENCE * N_SEQUENCES
    assert pipeline.triggers.n_sequences(FRAMES_PER_SEQUENCE) == N_SEQUENCES
    assert pipeline.triggers.frequency == pytest.approx(FS / FRAME_SAMPLES)
    assert pipeline.stimulus.shape == (N_SEQUENCES * FRAMES_PER_SEQUENCE // 2, 10, 10)

    results = pipeline.results
    n_electrodes = rec.GRID_SIZE ** 2 - len(rec.EXCLUDED_LABELS)
    assert results.labels.size == n_electrodes
    assert results.sta_3d.shape == (n_electrodes, 3, 10, 10)
    assert results.spatial.shape == (n_electrodes, 10, 10)
    assert results.temporal.shape == (n_electrodes, 3)
    assert results.meta["whiten"] is False
    assert len(results.raster(int(results.labels[0]))) == N_SEQUENCES


def test_everything_is_cached_after_a_run(setup):
    config, tmp_path = setup
    pipeline = build(config, tmp_path)
    seed_spikes(pipeline)
    pipeline.run_all()
    for step in pipeline.steps():
        assert pipeline.status(step) is ca.Status.CACHED, step


def test_a_new_threshold_does_not_re_read_the_trigger_channel(setup):
    """The reason the steps are split where they are."""
    config, tmp_path = setup
    pipeline = build(config, tmp_path)
    seed_spikes(pipeline)
    pipeline.run_all()
    trace_key = pipeline.step_key(ca.TRACE)

    config.trigger_threshold = 9000
    assert pipeline.step_key(ca.TRACE) == trace_key
    assert pipeline.status(ca.TRACE) is ca.Status.CACHED
    assert pipeline.status(ca.TRIGGERS) is ca.Status.STALE
    assert pipeline.status(ca.SPIKES) is ca.Status.CACHED


@pytest.mark.parametrize("field, value", [
    ("checks_per_side", 10),
    ("temporal_dimension", 5),
    ("whiten", True),
])
def test_stimulus_and_analysis_parameters_never_touch_the_spikes(setup, field, value):
    """The long step must be immune to every parameter of the stimulus and the analysis."""
    config, tmp_path = setup
    pipeline = build(config, tmp_path)
    seed_spikes(pipeline)
    pipeline.run_all()
    spikes_key = pipeline.step_key(ca.SPIKES)

    setattr(config, field, value)
    assert pipeline.step_key(ca.SPIKES) == spikes_key
    assert pipeline.status(ca.SPIKES) is ca.Status.CACHED
    assert pipeline.status(ca.TRACE) is ca.Status.CACHED
    assert pipeline.status(ca.TRIGGERS) is ca.Status.CACHED
    assert pipeline.status(ca.ANALYSIS) is ca.Status.STALE


def test_switching_stimulus_keeps_the_other_one_built(setup, tmp_path):
    """Rebuilding a shifting-white-noise stimulus costs minutes; switching must not lose it."""
    config, root = setup
    pipeline = build(config, root)
    seed_spikes(pipeline)
    pipeline.run_all()
    swn_key = pipeline.step_key(ca.STIMULUS)

    source = root / "binarysource"
    source.write_bytes(bytes(np.random.default_rng(0).integers(0, 256, 4096, dtype=np.uint8)))
    config.kind = st.CHECKERBOARD
    config.checkerboard_source = str(source)
    config.n_checks = 8
    pipeline.run_all()
    assert pipeline.status(ca.STIMULUS) is ca.Status.CACHED
    assert pipeline.results.sta_3d.shape[2:] == (8, 8)

    config.kind = st.SWN
    assert pipeline.step_key(ca.STIMULUS) == swn_key
    assert pipeline.status(ca.STIMULUS) is ca.Status.CACHED


def test_whitening_adds_the_covariance_step_only_for_swn(setup):
    config, tmp_path = setup
    config.whiten = True
    pipeline = build(config, tmp_path)
    assert ca.COV in pipeline.steps()
    config.kind = st.CHECKERBOARD
    assert ca.COV not in pipeline.steps()


def test_whitened_run_keeps_the_raw_sta_too(setup):
    config, tmp_path = setup
    config.whiten = True
    pipeline = build(config, tmp_path)
    seed_spikes(pipeline)
    pipeline.run_all()
    assert pipeline.results.meta["whiten"] is True
    assert pipeline.results.spatial.shape == pipeline.results.spatial_raw.shape
    assert pipeline.covariance.shape == (100, 100)


def test_unknown_steps_before_a_recording_is_chosen(setup):
    config, tmp_path = setup
    config.recording_path = ""
    pipeline = build(config, tmp_path)
    assert pipeline.status(ca.TRACE) is None
    assert pipeline.status(ca.ANALYSIS) is None
    with pytest.raises(ValueError, match="not enough is known"):
        pipeline.run(ca.TRIGGERS)


def test_force_recomputes_one_step(setup):
    config, tmp_path = setup
    pipeline = build(config, tmp_path)
    seed_spikes(pipeline)
    pipeline.run_all()
    path = pipeline.cache.path(ca.TRIGGERS, pipeline.step_key(ca.TRIGGERS))
    before = path.stat().st_mtime_ns
    pipeline.run(ca.TRIGGERS, force=True)
    assert path.stat().st_mtime_ns != before
    assert pipeline.status(ca.TRIGGERS) is ca.Status.CACHED


def test_whitening_that_buries_the_fields_is_reported(setup):
    """The default is whitening on, so the tool has to say when it has made things worse."""
    config, tmp_path = setup
    config.whiten = True
    messages = []
    pipeline = build(config, tmp_path)
    pipeline.log = lambda message, level="info": messages.append((level, message))
    seed_spikes(pipeline)
    pipeline.run_all()
    whitening = [m for level, m in messages if m.startswith("Whitening")]
    assert whitening, "the whitening step says nothing about what it did"
    assert any("quality" in m for m in whitening)


# ----------------------------------------------------------------- the three lines
def test_group_status_aggregates_its_steps(setup):
    config, tmp_path = setup
    pipeline = build(config, tmp_path)
    seed_spikes(pipeline)
    assert pipeline.group_status("recording") is ca.Status.MISSING      # trace not read yet
    pipeline.run(ca.TRACE)
    assert pipeline.group_status("recording") is ca.Status.MISSING      # triggers still missing
    pipeline.run(ca.TRIGGERS)
    assert pipeline.group_status("recording") is ca.Status.CACHED
    config.trigger_threshold = 9000
    assert pipeline.group_status("recording") is ca.Status.STALE        # one stale is enough


def test_the_covariance_belongs_to_the_stimulus_line_only_when_it_runs(setup):
    config, tmp_path = setup
    pipeline = build(config, tmp_path)
    config.whiten = False
    assert pipeline.group_steps("stimulus") == (ca.STIMULUS,)
    config.whiten = True
    assert pipeline.group_steps("stimulus") == (ca.STIMULUS, ca.COV)
    config.kind = st.CHECKERBOARD
    assert pipeline.group_steps("stimulus") == (ca.STIMULUS,)


def test_run_group_forces_only_when_the_line_is_already_green(setup):
    config, tmp_path = setup
    pipeline = build(config, tmp_path)
    seed_spikes(pipeline)
    pipeline.run(ca.TRACE)
    stamp = pipeline.cache.path(ca.TRACE, pipeline.step_key(ca.TRACE)).stat().st_mtime_ns

    pipeline.run_group("recording")                 # grey: only does what is missing
    assert pipeline.cache.path(ca.TRACE, pipeline.step_key(ca.TRACE)).stat().st_mtime_ns == stamp
    assert pipeline.group_status("recording") is ca.Status.CACHED

    pipeline.run_group("recording")                 # green: does it again
    assert pipeline.cache.path(ca.TRACE, pipeline.step_key(ca.TRACE)).stat().st_mtime_ns != stamp


def test_drop_group_forgets_every_step_of_the_line(setup):
    config, tmp_path = setup
    pipeline = build(config, tmp_path)
    seed_spikes(pipeline)
    pipeline.run_all()
    pipeline.drop_group("recording")
    assert pipeline.group_status("recording") is ca.Status.MISSING
    for step in (ca.TRACE, ca.TRIGGERS, ca.SPIKES):
        assert pipeline.status(step) is ca.Status.MISSING
    assert pipeline.group_status("stimulus") is ca.Status.CACHED


def test_anything_done_is_what_makes_run_ask(setup):
    config, tmp_path = setup
    pipeline = build(config, tmp_path)
    assert not pipeline.anything_done()
    seed_spikes(pipeline)
    assert pipeline.anything_done()                 # the spikes alone already count


def test_run_all_can_force_every_step(setup):
    config, tmp_path = setup
    pipeline = build(config, tmp_path)
    seed_spikes(pipeline)
    pipeline.run_all()
    stamps = {step: pipeline.cache.path(step, pipeline.step_key(step)).stat().st_mtime_ns
              for step in pipeline.steps()}
    pipeline.run_all(force=True)
    for step, stamp in stamps.items():
        assert pipeline.cache.path(step, pipeline.step_key(step)).stat().st_mtime_ns != stamp
