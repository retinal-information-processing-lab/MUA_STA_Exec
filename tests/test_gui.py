"""The small window, headless: that it is small, and that each control does what it claims."""
import numpy as np
import pytest

from mua_sta import cache as ca
from mua_sta import stimulus as st
from mua_sta.config import Config
from mua_sta.pipeline import GROUPS

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6 import QtWidgets                                         # noqa: E402

from mua_sta.dialogs import FramePreview, SettingsDialog, TriggerPlot   # noqa: E402
from mua_sta.gui import Window                                      # noqa: E402
from tests.test_pipeline import seed_spikes, setup                  # noqa: E402,F401


@pytest.fixture(scope="session")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app, setup):                                             # noqa: F811
    config, root = setup
    config.path = str(root / "config.json")                         # never the real one
    window = Window(config)
    window.pipeline.cache = ca.Cache.for_recording(config.recording_path, root=root / "cache")
    # run in this thread, refreshing afterwards as the real Worker does when it finishes
    window._start = lambda work, what: (work(lambda message: None), window._refresh())
    yield window
    window.close()


def dots(window):
    return {group: row.dot._colour for group, row in window.rows.items()}


GREEN, RED, GREY = "#00FF00", "#FF4444", "#555555"


# ------------------------------------------------------------------------- the window
def test_it_is_small_and_has_three_steps(window):
    assert window.windowTitle() == "MUA STA"
    assert window.width() <= 520
    assert list(window.rows) == list(GROUPS)
    assert [row.name.text() for row in window.rows.values()] == [
        "Triggers and spikes", "Stimulus", "STA"]


def test_the_bench_parameters_are_not_on_screen(window):
    """Channels, format, rate, threshold and jitter never change here, so they are not shown."""
    labels = {w.text() for w in window.findChildren(QtWidgets.QLabel)}
    for gone in ("channels", "sample format", "sampling rate", "threshold", "allowed jitter",
                 "peak sign", "band from", "STA depth", "pixels per check"):
        assert gone not in labels
    assert "trigger channel" in labels and "Setup" in labels and "Recording" in labels


def test_the_recording_is_described(window):
    assert "256 ch" in window.recording_label.text()


def test_a_bad_recording_path_is_reported_not_raised(window):
    window.recording_row.set_path("/nowhere/at/all.raw")
    assert window.recording_label.text()
    assert window.pipeline.status(ca.TRACE) is None


# ------------------------------------------------------------------------- the switch
def test_the_switch_moves_the_page_and_the_config(window):
    window._on_kind(st.CHECKERBOARD)
    assert window.pages.currentIndex() == 0
    assert window.kind_buttons[st.CHECKERBOARD].isChecked()
    assert window.config.kind == st.CHECKERBOARD
    window._on_kind(st.SWN)
    assert window.pages.currentIndex() == 1
    assert window.kind_buttons[st.SWN].isChecked()
    assert not window.kind_buttons[st.CHECKERBOARD].isChecked()


def test_the_checkerboard_has_one_squares_box_for_both_axes(window, tmp_path):
    source = tmp_path / "binarysource"
    source.write_bytes(bytes(4096))
    window._on_kind(st.CHECKERBOARD)
    window.source_row.set_path(str(source))
    window.checks.setValue(24)
    spec = window.config.stimulus_spec()
    assert (spec.n_checks_x, spec.n_checks_y) == (24, 24)


def test_choosing_a_library_does_not_read_it(window, monkeypatch):
    """Picking a file must not start any work: the lattice is measured when the step runs."""
    import mua_sta.stimulus as st_mod
    reads = []
    monkeypatch.setattr(st_mod, "detect_swn_lattice",
                        lambda *a, **k: reads.append(a) or (5, 2, ""))
    window.bin_row.set_path(window.config.swn_bin)
    window._refresh()
    assert not reads
    assert window.pipeline.stimulus is None


def test_the_lattice_is_measured_when_the_stimulus_step_runs(window):
    seed_spikes(window.pipeline)
    window._on_run_group("recording")
    window._on_run_group("stimulus")
    assert window.pipeline.stimulus is not None
    assert window.pipeline.stimulus.shape[1:] == (10, 10)       # 5 checks x 2 shifts
    assert "80x80" not in window.status.text()


# -------------------------------------------------------------------------- the setup
def test_the_setup_box_reloads_that_rig_s_hidden_threshold(window):
    window._on_mea(2)
    assert window.config.threshold == 15630
    window._on_mea(3)
    assert window.config.threshold == 17714


def test_changing_setup_reddens_the_file_line_without_losing_the_trace(window):
    """The whole point of keeping six steps under three lines."""
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    window._refresh()
    assert dots(window)["recording"] == GREEN
    window._on_mea(2)
    assert dots(window)["recording"] == RED
    assert window.pipeline.status(ca.TRACE) is ca.Status.CACHED      # not re-read
    assert window.pipeline.status(ca.TRIGGERS) is ca.Status.STALE
    window._on_mea(3)
    assert dots(window)["recording"] == GREEN


# -------------------------------------------------------------------- the three lines
def test_dots_start_grey_and_turn_green(window):
    assert set(dots(window).values()) == {GREY}
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    window._refresh()
    assert set(dots(window).values()) == {GREEN}


def test_a_line_is_green_only_when_all_of_it_is(window):
    seed_spikes(window.pipeline)                        # spikes cached, triggers not yet
    window.pipeline.run(ca.TRACE)
    window._refresh()
    assert dots(window)["recording"] == GREY            # half done is not done
    window.pipeline.run(ca.TRIGGERS)
    window._refresh()
    assert dots(window)["recording"] == GREEN


def test_play_on_a_grey_line_does_only_what_is_missing(window):
    seed_spikes(window.pipeline)
    window.pipeline.run(ca.TRACE)
    before = window.pipeline.cache.path(ca.TRACE, window.pipeline.step_key(ca.TRACE))
    stamp = before.stat().st_mtime_ns
    window._on_run_group("recording")
    assert before.stat().st_mtime_ns == stamp           # the trace was reused
    assert dots(window)["recording"] == GREEN


def test_play_on_a_green_line_does_it_again(window):
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    path = window.pipeline.cache.path(ca.STIMULUS, window.pipeline.step_key(ca.STIMULUS))
    stamp = path.stat().st_mtime_ns
    window._on_run_group("stimulus")
    assert path.stat().st_mtime_ns != stamp


def test_erase_forgets_a_whole_line(window):
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    window._on_drop_group("stimulus")
    assert dots(window)["stimulus"] == GREY
    assert dots(window)["recording"] == GREEN           # the others are untouched


# ----------------------------------------------------------------------------- the run
def test_run_on_a_fresh_recording_does_not_ask(window, monkeypatch):
    """Nothing cached at all, so there is nothing to warn about losing."""
    asked = []
    monkeypatch.setattr(Window, "_confirm_redo", lambda self: asked.append(True) or True)
    assert not window.pipeline.anything_done()
    window._on_run()
    assert not asked
    assert set(dots(window).values()) == {GREEN}


def test_run_asks_when_only_part_of_the_work_exists(window, monkeypatch):
    """A line reads grey while half done, but that half is still work Run would throw away."""
    seed_spikes(window.pipeline)
    assert dots(window)["recording"] == GREY
    monkeypatch.setattr(Window, "_confirm_redo", lambda self: False)
    assert window.pipeline.anything_done()
    window._on_run()
    assert set(dots(window).values()) == {GREY}         # cancelled, so nothing ran


def test_run_asks_once_something_is_done(window, monkeypatch):
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    monkeypatch.setattr(Window, "_confirm_redo", lambda self: False)
    stamp = window.pipeline.cache.path(
        ca.ANALYSIS, window.pipeline.step_key(ca.ANALYSIS)).stat().st_mtime_ns
    window._on_run()
    assert window.pipeline.cache.path(
        ca.ANALYSIS, window.pipeline.step_key(ca.ANALYSIS)).stat().st_mtime_ns == stamp

    monkeypatch.setattr(Window, "_confirm_redo", lambda self: True)
    window._on_run()
    assert window.pipeline.cache.path(
        ca.ANALYSIS, window.pipeline.step_key(ca.ANALYSIS)).stat().st_mtime_ns != stamp


# -------------------------------------------------------------------------- the views
def test_results_open_in_their_own_window(window):
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    window._show_results()
    results = window.results_window
    assert results is not None and results.isWindow()
    assert results.grid is not None
    label = int(window.pipeline.results.labels[0])
    results.show_electrode(label)
    assert results.selected == label
    assert results.highlight is not None            # the selection is a shape, not a caption
    results.close()


def test_the_results_window_has_no_settings(window):
    """No orientation combo, no crop or flip boxes: the rig decides and the tiles are always
    cropped to where the fields are."""
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    window._show_results()
    results = window.results_window
    assert not results.findChildren(QtWidgets.QComboBox)
    assert not results.findChildren(QtWidgets.QCheckBox)
    assert [b.text() for b in results.findChildren(QtWidgets.QPushButton)] == ["Save image"]
    results.close()


def test_the_results_figure_can_be_saved(window, tmp_path, monkeypatch):
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    window._show_results()
    results = window.results_window
    target = tmp_path / "sta.png"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    results._on_save()
    assert target.exists() and target.stat().st_size > 1000
    results.close()


def test_the_results_grid_click_maps_back_to_an_electrode(window):
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    window._show_results()
    grid = window.results_window.grid
    rows, cols = grid.tile
    for label, (row, col) in window._mapping.items():
        x = col * (cols + grid.padding) + 0.5
        y = row * (rows + grid.padding) + 0.5
        assert grid.electrode_at(x, y, window._mapping) == label
    window.results_window.close()


def test_the_grid_uses_the_rig_s_own_orientation(window):
    """The stimulus axes are placed on the array by the rig, as its polarity and optics are."""
    from mua_sta.rigs import rig, to_array_frame
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    window._show_results()
    results = window.results_window
    expected = to_array_frame(window.pipeline.results.spatial, window.config.mea)
    direct = to_array_frame(window.pipeline.results.spatial,
                            window.config.mea, rig(window.config.mea).array_registration)
    assert np.array_equal(expected, direct)

    # the tiles are always cropped to where the fields are, with no box to tick; on a fixture
    # of pure noise there is nowhere to crop to and the whole frame is kept
    from mua_sta import sta as sta_mod
    box = sta_mod.field_bounds(to_array_frame(window.pipeline.results.spatial_raw,
                                              window.config.mea))
    rows, cols = expected.shape[1:]
    assert results.grid.tile == ((box[1] - box[0], box[3] - box[2]) if box else (rows, cols))
    results.close()


def test_the_frame_preview_says_what_to_do_before_the_stimulus_is_built(window):
    preview = FramePreview(window.config, None, window)
    assert "run the stimulus step" in preview.note.text()
    preview.close()


def test_the_frame_preview_shows_the_built_grid(window):
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    preview = FramePreview(window.config, window.pipeline.stimulus, window)
    assert "novel frame 0" in preview.note.text()
    preview.full.setChecked(True)
    assert "full DMD resolution" in preview.note.text()
    preview.close()


def test_the_trigger_plot_summarises_the_triggers(window):
    seed_spikes(window.pipeline)
    window.pipeline.run(ca.TRIGGERS)
    plot = TriggerPlot(window.pipeline.trace, window.config.fs, window.config.threshold,
                       window.pipeline.triggers, window)
    text = plot.findChildren(QtWidgets.QLabel)[-1].text()
    assert "triggers" in text and "17714 levels" in text
    plot.close()


def test_the_settings_dialog_edits_the_config(window):
    dialog = SettingsDialog(window.config, window.pipeline.cache, window)
    dialog.sigma.setValue(2.5)
    assert window.config.sigma_fraction == pytest.approx(0.025)
    dialog.checks.setEnabled(True)
    dialog.checks.setValue(7)
    assert window.config.checks_per_side == 7
    dialog.close()


# ------------------------------------------------------------------------- messages
def test_the_status_line_carries_the_last_message(window):
    window._log("something happened", "warn")
    assert window.status.text() == "something happened"
    assert "FF" in window.status.styleSheet().upper()       # warnings are not grey
