"""The window, headless: that it builds, and that the controls do what they claim."""
import numpy as np
import pytest

from mua_sta import cache as ca
from mua_sta import stimulus as st
from mua_sta.config import Config

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6 import QtWidgets                                        # noqa: E402

from mua_sta.gui import Window                                     # noqa: E402
from tests.test_pipeline import FRAMES_PER_SEQUENCE, seed_spikes, setup   # noqa: E402,F401


@pytest.fixture(scope="session")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app, setup, tmp_path_factory):                          # noqa: F811
    config, root = setup
    config.path = str(root / "config.json")                        # never touch the real one
    window = Window(config)
    window.pipeline.cache = ca.Cache.for_recording(config.recording_path, root=root / "cache")
    yield window
    window.close()


def test_builds_and_describes_the_recording(window):
    assert window.windowTitle() == "MUA STA"
    assert "8 ch" not in window.lbl_recording.text()               # 256-channel fixture
    assert "256 ch" in window.lbl_recording.text()
    assert "µV at the electrode" in window.lbl_threshold_uv.text()


def test_steps_start_as_not_done_and_turn_green(window):
    seed_spikes(window.pipeline)
    assert window.step_rows[ca.TRACE].note.text() == "not done"
    assert window.step_rows[ca.STIMULUS].note.text() == "not known yet"
    window.pipeline.run_all()
    window._refresh()
    for step in window.pipeline.steps():
        assert "cached" in window.step_rows[step].note.text(), step


def test_the_covariance_row_is_hidden_unless_whitening_swn(window):
    # isHidden() rather than isVisible(): the window is never shown in a headless test, so
    # isVisible() is False for every widget regardless of what the refresh decided.
    window.check_whiten.setChecked(False)
    assert window.step_rows[ca.COV].isHidden()
    window.check_whiten.setChecked(True)
    assert not window.step_rows[ca.COV].isHidden()
    window._on_kind(st.CHECKERBOARD)
    assert window.step_rows[ca.COV].isHidden()


def test_the_stimulus_menu_switches_the_page(window):
    window._on_kind(st.CHECKERBOARD)
    assert window.stimulus_pages.currentIndex() == 0
    assert window.kind_actions[st.CHECKERBOARD].isChecked()
    window._on_kind(st.SWN)
    assert window.stimulus_pages.currentIndex() == 1


def test_the_setup_menu_reloads_that_rig_s_threshold(window):
    window._on_mea(2)
    assert window.spin_threshold.value() == 15630
    window._on_mea(3)
    assert window.spin_threshold.value() == 17714


def test_dragging_the_threshold_line_sets_the_parameter(window):
    window.line_threshold.setPos(12345)
    window._on_threshold_line()
    assert window.spin_threshold.value() == 12345
    assert window.config.threshold == 12345
    assert window.pipeline.status(ca.TRIGGERS) in (ca.Status.MISSING, ca.Status.STALE)


def test_detect_fills_in_the_lattice(window):
    window._on_detect()
    assert (window.spin_checks.value(), window.spin_shifts.value()) == (5, 2)
    assert window.config.checks_per_side == 5


def test_results_draw_and_an_electrode_can_be_selected(window):
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    window._draw_results()
    assert window.tabs.currentIndex() == 1
    assert "electrodes" in window.lbl_grid.text()
    assert "+y up" in window.lbl_grid.text()

    label = int(window.pipeline.results.labels[0])
    window._show_electrode(label)
    assert f"electrode {label}" in window.lbl_electrode.text()
    x, y = window.curve_temporal.getData()
    assert x is not None and len(x) == window.config.temporal_dimension
    assert x[-1] == pytest.approx(0.0)                             # lag 0 is the spike itself


def test_the_grid_click_maps_back_to_an_electrode(window):
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    window._draw_results()
    rows, cols = window.grid.tile
    for label, (row, col) in window._mapping.items():
        x = col * (cols + window.grid.padding) + 0.5
        y = row * (rows + window.grid.padding) + 0.5
        assert window.grid.electrode_at(x, y, window._mapping) == label


def test_preview_says_what_to_do_before_the_stimulus_is_built(window):
    window._draw_preview()
    assert "run the stimulus step" in window.lbl_preview.text()
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    window._draw_preview()
    assert "analysis grid" in window.lbl_preview.text()


def test_full_resolution_preview_reads_the_library_directly(window):
    window.check_full_res.setChecked(True)
    window._draw_preview()
    assert "full DMD resolution" in window.lbl_preview.text()


def test_forgetting_one_step_turns_its_dot_grey(window):
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    window._on_drop_step(ca.TRACE)
    assert window.step_rows[ca.TRACE].note.text() == "not done"
    assert "cached" in window.step_rows[ca.SPIKES].note.text()


def test_a_bad_recording_path_is_reported_not_raised(window):
    window.row_recording.set_path("/nowhere/at/all.raw")
    assert "nowhere" in window.lbl_recording.text() or "No such" in window.lbl_recording.text()
    assert window.pipeline.status(ca.TRACE) is None


def test_config_round_trips_through_the_window(window):
    window.spin_temporal.setValue(7)
    window.spin_sigma.setValue(2.5)
    saved = Config.load(window.config.path)
    assert saved.temporal_dimension == 7
    assert saved.sigma_fraction == pytest.approx(0.025)


def test_check_orientation_needs_an_analysis_first(window):
    window._on_check_orientation()
    assert "Run the analysis first" in window.log_view.toPlainText()


def test_check_orientation_reports_what_the_data_says(window, monkeypatch):
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    window._draw_results()

    from mua_sta import sta as sta_mod
    monkeypatch.setattr(sta_mod, "measure_registration",
                        lambda *a: [("transpose_flip_xy", 0.98), ("identity", 0.05)])
    window.combo_registration.setCurrentIndex(window.combo_registration.findData("identity"))
    window._on_check_orientation()
    text = window.log_view.toPlainText()
    assert "the data says transpose_flip_xy" in text
    assert window.config.registration == "transpose_flip_xy"

    window._on_check_orientation()
    assert "confirmed" in window.log_view.toPlainText()


def test_registration_is_a_display_setting_not_part_of_the_analysis(window):
    """Changing how the array sits must not throw the STA away."""
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    before = window.pipeline.step_key(ca.ANALYSIS)
    window.combo_registration.setCurrentIndex(window.combo_registration.findData("transpose"))
    assert window.pipeline.step_key(ca.ANALYSIS) == before
    assert window.pipeline.status(ca.ANALYSIS) is ca.Status.CACHED
    assert "orientation transpose" in window.lbl_grid.text()


def test_flipping_the_grid_is_also_only_a_display_setting(window):
    seed_spikes(window.pipeline)
    window.pipeline.run_all()
    before = window.pipeline.step_key(ca.ANALYSIS)
    window.check_flip_y.setChecked(True)
    assert window.pipeline.step_key(ca.ANALYSIS) == before
    assert "row 0 at the top" in window.lbl_grid.text()
