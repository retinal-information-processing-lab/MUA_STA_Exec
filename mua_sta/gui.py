"""The window: pick the files, run the steps that need running, look at the result.

Laid out like the lab's other tools: three cards side by side, a shared console log and
progress bar underneath, one big RUN button. The stimulus kind and the setup are chosen from
the menu bar, the way Oriole picks which Wavefront it drives, and both are remembered.

The parts that exist to stop a wrong parameter costing a rerun are the STEPS card — a status
dot per step, run-just-this and forget-this-result buttons — and the trigger plot in the
RECORDING card, where the threshold is a line to drag over the real trace rather than a number
to guess.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from . import cache as ca
from . import recording as rec
from . import stimulus as st
from . import triggers as tg
from .config import Config
from .pipeline import STEP_LABELS, Pipeline
from .rigs import ARRAY_REGISTRATIONS, RIGS, UV_PER_LEVEL, rig, to_array_frame
from .theme import (DIM, RECORDING_COLOUR, STEPS_COLOUR, STIMULUS_COLOUR, STOP_STYLE, WARN,
                    apply_dark_theme)
from .widgets import (ConsoleLog, FileRow, StepRow, Worker, action_button, button, card,
                      combo_box, human_size, spin_box, title_label)

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

#: Blue for negative, white for nothing, red for positive — as the earlier stitched figure.
STA_COLOURS = pg.ColorMap([0.0, 0.5, 1.0],
                          [(50, 90, 255), (255, 255, 255), (255, 60, 60)])
DTYPES = [("uint16", "uint16"), ("int16", "int16")]


class Window(QtWidgets.QMainWindow):
    """One recording at a time: its files, its parameters, its six steps and its results."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.pipeline = Pipeline(config, log=self._log)
        self.worker: Worker | None = None
        self._loading = True            # suppresses change signals while widgets are filled
        self._mapping = rec.load_mapping()
        self._selected: int | None = None

        self.setWindowTitle("MUA STA")
        self.resize(1480, 940)
        self._build_menu()
        self._build()
        self._loading = False
        if config.load_error:
            self._log(config.load_error, "warn")
        self._log(f"Cache in {ca.CACHE_ROOT}", "info")
        self._refresh()

    # --------------------------------------------------------------------- menus
    def _build_menu(self):
        stimulus_menu = self.menuBar().addMenu("Stimulus")
        self.kind_actions = {}
        group = QtGui.QActionGroup(self)
        for kind in (st.CHECKERBOARD, st.SWN):
            action = QtGui.QAction(st.KIND_LABELS[kind], self, checkable=True)
            action.setChecked(self.config.kind == kind)
            action.triggered.connect(lambda _=False, k=kind: self._on_kind(k))
            group.addAction(action)
            stimulus_menu.addAction(action)
            self.kind_actions[kind] = action

        setup_menu = self.menuBar().addMenu("Setup")
        self.mea_actions = {}
        setups = QtGui.QActionGroup(self)
        for mea, settings in sorted(RIGS.items()):
            text = settings.label + ("" if settings.display_ready else "  (not tested)")
            action = QtGui.QAction(text, self, checkable=True)
            action.setChecked(self.config.mea == mea)
            if not settings.display_ready:
                action.setToolTip("this rig's stimulus display settings are not verified: "
                                  "frames may come out mirrored, rotated or inverted")
            action.triggered.connect(lambda _=False, m=mea: self._on_mea(m))
            setups.addAction(action)
            setup_menu.addAction(action)
            self.mea_actions[mea] = action

        cache_menu = self.menuBar().addMenu("Cache")
        cache_menu.addAction("Open the cache folder", self._open_cache)
        cache_menu.addAction("Forget this recording", self._forget_recording)
        cache_menu.addAction("Forget everything", self._forget_all)

    # --------------------------------------------------------------------- layout
    def _build(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setSpacing(8)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._setup_tab(), "Setup")
        self.tabs.addTab(self._results_tab(), "Results")
        root.addWidget(self.tabs, stretch=1)

        self.log_view = ConsoleLog()
        self.log_view.setFixedHeight(130)
        root.addWidget(self.log_view)

        bar = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet(f"color: {DIM};")
        bar.addWidget(self.status, stretch=1)
        bar.addWidget(self.progress, stretch=1)
        root.addLayout(bar)

        self.btn_run = action_button("RUN", self._on_run)
        root.addWidget(self.btn_run)

    def _setup_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        columns = QtWidgets.QHBoxLayout(page)
        columns.setSpacing(10)
        columns.addWidget(card(self._recording_card(), "recCard", RECORDING_COLOUR, 360))
        columns.addWidget(card(self._stimulus_card(), "stimCard", STIMULUS_COLOUR), stretch=1)
        columns.addWidget(card(self._steps_card(), "stepCard", STEPS_COLOUR, 400))
        return page

    # ----------------------------------------------------------- recording card
    def _recording_card(self) -> QtWidgets.QVBoxLayout:
        column = QtWidgets.QVBoxLayout()
        column.addWidget(title_label("RECORDING", RECORDING_COLOUR))

        self.row_recording = FileRow(self.config.recording_path, "Select a recording",
                                     "Raw recordings (*.raw);;All files (*)")
        self.row_recording.changed.connect(self._on_recording)
        column.addWidget(self.row_recording)
        self.lbl_recording = QtWidgets.QLabel("no recording chosen")
        self.lbl_recording.setStyleSheet(f"color: {DIM};")
        self.lbl_recording.setWordWrap(True)
        column.addWidget(self.lbl_recording)

        acquisition = QtWidgets.QGroupBox("Acquisition")
        form = QtWidgets.QFormLayout(acquisition)
        self.spin_channels = spin_box(self.config.n_channels, 1, 1024, slot=self._on_channels)
        self.combo_dtype = combo_box(DTYPES, self.config.dtype, self._on_dtype)
        self.spin_fs = spin_box(self.config.fs, 1000, 100000, " Hz", slot=self._on_fs)
        self.spin_trigger_channel = spin_box(self.config.trigger_channel, 0, 1023,
                                             slot=self._on_trigger_channel)
        self.spin_trigger_channel.setToolTip("0-based channel index; MCS label 127 is index 126")
        form.addRow("channels", self.spin_channels)
        form.addRow("sample format", self.combo_dtype)
        form.addRow("sampling rate", self.spin_fs)
        form.addRow("trigger channel", self.spin_trigger_channel)
        column.addWidget(acquisition)

        trigger = QtWidgets.QGroupBox("Trigger")
        tform = QtWidgets.QFormLayout(trigger)
        self.spin_threshold = spin_box(self.config.threshold, 1, 32767, slot=self._on_threshold)
        self.spin_threshold.setToolTip("ADC levels above the 32768 zero; drag the line on the "
                                       "trace below to set it by eye")
        self.lbl_threshold_uv = QtWidgets.QLabel("")
        self.lbl_threshold_uv.setStyleSheet(f"color: {DIM};")
        self.spin_jitter = spin_box(self.config.maximal_jitter * 1000, 0.01, 100, " ms", 2,
                                    0.05, self._on_jitter)
        self.spin_jitter.setToolTip("how far a gap between triggers may differ from the usual "
                                   "one before it is reported")
        tform.addRow("threshold", self.spin_threshold)
        tform.addRow("", self.lbl_threshold_uv)
        tform.addRow("allowed jitter", self.spin_jitter)
        column.addWidget(trigger)

        self.plot_trigger = pg.PlotWidget()
        self.plot_trigger.setFixedHeight(130)
        self.plot_trigger.setLabel("bottom", "time (s)")
        self.plot_trigger.setLabel("left", "ADC levels")
        self.plot_trigger.showGrid(x=True, y=True, alpha=0.15)
        self.curve_trigger = self.plot_trigger.plot(pen=pg.mkPen("#6FA8DC", width=1))
        self.line_threshold = pg.InfiniteLine(
            pos=self.config.threshold, angle=0, movable=True,
            pen=pg.mkPen(WARN, width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.line_threshold.sigPositionChangeFinished.connect(self._on_threshold_line)
        self.plot_trigger.addItem(self.line_threshold)
        column.addWidget(self.plot_trigger)
        self.lbl_trigger_hint = QtWidgets.QLabel(
            "run the trigger-channel step to see the trace here")
        self.lbl_trigger_hint.setStyleSheet(f"color: {DIM};")
        self.lbl_trigger_hint.setWordWrap(True)
        column.addWidget(self.lbl_trigger_hint)
        column.addStretch(1)
        return column

    # ----------------------------------------------------------- stimulus card
    def _stimulus_card(self) -> QtWidgets.QVBoxLayout:
        column = QtWidgets.QVBoxLayout()
        column.addWidget(title_label("STIMULUS", STIMULUS_COLOUR))

        self.stimulus_pages = QtWidgets.QStackedWidget()
        self.stimulus_pages.addWidget(self._checkerboard_page())
        self.stimulus_pages.addWidget(self._swn_page())
        self.stimulus_pages.setCurrentIndex(0 if self.config.kind == st.CHECKERBOARD else 1)
        column.addWidget(self.stimulus_pages)

        sequences = QtWidgets.QGroupBox("Sequences")
        form = QtWidgets.QFormLayout(sequences)
        self.spin_frames = spin_box(self.config.frames_per_sequence, 2, 100000,
                                    slot=self._on_frames)
        self.spin_frames.setToolTip("frames in one sequence: a novel half, then a repeated half")
        self.spin_temporal = spin_box(self.config.temporal_dimension, 1, 500, " frames",
                                      slot=self._on_temporal)
        self.spin_temporal.setToolTip("how far back the STA looks before each spike")
        form.addRow("frames per sequence", self.spin_frames)
        form.addRow("STA depth", self.spin_temporal)
        self.lbl_sequences = QtWidgets.QLabel("—")
        self.lbl_sequences.setStyleSheet(f"color: {DIM};")
        self.lbl_sequences.setWordWrap(True)
        form.addRow("from the triggers", self.lbl_sequences)
        column.addWidget(sequences)

        preview = QtWidgets.QGroupBox("Frame preview")
        pbox = QtWidgets.QVBoxLayout(preview)
        row = QtWidgets.QHBoxLayout()
        self.spin_preview = spin_box(0, 0, 10_000_000, slot=lambda _: self._draw_preview())
        self.check_full_res = QtWidgets.QCheckBox("full resolution")
        self.check_full_res.setToolTip("the frame as the DMD showed it, to compare with the "
                                       "lab's StimulusDisplayer; otherwise the analysis grid")
        self.check_full_res.toggled.connect(lambda _: self._draw_preview())
        row.addWidget(QtWidgets.QLabel("frame"))
        row.addWidget(self.spin_preview)
        row.addWidget(self.check_full_res)
        row.addStretch(1)
        pbox.addLayout(row)
        self.view_preview = pg.GraphicsLayoutWidget()
        self.view_preview.setMinimumHeight(260)
        self.vb_preview = self.view_preview.addViewBox(lockAspect=True, enableMenu=False)
        self.vb_preview.setMouseEnabled(x=False, y=False)
        self.img_preview = pg.ImageItem()
        self.vb_preview.addItem(self.img_preview)
        # refit on resize: the first fit happens before the window has its final size
        self.vb_preview.sigResized.connect(
            lambda _: self.vb_preview.autoRange(items=[self.img_preview]))
        pbox.addWidget(self.view_preview)
        self.lbl_preview = QtWidgets.QLabel("—")
        self.lbl_preview.setStyleSheet(f"color: {DIM};")
        self.lbl_preview.setWordWrap(True)
        pbox.addWidget(self.lbl_preview)
        column.addWidget(preview, stretch=1)
        return column

    def _checkerboard_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        self.row_source = FileRow(self.config.checkerboard_source, "Select the binary source",
                                  "All files (*)")
        self.row_source.changed.connect(self._on_source)
        self.spin_checks_x = spin_box(self.config.n_checks_x, 1, 1024, slot=self._on_checks_x)
        self.spin_checks_y = spin_box(self.config.n_checks_y, 1, 1024, slot=self._on_checks_y)
        self.spin_pixels_per_check = spin_box(self.config.pixels_per_check, 1, 512, " px",
                                              slot=self._on_pixels_per_check)
        self.spin_pixels_per_check.setToolTip(
            "how wide a check was drawn on the DMD; sets the µm the result is reported in")
        form.addRow("binary source", self.row_source)
        form.addRow("checks across", self.spin_checks_x)
        form.addRow("checks down", self.spin_checks_y)
        form.addRow("pixels per check", self.spin_pixels_per_check)
        return page

    def _swn_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        self.row_swn_bin = FileRow(self.config.swn_bin, "Select the SWN frame library",
                                   "Stimulus bins (*.bin);;All files (*)")
        self.row_swn_bin.changed.connect(self._on_swn_bin)
        self.row_swn_vec = FileRow(self.config.swn_vec, "Select the SWN playlist",
                                   "Playlists (*.vec);;All files (*)")
        self.row_swn_vec.changed.connect(self._on_swn_vec)
        self.spin_checks = spin_box(self.config.checks_per_side, 1, 512, slot=self._on_checks)
        self.spin_checks.setToolTip("checks per side of the stimulus, from its design")
        self.spin_shifts = spin_box(self.config.shifts_per_check, 1, 64, slot=self._on_shifts)
        self.spin_shifts.setToolTip("shift steps per check: the analysis grid has one cell each")
        detect = button("Detect", self._on_detect,
                        "measure both from the frames in the chosen library")
        self.check_whiten = QtWidgets.QCheckBox("whiten the spatial STA")
        self.check_whiten.setChecked(self.config.whiten)
        self.check_whiten.setToolTip("shifting white noise is spatially correlated, so the raw "
                                     "STA is the receptive field blurred by the stimulus's own "
                                     "covariance; this divides it back out")
        self.check_whiten.toggled.connect(self._on_whiten)
        self.spin_sigma = spin_box(self.config.sigma_fraction * 100, 0.01, 100, " %", 2, 0.5,
                                   self._on_sigma)
        self.spin_sigma.setToolTip("regularisation, as a percentage of the pixel variance")
        form.addRow("frame library (.bin)", self.row_swn_bin)
        form.addRow("playlist (.vec)", self.row_swn_vec)
        form.addRow("checks per side", self.spin_checks)
        form.addRow("shifts per check", self.spin_shifts)
        form.addRow("", detect)
        form.addRow(self.check_whiten)
        form.addRow("regularisation", self.spin_sigma)
        return page

    # -------------------------------------------------------------- steps card
    def _steps_card(self) -> QtWidgets.QVBoxLayout:
        column = QtWidgets.QVBoxLayout()
        column.addWidget(title_label("STEPS", STEPS_COLOUR))
        hint = QtWidgets.QLabel(
            "RUN does only what is not green. A yellow dot means the step was done with "
            "different settings; grey means not yet.")
        hint.setStyleSheet(f"color: {DIM};")
        hint.setWordWrap(True)
        column.addWidget(hint)

        self.step_rows: dict[str, StepRow] = {}
        for step in (ca.TRACE, ca.TRIGGERS, ca.SPIKES, ca.STIMULUS, ca.COV, ca.ANALYSIS):
            row = StepRow(step, STEP_LABELS[step])
            row.run_requested.connect(self._on_run_step)
            row.drop_requested.connect(self._on_drop_step)
            self.step_rows[step] = row
            column.addWidget(row)

        detection = QtWidgets.QGroupBox("MUA detection")
        form = QtWidgets.QFormLayout(detection)
        self.spin_detect = spin_box(self.config.detect_threshold, 0.5, 50, " MAD", 1, 0.5,
                                    self._on_detect_threshold)
        self.combo_sign = combo_box([("negative", "neg"), ("positive", "pos"), ("both", "both")],
                                    self.config.peak_sign, self._on_peak_sign)
        self.spin_low = spin_box(self.config.freq_min, 1, 10000, " Hz", slot=self._on_freq_min)
        self.spin_high = spin_box(self.config.freq_max, 100, 20000, " Hz", slot=self._on_freq_max)
        self.check_car = QtWidgets.QCheckBox("common reference")
        self.check_car.setChecked(self.config.common_reference)
        self.check_car.toggled.connect(self._on_car)
        self.spin_jobs = spin_box(self.config.n_jobs, 1, 128, slot=self._on_jobs)
        form.addRow("threshold", self.spin_detect)
        form.addRow("peak sign", self.combo_sign)
        form.addRow("band from", self.spin_low)
        form.addRow("band to", self.spin_high)
        form.addRow(self.check_car)
        form.addRow("parallel jobs", self.spin_jobs)
        column.addWidget(detection)

        self.lbl_cache = QtWidgets.QLabel("—")
        self.lbl_cache.setStyleSheet(f"color: {DIM};")
        self.lbl_cache.setWordWrap(True)
        column.addWidget(self.lbl_cache)
        column.addStretch(1)
        return column

    # ------------------------------------------------------------- results tab
    def _results_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left = QtWidgets.QWidget()
        lbox = QtWidgets.QVBoxLayout(left)
        self.lbl_grid = QtWidgets.QLabel("no analysis yet — press RUN")
        self.lbl_grid.setStyleSheet(f"color: {DIM};")
        self.lbl_grid.setWordWrap(True)
        lbox.addWidget(self.lbl_grid)

        orientation = QtWidgets.QHBoxLayout()
        orientation.addWidget(QtWidgets.QLabel("array orientation"))
        self.combo_registration = combo_box(
            [("this rig's default", "")] + [(name, name) for name in ARRAY_REGISTRATIONS],
            self.config.array_registration, self._on_registration)
        self.combo_registration.setToolTip(
            "how the stimulus axes sit on the electrode grid — a property of the setup, not of "
            "the stimulus file. Check orientation measures it from the data.")
        orientation.addWidget(self.combo_registration)
        orientation.addWidget(button("Check orientation", self._on_check_orientation,
                                      "measure it: an electrode's position has to predict "
                                      "where its receptive field is"))
        self.check_crop = QtWidgets.QCheckBox("crop tiles to the fields")
        self.check_crop.setChecked(self.config.crop_tiles)
        self.check_crop.setToolTip("the stimulus covers much more of the retina than the array "
                                   "does, so most of each tile is empty")
        self.check_crop.toggled.connect(self._on_crop)
        orientation.addWidget(self.check_crop)
        self.check_flip_y = QtWidgets.QCheckBox("row 0 at the top")
        self.check_flip_y.setChecked(self.config.flip_grid_y)
        self.check_flip_y.toggled.connect(self._on_flip_y)
        orientation.addWidget(self.check_flip_y)
        orientation.addStretch(1)
        lbox.addLayout(orientation)
        self.view_grid = pg.GraphicsLayoutWidget()
        self.vb_grid = self.view_grid.addViewBox(lockAspect=True, enableMenu=False)
        self.img_grid = pg.ImageItem()
        self.vb_grid.addItem(self.img_grid)
        self.img_grid.setLookupTable(STA_COLOURS.getLookupTable(nPts=256))
        self.view_grid.scene().sigMouseClicked.connect(self._on_grid_click)
        lbox.addWidget(self.view_grid, stretch=1)
        splitter.addWidget(left)

        right = QtWidgets.QWidget()
        rbox = QtWidgets.QVBoxLayout(right)
        self.lbl_electrode = QtWidgets.QLabel("click an electrode on the grid")
        self.lbl_electrode.setStyleSheet(f"color: {DIM};")
        self.lbl_electrode.setWordWrap(True)
        rbox.addWidget(self.lbl_electrode)
        detail = pg.GraphicsLayoutWidget()
        self.vb_cell = detail.addViewBox(lockAspect=True, enableMenu=False, row=0, col=0)
        self.vb_cell.setMouseEnabled(x=False, y=False)
        self.img_cell = pg.ImageItem()
        self.img_cell.setLookupTable(STA_COLOURS.getLookupTable(nPts=256))
        self.vb_cell.addItem(self.img_cell)
        self.vb_cell.sigResized.connect(
            lambda _: self.vb_cell.autoRange(items=[self.img_cell]))
        self.plot_temporal = detail.addPlot(row=1, col=0)
        self.plot_temporal.setLabel("bottom", "time before the spike (s)")
        self.plot_temporal.setLabel("left", "STA")
        self.plot_temporal.showGrid(x=True, y=True, alpha=0.15)
        self.curve_temporal = self.plot_temporal.plot(pen=pg.mkPen("#FFD700", width=2))
        self.plot_raster = detail.addPlot(row=2, col=0)
        self.plot_raster.setLabel("bottom", "time in the repeated half (s)")
        self.plot_raster.setLabel("left", "sequence")
        self.curve_raster = pg.PlotCurveItem(pen=pg.mkPen("#E0E0E0", width=1), connect="pairs")
        self.plot_raster.addItem(self.curve_raster)
        detail.ci.layout.setRowStretchFactor(0, 3)
        detail.ci.layout.setRowStretchFactor(1, 2)
        detail.ci.layout.setRowStretchFactor(2, 3)
        rbox.addWidget(detail, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([820, 480])
        return page

    # ------------------------------------------------------------------ logging
    def _log(self, message: str, level: str = "info"):
        if hasattr(self, "log_view"):
            self.log_view.log(message, level)

    # ------------------------------------------------------------- config edits
    def _changed(self, reset: bool = False):
        """Save the config and bring the step states up to date."""
        if self._loading:
            return
        self.config.save()
        if reset:
            self.pipeline.reset()
        self._refresh()

    def _on_kind(self, kind: str):
        self.config.kind = kind
        self.kind_actions[kind].setChecked(True)        # also when called from code, not a click
        self.stimulus_pages.setCurrentIndex(0 if kind == st.CHECKERBOARD else 1)
        self._log(f"Stimulus: {st.KIND_LABELS[kind]}", "info")
        self._changed()

    def _on_mea(self, mea: int):
        self.config.mea = mea
        self.mea_actions[mea].setChecked(True)
        settings = rig(mea)
        self.config.trigger_threshold = None            # back to this rig's default
        self._loading = True
        self.spin_threshold.setValue(self.config.threshold)
        self._loading = False
        self._log(f"Setup: {settings.label} — trigger threshold {self.config.threshold} levels, "
                  f"{'inverted' if settings.invert_polarity else 'normal'} polarity, optics "
                  f"{settings.optical_transform or 'none'}", "info")
        if not settings.display_ready:
            self._log(f"{settings.label} stimulus display settings are not verified: frames may "
                      "come out mirrored, rotated or inverted", "warn")
        self._changed()

    def _on_recording(self, path: str):
        self.config.recording_path = path
        if path:
            self.config.last_dir = str(Path(path).parent)
        self._changed(reset=True)

    def _on_channels(self, value): self.config.n_channels = int(value); self._changed(reset=True)
    def _on_dtype(self, value): self.config.dtype = str(value); self._changed(reset=True)
    def _on_fs(self, value): self.config.fs = float(value); self._changed(reset=True)

    def _on_trigger_channel(self, value):
        self.config.trigger_channel = int(value)
        self._changed(reset=True)

    def _on_threshold(self, value):
        self.config.trigger_threshold = int(value)
        self.line_threshold.setPos(int(value))
        self._changed()

    def _on_threshold_line(self):
        value = max(1, int(round(self.line_threshold.value())))
        self._loading = True
        self.spin_threshold.setValue(value)
        self._loading = False
        self.config.trigger_threshold = value
        self._changed()

    def _on_jitter(self, value): self.config.maximal_jitter = float(value) / 1000; self._changed()
    def _on_frames(self, value): self.config.frames_per_sequence = int(value); self._changed()
    def _on_temporal(self, value): self.config.temporal_dimension = int(value); self._changed()
    def _on_source(self, path): self.config.checkerboard_source = path; self._changed()
    def _on_checks_x(self, value): self.config.n_checks_x = int(value); self._changed()
    def _on_checks_y(self, value): self.config.n_checks_y = int(value); self._changed()
    def _on_pixels_per_check(self, v): self.config.pixels_per_check = int(v); self._changed()
    def _on_swn_vec(self, path): self.config.swn_vec = path; self._changed()
    def _on_checks(self, value): self.config.checks_per_side = int(value); self._changed()
    def _on_shifts(self, value): self.config.shifts_per_check = int(value); self._changed()
    def _on_whiten(self, on): self.config.whiten = bool(on); self._changed()
    def _on_sigma(self, value): self.config.sigma_fraction = float(value) / 100; self._changed()
    def _on_detect_threshold(self, v): self.config.detect_threshold = float(v); self._changed()
    def _on_peak_sign(self, value): self.config.peak_sign = str(value); self._changed()
    def _on_freq_min(self, value): self.config.freq_min = float(value); self._changed()
    def _on_freq_max(self, value): self.config.freq_max = float(value); self._changed()
    def _on_car(self, on): self.config.common_reference = bool(on); self._changed()
    def _on_jobs(self, value): self.config.n_jobs = int(value); self._changed()

    def _on_registration(self, value):
        self.config.array_registration = str(value)
        self._changed()
        self._draw_results()

    def _on_crop(self, on: bool):
        self.config.crop_tiles = bool(on)
        self._changed()
        self._draw_results()

    def _on_flip_y(self, on: bool):
        self.config.flip_grid_y = bool(on)
        self._changed()
        self._draw_results()

    def _on_check_orientation(self):
        """Measure how the array sits in the stimulus, from the receptive fields themselves."""
        results = self.pipeline.results
        if results is None:
            self._log("Run the analysis first: there are no receptive fields to measure", "warn")
            return
        from . import sta as sta_mod
        # measured on the unwhitened STA: whitening is there to sharpen a field, not to find
        # one, and on noisy multi-unit data it scatters the peaks the measurement relies on
        scores = sta_mod.measure_registration(results.spatial_raw, results.labels, self._mapping)
        best, score = scores[0]
        runner_up = scores[1][1]
        self._log("Array orientation, by how well electrode position predicts receptive-field "
                  "position: " + ", ".join(f"{name} {value:+.2f}" for name, value in scores[:4]),
                  "info")
        if score < 0.5:
            self._log(f"No clear winner (best {best} at {score:+.2f}). Either there are too few "
                      "receptive fields, or the stimulus geometry is wrong.", "warn")
            return
        current = self.config.registration
        if best == current:
            self._log(f"{rig(self.config.mea).label}: {best} confirmed ({score:+.2f}, next "
                      f"{runner_up:+.2f}) — the grid is drawn the right way round", "info")
        else:
            self._log(f"{rig(self.config.mea).label}: the data says {best} ({score:+.2f}), not "
                      f"{current}. Setting it for this session; put it in mua_sta/rigs.py to "
                      "keep it.", "warn")
            self.combo_registration.setCurrentIndex(self.combo_registration.findData(best))

    def _on_swn_bin(self, path: str):
        self.config.swn_bin = path
        self._changed()
        if path:
            self._on_detect()

    def _on_detect(self):
        """Measure the lattice from the chosen library and offer it in the two boxes."""
        if not self.config.swn_bin:
            self._log("Choose a frame library (.bin) first", "warn")
            return
        try:
            checks, shifts, note = st.detect_swn_lattice(self.config.swn_bin, self.config.mea)
        except (OSError, ValueError) as error:
            self._log(f"Could not measure the lattice: {error}", "error")
            return
        self._log(note, "info")
        self._loading = True
        self.spin_checks.setValue(checks)
        self.spin_shifts.setValue(shifts)
        self._loading = False
        self.config.checks_per_side, self.config.shifts_per_check = checks, shifts
        self._changed()

    # -------------------------------------------------------------------- cache
    def _open_cache(self):
        ca.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(ca.CACHE_ROOT)))

    def _forget_recording(self):
        self.pipeline.cache.clear_recording()
        self.pipeline.reset()
        self._log("Forgot this recording's cached steps", "info")
        self._refresh()

    def _forget_all(self):
        answer = QtWidgets.QMessageBox.question(
            self, "Forget everything",
            "Delete every cached step of every recording, including rebuilt stimuli?")
        if answer is QtWidgets.QMessageBox.StandardButton.Yes:
            self.pipeline.cache.clear_all()
            self.pipeline.reset()
            self._log("Cache emptied", "info")
            self._refresh()

    def _on_drop_step(self, step: str):
        step_key = self.pipeline.step_key(step)
        if step_key is None:
            return
        self.pipeline.cache.drop(step, step_key)
        self.pipeline._loaded.pop(step, None)
        self._log(f"{STEP_LABELS[step]}: cached result forgotten", "info")
        self._refresh()

    # --------------------------------------------------------------------- runs
    def _busy(self, busy: bool):
        self.progress.setVisible(busy)
        self.btn_run.setEnabled(not busy)
        for row in self.step_rows.values():
            row.run.setEnabled(not busy)

    def _start(self, work, what: str):
        if self.worker is not None and self.worker.isRunning():
            self._log("Something is already running", "warn")
            return
        self._busy(True)
        self.status.setText(what)
        self.worker = Worker(work)
        self.worker.progressed.connect(self.status.setText)
        self.worker.logged.connect(self._log)
        self.worker.finished_ok.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _on_run(self):
        self._start(lambda progress: self.pipeline.run_all(progress), "running")

    def _on_run_step(self, step: str):
        self._start(lambda progress: self.pipeline.run(step, progress, force=True),
                    f"running {STEP_LABELS[step].lower()}")

    def _finished(self):
        self._busy(False)
        self.status.setText("done")
        self._refresh()
        self._draw_results()
        self._draw_trigger_trace()
        self._draw_preview()

    def _failed(self, message: str):
        self._busy(False)
        self.status.setText("failed")
        self._log(message, "error")
        self._refresh()

    # ------------------------------------------------------------------ refresh
    def _refresh(self):
        self._refresh_recording_label()
        self._refresh_steps()
        self.lbl_threshold_uv.setText(
            f"{tg.threshold_to_uv(self.config.threshold):,.0f} µV at the electrode")
        self.spin_sigma.setEnabled(self.config.whiten)
        n_sequences = self.pipeline.n_sequences()
        if n_sequences is None:
            self.lbl_sequences.setText("run the triggers step")
        else:
            frames = self.pipeline.stimulus_frame_count()
            self.lbl_sequences.setText(f"{n_sequences} sequences, {frames} novel frames")
        self._refresh_cache_label()
        self._refresh_grid_geometry()

    def _refresh_recording_label(self):
        if not self.config.recording_path:
            self.lbl_recording.setText("no recording chosen")
            return
        try:
            info = rec.open_info(self.config.recording_path, self.config.n_channels,
                                 self.config.dtype, self.config.fs)
        except (OSError, ValueError) as error:
            self.lbl_recording.setText(str(error))
            return
        self.lbl_recording.setText(info.describe())

    def _refresh_steps(self):
        applicable = set(self.pipeline.steps())
        for step, row in self.step_rows.items():
            row.setVisible(step in applicable)
            if step not in applicable:
                continue
            status = self.pipeline.status(step)
            step_key = self.pipeline.step_key(step)
            if status is None:
                row.set_state(None, "not known yet")
                continue
            size = self.pipeline.cache.size(step, step_key)
            note = {ca.Status.CACHED: f"cached · {human_size(size)}",
                    ca.Status.STALE: "settings changed",
                    ca.Status.MISSING: "not done"}[status]
            row.set_state(str(status), note)

    def _refresh_cache_label(self):
        try:
            total = self.pipeline.cache.total_size()
        except OSError:
            total = 0
        self.lbl_cache.setText(f"Cache: {human_size(total)} in {ca.CACHE_ROOT}")

    def _refresh_grid_geometry(self):
        try:
            spec = self.config.stimulus_spec()
        except (OSError, ValueError):
            return
        try:
            cell = spec.cell_size_um
        except (OSError, ValueError):
            cell = None
        rows, cols = spec.grid_shape
        text = f"{rows}x{cols} cells"
        if cell:
            text += f", {cell:.1f} µm each"
        self.lbl_preview.setText(text)

    # ------------------------------------------------------------------ drawing
    def _draw_trigger_trace(self):
        if self.pipeline.trace is None:
            return
        t, low, high = tg.envelope(self.pipeline.trace, self.config.fs)
        self.curve_trigger.setData(t, high)
        self.plot_trigger.setYRange(float(min(low.min(), 0)), float(high.max() * 1.1 + 1))
        self.line_threshold.setPos(self.config.threshold)
        self.lbl_trigger_hint.setText(
            "drag the dashed line to set the threshold, then run the triggers step")

    def _draw_preview(self):
        """Show one stimulus frame: the analysis grid, or the full DMD frame to compare with
        the lab's StimulusDisplayer."""
        index = int(self.spin_preview.value())
        frame, note = None, ""
        if self.check_full_res.isChecked() and self.config.kind == st.SWN:
            try:
                from .binfile import BinFile
                with BinFile(self.config.swn_bin, self.config.mea) as library:
                    index = min(index, len(library) - 1)
                    frame = library.frame(index)
                note = f"library frame {index}, full DMD resolution {frame.shape[1]}x{frame.shape[0]}"
            except (OSError, ValueError) as error:
                note = str(error)
        elif self.pipeline.stimulus is not None:
            index = min(index, self.pipeline.stimulus.shape[0] - 1)
            frame = self.pipeline.stimulus[index].astype(np.float32) / 255.0
            note = (f"novel frame {index} of {self.pipeline.stimulus.shape[0]}, "
                    f"analysis grid {frame.shape[1]}x{frame.shape[0]}")
        else:
            note = "run the stimulus step, or tick full resolution to read the library directly"
        if frame is not None:
            self.img_preview.setImage(frame, levels=(0.0, 1.0))
            self.vb_preview.autoRange(items=[self.img_preview])
        self.lbl_preview.setText(note + "  ·  shown as the retina saw it (+y up)")

    def _draw_results(self):
        results = self.pipeline.results
        if results is None:
            return
        from . import sta as sta_mod
        registered = to_array_frame(results.spatial, self.config.mea, self.config.registration)
        crop = None
        if self.config.crop_tiles:
            # measured on the unwhitened STA, where the fields are unambiguous
            crop = sta_mod.field_bounds(
                to_array_frame(results.spatial_raw, self.config.mea, self.config.registration))
        self.grid = sta_mod.stitch(registered, results.labels, self._mapping, crop=crop)
        image = np.nan_to_num(self.grid.image, nan=0.0)
        if self.config.flip_grid_y:
            image = image[::-1]
        self.img_grid.setImage(image, levels=(-1.0, 1.0))
        self.vb_grid.autoRange(items=[self.img_grid])
        whitened = " · whitened" if results.meta.get("whiten") else ""
        where = "row 0 at the top" if self.config.flip_grid_y else "row 0 at the bottom (+y up)"
        tile = f"{self.grid.tile[0]}x{self.grid.tile[1]} cells per tile"
        self.lbl_grid.setText(
            f"{results.labels.size} electrodes · {st.KIND_LABELS[results.meta['kind']]} · "
            f"{results.n_sequences} sequences · {results.meta['frequency']:.2f} Hz{whitened} · "
            f"orientation {self.config.registration} · {tile} · {where}")
        if self._selected is not None:
            self._show_electrode(self._selected)
        self.tabs.setCurrentIndex(1)

    def _on_grid_click(self, event):
        if self.pipeline.results is None or not hasattr(self, "grid"):
            return
        point = self.vb_grid.mapSceneToView(event.scenePos())
        y = point.y()
        if self.config.flip_grid_y:
            y = self.grid.image.shape[0] - 1 - y
        label = self.grid.electrode_at(point.x(), y, self._mapping)
        if label is not None:
            self._selected = label
            self._show_electrode(label)

    def _show_electrode(self, label: int):
        from .sta import rf_quality as sta_mod_quality
        results = self.pipeline.results
        index = results.index_of(label)
        if index is None:
            return
        row, col = self._mapping[label]
        lag = results.peak[index][0]
        frequency = results.meta["frequency"] or 1.0
        depth = results.sta_3d.shape[1]

        cell = to_array_frame(results.spatial[index], self.config.mea,
                              self.config.registration)
        self.img_cell.setImage(cell, levels=(-1.0, 1.0))
        self.vb_cell.autoRange(items=[self.img_cell])
        lags = (np.arange(depth) - (depth - 1)) / frequency
        self.curve_temporal.setData(lags, results.temporal[index])
        self.plot_temporal.setYRange(-1.05, 1.05)

        trains = results.raster(label)
        sizes = np.array([train.size for train in trains])
        if sizes.sum():
            x = np.repeat(np.concatenate(trains), 2)
            base = np.repeat(np.arange(len(trains)), 2 * sizes)
            y = base + np.tile([0.05, 0.95], int(sizes.sum()))
            self.curve_raster.setData(x, y)
        else:
            self.curve_raster.setData([], [])
        self.plot_raster.setYRange(0, max(1, len(trains)))

        quality = sta_mod_quality(results.spatial[index])
        self.lbl_electrode.setText(
            f"electrode {label} · grid row {row}, column {col} · peak at lag "
            f"{(lag - (depth - 1)) / frequency * 1000:.0f} ms · receptive-field strength "
            f"{quality:.0f} · {int(sizes.sum())} spikes in the repeated halves")

    # ------------------------------------------------------------------ closing
    def closeEvent(self, event):
        self.config.save()
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(2000)
        super().closeEvent(event)


def run() -> int:
    app = QtWidgets.QApplication(sys.argv)
    QtCore.QLocale.setDefault(QtCore.QLocale.c())       # a decimal point in every number box
    apply_dark_theme(app)
    window = Window(Config.load())
    window.show()
    return app.exec()
