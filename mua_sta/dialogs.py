"""The three things the small window keeps out of sight: settings, and two ways to look.

Nothing here is needed to run an analysis. The gear holds the parameters that do not change
between experiments, and the two viewers exist to answer "is this right?" — is the trigger
channel what it should be, and is the stimulus coming back the way it was projected.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from . import cache as ca
from . import stimulus as st
from . import triggers as tg
from .config import Config
from .theme import DIM, WARN
from .widgets import button, combo_box, human_size, spin_box


class SettingsDialog(QtWidgets.QDialog):
    """Everything the bench does not need to see, one click away under the gear."""

    def __init__(self, config: Config, cache: ca.Cache, parent=None):
        super().__init__(parent)
        self.config = config
        self.cache = cache
        self.setWindowTitle("Settings")
        self.setMinimumWidth(380)
        layout = QtWidgets.QVBoxLayout(self)

        detection = QtWidgets.QGroupBox("MUA detection")
        form = QtWidgets.QFormLayout(detection)
        form.addRow("threshold", spin_box(config.detect_threshold, 0.5, 50, " MAD", 1, 0.5,
                                          self._set("detect_threshold", float)))
        form.addRow("peak sign", combo_box([("negative", "neg"), ("positive", "pos"),
                                            ("both", "both")], config.peak_sign,
                                           self._set("peak_sign", str)))
        form.addRow("band from", spin_box(config.freq_min, 1, 10000, " Hz",
                                          slot=self._set("freq_min", float)))
        form.addRow("band to", spin_box(config.freq_max, 100, 20000, " Hz",
                                        slot=self._set("freq_max", float)))
        form.addRow(self._check("common reference", "common_reference"))
        form.addRow("parallel jobs", spin_box(config.n_jobs, 1, 128,
                                              slot=self._set("n_jobs", int)))
        layout.addWidget(detection)

        analysis = QtWidgets.QGroupBox("Sequences and STA")
        form = QtWidgets.QFormLayout(analysis)
        form.addRow("frames per sequence", spin_box(config.frames_per_sequence, 2, 100000,
                                                    slot=self._set("frames_per_sequence", int)))
        form.addRow("STA depth", spin_box(config.temporal_dimension, 1, 500, " frames",
                                          slot=self._set("temporal_dimension", int)))
        layout.addWidget(analysis)

        stimulus = QtWidgets.QGroupBox("Stimulus")
        form = QtWidgets.QFormLayout(stimulus)
        pixels = spin_box(config.pixels_per_check, 1, 512, " px",
                          slot=self._set("pixels_per_check", int))
        pixels.setToolTip("checkerboard only: how wide a check was drawn on the DMD. It sets "
                          "the µm a result is reported in and nothing else.")
        form.addRow("pixels per check", pixels)
        self.auto = self._check("measure the noise lattice from the frames", "swn_lattice_auto")
        self.auto.toggled.connect(self._toggle_lattice)
        form.addRow(self.auto)
        self.checks = spin_box(config.checks_per_side, 1, 512,
                               slot=self._set("checks_per_side", int))
        self.shifts = spin_box(config.shifts_per_check, 1, 64,
                               slot=self._set("shifts_per_check", int))
        form.addRow("checks per side", self.checks)
        form.addRow("shifts per check", self.shifts)
        self._toggle_lattice(config.swn_lattice_auto)
        layout.addWidget(stimulus)

        whitening = QtWidgets.QGroupBox("Whitening")
        form = QtWidgets.QFormLayout(whitening)
        note = QtWidgets.QLabel(
            "Shifting white noise is spatially correlated, so the raw STA is the receptive "
            "field blurred by the stimulus. On multi-unit data this often makes things worse "
            "rather than better; the log says so when it does.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {DIM}; font-size: 8pt;")
        form.addRow(note)
        form.addRow(self._check("whiten the spatial STA", "whiten"))
        self.sigma = spin_box(config.sigma_fraction * 100, 0.01, 2000, " %", 2, 0.5,
                              lambda v: setattr(config, "sigma_fraction", float(v) / 100))
        self.sigma.setToolTip("regularisation, as a percentage of the pixel variance")
        form.addRow("regularisation", self.sigma)
        layout.addWidget(whitening)

        cache_box = QtWidgets.QGroupBox("Cache")
        box = QtWidgets.QVBoxLayout(cache_box)
        self.cache_label = QtWidgets.QLabel()
        self.cache_label.setWordWrap(True)
        self.cache_label.setStyleSheet(f"color: {DIM}; font-size: 8pt;")
        box.addWidget(self.cache_label)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(button("Open folder", self._open_cache))
        row.addWidget(button("Forget this recording", self._forget_recording))
        row.addWidget(button("Forget everything", self._forget_all))
        box.addLayout(row)
        layout.addWidget(cache_box)
        self._refresh_cache()

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ helpers
    def _set(self, name: str, cast):
        return lambda value: setattr(self.config, name, cast(value))

    def _check(self, text: str, name: str) -> QtWidgets.QCheckBox:
        box = QtWidgets.QCheckBox(text)
        box.setChecked(bool(getattr(self.config, name)))
        box.toggled.connect(lambda on: setattr(self.config, name, bool(on)))
        return box

    def _toggle_lattice(self, auto: bool):
        for widget in (self.checks, self.shifts):
            widget.setEnabled(not auto)

    def _refresh_cache(self):
        try:
            total = self.cache.total_size()
        except OSError:
            total = 0
        self.cache_label.setText(f"{human_size(total)} in {ca.CACHE_ROOT}")

    def _open_cache(self):
        from PyQt6 import QtGui
        ca.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(ca.CACHE_ROOT)))

    def _forget_recording(self):
        self.cache.clear_recording()
        self._refresh_cache()

    def _forget_all(self):
        if QtWidgets.QMessageBox.question(
                self, "Forget everything",
                "Delete every cached step of every recording, rebuilt stimuli included?"
        ) is QtWidgets.QMessageBox.StandardButton.Yes:
            self.cache.clear_all()
            self._refresh_cache()


class TriggerPlot(QtWidgets.QDialog):
    """The trigger channel with the threshold drawn on it, to see that the pulses are there."""

    def __init__(self, trace: np.ndarray, fs: float, threshold: int,
                 triggers=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trigger channel")
        self.resize(820, 380)
        layout = QtWidgets.QVBoxLayout(self)

        plot = pg.PlotWidget()
        plot.setLabel("bottom", "time (s)")
        plot.setLabel("left", "ADC levels above zero")
        plot.showGrid(x=True, y=True, alpha=0.15)
        time, low, high = tg.envelope(trace, fs)
        plot.plot(time, high, pen=pg.mkPen("#6FA8DC", width=1))
        plot.addItem(pg.InfiniteLine(pos=threshold, angle=0,
                                     pen=pg.mkPen(WARN, width=2,
                                                  style=QtCore.Qt.PenStyle.DashLine)))
        plot.setYRange(float(min(low.min(), 0)), float(max(high.max(), threshold) * 1.1 + 1))
        layout.addWidget(plot)

        summary = f"threshold {threshold} levels ({tg.threshold_to_uv(threshold):,.0f} µV)"
        if triggers is not None:
            summary = f"{triggers.describe()} · {summary}"
        label = QtWidgets.QLabel(summary)
        label.setStyleSheet(f"color: {DIM};")
        layout.addWidget(label)


class FramePreview(QtWidgets.QDialog):
    """One stimulus frame as the retina saw it, to compare against the StimulusDisplayer."""

    def __init__(self, config: Config, frames: np.ndarray | None, parent=None):
        super().__init__(parent)
        self.config = config
        self.frames = frames
        self.setWindowTitle("Stimulus frame")
        self.resize(560, 620)
        layout = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        self.index = spin_box(0, 0, 10_000_000, slot=lambda _: self.draw())
        self.full = QtWidgets.QCheckBox("full resolution")
        self.full.setToolTip("the frame as the DMD showed it, to compare with the lab's "
                             "StimulusDisplayer; otherwise the grid the STA is computed on")
        self.full.toggled.connect(lambda _: self.draw())
        row.addWidget(QtWidgets.QLabel("frame"))
        row.addWidget(self.index)
        row.addWidget(self.full)
        row.addStretch(1)
        layout.addLayout(row)

        view = pg.GraphicsLayoutWidget()
        self.box = view.addViewBox(lockAspect=True, enableMenu=False)
        self.box.setMouseEnabled(x=False, y=False)
        self.image = pg.ImageItem()
        self.box.addItem(self.image)
        self.box.sigResized.connect(lambda _: self.box.autoRange(items=[self.image]))
        layout.addWidget(view, stretch=1)

        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {DIM};")
        layout.addWidget(self.note)
        self.draw()

    def draw(self):
        index = int(self.index.value())
        frame, note = None, ""
        if self.full.isChecked() and self.config.kind == st.SWN:
            try:
                from .binfile import BinFile
                with BinFile(self.config.swn_bin, self.config.mea) as library:
                    index = min(index, len(library) - 1)
                    frame = library.frame(index)
                note = (f"library frame {index}, full DMD resolution "
                        f"{frame.shape[1]}x{frame.shape[0]}")
            except (OSError, ValueError) as error:
                note = str(error)
        elif self.frames is not None:
            index = min(index, self.frames.shape[0] - 1)
            frame = self.frames[index].astype(np.float32) / 255.0
            note = (f"novel frame {index} of {self.frames.shape[0]}, grid "
                    f"{frame.shape[1]}x{frame.shape[0]}")
        else:
            note = "run the stimulus step, or tick full resolution to read the library directly"
        if frame is not None:
            self.image.setImage(frame, levels=(0.0, 1.0))
            self.box.autoRange(items=[self.image])
        self.note.setText(note + "  ·  shown as the retina saw it (+y up)")
