"""The results window: every electrode's receptive field, and how the array sits in it.

Kept apart from the control window so that one can stay small. Everything here is about
looking at a finished analysis — nothing in it changes what was computed, so the orientation
controls never invalidate a cached step.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from . import sta as sta_mod
from . import stimulus as st
from .config import Config
from .rigs import ARRAY_REGISTRATIONS, rig, to_array_frame
from .theme import DIM
from .widgets import button, combo_box

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

#: Blue for negative, white for nothing, red for positive — as the lab's stitched figure.
STA_COLOURS = pg.ColorMap([0.0, 0.5, 1.0], [(50, 90, 255), (255, 255, 255), (255, 60, 60)])


class ResultsWindow(QtWidgets.QWidget):
    """The 16x16 grid of spatial STAs, and one electrode in detail when you click it."""

    def __init__(self, config: Config, mapping: dict[int, tuple[int, int]], log, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.config = config
        self.mapping = mapping
        self.log = log
        self.results = None
        self.grid = None
        self.selected: int | None = None
        self.setWindowTitle("MUA STA — results")
        self.resize(1380, 900)
        self._build()

    def _build(self):
        layout = QtWidgets.QHBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(left)
        self.summary = QtWidgets.QLabel("no analysis yet")
        self.summary.setStyleSheet(f"color: {DIM};")
        self.summary.setWordWrap(True)
        column.addWidget(self.summary)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("array orientation"))
        self.registration = combo_box(
            [("this rig's default", "")] + [(name, name) for name in ARRAY_REGISTRATIONS],
            self.config.array_registration, self._on_registration)
        self.registration.setToolTip(
            "how the stimulus axes sit on the electrode grid — a property of the setup, not of "
            "the stimulus file. Check orientation measures it from the data.")
        row.addWidget(self.registration)
        row.addWidget(button("Check orientation", self._on_check_orientation,
                             "measure it: an electrode's position has to predict where its "
                             "receptive field is"))
        self.crop = QtWidgets.QCheckBox("crop tiles to the fields")
        self.crop.setChecked(self.config.crop_tiles)
        self.crop.setToolTip("the stimulus covers much more of the retina than the array does, "
                             "so most of each tile is empty")
        self.crop.toggled.connect(self._on_crop)
        row.addWidget(self.crop)
        self.flip = QtWidgets.QCheckBox("row 0 at the top")
        self.flip.setChecked(self.config.flip_grid_y)
        self.flip.toggled.connect(self._on_flip)
        row.addWidget(self.flip)
        row.addStretch(1)
        column.addLayout(row)

        view = pg.GraphicsLayoutWidget()
        self.grid_box = view.addViewBox(lockAspect=True, enableMenu=False)
        self.grid_image = pg.ImageItem()
        self.grid_image.setLookupTable(STA_COLOURS.getLookupTable(nPts=256))
        self.grid_box.addItem(self.grid_image)
        view.scene().sigMouseClicked.connect(self._on_click)
        column.addWidget(view, stretch=1)
        splitter.addWidget(left)

        right = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(right)
        self.electrode = QtWidgets.QLabel("click an electrode on the grid")
        self.electrode.setStyleSheet(f"color: {DIM};")
        self.electrode.setWordWrap(True)
        column.addWidget(self.electrode)
        detail = pg.GraphicsLayoutWidget()
        self.cell_box = detail.addViewBox(lockAspect=True, enableMenu=False, row=0, col=0)
        self.cell_box.setMouseEnabled(x=False, y=False)
        self.cell_image = pg.ImageItem()
        self.cell_image.setLookupTable(STA_COLOURS.getLookupTable(nPts=256))
        self.cell_box.addItem(self.cell_image)
        self.cell_box.sigResized.connect(
            lambda _: self.cell_box.autoRange(items=[self.cell_image]))
        self.temporal_plot = detail.addPlot(row=1, col=0)
        self.temporal_plot.setLabel("bottom", "time before the spike (s)")
        self.temporal_plot.setLabel("left", "STA")
        self.temporal_plot.showGrid(x=True, y=True, alpha=0.15)
        self.temporal = self.temporal_plot.plot(pen=pg.mkPen("#FFD700", width=2))
        self.raster_plot = detail.addPlot(row=2, col=0)
        self.raster_plot.setLabel("bottom", "time in the repeated half (s)")
        self.raster_plot.setLabel("left", "sequence")
        self.raster = pg.PlotCurveItem(pen=pg.mkPen("#E0E0E0", width=1), connect="pairs")
        self.raster_plot.addItem(self.raster)
        detail.ci.layout.setRowStretchFactor(0, 3)
        detail.ci.layout.setRowStretchFactor(1, 2)
        detail.ci.layout.setRowStretchFactor(2, 3)
        column.addWidget(detail, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([780, 460])

    # ------------------------------------------------------------------- drawing
    def show_results(self, results):
        """Take a finished analysis and draw it; call again to redraw after a new run."""
        self.results = results
        self.draw()
        self.show()
        self.raise_()
        self.activateWindow()

    def draw(self):
        if self.results is None:
            return
        registered = to_array_frame(self.results.spatial, self.config.mea,
                                    self.config.registration)
        crop = None
        if self.config.crop_tiles:
            # measured on the unwhitened STA, where the fields are unambiguous
            crop = sta_mod.field_bounds(to_array_frame(
                self.results.spatial_raw, self.config.mea, self.config.registration))
        self.grid = sta_mod.stitch(registered, self.results.labels, self.mapping, crop=crop)
        image = np.nan_to_num(self.grid.image, nan=0.0)
        if self.config.flip_grid_y:
            image = image[::-1]
        self.grid_image.setImage(image, levels=(-1.0, 1.0))
        self.grid_box.autoRange(items=[self.grid_image])

        meta = self.results.meta
        whitened = " · whitened" if meta.get("whiten") else ""
        where = "row 0 at the top" if self.config.flip_grid_y else "row 0 at the bottom (+y up)"
        self.summary.setText(
            f"{self.results.labels.size} electrodes · {st.KIND_LABELS[meta['kind']]} · "
            f"{self.results.n_sequences} sequences · {meta['frequency']:.2f} Hz{whitened} · "
            f"orientation {self.config.registration} · "
            f"{self.grid.tile[0]}x{self.grid.tile[1]} cells per tile · {where}")
        if self.selected is not None:
            self.show_electrode(self.selected)

    def _on_click(self, event):
        if self.results is None or self.grid is None:
            return
        point = self.grid_box.mapSceneToView(event.scenePos())
        y = point.y()
        if self.config.flip_grid_y:
            y = self.grid.image.shape[0] - 1 - y
        label = self.grid.electrode_at(point.x(), y, self.mapping)
        if label is not None:
            self.selected = label
            self.show_electrode(label)

    def show_electrode(self, label: int):
        results = self.results
        index = results.index_of(label)
        if index is None:
            return
        row, col = self.mapping[label]
        lag = results.peak[index][0]
        frequency = results.meta["frequency"] or 1.0
        depth = results.sta_3d.shape[1]

        cell = to_array_frame(results.spatial[index], self.config.mea,
                              self.config.registration)
        self.cell_image.setImage(cell, levels=(-1.0, 1.0))
        self.cell_box.autoRange(items=[self.cell_image])
        self.temporal.setData((np.arange(depth) - (depth - 1)) / frequency,
                              results.temporal[index])
        self.temporal_plot.setYRange(-1.05, 1.05)

        trains = results.raster(label)
        sizes = np.array([train.size for train in trains])
        if sizes.sum():
            x = np.repeat(np.concatenate(trains), 2)
            base = np.repeat(np.arange(len(trains)), 2 * sizes)
            self.raster.setData(x, base + np.tile([0.05, 0.95], int(sizes.sum())))
        else:
            self.raster.setData([], [])
        self.raster_plot.setYRange(0, max(1, len(trains)))

        quality = sta_mod.rf_quality(results.spatial[index])
        self.electrode.setText(
            f"electrode {label} · grid row {row}, column {col} · peak at lag "
            f"{(lag - (depth - 1)) / frequency * 1000:.0f} ms · receptive-field strength "
            f"{quality:.0f} · {int(sizes.sum())} spikes in the repeated halves")

    # --------------------------------------------------------------- orientation
    def _on_registration(self, value):
        self.config.array_registration = str(value)
        self.config.save()
        self.draw()

    def _on_crop(self, on: bool):
        self.config.crop_tiles = bool(on)
        self.config.save()
        self.draw()

    def _on_flip(self, on: bool):
        self.config.flip_grid_y = bool(on)
        self.config.save()
        self.draw()

    def _on_check_orientation(self):
        """Measure how the array sits in the stimulus, from the receptive fields themselves."""
        if self.results is None:
            self.log("Run the analysis first: there are no receptive fields to measure", "warn")
            return
        # measured on the unwhitened STA: whitening is there to sharpen a field, not to find
        # one, and on noisy multi-unit data it scatters the peaks the measurement relies on
        scores = sta_mod.measure_registration(self.results.spatial_raw, self.results.labels,
                                              self.mapping)
        best, score = scores[0]
        self.log("Array orientation: "
                 + ", ".join(f"{name} {value:+.2f}" for name, value in scores[:4]), "info")
        if score < 0.5:
            self.log(f"No clear winner (best {best} at {score:+.2f}). Either there are too few "
                     "receptive fields, or the stimulus geometry is wrong.", "warn")
            return
        current = self.config.registration
        if best == current:
            self.log(f"{rig(self.config.mea).label}: {best} confirmed ({score:+.2f}, next "
                     f"{scores[1][1]:+.2f}) — the grid is drawn the right way round", "info")
        else:
            self.log(f"{rig(self.config.mea).label}: the data says {best} ({score:+.2f}), not "
                     f"{current}. Setting it for this session; put it in mua_sta/rigs.py to "
                     "keep it.", "warn")
            self.registration.setCurrentIndex(self.registration.findData(best))
