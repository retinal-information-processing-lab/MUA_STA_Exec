"""The results window: one figure, and a button to save it.

Deliberately plain. Every electrode's receptive field tiled onto the array on the left, the one
you click on the right. No settings: the stimulus axes are placed on the electrode grid by the
rig's own registration, the way its polarity and optics are, and the tiles are always cropped
to where the fields actually are — the stimulus covers far more retina than the array does, so
otherwise most of every tile is empty.

Drawn with matplotlib rather than pyqtgraph, as the standard pipeline draws it, which also
makes saving the figure the same operation as showing it.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt6 import QtCore, QtWidgets

from . import sta as sta_mod
from .config import Config
from .rigs import to_array_frame
from .widgets import button

#: Blue negative, white nothing, red positive — the colour map the pipeline uses for STAs.
CMAP = "RdBu_r"
TEXT = "#222222"


class ResultsWindow(QtWidgets.QWidget):
    """The array of receptive fields, and one electrode in detail when you click it."""

    def __init__(self, config: Config, mapping: dict[int, tuple[int, int]], log, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.config = config
        self.mapping = mapping
        self.log = log
        self.results = None
        self.grid = None
        self.selected: int | None = None
        self.setWindowTitle("MUA STA — results")
        self.resize(1180, 740)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 6)
        self.figure = Figure(figsize=(11.8, 7.0), facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        layout.addWidget(self.canvas, stretch=1)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(8, 0, 8, 0)
        bar.addStretch(1)
        bar.addWidget(button("Save image", self._on_save))
        layout.addLayout(bar)

        grid = self.figure.add_gridspec(3, 2, width_ratios=[1.75, 1.0],
                                        height_ratios=[3.0, 2.0, 2.0],
                                        left=0.015, right=0.975, top=0.975, bottom=0.07,
                                        wspace=0.10, hspace=0.45)
        self.ax_grid = self.figure.add_subplot(grid[:, 0])
        self.ax_cell = self.figure.add_subplot(grid[0, 1])
        self.ax_time = self.figure.add_subplot(grid[1, 1])
        self.ax_raster = self.figure.add_subplot(grid[2, 1])
        for ax in (self.ax_grid, self.ax_cell):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        for ax in (self.ax_time, self.ax_raster):
            ax.tick_params(labelsize=8, colors=TEXT)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        self.highlight: Rectangle | None = None

    # ------------------------------------------------------------------- drawing
    def show_results(self, results):
        self.results = results
        self.selected = None
        self.draw()
        self.show()
        self.raise_()
        self.activateWindow()

    def draw(self):
        if self.results is None:
            return
        spatial = to_array_frame(self.results.spatial, self.config.mea)
        # the window to crop to is taken from the unwhitened STA, where the fields are plain
        crop = sta_mod.field_bounds(to_array_frame(self.results.spatial_raw, self.config.mea))
        self.grid = sta_mod.stitch(spatial, self.results.labels, self.mapping, crop=crop)

        self.ax_grid.clear()
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        for spine in self.ax_grid.spines.values():
            spine.set_visible(False)
        self.ax_grid.imshow(np.ma.masked_invalid(self.grid.image), cmap=CMAP, vmin=-1, vmax=1,
                            origin="lower", interpolation="nearest")
        self.highlight = None
        if self.selected is not None:
            self.show_electrode(self.selected, redraw=False)
        else:
            for ax in (self.ax_cell, self.ax_time, self.ax_raster):
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
        self.canvas.draw_idle()

    def _on_click(self, event):
        if self.results is None or self.grid is None or event.inaxes is not self.ax_grid:
            return
        if event.xdata is None or event.ydata is None:
            return
        label = self.grid.electrode_at(event.xdata, event.ydata, self.mapping)
        if label is not None:
            self.selected = label
            self.show_electrode(label)

    def show_electrode(self, label: int, redraw: bool = True):
        results = self.results
        index = results.index_of(label)
        if index is None:
            return
        self.selected = label
        rows, cols = self.grid.tile
        row, col = self.mapping[label]
        if self.highlight is not None:
            self.highlight.remove()
        self.highlight = Rectangle((col * (cols + self.grid.padding) - 0.5,
                                    row * (rows + self.grid.padding) - 0.5), cols, rows,
                                   fill=False, edgecolor="#111111", linewidth=1.4)
        self.ax_grid.add_patch(self.highlight)

        depth = results.sta_3d.shape[1]
        frequency = results.meta["frequency"] or 1.0

        self.ax_cell.clear()
        self.ax_cell.set_xticks([])
        self.ax_cell.set_yticks([])
        for spine in self.ax_cell.spines.values():
            spine.set_visible(False)
        cell = to_array_frame(results.spatial[index], self.config.mea)
        if crop_box := sta_mod.field_bounds(
                to_array_frame(results.spatial_raw, self.config.mea)):
            r0, r1, c0, c1 = crop_box
            cell = cell[r0:r1, c0:c1]
        self.ax_cell.imshow(cell, cmap=CMAP, vmin=-1, vmax=1, origin="lower",
                            interpolation="nearest")

        # the pipeline's time vector: one bin per frame, ending at the spike
        time = np.flip(np.arange(0, -depth / frequency, -1 / frequency))[:depth]
        self.ax_time.clear()
        self.ax_time.axhline(0, color="#BBBBBB", linewidth=0.8)
        self.ax_time.plot(time, results.temporal[index], color="#D1495B", linewidth=1.6)
        self.ax_time.set_xlim(time[0], 0)
        self.ax_time.set_ylim(-1.05, 1.05)
        self.ax_time.set_xlabel("Time (s)", fontsize=8, color=TEXT)
        self.ax_time.tick_params(labelsize=8, colors=TEXT)
        for spine in ("top", "right"):
            self.ax_time.spines[spine].set_visible(False)

        trains = results.raster(label)
        self.ax_raster.clear()
        if any(train.size for train in trains):
            self.ax_raster.eventplot(trains, colors="#222222", linewidths=0.4)
        self.ax_raster.set_xlim(0, max(1e-3, max((t.max() for t in trains if t.size),
                                                 default=1.0)))
        self.ax_raster.set_ylim(-0.5, max(1, len(trains)) - 0.5)
        self.ax_raster.set_xlabel("Time (s)", fontsize=8, color=TEXT)
        self.ax_raster.tick_params(labelsize=8, colors=TEXT)
        for spine in ("top", "right"):
            self.ax_raster.spines[spine].set_visible(False)

        if redraw:
            self.canvas.draw_idle()

    # ---------------------------------------------------------------------- saving
    def _on_save(self):
        if self.results is None:
            return
        start = str(self.config.last_dir or "")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the figure", f"{start}/mua_sta.png" if start else "mua_sta.png",
            "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return
        self.figure.savefig(path, dpi=150, facecolor="white")
        self.log(f"Saved {path}")
