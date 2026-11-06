"""The window: pick a stimulus, a setup and a recording, press RUN.

Small on purpose. This is used at the bench while a retina is alive, so it shows only what
changes between experiments. Everything that does not — 256 uint16 channels at 20 kHz, the
visual trigger channel, each rig's trigger threshold, the spike-detection band — lives in the
code or in the per-rig table, and the few parameters that are occasionally worth touching are
behind the gear.

The three lines above RUN are the run itself, split where the work is: everything about the
file, everything about the stimulus, then the STA. Each has a dot (green done, red has to be
redone, grey not done yet), a play button that does just that part, and an erase button. The
six cached steps underneath are unchanged — collapsing them into three is only presentation,
and it is what lets a changed setup re-detect the triggers without reading the recording again.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from . import cache as ca
from . import recording as rec
from . import stimulus as st
from .config import Config
from .dialogs import FramePreview, SettingsDialog, TriggerPlot
from .pipeline import GROUP_LABELS, GROUPS, Pipeline
from .results import ResultsWindow
from .rigs import RIGS, rig
from .theme import ACTION_STYLE, DIM, STOP_STYLE, apply_dark_theme
from .widgets import (FileRow, StatusLine, StepRow, Worker, action_button, button, combo_box,
                      segmented, spin_box)

GROUP_EXTRAS = {
    "recording": ("▤", "plot the trigger channel"),
    "stimulus": ("▣", "look at one stimulus frame"),
    "sta": ("⊞", "open the results"),
}


class Window(QtWidgets.QMainWindow):
    """One recording at a time: what it is, what was shown, and the three steps of the run."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.pipeline = Pipeline(config, log=self._log)
        self.worker: Worker | None = None
        self._show_when_done = False
        self.results_window: ResultsWindow | None = None
        self._loading = True
        self._mapping = rec.load_mapping()

        self.setWindowTitle("MUA STA")
        self._build()
        self._loading = False
        if config.load_error:
            self._log(config.load_error, "warn")
        self._refresh()

    # --------------------------------------------------------------------- layout
    def _build(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 10)

        self.switch, self.kind_buttons = segmented(
            [(st.KIND_LABELS[st.CHECKERBOARD], st.CHECKERBOARD),
             (st.KIND_LABELS[st.SWN], st.SWN)], self.config.kind, self._on_kind)
        root.addWidget(self.switch)

        self.pages = QtWidgets.QStackedWidget()
        self.pages.addWidget(self._checkerboard_page())
        self.pages.addWidget(self._swn_page())
        self.pages.setCurrentIndex(0 if self.config.kind == st.CHECKERBOARD else 1)
        root.addWidget(self.pages)
        root.addWidget(self._rule())

        setup = QtWidgets.QHBoxLayout()
        setup.addWidget(QtWidgets.QLabel("Setup"))
        self.setup_combo = combo_box(
            [(settings.label, mea) for mea, settings in sorted(RIGS.items())],
            self.config.mea, self._on_mea)
        setup.addWidget(self.setup_combo, stretch=1)
        setup.addSpacing(8)
        setup.addWidget(QtWidgets.QLabel("trigger channel"))
        self.trigger_channel = spin_box(self.config.trigger_channel, 0, 1023,
                                        slot=self._on_trigger_channel)
        self.trigger_channel.setToolTip("0-based index of the visual trigger: MCS label 127 is "
                                        "index 126, as in the standard pipeline")
        self.trigger_channel.setFixedWidth(60)
        setup.addWidget(self.trigger_channel)
        root.addLayout(setup)
        root.addWidget(self._rule())

        recording = QtWidgets.QHBoxLayout()
        recording.addWidget(QtWidgets.QLabel("Recording"))
        self.recording_row = FileRow(self.config.recording_path, "Select a recording",
                                     "Recordings (*.mcd *.raw);;MCD recordings (*.mcd);;"
                                     "Raw recordings (*.raw);;All files (*)")
        self.recording_row.changed.connect(self._on_recording)
        recording.addWidget(self.recording_row, stretch=1)
        root.addLayout(recording)
        self.recording_label = QtWidgets.QLabel()
        self.recording_label.setStyleSheet(f"color: {DIM}; font-size: 8pt;")
        self.recording_label.setWordWrap(True)
        root.addWidget(self.recording_label)
        root.addWidget(self._rule())

        self.rows: dict[str, StepRow] = {}
        for group in GROUPS:
            icon, tip = GROUP_EXTRAS[group]
            row = StepRow(group, GROUP_LABELS[group], icon, tip)
            row.run_requested.connect(self._on_run_group)
            row.drop_requested.connect(self._on_drop_group)
            row.extra_requested.connect(self._on_extra)
            self.rows[group] = row
            root.addWidget(row)
        root.addWidget(self._rule())

        bottom = QtWidgets.QHBoxLayout()
        self.run_button = action_button("RUN", self._on_run)
        bottom.addWidget(self.run_button, stretch=1)
        gear = button("⚙", self._on_settings, "settings")
        gear.setFixedSize(44, 44)
        gear.setStyleSheet("font-size: 15pt; padding: 0;")
        bottom.addWidget(gear)
        root.addLayout(bottom)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.status = StatusLine()
        root.addWidget(self.status)
        self.resize(460, self.sizeHint().height())

    def _rule(self) -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color: #2A2A2A;")
        return line

    def _checkerboard_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setContentsMargins(0, 4, 0, 0)
        self.source_row = FileRow(self.config.checkerboard_source, "Select the binary source")
        self.source_row.changed.connect(self._on_source)
        self.checks = spin_box(self.config.n_checks, 1, 1024, slot=self._on_checks)
        self.checks.setFixedWidth(70)
        form.addRow("binary source", self.source_row)
        form.addRow("squares per side", self.checks)
        return page

    def _swn_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setContentsMargins(0, 4, 0, 0)
        self.bin_row = FileRow(self.config.swn_bin, "Select the bin file",
                               "Bin files (*.bin);;All files (*)")
        self.bin_row.changed.connect(self._on_swn_bin)
        self.vec_row = FileRow(self.config.swn_vec, "Select the vec file",
                               "Vec files (*.vec);;All files (*)")
        self.vec_row.changed.connect(self._on_swn_vec)
        form.addRow("Vec file", self.vec_row)
        form.addRow("Bin file", self.bin_row)
        return page

    # -------------------------------------------------------------------- messages
    def _log(self, message: str, level: str = "info"):
        """One line on screen, the whole thing to ~/.mua_sta/mua_sta.log."""
        print(f"{datetime.now():%H:%M:%S} [{level}] {message}", flush=True)
        if hasattr(self, "status"):
            self.status.say(str(message), level)

    # --------------------------------------------------------------- config edits
    def _changed(self, reset: bool = False):
        if self._loading:
            return
        self.config.save()
        if reset:
            self.pipeline.reset()
        self._refresh()

    def _on_kind(self, kind: str):
        self.config.kind = kind
        self.kind_buttons[kind].setChecked(True)
        self.pages.setCurrentIndex(0 if kind == st.CHECKERBOARD else 1)
        self._changed()

    def _on_mea(self, mea: int):
        self.config.mea = mea
        settings = rig(mea)
        self.config.trigger_threshold = None            # back to this rig's threshold
        self._log(f"{settings.label}: trigger threshold {self.config.threshold} levels, "
                  f"{'inverted' if settings.invert_polarity else 'normal'} polarity",
                  "info" if settings.display_ready else "warn")
        self._changed()

    def _on_trigger_channel(self, value):
        self.config.trigger_channel = int(value)
        self._changed(reset=True)

    def _on_recording(self, path: str):
        self.config.recording_path = path
        if path:
            self.config.last_dir = str(Path(path).parent)
        self._changed(reset=True)

    def _on_source(self, path: str):
        self.config.checkerboard_source = path
        self._changed()

    def _on_checks(self, value):
        self.config.n_checks = int(value)
        self._changed()

    def _on_swn_vec(self, path: str):
        self.config.swn_vec = path
        self._changed()

    def _on_swn_bin(self, path: str):
        self.config.swn_bin = path
        self._changed()

    # ------------------------------------------------------------------- the run
    def _busy(self, busy: bool):
        """RUN turns into STOP while something runs; the per-line buttons wait their turn."""
        self.progress.setVisible(busy)
        self.run_button.setText("STOP" if busy else "RUN")
        self.run_button.setStyleSheet(STOP_STYLE if busy else ACTION_STYLE)
        self.run_button.setEnabled(True)
        for row in self.rows.values():
            row.run.setEnabled(not busy)
            row.drop.setEnabled(not busy)
        if not busy:
            self.progress.setRange(0, 0)

    def _start(self, work, what: str, then_show: bool = False):
        """Run ``work`` off the GUI thread; ``then_show`` opens the results when it finishes."""
        if self._running():
            self._log("Something is already running", "warn")
            return
        self._show_when_done = then_show
        self._busy(True)
        self.status.say(what)
        self.worker = Worker(work)
        self.worker.progressed.connect(self._on_progress)
        self.worker.logged.connect(self._log)
        self.worker.finished_ok.connect(self._finished)
        self.worker.cancelled.connect(self._stopped)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def _on_progress(self, text: str, fraction: float):
        """Fill the bar when the step can say how far along it is, spin it when it cannot."""
        if fraction < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(fraction * 1000))
        self.status.say(text)

    def _on_run(self):
        """Run everything — asking first if any of it has been done already. While something
        is running this same button stops it."""
        if self._running():
            self.worker.stop()
            self.status.say("stopping at the end of the current step…", "warn")
            self.run_button.setEnabled(False)
            return
        if not self.pipeline.anything_done():
            self._start(lambda progress: self.pipeline.run_all(progress), "running", True)
        elif self._confirm_redo():
            self._start(lambda progress: self.pipeline.run_all(progress, force=True),
                        "redoing everything", True)

    def _confirm_redo(self) -> bool:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Run")
        box.setText("Some steps are already done.")
        box.setInformativeText("Redo everything from the start?\n\nTo redo just one step, "
                               "close this and use its ▶ button.")
        redo = box.addButton("Redo everything", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.exec()
        return box.clickedButton() is redo

    def _on_run_group(self, group: str):
        self._start(lambda progress: self.pipeline.run_group(group, progress),
                    f"running {GROUP_LABELS[group].lower()}", then_show=group == "sta")

    def _on_drop_group(self, group: str):
        self.pipeline.drop_group(group)
        self._log(f"{GROUP_LABELS[group]}: forgotten")
        self._refresh()

    def _finished(self):
        self._busy(False)
        self._refresh()
        # only what computed an STA opens the results; running the triggers again should not
        if self._show_when_done and self.pipeline.results is not None:
            self._show_results()

    def _stopped(self):
        self._busy(False)
        self._log("Stopped. What had finished is kept; the step that was running is not.",
                  "warn")
        self._refresh()

    def _failed(self, message: str):
        self._busy(False)
        self._log(message, "error")
        self._refresh()

    # ------------------------------------------------------------- the extra views
    def _on_extra(self, group: str):
        if group == "recording":
            self._show_trigger_plot()
        elif group == "stimulus":
            if self.pipeline.stimulus is None:
                self.pipeline.load_cached(ca.STIMULUS)
            FramePreview(self.config, self.pipeline.stimulus, self).exec()
        else:
            self._show_results()

    def _show_trigger_plot(self):
        """Show the trace that has been read or cached, never read it: that is the step's job."""
        if self.pipeline.trace is None:
            self.pipeline.load_cached(ca.TRACE)
        if self.pipeline.triggers is None:
            self.pipeline.load_cached(ca.TRIGGERS)
        if self.pipeline.trace is None:
            self._log("Run Triggers and spikes first: the trigger channel has not been read",
                      "warn")
            return
        TriggerPlot(self.pipeline.trace, self.config.fs, self.config.threshold,
                    self.pipeline.triggers, self).exec()

    def _show_results(self):
        if self.pipeline.results is None:
            self.pipeline.load_cached(ca.ANALYSIS)
        if self.pipeline.results is None:
            self._log("Nothing to show yet: run the STA step", "warn")
            return
        if self.results_window is None:
            self.results_window = ResultsWindow(self.config, self._mapping, self._log, self)
        self.results_window.show_results(self.pipeline.results)

    def _on_settings(self):
        SettingsDialog(self.config, self.pipeline.cache, self).exec()
        self.config.save()
        self._refresh()

    # -------------------------------------------------------------------- refresh
    def _refresh(self):
        self._refresh_recording_label()
        for group, row in self.rows.items():
            status = self.pipeline.group_status(group)
            row.set_state(str(status), self._group_tip(group))

    def _group_tip(self, group: str) -> str:
        parts = []
        for step in self.pipeline.group_steps(group):
            status = self.pipeline.status(step)
            parts.append(f"{step}: {status if status else 'not known yet'}")
        return "\n".join(parts)

    def _refresh_recording_label(self):
        if not self.config.recording_path:
            self.recording_label.setText("no recording chosen")
            return
        try:
            info = rec.open_info(self.config.recording_path, self.config.n_channels,
                                 self.config.dtype, self.config.fs)
        except (OSError, ValueError) as error:
            self.recording_label.setText(str(error))
            return
        self.recording_label.setText(info.describe())

    # -------------------------------------------------------------------- closing
    def closeEvent(self, event):
        self.config.save()
        if self.results_window is not None:
            self.results_window.close()
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
