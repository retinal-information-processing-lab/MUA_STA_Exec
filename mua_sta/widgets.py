"""Small widgets the window is built from: cards, file rows, the log, the step rows."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6 import QtCore, QtGui, QtWidgets

from .theme import ACTION_STYLE, DIM, LOG_COLOURS, PANEL_BG, STATUS_COLOURS


# ----------------------------------------------------------------------- factories
def title_label(text: str, colour: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setFont(QtGui.QFont("Consolas", 12, QtGui.QFont.Weight.Bold))
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet(f"background: {colour}; color: white; padding: 6px; border-radius: 4px;")
    return label


def card(layout: QtWidgets.QLayout, name: str, colour: str, width: int | None = None):
    """One part of the window: a framed card with a coloured top edge."""
    frame = QtWidgets.QFrame()
    frame.setObjectName(name)
    frame.setStyleSheet(f"QFrame#{name} {{ background: {PANEL_BG}; border: 1px solid #2E2E2E;"
                        f" border-top: 3px solid {colour}; border-radius: 6px; }}")
    frame.setLayout(layout)
    if width:
        frame.setFixedWidth(width)
    return frame


def button(text: str, slot, tip: str = "", style: str = "") -> QtWidgets.QPushButton:
    btn = QtWidgets.QPushButton(text)
    btn.clicked.connect(slot)
    btn.setToolTip(tip)
    if style:
        btn.setStyleSheet(style)
    return btn


def action_button(text: str, slot) -> QtWidgets.QPushButton:
    btn = button(text, slot, style=ACTION_STYLE)
    btn.setMinimumHeight(44)
    return btn


def spin_box(value, lo, hi, suffix: str = "", decimals: int | None = None,
             step: float | None = None, slot=None) -> QtWidgets.QAbstractSpinBox:
    """A QSpinBox, or a QDoubleSpinBox with ``decimals``; ``slot`` gets every new value."""
    box = QtWidgets.QSpinBox() if decimals is None else QtWidgets.QDoubleSpinBox()
    box.setRange(lo, hi)
    if decimals is not None:
        box.setDecimals(decimals)
    if step:
        box.setSingleStep(step)
    box.setSuffix(suffix)
    box.setValue(float(value) if decimals is not None else int(round(value)))
    box.setKeyboardTracking(False)      # one change per edit, not one per keystroke
    if slot is not None:
        box.valueChanged.connect(slot)
    return box


def combo_box(items: list[tuple[str, object]], current, slot=None) -> QtWidgets.QComboBox:
    box = QtWidgets.QComboBox()
    for label, value in items:
        box.addItem(label, value)
    index = box.findData(current)
    box.setCurrentIndex(max(0, index))
    if slot is not None:
        box.currentIndexChanged.connect(lambda _: slot(box.currentData()))
    return box


# ------------------------------------------------------------------------- widgets
class FileRow(QtWidgets.QWidget):
    """A file path with a browse button; ``changed`` carries the new path."""

    changed = QtCore.pyqtSignal(str)

    def __init__(self, value: str, caption: str, pattern: str = "All files (*)", parent=None):
        super().__init__(parent)
        self.caption, self.pattern = caption, pattern
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QtWidgets.QLineEdit(value)
        self.edit.setToolTip(value)
        self.edit.editingFinished.connect(lambda: self.set_path(self.edit.text()))
        browse = button("…", self._browse, style="padding: 4px 0;")
        browse.setFixedWidth(30)
        row.addWidget(self.edit, stretch=1)
        row.addWidget(browse)

    def _browse(self):
        start = self.edit.text() or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.caption, start, self.pattern)
        if path:
            self.set_path(path)

    def set_path(self, path: str, notify: bool = True):
        path = path.strip()
        if path == self.edit.text() and not notify:
            return
        self.edit.setText(path)
        self.edit.setToolTip(path)
        if notify:
            self.changed.emit(path)

    def path(self) -> str:
        return self.edit.text().strip()


class ConsoleLog(QtWidgets.QPlainTextEdit):
    """The shared log: black background, green monospace, one colour per level."""

    appended = QtCore.pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)
        self.setFont(QtGui.QFont("Consolas", 9))
        self.setStyleSheet("background: #0A0A0A; color: #00FF00; border: none;")
        self.appended.connect(self._append)       # so worker threads can log safely

    def log(self, message: str, level: str = "info"):
        self.appended.emit(str(message), level)

    def _append(self, message: str, level: str):
        colour = LOG_COLOURS.get(level, LOG_COLOURS["info"])
        stamp = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self.appendHtml(f'<span style="color:{DIM}">{stamp}</span> '
                        f'<span style="color:{colour}">{message}</span>')
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class StatusDot(QtWidgets.QWidget):
    """A small filled circle saying whether a step is cached, stale or not done."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._colour = STATUS_COLOURS["unknown"]
        self.setFixedSize(14, 14)

    def set_status(self, status: str):
        self._colour = STATUS_COLOURS.get(status, STATUS_COLOURS["unknown"])
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor("#000000"), 1))
        painter.setBrush(QtGui.QColor(self._colour))
        painter.drawEllipse(2, 2, 10, 10)
        painter.end()


class StepRow(QtWidgets.QWidget):
    """One step of the pipeline: status dot, name, note, run-just-this and drop-cache buttons."""

    run_requested = QtCore.pyqtSignal(str)
    drop_requested = QtCore.pyqtSignal(str)

    def __init__(self, step: str, label: str, parent=None):
        super().__init__(parent)
        self.step = step
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(6)
        self.dot = StatusDot()
        self.name = QtWidgets.QLabel(label)
        self.name.setMinimumWidth(130)
        self.note = QtWidgets.QLabel("—")
        self.note.setStyleSheet(f"color: {DIM};")
        self.note.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                               | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.run = button("▶", lambda: self.run_requested.emit(self.step), "run this step now")
        self.drop = button("✕", lambda: self.drop_requested.emit(self.step),
                           "forget this step's cached result")
        for btn in (self.run, self.drop):
            btn.setFixedWidth(28)
            btn.setStyleSheet("padding: 2px 0;")
        row.addWidget(self.dot)
        row.addWidget(self.name)
        row.addWidget(self.note, stretch=1)
        row.addWidget(self.run)
        row.addWidget(self.drop)

    def set_state(self, status: str | None, note: str):
        self.dot.set_status(status or "unknown")
        self.note.setText(note)
        self.drop.setEnabled(status in ("cached", "stale"))


def human_size(n_bytes: int) -> str:
    if n_bytes <= 0:
        return "—"
    for unit in ("B", "kB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.0f} {unit}" if unit == "B" else f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} GB"


class Worker(QtCore.QThread):
    """Runs one callable off the GUI thread, reporting progress, log lines and the outcome."""

    progressed = QtCore.pyqtSignal(str)
    logged = QtCore.pyqtSignal(str, str)
    finished_ok = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)

    def __init__(self, work: Callable[[Callable[[str], None]], None], parent=None):
        super().__init__(parent)
        self._work = work

    def run(self):
        try:
            self._work(self.progressed.emit)
        except Exception as error:                      # shown in the log, never a traceback popup
            self.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.finished_ok.emit()
