"""Small widgets the window is built from: cards, file rows, the log, the step rows."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6 import QtCore, QtGui, QtWidgets

from .theme import ACTION_STYLE, DIM, LOG_COLOURS, SEGMENT_STYLE, STATUS_COLOURS


# ----------------------------------------------------------------------- factories
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


def segmented(options: list[tuple[str, object]], current, slot) -> tuple[QtWidgets.QWidget, dict]:
    """A two-or-more position switch: one button per option, only one pressed at a time.

    Returns the widget and the buttons by value, so a caller can set the position itself.
    """
    bar = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(bar)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)
    group = QtWidgets.QButtonGroup(bar)
    group.setExclusive(True)
    buttons = {}
    for index, (label, value) in enumerate(options):
        btn = QtWidgets.QPushButton(label)
        btn.setCheckable(True)
        btn.setChecked(value == current)
        btn.setMinimumHeight(30)
        edge = ("border-top-left-radius: 4px; border-bottom-left-radius: 4px;" if index == 0
                else "border-top-right-radius: 4px; border-bottom-right-radius: 4px;"
                if index == len(options) - 1 else "")
        btn.setStyleSheet(SEGMENT_STYLE + f"QPushButton {{ {edge} }}")
        btn.clicked.connect(lambda _=False, v=value: slot(v))
        group.addButton(btn)
        row.addWidget(btn, stretch=1)
        buttons[value] = btn
    return bar, buttons


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
    """One line of the run: a status dot, its name, play, erase, and one look-at-it button."""

    run_requested = QtCore.pyqtSignal(str)
    drop_requested = QtCore.pyqtSignal(str)
    extra_requested = QtCore.pyqtSignal(str)

    def __init__(self, step: str, label: str, extra: str = "", extra_tip: str = "", parent=None):
        super().__init__(parent)
        self.step = step
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 1, 0, 1)
        row.setSpacing(4)
        self.dot = StatusDot()
        self.name = QtWidgets.QLabel(label)
        self.run = button("\u25b6", lambda: self.run_requested.emit(self.step),
                          "run this step (again, if it is already done)")
        self.drop = button("\u2715", lambda: self.drop_requested.emit(self.step),
                           "forget what this step computed")
        self.extra = button(extra, lambda: self.extra_requested.emit(self.step), extra_tip)
        self.extra.setVisible(bool(extra))
        for btn in (self.run, self.drop, self.extra):
            btn.setFixedWidth(26)
            btn.setStyleSheet("padding: 3px 0;")
        row.addWidget(self.dot)
        row.addWidget(self.name, stretch=1)
        for btn in (self.run, self.drop, self.extra):
            row.addWidget(btn)

    def set_state(self, status: str, tip: str = ""):
        self.dot.set_status(status)
        self.dot.setToolTip(tip)
        self.name.setToolTip(tip)
        self.drop.setEnabled(status in ("cached", "stale"))


class StatusLine(QtWidgets.QLabel):
    """The one line of feedback the window has room for: the last thing that happened."""

    def __init__(self, parent=None):
        super().__init__("", parent)
        self.setWordWrap(True)
        self.setMinimumHeight(32)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.say("")

    def say(self, message: str, level: str = "info"):
        colour = DIM if level == "info" else LOG_COLOURS.get(level, DIM)
        self.setStyleSheet(f"color: {colour}; font-size: 8pt;")
        self.setText(message)


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
