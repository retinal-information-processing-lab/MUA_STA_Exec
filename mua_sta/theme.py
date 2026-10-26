"""Colours and styles: the lab's dark theme, as Plerion and Oriole use it."""
from __future__ import annotations

from PyQt6 import QtGui, QtWidgets

BG, CARD, TEXT, DIM, BTN, ACCENT = "#121212", "#1E1E1E", "#E0E0E0", "#555555", "#2A2A2A", "#2D6A9F"
LOG_COLOURS = {"info": "#00FF00", "warn": "#FFD700", "error": "#FF4444"}

#: One colour per state of a step: done, has to be redone, not done yet.
STATUS_COLOURS = {"cached": "#00FF00", "stale": "#FF4444", "missing": DIM, "unknown": DIM}

SEGMENT_STYLE = """
    QPushButton { background: #1E1E1E; color: #999999; border: 1px solid #333;
                  border-radius: 0; padding: 6px 10px; }
    QPushButton:hover { background: #2A2A2A; }
    QPushButton:checked { background: #7A3E9D; color: #FFFFFF; border-color: #7A3E9D;
                          font-weight: bold; }
"""
ACTION_STYLE = """
    QPushButton { background: #1A4A1A; color: #00FF00; border: none; border-radius: 4px;
                  font: bold 11pt 'Consolas'; padding: 10px; }
    QPushButton:hover { background: #2A6A2A; }
    QPushButton:disabled { background: #3A3A3A; color: #666666; }
"""


def apply_dark_theme(app: QtWidgets.QApplication):
    """The lab's dark palette and style sheet."""
    role = QtGui.QPalette.ColorRole
    palette = QtGui.QPalette()
    for which, colour in ((role.Window, BG), (role.WindowText, TEXT), (role.Base, CARD),
                          (role.AlternateBase, BG), (role.ToolTipBase, CARD),
                          (role.ToolTipText, TEXT), (role.Text, TEXT), (role.Button, BTN),
                          (role.ButtonText, TEXT), (role.BrightText, "#FF4444"),
                          (role.Link, ACCENT), (role.Highlight, ACCENT),
                          (role.HighlightedText, "#FFFFFF")):
        palette.setColor(which, QtGui.QColor(colour))
    for which in (role.Text, role.ButtonText):
        palette.setColor(QtGui.QPalette.ColorGroup.Disabled, which, QtGui.QColor(DIM))
    app.setPalette(palette)
    app.setFont(QtGui.QFont("Segoe UI", 9))
    app.setStyleSheet("""
        QGroupBox { border: 1px solid #333; border-radius: 4px; margin-top: 10px;
                    padding: 12px 6px 6px 6px; font: bold 9pt 'Segoe UI'; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #E0E0E0; }
        QPushButton { border: none; border-radius: 3px; padding: 6px 14px;
                      background: #2A2A2A; color: #E0E0E0; }
        QPushButton:hover { background: #3A3A3A; }
        QPushButton:pressed { background: #2D6A9F; }
        QPushButton:disabled { color: #555555; }
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
            border: 1px solid #333; border-radius: 3px; padding: 4px;
            background: #1E1E1E; color: #E0E0E0; }
        QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { width: 0; border: none; }
        QComboBox QAbstractItemView { background: #1E1E1E; color: #E0E0E0;
                                      selection-background-color: #2D6A9F;
                                      selection-color: #FFFFFF; }
        QTabWidget::pane { border: 1px solid #2E2E2E; border-radius: 4px; }
        QTabBar::tab { background: #1A1A1A; color: #AAAAAA; padding: 6px 16px;
                       border-top-left-radius: 4px; border-top-right-radius: 4px; }
        QTabBar::tab:selected { background: #242424; color: #E0E0E0; }
        QProgressBar { border: 1px solid #333; border-radius: 3px; background: #1A1A1A;
                       text-align: center; color: #E0E0E0; }
        QProgressBar::chunk { background: #2D6A9F; }
        QToolTip { background: #1E1E1E; color: #E0E0E0; border: 1px solid #333; }
    """)
