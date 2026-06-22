"""QApplication entry point (§7.1, §11 prompt 11).

This module and everything under ``qndt.gui`` is the only place in the
codebase permitted to import PySide6 (§3.6 GUI Isolation Law).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from qndt.gui.main_window import QuasarMainWindow
from qndt.gui.simulation_controller import SimulationController

_STYLE_PATH = Path(__file__).parent / "styles" / "dark.qss"


def load_stylesheet(app: QApplication) -> None:
    """Load and apply the dark scientific-instrument QSS theme.

    Args:
        app: The application instance to style.
    """
    app.setStyleSheet(_STYLE_PATH.read_text())


def main() -> None:
    """Construct the QApplication, apply the theme, and show the main window."""
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Quasar")
    app.setOrganizationName("Quasar")
    load_stylesheet(app)

    window = QuasarMainWindow()
    SimulationController(window, window.dashboard, window._topology_model, parent=window)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
