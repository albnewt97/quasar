"""Smoke tests for QuasarMainWindow (§11 prompt 11).

Run headless via ``QT_QPA_PLATFORM=offscreen`` (see conftest.py qapp_env).
"""
from __future__ import annotations

from pytestqt.qtbot import QtBot

from qndt.gui.main_window import QuasarMainWindow


def test_main_window_creates(qtbot: QtBot) -> None:
    """QuasarMainWindow constructs without raising."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    assert window is not None


def test_main_window_title(qtbot: QtBot) -> None:
    """Window title identifies the application."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == "Quasar — Quantum Network Digital Twin"


def test_status_idle_on_create(qtbot: QtBot) -> None:
    """Status label reads 'Idle' immediately after construction."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    assert window._status_label.text() == "Idle"


def test_set_status_running(qtbot: QtBot) -> None:
    """set_simulation_status() updates the status label text."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    window.set_simulation_status("Running")
    assert window._status_label.text() == "Running"


def test_menu_bar_exists(qtbot: QtBot) -> None:
    """Menu bar exposes File, Simulation, View, and Help top-level menus."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    menu_titles = [action.text() for action in window.menuBar().actions()]
    assert "&File" in menu_titles
    assert "&Simulation" in menu_titles
    assert "&View" in menu_titles
    assert "&Help" in menu_titles


def test_dock_widgets_present(qtbot: QtBot) -> None:
    """The parameter, dashboard, and telemetry docks are all created."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    assert window._parameter_dock.windowTitle() == "Parameters"
    assert window._dashboard_dock.windowTitle() == "Dashboard"
    assert window._telemetry_dock.windowTitle() == "Telemetry Viewer"


def test_update_clock(qtbot: QtBot) -> None:
    """update_clock() formats the simulation time into the clock label."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    window.update_clock(1.5)
    assert window._clock_label.text() == "t = 1.500 s"
