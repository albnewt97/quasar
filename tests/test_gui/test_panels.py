"""Tests for parameter panels (§7.3).

Run headless via ``QT_QPA_PLATFORM=offscreen`` (see conftest.py qapp_env).
"""
from __future__ import annotations

from pytestqt.qtbot import QtBot

from qndt.gui.panels.aging_panel import AgingPanel
from qndt.gui.panels.channel_panel import ChannelPanel
from qndt.gui.panels.coexistence_panel import CoexistencePanel
from qndt.gui.panels.control_plane_panel import ControlPlanePanel
from qndt.gui.panels.telemetry_panel import TelemetryPanel


def test_channel_panel_creates(qtbot: QtBot) -> None:
    """ChannelPanel() creates without error."""
    panel = ChannelPanel()
    qtbot.addWidget(panel)


def test_channel_panel_get_config_defaults(qtbot: QtBot) -> None:
    """get_config() returns the default quantum wavelength."""
    panel = ChannelPanel()
    qtbot.addWidget(panel)
    assert panel.get_config()["lambda_q_nm"] == 1550.0


def test_channel_panel_emits_on_apply(qtbot: QtBot) -> None:
    """Clicking Apply emits config_changed with engine_id='channel'."""
    panel = ChannelPanel()
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.config_changed, timeout=1000) as blocker:
        panel._apply_button.click()
    assert blocker.args[0] == "channel"


def test_coexistence_panel_add_channel(qtbot: QtBot) -> None:
    """Adding a channel grows the table by one row."""
    panel = CoexistencePanel()
    qtbot.addWidget(panel)
    rows_before = panel._table.rowCount()
    panel.add_channel()
    assert panel._table.rowCount() == rows_before + 1


def test_coexistence_panel_remove_channel(qtbot: QtBot) -> None:
    """Adding then removing a channel returns the table to zero rows."""
    panel = CoexistencePanel()
    qtbot.addWidget(panel)
    panel._table.setRowCount(0)
    panel.add_channel()
    panel._table.selectRow(0)
    panel.remove_selected()
    assert panel._table.rowCount() == 0


def test_telemetry_panel_creates(qtbot: QtBot) -> None:
    """TelemetryPanel() creates without error."""
    panel = TelemetryPanel()
    qtbot.addWidget(panel)


def test_telemetry_panel_sensitivity_matrix(qtbot: QtBot) -> None:
    """get_config() returns a 3x3 sensitivity matrix."""
    panel = TelemetryPanel()
    qtbot.addWidget(panel)
    sensitivity = panel.get_config()["sensitivity"]
    assert len(sensitivity) == 3
    assert all(len(row) == 3 for row in sensitivity)


def test_aging_panel_creates(qtbot: QtBot) -> None:
    """AgingPanel() creates without error."""
    panel = AgingPanel()
    qtbot.addWidget(panel)


def test_aging_panel_update_node_status(qtbot: QtBot) -> None:
    """update_node_status() adds a row to the status table."""
    panel = AgingPanel()
    qtbot.addWidget(panel)
    panel.update_node_status("node_A", 1000, 0.8)
    assert panel._status_table.rowCount() == 1


def test_control_plane_panel_creates(qtbot: QtBot) -> None:
    """ControlPlanePanel() creates without error."""
    panel = ControlPlanePanel()
    qtbot.addWidget(panel)


def test_control_plane_update_traffic(qtbot: QtBot) -> None:
    """update_link_traffic() adds a row to the traffic table."""
    panel = ControlPlanePanel()
    qtbot.addWidget(panel)
    panel.update_link_traffic("link_01", 0.3, 0.002)
    assert panel._traffic_table.rowCount() == 1


def test_base_panel_section_label(qtbot: QtBot) -> None:
    """_make_section_label() returns a QLabel with objectName 'section_header'."""
    panel = ChannelPanel()
    qtbot.addWidget(panel)
    label = panel._make_section_label("test")
    assert label.objectName() == "section_header"
