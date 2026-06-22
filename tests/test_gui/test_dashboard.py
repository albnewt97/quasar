"""Tests for dashboard plots (§7.4).

Run headless via ``QT_QPA_PLATFORM=offscreen`` (see conftest.py qapp_env).
"""
from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from pytestqt.qtbot import QtBot

from qndt.gui.dashboard.aging_plot import AgingPlot
from qndt.gui.dashboard.dashboard_window import DashboardWindow
from qndt.gui.dashboard.fidelity_plot import FidelityPlot
from qndt.gui.dashboard.key_rate_plot import _BB84_THRESHOLD as _KR_THRESHOLD
from qndt.gui.dashboard.key_rate_plot import KeyRatePlot
from qndt.gui.dashboard.nonmarkov_plot import NonMarkovPlot
from qndt.gui.dashboard.plot_utils import (
    CrosshairOverlay,
    configure_pyqtgraph,
    enable_interactions,
)
from qndt.gui.dashboard.qber_plot import _BB84_THRESHOLD as _QBER_THRESHOLD
from qndt.gui.dashboard.qber_plot import QBERPlot
from qndt.gui.dashboard.raman_plot import RamanPlot
from qndt.gui.topology.topology_model import TopologyModel
from qndt.physics.key_rate import BB84KeyRateCalculator, KeyRateParams

configure_pyqtgraph()


def test_qber_plot_creates(qtbot: QtBot) -> None:
    """QBERPlot() creates without error."""
    plot = QBERPlot()
    qtbot.addWidget(plot)


def test_qber_plot_update(qtbot: QtBot) -> None:
    """add_link + update(link_id, 0.1, 0.05) doesn't raise."""
    plot = QBERPlot()
    qtbot.addWidget(plot)
    plot.add_link("l1")
    plot.update_sample("l1", 0.1, 0.05)


def test_qber_plot_threshold_line(qtbot: QtBot) -> None:
    """Plot has at least 2 items (curve + threshold)."""
    plot = QBERPlot()
    qtbot.addWidget(plot)
    plot.update_sample("l1", 0.1, 0.05)
    assert len(plot._plot.getPlotItem().items) >= 2


def test_fidelity_plot_creates(qtbot: QtBot) -> None:
    """FidelityPlot() creates without error."""
    plot = FidelityPlot()
    qtbot.addWidget(plot)


def test_fidelity_plot_shaded_region(qtbot: QtBot) -> None:
    """Plot has LinearRegionItem child."""
    plot = FidelityPlot()
    qtbot.addWidget(plot)
    items = plot._plot.getPlotItem().items
    assert any(isinstance(item, pg.LinearRegionItem) for item in items)


def test_raman_plot_creates(qtbot: QtBot) -> None:
    """RamanPlot() creates without error."""
    plot = RamanPlot()
    qtbot.addWidget(plot)
    plot.update_sample("l1", 0.1, 1e-4, 3)


def test_nonmarkov_plot_creates(qtbot: QtBot) -> None:
    """NonMarkovPlot() creates without error."""
    plot = NonMarkovPlot()
    qtbot.addWidget(plot)
    plot.update_sample("l1", 0.1, 0.01, 0.05, 0.05, 0.05)


def test_nonmarkov_backflow_detection(qtbot: QtBot) -> None:
    """gamma_x negative then positive yields one backflow region."""
    plot = NonMarkovPlot()
    qtbot.addWidget(plot)
    plot.update_sample("l1", 0.0, 0.0, -0.1, 0.05, 0.05)
    plot.update_sample("l1", 1.0, 0.0, 0.1, 0.05, 0.05)
    assert len(plot._backflow_regions) == 1


def test_aging_plot_creates(qtbot: QtBot) -> None:
    """AgingPlot() creates without error."""
    plot = AgingPlot()
    qtbot.addWidget(plot)


def test_aging_plot_add_node(qtbot: QtBot) -> None:
    """add_node("n1", 1.0, 1e6) doesn't raise."""
    plot = AgingPlot()
    qtbot.addWidget(plot)
    plot.add_node("n1", 1.0, 1e6)


def test_dashboard_window_creates(qtbot: QtBot) -> None:
    """DashboardWindow(TopologyModel()) creates."""
    model = TopologyModel()
    dashboard = DashboardWindow(model)
    qtbot.addWidget(dashboard)


def test_dashboard_update_delegates(qtbot: QtBot) -> None:
    """dashboard.update_qber('l1', 0.1, 0.05) updates qber_plot._xs['l1']."""
    model = TopologyModel()
    dashboard = DashboardWindow(model)
    qtbot.addWidget(dashboard)
    dashboard.update_qber("l1", 0.1, 0.05)
    assert dashboard.qber_plot._xs["l1"] == [0.1]


def test_dashboard_clear_all(qtbot: QtBot) -> None:
    """update then clear_all() fully resets all plot dicts (keys removed, not zeroed)."""
    model = TopologyModel()
    dashboard = DashboardWindow(model)
    qtbot.addWidget(dashboard)
    dashboard.update_qber("l1", 0.1, 0.05)
    dashboard.update_fidelity("l1", 0.1, 0.9)
    dashboard.update_raman("l1", 0.1, 1e-4, 2)
    dashboard.update_nonmarkov("l1", 0.1, 0.0, 0.05, 0.05, 0.05)
    dashboard.update_aging("n1", 10, 0.9, 0.0, 0.1)
    dashboard.clear_all()
    assert len(dashboard.qber_plot._xs) == 0
    assert len(dashboard.fidelity_plot._xs) == 0
    assert len(dashboard.raman_plot._rate_xs) == 0
    assert len(dashboard.nonmarkov_plot._witness_xs) == 0
    assert len(dashboard.aging_plot._t2_xs) == 0


# ---------------------------------------------------------------------------
# Phase 3 interactive feature tests
# ---------------------------------------------------------------------------


def test_crosshair_overlay_creates(qtbot: QtBot) -> None:
    """CrosshairOverlay attaches to a PlotWidget without error."""
    configure_pyqtgraph()
    pw = pg.PlotWidget()
    qtbot.addWidget(pw)
    overlay = CrosshairOverlay(pw, x_label="t", y_label="QBER")
    assert overlay.vline is not None
    assert overlay.hline is not None
    assert overlay.label is not None


def test_crosshair_attach_series(qtbot: QtBot) -> None:
    """attach_series() registers the curve in the overlay's _series dict."""
    configure_pyqtgraph()
    pw = pg.PlotWidget()
    qtbot.addWidget(pw)
    overlay = CrosshairOverlay(pw)
    curve = pw.plot([], [])
    overlay.attach_series("link_01", curve)
    assert "link_01" in overlay._series
    assert overlay._series["link_01"] is curve


def test_enable_interactions(qtbot: QtBot) -> None:
    """enable_interactions() puts the ViewBox into PanMode."""
    configure_pyqtgraph()
    pw = pg.PlotWidget()
    qtbot.addWidget(pw)
    pw.getViewBox().setMouseMode(pg.ViewBox.RectMode)
    enable_interactions(pw)
    assert pw.getViewBox().state["mouseMode"] == pg.ViewBox.PanMode


def test_plot_pause_flag(qtbot: QtBot) -> None:
    """_paused=True prevents update_qber from appending to the rolling buffer."""
    model = TopologyModel()
    dashboard = DashboardWindow(model)
    qtbot.addWidget(dashboard)
    dashboard.update_qber("l1", 0.1, 0.05)
    assert dashboard.qber_plot._xs["l1"] == [0.1]
    dashboard._paused = True
    dashboard.update_qber("l1", 0.2, 0.06)
    assert dashboard.qber_plot._xs["l1"] == [0.1]


def test_reset_zoom_runs(qtbot: QtBot) -> None:
    """reset_zoom() calls autoRange on all plots in the active tab."""
    model = TopologyModel()
    dashboard = DashboardWindow(model)
    qtbot.addWidget(dashboard)
    dashboard.update_qber("l1", 0.1, 0.05)
    dashboard.reset_zoom()


def test_export_plot_creates_file(qtbot: QtBot, tmp_path: Path) -> None:
    """export_current_plot(path) writes a non-empty PNG to disk."""
    model = TopologyModel()
    dashboard = DashboardWindow(model)
    qtbot.addWidget(dashboard)
    dashboard.resize(600, 400)
    dashboard.show()
    dashboard.update_qber("l1", 0.1, 0.05)

    out = str(tmp_path / "test_export.png")
    dashboard.export_current_plot(out)

    assert Path(out).exists(), "export_current_plot() did not create the file"
    assert Path(out).stat().st_size > 0, "exported PNG is empty"


# ---------------------------------------------------------------------------
# Regression: QBER threshold must be the computed BB84 bound, not 0.11
# ---------------------------------------------------------------------------


def test_qber_threshold_matches_key_rate_calculator() -> None:
    """_BB84_THRESHOLD in qber_plot and key_rate_plot equals the computed GLLP bound.

    With the default KeyRateParams (f_ec=1.16), the threshold is ~0.098, not
    the cruder 0.11 approximation valid only for f_ec=1.0.  This regression
    test catches any reintroduction of a hardcoded 0.11.
    """
    computed = BB84KeyRateCalculator(KeyRateParams()).qber_threshold()
    assert abs(_QBER_THRESHOLD - computed) < 1e-8, (
        f"QBERPlot._BB84_THRESHOLD ({_QBER_THRESHOLD:.4f}) drifted from "
        f"computed threshold ({computed:.4f})"
    )
    assert abs(_KR_THRESHOLD - computed) < 1e-8, (
        f"KeyRatePlot._BB84_THRESHOLD ({_KR_THRESHOLD:.4f}) drifted from "
        f"computed threshold ({computed:.4f})"
    )
    # Verify we're using the precise f_ec=1.16 value (~0.098), not 0.11
    assert _QBER_THRESHOLD < 0.105, (
        f"Threshold {_QBER_THRESHOLD:.4f} looks like the f_ec=1.0 approximation (0.11); "
        "expected ~0.098 for f_ec=1.16"
    )


def test_clear_removes_curves(qtbot: QtBot) -> None:
    """add_link x2 + update + clear() leaves _curves empty and no PlotDataItem in plot."""
    plot = QBERPlot()
    qtbot.addWidget(plot)
    plot.add_link("l1")
    plot.add_link("l2")
    plot.update_sample("l1", 0.1, 0.05)
    plot.clear()
    assert len(plot._curves) == 0
    data_items = [
        item for item in plot._plot.getPlotItem().items
        if isinstance(item, pg.PlotDataItem)
    ]
    assert len(data_items) == 0


def test_add_link_idempotent(qtbot: QtBot) -> None:
    """add_link('l1') called twice produces exactly one curve."""
    plot = QBERPlot()
    qtbot.addWidget(plot)
    plot.add_link("l1")
    plot.add_link("l1")
    assert len(plot._curves) == 1
    data_items = [
        item for item in plot._plot.getPlotItem().items
        if isinstance(item, pg.PlotDataItem)
    ]
    assert len(data_items) == 1


def test_legend_cleared(qtbot: QtBot) -> None:
    """After clear(), the interactive legend has no entries."""
    plot = QBERPlot()
    qtbot.addWidget(plot)
    plot.add_link("l1")
    plot.add_link("l2")
    plot.clear()
    assert len(plot._legend.items) == 0


def test_key_rate_clear_removes_curves(qtbot: QtBot) -> None:
    """KeyRatePlot.clear() removes all curves from skr and qber sub-plots."""
    plot = KeyRatePlot()
    qtbot.addWidget(plot)
    plot.add_link("l1")
    plot.update_sample("l1", 0.1, 0.05, 1000.0)
    plot.clear()
    assert len(plot._skr_curves) == 0
    assert len(plot._qber_curves) == 0


def test_key_rate_clear_resets_all_crosshair_series(qtbot: QtBot) -> None:
    """KeyRatePlot.clear() clears _series on all three crosshair overlays."""
    plot = KeyRatePlot()
    qtbot.addWidget(plot)
    plot.add_link("l1")
    plot.update_sample("l1", 0.1, 0.05, 1000.0)
    plot.clear()
    assert len(plot._skr_crosshair._series) == 0
    assert len(plot._qber_crosshair._series) == 0
    assert len(plot._rate_curve_crosshair._series) == 0


def test_aging_config_applied(qtbot: QtBot) -> None:
    """NodePropertiesDialog.get_config() returns t2_nominal and wear_const_nc keys."""
    from qndt.gui.topology.node_item import NodePropertiesDialog

    dialog = NodePropertiesDialog("n1", "memory_node")
    qtbot.addWidget(dialog)
    cfg = dialog.get_config()
    assert "t2_nominal" in cfg, "get_config() must include t2_nominal"
    assert "wear_const_nc" in cfg, "get_config() must include wear_const_nc"
    assert cfg["node_type"] == "memory_node"
