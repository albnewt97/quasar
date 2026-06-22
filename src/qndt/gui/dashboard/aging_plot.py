"""AgingPlot: device aging -- coherence decay and gate overrotation (§5.5, §7.4).

Two stacked plots per node: T2 degradation against cumulative op count
(with the theoretical exponential wear curve overlaid), and gate
overrotation ε(t) drift.
"""
from __future__ import annotations

import math

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from qndt.gui.dashboard.plot_utils import (
    QUASAR_PLOT_THEME,
    CrosshairOverlay,
    configure_pyqtgraph,
    enable_interactions,
    make_pen,
    make_plot_widget,
    rolling_append,
)

_NODE_COLOURS = (
    QUASAR_PLOT_THEME["accent_cyan"],
    QUASAR_PLOT_THEME["accent_green"],
    QUASAR_PLOT_THEME["accent_amber"],
    QUASAR_PLOT_THEME["accent_purple"],
    QUASAR_PLOT_THEME["accent_red"],
)

_THEORETICAL_CURVE_POINTS = 200


class AgingPlot(QWidget):
    """Live T2 coherence decay and gate overrotation drift per node."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        configure_pyqtgraph()

        self._t2_plot = make_plot_widget(
            title="Coherence Time Degradation",
            x_label="Cumulative ops N",
            y_label="T2 (s)",
        )
        self._overrotation_plot = make_plot_widget(
            title="Gate Overrotation ε(t)",
            x_label="Simulation Time (s)",
            y_label="ε",
        )
        enable_interactions(self._t2_plot)
        enable_interactions(self._overrotation_plot)

        zero_line = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            pen=make_pen(
                QUASAR_PLOT_THEME["foreground"], width=1, style=Qt.PenStyle.DashLine
            ),
        )
        self._overrotation_plot.addItem(zero_line)

        self._t2_curves: dict[str, pg.PlotDataItem] = {}
        self._theoretical_curves: dict[str, pg.PlotDataItem] = {}
        self._t2_xs: dict[str, list[float]] = {}
        self._t2_ys: dict[str, list[float]] = {}

        self._overrotation_curves: dict[str, pg.PlotDataItem] = {}
        self._overrotation_xs: dict[str, list[float]] = {}
        self._overrotation_ys: dict[str, list[float]] = {}

        self._node_params: dict[str, tuple[float, float]] = {}

        self._t2_crosshair = CrosshairOverlay(
            self._t2_plot,
            x_label="N",
            y_label="T2",
            x_fmt="{:.0f}",
            y_fmt="{:.4f}",
        )
        self._overrotation_crosshair = CrosshairOverlay(
            self._overrotation_plot,
            x_label="t",
            y_label="ε",
            x_fmt="{:.2f}",
            y_fmt="{:.4f}",
        )

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._t2_plot)
        splitter.addWidget(self._overrotation_plot)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def add_node(self, node_id: str, t2_nominal: float, wear_const_nc: float) -> None:
        """Register a node and plot its theoretical exponential wear curve.

        Args:
            node_id: Node to track.
            t2_nominal: Nominal (N=0) coherence time T2_0 [s].
            wear_const_nc: Wear constant Nc (§5.5).
        """
        if node_id in self._t2_curves:
            return
        colour = _NODE_COLOURS[len(self._t2_curves) % len(_NODE_COLOURS)]
        self._node_params[node_id] = (t2_nominal, wear_const_nc)

        self._t2_curves[node_id] = self._t2_plot.plot([], [], pen=make_pen(colour), name=node_id)
        self._t2_xs[node_id] = []
        self._t2_ys[node_id] = []

        max_n = wear_const_nc * 5.0
        ns = [max_n * i / (_THEORETICAL_CURVE_POINTS - 1) for i in range(_THEORETICAL_CURVE_POINTS)]
        theoretical_t2 = [t2_nominal * math.exp(-n / wear_const_nc) for n in ns]
        self._theoretical_curves[node_id] = self._t2_plot.plot(
            ns,
            theoretical_t2,
            pen=make_pen(
                QUASAR_PLOT_THEME["foreground"], width=1, style=Qt.PenStyle.DashLine
            ),
        )

        self._overrotation_curves[node_id] = self._overrotation_plot.plot(
            [], [], pen=make_pen(colour), name=node_id
        )
        self._overrotation_xs[node_id] = []
        self._overrotation_ys[node_id] = []
        self._t2_crosshair.attach_series(node_id, self._t2_curves[node_id])
        self._overrotation_crosshair.attach_series(node_id, self._overrotation_curves[node_id])

    def update_sample(
        self, node_id: str, op_count: int, t2_current: float, overrotation: float, t: float
    ) -> None:
        """Append a new aging sample for ``node_id``.

        Args:
            node_id: Node to update.
            op_count: Cumulative operation count N.
            t2_current: Current coherence time T2(N) [s].
            overrotation: Current gate overrotation ε(t).
            t: Simulation time [s].
        """
        if node_id not in self._t2_curves:
            t2_nominal, wear_const_nc = self._node_params.get(node_id, (t2_current, 1.0))
            self.add_node(node_id, t2_nominal, wear_const_nc)
        rolling_append(self._t2_xs[node_id], self._t2_ys[node_id], float(op_count), t2_current)
        self._t2_curves[node_id].setData(self._t2_xs[node_id], self._t2_ys[node_id])

        rolling_append(
            self._overrotation_xs[node_id], self._overrotation_ys[node_id], t, overrotation
        )
        self._overrotation_curves[node_id].setData(
            self._overrotation_xs[node_id], self._overrotation_ys[node_id]
        )

    def clear(self) -> None:
        """Fully reset all node data and remove all curves from both sub-plots."""
        for curve in self._t2_curves.values():
            self._t2_plot.removeItem(curve)
        for curve in self._theoretical_curves.values():
            self._t2_plot.removeItem(curve)
        for curve in self._overrotation_curves.values():
            self._overrotation_plot.removeItem(curve)
        self._t2_curves.clear()
        self._theoretical_curves.clear()
        self._overrotation_curves.clear()
        self._t2_xs.clear()
        self._t2_ys.clear()
        self._overrotation_xs.clear()
        self._overrotation_ys.clear()
        self._node_params.clear()
        self._t2_crosshair._series.clear()
        self._overrotation_crosshair._series.clear()
