"""NonMarkovPlot: rate-based divisibility witness N_rate and canonical rates (§5.6, §7.4).

N_rate is the rate-based divisibility witness that accumulates Σ_k ∫ |γk(t)| dt
over periods where canonical decoherence rates go negative.  It is NOT the
RHP entanglement measure I^(E) and is not on the same scale.  A positive
N_rate value indicates non-divisibility of the dynamical map (information
backflow), a structurally non-Markovian signature.  Periods where any
canonical rate goes negative are highlighted as backflow regions.
"""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from qndt.gui.dashboard.plot_utils import (
    MAX_PLOT_POINTS,
    QUASAR_PLOT_THEME,
    CrosshairOverlay,
    configure_pyqtgraph,
    enable_interactions,
    make_pen,
    make_plot_widget,
    rolling_append,
)


class NonMarkovPlot(QWidget):
    """Live RHP witness timeline with canonical-rate sign-change markers."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        configure_pyqtgraph()

        self._witness_plot = make_plot_widget(
            title="Rate-Based Divisibility Witness (N_rate)",
            x_label="Simulation Time (s)",
            y_label="N_rate",
        )
        self._rates_plot = make_plot_widget(
            title="Canonical Rates",
            x_label="Simulation Time (s)",
            y_label="γ (1/s)",
        )
        enable_interactions(self._witness_plot)
        enable_interactions(self._rates_plot)

        self._witness_curves: dict[str, pg.PlotDataItem] = {}
        self._witness_xs: dict[str, list[float]] = {}
        self._witness_ys: dict[str, list[float]] = {}

        self._gamma_x_curves: dict[str, pg.PlotDataItem] = {}
        self._gamma_y_curves: dict[str, pg.PlotDataItem] = {}
        self._gamma_z_curves: dict[str, pg.PlotDataItem] = {}
        self._rates_xs: dict[str, list[float]] = {}
        self._gamma_x_ys: dict[str, list[float]] = {}
        self._gamma_y_ys: dict[str, list[float]] = {}
        self._gamma_z_ys: dict[str, list[float]] = {}

        annotation = pg.TextItem(
            text=(
                "N_rate > 0 — rate-based divisibility witness, "
                "NOT the RHP entanglement measure Iᵉ⁺⁾, "
                "not on the same scale."
            ),
            color=QUASAR_PLOT_THEME["accent_purple"],
            anchor=(0, 1),
        )
        annotation.setPos(0, 0)
        self._witness_plot.addItem(annotation)

        zero_line = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            pen=make_pen(
                QUASAR_PLOT_THEME["foreground"], width=1, style=Qt.PenStyle.DashLine
            ),
        )
        self._rates_plot.addItem(zero_line)

        self._backflow_regions: list[pg.LinearRegionItem] = []
        self._backflow_start: float | None = None

        self._witness_crosshair = CrosshairOverlay(
            self._witness_plot,
            x_label="t",
            y_label="N_rate",
            x_fmt="{:.2f}",
            y_fmt="{:.4f}",
        )
        self._rates_crosshair = CrosshairOverlay(
            self._rates_plot,
            x_label="t",
            y_label="γ",
            x_fmt="{:.2f}",
            y_fmt="{:.4f}",
            check_backflow=True,
        )

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._witness_plot)
        splitter.addWidget(self._rates_plot)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def _ensure_link(self, link_id: str) -> None:
        if link_id in self._witness_curves:
            return
        self._witness_curves[link_id] = self._witness_plot.plot(
            [], [], pen=make_pen(QUASAR_PLOT_THEME["accent_purple"]), name=link_id
        )
        self._witness_xs[link_id] = []
        self._witness_ys[link_id] = []
        self._witness_crosshair.attach_series(link_id, self._witness_curves[link_id])

        self._gamma_x_curves[link_id] = self._rates_plot.plot(
            [], [], pen=make_pen(QUASAR_PLOT_THEME["accent_cyan"]), name=f"{link_id}:γx"
        )
        self._gamma_y_curves[link_id] = self._rates_plot.plot(
            [], [], pen=make_pen(QUASAR_PLOT_THEME["accent_green"]), name=f"{link_id}:γy"
        )
        self._gamma_z_curves[link_id] = self._rates_plot.plot(
            [], [], pen=make_pen(QUASAR_PLOT_THEME["accent_amber"]), name=f"{link_id}:γz"
        )
        self._rates_xs[link_id] = []
        self._gamma_x_ys[link_id] = []
        self._gamma_y_ys[link_id] = []
        self._gamma_z_ys[link_id] = []
        self._rates_crosshair.attach_series(f"{link_id}:γx", self._gamma_x_curves[link_id])
        self._rates_crosshair.attach_series(f"{link_id}:γy", self._gamma_y_curves[link_id])
        self._rates_crosshair.attach_series(f"{link_id}:γz", self._gamma_z_curves[link_id])

    def update_sample(
        self,
        link_id: str,
        t: float,
        rhp_value: float,
        gamma_x: float,
        gamma_y: float,
        gamma_z: float,
    ) -> None:
        """Append a new RHP/rate sample for ``link_id`` and refresh curves.

        Args:
            link_id: Link to update.
            t: Simulation time [s].
            rhp_value: Current RHP non-Markovianity witness value.
            gamma_x: Canonical decoherence rate γ_x [1/s].
            gamma_y: Canonical decoherence rate γ_y [1/s].
            gamma_z: Canonical decoherence rate γ_z [1/s].
        """
        self._ensure_link(link_id)

        rolling_append(self._witness_xs[link_id], self._witness_ys[link_id], t, rhp_value)
        self._witness_curves[link_id].setData(
            self._witness_xs[link_id], self._witness_ys[link_id]
        )

        self._rates_xs[link_id].append(t)
        self._gamma_x_ys[link_id].append(gamma_x)
        self._gamma_y_ys[link_id].append(gamma_y)
        self._gamma_z_ys[link_id].append(gamma_z)
        if len(self._rates_xs[link_id]) > MAX_PLOT_POINTS:
            self._rates_xs[link_id].pop(0)
            self._gamma_x_ys[link_id].pop(0)
            self._gamma_y_ys[link_id].pop(0)
            self._gamma_z_ys[link_id].pop(0)
        self._gamma_x_curves[link_id].setData(self._rates_xs[link_id], self._gamma_x_ys[link_id])
        self._gamma_y_curves[link_id].setData(self._rates_xs[link_id], self._gamma_y_ys[link_id])
        self._gamma_z_curves[link_id].setData(self._rates_xs[link_id], self._gamma_z_ys[link_id])

        any_negative = gamma_x < 0 or gamma_y < 0 or gamma_z < 0
        if any_negative and self._backflow_start is None:
            self._backflow_start = t
        elif not any_negative and self._backflow_start is not None:
            region = pg.LinearRegionItem(
                values=(self._backflow_start, t),
                orientation="vertical",
                brush=pg.mkBrush(188, 140, 255, 40),
                movable=False,
            )
            self._witness_plot.addItem(region)
            self._backflow_regions.append(region)
            self._backflow_start = None

    def clear(self) -> None:
        """Fully reset all link data and remove all curves and backflow regions."""
        for curve in self._witness_curves.values():
            self._witness_plot.removeItem(curve)
        for curve in self._gamma_x_curves.values():
            self._rates_plot.removeItem(curve)
        for curve in self._gamma_y_curves.values():
            self._rates_plot.removeItem(curve)
        for curve in self._gamma_z_curves.values():
            self._rates_plot.removeItem(curve)
        self._witness_curves.clear()
        self._gamma_x_curves.clear()
        self._gamma_y_curves.clear()
        self._gamma_z_curves.clear()
        self._witness_xs.clear()
        self._witness_ys.clear()
        self._rates_xs.clear()
        self._gamma_x_ys.clear()
        self._gamma_y_ys.clear()
        self._gamma_z_ys.clear()
        for region in self._backflow_regions:
            self._witness_plot.removeItem(region)
        self._backflow_regions.clear()
        self._backflow_start = None
        self._witness_crosshair._series.clear()
        self._rates_crosshair._series.clear()
