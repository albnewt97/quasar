"""KeyRatePlot: live BB84 secret key rate dashboard (§7.4).

Three sub-views in a vertical QSplitter:
  Top:    Secret key rate [bps] vs simulation time (log-scale Y).
  Middle: QBER vs time with the security threshold and operating-point marker.
  Bottom: Static R(Q) rate curve with the live operating-point dot.
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
from qndt.physics.key_rate import BB84KeyRateCalculator, KeyRateParams

_LINK_COLOURS = (
    QUASAR_PLOT_THEME["accent_cyan"],
    QUASAR_PLOT_THEME["accent_green"],
    QUASAR_PLOT_THEME["accent_amber"],
    QUASAR_PLOT_THEME["accent_purple"],
    QUASAR_PLOT_THEME["accent_red"],
)

# Computed from BB84KeyRateCalculator(KeyRateParams()) — ~0.098 with f_ec=1.16.
_BB84_THRESHOLD: float = BB84KeyRateCalculator(KeyRateParams()).qber_threshold()
_SKR_FLOOR = 1.0  # minimum bps shown on log-scale to keep point visible


class KeyRatePlot(QWidget):
    """Live secret key rate dashboard with three coordinated sub-plots.

    Top sub-plot:    SKR [bps] vs time (log-scale Y), one curve per link.
    Middle sub-plot: QBER vs time with the BB84 security threshold line.
    Bottom sub-plot: Static BB84 rate curve R(Q) with live operating-point dot.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        configure_pyqtgraph()

        # --- top: SKR over time (log Y) ------------------------------------
        self._skr_plot = make_plot_widget(
            title="Secret Key Rate",
            x_label="Time (s)",
            y_label="SKR (bps)",
        )
        self._skr_plot.setLogMode(y=True)
        enable_interactions(self._skr_plot)
        skr_zero = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            pen=make_pen(
                QUASAR_PLOT_THEME["accent_red"], width=1, style=Qt.PenStyle.DashLine
            ),
            label="Rate = 0 threshold",
            labelOpts={
                "color": QUASAR_PLOT_THEME["accent_red"],
                "position": 0.95,
            },
        )
        self._skr_plot.addItem(skr_zero)

        # --- middle: QBER over time with threshold --------------------------
        self._qber_plot = make_plot_widget(
            title="QBER with Security Bound",
            x_label="Time (s)",
            y_label="QBER",
            y_min=0.0,
            y_max=0.5,
        )
        enable_interactions(self._qber_plot)
        self._qber_threshold_line = pg.InfiniteLine(
            pos=_BB84_THRESHOLD,
            angle=0,
            pen=make_pen(
                QUASAR_PLOT_THEME["accent_red"], width=1, style=Qt.PenStyle.DashLine
            ),
            label=f"BB84 threshold ({_BB84_THRESHOLD:.3f})",
            labelOpts={
                "color": QUASAR_PLOT_THEME["accent_red"],
                "position": 0.95,
            },
        )
        self._qber_plot.addItem(self._qber_threshold_line)

        # --- bottom: static rate curve R(Q) with live operating point ------
        self._calculator = BB84KeyRateCalculator(KeyRateParams())
        self._rate_curve_plot = make_plot_widget(
            title="BB84 Key Rate Curve",
            x_label="QBER",
            y_label="Rate (bps)",
            y_min=0.0,
        )
        enable_interactions(self._rate_curve_plot)
        # Create _op_dot before _draw_rate_curve() so the method can re-add it.
        self._op_dot = pg.ScatterPlotItem(size=12)
        self._draw_rate_curve()

        # --- crosshairs -------------------------------------------------------
        self._skr_crosshair = CrosshairOverlay(
            self._skr_plot,
            x_label="t",
            y_label="SKR",
            x_fmt="{:.2f}",
            y_fmt="{:.2e}",
        )
        self._qber_crosshair = CrosshairOverlay(
            self._qber_plot,
            x_label="t",
            y_label="QBER",
            x_fmt="{:.2f}",
            y_fmt="{:.4f}",
            threshold=_BB84_THRESHOLD,
        )
        self._rate_curve_crosshair = CrosshairOverlay(
            self._rate_curve_plot,
            x_label="QBER",
            y_label="rate",
            x_fmt="{:.4f}",
            y_fmt="{:.2e}",
        )

        # --- splitter layout ------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Vertical)
        for plot in (self._skr_plot, self._qber_plot, self._rate_curve_plot):
            splitter.addWidget(plot)
        splitter.setSizes([200, 200, 200])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- per-link data buffers ------------------------------------------
        self._skr_curves: dict[str, pg.PlotDataItem] = {}
        self._qber_curves: dict[str, pg.PlotDataItem] = {}
        self._xs: dict[str, list[float]] = {}
        self._ys_skr: dict[str, list[float]] = {}
        self._ys_qber: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_link(self, link_id: str, colour: str | None = None) -> None:
        """Register a fiber link and create its curves in all three sub-plots.

        Args:
            link_id: Fiber link identifier.
            colour: Hex colour override; cycles through accent colours if omitted.
        """
        if link_id in self._skr_curves:
            return
        if colour is None:
            colour = _LINK_COLOURS[len(self._skr_curves) % len(_LINK_COLOURS)]
        pen = make_pen(colour)
        self._skr_curves[link_id] = self._skr_plot.plot([], [], pen=pen, name=link_id)
        self._qber_curves[link_id] = self._qber_plot.plot([], [], pen=pen, name=link_id)
        self._xs[link_id] = []
        self._ys_skr[link_id] = []
        self._ys_qber[link_id] = []
        self._skr_crosshair.attach_series(link_id, self._skr_curves[link_id])
        self._qber_crosshair.attach_series(link_id, self._qber_curves[link_id])

    def update_sample(
        self, link_id: str, t: float, qber: float, skr_bps: float
    ) -> None:
        """Append a live simulation sample and refresh all sub-plots.

        Args:
            link_id: Fiber link identifier.
            t: Simulation time [s].
            qber: Quantum bit error rate.
            skr_bps: Secret key rate [bits/sec].
        """
        if link_id not in self._skr_curves:
            self.add_link(link_id)

        rolling_append(self._xs[link_id], self._ys_skr[link_id], t, max(skr_bps, _SKR_FLOOR))
        self._ys_qber[link_id].append(qber)
        if len(self._ys_qber[link_id]) > MAX_PLOT_POINTS:
            self._ys_qber[link_id].pop(0)

        self._skr_curves[link_id].setData(self._xs[link_id], self._ys_skr[link_id])
        self._qber_curves[link_id].setData(self._xs[link_id], self._ys_qber[link_id])

        theme = QUASAR_PLOT_THEME
        dot_colour = theme["accent_green"] if skr_bps > 0 else theme["accent_red"]
        self._op_dot.setData(
            [qber],
            [max(skr_bps, _SKR_FLOOR)],
            brush=pg.mkBrush(dot_colour),
            pen=pg.mkPen(None),
        )

    def set_params(self, params: KeyRateParams) -> None:
        """Update the key rate calculator and redraw the static rate curve.

        Args:
            params: New ``KeyRateParams`` to use for all subsequent calculations.
        """
        self._calculator = BB84KeyRateCalculator(params)
        self._draw_rate_curve()

    def clear(self) -> None:
        """Fully reset all link data and remove curves from both sub-plots."""
        for curve in self._skr_curves.values():
            self._skr_plot.removeItem(curve)
        for curve in self._qber_curves.values():
            self._qber_plot.removeItem(curve)
        self._skr_curves.clear()
        self._qber_curves.clear()
        self._xs.clear()
        self._ys_skr.clear()
        self._ys_qber.clear()
        self._skr_crosshair._series.clear()
        self._qber_crosshair._series.clear()
        self._rate_curve_crosshair._series.clear()

    @property
    def key_rate_plot(self) -> "KeyRatePlot":
        """Self-reference property for symmetry with other dashboard widgets."""
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_rate_curve(self) -> None:
        """Draw the static BB84 rate curve and shaded unsafe region."""
        self._rate_curve_plot.clear()
        self._rate_curve_plot.addItem(self._op_dot)

        qbers, rates = self._calculator.rate_vs_qber(200)
        threshold = self._calculator.qber_threshold()

        self._rate_curve_plot.plot(
            qbers,
            rates,
            pen=make_pen(QUASAR_PLOT_THEME["foreground"], width=2),
        )

        unsafe_fill = pg.FillBetweenItem(
            pg.PlotDataItem(
                [threshold, 0.5],
                [0.0, 0.0],
            ),
            pg.PlotDataItem(
                [threshold, 0.5],
                [0.0, 0.0],
            ),
            brush=pg.mkBrush(248, 81, 73, 40),
        )
        self._rate_curve_plot.addItem(unsafe_fill)

        threshold_line = pg.InfiniteLine(
            pos=threshold,
            angle=90,
            pen=make_pen(
                QUASAR_PLOT_THEME["accent_red"], width=1, style=Qt.PenStyle.DashLine
            ),
        )
        self._rate_curve_plot.addItem(threshold_line)
