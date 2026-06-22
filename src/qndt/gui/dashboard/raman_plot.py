"""RamanPlot: SpRS Raman cross-talk noise vs classical WDM load (§5.4, §7.4).

Two stacked plots: the resulting dark-count rate on top, and the count
of active classical channels driving it on the bottom.
"""
from __future__ import annotations

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

_DEFAULT_DARK_COUNT_BASELINE = 1e-5


class RamanPlot(QWidget):
    """Live Raman-induced dark count rate vs active classical channel count."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        configure_pyqtgraph()

        self._rate_plot = make_plot_widget(
            title="Raman Dark Count Rate",
            x_label="Simulation Time (s)",
            y_label="Dark count rate (Hz)",
        )
        self._channel_plot = make_plot_widget(
            title="Active Classical Channels",
            x_label="Simulation Time (s)",
            y_label="Channel count",
        )
        enable_interactions(self._rate_plot)
        enable_interactions(self._channel_plot)

        self._rate_curves: dict[str, pg.PlotDataItem] = {}
        self._channel_curves: dict[str, pg.PlotDataItem] = {}
        self._rate_xs: dict[str, list[float]] = {}
        self._rate_ys: dict[str, list[float]] = {}
        self._channel_xs: dict[str, list[float]] = {}
        self._channel_ys: dict[str, list[float]] = {}

        self._dark_count_baseline = _DEFAULT_DARK_COUNT_BASELINE
        self._baseline_line = pg.InfiniteLine(
            pos=self._dark_count_baseline,
            angle=0,
            pen=make_pen(
                QUASAR_PLOT_THEME["foreground"], width=1, style=Qt.PenStyle.DashLine
            ),
            label="Intrinsic dark count baseline",
            labelOpts={"color": QUASAR_PLOT_THEME["foreground"], "position": 0.95},
        )
        self._rate_plot.addItem(self._baseline_line)
        self._rate_crosshair = CrosshairOverlay(
            self._rate_plot,
            x_label="t",
            y_label="rate",
            x_fmt="{:.2f}",
            y_fmt="{:.2e}",
        )
        self._channel_crosshair = CrosshairOverlay(
            self._channel_plot,
            x_label="t",
            y_label="channels",
            x_fmt="{:.2f}",
            y_fmt="{:.0f}",
        )

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._rate_plot)
        splitter.addWidget(self._channel_plot)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def _ensure_link(self, link_id: str) -> None:
        if link_id in self._rate_curves:
            return
        self._rate_curves[link_id] = self._rate_plot.plot(
            [], [], pen=make_pen(QUASAR_PLOT_THEME["accent_cyan"]), name=link_id
        )
        self._channel_curves[link_id] = self._channel_plot.plot(
            [], [], pen=make_pen(QUASAR_PLOT_THEME["accent_amber"]), name=link_id
        )
        self._rate_xs[link_id] = []
        self._rate_ys[link_id] = []
        self._channel_xs[link_id] = []
        self._channel_ys[link_id] = []
        self._rate_crosshair.attach_series(link_id, self._rate_curves[link_id])
        self._channel_crosshair.attach_series(link_id, self._channel_curves[link_id])

    def update_sample(self, link_id: str, t: float, raman_rate: float, n_classical: int) -> None:
        """Append new Raman noise and channel-count samples for ``link_id``.

        Args:
            link_id: Link to update.
            t: Simulation time [s].
            raman_rate: Raman-induced dark count rate [Hz].
            n_classical: Number of currently active classical WDM channels.
        """
        self._ensure_link(link_id)
        rolling_append(self._rate_xs[link_id], self._rate_ys[link_id], t, raman_rate)
        self._rate_curves[link_id].setData(self._rate_xs[link_id], self._rate_ys[link_id])
        rolling_append(
            self._channel_xs[link_id], self._channel_ys[link_id], t, float(n_classical)
        )
        self._channel_curves[link_id].setData(
            self._channel_xs[link_id], self._channel_ys[link_id]
        )

    def set_dark_count_baseline(self, p_dc: float) -> None:
        """Move the intrinsic dark count baseline reference line.

        Args:
            p_dc: New intrinsic dark count rate [Hz].
        """
        self._dark_count_baseline = p_dc
        self._baseline_line.setPos(p_dc)

    def clear(self) -> None:
        """Fully reset all link data and remove curves from both sub-plots."""
        for curve in self._rate_curves.values():
            self._rate_plot.removeItem(curve)
        for curve in self._channel_curves.values():
            self._channel_plot.removeItem(curve)
        self._rate_curves.clear()
        self._channel_curves.clear()
        self._rate_xs.clear()
        self._rate_ys.clear()
        self._channel_xs.clear()
        self._channel_ys.clear()
        self._rate_crosshair._series.clear()
        self._channel_crosshair._series.clear()
