"""QBERPlot: live quantum bit error rate per link (§7.4).

QBER is the headline link-quality metric: the BB84 security bound at
QBER = 0.11 is drawn as a permanent reference line.
"""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from qndt.gui.dashboard.plot_utils import (
    QUASAR_PLOT_THEME,
    CrosshairOverlay,
    add_interactive_legend,
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

# Computed from BB84KeyRateCalculator(KeyRateParams()) — ~0.098 with f_ec=1.16,
# not the cruder 0.11 approximation (valid only for f_ec=1.0).
_BB84_THRESHOLD: float = BB84KeyRateCalculator(KeyRateParams()).qber_threshold()


class QBERPlot(QWidget):
    """Live QBER per link, with the BB84 security bound annotated."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        configure_pyqtgraph()

        self._plot = make_plot_widget(
            title="Quantum Bit Error Rate",
            x_label="Simulation Time (s)",
            y_label="QBER",
            y_min=0.0,
            y_max=0.5,
        )
        # Prevent pyqtgraph from zooming in when low-QBER data arrives, which
        # would push the BB84 security threshold line (0.11) off-screen.
        self._plot.getViewBox().setAutoVisible(y=False)
        self._plot.enableAutoRange('y', False)
        self._legend = add_interactive_legend(self._plot)
        enable_interactions(self._plot)
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._xs: dict[str, list[float]] = {}
        self._ys: dict[str, list[float]] = {}

        threshold = pg.InfiniteLine(
            pos=_BB84_THRESHOLD,
            angle=0,
            pen=make_pen(QUASAR_PLOT_THEME["accent_red"], width=1, style=Qt.PenStyle.DashLine),
            label=f"BB84 security bound ({_BB84_THRESHOLD:.3f})",
            labelOpts={"color": QUASAR_PLOT_THEME["accent_red"], "position": 0.95},
        )
        self._plot.addItem(threshold)
        self._crosshair = CrosshairOverlay(
            self._plot,
            x_label="t",
            y_label="QBER",
            x_fmt="{:.2f}",
            y_fmt="{:.4f}",
            threshold=_BB84_THRESHOLD,
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self._plot)
        self._legend_label = QLabel("")
        layout.addWidget(self._legend_label)

    def add_link(self, link_id: str, colour: str | None = None) -> None:
        """Register a link and assign it a curve and colour.

        Args:
            link_id: Link to track.
            colour: Hex colour override; cycles through accent colours if
                omitted.
        """
        if link_id in self._curves:
            return
        if colour is None:
            colour = _LINK_COLOURS[len(self._curves) % len(_LINK_COLOURS)]
        curve = self._plot.plot([], [], pen=make_pen(colour), name=link_id)
        self._curves[link_id] = curve
        self._xs[link_id] = []
        self._ys[link_id] = []
        self._crosshair.attach_series(link_id, curve)
        self._update_legend()

    def update_sample(self, link_id: str, t: float, qber: float) -> None:
        """Append a new QBER sample for ``link_id`` and refresh its curve.

        Args:
            link_id: Link to update.
            t: Simulation time [s].
            qber: Quantum bit error rate in ``[0, 1]``.
        """
        if link_id not in self._curves:
            self.add_link(link_id)
        rolling_append(self._xs[link_id], self._ys[link_id], t, qber)
        self._curves[link_id].setData(self._xs[link_id], self._ys[link_id])

    def clear(self) -> None:
        """Fully reset all link data, remove curves from the plot, and clear the legend."""
        for curve in self._curves.values():
            self._plot.removeItem(curve)
        self._curves.clear()
        self._xs.clear()
        self._ys.clear()
        if self._legend is not None:
            self._legend.clear()
        self._crosshair._series.clear()
        self._legend_label.setText("")

    def _update_legend(self) -> None:
        colours = {
            link_id: _LINK_COLOURS[i % len(_LINK_COLOURS)]
            for i, link_id in enumerate(self._curves)
        }
        html = "  ".join(
            f'<span style="color:{colour}">&#9632;</span> {link_id}'
            for link_id, colour in colours.items()
        )
        self._legend_label.setText(html)
