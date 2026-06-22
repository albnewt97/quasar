"""Shared pyqtgraph configuration, helpers, and crosshair overlay (§7.4).

All dashboard plots use pyqtgraph, never matplotlib.  ``configure_pyqtgraph``
must run once, before any ``PlotWidget`` is constructed, so every plot picks
up the dark theme.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QPen

QUASAR_PLOT_THEME: dict[str, str] = {
    "background": "#0D1117",
    "foreground": "#8B949E",
    "accent_cyan": "#58A6FF",
    "accent_green": "#3FB950",
    "accent_amber": "#D29922",
    "accent_red": "#F85149",
    "accent_purple": "#BC8CFF",
}

_GRID_ALPHA = 40
MAX_PLOT_POINTS = 500

_configured = False


def configure_pyqtgraph() -> None:
    """Apply the Quasar dark theme to pyqtgraph's global config (once)."""
    global _configured
    if _configured:
        return
    pg.setConfigOptions(
        antialias=True,
        background=QUASAR_PLOT_THEME["background"],
        foreground=QUASAR_PLOT_THEME["foreground"],
    )
    _configured = True


def make_plot_widget(
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    y_min: float | None = None,
    y_max: float | None = None,
) -> pg.PlotWidget:
    """Build a pre-themed ``PlotWidget``.

    Args:
        title: Plot title.
        x_label: Bottom-axis label.
        y_label: Left-axis label.
        y_min: Optional fixed lower y-range bound.
        y_max: Optional fixed upper y-range bound.
    """
    pw = pg.PlotWidget(title=title)
    pw.setBackground(QUASAR_PLOT_THEME["background"])
    pw.showGrid(x=True, y=True, alpha=_GRID_ALPHA / 255)
    pw.setLabel("bottom", x_label)
    pw.setLabel("left", y_label)
    if y_min is not None and y_max is not None:
        pw.setYRange(y_min, y_max)
    pw.getAxis("bottom").setTextPen(QUASAR_PLOT_THEME["foreground"])
    pw.getAxis("left").setTextPen(QUASAR_PLOT_THEME["foreground"])
    pw.getAxis("bottom").enableAutoSIPrefix(False)
    pw.getAxis("left").enableAutoSIPrefix(False)
    return pw


def make_pen(
    colour: str, width: int = 2, style: Qt.PenStyle = Qt.PenStyle.SolidLine
) -> QPen:
    """Build a themed pen for plot curves and reference lines.

    Args:
        colour: Hex colour string.
        width: Line width in pixels.
        style: Qt pen style (solid, dashed, ...).
    """
    return cast(QPen, pg.mkPen(color=colour, width=width, style=style))


def rolling_append(xs: list[float], ys: list[float], x: float, y: float) -> None:
    """Append a sample, evicting the oldest once ``MAX_PLOT_POINTS`` is exceeded.

    Args:
        xs: Rolling x-value buffer, mutated in place.
        ys: Rolling y-value buffer, mutated in place.
        x: New x value.
        y: New y value.
    """
    xs.append(x)
    ys.append(y)
    if len(xs) > MAX_PLOT_POINTS:
        xs.pop(0)
        ys.pop(0)


# ---------------------------------------------------------------------------
# CrosshairOverlay
# ---------------------------------------------------------------------------

class CrosshairOverlay:
    """Attaches a crosshair + value label to a ``pyqtgraph.PlotWidget``.

    On mouse move over the plot shows a vertical+horizontal infinite-line
    crosshair and a text label reading the data coordinates.  For each
    registered series the label shows the nearest-y value at the cursor x.

    Args:
        plot_widget: The ``PlotWidget`` to attach to.
        x_label: Name for the x axis in the readout (e.g. "t").
        y_label: Name for the y axis in the readout.
        x_fmt: Python format string for x values.
        y_fmt: Python format string for y values.
        threshold: Optional y threshold; values above it get a "⚠" marker.
        check_backflow: If ``True``, appends "backflow (γ < 0)" when any
            registered series is negative at the cursor x.
    """

    def __init__(
        self,
        plot_widget: pg.PlotWidget,
        x_label: str = "t",
        y_label: str = "value",
        x_fmt: str = "{:.2f}",
        y_fmt: str = "{:.4f}",
        threshold: float | None = None,
        check_backflow: bool = False,
    ) -> None:
        self._pw = plot_widget
        self._x_label = x_label
        self._y_label = y_label
        self._x_fmt = x_fmt
        self._y_fmt = y_fmt
        self._threshold = threshold
        self._check_backflow = check_backflow
        self._series: dict[str, pg.PlotDataItem] = {}

        dash_pen = pg.mkPen("#8B949E", width=0.5, style=Qt.PenStyle.DashLine)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=dash_pen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=dash_pen)
        self.label = pg.TextItem(
            anchor=(0, 1),
            color="#E6EDF3",
            fill=pg.mkBrush(22, 27, 34, 220),
        )

        for item in (self.vline, self.hline, self.label):
            plot_widget.addItem(item, ignoreBounds=True)
            item.setVisible(False)

        plot_widget.scene().sigMouseMoved.connect(self._on_move)

    def attach_series(self, name: str, curve: pg.PlotDataItem) -> None:
        """Register a named curve for per-series value readout.

        Args:
            name: Display name shown in the crosshair label.
            curve: The ``PlotDataItem`` whose data to read.
        """
        self._series[name] = curve

    def _on_move(self, scene_pos: object) -> None:
        if not self._pw.sceneBoundingRect().contains(scene_pos):
            self.set_visible(False)
            return

        vb = self._pw.getViewBox()
        mouse_point = vb.mapSceneToView(scene_pos)
        x = float(mouse_point.x())
        y = float(mouse_point.y())

        self.vline.setPos(x)
        self.hline.setPos(y)

        lines = [f"{self._x_label} = {self._x_fmt.format(x)}"]
        any_negative = False

        for name, curve in self._series.items():
            xs_arr, ys_arr = curve.getData()
            if xs_arr is not None and len(xs_arr) > 0:
                idx = int(np.argmin(np.abs(xs_arr - x)))
                y_val = float(ys_arr[idx])
                entry = f"{name}: {self._y_fmt.format(y_val)}"
                if self._threshold is not None and y_val > self._threshold:
                    entry += " ⚠"
                if y_val < 0:
                    any_negative = True
                lines.append(entry)

        if self._check_backflow and any_negative:
            lines.append("backflow (γ < 0)")

        self.label.setText("\n".join(lines))

        view_rect = vb.viewRect()
        off_x = view_rect.width() * 0.02
        off_y = view_rect.height() * 0.02
        label_x = x + off_x
        if label_x + view_rect.width() * 0.25 > view_rect.right():
            label_x = x - off_x
            self.label.setAnchor((1, 1))
        else:
            self.label.setAnchor((0, 1))
        self.label.setPos(label_x, y + off_y)

        self.set_visible(True)

    def set_visible(self, visible: bool) -> None:
        """Toggle crosshair and label visibility.

        Args:
            visible: ``True`` to show; ``False`` to hide.
        """
        self.vline.setVisible(visible)
        self.hline.setVisible(visible)
        self.label.setVisible(visible)


# ---------------------------------------------------------------------------
# Interaction helpers
# ---------------------------------------------------------------------------

def enable_interactions(
    plot_widget: pg.PlotWidget, auto_range_y: bool = True
) -> None:
    """Enable pan/zoom, right-click menu, and auto-range on a ``PlotWidget``.

    Sets the view box to PanMode (left-drag pans, scroll zooms), enables the
    pyqtgraph right-click context menu, and turns on auto-range.

    Args:
        plot_widget: The widget to configure.
        auto_range_y: If ``False``, only the X axis gets auto-range (useful
            for plots with fixed Y ranges like QBER 0–0.5).
    """
    vb = plot_widget.getViewBox()
    vb.setMouseMode(pg.ViewBox.PanMode)
    plot_widget.setMouseEnabled(x=True, y=True)
    plot_widget.setMenuEnabled(True)
    if auto_range_y:
        vb.enableAutoRange()
    else:
        vb.enableAutoRange(axis=pg.ViewBox.XAxis)


def add_interactive_legend(plot_widget: pg.PlotWidget) -> pg.LegendItem:
    """Add a pyqtgraph legend to ``plot_widget`` and return it.

    Curves added after this call with a ``name`` keyword argument are
    auto-registered in the legend.  The legend is draggable within the plot.

    Args:
        plot_widget: The widget to add the legend to.

    Returns:
        The ``LegendItem`` for further customisation.
    """
    return plot_widget.addLegend(offset=(10, 10))
