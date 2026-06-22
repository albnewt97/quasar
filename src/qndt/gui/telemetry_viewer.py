"""TelemetryViewer: live scrolling view of raw environmental telemetry (§7.5).

Shows the last ``WINDOW_S`` seconds of temperature, seismic acceleration,
and wind force as three stacked pyqtgraph plots, fed by ``ingest_sample``
from the simulation thread via Qt Signal/Slot (§4.2).
"""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from qndt.gui.dashboard.plot_utils import (
    QUASAR_PLOT_THEME,
    configure_pyqtgraph,
    make_pen,
    make_plot_widget,
    rolling_append,
)
from qndt.telemetry.sources import TelemetrySample

_STALE_GAP_S = 10.0
_STALE_BORDER_STYLE = f"border: 2px solid {QUASAR_PLOT_THEME['accent_amber']};"


class TelemetryViewer(QWidget):
    """Live scrolling view of the raw environmental telemetry stream."""

    WINDOW_S = 60.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        configure_pyqtgraph()

        self._paused = False
        self._xs: dict[str, list[float]] = {"temp": [], "seis": [], "wind": []}
        self._ys: dict[str, list[float]] = {"temp": [], "seis": [], "wind": []}
        self._last_t: float = 0.0

        self._temp_plot = make_plot_widget("Temperature", "Time (s)", "°C")
        self._temp_curve: pg.PlotDataItem = self._temp_plot.plot(
            [], [], pen=make_pen(QUASAR_PLOT_THEME["accent_cyan"])
        )

        self._seis_plot = make_plot_widget("Seismic", "Time (s)", "m/s²")
        self._seis_curve: pg.PlotDataItem = self._seis_plot.plot(
            [], [], pen=make_pen(QUASAR_PLOT_THEME["accent_purple"])
        )

        self._wind_plot = make_plot_widget("Wind", "Time (s)", "N")
        self._wind_curve: pg.PlotDataItem = self._wind_plot.plot(
            [], [], pen=make_pen(QUASAR_PLOT_THEME["accent_green"])
        )

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.addWidget(QLabel("SOURCE:"))
        self._source_label = QLabel("SYNTHETIC")
        toolbar_layout.addWidget(self._source_label)
        toolbar_layout.addWidget(QLabel("LINK:"))
        self._link_combo = QComboBox()
        toolbar_layout.addWidget(self._link_combo)
        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.clicked.connect(self._toggle_pause)
        toolbar_layout.addWidget(self._pause_btn)
        self._stale_label = QLabel("⚠ STALE DATA")
        self._stale_label.setStyleSheet(f"color: {QUASAR_PLOT_THEME['accent_amber']};")
        self._stale_label.setVisible(False)
        toolbar_layout.addWidget(self._stale_label)
        toolbar_layout.addStretch(1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._temp_plot)
        splitter.addWidget(self._seis_plot)
        splitter.addWidget(self._wind_plot)

        layout = QVBoxLayout(self)
        layout.addWidget(toolbar)
        layout.addWidget(splitter)

    def ingest_sample(self, sample: TelemetrySample) -> None:
        """Push one environmental sample into the rolling plots.

        Args:
            sample: Environmental sample whose ``E`` holds
                ``[temperature, seismic, wind]``.
        """
        if self._paused:
            return

        temp, seismic, wind = float(sample.E[0]), float(sample.E[1]), float(sample.E[2])
        rolling_append(self._xs["temp"], self._ys["temp"], sample.t, temp)
        rolling_append(self._xs["seis"], self._ys["seis"], sample.t, seismic)
        rolling_append(self._xs["wind"], self._ys["wind"], sample.t, wind)
        self._trim_window("temp")
        self._trim_window("seis")
        self._trim_window("wind")

        self._temp_curve.setData(self._xs["temp"], self._ys["temp"])
        self._seis_curve.setData(self._xs["seis"], self._ys["seis"])
        self._wind_curve.setData(self._xs["wind"], self._ys["wind"])

        is_stale = sample.t - self._last_t > _STALE_GAP_S
        self._stale_label.setVisible(is_stale)
        self._temp_plot.setStyleSheet(_STALE_BORDER_STYLE if is_stale else "")
        self._last_t = sample.t

    def _trim_window(self, key: str) -> None:
        """Evict samples older than ``WINDOW_S`` from a channel's buffers."""
        xs = self._xs[key]
        ys = self._ys[key]
        if not xs:
            return
        cutoff = xs[-1] - self.WINDOW_S
        while xs and xs[0] < cutoff:
            xs.pop(0)
            ys.pop(0)

    def set_source_label(self, source_type: str) -> None:
        """Update the toolbar's source-type indicator.

        Args:
            source_type: Human-readable source kind (e.g. "Synthetic").
        """
        self._source_label.setText(source_type.upper())

    def update_link_list(self, link_ids: list[str]) -> None:
        """Repopulate the link selector, preserving the selection if possible.

        Args:
            link_ids: Link identifiers to populate the combo box with.
        """
        current = self._link_combo.currentText()
        self._link_combo.blockSignals(True)
        self._link_combo.clear()
        self._link_combo.addItems(link_ids)
        if current in link_ids:
            self._link_combo.setCurrentText(current)
        self._link_combo.blockSignals(False)

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self._pause_btn.setText("▶ Resume" if self._paused else "⏸ Pause")

    def clear(self) -> None:
        """Clear all buffered samples and reset the plots and stale indicator."""
        for key in ("temp", "seis", "wind"):
            self._xs[key] = []
            self._ys[key] = []
        self._temp_curve.setData([], [])
        self._seis_curve.setData([], [])
        self._wind_curve.setData([], [])
        self._stale_label.setVisible(False)
        self._temp_plot.setStyleSheet("")
