"""DashboardWindow: tabbed container for all live simulation plots (§7.4).

Delegates each ``update_*`` call to the matching plot widget.  A shared
toolbar above the tabs provides pause/resume, reset-zoom, PNG export, and a
simulation-clock readout.  When paused the update methods are no-ops; the
simulation engine continues running but the GUI stops redrawing.
"""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from qndt.gui.dashboard.aging_plot import AgingPlot
from qndt.gui.dashboard.fidelity_plot import FidelityPlot
from qndt.gui.dashboard.key_rate_plot import KeyRatePlot
from qndt.gui.dashboard.network_heatmap import NetworkHeatmap
from qndt.gui.dashboard.nonmarkov_plot import NonMarkovPlot
from qndt.gui.dashboard.plot_utils import configure_pyqtgraph
from qndt.gui.dashboard.qber_plot import QBERPlot
from qndt.gui.dashboard.raman_plot import RamanPlot
from qndt.gui.topology.topology_model import TopologyModel


class DashboardWindow(QWidget):
    """Tabbed dashboard hosting all live simulation plots."""

    def __init__(self, topology_model: TopologyModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        configure_pyqtgraph()

        self._qber_plot = QBERPlot()
        self._fidelity_plot = FidelityPlot()
        self._raman_plot = RamanPlot()
        self._nonmarkov_plot = NonMarkovPlot()
        self._aging_plot = AgingPlot()
        self._heatmap = NetworkHeatmap(topology_model)
        self._key_rate_plot = KeyRatePlot()
        self._paused: bool = False

        toolbar = self._build_toolbar()
        self._tabs = QTabWidget()
        self._tabs.addTab(self._qber_plot, "QBER")
        self._tabs.addTab(self._fidelity_plot, "Fidelity")
        self._tabs.addTab(self._raman_plot, "Raman Noise")
        self._tabs.addTab(self._nonmarkov_plot, "Non-Markovian")
        self._tabs.addTab(self._aging_plot, "Device Aging")
        self._tabs.addTab(self._heatmap, "Network Map")
        self._tabs.addTab(self._key_rate_plot, "Key Rate")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self._tabs, stretch=1)

    # ------------------------------------------------------------------
    # Toolbar construction
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> QWidget:
        bar = QWidget(self)
        bar.setObjectName("canvas_toolbar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(6)

        self._pause_btn = QPushButton("Pause Updates")
        self._pause_btn.setCheckable(True)
        self._pause_btn.setObjectName("canvas_toolbar_btn")
        self._pause_btn.clicked.connect(self._on_pause_toggle)
        layout.addWidget(self._pause_btn)

        sep = QWidget(bar)
        sep.setObjectName("canvas_toolbar_sep")
        sep.setFixedWidth(1)
        layout.addWidget(sep)

        reset_btn = QPushButton("Reset Zoom")
        reset_btn.setObjectName("canvas_toolbar_btn")
        reset_btn.clicked.connect(self.reset_zoom)
        layout.addWidget(reset_btn)

        export_btn = QPushButton("Export PNG")
        export_btn.setObjectName("canvas_toolbar_btn")
        export_btn.clicked.connect(lambda: self.export_current_plot(""))
        layout.addWidget(export_btn)

        layout.addStretch()

        self._clock_label = QLabel("t = 0.000 s")
        self._clock_label.setObjectName("palette_header")
        layout.addWidget(self._clock_label)

        return bar

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------

    def _on_pause_toggle(self, checked: bool) -> None:
        self._paused = checked
        self._pause_btn.setText("Resume" if checked else "Pause Updates")

    def update_clock(self, t: float) -> None:
        """Update the simulation-time readout in the toolbar.

        Args:
            t: Simulation time [s].
        """
        self._clock_label.setText(f"t = {t:.3f} s")

    def reset_zoom(self) -> None:
        """Auto-range all ``PlotWidget`` children of the active tab."""
        current = self._tabs.currentWidget()
        if current is not None:
            for pw in current.findChildren(pg.PlotWidget):
                pw.autoRange()

    def export_current_plot(self, path: str = "") -> None:
        """Export the first plot in the active tab to a PNG file.

        If ``path`` is empty a ``QFileDialog`` is shown; if the user cancels,
        the method returns without doing anything.

        Args:
            path: Destination file path.  Pass a non-empty string from tests to
                bypass the dialog.
        """
        current = self._tabs.currentWidget()
        if current is None:
            return
        plot_widgets = current.findChildren(pg.PlotWidget)
        if not plot_widgets:
            return
        if not path:
            from PySide6.QtWidgets import QFileDialog

            path, _ = QFileDialog.getSaveFileName(self, "Export Plot", "", "PNG Files (*.png)")
            if not path:
                return
        from pyqtgraph.exporters import ImageExporter

        exporter = ImageExporter(plot_widgets[0].getPlotItem())
        exporter.export(path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def qber_plot(self) -> QBERPlot:
        """The QBER plot widget."""
        return self._qber_plot

    @property
    def fidelity_plot(self) -> FidelityPlot:
        """The fidelity plot widget."""
        return self._fidelity_plot

    @property
    def raman_plot(self) -> RamanPlot:
        """The Raman noise plot widget."""
        return self._raman_plot

    @property
    def nonmarkov_plot(self) -> NonMarkovPlot:
        """The non-Markovianity witness plot widget."""
        return self._nonmarkov_plot

    @property
    def aging_plot(self) -> AgingPlot:
        """The device aging plot widget."""
        return self._aging_plot

    @property
    def heatmap(self) -> NetworkHeatmap:
        """The network fidelity heatmap widget."""
        return self._heatmap

    @property
    def key_rate_plot(self) -> KeyRatePlot:
        """The BB84 key rate plot widget."""
        return self._key_rate_plot

    # ------------------------------------------------------------------
    # Update methods — all respect the pause flag
    # ------------------------------------------------------------------

    def update_qber(self, link_id: str, t: float, qber: float) -> None:
        """Forward a QBER sample to the QBER plot (no-op when paused)."""
        if self._paused:
            return
        self._qber_plot.update_sample(link_id, t, qber)

    def update_fidelity(self, link_id: str, t: float, fidelity: float) -> None:
        """Forward a fidelity sample to the fidelity plot (no-op when paused)."""
        if self._paused:
            return
        self._fidelity_plot.update_sample(link_id, t, fidelity)

    def update_raman(self, link_id: str, t: float, raman_rate: float, n_classical: int) -> None:
        """Forward a Raman noise sample to the Raman plot (no-op when paused)."""
        if self._paused:
            return
        self._raman_plot.update_sample(link_id, t, raman_rate, n_classical)

    def update_nonmarkov(
        self,
        link_id: str,
        t: float,
        rhp_value: float,
        gamma_x: float,
        gamma_y: float,
        gamma_z: float,
    ) -> None:
        """Forward an RHP witness sample to the non-Markovian plot (no-op when paused)."""
        if self._paused:
            return
        self._nonmarkov_plot.update_sample(link_id, t, rhp_value, gamma_x, gamma_y, gamma_z)

    def update_aging(
        self, node_id: str, op_count: int, t2_current: float, overrotation: float, t: float
    ) -> None:
        """Forward an aging sample to the device aging plot (no-op when paused)."""
        if self._paused:
            return
        self._aging_plot.update_sample(node_id, op_count, t2_current, overrotation, t)

    def update_heatmap(self, fidelity_map: dict[str, float]) -> None:
        """Forward a fidelity map to the network heatmap."""
        self._heatmap.update_fidelities(fidelity_map)

    def update_key_rate(
        self, link_id: str, t: float, qber: float, skr_bps: float
    ) -> None:
        """Forward a key rate sample to the key rate plot (no-op when paused).

        Args:
            link_id: Fiber link identifier.
            t: Simulation time [s].
            qber: Quantum bit error rate.
            skr_bps: Secret key rate [bits/sec].
        """
        if self._paused:
            return
        self._key_rate_plot.update_sample(link_id, t, qber, skr_bps)

    def add_link(self, link_id: str, colour: str | None = None) -> None:
        """Register a link with the QBER, fidelity, and key rate plots."""
        self._qber_plot.add_link(link_id, colour)
        self._fidelity_plot.add_link(link_id, colour)
        self._key_rate_plot.add_link(link_id, colour)

    def add_node(self, node_id: str, t2_nominal: float, wear_const_nc: float) -> None:
        """Register a node with the device aging plot."""
        self._aging_plot.add_node(node_id, t2_nominal, wear_const_nc)

    def clear_all(self) -> None:
        """Clear all plot data across every dashboard tab."""
        self._qber_plot.clear()
        self._fidelity_plot.clear()
        self._raman_plot.clear()
        self._nonmarkov_plot.clear()
        self._aging_plot.clear()
        self._key_rate_plot.clear()
