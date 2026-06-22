"""NetworkHeatmap: topology overlay coloured by live link fidelity (§7.4).

Wraps a read-only ``TopologyCanvas`` so the same visual representation
used in the editor doubles as a live fidelity heatmap, without letting
the dashboard accidentally mutate the topology.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from qndt.gui.topology.canvas import TopologyCanvas
from qndt.gui.topology.topology_model import TopologyModel

_LEGEND_WIDTH = 24
_LEGEND_HEIGHT = 160


def _build_legend_pixmap() -> QPixmap:
    pixmap = QPixmap(_LEGEND_WIDTH, _LEGEND_HEIGHT)
    gradient = QLinearGradient(0, 0, 0, _LEGEND_HEIGHT)
    gradient.setColorAt(0.0, QColor("#3FB950"))
    gradient.setColorAt(1.0, QColor("#F85149"))
    painter = QPainter(pixmap)
    painter.fillRect(pixmap.rect(), gradient)
    painter.end()
    return pixmap


class NetworkHeatmap(QWidget):
    """Read-only topology view with fidelity-mapped link/node colouring."""

    def __init__(self, topology_model: TopologyModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._canvas = TopologyCanvas(topology_model, self, read_only=True)

        legend_image = QLabel()
        legend_image.setPixmap(_build_legend_pixmap())

        legend_with_mid = QVBoxLayout()
        legend_with_mid.addWidget(QLabel("Fidelity"))
        legend_with_mid.addWidget(QLabel("1.0"))
        legend_with_mid.addWidget(legend_image)
        legend_with_mid.addWidget(QLabel("0.5"))
        legend_with_mid.addWidget(QLabel("0.0"))
        legend_with_mid.addStretch(1)

        self._export_button = QPushButton("Export PNG")
        self._export_button.clicked.connect(self._on_export_clicked)

        side_layout = QVBoxLayout()
        side_layout.addLayout(legend_with_mid)
        side_layout.addWidget(self._export_button)

        layout = QHBoxLayout(self)
        layout.addWidget(self._canvas, 1)
        layout.addLayout(side_layout)

    def update_fidelities(self, fidelity_map: dict[str, float]) -> None:
        """Push a batch of live fidelity readings to the canvas.

        Args:
            fidelity_map: Mapping of node or link id to fidelity in
                ``[0, 1]``.
        """
        for item_id, fidelity in fidelity_map.items():
            self._canvas.update_node_fidelity(item_id, fidelity)
            self._canvas.update_link_fidelity(item_id, fidelity)

    def export_png(self, path: str) -> None:
        """Export the current canvas view as a PNG image.

        Args:
            path: Destination file path.
        """
        self._canvas.grab().save(path)

    def _on_export_clicked(self) -> None:
        self.export_png("network_heatmap.png")
