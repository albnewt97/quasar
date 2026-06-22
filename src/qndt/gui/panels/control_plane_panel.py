"""ControlPlanePanel: classical control plane configuration (§7.3).

Drives ``AsynchronousControlPlane`` (§3.3): jitter model, routing policy,
and the live per-link traffic table.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from qndt.gui.panels.base_panel import BasePanel

_TRAFFIC_COLUMNS = ("Link", "Utilisation", "Induced Idle")
_GOOD_COLOUR = QColor("#3FB950")
_DEGRADED_COLOUR = QColor("#D29922")
_CRITICAL_COLOUR = QColor("#F85149")


class ControlPlaneConfigModel(BaseModel):
    """Validated classical control plane configuration.

    Args:
        base_latency_s: Fixed propagation/processing latency [s].
        jitter_std_s: Standard deviation of latency jitter [s].
        congestion_multiplier: Latency multiplier under congestion.
        max_retries: Maximum retransmission attempts.
        loop_detection: Whether routing-loop detection is enabled.
    """

    base_latency_s: float = Field(default=1e-3, ge=1e-6, le=1.0)
    jitter_std_s: float = Field(default=1e-4, ge=0.0, le=0.1)
    congestion_multiplier: float = Field(default=2.0, ge=1.0, le=20.0)
    max_retries: int = Field(default=5, ge=1, le=20)
    loop_detection: bool = True


class ControlPlanePanel(BasePanel):
    """Classical control plane configuration panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("control_plane", parent)

        self._layout.addWidget(self._make_section_label("Jitter Model"))

        self._base_latency = QDoubleSpinBox()
        self._base_latency.setRange(1e-6, 1.0)
        self._base_latency.setSingleStep(1e-4)
        self._base_latency.setDecimals(6)
        self._base_latency.setValue(1e-3)
        self._layout.addLayout(
            self._make_field_row("Base latency", self._base_latency, "s")
        )

        self._jitter_std = QDoubleSpinBox()
        self._jitter_std.setRange(0.0, 0.1)
        self._jitter_std.setSingleStep(1e-5)
        self._jitter_std.setDecimals(6)
        self._jitter_std.setValue(1e-4)
        self._layout.addLayout(self._make_field_row("Jitter std", self._jitter_std, "s"))

        self._congestion_multiplier = QDoubleSpinBox()
        self._congestion_multiplier.setRange(1.0, 20.0)
        self._congestion_multiplier.setSingleStep(0.1)
        self._congestion_multiplier.setDecimals(1)
        self._congestion_multiplier.setValue(2.0)
        self._layout.addLayout(
            self._make_field_row("Congestion ×", self._congestion_multiplier, "×")
        )

        self._layout.addWidget(self._make_section_label("Routing"))

        self._max_retries = QSpinBox()
        self._max_retries.setRange(1, 20)
        self._max_retries.setValue(5)
        self._layout.addLayout(self._make_field_row("Max retries", self._max_retries))

        self._loop_detection = QCheckBox("Loop detection")
        self._loop_detection.setChecked(True)
        self._layout.addWidget(self._loop_detection)

        self._layout.addWidget(self._make_section_label("Live Traffic"))
        self._traffic_table = QTableWidget(0, len(_TRAFFIC_COLUMNS))
        self._traffic_table.setHorizontalHeaderLabels(_TRAFFIC_COLUMNS)
        self._traffic_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._layout.addWidget(self._traffic_table)

        self._apply_button = QPushButton("Apply Control Plane Config")
        self._apply_button.setObjectName("primary")
        self._apply_button.clicked.connect(self._on_apply_clicked)
        self._layout.addWidget(self._apply_button)
        self._layout.addStretch(1)

        self._link_rows: dict[str, int] = {}

    def _config_model(self) -> type[ControlPlaneConfigModel]:
        return ControlPlaneConfigModel

    def update_link_traffic(
        self, link_id: str, utilisation: float, induced_idle: float
    ) -> None:
        """Update (or create) a link's row in the live traffic table.

        Args:
            link_id: Link identifier.
            utilisation: Fraction of classical capacity in use, in ``[0, 1]``.
            induced_idle: Quantum idle time induced by congestion [s].
        """
        row = self._link_rows.get(link_id)
        if row is None:
            row = self._traffic_table.rowCount()
            self._traffic_table.insertRow(row)
            self._traffic_table.setItem(row, 0, QTableWidgetItem(link_id))
            self._link_rows[link_id] = row

        if utilisation < 0.5:
            colour = _GOOD_COLOUR
        elif utilisation <= 0.8:
            colour = _DEGRADED_COLOUR
        else:
            colour = _CRITICAL_COLOUR
        utilisation_item = QTableWidgetItem(f"{utilisation * 100.0:.1f}%")
        utilisation_item.setForeground(colour)
        self._traffic_table.setItem(row, 1, utilisation_item)

        self._traffic_table.setItem(
            row, 2, QTableWidgetItem(f"{induced_idle * 1000.0:.2f} ms")
        )

    def _on_apply_clicked(self) -> None:
        self._emit_if_valid(self.get_config())

    def get_config(self) -> dict[str, object]:
        """Return the panel's current control plane configuration."""
        return {
            "base_latency_s": self._base_latency.value(),
            "jitter_std_s": self._jitter_std.value(),
            "congestion_multiplier": self._congestion_multiplier.value(),
            "max_retries": self._max_retries.value(),
            "loop_detection": self._loop_detection.isChecked(),
        }
