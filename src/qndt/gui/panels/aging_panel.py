"""AgingPanel: device aging and calibration drift configuration (§7.3).

Drives ``DeviceAgingModel`` (§5.5): T2 Matthiessen wear curve, gate overrotation
drift, and the live per-node operational status table.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDoubleSpinBox, QPushButton, QTableWidget, QTableWidgetItem, QWidget

from qndt.gui.panels.base_panel import BasePanel

_STATUS_COLUMNS = ("Node ID", "Ops Count", "T₂ Current", "Status")
_GOOD_COLOUR = QColor("#3FB950")
_DEGRADED_COLOUR = QColor("#D29922")
_CRITICAL_COLOUR = QColor("#F85149")


class _ScientificDoubleSpinBox(QDoubleSpinBox):
    """A ``QDoubleSpinBox`` that displays its value in scientific notation."""

    def textFromValue(self, value: float) -> str:  # noqa: N802
        return f"{value:.3e}"

    def valueFromText(self, text: str) -> float:  # noqa: N802
        try:
            return float(text)
        except ValueError:
            return self.value()


class AgingConfigModel(BaseModel):
    """Validated device aging and calibration drift configuration.

    Args:
        t2_nominal: Initial T2 coherence time at zero duty cycle [s].
        wear_rate_kappa: Matthiessen wear rate κ [s⁻²]; 0 means no wear.
        drift_rate_kappa: Gate overrotation drift rate κ_drift [rad/s].
        epsilon_0: Initial gate overrotation [rad].
    """

    t2_nominal: float = Field(default=1.0, ge=1e-6, le=100.0)
    wear_rate_kappa: float = Field(default=1e-4, ge=0.0, le=10.0)
    drift_rate_kappa: float = Field(default=1e-6, ge=0.0, le=1e-3)
    epsilon_0: float = Field(default=0.0, ge=-0.1, le=0.1)


class AgingPanel(BasePanel):
    """Device aging and calibration drift configuration panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("aging", parent)

        self._layout.addWidget(self._make_section_label("Coherence Time"))

        self._t2_nominal = QDoubleSpinBox()
        self._t2_nominal.setRange(1e-6, 100.0)
        self._t2_nominal.setDecimals(6)
        self._t2_nominal.setValue(1.0)
        self._layout.addLayout(self._make_field_row("T₂ nominal", self._t2_nominal, "s"))

        self._wear_rate_kappa = _ScientificDoubleSpinBox()
        self._wear_rate_kappa.setRange(0.0, 10.0)
        self._wear_rate_kappa.setValue(1e-4)
        self._layout.addLayout(
            self._make_field_row("κ", self._wear_rate_kappa, "s⁻²")
        )

        self._layout.addWidget(self._make_section_label("Calibration Drift"))

        self._drift_rate = QDoubleSpinBox()
        self._drift_rate.setRange(0.0, 1e-3)
        self._drift_rate.setSingleStep(1e-7)
        self._drift_rate.setDecimals(9)
        self._drift_rate.setValue(1e-6)
        self._layout.addLayout(
            self._make_field_row("Drift rate κ", self._drift_rate, "rad/s")
        )

        self._epsilon_0 = QDoubleSpinBox()
        self._epsilon_0.setRange(-0.1, 0.1)
        self._epsilon_0.setSingleStep(1e-4)
        self._epsilon_0.setDecimals(6)
        self._epsilon_0.setValue(0.0)
        self._layout.addLayout(self._make_field_row("Initial ε₀", self._epsilon_0, "rad"))

        self._layout.addWidget(self._make_section_label("Live Node Status"))
        self._status_table = QTableWidget(0, len(_STATUS_COLUMNS))
        self._status_table.setHorizontalHeaderLabels(_STATUS_COLUMNS)
        self._status_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._layout.addWidget(self._status_table)

        self._apply_button = QPushButton("Apply Aging Config")
        self._apply_button.setObjectName("primary")
        self._apply_button.clicked.connect(self._on_apply_clicked)
        self._layout.addWidget(self._apply_button)

        self._reset_button = QPushButton("Reset Wear Counters")
        self._reset_button.setObjectName("danger")
        self._reset_button.clicked.connect(self._on_reset_clicked)
        self._layout.addWidget(self._reset_button)
        self._layout.addStretch(1)

        self._node_rows: dict[str, int] = {}

    def _config_model(self) -> type[AgingConfigModel]:
        return AgingConfigModel

    def update_node_status(self, node_id: str, op_count: int, t2_current: float) -> None:
        """Update (or create) a node's row in the live status table.

        Args:
            node_id: Node identifier.
            op_count: Cumulative operation count for this node.
            t2_current: Current T2 coherence time [s].
        """
        row = self._node_rows.get(node_id)
        if row is None:
            row = self._status_table.rowCount()
            self._status_table.insertRow(row)
            self._status_table.setItem(row, 0, QTableWidgetItem(node_id))
            self._node_rows[node_id] = row

        self._status_table.setItem(row, 1, QTableWidgetItem(str(op_count)))
        self._status_table.setItem(row, 2, QTableWidgetItem(f"{t2_current:.4f} s"))

        t2_nominal = self._t2_nominal.value()
        if t2_current > 0.5 * t2_nominal:
            status_text, colour = "Good", _GOOD_COLOUR
        elif t2_current > 0.1 * t2_nominal:
            status_text, colour = "Degraded", _DEGRADED_COLOUR
        else:
            status_text, colour = "Critical", _CRITICAL_COLOUR
        status_item = QTableWidgetItem(status_text)
        status_item.setForeground(colour)
        self._status_table.setItem(row, 3, status_item)

    def _on_apply_clicked(self) -> None:
        self._emit_if_valid(self.get_config())

    def _on_reset_clicked(self) -> None:
        self._status_table.setRowCount(0)
        self._node_rows.clear()

    def get_config(self) -> dict[str, object]:
        """Return the panel's current aging configuration."""
        return {
            "t2_nominal": self._t2_nominal.value(),
            "wear_rate_kappa": self._wear_rate_kappa.value(),
            "drift_rate_kappa": self._drift_rate.value(),
            "epsilon_0": self._epsilon_0.value(),
        }
