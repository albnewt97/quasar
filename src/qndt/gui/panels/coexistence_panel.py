"""CoexistencePanel: WDM classical channel coexistence configuration (§7.3).

Drives ``CoexistenceNoiseEngine`` (§3.3) with the list of active classical
channels and the Raman profile parameters (ρ_peak, T).
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from qndt.gui.panels.base_panel import BasePanel

_COLUMNS = ("Channel ID", "λ_c (nm)", "Power (mW)", "Active")
_MIN_ROWS = 5

# Calibrated ρ_peak from smf28_default() (displayed as default in spinbox)
_RHO_PEAK_DEFAULT: float = 1.19e-9   # 1/(km·nm)
_T_DEFAULT_K: float = 300.0          # K


class CoexistenceConfigModel(BaseModel):
    """Validated coexistence configuration payload.

    Args:
        channels: Active classical WDM channels.
        rho_peak: Raman cross-section peak ρ_peak in 1/(km·nm).
        temperature_k: Fiber temperature for Bose–Einstein factor in K.
    """

    channels: list[dict[str, object]]
    rho_peak: float = Field(gt=0.0, default=_RHO_PEAK_DEFAULT)
    temperature_k: float = Field(gt=0.0, default=_T_DEFAULT_K)


class CoexistencePanel(BasePanel):
    """WDM classical channel configuration panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("coexistence", parent)
        self._next_channel_index = 0

        self._layout.addWidget(self._make_section_label("Classical Channels"))

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.setMinimumHeight(150)
        self._layout.addWidget(self._table)
        self._table.itemChanged.connect(self._on_table_changed)

        button_row = QHBoxLayout()
        self._add_button = QPushButton("+ Add Channel")
        self._add_button.setObjectName("primary")
        self._add_button.clicked.connect(self.add_channel)
        self._remove_button = QPushButton("- Remove Selected")
        self._remove_button.setObjectName("danger")
        self._remove_button.clicked.connect(self.remove_selected)
        button_row.addWidget(self._add_button)
        button_row.addWidget(self._remove_button)
        self._layout.addLayout(button_row)

        # ── Raman profile parameters ─────────────────────────────────────
        # ρ(Δν) = ρ_peak · g(|Δν|) · A(Δν, T)  [whitepaper eq 16]
        self._layout.addWidget(self._make_section_label("Raman Profile (eq 16)"))

        self._rho_peak_spin = QDoubleSpinBox()
        self._rho_peak_spin.setDecimals(3)
        self._rho_peak_spin.setRange(1e-12, 1e-6)
        self._rho_peak_spin.setSingleStep(1e-11)
        self._rho_peak_spin.setValue(_RHO_PEAK_DEFAULT)
        self._rho_peak_spin.setToolTip(
            "ρ_peak: absolute scale of the silica Raman cross-section profile.\n"
            "Calibrated so β(1310→1550 nm) = 4×10⁻¹¹ 1/(km·nm) [Eraerds 2010]."
        )
        self._layout.addLayout(
            self._make_field_row("ρ_peak", self._rho_peak_spin, "1/(km·nm)")
        )

        self._temp_spin = QDoubleSpinBox()
        self._temp_spin.setDecimals(1)
        self._temp_spin.setRange(77.0, 1000.0)
        self._temp_spin.setSingleStep(10.0)
        self._temp_spin.setValue(_T_DEFAULT_K)
        self._temp_spin.setToolTip(
            "Fiber temperature T [K] used for the Bose–Einstein asymmetry factor\n"
            "A(Δν,T): Stokes ∝ (n+1), anti-Stokes ∝ n where n=1/(exp(hΔν/kT)−1)."
        )
        self._layout.addLayout(
            self._make_field_row("Temperature", self._temp_spin, "K")
        )

        # Read-only preview: computed ρ at the Eraerds calibration point
        self._rho_preview_label = QLabel("")
        self._layout.addWidget(self._rho_preview_label)

        self._rho_peak_spin.valueChanged.connect(self._on_profile_changed)
        self._temp_spin.valueChanged.connect(self._on_profile_changed)
        self._on_profile_changed()  # populate preview

        # ── Live Raman rate ───────────────────────────────────────────────
        self._layout.addWidget(self._make_section_label("Live Raman Rate"))
        self._rate_label = QLabel("--- Hz")
        self._layout.addWidget(self._rate_label)
        self._noise_bar = QProgressBar()
        self._noise_bar.setRange(0, 100)
        self._set_noise_bar_colour("#58A6FF")
        self._layout.addWidget(self._noise_bar)
        self._layout.addStretch(1)

        for _ in range(_MIN_ROWS):
            self.add_channel()

    def _config_model(self) -> type[CoexistenceConfigModel]:
        return CoexistenceConfigModel

    def add_channel(self) -> None:
        """Append a new classical channel row with default values."""
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.blockSignals(True)
        self._table.setItem(row, 0, QTableWidgetItem(f"ch_{self._next_channel_index}"))
        self._table.setItem(row, 1, QTableWidgetItem("1310.0"))
        self._table.setItem(row, 2, QTableWidgetItem("1.0"))
        active_item = QTableWidgetItem()
        active_item.setFlags(active_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        active_item.setCheckState(Qt.CheckState.Checked)
        self._table.setItem(row, 3, active_item)
        self._table.blockSignals(False)
        self._next_channel_index += 1
        self._on_table_changed(None)

    def remove_selected(self) -> None:
        """Remove the currently selected table row, if any."""
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)
            self._on_table_changed(None)

    def load_config(self, channels: list[dict[str, object]]) -> None:
        """Repopulate the channel table from a list of channel dicts.

        Args:
            channels: Each dict may have ``channel_id``, ``lambda_c_nm``,
                ``launch_power_mw``, and ``active`` keys.
        """
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        self._next_channel_index = 0
        for ch in channels:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(
                row, 0,
                QTableWidgetItem(str(ch.get("channel_id", f"ch_{self._next_channel_index}")))
            )
            self._table.setItem(row, 1, QTableWidgetItem(str(ch.get("lambda_c_nm", "1310.0"))))
            self._table.setItem(row, 2, QTableWidgetItem(str(ch.get("launch_power_mw", "1.0"))))
            active_item = QTableWidgetItem()
            active_item.setFlags(active_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            active_item.setCheckState(
                Qt.CheckState.Checked if ch.get("active", True) else Qt.CheckState.Unchecked
            )
            self._table.setItem(row, 3, active_item)
            self._next_channel_index += 1
        self._table.blockSignals(False)

    def get_channels(self) -> list[dict[str, object]]:
        """Return the active classical channels from the table.

        Returns:
            One dict per row whose "Active" checkbox is checked.
        """
        channels: list[dict[str, object]] = []
        for row in range(self._table.rowCount()):
            active_item = self._table.item(row, 3)
            if active_item is None or active_item.checkState() != Qt.CheckState.Checked:
                continue
            channel_id_item = self._table.item(row, 0)
            lambda_item = self._table.item(row, 1)
            power_item = self._table.item(row, 2)
            channels.append(
                {
                    "channel_id": channel_id_item.text() if channel_id_item else "",
                    "lambda_c_nm": float(lambda_item.text()) if lambda_item else 0.0,
                    "launch_power_mw": float(power_item.text()) if power_item else 0.0,
                    "active": True,
                }
            )
        return channels

    def update_raman_display(self, rate_hz: float) -> None:
        """Update the live Raman rate label and progress bar.

        Args:
            rate_hz: Current total Raman dark-count rate contribution [Hz].
        """
        self._rate_label.setText(f"{rate_hz:.2e} Hz")
        self._noise_bar.setValue(min(100, int(rate_hz / 1e6 * 100)))
        if rate_hz > 1e6:
            self._set_noise_bar_colour("#F85149")
        else:
            self._set_noise_bar_colour("#58A6FF")

    def _set_noise_bar_colour(self, colour: str) -> None:
        self._noise_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {colour}; }}"
        )

    def _on_profile_changed(self) -> None:
        """Update the ρ(Δν) preview label and emit config."""
        rho_peak = self._rho_peak_spin.value()
        self._rho_preview_label.setText(
            f"ρ(35.4 THz, 1310→1550 nm) = {rho_peak * 0.0341:.2e} 1/(km·nm)"
            "  [preview at calibration offset]"
        )
        self._emit_config()

    def _on_table_changed(self, _item: QTableWidgetItem | None) -> None:
        self._emit_config()

    def _emit_config(self) -> None:
        self.config_changed.emit("coexistence", self.get_config())

    def get_config(self) -> dict[str, object]:
        """Return the panel's current coexistence configuration."""
        return {
            "channels": self.get_channels(),
            "rho_peak": self._rho_peak_spin.value(),
            "temperature_k": self._temp_spin.value(),
        }
