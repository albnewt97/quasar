"""TelemetryPanel: telemetry source, sensitivity matrix, and kernel selection (§7.3).

Feeds ``EnvironmentalTelemetryEngine`` (§5.2, §5.3): source selection, the
3×3 sensitivity matrix S, and the memory kernel K(τ) parameterisation.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from qndt.gui.panels.base_panel import BasePanel

_SENSITIVITY_DEFAULTS = (
    (0.0, 0.001, 0.0005),
    (0.0, 0.001, 0.0),
    (0.002, 0.0, 0.0005),
)
_SENSITIVITY_ROW_LABELS = ("p_x", "p_y", "p_z")
_SENSITIVITY_COL_LABELS = ("T", "Seis", "Wind")


class TelemetryConfigModel(BaseModel):
    """Validated telemetry source + sensitivity + kernel configuration.

    Args:
        source_type: Selected telemetry source kind.
        sensitivity: 3×3 sensitivity matrix S (env axis → Pauli axis).
        kernel: Memory kernel selection and parameters.
        csv: CSV replay source parameters (when ``source_type == "csv"``).
        live: Live JSON stream source parameters.
        synthetic: Synthetic source generation parameters.
    """

    source_type: Literal["csv", "live_json", "synthetic"]
    sensitivity: list[list[float]]
    kernel: dict[str, object]
    csv: dict[str, object] | None = None
    live: dict[str, object] | None = None
    synthetic: dict[str, object] | None = None


class TelemetryPanel(BasePanel):
    """Telemetry source configuration panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("telemetry", parent)

        self._layout.addWidget(self._make_section_label("Data Source"))
        self._source_combo = QComboBox()
        self._source_combo.addItems(["CSV Replay", "Live JSON Stream", "Synthetic"])
        self._layout.addWidget(self._source_combo)

        self._source_stack = QStackedWidget()
        self._source_stack.addWidget(self._build_csv_form())
        self._source_stack.addWidget(self._build_live_form())
        self._source_stack.addWidget(self._build_synthetic_form())
        self._source_combo.currentIndexChanged.connect(self._source_stack.setCurrentIndex)
        self._layout.addWidget(self._source_stack)

        self._layout.addWidget(self._make_section_label("Sensitivity Matrix S (3×3)"))
        self._layout.addLayout(self._build_sensitivity_grid())

        self._layout.addWidget(self._make_section_label("Memory Kernel"))
        self._kernel_combo = QComboBox()
        self._kernel_combo.addItems(["Exponential", "Lorentzian", "Gaussian"])
        self._layout.addWidget(self._kernel_combo)

        self._kernel_stack = QStackedWidget()
        self._kernel_stack.addWidget(self._build_exponential_kernel_form())
        self._kernel_stack.addWidget(self._build_lorentzian_kernel_form())
        self._kernel_stack.addWidget(self._build_gaussian_kernel_form())
        self._kernel_combo.currentIndexChanged.connect(self._kernel_stack.setCurrentIndex)
        self._layout.addWidget(self._kernel_stack)

        self._apply_button = QPushButton("Apply Telemetry Config")
        self._apply_button.setObjectName("primary")
        self._apply_button.clicked.connect(self._on_apply_clicked)
        self._layout.addWidget(self._apply_button)
        self._layout.addStretch(1)

    # ------------------------------------------------------------------
    # Source forms
    # ------------------------------------------------------------------

    def _build_csv_form(self) -> QWidget:
        form = QWidget()
        layout = QVBoxLayout(form)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self._csv_path = QLineEdit()
        self._csv_path.setReadOnly(True)
        file_row.addWidget(self._csv_path, 1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.load_file_dialog)
        file_row.addWidget(browse_button)
        layout.addLayout(file_row)

        self._csv_speedup = QDoubleSpinBox()
        self._csv_speedup.setRange(0.1, 1000.0)
        self._csv_speedup.setValue(1.0)
        layout.addLayout(self._make_field_row("Speedup:", self._csv_speedup))

        self._csv_t_column = QSpinBox()
        self._csv_t_column.setRange(0, 20)
        self._csv_t_column.setValue(0)
        layout.addLayout(self._make_field_row("t column:", self._csv_t_column))

        self._csv_env_columns = QLineEdit("1,2,3")
        layout.addLayout(self._make_field_row("Env columns:", self._csv_env_columns))

        layout.addStretch(1)
        return form

    def _build_live_form(self) -> QWidget:
        form = QWidget()
        layout = QVBoxLayout(form)

        self._live_url = QLineEdit("http://localhost:8080/telemetry")
        layout.addLayout(self._make_field_row("URL:", self._live_url))

        self._live_poll_hz = QDoubleSpinBox()
        self._live_poll_hz.setRange(0.1, 100.0)
        self._live_poll_hz.setValue(10.0)
        layout.addLayout(self._make_field_row("Poll (Hz):", self._live_poll_hz))

        layout.addWidget(QLabel("Field map:"))
        self._live_field_map = QPlainTextEdit(
            '{"temperature": "sensors.temp", "seismic": "sensors.seis"}'
        )
        self._live_field_map.setFixedHeight(72)
        layout.addWidget(self._live_field_map)

        layout.addStretch(1)
        return form

    def _build_synthetic_form(self) -> QWidget:
        form = QWidget()
        layout = QVBoxLayout(form)

        self._synth_duration = QDoubleSpinBox()
        self._synth_duration.setRange(1.0, 86400.0)
        self._synth_duration.setValue(3600.0)
        layout.addLayout(self._make_field_row("Duration (s):", self._synth_duration))

        self._synth_temp_mean = QDoubleSpinBox()
        self._synth_temp_mean.setRange(-40.0, 80.0)
        self._synth_temp_mean.setValue(20.0)
        layout.addLayout(self._make_field_row("Temp mean (°C):", self._synth_temp_mean))

        self._synth_temp_amplitude = QDoubleSpinBox()
        self._synth_temp_amplitude.setRange(0.0, 30.0)
        self._synth_temp_amplitude.setValue(5.0)
        layout.addLayout(
            self._make_field_row("Temp amplitude:", self._synth_temp_amplitude)
        )

        self._synth_seismic_noise = QDoubleSpinBox()
        self._synth_seismic_noise.setRange(0.0, 1.0)
        self._synth_seismic_noise.setDecimals(4)
        self._synth_seismic_noise.setValue(0.001)
        layout.addLayout(
            self._make_field_row("Seismic noise:", self._synth_seismic_noise)
        )

        self._synth_seed = QSpinBox()
        self._synth_seed.setRange(0, 99999)
        self._synth_seed.setValue(42)
        layout.addLayout(self._make_field_row("Seed:", self._synth_seed))

        layout.addStretch(1)
        return form

    # ------------------------------------------------------------------
    # Sensitivity matrix
    # ------------------------------------------------------------------

    def _build_sensitivity_grid(self) -> QGridLayout:
        grid = QGridLayout()
        for col, label in enumerate(_SENSITIVITY_COL_LABELS):
            grid.addWidget(QLabel(label), 0, col + 1)

        self._sensitivity: list[list[QDoubleSpinBox]] = []
        for row, row_label in enumerate(_SENSITIVITY_ROW_LABELS):
            grid.addWidget(QLabel(row_label), row + 1, 0)
            spin_row: list[QDoubleSpinBox] = []
            for col in range(3):
                spinbox = QDoubleSpinBox()
                spinbox.setRange(-10.0, 10.0)
                spinbox.setSingleStep(0.001)
                spinbox.setDecimals(4)
                spinbox.setValue(_SENSITIVITY_DEFAULTS[row][col])
                grid.addWidget(spinbox, row + 1, col + 1)
                spin_row.append(spinbox)
            self._sensitivity.append(spin_row)
        return grid

    # ------------------------------------------------------------------
    # Kernel forms
    # ------------------------------------------------------------------

    def _build_exponential_kernel_form(self) -> QWidget:
        form = QWidget()
        layout = QVBoxLayout(form)
        self._tau_x = QDoubleSpinBox()
        self._tau_x.setRange(0.1, 3600.0)
        self._tau_x.setValue(30.0)
        layout.addLayout(self._make_field_row("τ_x", self._tau_x, "s"))
        self._tau_y = QDoubleSpinBox()
        self._tau_y.setRange(0.1, 3600.0)
        self._tau_y.setValue(30.0)
        layout.addLayout(self._make_field_row("τ_y", self._tau_y, "s"))
        self._tau_z = QDoubleSpinBox()
        self._tau_z.setRange(0.1, 3600.0)
        self._tau_z.setValue(120.0)
        layout.addLayout(self._make_field_row("τ_z", self._tau_z, "s"))
        layout.addStretch(1)
        return form

    def _build_lorentzian_kernel_form(self) -> QWidget:
        form = QWidget()
        layout = QVBoxLayout(form)
        self._gamma = QDoubleSpinBox()
        self._gamma.setRange(0.001, 100.0)
        self._gamma.setDecimals(3)
        self._gamma.setValue(0.1)
        layout.addLayout(self._make_field_row("γ", self._gamma, "Hz"))
        self._omega_0 = QDoubleSpinBox()
        self._omega_0.setRange(0.001, 100.0)
        self._omega_0.setDecimals(3)
        self._omega_0.setValue(1.0)
        layout.addLayout(self._make_field_row("ω₀", self._omega_0, "Hz"))
        layout.addStretch(1)
        return form

    def _build_gaussian_kernel_form(self) -> QWidget:
        form = QWidget()
        layout = QVBoxLayout(form)
        self._sigma = QDoubleSpinBox()
        self._sigma.setRange(0.1, 3600.0)
        self._sigma.setValue(10.0)
        layout.addLayout(self._make_field_row("σ", self._sigma, "s"))
        self._gaussian_amplitude = QDoubleSpinBox()
        self._gaussian_amplitude.setRange(0.0, 10.0)
        self._gaussian_amplitude.setValue(1.0)
        layout.addLayout(self._make_field_row("Amplitude", self._gaussian_amplitude))
        layout.addStretch(1)
        return form

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------

    def _config_model(self) -> type[TelemetryConfigModel]:
        return TelemetryConfigModel

    def load_file_dialog(self) -> None:
        """Open a file dialog to select a CSV telemetry replay file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV Telemetry File", "", "CSV Files (*.csv)"
        )
        if path:
            self._csv_path.setText(path)

    def _on_apply_clicked(self) -> None:
        self._emit_if_valid(self.get_config())

    def load_config(self, config: dict[str, object]) -> None:
        """Populate panel widgets from a loaded scenario config.

        Args:
            config: Dict with optional keys ``"kernel"`` (dict) and
                ``"sensitivity"`` (list[list[float]] or None).
                Missing or ``None`` values are skipped.
        """
        kernel = config.get("kernel")
        if isinstance(kernel, dict):
            ktype = str(kernel.get("type", "exponential"))
            index = {"exponential": 0, "lorentzian": 1, "gaussian": 2}.get(ktype, 0)
            self._kernel_combo.setCurrentIndex(index)
            self._kernel_stack.setCurrentIndex(index)
            if ktype == "exponential":
                if "tau_x" in kernel:
                    self._tau_x.setValue(float(kernel["tau_x"]))
                if "tau_y" in kernel:
                    self._tau_y.setValue(float(kernel["tau_y"]))
                if "tau_z" in kernel:
                    self._tau_z.setValue(float(kernel["tau_z"]))
            elif ktype == "lorentzian":
                if "gamma" in kernel:
                    self._gamma.setValue(float(kernel["gamma"]))
                if "omega_0" in kernel:
                    self._omega_0.setValue(float(kernel["omega_0"]))
            elif ktype == "gaussian":
                if "sigma" in kernel:
                    self._sigma.setValue(float(kernel["sigma"]))

        sensitivity = config.get("sensitivity")
        if isinstance(sensitivity, list) and len(sensitivity) == 3:
            for row_idx, row in enumerate(sensitivity):
                if not isinstance(row, list):
                    continue
                for col_idx, val in enumerate(row):
                    if col_idx < len(self._sensitivity[row_idx]):
                        try:
                            self._sensitivity[row_idx][col_idx].setValue(float(val))
                        except (TypeError, ValueError):
                            pass

    def get_config(self) -> dict[str, object]:
        """Return the full telemetry configuration dict."""
        source_index = self._source_combo.currentIndex()
        source_type: Literal["csv", "live_json", "synthetic"] = (
            "csv" if source_index == 0 else "live_json" if source_index == 1 else "synthetic"
        )

        try:
            env_columns = [
                int(token) for token in self._csv_env_columns.text().split(",") if token.strip()
            ]
        except ValueError:
            env_columns = []
        csv_config: dict[str, object] = {
            "path": self._csv_path.text(),
            "speedup": self._csv_speedup.value(),
            "t_column": self._csv_t_column.value(),
            "env_columns": env_columns,
        }

        try:
            field_map = json.loads(self._live_field_map.toPlainText())
        except json.JSONDecodeError:
            field_map = {}
        live_config: dict[str, object] = {
            "url": self._live_url.text(),
            "poll_hz": self._live_poll_hz.value(),
            "field_map": field_map,
        }

        synthetic_config: dict[str, object] = {
            "duration_s": self._synth_duration.value(),
            "temp_mean_c": self._synth_temp_mean.value(),
            "temp_amplitude": self._synth_temp_amplitude.value(),
            "seismic_noise": self._synth_seismic_noise.value(),
            "seed": self._synth_seed.value(),
        }

        sensitivity = [
            [spinbox.value() for spinbox in row] for row in self._sensitivity
        ]

        kernel_index = self._kernel_combo.currentIndex()
        if kernel_index == 0:
            kernel: dict[str, object] = {
                "type": "exponential",
                "tau_x": self._tau_x.value(),
                "tau_y": self._tau_y.value(),
                "tau_z": self._tau_z.value(),
            }
        elif kernel_index == 1:
            kernel = {
                "type": "lorentzian",
                "gamma": self._gamma.value(),
                "omega_0": self._omega_0.value(),
            }
        else:
            kernel = {
                "type": "gaussian",
                "sigma": self._sigma.value(),
                "amplitude": self._gaussian_amplitude.value(),
            }

        return {
            "source_type": source_type,
            "csv": csv_config,
            "live": live_config,
            "synthetic": synthetic_config,
            "sensitivity": sensitivity,
            "kernel": kernel,
        }
