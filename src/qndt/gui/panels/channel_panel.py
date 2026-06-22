"""ChannelPanel: per-link fiber channel configuration (§7.3).

Validates against ``ChannelConfigModel`` before emitting ``config_changed``.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from PySide6.QtWidgets import QDoubleSpinBox, QLabel, QPushButton, QSpinBox, QWidget

from qndt.gui.panels.base_panel import BasePanel
from qndt.physics.key_rate import KeyRateResult


class ChannelConfigModel(BaseModel):
    """Validated per-link fiber channel parameters (mirrors §9.3 ``FiberParamsModel``).

    Args:
        lambda_q_nm: Quantum channel wavelength in nm.
        length_km: Fiber span length in km.
        attenuation_db_per_km: Power loss coefficient in dB/km.
        eta_detector: Single-photon detector efficiency in (0, 1].
        t_opt: Optical transmission of filter + coupling in (0, 1].
        p_dc: Intrinsic dark-count probability per gate in (0, 1].
        gate_width_s: Quantum gate duration in seconds.
        duration_s: Total simulation duration [s].
        dt_s: Simulation time step [s].
        chi_max: Maximum MPDO bond dimension for TensorStateTracker.
    """

    lambda_q_nm: float = Field(default=1550.0, ge=1200.0, le=1700.0)
    length_km: float = Field(default=25.0, ge=0.1, le=1000.0)
    attenuation_db_per_km: float = Field(default=0.2, ge=0.01, le=2.0)
    eta_detector: float = Field(default=0.8, ge=0.01, le=1.0)
    t_opt: float = Field(default=0.5, ge=0.01, le=1.0)
    p_dc: float = Field(default=1e-5, ge=1e-8, le=0.1)
    gate_width_s: float = Field(default=1e-9, ge=1e-12, le=1e-6)
    duration_s: float = Field(default=10.0, gt=0.0, le=86400.0)
    dt_s: float = Field(default=0.1, gt=0.0, le=10.0)
    chi_max: int = Field(default=4, ge=1, le=256)


class ChannelPanel(BasePanel):
    """Per-link fiber channel configuration panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("channel", parent)

        self._layout.addWidget(self._make_section_label("Fiber Channel"))

        self._lambda_q = QDoubleSpinBox()
        self._lambda_q.setRange(1200.0, 1700.0)
        self._lambda_q.setSingleStep(1.0)
        self._lambda_q.setDecimals(1)
        self._lambda_q.setValue(1550.0)
        self._layout.addLayout(self._make_field_row("Quantum λ", self._lambda_q, "nm"))

        self._length_km = QDoubleSpinBox()
        self._length_km.setRange(0.1, 1000.0)
        self._length_km.setSingleStep(1.0)
        self._length_km.setDecimals(1)
        self._length_km.setValue(25.0)
        self._layout.addLayout(
            self._make_field_row("Fiber Length", self._length_km, "km")
        )

        self._attenuation = QDoubleSpinBox()
        self._attenuation.setRange(0.01, 2.0)
        self._attenuation.setSingleStep(0.01)
        self._attenuation.setDecimals(3)
        self._attenuation.setValue(0.2)
        self._layout.addLayout(
            self._make_field_row("Attenuation", self._attenuation, "dB/km")
        )

        self._eta_detector = QDoubleSpinBox()
        self._eta_detector.setRange(0.01, 1.0)
        self._eta_detector.setSingleStep(0.01)
        self._eta_detector.setDecimals(3)
        self._eta_detector.setValue(0.8)
        self._layout.addLayout(
            self._make_field_row("Detector η", self._eta_detector, "")
        )

        self._t_opt = QDoubleSpinBox()
        self._t_opt.setRange(0.01, 1.0)
        self._t_opt.setSingleStep(0.01)
        self._t_opt.setDecimals(3)
        self._t_opt.setValue(0.5)
        self._layout.addLayout(self._make_field_row("Filter T_opt", self._t_opt, ""))

        self._p_dc = QDoubleSpinBox()
        self._p_dc.setRange(1e-8, 0.1)
        self._p_dc.setSingleStep(1e-6)
        self._p_dc.setDecimals(8)
        self._p_dc.setValue(1e-5)
        self._layout.addLayout(self._make_field_row("Dark Count p", self._p_dc, ""))

        self._gate_width = QDoubleSpinBox()
        self._gate_width.setRange(1e-12, 1e-6)
        self._gate_width.setSingleStep(1e-10)
        self._gate_width.setDecimals(12)
        self._gate_width.setValue(1e-9)
        self._layout.addLayout(self._make_field_row("Gate Width", self._gate_width, "s"))

        self._layout.addWidget(self._make_section_label("Simulation"))

        self._duration_s = QDoubleSpinBox()
        self._duration_s.setRange(0.1, 86400.0)
        self._duration_s.setSingleStep(1.0)
        self._duration_s.setDecimals(2)
        self._duration_s.setValue(10.0)
        self._layout.addLayout(self._make_field_row("Duration", self._duration_s, "s"))

        self._dt_s = QDoubleSpinBox()
        self._dt_s.setRange(1e-4, 10.0)
        self._dt_s.setSingleStep(0.01)
        self._dt_s.setDecimals(4)
        self._dt_s.setValue(0.1)
        self._layout.addLayout(self._make_field_row("Time step Δt", self._dt_s, "s"))

        self._chi_max = QSpinBox()
        self._chi_max.setRange(1, 256)
        self._chi_max.setValue(4)
        self._layout.addLayout(self._make_field_row("χ_max", self._chi_max, ""))

        for spinbox in (
            self._lambda_q,
            self._length_km,
            self._attenuation,
            self._eta_detector,
            self._t_opt,
            self._p_dc,
            self._gate_width,
            self._duration_s,
            self._dt_s,
        ):
            spinbox.valueChanged.connect(self._on_value_changed)
        self._chi_max.valueChanged.connect(self._on_value_changed)

        self._apply_button = QPushButton("Apply to Selected Link")
        self._apply_button.setObjectName("primary")
        self._apply_button.clicked.connect(self._on_apply_clicked)
        self._layout.addWidget(self._apply_button)

        self._layout.addWidget(self._make_section_label("ESTIMATED KEY RATE"))

        self._skr_label = QLabel("--- bps")
        self._layout.addWidget(self._skr_label)

        self._margin_label = QLabel("Margin: ---")
        self._layout.addWidget(self._margin_label)

        self._threshold_label = QLabel("Threshold QBER: ---")
        self._layout.addWidget(self._threshold_label)

        self._layout.addStretch(1)

    def _config_model(self) -> type[ChannelConfigModel]:
        return ChannelConfigModel

    def _on_value_changed(self, _value: float | int) -> None:
        self._dirty = True

    def _on_apply_clicked(self) -> None:
        self._emit_if_valid(self.get_config())

    def get_config(self) -> dict[str, object]:
        """Return the panel's current fiber channel and simulation values."""
        return {
            "lambda_q_nm": self._lambda_q.value(),
            "length_km": self._length_km.value(),
            "attenuation_db_per_km": self._attenuation.value(),
            "eta_detector": self._eta_detector.value(),
            "t_opt": self._t_opt.value(),
            "p_dc": self._p_dc.value(),
            "gate_width_s": self._gate_width.value(),
            "duration_s": self._duration_s.value(),
            "dt_s": self._dt_s.value(),
            "chi_max": self._chi_max.value(),
        }

    def update_key_rate_display(self, result: KeyRateResult) -> None:
        """Refresh the key rate section from a live ``KeyRateResult``.

        Args:
            result: Freshly computed key rate result from ``BB84KeyRateCalculator``.
        """
        self._skr_label.setText(f"{result.secret_key_rate_bps:.3e} bps")
        margin_text = f"Margin: {result.security_margin:+.4f}"
        colour = "#3FB950" if result.security_margin > 0 else "#F85149"
        self._margin_label.setText(margin_text)
        self._margin_label.setStyleSheet(f"color: {colour};")
        self._threshold_label.setText(f"Threshold QBER: {result.qber_threshold:.4f}")

    def load_config(self, config: dict[str, float | int]) -> None:
        """Populate spinbox values from a config dict (e.g. on link selection).

        Args:
            config: Mapping of field name to value; missing keys are skipped.
        """
        float_widgets: dict[str, QDoubleSpinBox] = {
            "lambda_q_nm": self._lambda_q,
            "length_km": self._length_km,
            "attenuation_db_per_km": self._attenuation,
            "eta_detector": self._eta_detector,
            "t_opt": self._t_opt,
            "p_dc": self._p_dc,
            "gate_width_s": self._gate_width,
            "duration_s": self._duration_s,
            "dt_s": self._dt_s,
        }
        for key, widget in float_widgets.items():
            if key in config:
                widget.setValue(float(config[key]))
        if "chi_max" in config:
            self._chi_max.setValue(int(config["chi_max"]))
