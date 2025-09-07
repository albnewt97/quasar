# configs/schema.py
"""
Typed configuration schema for QUASAR
====================================

This module defines production-grade, Pydantic v2 models for validating and
loading QUASAR simulation configurations from YAML files.

Design goals
------------
- Strict, explicit fields with validation and helpful error messages.
- Stable public API (`Config`) used by CLI, dashboards, and scenarios.
- Minimal coupling: schema is independent from runtime modules; it encodes
  only data, not behavior.

Conventions
-----------
- Times in seconds (float), pulse rates in Hertz (int), lengths handled in
  scenario/physics layers (not here).
- Output paths are strings; callers resolve them to Path objects.

Quick start
-----------
from configs.schema import Config, load_config

cfg = load_config("configs/scenario1_equidistant.yaml")
print(cfg.scenario.name, cfg.scenario.pulse_rate_hz)

Advanced
--------
- `weather` supports either a **preset name** (e.g., "clear", "fog") or
  an **inline** object with numeric fields. Resolution to runtime presets
  is performed in the scenario layer.
"""

from __future__ import annotations

from typing import Literal, Optional, Union, Any, Dict
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, ValidationError


# -----------------------------------------------------------------------------
# Weather (inline or preset reference)
# -----------------------------------------------------------------------------
class WeatherInline(BaseModel):
    """
    Inline weather configuration for free-space channels.

    Attributes
    ----------
    name : str | None
        Optional label for this inline profile (e.g., "custom").
    attenuation_db_per_km : float >= 0
        Atmospheric attenuation per km (dB/km).
    cn2 : float > 0
        Turbulence structure parameter (m^-2/3).
    pointing_sigma_urad : float > 0
        Pointing jitter (1-sigma) in microradians.
    """
    name: Optional[str] = None
    attenuation_db_per_km: float = Field(..., ge=0.0)
    cn2: float = Field(..., gt=0.0)
    pointing_sigma_urad: float = Field(..., gt=0.0)


WeatherConfig = Union[str, WeatherInline]  # preset key or inline definition


# -----------------------------------------------------------------------------
# Device configuration (detectors/BSM)
# -----------------------------------------------------------------------------
class Devices(BaseModel):
    """
    Device parameterization (detectors & BSM).

    These map directly to detector and BSM model parameters used in scenarios.
    """
    detector_eta: float = Field(0.8, ge=0.0, le=1.0, description="Detector quantum efficiency")
    detector_dark_per_gate: float = Field(1e-6, ge=0.0, description="Dark count probability per gate")
    detector_dead_time_ns: float = Field(60.0, ge=0.0, description="Detector dead time (ns)")
    detector_afterpulse: float = Field(0.02, ge=0.0, le=1.0, description="Afterpulsing probability per detection")
    bsm_visibility: float = Field(0.98, ge=0.0, le=1.0, description="Interference visibility at BSM")
    coincidence_window_ps: int = Field(500, ge=50, le=5000, description="Coincidence window width (ps)")


# -----------------------------------------------------------------------------
# Scenario configuration
# -----------------------------------------------------------------------------
ScenarioName = Literal[
    "scenario1_equidistant",
    "scenario1_uneven",
    "scenario2_moving",
    "scenario3_city",
    "scenario4_uk_opt",
]


class Scenario(BaseModel):
    """
    Scenario control parameters.

    Attributes
    ----------
    name : ScenarioName
        Which scenario implementation to run.
    pulse_rate_hz : int
        Source pulse repetition rate before channel/device reductions.
    duration_s : float
        Total wall-clock simulation horizon in seconds.
    output_dir : str
        Directory where results will be written.
    """
    name: ScenarioName
    pulse_rate_hz: int = Field(50_000_000, ge=1_000, le=2_000_000_000)
    duration_s: float = Field(1.0, gt=0.0)
    output_dir: str = Field("data/runs/latest", min_length=1)


# -----------------------------------------------------------------------------
# Root configuration
# -----------------------------------------------------------------------------
class Config(BaseModel):
    """
    Root configuration for a QUASAR run.

    Sections
    --------
    scenario : Scenario
        Core run parameters.
    devices : Devices
        Detector/BSM parameters.
    weather : WeatherConfig | None
        Optional free-space weather profile. May be:
          - a preset name (e.g., "clear", "fog"), or
          - an inline object (attenuation_db_per_km/cn2/pointing_sigma_urad).
        Only used by scenarios requiring free-space channels.
    """
    scenario: Scenario
    devices: Devices
    weather: Optional[WeatherConfig] = None

    @field_validator("weather", mode="before")
    @classmethod
    def _coerce_weather(cls, v: Any) -> Any:
        """
        Accept either a string (preset key) or dict-like for inline definition.
        """
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, dict):
            # Validate as WeatherInline later by Pydantic union.
            return v
        raise TypeError("weather must be a preset name (str), an object with "
                        "attenuation_db_per_km/cn2/pointing_sigma_urad, or null.")


# -----------------------------------------------------------------------------
# YAML helpers (production-safe)
# -----------------------------------------------------------------------------
def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("Top-level YAML must be a mapping/object.")
            return data
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e


def load_config(path: str | Path) -> Config:
    """
    Load and validate a Config from a YAML file.

    Parameters
    ----------
    path : str | Path
        Path to YAML configuration.

    Returns
    -------
    Config
        Validated configuration model.

    Raises
    ------
    FileNotFoundError, ValueError, ValidationError
        On missing file, bad YAML, or schema validation failure.
    """
    p = Path(path)
    raw = _read_yaml(p)
    try:
        return Config.model_validate(raw)
    except ValidationError as e:
        # Enrich error with file context for better UX
        raise ValidationError.from_exception_data(
            title=f"Invalid config: {p}",
            line_errors=e.errors(),
        )


# Convenience alias for symmetry with pydantic naming
def model_validate_yaml(path: str | Path) -> Config:
    """Alias for load_config()."""
    return load_config(path)


# -----------------------------------------------------------------------------
# Example factories (useful for tests or programmatic configs)
# -----------------------------------------------------------------------------
def example_scenario1_equidistant(output_dir: str = "data/runs/sc1_equ") -> Config:
    """Programmatic example matching the provided sample YAML."""
    return Config(
        scenario=Scenario(
            name="scenario1_equidistant",
            pulse_rate_hz=50_000_000,
            duration_s=2.0,
            output_dir=output_dir,
        ),
        devices=Devices(
            detector_eta=0.8,
            detector_dark_per_gate=1e-6,
            detector_dead_time_ns=60,
            detector_afterpulse=0.02,
            bsm_visibility=0.98,
            coincidence_window_ps=500,
        ),
        weather=None,
    )


__all__ = [
    "WeatherInline",
    "WeatherConfig",
    "Devices",
    "Scenario",
    "ScenarioName",
    "Config",
    "load_config",
    "model_validate_yaml",
    "example_scenario1_equidistant",
]
