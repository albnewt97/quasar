# sequence_ext/physics/profiles.py
"""
Profiles
========

Convenience presets for physics/device parameters used in scenarios.

This module provides dictionaries and helper functions for:
- Weather presets (attenuation, turbulence, pointing).
- Device presets (detectors, BSM).
- Fiber presets (standard telecom fiber, ultra-low-loss).

These are intended for quick configuration in scenarios or YAML config files.
"""

from __future__ import annotations

from dataclasses import dataclass


# -------------------------------------------------------------------------
# Weather presets
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class WeatherPreset:
    name: str
    attenuation_db_per_km: float
    cn2: float
    pointing_sigma_urad: float


CLEAR = WeatherPreset("clear", attenuation_db_per_km=0.05, cn2=1e-15, pointing_sigma_urad=3.0)
FOG = WeatherPreset("fog", attenuation_db_per_km=0.3, cn2=1e-14, pointing_sigma_urad=8.0)
RAIN = WeatherPreset("rain", attenuation_db_per_km=0.15, cn2=5e-15, pointing_sigma_urad=5.0)

WEATHER_PRESETS = {w.name: w for w in (CLEAR, FOG, RAIN)}


# -------------------------------------------------------------------------
# Device presets
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class DevicePreset:
    name: str
    detector_eta: float
    detector_dark_per_gate: float
    detector_dead_time_ns: float
    detector_afterpulse: float
    bsm_visibility: float
    coincidence_window_ps: int


BASELINE = DevicePreset(
    "baseline",
    detector_eta=0.8,
    detector_dark_per_gate=1e-6,
    detector_dead_time_ns=60,
    detector_afterpulse=0.02,
    bsm_visibility=0.98,
    coincidence_window_ps=500,
)

SUPERCONDUCTING = DevicePreset(
    "snsdp",
    detector_eta=0.9,
    detector_dark_per_gate=1e-8,
    detector_dead_time_ns=40,
    detector_afterpulse=0.005,
    bsm_visibility=0.99,
    coincidence_window_ps=200,
)

DEVICE_PRESETS = {d.name: d for d in (BASELINE, SUPERCONDUCTING)}


# -------------------------------------------------------------------------
# Fiber presets
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class FiberPreset:
    name: str
    attenuation_db_per_km: float
    dispersion_ps_nm_km: float
    pmd_ps_sqrt_km: float


SMF28 = FiberPreset("smf28", attenuation_db_per_km=0.2, dispersion_ps_nm_km=17.0, pmd_ps_sqrt_km=0.1)
ULL = FiberPreset("ull", attenuation_db_per_km=0.16, dispersion_ps_nm_km=18.0, pmd_ps_sqrt_km=0.05)

FIBER_PRESETS = {f.name: f for f in (SMF28, ULL)}


__all__ = [
    "WeatherPreset",
    "WEATHER_PRESETS",
    "DevicePreset",
    "DEVICE_PRESETS",
    "FiberPreset",
    "FIBER_PRESETS",
]
