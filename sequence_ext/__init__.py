# sequence_ext/__init__.py
"""
sequence_ext
============

Production-ready extensions for SeQUeNCe-based MDI-QKD simulations.

This package provides:
- Orchestration (`orchestrator`): event definitions, central runner, scenario integration.
- I/O (`io`): logging, metrics containers, result writers.
- Physics (`physics`): models for fiber, free-space channels, detectors, and presets.
- Scenarios (`scenarios`): ready-to-run experiment setups.
- Topology (`topo`): geometry and routing utilities.
- Visualization (`viz`): static plots and interactive dashboard.

Design principles
-----------------
- Explicit, typed, minimal public API surface.
- Import-light: heavy dependencies only loaded when needed.
- Logging policy: centralised in `sequence_ext.io.logging`, no global side effects.
- Results pipeline: every orchestrator/scenario returns a MetricFrame, written
  to disk via a ResultWriter.
"""

from __future__ import annotations

from importlib import metadata as _metadata
from typing import Final

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
try:
    __version__: Final[str] = _metadata.version("quasar-mdiqkd")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0+local"


def version() -> str:
    """Return the installed package version string."""
    return __version__


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------
# Logging
from .io.logging import logger, reconfigure_logging, set_run_id, bind_component

# Metrics + writers
from .io.metrics import MetricFrame
from .io.writers import ResultWriter

# Orchestration core
from .orchestrator.orchestrator import Orchestrator
from .orchestrator.events import BSMEvent, DetectorEvent

# Physics exports (commonly used)
from .physics.fiber import FiberChannel
from .physics.free_space import FreeSpaceChannel
from .physics.detectors import Detector
from .physics.profiles import WEATHER_PRESETS, DEVICE_PRESETS, FIBER_PRESETS

# Scenarios (baseline ready-to-run)
from .scenarios.scenario1_static import Scenario1Equidistant, Scenario1Uneven
from .scenarios.scenario2_moving import Scenario2Moving
from .scenarios.scenario3_city import Scenario3City
from .scenarios.scenario4_uk_opt import Scenario4UKOpt

__all__ = [
    "__version__",
    "version",
    # Logging
    "logger",
    "reconfigure_logging",
    "set_run_id",
    "bind_component",
    # Metrics
    "MetricFrame",
    "ResultWriter",
    # Orchestrator
    "Orchestrator",
    "BSMEvent",
    "DetectorEvent",
    # Physics
    "FiberChannel",
    "FreeSpaceChannel",
    "Detector",
    "WEATHER_PRESETS",
    "DEVICE_PRESETS",
    "FIBER_PRESETS",
    # Scenarios
    "Scenario1Equidistant",
    "Scenario1Uneven",
    "Scenario2Moving",
    "Scenario3City",
    "Scenario4UKOpt",
]
