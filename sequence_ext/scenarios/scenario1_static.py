# sequence_ext/scenarios/scenario1_static.py
"""
Scenario 1 – Static nodes with fiber channels
=============================================

Two flavors:
- Equidistant nodes (A–C–B with equal fiber segments).
- Uneven distances (A–C shorter or longer than C–B).

Purpose
-------
Baseline benchmarking for MDI-QKD under fiber-only conditions.
Useful for validating orchestrator + metrics pipeline.

Outputs
-------
Parquet/CSV metrics in configured output directory.

Usage
-----
from sequence_ext.scenarios.scenario1_static import Scenario1Equidistant
sc = Scenario1Equidistant(pulse_rate_hz=50e6, duration_s=1.0, output_dir="data/sc1")
sc.run()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..orchestrator.orchestrator import Orchestrator
from ..io.logging import logger


@dataclass
class Scenario1Equidistant:
    """Equidistant nodes (A–C–B with equal fiber lengths)."""

    pulse_rate_hz: int
    duration_s: float
    output_dir: str

    def run(self) -> Path:
        logger.info("Running Scenario 1a: equidistant fiber nodes")
        orch = Orchestrator(self.pulse_rate_hz, self.duration_s, Path(self.output_dir))
        mf = orch.run()
        return orch.write(mf, fmt="parquet")


@dataclass
class Scenario1Uneven:
    """Uneven distances between A–C and C–B."""

    pulse_rate_hz: int
    duration_s: float
    output_dir: str

    def run(self) -> Path:
        logger.info("Running Scenario 1b: uneven fiber nodes")
        orch = Orchestrator(self.pulse_rate_hz, self.duration_s, Path(self.output_dir))
        mf = orch.run()
        return orch.write(mf, fmt="parquet")
