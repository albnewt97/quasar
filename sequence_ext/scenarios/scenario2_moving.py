# sequence_ext/scenarios/scenario2_moving.py
"""
Scenario 2 – Moving terminals (free-space)
==========================================

Models MDI-QKD with at least one moving node (e.g. UAV, satellite).
Channel is free-space optical (FSO) with weather/turbulence/pointing.

Inputs
------
- pulse_rate_hz : photon pulse repetition rate.
- duration_s    : total run time in seconds.
- output_dir    : directory to write results.
- weather       : weather preset (attenuation, turbulence, pointing).

Purpose
-------
Stress-test protocol under dynamic link conditions and weather variability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..orchestrator.orchestrator import Orchestrator
from ..io.logging import logger
from ..physics.free_space import FreeSpaceChannel
from ..physics.profiles import WEATHER_PRESETS


@dataclass
class Scenario2Moving:
    pulse_rate_hz: int
    duration_s: float
    output_dir: str
    weather: str = "clear"  # key from WEATHER_PRESETS

    def run(self) -> Path:
        if self.weather not in WEATHER_PRESETS:
            raise ValueError(f"Unknown weather preset {self.weather}")
        w = WEATHER_PRESETS[self.weather]
        logger.info("Running Scenario 2: moving terminal with weather preset '{}'", self.weather)

        # Example: free-space link 500 km, modulated by weather profile
        fso = FreeSpaceChannel(
            distance_km=500,
            attenuation_db_per_km=w.attenuation_db_per_km,
            cn2=w.cn2,
            pointing_sigma_urad=w.pointing_sigma_urad,
        )

        # Modify orchestrator pulse rate depending on channel transmission
        eta = fso.transmission_eta()
        effective_rate = int(self.pulse_rate_hz * eta)

        logger.info("FSO transmission efficiency = {:.3e}, effective rate = {}", eta, effective_rate)

        orch = Orchestrator(effective_rate, self.duration_s, Path(self.output_dir))
        mf = orch.run()
        return orch.write(mf, fmt="parquet")
