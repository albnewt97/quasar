#!/usr/bin/env python3
"""
Run a QUASAR scenario from a YAML config
=======================================

Usage
-----
python scripts/run_scenario.py configs/scenario1_equidistant.yaml

This script:
1. Loads and validates the YAML config using `configs.schema.Config`.
2. Dispatches to the appropriate scenario class.
3. Runs the simulation and writes results to the configured output_dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from configs.schema import load_config, Config
from sequence_ext.io.logging import logger, reconfigure_logging
from sequence_ext import (
    Scenario1Equidistant,
    Scenario1Uneven,
    Scenario2Moving,
    Scenario3City,
    Scenario4UKOpt,
)


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------
def _run_from_config(cfg: Config) -> Path:
    s = cfg.scenario

    if s.name == "scenario1_equidistant":
        sc = Scenario1Equidistant(
            pulse_rate_hz=s.pulse_rate_hz,
            duration_s=s.duration_s,
            output_dir=s.output_dir,
        )
    elif s.name == "scenario1_uneven":
        sc = Scenario1Uneven(
            pulse_rate_hz=s.pulse_rate_hz,
            duration_s=s.duration_s,
            output_dir=s.output_dir,
        )
    elif s.name == "scenario2_moving":
        # Weather may be a preset string or inline object
        weather = None
        if isinstance(cfg.weather, str):
            weather = cfg.weather
        elif isinstance(cfg.weather, dict) and "attenuation_db_per_km" in cfg.weather:
            weather = "custom"  # scenario2_moving expects a preset key
        sc = Scenario2Moving(
            pulse_rate_hz=s.pulse_rate_hz,
            duration_s=s.duration_s,
            output_dir=s.output_dir,
            weather=weather or "clear",
        )
    elif s.name == "scenario3_city":
        sc = Scenario3City(
            pulse_rate_hz=s.pulse_rate_hz,
            duration_s=s.duration_s,
            output_dir=s.output_dir,
        )
    elif s.name == "scenario4_uk_opt":
        # In real configs, relay_candidates should be added under scenario.
        relay_candidates = ["Birmingham", "Manchester", "Cambridge"]
        sc = Scenario4UKOpt(
            src="London",
            dst="Edinburgh",
            relay_candidates=relay_candidates,
            fiber_profile="smf28",
            pulse_rate_hz=s.pulse_rate_hz,
            duration_s=s.duration_s,
            output_dir=s.output_dir,
        )
    else:
        raise ValueError(f"Unsupported scenario name {s.name}")

    return sc.run()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a QUASAR scenario from YAML config.")
    parser.add_argument("config", type=str, help="Path to YAML config file.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    reconfigure_logging()
    logger.info("Loaded config from {}", args.config)

    try:
        out_dir = _run_from_config(cfg)
        logger.info("Scenario finished. Results in {}", out_dir)
        return 0
    except Exception as e:
        logger.exception("Scenario failed: {}", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
