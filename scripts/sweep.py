#!/usr/bin/env python3
# scripts/sweep.py
"""
Parameter sweep runner for QUASAR
=================================

Run a series of QUASAR simulations by varying configuration parameters,
optionally in parallel, and produce a consolidated CSV summary.

Usage
-----
# 1) Simple built-in sweep (vary pulse_rate_hz and duration)
python scripts/sweep.py configs/scenario1_equidistant.yaml \
  --pulse-rate 20_000_000 50_000_000 100_000_000 \
  --duration 1.0 2.0 \
  --workers 4 \
  --out data/sweeps/sc1_equ

# 2) YAML-defined grid of overrides
python scripts/sweep.py configs/scenario2_moving.yaml \
  --grid-yaml configs/grids/sc2_grid.yaml \
  --workers 4 \
  --out data/sweeps/sc2_grid

Grid YAML format
----------------
# configs/grids/sc2_grid.yaml
overrides:
  - tag: clear_fast
    scenario:
      pulse_rate_hz: 100000000
      duration_s: 2.0
    weather: "clear"
  - tag: fog_slow
    scenario:
      pulse_rate_hz: 30000000
      duration_s: 5.0
    weather: "fog"

Notes
-----
- Each sweep run writes its own results under <out>/<tag or auto>.
- A summary CSV is written to <out>/summary.csv with mean secret rates, etc.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import concurrent.futures as cf

import numpy as np
import pandas as pd
import yaml

from configs.schema import Config, load_config
from sequence_ext.io.logging import logger, reconfigure_logging, set_run_id
from sequence_ext import (
    Scenario1Equidistant,
    Scenario1Uneven,
    Scenario2Moving,
    Scenario3City,
    Scenario4UKOpt,
)


# -----------------------------------------------------------------------------
# Scenario dispatcher
# -----------------------------------------------------------------------------
def _dispatch_and_run(cfg: Config, out_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    """
    Instantiate the appropriate scenario from cfg and run it.

    Returns
    -------
    (run_dir, metrics_summary)
    """
    s = cfg.scenario
    name = s.name

    # Create a scenario instance
    if name == "scenario1_equidistant":
        sc = Scenario1Equidistant(
            pulse_rate_hz=s.pulse_rate_hz,
            duration_s=s.duration_s,
            output_dir=str(out_dir),
        )
    elif name == "scenario1_uneven":
        sc = Scenario1Uneven(
            pulse_rate_hz=s.pulse_rate_hz,
            duration_s=s.duration_s,
            output_dir=str(out_dir),
        )
    elif name == "scenario2_moving":
        weather = cfg.weather if isinstance(cfg.weather, str) else "clear"
        sc = Scenario2Moving(
            pulse_rate_hz=s.pulse_rate_hz,
            duration_s=s.duration_s,
            output_dir=str(out_dir),
            weather=weather,
        )
    elif name == "scenario3_city":
        sc = Scenario3City(
            pulse_rate_hz=s.pulse_rate_hz,
            duration_s=s.duration_s,
            output_dir=str(out_dir),
        )
    elif name == "scenario4_uk_opt":
        # A minimal default sweep; for proper use, supply relay list in codebase or extend schema.
        relay_candidates = ["Birmingham", "Manchester", "Cambridge"]
        sc = Scenario4UKOpt(
            src="London",
            dst="Edinburgh",
            relay_candidates=relay_candidates,
            fiber_profile="smf28",
            pulse_rate_hz=s.pulse_rate_hz,
            duration_s=s.duration_s,
            output_dir=str(out_dir),
        )
    else:
        raise ValueError(f"Unsupported scenario {name}")

    # Run simulation
    run_dir = sc.run()

    # Minimal metrics summary: read security.parquet and compute mean secret rate
    sec_path = Path(run_dir) / "security.parquet"
    secret_mean = float("nan")
    try:
        sec = pd.read_parquet(sec_path)
        secret_mean = float(sec["secret_rate"].mean()) if not sec.empty else float("nan")
    except Exception:
        pass

    summary = {
        "scenario": name,
        "pulse_rate_hz": s.pulse_rate_hz,
        "duration_s": s.duration_s,
        "output_dir": str(run_dir),
        "secret_rate_mean": secret_mean,
    }
    return run_dir, summary


# -----------------------------------------------------------------------------
# Overrides / grid helpers
# -----------------------------------------------------------------------------
def _apply_overrides(base: Config, override: Dict[str, Any]) -> Config:
    """
    Return a new Config with 'override' dict merged into base.
    Supports keys: scenario.*, devices.*, weather.
    """
    data = base.model_dump()
    for top_key in ("scenario", "devices"):
        if top_key in override and isinstance(override[top_key], dict):
            data[top_key].update(override[top_key])

    if "weather" in override:
        data["weather"] = override["weather"]

    return Config.model_validate(data)


def _iter_builtin_grid(
    base: Config,
    pulse_rates: Iterable[int],
    durations: Iterable[float],
) -> Iterable[Tuple[str, Config]]:
    """
    Generate a grid by varying pulse_rate_hz and duration_s.
    """
    for pr, du in itertools.product(pulse_rates, durations):
        tag = f"pr{pr}_du{str(du).replace('.', 'p')}"
        override = {"scenario": {"pulse_rate_hz": pr, "duration_s": du}}
        yield tag, _apply_overrides(base, override)


def _iter_yaml_grid(base: Config, grid_yaml: Path) -> Iterable[Tuple[str, Config]]:
    """
    Load a YAML grid file with a list of overrides.

    Format:
    overrides:
      - tag: name1
        scenario: { pulse_rate_hz: 1, duration_s: 2.0, output_dir: "..." }
        devices: { ... }
        weather: "clear" | { attenuation_db_per_km: ..., cn2: ..., pointing_sigma_urad: ... }
    """
    with grid_yaml.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    items = raw.get("overrides")
    if not isinstance(items, list):
        raise ValueError("Grid YAML must contain a top-level 'overrides: [ ... ]' list.")

    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"Override #{idx} must be a mapping.")
        tag = it.get("tag") or f"run_{idx:03d}"
        override = {k: v for k, v in it.items() if k != "tag"}
        yield tag, _apply_overrides(base, override)


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------
def _worker(task: Tuple[str, Config, Path]) -> Dict[str, Any]:
    """
    Process worker for concurrent execution.
    """
    tag, cfg, out_root = task
    run_id = f"{tag}"
    with set_run_id(run_id):
        run_dir = out_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        _, summary = _dispatch_and_run(cfg, run_dir)
        summary["tag"] = tag
        return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a QUASAR parameter sweep.")
    p.add_argument("config", type=str, help="Base YAML config path.")
    p.add_argument("--out", type=str, required=True, help="Output root directory for sweep results.")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2, help="Max parallel workers.")

    # Built-in grid
    p.add_argument("--pulse-rate", type=str, nargs="*", default=None,
                   help="List of pulse rates (Hz). Accepts integers with _ separators (e.g., 50_000_000).")
    p.add_argument("--duration", type=float, nargs="*", default=None,
                   help="List of durations (seconds).")

    # YAML-defined grid
    p.add_argument("--grid-yaml", type=str, default=None, help="Path to grid YAML with overrides list.")

    return p.parse_args(argv)


def _parse_pulse_rates(values: Optional[List[str]]) -> Optional[List[int]]:
    if not values:
        return None
    out: List[int] = []
    for v in values:
        # allow underscores and commas
        v2 = v.replace("_", "").replace(",", "")
        out.append(int(v2))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    reconfigure_logging()

    base_cfg = load_config(args.config)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks: List[Tuple[str, Config, Path]] = []

    if args.grid_yaml:
        for tag, cfg in _iter_yaml_grid(base_cfg, Path(args.grid_yaml)):
            tasks.append((tag, cfg, out_root))
    else:
        prs = _parse_pulse_rates(args.pulse_rate) or [base_cfg.scenario.pulse_rate_hz]
        dus = args.duration or [base_cfg.scenario.duration_s]
        for tag, cfg in _iter_builtin_grid(base_cfg, prs, dus):
            tasks.append((tag, cfg, out_root))

    logger.info("Prepared {} sweep tasks → {}", len(tasks), out_root)

    # Execute
    rows: List[Dict[str, Any]] = []
    if args.workers and args.workers > 1:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(_worker, tasks):
                rows.append(res)
                logger.info("Completed {}", res.get("tag"))
    else:
        for t in tasks:
            res = _worker(t)
            rows.append(res)
            logger.info("Completed {}", res.get("tag"))

    # Summarize
    df = pd.DataFrame(rows)
    df.sort_values(["secret_rate_mean"], ascending=[False], inplace=True)
    summary_path = out_root / "summary.csv"
    df.to_csv(summary_path, index=False)
    logger.info("Wrote sweep summary → {}", summary_path)

    # Also dump JSON for quick programmatic consumption
    (out_root / "summary.json").write_text(json.dumps(df.to_dict(orient="records"), indent=2))

    # Print top 5 to stdout
    top5 = df.head(5).to_string(index=False)
    print("\nTop 5 runs by mean secret rate:\n", top5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
