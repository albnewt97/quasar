# sequence_ext/scenarios/scenario4_uk_opt.py
"""
Scenario 4 – UK-wide optimization (hybrid/geo-aware)
====================================================

Performs a coarse UK-wide optimization by selecting relay (BSM) locations and
routes that maximize an estimated secret key throughput proxy.

Approach
--------
- Uses a small built-in UK node map with great-circle distances (km).
- Models fiber attenuation on each hop using fiber presets.
- Converts total path transmission into an effective pulse rate for the
  orchestrator, runs the sim, and aggregates the resulting secret rate.
- Sweeps candidate relay cities and returns the best configuration.

This is a pragmatic, production-friendly baseline without external datasets.
Downstream users can replace the built-in map with a geojson/CSV loader.

Outputs
-------
- Full MetricFrame written to <output_dir>/ for the best configuration.
- A summary CSV with the sweep results at <output_dir>/uk_opt_summary.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import math
import numpy as np
import pandas as pd

from ..orchestrator.orchestrator import Orchestrator
from ..io.logging import logger
from ..io.metrics import MetricFrame
from ..physics.fiber import FiberChannel
from ..physics.profiles import FIBER_PRESETS


# -----------------------------------------------------------------------------
# Built-in UK node map (lat, lon)
# -----------------------------------------------------------------------------
UK_NODES: Dict[str, Tuple[float, float]] = {
    "London": (51.5074, -0.1278),
    "Birmingham": (52.4862, -1.8904),
    "Manchester": (53.4808, -2.2426),
    "Leeds": (53.8008, -1.5491),
    "Bristol": (51.4545, -2.5879),
    "Cardiff": (51.4816, -3.1791),
    "Liverpool": (53.4084, -2.9916),
    "Newcastle": (54.9783, -1.6178),
    "Edinburgh": (55.9533, -3.1883),
    "Glasgow": (55.8642, -4.2518),
    "Sheffield": (53.3811, -1.4701),
    "Nottingham": (52.9548, -1.1581),
    "Leicester": (52.6369, -1.1398),
    "Cambridge": (52.2053, 0.1218),
    "Oxford": (51.7520, -1.2577),
}


def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) in km."""
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    s = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(s))


def _path_length_km(path: List[str]) -> float:
    """Total fiber length along an ordered list of node names."""
    total = 0.0
    for i in range(len(path) - 1):
        total += _haversine_km(UK_NODES[path[i]], UK_NODES[path[i + 1]])
    return total


def _fiber_eta(length_km: float, preset_name: str) -> float:
    """Linear transmission efficiency for a fiber span under a preset."""
    if preset_name not in FIBER_PRESETS:
        raise ValueError(f"Unknown fiber preset '{preset_name}'. Choices: {list(FIBER_PRESETS)}")
    p = FIBER_PRESETS[preset_name]
    ch = FiberChannel(
        length_km=length_km,
        attenuation_db_per_km=p.attenuation_db_per_km,
        dispersion_ps_nm_km=p.dispersion_ps_nm_km,
        pmd_ps_sqrt_km=p.pmd_ps_sqrt_km,
    )
    return ch.transmission_eta()


@dataclass
class Scenario4UKOpt:
    """
    Sweep relay cities and pick the best for MDI-QKD between two endpoints.

    Parameters
    ----------
    src : str
        Source city name (must be in UK_NODES).
    dst : str
        Destination city name (must be in UK_NODES).
    relay_candidates : List[str]
        Cities to consider as BSM relay (subset of UK_NODES).
    fiber_profile : str
        Key from FIBER_PRESETS (e.g., "smf28", "ull").
    pulse_rate_hz : int
        Source pulse repetition rate (before channel losses).
    duration_s : float
        Orchestrator run duration in seconds.
    output_dir : str
        Output directory for artifacts.
    """

    src: str
    dst: str
    relay_candidates: List[str]
    fiber_profile: str = "smf28"
    pulse_rate_hz: int = 50_000_000
    duration_s: float = 1.0
    output_dir: str = "data/runs/uk_opt"

    def _validate(self) -> None:
        if self.src not in UK_NODES or self.dst not in UK_NODES:
            raise ValueError("src/dst must be valid UK_NODES keys")
        for r in self.relay_candidates:
            if r not in UK_NODES:
                raise ValueError(f"Relay candidate '{r}' not found in UK_NODES")
        if self.src == self.dst:
            raise ValueError("src and dst must be different cities")

    def _effective_rate_for_path(self, path: List[str]) -> int:
        """
        Convert path losses into an effective pulse rate seen by the orchestrator.

        For an A–C–B MDI setup, the relevant throughput proxy scales roughly with
        the product of the two arms' transmissions to the BSM (central relay).
        Here, we compute eta = eta_left * eta_right and scale the base pulse rate.
        """
        if len(path) != 3:
            raise ValueError("Path must be [src, relay, dst] for MDI.")

        left_len = _path_length_km([path[0], path[1]])
        right_len = _path_length_km([path[1], path[2]])

        eta_left = _fiber_eta(left_len, self.fiber_profile)
        eta_right = _fiber_eta(right_len, self.fiber_profile)
        eta_mdi = eta_left * eta_right

        eff = max(1, int(self.pulse_rate_hz * eta_mdi))
        logger.info(
            "Path {} -> {} -> {} | lengths = ({:.1f} km, {:.1f} km), "
            "etas = ({:.3e}, {:.3e}), eta_mdi={:.3e}, eff_rate={}",
            path[0], path[1], path[2],
            left_len, right_len, eta_left, eta_right, eta_mdi, eff,
        )
        return eff

    def run(self) -> Path:
        self._validate()
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Running Scenario 4: UK-wide optimization | {} -> {} | candidates: {} | fiber={}",
            self.src,
            self.dst,
            self.relay_candidates,
            self.fiber_profile,
        )

        summary_rows = []
        best_row = None
        best_secret_rate = -1.0
        best_metrics: MetricFrame | None = None

        for relay in self.relay_candidates:
            if relay in (self.src, self.dst):
                # Relay must be distinct to model A–C–B layout properly.
                continue

            path = [self.src, relay, self.dst]
            effective_rate = self._effective_rate_for_path(path)

            # Run orchestrator at the effective rate dictated by physics
            orch = Orchestrator(
                pulse_rate_hz=effective_rate,
                duration_s=self.duration_s,
                output_dir=out_dir / f"relay_{relay.replace(' ', '_').lower()}",
            )
            mf = orch.run()

            # Aggregate a scalar secret rate proxy (mean over time)
            sec_mean = float(np.mean(mf.security["secret_rate"])) if len(mf.security) else 0.0

            row = {
                "relay": relay,
                "effective_rate_hz": effective_rate,
                "secret_rate_mean": sec_mean,
                "path": "->".join(path),
            }
            summary_rows.append(row)

            if sec_mean > best_secret_rate:
                best_secret_rate = sec_mean
                best_row = row
                best_metrics = mf

        # Persist summary
        summary_df = pd.DataFrame(summary_rows).sort_values("secret_rate_mean", ascending=False)
        summary_path = out_dir / "uk_opt_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Wrote sweep summary → {}", summary_path)

        # Write best metrics to a canonical subdir
        if best_row is None or best_metrics is None:
            raise RuntimeError("No valid relay produced results; check candidates.")

        best_dir = out_dir / "best"
        best_orch = Orchestrator(
            pulse_rate_hz=int(best_row["effective_rate_hz"]),
            duration_s=self.duration_s,
            output_dir=best_dir,
        )
        # Write using the orchestrator's standard writer to keep format consistent
        best_orch.write(best_metrics, fmt="parquet")
        logger.info(
            "Best relay: {} | path={} | secret_rate_mean={:.3e} → results in {}",
            best_row["relay"], best_row["path"], best_row["secret_rate_mean"], best_dir
        )

        return best_dir
