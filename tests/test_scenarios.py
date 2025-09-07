# tests/test_scenarios.py
"""
Scenario smoke tests
====================

Covers:
- Each scenario class runs end-to-end with small parameters
- Produces a MetricFrame with expected tables
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sequence_ext.scenarios.scenario1_static import Scenario1Equidistant, Scenario1Uneven
from sequence_ext.scenarios.scenario2_moving import Scenario2Moving
from sequence_ext.scenarios.scenario3_city import Scenario3City
from sequence_ext.scenarios.scenario4_uk_opt import Scenario4UKOpt
from sequence_ext.io.metrics import MetricFrame


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (Scenario1Equidistant, dict(pulse_rate_hz=1_000_000, duration_s=0.01)),
        (Scenario1Uneven, dict(pulse_rate_hz=1_000_000, duration_s=0.01)),
        (Scenario2Moving, dict(pulse_rate_hz=1_000_000, duration_s=0.01, weather="clear")),
        (Scenario3City, dict(pulse_rate_hz=1_000_000, duration_s=0.01)),
        (
            Scenario4UKOpt,
            dict(
                src="London",
                dst="Edinburgh",
                relay_candidates=["Birmingham", "Manchester"],
                fiber_profile="smf28",
                pulse_rate_hz=1_000_000,
                duration_s=0.01,
            ),
        ),
    ],
)
def test_scenario_runs(tmp_path: Path, cls, kwargs):
    kwargs["output_dir"] = str(tmp_path)
    sc = cls(**kwargs)
    out_dir = sc.run()
    assert Path(out_dir).exists()

    # Check metrics
    from sequence_ext.io.metrics import MetricFrame

    mf = MetricFrame()
    for stem in ("physical", "network", "protocol", "security"):
        p = Path(out_dir) / f"{stem}.parquet"
        assert p.exists(), f"{stem} parquet missing"
