# tests/test_orchestrator.py
"""
Orchestrator tests
==================

Covers:
- Orchestrator.run() returns a well-formed MetricFrame
- Orchestrator.write() persists expected artifacts (Parquet by default)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sequence_ext.orchestrator.orchestrator import Orchestrator
from sequence_ext.io.metrics import MetricFrame


def test_orchestrator_run_returns_metricframe(tmp_path: Path):
    orch = Orchestrator(pulse_rate_hz=1_000_000, duration_s=0.01, output_dir=tmp_path)
    mf = orch.run()
    assert isinstance(mf, MetricFrame)

    # basic shape checks
    assert not mf.physical.empty
    assert not mf.protocol.empty
    assert not mf.security.empty
    for table in (mf.physical, mf.network, mf.protocol, mf.security):
        assert "time" in table.columns
        # should be monotonic increasing time
        assert (table["time"].diff().fillna(0) >= 0).all()


def test_orchestrator_write_persists_parquet(tmp_path: Path):
    orch = Orchestrator(pulse_rate_hz=500_000, duration_s=0.01, output_dir=tmp_path)
    mf = orch.run()
    out_dir = orch.write(mf, fmt="parquet")

    # Files exist
    for stem in ("physical", "network", "protocol", "security"):
        p = Path(out_dir) / f"{stem}.parquet"
        assert p.exists(), f"Missing {p}"
        # Read back a couple to ensure validity
        df = pd.read_parquet(p)
        assert "time" in df.columns
        assert len(df) > 0
