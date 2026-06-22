"""Integration tests for TelemetryResampler, telemetry sources, and the engine.

No physics_regression mark: these are integration-path tests covering the
full §3.5 data pipeline from source → resampler → engine → ptm().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qndt.core.context import OpContext
from qndt.physics.channels import validate_ptm
from qndt.physics.kernels import ExponentialKernel
from qndt.telemetry.engine import EnvironmentalTelemetryEngine
from qndt.telemetry.resampler import TelemetryResampler
from qndt.telemetry.sources import (
    CSVReplaySource,
    SyntheticTelemetrySource,
    TelemetrySample,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_LINK = "link_test"


def _sample(t: float, e: list[float]) -> TelemetrySample:
    return TelemetrySample(t=t, E=np.array(e, dtype=np.float64), link_id=_LINK)


# ---------------------------------------------------------------------------
# TelemetryResampler tests
# ---------------------------------------------------------------------------


def test_push_and_at_single() -> None:
    """Push one sample; at() must return its E vector exactly."""
    rs = TelemetryResampler()
    E = np.array([25.0, 0.001, 0.5])
    rs.push(_sample(0.0, E.tolist()))
    result = rs.at(_LINK, 0.0)
    np.testing.assert_array_almost_equal(result, E)


def test_linear_interpolation() -> None:
    """at() must linearly interpolate between two bracketing samples."""
    rs = TelemetryResampler()
    rs.push(_sample(0.0, [0.0, 0.0, 0.0]))
    rs.push(_sample(2.0, [2.0, 2.0, 2.0]))
    result = rs.at(_LINK, 1.0)
    np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])


def test_hold_last_value() -> None:
    """t beyond last sample within max_gap must return last E (not stale)."""
    rs = TelemetryResampler(max_gap_s=10.0)
    E = np.array([30.0, 0.0, 0.0])
    rs.push(_sample(0.0, [0.0, 0.0, 0.0]))
    rs.push(_sample(1.0, E.tolist()))
    result = rs.at(_LINK, 3.0)   # gap = 2 < max_gap=10
    np.testing.assert_array_almost_equal(result, E)
    assert not rs.is_stale(_LINK)


def test_stale_detection() -> None:
    """t beyond last sample by more than max_gap_s must set the stale flag."""
    rs = TelemetryResampler(max_gap_s=5.0)
    rs.push(_sample(0.0, [1.0, 0.0, 0.0]))
    rs.push(_sample(1.0, [1.0, 0.0, 0.0]))
    rs.at(_LINK, 7.0)   # gap = 6 > max_gap=5 → stale
    assert rs.is_stale(_LINK)


def test_hold_first_value() -> None:
    """t before the first sample must return the first E (hold-first)."""
    rs = TelemetryResampler()
    E = np.array([20.0, 0.0, 0.0])
    rs.push(_sample(5.0, E.tolist()))
    rs.push(_sample(10.0, [30.0, 0.0, 0.0]))
    result = rs.at(_LINK, 1.0)   # before all samples
    np.testing.assert_array_almost_equal(result, E)


def test_eviction() -> None:
    """Samples older than window_s must be evicted on the next push."""
    rs = TelemetryResampler(window_s=10.0, max_gap_s=60.0)
    E = np.array([1.0, 0.0, 0.0])
    for t in [0.0, 5.0, 10.0]:
        rs.push(_sample(t, E.tolist()))
    # Push at t=20: cutoff = 20-10 = 10. Evict t < 10: {0.0, 5.0}. Keep t=10 and t=20.
    rs.push(_sample(20.0, E.tolist()))
    stats = rs.stats(_LINK)
    assert stats["buffer_size"] == 2
    assert stats["oldest_t"] == pytest.approx(10.0)


def test_window_returns_sorted() -> None:
    """window() must always return samples in strictly ascending t order."""
    rs = TelemetryResampler()
    for t in [3.0, 1.0, 2.0]:   # insert out of order to stress-test sort
        rs.push(_sample(t, [float(t), 0.0, 0.0]))
    samples = rs.window(_LINK, 10.0)
    times = [s.t for s in samples]
    assert times == sorted(times)


def test_unknown_link_raises() -> None:
    """at() on an unknown link_id must raise KeyError."""
    rs = TelemetryResampler()
    with pytest.raises(KeyError, match="unknown_link"):
        rs.at("unknown_link", 0.0)


# ---------------------------------------------------------------------------
# TelemetrySource tests
# ---------------------------------------------------------------------------


def test_synthetic_source_yields_correct_count() -> None:
    """SyntheticTelemetrySource with duration=10, dt=1 must yield 10 samples."""
    src = SyntheticTelemetrySource(
        link_id=_LINK, duration_s=10.0, dt_s=1.0, seed=0
    )
    samples = list(src)
    assert len(samples) == 10
    assert samples[0].t == pytest.approx(0.0)
    assert samples[-1].t == pytest.approx(9.0)


def test_csv_replay_source(tmp_path: Path) -> None:
    """CSVReplaySource must skip comments, parse values, and apply speedup."""
    csv_content = (
        "# Telemetry header comment\n"
        "0.0,25.0,0.001,0.5\n"
        "1.0,26.0,0.002,0.6\n"
    )
    csv_file = tmp_path / "telemetry.csv"
    csv_file.write_text(csv_content)

    src = CSVReplaySource(
        path=str(csv_file),
        t_col=0,
        env_cols=[1, 2, 3],
        link_id=_LINK,
        speedup=2.0,       # 1.0 raw → 0.5 in sim time
        epoch_offset=0.0,
    )
    samples = list(src)

    assert len(samples) == 2
    assert samples[0].t == pytest.approx(0.0)       # 0.0 / 2.0
    assert samples[1].t == pytest.approx(0.5)       # 1.0 / 2.0
    assert samples[0].E[0] == pytest.approx(25.0)
    assert samples[0].E[1] == pytest.approx(0.001)
    assert samples[0].link_id == _LINK


# ---------------------------------------------------------------------------
# EnvironmentalTelemetryEngine tests
# ---------------------------------------------------------------------------


def _make_engine() -> EnvironmentalTelemetryEngine:
    return EnvironmentalTelemetryEngine(
        sensitivity=np.eye(3),
        kernel=ExponentialKernel(tau_x=1.0, tau_y=1.0, tau_z=1.0),
    )


def test_engine_pauli_rates_zero_with_no_data() -> None:
    """pauli_rates() must return (0,0,0) when no telemetry has been ingested."""
    engine = _make_engine()
    rates = engine.pauli_rates(_LINK, 1.0)
    assert rates.px == pytest.approx(0.0)
    assert rates.py == pytest.approx(0.0)
    assert rates.pz == pytest.approx(0.0)


def test_engine_pauli_rates_nonzero_with_data() -> None:
    """pauli_rates() must return non-zero rates after two samples are ingested."""
    engine = _make_engine()
    # Two samples give one non-empty convolution interval.
    engine.ingest(_sample(0.0, [1.0, 1.0, 1.0]))
    engine.ingest(_sample(1.0, [1.0, 1.0, 1.0]))
    rates = engine.pauli_rates(_LINK, 2.0)
    assert rates.px > 0.0 or rates.py > 0.0 or rates.pz > 0.0


def test_engine_ptm_valid() -> None:
    """engine.ptm(ctx) must pass validate_ptm() after data ingestion."""
    engine = _make_engine()
    engine.ingest(_sample(0.0, [1.0, 1.0, 1.0]))
    engine.ingest(_sample(1.0, [1.0, 1.0, 1.0]))
    ctx = OpContext(
        link_id=_LINK, node_id=None, t=2.0, lambda_q=1550e-9, gate_width=1e-9
    )
    result = engine.ptm(ctx)
    assert validate_ptm(result), f"ptm() returned invalid PTM: {result}"


def test_engine_cache_invalidation() -> None:
    """Ingesting a new sample must invalidate the cache for its link.

    With one sample the convolution loop is empty → rates = (0,0,0).
    After ingesting a second sample the cache is invalidated and rates become
    non-zero, proving the stale entry was not reused.
    """
    engine = _make_engine()

    # One sample → convolution loop has no iterations (range(1,1) = ∅) → zero rates.
    engine.ingest(_sample(1.0, [0.0, 0.0, 0.0]))
    r1 = engine.pauli_rates(_LINK, 2.0)
    assert r1.px == pytest.approx(0.0), "Expected zero rates with one sample"

    # Second sample invalidates cache.  Now the loop executes once.
    engine.ingest(_sample(1.5, [1.0, 1.0, 1.0]))
    r2 = engine.pauli_rates(_LINK, 2.0)

    # Cache invalidated: result differs from the previously cached (zero) result.
    # The temperature component of E=[1,1,1] minus env_ref[0]=20 is negative and
    # clips to zero (so r2.px == 0), but seismic and wind components are unaffected
    # and produce non-zero py and pz, confirming the stale entry was not reused.
    total_r2 = r2.px + r2.py + r2.pz
    total_r1 = r1.px + r1.py + r1.pz
    assert total_r2 > total_r1, (
        f"Cache invalidation failed: total rates r2={total_r2:.4f} should exceed r1={total_r1:.4f}"
    )
