"""Physics regression tests for TCLSolver and RHPWitness (§5.6).

Verifies canonical rate extraction, sign detection, and online accumulation
of the Rivas-Huelga-Plenio non-Markovianity measure.

Reference: Rivas, Huelga, Plenio, Rep. Prog. Phys. 77, 094001 (2014) §II.C.
"""
from __future__ import annotations

import math

import pytest

from qndt.core.context import PauliRateVector
from qndt.physics.master_equation import CanonicalRates, RHPWitness, TCLSolver

pytestmark = pytest.mark.physics_regression

# ---------------------------------------------------------------------------
# Shared solver fixture
# ---------------------------------------------------------------------------

_SOLVER = TCLSolver(dt=0.01)


# ---------------------------------------------------------------------------
# test_canonical_rates_positive_markovian
# ---------------------------------------------------------------------------


def test_canonical_rates_positive_markovian() -> None:
    """Static Pauli rates (dλ/dt = 0) must give all canonical rates ≈ 0.

    A channel that does not evolve is Markovian; the TCL rates vanish because
    all finite-difference numerators are zero.
    """
    rates = PauliRateVector(px=0.1, py=0.05, pz=0.08)
    result = _SOLVER.canonical_rates(rates_t=rates, rates_t_minus_dt=rates, t=1.0)

    assert result.gamma_x == pytest.approx(0.0, abs=1e-12)
    assert result.gamma_y == pytest.approx(0.0, abs=1e-12)
    assert result.gamma_z == pytest.approx(0.0, abs=1e-12)
    assert not result.is_non_markovian


# ---------------------------------------------------------------------------
# test_canonical_rates_sign
# ---------------------------------------------------------------------------


def test_canonical_rates_sign() -> None:
    """Increasing λx should produce a negative γx (non-Markovian backflow).

    Construction (§5.6 TCL formula):
      rates_t_minus_dt: (px=0.2, py=0.1, pz=0.1) → λx=0.6, λy=0.4, λz=0.4
      rates_t:          (px=0.1, py=0.05, pz=0.05) → λx=0.8, λy=0.7, λz=0.7

    dλy/dt = (0.7-0.4)/0.01 = 30, dλz/dt = 30
    γx = -(30+30) / (4·0.8) = -18.75 < 0.
    """
    rates_prev = PauliRateVector(px=0.2, py=0.1, pz=0.1)
    rates_curr = PauliRateVector(px=0.1, py=0.05, pz=0.05)

    result = _SOLVER.canonical_rates(
        rates_t=rates_curr, rates_t_minus_dt=rates_prev, t=0.01
    )

    assert result.gamma_x < 0.0, f"Expected gamma_x < 0; got {result.gamma_x}"
    assert result.is_non_markovian


# ---------------------------------------------------------------------------
# test_rhp_witness_accumulates
# ---------------------------------------------------------------------------


def test_rhp_witness_accumulates() -> None:
    """RHPWitness must accumulate positive N_RHP when any gamma < 0.

    Feeds three snapshots at t=0, 0.1, 0.2 with gamma_x = -1.0.
    Expected N_RHP = 2 × (1.0 × 0.1) = 0.2 (two intervals, no contribution
    from the first snapshot alone).
    """
    witness = RHPWitness()
    for step in range(3):
        witness.update(
            CanonicalRates(gamma_x=-1.0, gamma_y=0.0, gamma_z=0.0, t=step * 0.1)
        )

    assert witness.current_value() == pytest.approx(0.2, rel=1e-10)
    assert witness.is_non_markovian()


# ---------------------------------------------------------------------------
# test_rhp_witness_zero_for_markovian
# ---------------------------------------------------------------------------


def test_rhp_witness_zero_for_markovian() -> None:
    """RHPWitness must remain at 0 when all canonical rates are non-negative."""
    witness = RHPWitness()
    for step in range(5):
        witness.update(
            CanonicalRates(gamma_x=0.5, gamma_y=0.3, gamma_z=0.1, t=step * 0.1)
        )

    assert witness.current_value() == pytest.approx(0.0, abs=1e-15)
    assert not witness.is_non_markovian()


# ---------------------------------------------------------------------------
# test_sign_change_times_detected
# ---------------------------------------------------------------------------


def test_sign_change_times_detected() -> None:
    """sign_change_times() must return the timestamp of a positive→negative crossing.

    Feeds a positive snapshot at t=0 followed by a negative snapshot at t=0.5.
    The crossing is at t=0.5.
    """
    witness = RHPWitness()
    witness.update(CanonicalRates(gamma_x=1.0, gamma_y=0.0, gamma_z=0.0, t=0.0))
    witness.update(CanonicalRates(gamma_x=-1.0, gamma_y=0.0, gamma_z=0.0, t=0.5))

    times = witness.sign_change_times()
    assert len(times) > 0, "Expected at least one sign-change time to be recorded"
    assert times[0] == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------------------
# test_rhp_reset
# ---------------------------------------------------------------------------


def test_rhp_reset() -> None:
    """After reset(), current_value() == 0 and is_non_markovian() == False."""
    witness = RHPWitness()
    for step in range(3):
        witness.update(
            CanonicalRates(gamma_x=-2.0, gamma_y=-0.5, gamma_z=0.0, t=step * 0.1)
        )
    assert witness.is_non_markovian()

    witness.reset()

    assert witness.current_value() == pytest.approx(0.0, abs=1e-15)
    assert not witness.is_non_markovian()
    assert witness.sign_change_times() == []


# ---------------------------------------------------------------------------
# CRITICAL: Markovian recovery test (audit requirement, Rivas+Huelga+Plenio 2014)
# ---------------------------------------------------------------------------


def test_markovian_dephasing_recovery() -> None:
    """TCLSolver must recover γz ≈ 1.0 for pure-Z dephasing with λ(t) = exp(-t).

    Reference:
        Rivas, Huelga, Plenio, Rep. Prog. Phys. 77, 094001 (2014) §II.C.

    Physics:
        Pure Z dephasing: λx(t) = λy(t) = exp(−t), λz(t) = 1.
        TCL inversion: γz = (dλz/λz − dλx/λx − dλy/λy)/2 = (0 − (−1) − (−1))/2 = 1.
        γx = γy = (0 − (−1) − 0)/2 ... but symmetric roles: γx = γy = 0 for pure Z.

    The recovered γz must be 1.0 ± 0.01 (within finite-difference error) and
    γx, γy must be 0.0 ± 0.01.
    """
    dt = 1e-6  # very small step to minimise finite-difference error
    solver = TCLSolver(dt=dt)

    t = 0.5  # arbitrary mid-run time
    lx_t = math.exp(-t)
    lx_p = math.exp(-(t - dt))
    # Pure Z dephasing: λx = λy = exp(-t), λz = 1
    rates_t = PauliRateVector(px=0.0, py=0.0, pz=(1 - lx_t) / 2.0)
    rates_p = PauliRateVector(px=0.0, py=0.0, pz=(1 - lx_p) / 2.0)

    result = solver.canonical_rates(rates_t=rates_t, rates_t_minus_dt=rates_p, t=t)

    assert result.gamma_z == pytest.approx(1.0, abs=0.01), (
        f"Markovian dephasing: expected γz ≈ 1.0, got {result.gamma_z:.6f}"
    )
    assert result.gamma_x == pytest.approx(0.0, abs=0.01), (
        f"Markovian dephasing: expected γx ≈ 0.0, got {result.gamma_x:.6f}"
    )
    assert result.gamma_y == pytest.approx(0.0, abs=0.01), (
        f"Markovian dephasing: expected γy ≈ 0.0, got {result.gamma_y:.6f}"
    )
    assert not result.is_non_markovian, "Markovian channel must not trigger non-Markovian flag"


def test_nonmonotonic_lambda_gives_negative_rate() -> None:
    """A non-monotonically increasing λx must give γz < 0 (information backflow).

    Construction: λx, λy increase from 0.4 → 0.6 over one step (λz = 1).
    This means coherence is being restored — a hallmark of non-Markovian dynamics.
    The recovered γz must be negative.
    """
    dt = 0.01
    solver = TCLSolver(dt=dt)

    # λx, λy increasing (non-Markovian backflow); λz = 1 (pure Z axis)
    lx_prev = 0.4
    lx_curr = 0.6
    pz_prev = (1 - lx_prev) / 2.0   # 0.3
    pz_curr = (1 - lx_curr) / 2.0   # 0.2

    rates_prev = PauliRateVector(px=0.0, py=0.0, pz=pz_prev)
    rates_curr = PauliRateVector(px=0.0, py=0.0, pz=pz_curr)

    result = solver.canonical_rates(rates_t=rates_curr, rates_t_minus_dt=rates_prev, t=dt)

    assert result.gamma_z < 0.0, (
        f"Non-monotonic λ must give γz < 0; got {result.gamma_z:.4f}"
    )
    assert result.is_non_markovian
