"""Physics regression tests for TensorStateTracker and SimulationStep.

All tests are marked ``physics_regression`` and verify analytic limits from
docs/architecture.md §5.1 (PTM algebra), §5.5 (dephasing), and Bell-state fidelity.

Key analytic results used:
  - |00⟩⟨00| has fidelity 0.5 with |Φ+⟩.
  - After H + CNOT on |00⟩: perfect |Φ+⟩, purity = 1, entropy(q0) = log(2).
  - depolarising_ptm(0.5) on q0 of |Φ+⟩ → F(Φ+) = 0.625, purity = 0.4375.
  - dephasing_ptm(0.3)    on q0 of |Φ+⟩ → F(Φ+) = 0.7,   purity = 0.58.
  - Identity PTM leaves fidelity unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from qndt.physics.channels import dephasing_ptm, depolarising_ptm
from qndt.quantum.tracker import TensorStateTracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker2() -> TensorStateTracker:
    """Fresh 2-qubit TensorStateTracker in the |00⟩ state."""
    return TensorStateTracker(n_sites=2)


@pytest.fixture
def bell_tracker() -> TensorStateTracker:
    """2-qubit TensorStateTracker with q0,q1 prepared in the |Φ+⟩ Bell state."""
    t = TensorStateTracker(n_sites=2)
    t.entangle(0, 1, bell_state="phi+")
    return t


# ---------------------------------------------------------------------------
# 1. Initial state
# ---------------------------------------------------------------------------


@pytest.mark.physics_regression
def test_initial_state_purity_unity(tracker2: TensorStateTracker) -> None:
    """Fresh |00⟩ state is pure: Tr(ρ²) = 1."""
    assert tracker2.purity() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.physics_regression
def test_initial_fidelity_with_phi_plus(tracker2: TensorStateTracker) -> None:
    """Fidelity of |00⟩⟨00| with |Φ+⟩ is exactly 0.5 (analytic: |⟨Φ+|00⟩|² = 1/2)."""
    assert tracker2.fidelity(0, 1, "phi+") == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Entanglement
# ---------------------------------------------------------------------------


@pytest.mark.physics_regression
def test_entangle_creates_unit_fidelity_bell_pair(tracker2: TensorStateTracker) -> None:
    """H + CNOT on |00⟩ produces |Φ+⟩ with fidelity = 1."""
    f = tracker2.entangle(0, 1, "phi+")
    assert f == pytest.approx(1.0, abs=1e-12)


@pytest.mark.physics_regression
def test_bell_pair_purity_is_unity(bell_tracker: TensorStateTracker) -> None:
    """|Φ+⟩ is a pure state — global purity must equal 1."""
    assert bell_tracker.purity() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.physics_regression
def test_bell_pair_entropy_is_maximal(bell_tracker: TensorStateTracker) -> None:
    """Single-qubit entropy of q0 in |Φ+⟩ is log(2) in nats (maximally mixed marginal)."""
    assert bell_tracker.entropy(0) == pytest.approx(np.log(2), abs=1e-10)


# ---------------------------------------------------------------------------
# 3. Pauli channel — fidelity analytic limits
# ---------------------------------------------------------------------------


@pytest.mark.physics_regression
def test_identity_ptm_preserves_fidelity(bell_tracker: TensorStateTracker) -> None:
    """Applying the identity PTM [1,1,1,1] leaves Bell fidelity unchanged."""
    f_before = bell_tracker.fidelity(0, 1, "phi+")
    identity_ptm = np.array([1.0, 1.0, 1.0, 1.0])
    bell_tracker.apply_channel(0, identity_ptm)
    assert bell_tracker.fidelity(0, 1, "phi+") == pytest.approx(f_before, abs=1e-10)


@pytest.mark.physics_regression
def test_depolarising_channel_fidelity(bell_tracker: TensorStateTracker) -> None:
    """depolarising_ptm(0.5) on q0 of |Φ+⟩ → F(Φ+) = 0.625.

    Analytic: ρ_out = 0.625|Φ+⟩⟨Φ+| + 0.125(|Ψ+⟩⟨Ψ+| + |Ψ-⟩⟨Ψ-| + |Φ-⟩⟨Φ-|)
    F = ⟨Φ+|ρ_out|Φ+⟩ = 0.625.
    """
    bell_tracker.apply_channel(0, depolarising_ptm(0.5))
    assert bell_tracker.fidelity(0, 1, "phi+") == pytest.approx(0.625, abs=1e-10)


@pytest.mark.physics_regression
def test_dephasing_channel_fidelity(bell_tracker: TensorStateTracker) -> None:
    """dephasing_ptm(0.3) on q0 of |Φ+⟩ → F(Φ+) = 0.7.

    Analytic: ρ_out = 0.7|Φ+⟩⟨Φ+| + 0.3|Φ-⟩⟨Φ-| → F = 0.7.
    """
    bell_tracker.apply_channel(0, dephasing_ptm(0.3))
    assert bell_tracker.fidelity(0, 1, "phi+") == pytest.approx(0.7, abs=1e-10)


@pytest.mark.physics_regression
def test_depolarising_purity(bell_tracker: TensorStateTracker) -> None:
    """depolarising_ptm(0.5) on |Φ+⟩ → purity = 0.4375.

    Analytic: Tr(ρ²) = 0.625² + 3×0.125² = 0.390625 + 0.046875 = 0.4375.
    """
    bell_tracker.apply_channel(0, depolarising_ptm(0.5))
    assert bell_tracker.purity() == pytest.approx(0.4375, abs=1e-10)


@pytest.mark.physics_regression
def test_invalid_ptm_raises_value_error(tracker2: TensorStateTracker) -> None:
    """apply_channel with an unphysical PTM raises ValueError."""
    bad_ptm = np.array([1.0, 2.0, 0.0, 0.0])  # |λx| > 1, violates CP
    with pytest.raises(ValueError, match="PTM"):
        tracker2.apply_channel(0, bad_ptm)


# ---------------------------------------------------------------------------
# 4. Step log
# ---------------------------------------------------------------------------


@pytest.mark.physics_regression
def test_apply_channel_logs_step(tracker2: TensorStateTracker) -> None:
    """apply_channel appends exactly one step with correct metadata."""
    ptm = depolarising_ptm(0.1)
    tracker2.apply_channel(0, ptm, t=1.5, link_id="link_ab")
    log = tracker2.step_log()
    assert len(log) == 1
    assert log[0].operation == "channel"
    assert log[0].t == pytest.approx(1.5)
    assert log[0].link_id == "link_ab"


@pytest.mark.physics_regression
def test_step_log_ptm_stored_correctly(tracker2: TensorStateTracker) -> None:
    """PTM stored in the step log matches the one that was applied."""
    ptm = dephasing_ptm(0.2)
    tracker2.apply_channel(0, ptm, t=2.0)
    log = tracker2.step_log()
    assert log[0].ptm_applied is not None
    np.testing.assert_allclose(log[0].ptm_applied, ptm, atol=1e-15)


@pytest.mark.physics_regression
def test_step_log_node_and_link_id_preserved(tracker2: TensorStateTracker) -> None:
    """node_id and link_id metadata survive the step log round-trip."""
    tracker2.apply_channel(
        0, depolarising_ptm(0.1), t=3.0, link_id="link_xy", node_id="node_a"
    )
    step = tracker2.step_log()[0]
    assert step.node_id == "node_a"
    assert step.link_id == "link_xy"


@pytest.mark.physics_regression
def test_step_log_is_a_copy(tracker2: TensorStateTracker) -> None:
    """step_log() returns a copy; mutating the returned list does not affect internals."""
    tracker2.apply_channel(0, depolarising_ptm(0.1))
    log = tracker2.step_log()
    log.clear()
    # The internal log must still have 1 entry.
    assert len(tracker2.step_log()) == 1


# ---------------------------------------------------------------------------
# 5. Fidelity timeseries and reset
# ---------------------------------------------------------------------------


@pytest.mark.physics_regression
def test_fidelity_timeseries_channel_ops(tracker2: TensorStateTracker) -> None:
    """fidelity_timeseries('channel') returns (t, purity) pairs in order."""
    tracker2.entangle(0, 1)
    tracker2.apply_channel(0, depolarising_ptm(0.1), t=1.0)
    tracker2.apply_channel(0, dephasing_ptm(0.2), t=2.0)
    ts = tracker2.fidelity_timeseries(operation="channel")
    assert len(ts) == 2
    assert ts[0][0] == pytest.approx(1.0)
    assert ts[1][0] == pytest.approx(2.0)
    # Purity values must be in (0, 1].
    assert 0.0 < ts[0][1] <= 1.0
    assert 0.0 < ts[1][1] <= 1.0


@pytest.mark.physics_regression
def test_fidelity_timeseries_filters_by_operation(tracker2: TensorStateTracker) -> None:
    """fidelity_timeseries excludes entangle steps when filtering on 'channel'."""
    tracker2.entangle(0, 1)  # logged as "entangle"
    tracker2.apply_channel(0, depolarising_ptm(0.05), t=5.0)
    ts = tracker2.fidelity_timeseries(operation="channel")
    assert len(ts) == 1
    assert ts[0][0] == pytest.approx(5.0)


@pytest.mark.physics_regression
def test_reset_reinitializes_state(tracker2: TensorStateTracker) -> None:
    """After entangle + channel + reset, state returns to |00⟩ and log is cleared."""
    tracker2.entangle(0, 1)
    tracker2.apply_channel(0, depolarising_ptm(0.3))
    tracker2.reset()
    assert tracker2.purity() == pytest.approx(1.0, abs=1e-12)
    assert tracker2.fidelity(0, 1, "phi+") == pytest.approx(0.5, abs=1e-12)
    assert tracker2.step_log() == []


# ---------------------------------------------------------------------------
# 6. Measurement
# ---------------------------------------------------------------------------


@pytest.mark.physics_regression
def test_measure_valid_outcome_and_probability(
    bell_tracker: TensorStateTracker,
) -> None:
    """measure() returns outcome ∈ {0,1} and a valid Born probability."""
    outcome, prob = bell_tracker.measure(0, basis="Z")
    assert outcome in (0, 1)
    assert 0.0 < prob <= 1.0


@pytest.mark.physics_regression
def test_measure_logs_step(bell_tracker: TensorStateTracker) -> None:
    """measure() appends a 'measure' step to the log after entangle's step."""
    bell_tracker.measure(0)
    log = bell_tracker.step_log()
    # bell_tracker fixture already called entangle(), so len >= 2
    assert len(log) == 2
    assert log[-1].operation == "measure"
    assert log[-1].ptm_applied is None


# ---------------------------------------------------------------------------
# 7. min_eigenvalue monitor (Stage 4, §8)
# ---------------------------------------------------------------------------


@pytest.mark.physics_regression
def test_min_eigenvalue_pure_state_nonnegative(tracker2: TensorStateTracker) -> None:
    """min_eigenvalue() of the initial |00⟩ state must be ≥ 0."""
    assert tracker2.min_eigenvalue() >= 0.0


@pytest.mark.physics_regression
def test_min_eigenvalue_bell_state_nonnegative(bell_tracker: TensorStateTracker) -> None:
    """|Φ+⟩ is a valid density matrix — all eigenvalues must be ≥ 0."""
    assert bell_tracker.min_eigenvalue() >= 0.0


@pytest.mark.physics_regression
def test_min_eigenvalue_after_depolarising_still_valid(
    bell_tracker: TensorStateTracker,
) -> None:
    """After a Pauli channel the DM stays physical — min eigenvalue ≥ -1e-10."""
    bell_tracker.apply_channel(0, depolarising_ptm(0.3))
    assert bell_tracker.min_eigenvalue() >= -1e-10


@pytest.mark.physics_regression
def test_min_eigenvalue_is_float_and_approx_zero_for_pure(
    tracker2: TensorStateTracker,
) -> None:
    """TensorStateTracker.min_eigenvalue() delegates to the adapter and returns a float.

    |00⟩ is a pure state with eigenvalues {1, 0, 0, 0} — min must be ≈ 0.
    """
    result = tracker2.min_eigenvalue()
    assert isinstance(result, float)
    assert result == pytest.approx(0.0, abs=1e-12)
