"""Physics regression tests for channels.py and kernels.py.

All tests are marked physics_regression and verify equations from docs/architecture.md §5.1.
"""
from __future__ import annotations

import numpy as np
import pytest

from qndt.core.context import PauliRateVector
from qndt.physics.channels import (
    compose_ptms,
    dephasing_ptm,
    depolarising_ptm,
    ptm_fidelity,
    ptm_to_pauli_rates,
    validate_ptm,
)
from qndt.physics.kernels import ExponentialKernel, GaussianKernel, LorentzianKernel

pytestmark = pytest.mark.physics_regression


# ---------------------------------------------------------------------------
# PTM round-trip
# ---------------------------------------------------------------------------


def test_ptm_roundtrip() -> None:
    """ptm_to_pauli_rates must recover the original rates within 1e-10."""
    original = PauliRateVector(px=0.05, py=0.03, pz=0.10)
    recovered = ptm_to_pauli_rates(original.ptm())
    assert recovered.px == pytest.approx(original.px, abs=1e-10)
    assert recovered.py == pytest.approx(original.py, abs=1e-10)
    assert recovered.pz == pytest.approx(original.pz, abs=1e-10)


# ---------------------------------------------------------------------------
# Depolarising channel
# ---------------------------------------------------------------------------


def test_depolarising_symmetry() -> None:
    """Depolarising PTM must have λx == λy == λz (all equal by symmetry)."""
    ptm = depolarising_ptm(0.12)
    assert ptm[1] == pytest.approx(ptm[2])
    assert ptm[2] == pytest.approx(ptm[3])


# ---------------------------------------------------------------------------
# Dephasing channel
# ---------------------------------------------------------------------------


def test_dephasing_structure() -> None:
    """Pure dephasing (Z-error only) channel structural checks.

    For px=0, py=0, pz=pz_val (§5.1):
      λx = 1 - 2(py + pz) = 1 - 2·pz_val   (X coherence decays)
      λy = 1 - 2(px + pz) = 1 - 2·pz_val   (Y coherence decays)
      λz = 1 - 2(px + py) = 1               (Z coherence preserved)
    """
    pz_val = 0.15
    ptm = dephasing_ptm(pz_val)
    assert ptm[1] == pytest.approx(ptm[2])                   # λx == λy
    assert ptm[1] == pytest.approx(1.0 - 2.0 * pz_val)      # common decay value
    assert ptm[3] == pytest.approx(1.0)                      # λz preserved


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_compose_associativity() -> None:
    """compose_ptms must be associative: (A⊙B)⊙C == A⊙(B⊙C) == A⊙B⊙C."""
    a = depolarising_ptm(0.05)
    b = dephasing_ptm(0.03)
    c = PauliRateVector(px=0.02, py=0.01, pz=0.04).ptm()

    lhs = compose_ptms(compose_ptms(a, b), c)
    mid = compose_ptms(a, b, c)
    rhs = compose_ptms(a, compose_ptms(b, c))

    np.testing.assert_array_almost_equal(lhs, mid)
    np.testing.assert_array_almost_equal(mid, rhs)


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------


def test_fidelity_identity() -> None:
    """Identity channel PTM ones(4) must have average gate fidelity == 1.0."""
    assert ptm_fidelity(np.ones(4)) == pytest.approx(1.0)


def test_fidelity_fully_depolarising() -> None:
    """Fully depolarising channel (p=1) must have fidelity == 0.25.

    With p=1: px=py=pz=0.25, λx=λy=λz=0, F=(1+0+0+0)/4=0.25.
    """
    assert ptm_fidelity(depolarising_ptm(1.0)) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_ptm_rejects_invalid() -> None:
    """PTM with |λx|=2 > 1 must be rejected by validate_ptm."""
    assert validate_ptm(np.array([1.0, 2.0, 0.0, 0.0])) is False


def test_validate_ptm_accepts_valid() -> None:
    """Valid depolarising PTM must pass validate_ptm."""
    assert validate_ptm(depolarising_ptm(0.2)) is True


def test_validate_ptm_rejects_wrong_norm() -> None:
    """PTM with ptm[0] != 1.0 must be rejected."""
    assert validate_ptm(np.array([0.9, 0.8, 0.8, 0.8])) is False


# ---------------------------------------------------------------------------
# ExponentialKernel
# ---------------------------------------------------------------------------


def test_exponential_kernel_decay() -> None:
    """ExponentialKernel.eval(0) == diag(1/τx, 1/τy, 1/τz); eval(1000) ≈ zero.

    After unit-area normalisation (Appendix A §3) each diagonal entry
    K_α(τ) = (1/τ_α)·exp(-τ/τ_α) peaks at 1/τ_α, not 1.
    """
    kernel = ExponentialKernel(tau_x=30.0, tau_y=30.0, tau_z=120.0)

    k0 = kernel.eval(0.0)
    expected_k0 = np.diag([1.0 / 30.0, 1.0 / 30.0, 1.0 / 120.0])
    np.testing.assert_array_almost_equal(k0, expected_k0)

    # tau=1000s is ~8× tau_z (slowest axis); (1/τz)·exp(-8.3)≈2e-6 — effectively zero.
    k_large = kernel.eval(1000.0)
    np.testing.assert_allclose(k_large, np.zeros((3, 3)), atol=1e-4)


# ---------------------------------------------------------------------------
# LorentzianKernel
# ---------------------------------------------------------------------------


def test_lorentzian_oscillation() -> None:
    """LorentzianKernel must produce sign changes confirming oscillatory behaviour."""
    kernel = LorentzianKernel(gamma=0.1, omega_0=1.0)

    val_0 = kernel.eval(0.0)          # cos(0)  = +1  → positive
    val_pi = kernel.eval(np.pi)       # cos(π)  = -1  → negative (times exp > 0)

    assert val_0[0, 0] > 0.0
    assert val_pi[0, 0] < 0.0


# ---------------------------------------------------------------------------
# tau < 0 guard (all kernel types)
# ---------------------------------------------------------------------------


def test_kernel_positive_tau_only() -> None:
    """All kernel types must raise ValueError when eval is called with tau < 0."""
    exp_kernel = ExponentialKernel(tau_x=30.0, tau_y=30.0, tau_z=120.0)
    lor_kernel = LorentzianKernel(gamma=0.1, omega_0=1.0)
    gau_kernel = GaussianKernel(sigma=10.0)

    with pytest.raises(ValueError, match="tau"):
        exp_kernel.eval(-1.0)
    with pytest.raises(ValueError, match="tau"):
        lor_kernel.eval(-1.0)
    with pytest.raises(ValueError, match="tau"):
        gau_kernel.eval(-1.0)
