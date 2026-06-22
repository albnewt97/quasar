"""Tests for ChannelComposer, NoiseContributor protocol, and PauliRateVector.ptm().

Covers:
- Identity (no contributors) returns ones(4)
- Single contributor PTM passthrough via mock
- Exactness of Hadamard composition against first-principles Pauli probability arithmetic
- Validation of invalid Pauli rates
- Runtime isinstance check for the @runtime_checkable NoiseContributor protocol
- Cross-module: Hadamard PTM == sequential Kraus (validates the central architecture claim)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from qndt.core.composer import ChannelComposer, NoiseContributor
from qndt.core.context import OpContext, PauliRateVector
from qndt.physics.channels import compose_ptms, dephasing_ptm, depolarising_ptm
from qndt.quantum.backends.quimb_adapter import MPDOConfig, QuimbAdapter

# Minimal context reused across tests — field values do not affect PTM composition.
_CTX = OpContext(
    link_id="link_test",
    node_id=None,
    t=0.0,
    lambda_q=1550.0,
    gate_width=1e-9,
)


class _PauliContributor:
    """Thin adapter exposing a PauliRateVector as a NoiseContributor.

    Used in test_composition_exactness to exercise real PTM arithmetic
    without mock indirection.
    """

    def __init__(self, rates: PauliRateVector) -> None:
        self._rates = rates

    def ptm(self, ctx: OpContext) -> np.ndarray:
        return self._rates.ptm()


# ---------------------------------------------------------------------------
# test_identity
# ---------------------------------------------------------------------------

def test_identity() -> None:
    """A composer with no contributors must return the identity PTM ones(4)."""
    composer = ChannelComposer()
    result = composer.effective_ptm(_CTX)
    np.testing.assert_array_equal(result, np.ones(4))


# ---------------------------------------------------------------------------
# test_single_pauli_z
# ---------------------------------------------------------------------------

def test_single_pauli_z() -> None:
    """A single pure-Z channel mock contributor must pass through effective_ptm.

    PauliRateVector(0, 0, 0.1):
        λx = 1 - 2(0 + 0.1) = 0.8
        λy = 1 - 2(0 + 0.1) = 0.8
        λz = 1 - 2(0 + 0)   = 1.0
    expected PTM = [1.0, 0.8, 0.8, 1.0].
    """
    expected = PauliRateVector(px=0.0, py=0.0, pz=0.1).ptm()

    mock_contributor = MagicMock()
    mock_contributor.ptm.return_value = expected

    composer = ChannelComposer()
    composer.register(mock_contributor)

    result = composer.effective_ptm(_CTX)
    np.testing.assert_array_almost_equal(result, expected)
    mock_contributor.ptm.assert_called_once_with(_CTX)


# ---------------------------------------------------------------------------
# test_composition_exactness
# ---------------------------------------------------------------------------

def test_composition_exactness() -> None:
    """Hadamard PTM product must equal the PTM of the analytically composed channel.

    For two Pauli channels (px1,py1,pz1) and (px2,py2,pz2), the composed
    channel (channel-1 first, then channel-2) has Pauli probabilities given
    by the Klein-four group convolution:

        P'x = pI1·px2 + px1·pI2 + py1·pz2 + pz1·py2
        P'y = pI1·py2 + py1·pI2 + px1·pz2 + pz1·px2
        P'z = pI1·pz2 + pz1·pI2 + px1·py2 + py1·px2

    The PTM of this composed channel must equal the element-wise product
    R1 ⊙ R2 returned by ChannelComposer (§3.2, §5.1).
    """
    prv1 = PauliRateVector(px=0.05, py=0.03, pz=0.02)
    prv2 = PauliRateVector(px=0.04, py=0.02, pz=0.06)

    composer = ChannelComposer()
    composer.register(_PauliContributor(prv1))
    composer.register(_PauliContributor(prv2))
    composed = composer.effective_ptm(_CTX)

    pI1 = 1.0 - prv1.px - prv1.py - prv1.pz
    pI2 = 1.0 - prv2.px - prv2.py - prv2.pz

    p_x = pI1 * prv2.px + prv1.px * pI2 + prv1.py * prv2.pz + prv1.pz * prv2.py
    p_y = pI1 * prv2.py + prv1.py * pI2 + prv1.px * prv2.pz + prv1.pz * prv2.px
    p_z = pI1 * prv2.pz + prv1.pz * pI2 + prv1.px * prv2.py + prv1.py * prv2.px

    direct = PauliRateVector(px=p_x, py=p_y, pz=p_z).ptm()
    np.testing.assert_array_almost_equal(composed, direct)


# ---------------------------------------------------------------------------
# test_invalid_rates
# ---------------------------------------------------------------------------

def test_invalid_rates() -> None:
    """PauliRateVector must raise ValueError when px + py + pz > 1.0."""
    with pytest.raises(ValueError, match="Sum of Pauli rates"):
        PauliRateVector(px=0.5, py=0.3, pz=0.3)  # sum = 1.1


def test_invalid_rates_negative() -> None:
    """PauliRateVector must raise ValueError when any rate is negative."""
    with pytest.raises(ValueError, match="non-negative"):
        PauliRateVector(px=-0.1, py=0.0, pz=0.0)


# ---------------------------------------------------------------------------
# test_noise_contributor_protocol
# ---------------------------------------------------------------------------

def test_noise_contributor_protocol() -> None:
    """Any object with a ptm() method must satisfy the runtime-checkable protocol."""
    mock_contributor = MagicMock()
    mock_contributor.ptm = MagicMock(return_value=np.ones(4))
    assert isinstance(mock_contributor, NoiseContributor)


# ---------------------------------------------------------------------------
# test_hadamard_ptm_equals_sequential_kraus  (cross-module §3.2 / MODULE 7)
# ---------------------------------------------------------------------------

def test_hadamard_ptm_equals_sequential_kraus() -> None:
    """Hadamard PTM composition must equal sequential Kraus application to 1e-12.

    This validates the central architectural claim in §3.2: composing two
    Pauli channels via element-wise PTM product is EXACT (not an approximation)
    and produces the same quantum state as applying the channels one after
    the other.

    Test procedure:
      1. Start a 1-qubit state in |+⟩.
      2. Sequential path: apply channel A then channel B via apply_pauli_channel.
      3. Composed path: apply compose_ptms(A, B) as a single channel.
      4. Assert the resulting density matrices agree element-wise to 1e-12.

    Verified for both 2-channel and 3-channel composition.
    """
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    cfg = MPDOConfig(n_sites=1)

    ptm_A = dephasing_ptm(0.3)
    ptm_B = depolarising_ptm(0.2)
    ptm_C = PauliRateVector(px=0.02, py=0.05, pz=0.01).ptm()

    # --- 2-channel test ---
    adapter_seq = QuimbAdapter(cfg)
    adapter_seq.apply_single_qubit_gate(0, H)
    adapter_seq.apply_pauli_channel(0, ptm_A)
    adapter_seq.apply_pauli_channel(0, ptm_B)

    adapter_had = QuimbAdapter(cfg)
    adapter_had.apply_single_qubit_gate(0, H)
    adapter_had.apply_pauli_channel(0, compose_ptms(ptm_A, ptm_B))

    diff2 = np.max(np.abs(adapter_seq._rho - adapter_had._rho))
    assert diff2 < 1e-12, (
        f"2-channel Hadamard≠Kraus: max element diff = {diff2:.2e}"
    )

    # --- 3-channel test ---
    adapter_seq3 = QuimbAdapter(cfg)
    adapter_seq3.apply_single_qubit_gate(0, H)
    adapter_seq3.apply_pauli_channel(0, ptm_A)
    adapter_seq3.apply_pauli_channel(0, ptm_B)
    adapter_seq3.apply_pauli_channel(0, ptm_C)

    adapter_had3 = QuimbAdapter(cfg)
    adapter_had3.apply_single_qubit_gate(0, H)
    adapter_had3.apply_pauli_channel(0, compose_ptms(ptm_A, ptm_B, ptm_C))

    diff3 = np.max(np.abs(adapter_seq3._rho - adapter_had3._rho))
    assert diff3 < 1e-12, (
        f"3-channel Hadamard≠Kraus: max element diff = {diff3:.2e}"
    )
