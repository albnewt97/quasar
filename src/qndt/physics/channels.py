"""PTM algebra: channel constructors, composition, inversion, and validation.

All functions operate on the diagonal PTM representation ``[1, λx, λy, λz]``
as defined in docs/architecture.md §5.1.  No I/O, no Qt, no state.
"""
from __future__ import annotations

import numpy as np

from qndt.core.context import PauliRateVector

# Upper bound for the Φ squashing map applied before returning recovered rates.
# Pauli rates are in [0, 0.5) per §5.2.
_RATE_MAX: float = 0.5 - 1e-15


def ptm_to_pauli_rates(ptm: np.ndarray) -> PauliRateVector:
    """Recover (px, py, pz) from a diagonal PTM ``[1, λx, λy, λz]``.

    Solves the linear system from §5.1 exactly:

    ``px = (1 + λx - λy - λz) / 4``
    ``py = (1 - λx + λy - λz) / 4``
    ``pz = (1 - λx - λy + λz) / 4``

    Results are clamped to ``[0, 0.5)`` (the Φ squashing range from §5.2)
    before being wrapped in a ``PauliRateVector``.

    Args:
        ptm: Length-4 array ``[1, λx, λy, λz]``.

    Returns:
        ``PauliRateVector`` with recovered Pauli error probabilities.

    Raises:
        ValueError: If the clamped rates still violate ``PauliRateVector``
            invariants (indicates a physically invalid input PTM).
    """
    lx, ly, lz = float(ptm[1]), float(ptm[2]), float(ptm[3])
    px = float(np.clip((1.0 + lx - ly - lz) / 4.0, 0.0, _RATE_MAX))
    py = float(np.clip((1.0 - lx + ly - lz) / 4.0, 0.0, _RATE_MAX))
    pz = float(np.clip((1.0 - lx - ly + lz) / 4.0, 0.0, _RATE_MAX))
    return PauliRateVector(px=px, py=py, pz=pz)


def compose_ptms(*ptms: np.ndarray) -> np.ndarray:
    """Return the Hadamard (element-wise) product of arbitrarily many PTMs.

    Exact for Pauli channels (§5.1, §3.2).  Degenerate cases:

    - Zero PTMs → identity PTM ``np.ones(4)``
    - One PTM → the PTM itself (same object returned)

    Args:
        *ptms: Zero or more length-4 diagonal PTM arrays.

    Returns:
        Length-4 numpy array representing the composed channel.
    """
    if len(ptms) == 0:
        return np.ones(4, dtype=np.float64)
    if len(ptms) == 1:
        return ptms[0]
    result = np.ones(4, dtype=np.float64)
    for ptm in ptms:
        result = result * ptm
    return result


def depolarising_ptm(p: float) -> np.ndarray:
    """Return the diagonal PTM for a standard depolarising channel.

    Models ``E(ρ) = (1-p)ρ + (p/4)(ρ + XρX + YρY + ZρZ)`` with
    ``px = py = pz = p/4``.  Valid for ``p ∈ [0, 1]``.

    Args:
        p: Total error probability.  ``p=0`` → identity; ``p=1`` → maximally
            mixed output (fidelity 0.25).

    Returns:
        Length-4 PTM ``[1, 1-p, 1-p, 1-p]``.
    """
    return PauliRateVector(px=p / 4.0, py=p / 4.0, pz=p / 4.0).ptm()


def dephasing_ptm(pz: float) -> np.ndarray:
    """Return the diagonal PTM for a pure dephasing (Z-error only) channel.

    Models ``E(ρ) = (1-pz)ρ + pz·ZρZ`` with ``px=0, py=0``.

    The Z coherence (λz) is preserved; X and Y coherences (λx = λy) decay:

    ``λx = λy = 1 - 2·pz``  ``λz = 1``

    Args:
        pz: Z-error probability.  ``pz=0`` → identity; ``pz=0.5`` → fully
            dephased (λx = λy = 0).

    Returns:
        Length-4 PTM ``[1, 1-2·pz, 1-2·pz, 1]``.
    """
    return PauliRateVector(px=0.0, py=0.0, pz=pz).ptm()


def ptm_fidelity(ptm: np.ndarray) -> float:
    """Compute the process fidelity (entanglement fidelity) of a Pauli channel.

    ``F_process = pI = (1 + λx + λy + λz) / 4``

    This is the **process fidelity** (also called entanglement fidelity), equal
    to the identity-error probability ``pI = 1 - px - py - pz``.  It is NOT
    the average gate fidelity; the two are related by:

    ``F_avg = (d·F_process + 1)/(d+1) = (2·F_process + 1)/3``  for d=2.

    For a fully depolarising channel, F_process = 0.25 while F_avg = 0.5.

    Valid range is ``[0.0, 1.0]`` for general Pauli channels (e.g. a pure
    X-flip channel has F_process = 0).  ``F=1`` for the identity channel.

    Args:
        ptm: Length-4 diagonal PTM ``[1, λx, λy, λz]``.

    Returns:
        Process fidelity as a float in ``[0.0, 1.0]``.
    """
    return float((1.0 + ptm[1] + ptm[2] + ptm[3]) / 4.0)


def validate_ptm(ptm: np.ndarray) -> bool:
    """Return ``True`` iff ``ptm`` represents a valid Pauli channel PTM.

    Checks (in order):

    1. Shape is ``(4,)``.
    2. ``ptm[0] == 1.0`` (normalisation).
    3. ``|ptm[i]| <= 1.0`` for ``i in 1..3`` (eigenvalue bound).
    4. Implied Pauli rates are all non-negative (complete positivity).

    Args:
        ptm: Array to validate.

    Returns:
        ``True`` if all checks pass, ``False`` otherwise.
    """
    if ptm.shape != (4,):
        return False
    if ptm[0] != 1.0:
        return False
    lx, ly, lz = float(ptm[1]), float(ptm[2]), float(ptm[3])
    if abs(lx) > 1.0 or abs(ly) > 1.0 or abs(lz) > 1.0:
        return False
    # Complete positivity: all Pauli rates must be non-negative.
    # Tolerance absorbs IEEE 754 cancellation errors (e.g. 1+λ-λ-1 ≈ -1e-16).
    _cp_tol: float = 1e-10
    px = (1.0 + lx - ly - lz) / 4.0
    py = (1.0 - lx + ly - lz) / 4.0
    pz = (1.0 - lx - ly + lz) / 4.0
    p_id = (1.0 + lx + ly + lz) / 4.0
    return (
        px >= -_cp_tol
        and py >= -_cp_tol
        and pz >= -_cp_tol
        and p_id >= -_cp_tol
    )
