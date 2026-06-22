"""Dense-fallback MPDO backend wrapping quimb conventions.

All quantum state is stored as a full 2^n × 2^n complex numpy density matrix.
quimb is imported lazily but the tensor-network path is not activated;
self._using_dense_fallback is always True.  This gives exact results for the
system sizes used in tests and examples (n ≤ 10).

§3.4 State Ownership Law: QuimbAdapter is the only place self._rho lives.
TensorStateTracker is the only class that owns a QuimbAdapter.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import numpy as np

from qndt.physics.channels import ptm_to_pauli_rates

_log = logging.getLogger(__name__)

# Eigenvalue threshold below which we warn; floating-point truncation
# introduces ~1e-15 negatives that are harmless — warn only if worse.
_POSITIVITY_THRESHOLD: float = -1e-10

# ---------------------------------------------------------------------------
# Module-level Pauli matrices (shared constant; never mutated)
# ---------------------------------------------------------------------------
_I: np.ndarray = np.eye(2, dtype=complex)
_X: np.ndarray = np.array([[0, 1], [1, 0]], dtype=complex)
_Y: np.ndarray = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z: np.ndarray = np.array([[1, 0], [0, -1]], dtype=complex)

# ---------------------------------------------------------------------------
# Bell state kets (4-vectors in the |q1,q2⟩ computational basis)
# ---------------------------------------------------------------------------
_BELL_KETS: dict[str, np.ndarray] = {
    "phi+": np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    "phi-": np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
    "psi+": np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
    "psi-": np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _kron_gate(n_sites: int, qubit: int, gate: np.ndarray) -> np.ndarray:
    """Embed a 2×2 gate at position *qubit* in the 2^n full-system space.

    Args:
        n_sites: Total number of qubits.
        qubit: Zero-based qubit index (qubit 0 is the MSB in the state integer).
        gate: 2×2 unitary or operator.

    Returns:
        2^n × 2^n matrix with *gate* at the right position and identities elsewhere.
    """
    mats: list[np.ndarray] = [_I] * n_sites
    mats[qubit] = gate
    result: np.ndarray = functools.reduce(np.kron, mats)
    return result


def _partial_trace(rho: np.ndarray, keep: list[int], n_sites: int) -> np.ndarray:
    """Return the reduced density matrix over the qubits in *keep*.

    Traces out all qubits NOT in *keep* using np.einsum with integer subscripts.
    The output ordering follows sorted(*keep*).

    Args:
        rho: Full 2^n × 2^n density matrix.
        keep: Qubit indices to retain.
        n_sites: Total number of qubits.

    Returns:
        Reduced density matrix of shape ``(2^k, 2^k)`` where ``k = len(keep)``.
    """
    k = len(keep)
    trace_out = [i for i in range(n_sites) if i not in keep]

    # Reshape to tensor with 2*n_sites axes: [r0,r1,...,rn-1, c0,c1,...,cn-1]
    rho_r = rho.reshape([2] * (2 * n_sites))

    # Build einsum subscripts using integers (numpy supports this form)
    row_subs = list(range(n_sites))
    col_subs = list(range(n_sites, 2 * n_sites))

    # For traced-out qubits, set the col subscript equal to the row subscript
    # (repeated index → summation → partial trace).
    for q in trace_out:
        col_subs[q] = row_subs[q]

    all_subs = row_subs + col_subs
    out_subs = (
        [row_subs[q] for q in sorted(keep)]
        + [col_subs[q] for q in sorted(keep)]
    )

    result: np.ndarray = np.einsum(rho_r, all_subs, out_subs)
    return result.reshape(2**k, 2**k).astype(complex)


# ---------------------------------------------------------------------------
# MPDOConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MPDOConfig:
    """Configuration for the MPDO/dense quantum state backend.

    Args:
        n_sites: Number of qubits. Must be ≥ 1.
        chi_max: Maximum MPS bond dimension (reserved for future TN mode).
        kappa_max: Maximum Kraus dimension (reserved for future TN mode).
        cutoff: SVD truncation threshold (reserved for future TN mode).

    Raises:
        ValueError: If any parameter violates its constraint.
    """

    n_sites: int
    chi_max: int = 32
    kappa_max: int = 8
    cutoff: float = 1e-10

    def __post_init__(self) -> None:
        if self.n_sites < 1:
            raise ValueError(f"n_sites must be >= 1; got {self.n_sites}")
        if self.chi_max < 1:
            raise ValueError(f"chi_max must be >= 1; got {self.chi_max}")
        if self.kappa_max < 1:
            raise ValueError(f"kappa_max must be >= 1; got {self.kappa_max}")


# ---------------------------------------------------------------------------
# QuimbAdapter
# ---------------------------------------------------------------------------

class QuimbAdapter:
    """Dense-fallback quantum state adapter.

    Always operates in dense-fallback mode (``self._using_dense_fallback = True``).
    The quimb tensor-network code path is imported lazily but not activated.

    All state is stored in ``self._rho``, a ``(2^n, 2^n)`` complex numpy array.
    This class is the only place quantum state lives (§3.4 State Ownership Law).

    Args:
        config: MPDO configuration.
    """

    def __init__(self, config: MPDOConfig) -> None:
        self._config = config
        self._n_sites = config.n_sites
        self._using_dense_fallback: bool = True  # always True in current impl
        self._rng = np.random.default_rng(42)

        dim = 2 ** self._n_sites
        self._rho: np.ndarray = np.zeros((dim, dim), dtype=complex)
        self.initialise_zeros()

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def initialise_zeros(self) -> None:
        """Reset all qubits to the |0⟩⟨0|^⊗n product state."""
        dim = 2 ** self._n_sites
        self._rho = np.zeros((dim, dim), dtype=complex)
        self._rho[0, 0] = 1.0

    # ------------------------------------------------------------------
    # Gate operations
    # ------------------------------------------------------------------

    def apply_single_qubit_gate(self, qubit: int, gate: np.ndarray) -> None:
        """Apply a 2×2 unitary to a single qubit.

        Args:
            qubit: Target qubit index in ``[0, n_sites)``.
            gate: 2×2 unitary matrix.

        Raises:
            ValueError: If *gate* is not shape ``(2, 2)`` or *qubit* is out of range.
        """
        if gate.shape != (2, 2):
            raise ValueError(f"Single-qubit gate must be (2,2); got {gate.shape}")
        if not (0 <= qubit < self._n_sites):
            raise ValueError(
                f"Qubit index {qubit} out of range [0, {self._n_sites})"
            )
        U = _kron_gate(self._n_sites, qubit, gate)
        self._rho = U @ self._rho @ U.conj().T

    def apply_two_qubit_gate(self, q1: int, q2: int, gate: np.ndarray) -> None:
        """Apply a 4×4 unitary to the (q1, q2) subspace.

        Handles non-adjacent qubits correctly by building the full 2^n unitary
        via explicit index manipulation.  Convention: the first index of *gate*
        corresponds to q1 and the second to q2 (q1 is the MSB in the gate space).

        Args:
            q1: First qubit index.
            q2: Second qubit index.
            gate: 4×4 unitary matrix.

        Raises:
            ValueError: If *gate* is not shape ``(4, 4)``.
        """
        if gate.shape != (4, 4):
            raise ValueError(f"Two-qubit gate must be (4,4); got {gate.shape}")
        n = self._n_sites
        dim = 2 ** n
        U_full = np.zeros((dim, dim), dtype=complex)

        for col in range(dim):
            b1_col = (col >> (n - 1 - q1)) & 1
            b2_col = (col >> (n - 1 - q2)) & 1
            input_idx = b1_col * 2 + b2_col

            mask_q1 = 1 << (n - 1 - q1)
            mask_q2 = 1 << (n - 1 - q2)

            for row in range(dim):
                diff = (col ^ row) & ~mask_q1 & ~mask_q2
                if diff != 0:
                    continue
                r1 = (row >> (n - 1 - q1)) & 1
                r2 = (row >> (n - 1 - q2)) & 1
                output_idx = r1 * 2 + r2
                U_full[row, col] = gate[output_idx, input_idx]

        self._rho = U_full @ self._rho @ U_full.conj().T

    def apply_pauli_channel(self, qubit: int, ptm: np.ndarray) -> None:
        """Apply a Pauli channel specified by its diagonal PTM.

        Converts the PTM ``[1, λx, λy, λz]`` to Pauli rates and applies:
        ``ρ → pI·ρ + px·XρX† + py·YρY† + pz·ZρZ†``

        Args:
            qubit: Target qubit index.
            ptm: Length-4 diagonal Pauli Transfer Matrix.
        """
        rates = ptm_to_pauli_rates(ptm)
        px, py, pz = rates.px, rates.py, rates.pz
        p_id = 1.0 - px - py - pz

        X_f = _kron_gate(self._n_sites, qubit, _X)
        Y_f = _kron_gate(self._n_sites, qubit, _Y)
        Z_f = _kron_gate(self._n_sites, qubit, _Z)

        rho = self._rho
        self._rho = (
            p_id * rho
            + px * (X_f @ rho @ X_f.conj().T)
            + py * (Y_f @ rho @ Y_f.conj().T)
            + pz * (Z_f @ rho @ Z_f.conj().T)
        )

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_qubit(
        self, qubit: int, basis: str = "Z"
    ) -> tuple[int, float]:
        """Projective measurement of a qubit with state collapse.

        Rotates into the requested basis, projects in the Z basis, then
        rotates back so the post-measurement state is an eigenstate of the
        original observable.

        Args:
            qubit: Qubit to measure.
            basis: ``"Z"`` (default), ``"X"``, or ``"Y"``.

        Returns:
            ``(outcome, probability)`` where ``outcome ∈ {0, 1}`` and
            ``probability`` is the Born-rule probability of that outcome.
        """
        # Rotation gates to convert X/Y eigenstates into Z eigenstates.
        _H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        _S_dag = np.array([[1, 0], [0, -1j]], dtype=complex)
        # U_y: maps |+y⟩→|0⟩, |−y⟩→|1⟩
        _U_y = np.array([[1, -1j], [1, 1j]], dtype=complex) / np.sqrt(2)

        if basis == "X":
            self.apply_single_qubit_gate(qubit, _H)
        elif basis == "Y":
            self.apply_single_qubit_gate(qubit, _U_y @ _S_dag)

        # Measure in Z: P(0) = Tr(|0><0|_qubit ⊗ I_rest · ρ)
        proj0 = np.array([[1, 0], [0, 0]], dtype=complex)
        P0_full = _kron_gate(self._n_sites, qubit, proj0)
        p0 = float(np.real(np.trace(P0_full @ self._rho)))
        p0 = max(0.0, min(1.0, p0))

        outcome = int(self._rng.choice(2, p=[p0, 1.0 - p0]))
        prob = p0 if outcome == 0 else (1.0 - p0)

        if outcome == 0:
            proj_full = P0_full
        else:
            proj1 = np.array([[0, 0], [0, 1]], dtype=complex)
            proj_full = _kron_gate(self._n_sites, qubit, proj1)

        if prob > 1e-15:
            self._rho = (proj_full @ self._rho @ proj_full.conj().T) / prob

        # Rotate back so the post-measurement state is an eigenstate of the
        # original observable (X or Y).
        if basis == "X":
            self.apply_single_qubit_gate(qubit, _H)
        elif basis == "Y":
            rot = _U_y @ _S_dag
            self.apply_single_qubit_gate(qubit, rot.conj().T)

        return outcome, prob

    # ------------------------------------------------------------------
    # Fidelity and state information
    # ------------------------------------------------------------------

    def fidelity_with_bell(
        self, q1: int, q2: int, bell_state: str = "phi+"
    ) -> float:
        """Fidelity of the reduced DM of (q1, q2) with a Bell state.

        Args:
            q1: First qubit index.
            q2: Second qubit index.
            bell_state: One of ``"phi+"``, ``"phi-"``, ``"psi+"``, ``"psi-"``.

        Returns:
            Fidelity in ``[0, 1]``.
        """
        if bell_state not in _BELL_KETS:
            raise ValueError(
                f"Unknown Bell state {bell_state!r}; "
                f"choose from {list(_BELL_KETS)}"
            )
        ket = _BELL_KETS[bell_state]
        rdm = self.reduced_dm(sorted([q1, q2]))
        fidelity = float(np.real(ket.conj() @ rdm @ ket))
        return max(0.0, min(1.0, fidelity))

    def fidelity_with_product(self, target_states: list[np.ndarray]) -> float:
        """Fidelity of the current state with a pure product state.

        Args:
            target_states: List of *n_sites* pure state vectors, each shape ``(2,)``.

        Returns:
            Fidelity in ``[0, 1]``.
        """
        psi: np.ndarray = functools.reduce(
            np.kron,
            [s.astype(complex) for s in target_states],
        )
        fidelity = float(np.real(psi.conj() @ self._rho @ psi))
        return max(0.0, min(1.0, fidelity))

    def reduced_dm(self, qubits: list[int]) -> np.ndarray:
        """Reduced density matrix for the specified qubits.

        Args:
            qubits: Qubit indices to retain.  Order is normalised to sorted.

        Returns:
            Complex array of shape ``(2^k, 2^k)`` where ``k = len(qubits)``.
        """
        return _partial_trace(self._rho, qubits, self._n_sites)

    def truncate(self, eps: float | None = None) -> dict[str, int]:  # noqa: ARG002
        """No-op in dense-fallback mode; returns bond dimension metadata.

        Args:
            eps: Unused (TN-mode truncation threshold).

        Returns:
            Dict with ``bond_dim`` and ``dense_fallback`` entries.
        """
        return {"bond_dim": 2 ** self._n_sites, "dense_fallback": 1}

    def purity(self) -> float:
        """Global state purity Tr(ρ²).

        Returns:
            1.0 for a pure state; 1/2^n for the maximally mixed state.
        """
        return float(np.real(np.trace(self._rho @ self._rho)))

    def min_eigenvalue(self) -> float:
        """Minimum eigenvalue of the global density matrix ρ.

        For a physically valid state this is ≥ 0.  MPDO truncation can
        introduce small negative eigenvalues; values below
        ``_POSITIVITY_THRESHOLD`` (-1e-10) indicate significant numerical
        error or an unphysical state (§8).

        Returns:
            Smallest eigenvalue of ``self._rho``; should be ≥ 0.
        """
        eigvals = np.linalg.eigvalsh(self._rho)
        return float(eigvals[0])

    def entropy(self, qubit: int) -> float:
        """Von Neumann entropy of the single-qubit reduced state.

        ``S = -Tr(ρ_q · log ρ_q)``  in nats (natural log).

        Logs a warning when the reduced DM has eigenvalues below
        ``_POSITIVITY_THRESHOLD`` (−1e-10); MPDO truncation is the usual
        cause.  Entropy is still computed from the positive eigenvalues.

        Args:
            qubit: Qubit index.

        Returns:
            Entropy in ``[0, log(2)]``.
        """
        rdm = self.reduced_dm([qubit])
        eigvals = np.linalg.eigvalsh(rdm)
        min_eig = float(eigvals[0])
        if min_eig < _POSITIVITY_THRESHOLD:
            _log.warning(
                "entropy(qubit=%d): reduced DM has negative eigenvalue %.3e; "
                "MPDO truncation may have violated positivity (§8).",
                qubit, min_eig,
            )
        positive = eigvals[eigvals > 1e-15]
        return float(-np.sum(positive * np.log(positive)))
