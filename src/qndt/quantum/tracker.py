"""TensorStateTracker: the sole owner of quantum state in the simulation (§3.4).

Wraps QuimbAdapter (dense-fallback MPDO backend) and provides the high-level
simulation interface: entanglement creation, Pauli channel application,
measurement, and step logging.

§3.4 State Ownership Law: TensorStateTracker is the only class in qndt
(outside QuimbAdapter) that holds a reference to the quantum density matrix.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from qndt.physics.channels import validate_ptm
from qndt.quantum.backends.quimb_adapter import MPDOConfig, QuimbAdapter

_log = logging.getLogger(__name__)

# Warn when global ρ min-eigenvalue drops below this (§8 MPDO validity).
_POSITIVITY_WARN_THRESHOLD: float = -1e-10


@dataclass(frozen=True, slots=True)
class SimulationStep:
    """Immutable record of a single simulation operation.

    Stored in the TensorStateTracker step log. Provides a full audit trail
    of operations and the resulting state quality metric.

    Args:
        t: Simulation time of this operation [s].
        node_id: Quantum memory node identifier, or None.
        link_id: Fiber link identifier, or None.
        operation: Operation type: ``"entangle"``, ``"gate"``,
                   ``"channel"``, or ``"measure"``.
        fidelity_after: Bell fidelity (entangle) or purity (channel) after
                        the operation, or None for gate/measure steps.
        ptm_applied: Copy of the PTM applied (channel steps only), or None.
        bond_dims: Bond dimension metadata from the backend.

    Notes:
        ``ptm_applied`` is stored as a fresh copy via ``object.__setattr__`` in
        ``__post_init__`` because ``frozen=True`` prevents normal field assignment.
    """

    t: float
    node_id: str | None
    link_id: str | None
    operation: str
    fidelity_after: float | None
    ptm_applied: np.ndarray | None
    bond_dims: dict[str, int]

    def __post_init__(self) -> None:
        if self.ptm_applied is not None:
            # frozen=True blocks self.ptm_applied = ..., so bypass via object.
            object.__setattr__(
                self,
                "ptm_applied",
                np.array(self.ptm_applied, dtype=np.float64),
            )


# Module-level gate constants — allocated once, never mutated.
_H: np.ndarray = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CNOT: np.ndarray = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)


class TensorStateTracker:
    """Manages quantum state for the whole simulation network (§3.4).

    This is the ONLY class in qndt that holds quantum state — via the
    QuimbAdapter it owns.  Physics engines apply noise through
    ``apply_channel()``; gates via ``apply_gate()``; entangled pairs are
    initialised via ``entangle()``.

    All operations append to an internal step log accessible via
    ``step_log()``.  Call ``reset()`` to start a fresh simulation run.

    Args:
        n_sites: Number of qubits in the global system.
        chi_max: Maximum MPDO bond dimension (forwarded to QuimbAdapter).
        kappa_max: Maximum Kraus bond dimension (forwarded).
        cutoff: SVD truncation threshold (forwarded).
    """

    def __init__(
        self,
        n_sites: int,
        chi_max: int = 32,
        kappa_max: int = 8,
        cutoff: float = 1e-10,
    ) -> None:
        self._config = MPDOConfig(
            n_sites=n_sites,
            chi_max=chi_max,
            kappa_max=kappa_max,
            cutoff=cutoff,
        )
        self._adapter = QuimbAdapter(self._config)
        self._log: list[SimulationStep] = []

    # ------------------------------------------------------------------
    # State control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset quantum state to |0⟩^⊗n and clear the step log."""
        self._adapter.initialise_zeros()
        self._log.clear()

    # ------------------------------------------------------------------
    # Quantum operations
    # ------------------------------------------------------------------

    def entangle(self, q1: int, q2: int, bell_state: str = "phi+") -> float:
        """Create a Bell pair on qubits (q1, q2) via Hadamard + CNOT.

        Starting from |00⟩ this produces the phi+ Bell state
        (|00⟩ + |11⟩)/√2.  Both qubits must be in |0⟩ for unit fidelity.

        Args:
            q1: Control qubit (Hadamard is applied here; CNOT control).
            q2: Target qubit (CNOT target).
            bell_state: Bell state to compute fidelity against.

        Returns:
            Fidelity of the (q1, q2) reduced DM with ``bell_state``.
        """
        self._adapter.apply_single_qubit_gate(q1, _H)
        self._adapter.apply_two_qubit_gate(q1, q2, _CNOT)
        fidelity = self._adapter.fidelity_with_bell(q1, q2, bell_state)
        self._log.append(
            SimulationStep(
                t=0.0,
                node_id=None,
                link_id=None,
                operation="entangle",
                fidelity_after=fidelity,
                ptm_applied=None,
                bond_dims=self._adapter.truncate(),
            )
        )
        return fidelity

    def apply_gate(self, gate: np.ndarray, qubits: list[int]) -> None:
        """Apply a unitary gate to one or two qubits.

        Args:
            gate: Unitary matrix; shape (2, 2) for 1-qubit or (4, 4) for 2-qubit.
            qubits: List of qubit indices; length must be 1 or 2.

        Raises:
            ValueError: If ``len(qubits)`` is not 1 or 2.
        """
        if len(qubits) == 1:
            self._adapter.apply_single_qubit_gate(qubits[0], gate)
        elif len(qubits) == 2:
            self._adapter.apply_two_qubit_gate(qubits[0], qubits[1], gate)
        else:
            raise ValueError(
                f"apply_gate supports 1 or 2 qubits; got {len(qubits)}"
            )
        self._log.append(
            SimulationStep(
                t=0.0,
                node_id=None,
                link_id=None,
                operation="gate",
                fidelity_after=None,
                ptm_applied=None,
                bond_dims=self._adapter.truncate(),
            )
        )

    def apply_channel(
        self,
        qubit: int,
        ptm: np.ndarray,
        t: float = 0.0,
        link_id: str | None = None,
        node_id: str | None = None,
    ) -> float:
        """Apply a Pauli noise channel to a single qubit.

        Args:
            qubit: Target qubit index.
            ptm: Diagonal PTM ``[1, λx, λy, λz]`` (§5.1).
            t: Simulation time of this operation [s]; stored in the step log.
            link_id: Fiber link identifier for the step log.
            node_id: Quantum node identifier for the step log.

        Returns:
            Global state purity ``Tr(ρ²)`` after the channel — a proxy for
            how much coherence has been destroyed across the entire network.

        Raises:
            ValueError: If ``ptm`` fails ``validate_ptm()`` (unphysical channel).
        """
        if not validate_ptm(ptm):
            raise ValueError(
                f"PTM failed validation — must be [1,λx,λy,λz] with |λ|≤1 "
                f"and non-negative Pauli rates. Got: {ptm}"
            )
        self._adapter.apply_pauli_channel(qubit, ptm)
        purity = self._adapter.purity()
        min_eig = self._adapter.min_eigenvalue()
        if min_eig < _POSITIVITY_WARN_THRESHOLD:
            _log.warning(
                "apply_channel(qubit=%d): global ρ min eigenvalue %.3e after "
                "channel — MPDO truncation may have violated positivity (§8).",
                qubit, min_eig,
            )
        self._log.append(
            SimulationStep(
                t=t,
                node_id=node_id,
                link_id=link_id,
                operation="channel",
                fidelity_after=purity,
                ptm_applied=ptm,
                bond_dims=self._adapter.truncate(),
            )
        )
        return purity

    def measure(self, qubit: int, basis: str = "Z") -> tuple[int, float]:
        """Projective measurement of a qubit with Born-rule state collapse.

        Args:
            qubit: Qubit to measure.
            basis: ``"Z"`` (default), ``"X"``, or ``"Y"``.

        Returns:
            ``(outcome, probability)`` where ``outcome ∈ {0, 1}`` and
            ``probability`` is the Born-rule probability of that outcome.
        """
        outcome, prob = self._adapter.measure_qubit(qubit, basis)
        self._log.append(
            SimulationStep(
                t=0.0,
                node_id=None,
                link_id=None,
                operation="measure",
                fidelity_after=None,
                ptm_applied=None,
                bond_dims=self._adapter.truncate(),
            )
        )
        return outcome, prob

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def fidelity(self, q1: int, q2: int, bell_state: str = "phi+") -> float:
        """Fidelity of the (q1, q2) pair with a Bell state.

        Args:
            q1: First qubit index.
            q2: Second qubit index.
            bell_state: One of ``"phi+"``, ``"phi-"``, ``"psi+"``, ``"psi-"``.

        Returns:
            Fidelity in ``[0, 1]``.
        """
        return self._adapter.fidelity_with_bell(q1, q2, bell_state)

    def purity(self) -> float:
        """Global state purity Tr(ρ²).

        Returns:
            1.0 for a pure state; approaches 1/2^n for the maximally mixed state.
        """
        return self._adapter.purity()

    def min_eigenvalue(self) -> float:
        """Minimum eigenvalue of the global density matrix (§8 positivity check).

        Returns:
            Smallest eigenvalue; should be ≥ 0 for a physically valid state.
        """
        return self._adapter.min_eigenvalue()

    def entropy(self, qubit: int) -> float:
        """Von Neumann entropy of the single-qubit reduced state (nats).

        Args:
            qubit: Qubit index.

        Returns:
            Entropy in ``[0, log(2)]``.
        """
        return self._adapter.entropy(qubit)

    # ------------------------------------------------------------------
    # Step log access
    # ------------------------------------------------------------------

    def step_log(self) -> list[SimulationStep]:
        """Return a shallow copy of the operation step log.

        Returns:
            Ordered list of ``SimulationStep`` records since the last ``reset()``.
        """
        return list(self._log)

    def fidelity_timeseries(
        self, operation: str = "channel"
    ) -> list[tuple[float, float]]:
        """Extract (t, fidelity_after) pairs from the step log.

        Args:
            operation: Only include steps whose ``operation`` field matches this
                       string (default ``"channel"``).

        Returns:
            List of ``(t, fidelity_after)`` tuples in chronological order.
            Steps with ``fidelity_after is None`` are silently excluded.
        """
        result: list[tuple[float, float]] = []
        for step in self._log:
            if step.operation == operation and step.fidelity_after is not None:
                result.append((step.t, step.fidelity_after))
        return result
