"""OpContext and PauliRateVector — immutable value objects for the simulation engine."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OpContext:
    """Immutable context passed to every NoiseContributor.ptm() call.

    Args:
        link_id: Identifier of the fiber link being evaluated.
        node_id: Identifier of the quantum memory node, or None for link-only ops.
        t: Current simulation time in seconds.
        lambda_q: Quantum channel wavelength in **SI metres** (not nanometres).
            Engines that need nm (e.g. ``CoexistenceNoiseEngine``) convert via
            ``lambda_q * 1e9`` internally.  This is the single unit-conversion
            point mandated by §3.6 of docs/architecture.md.
        gate_width: Duration of the quantum gate in seconds.
        idle_time: Time the qubit spent idle before this operation, in seconds.
    """

    link_id: str
    node_id: str | None
    t: float
    lambda_q: float
    gate_width: float
    idle_time: float = 0.0


@dataclass(frozen=True, slots=True)
class PauliRateVector:
    """Pauli error probabilities (px, py, pz) for a single-qubit channel.

    Represents the error model
    ``E(ρ) = pI·ρ + px·X·ρ·X + py·Y·ρ·Y + pz·Z·ρ·Z``
    where ``pI = 1 - px - py - pz``.

    Args:
        px: Probability of an X (bit-flip) error.
        py: Probability of a Y (bit-and-phase-flip) error.
        pz: Probability of a Z (phase-flip) error.

    Raises:
        ValueError: If any rate is negative, or if px + py + pz > 1.0.
    """

    px: float
    py: float
    pz: float

    def __post_init__(self) -> None:
        if self.px < 0.0 or self.py < 0.0 or self.pz < 0.0:
            raise ValueError(
                f"Pauli rates must be non-negative; "
                f"got px={self.px}, py={self.py}, pz={self.pz}"
            )
        total = self.px + self.py + self.pz
        if total > 1.0:
            raise ValueError(
                f"Sum of Pauli rates must be <= 1.0; got {total:.6g}"
            )

    def ptm(self) -> np.ndarray:
        """Return the diagonal Pauli Transfer Matrix as a length-4 array.

        Computes ``R = [1, λx, λy, λz]`` where:

        ``λx = 1 - 2(py + pz)``
        ``λy = 1 - 2(px + pz)``
        ``λz = 1 - 2(px + py)``

        Returns:
            Length-4 numpy array ``[1.0, λx, λy, λz]`` with dtype float64.
        """
        lx = 1.0 - 2.0 * (self.py + self.pz)
        ly = 1.0 - 2.0 * (self.px + self.pz)
        lz = 1.0 - 2.0 * (self.px + self.py)
        return np.array([1.0, lx, ly, lz], dtype=np.float64)
