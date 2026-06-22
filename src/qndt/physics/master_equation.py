"""Time-Convolutionless (TCL) Pauli master equation solver and RHP witness.

Implements §5.6: online computation of canonical rates from the time derivative
of PTM eigenvalues, and the Rivas-Huelga-Plenio non-Markovianity measure.
No I/O, no Qt, no state beyond the witness accumulator.
"""
from __future__ import annotations

from dataclasses import dataclass

from qndt.core.context import PauliRateVector


@dataclass(frozen=True, slots=True)
class CanonicalRates:
    """Instantaneous TCL canonical rates for the three Pauli jump operators.

    A negative rate signals information backflow into the system (non-Markovian
    dynamics).  Values are NOT clamped to zero; negativity is the RHP witness.

    Args:
        gamma_x: Canonical rate for X jump operator [Hz].
        gamma_y: Canonical rate for Y jump operator [Hz].
        gamma_z: Canonical rate for Z jump operator [Hz].
        t: Simulation time at which this snapshot was taken [s].
    """

    gamma_x: float
    gamma_y: float
    gamma_z: float
    t: float

    @property
    def is_non_markovian(self) -> bool:
        """True if any canonical rate is negative (information backflow)."""
        return self.gamma_x < 0.0 or self.gamma_y < 0.0 or self.gamma_z < 0.0


class TCLSolver:
    """Derives instantaneous TCL canonical rates from consecutive PTM snapshots.

    Uses finite differences on the diagonal PTM eigenvalues (λx, λy, λz) and
    the exact TCL inversion to extract γk.  The TCL master equation for a Pauli
    qubit channel is ``dρ/dt = Σk γk(t)(σkρσk − ρ)``, giving:

    ``dλx/dt = −(γy + γz)·λx``
    ``dλy/dt = −(γx + γz)·λy``
    ``dλz/dt = −(γx + γy)·λz``

    Inverting this 3×3 system (add any two equations, subtract the third):

    ``γx = (λ̇x/λx − λ̇y/λy − λ̇z/λz) / 2``
    ``γy = (λ̇y/λy − λ̇x/λx − λ̇z/λz) / 2``
    ``γz = (λ̇z/λz − λ̇x/λx − λ̇y/λy) / 2``

    Limiting case: for pure Z dephasing with λx(t) = exp(−t), this recovers
    γz = 1.0 exactly (constant Markovian rate), while a non-monotonic λ gives
    negative γ (non-Markovian information backflow).

    Negative rates are physically valid and indicate non-Markovian backflow;
    they are never clamped to zero.

    Reference: Rivas, Huelga, Plenio, Rep. Prog. Phys. 77, 094001 (2014) §II.C.

    Args:
        dt: Finite-difference timestep for rate estimation [s].

    Raises:
        ValueError: If dt is not positive.
    """

    def __init__(self, dt: float = 1e-3) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0; got {dt}")
        self.dt = dt

    def canonical_rates(
        self,
        rates_t: PauliRateVector,
        rates_t_minus_dt: PauliRateVector,
        t: float,
    ) -> CanonicalRates:
        """Compute instantaneous TCL canonical rates from two consecutive snapshots.

        Args:
            rates_t: Pauli rate vector at the current time step t.
            rates_t_minus_dt: Pauli rate vector at the previous step t − dt.
            t: Current simulation time [s].

        Returns:
            ``CanonicalRates`` snapshot at time ``t``.
        """
        ptm_t = rates_t.ptm()
        ptm_prev = rates_t_minus_dt.ptm()

        lx_t, ly_t, lz_t = float(ptm_t[1]), float(ptm_t[2]), float(ptm_t[3])
        lx_p, ly_p, lz_p = float(ptm_prev[1]), float(ptm_prev[2]), float(ptm_prev[3])

        d_lx = (lx_t - lx_p) / self.dt
        d_ly = (ly_t - ly_p) / self.dt
        d_lz = (lz_t - lz_p) / self.dt

        # Logarithmic derivatives: λ̇k / λk (protected against near-zero λ).
        dlx_over_lx = d_lx / lx_t if abs(lx_t) > 1e-15 else 0.0
        dly_over_ly = d_ly / ly_t if abs(ly_t) > 1e-15 else 0.0
        dlz_over_lz = d_lz / lz_t if abs(lz_t) > 1e-15 else 0.0

        # Exact TCL inversion: γx = (λ̇x/λx − λ̇y/λy − λ̇z/λz) / 2
        gamma_x = (dlx_over_lx - dly_over_ly - dlz_over_lz) / 2.0
        gamma_y = (dly_over_ly - dlx_over_lx - dlz_over_lz) / 2.0
        gamma_z = (dlz_over_lz - dlx_over_lx - dly_over_ly) / 2.0

        return CanonicalRates(gamma_x=gamma_x, gamma_y=gamma_y, gamma_z=gamma_z, t=t)


class RHPWitness:
    """Online accumulator for the Rivas-Huelga-Plenio non-Markovianity measure.

    Computes ``N_RHP = ∫_{γk(t)<0} |γk(t)| dt`` incrementally as new canonical
    rate snapshots arrive.  A positive N_RHP value certifies non-Markovian
    dynamics on the link.

    The witness integrates over all three canonical rates independently; any
    negative contribution from any rate channel is accumulated.
    """

    def __init__(self) -> None:
        self._history: list[CanonicalRates] = []
        self._N_RHP: float = 0.0
        self._sign_change_times: list[float] = []

    def update(self, rates: CanonicalRates) -> None:
        """Ingest a new canonical rates snapshot and update the witness.

        If at least two snapshots are available, integrates any negative-γ
        contributions over the interval [history[-2].t, rates.t].  Records
        backflow onset times when any rate crosses from positive to negative.

        Args:
            rates: Newly computed canonical rates snapshot.
        """
        self._history.append(rates)
        if len(self._history) < 2:
            return

        prev = self._history[-2]
        curr = rates
        dt = curr.t - prev.t
        if dt <= 0.0:
            return

        had_sign_change = False
        pairs: list[tuple[float, float]] = [
            (prev.gamma_x, curr.gamma_x),
            (prev.gamma_y, curr.gamma_y),
            (prev.gamma_z, curr.gamma_z),
        ]
        for gamma_prev, gamma_curr in pairs:
            if gamma_curr < 0.0:
                self._N_RHP += abs(gamma_curr) * dt
            if gamma_prev > 0.0 and gamma_curr < 0.0:
                had_sign_change = True

        if had_sign_change:
            self._sign_change_times.append(curr.t)

    def current_value(self) -> float:
        """Return the accumulated N_RHP witness value.

        Returns:
            Accumulated ``N_RHP`` in units of [Hz·s] = [dimensionless].
        """
        return self._N_RHP

    def is_non_markovian(self) -> bool:
        """True if ``N_RHP > 1e-10`` (certified information backflow).

        Returns:
            ``True`` when non-Markovian dynamics have been detected.
        """
        return self._N_RHP > 1e-10

    def reset(self) -> None:
        """Clear all history and reset the accumulated witness to zero."""
        self._history.clear()
        self._N_RHP = 0.0
        self._sign_change_times.clear()

    def sign_change_times(self) -> list[float]:
        """Return timestamps where any canonical rate crossed from positive to negative.

        Returns:
            Copy of the list of backflow onset times [s].
        """
        return list(self._sign_change_times)
