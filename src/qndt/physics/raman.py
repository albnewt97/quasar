"""Spontaneous Raman Scattering (SpRS) noise model for WDM co-existence.

Implements §5.4: forward/backward Raman power, total photon rate, and the
resulting PTM contribution to the quantum channel.  No I/O, no Qt, no state
shared with other engines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from qndt.core.context import OpContext

# Physical constants
_H_JS: float = 6.626e-34   # Planck constant [J·s]
_C_MPS: float = 3.0e8       # Speed of light   [m/s]

# Default per-channel filter bandwidth used in the Raman integral
_DELTA_LAMBDA_NM: float = 1.0   # [nm]


@dataclass(frozen=True, slots=True)
class FiberParams:
    """Physical parameters of a fiber span used in the SpRS calculation.

    Args:
        length_km: Fiber span length in km.
        attenuation_db_per_km: Power loss coefficient in dB/km.
        eta_detector: Single-photon detector efficiency in (0, 1].
        t_opt: Optical transmission of filter + coupling in (0, 1].
        p_dc: Intrinsic dark-count probability per gate in (0, 1].

    Raises:
        ValueError: If any field violates its domain constraint.
    """

    length_km: float
    attenuation_db_per_km: float
    eta_detector: float
    t_opt: float
    p_dc: float

    def __post_init__(self) -> None:
        if self.length_km <= 0.0:
            raise ValueError(f"length_km must be > 0; got {self.length_km}")
        if self.attenuation_db_per_km <= 0.0:
            raise ValueError(
                f"attenuation_db_per_km must be > 0; got {self.attenuation_db_per_km}"
            )
        for name, val in (
            ("eta_detector", self.eta_detector),
            ("t_opt", self.t_opt),
            ("p_dc", self.p_dc),
        ):
            if val <= 0.0 or val > 1.0:
                raise ValueError(f"{name} must be in (0, 1]; got {val}")

    @property
    def alpha(self) -> float:
        """Linear attenuation coefficient α [1/km].

        Derived from the dB/km loss via
        ``α = attenuation_db_per_km / (10·log₁₀(e))``.
        """
        return self.attenuation_db_per_km / (10.0 * math.log10(math.e))


class RamanProfile:
    """Raman cross-section table β(λc, λq) in units of [1/(km·nm)].

    Values map ``(lambda_c_nm, lambda_q_nm)`` pairs to the spontaneous Raman
    scattering coefficient used in the §5.4 power integrals.

    Args:
        profile_table: Dict keyed by ``(lambda_c_nm, lambda_q_nm)`` pairs.
    """

    def __init__(self, profile_table: dict[tuple[float, float], float]) -> None:
        self._table: dict[tuple[float, float], float] = dict(profile_table)

    def beta(self, lambda_c_nm: float, lambda_q_nm: float) -> float:
        """Return the Raman cross-section β(λc, λq) in [1/(km·nm)].

        Checks for an exact table entry first; otherwise finds the nearest
        ``lambda_c_nm`` entry in the table and linearly interpolates in
        ``lambda_q_nm``.

        Args:
            lambda_c_nm: Classical pump wavelength in nm.
            lambda_q_nm: Quantum signal wavelength in nm.

        Returns:
            Raman cross-section in [1/(km·nm)].
        """
        key = (lambda_c_nm, lambda_q_nm)
        if key in self._table:
            return self._table[key]

        # Nearest lambda_c in table.
        c_values = sorted({k[0] for k in self._table})
        nearest_c = min(c_values, key=lambda c: abs(c - lambda_c_nm))

        # All (lambda_q, beta) pairs for that lambda_c, sorted by wavelength.
        q_beta = sorted(
            ((k[1], v) for k, v in self._table.items() if k[0] == nearest_c)
        )

        if len(q_beta) == 1:
            return q_beta[0][1]

        q_vals = [qb[0] for qb in q_beta]
        b_vals = [qb[1] for qb in q_beta]
        return float(np.interp(lambda_q_nm, q_vals, b_vals))

    @classmethod
    def smf28_default(cls) -> RamanProfile:
        """Pre-loaded SMF-28 Raman cross-section table from published measurements.

        Values are consistent with Eraerds et al., New J. Phys. 12, 063027 (2010),
        which reports SpRS noise power of order 1e-14 W/nm for 1mW pump at 25km —
        corresponding to beta ~= 4e-11 1/(km*nm), not 1e-8 (an earlier version of
        this table was off by ~3 orders of magnitude and produced GHz-scale dark
        click rates instead of the kHz-MHz range reported in the literature).

        Returns:
            ``RamanProfile`` instance pre-loaded with SMF-28 cross-sections.
        """
        table: dict[tuple[float, float], float] = {
            # 1310 nm classical pump → 1550 nm quantum channel (Stokes tail)
            (1310.0, 1490.0): 5.2e-11,
            (1310.0, 1550.0): 4.0e-11,
            (1310.0, 1610.0): 3.2e-11,
            # 1550 nm classical pump → 1310 nm quantum channel (anti-Stokes)
            (1550.0, 1310.0): 3.5e-11,
            (1550.0, 1450.0): 4.8e-11,
            (1550.0, 1650.0): 3.7e-11,
        }
        return cls(table)


@dataclass(frozen=True, slots=True)
class ClassicalChannelSpec:
    """Specification for a single WDM classical co-propagating channel.

    Args:
        channel_id: Unique identifier for this channel.
        lambda_c_nm: Classical channel centre wavelength in nm.
        launch_power_mw: Launch power into the fiber in milliwatts.

    Raises:
        ValueError: If ``lambda_c_nm`` or ``launch_power_mw`` is not positive.
    """

    channel_id: str
    lambda_c_nm: float
    launch_power_mw: float

    def __post_init__(self) -> None:
        if self.lambda_c_nm <= 0.0:
            raise ValueError(f"lambda_c_nm must be > 0; got {self.lambda_c_nm}")
        if self.launch_power_mw <= 0.0:
            raise ValueError(f"launch_power_mw must be > 0; got {self.launch_power_mw}")


class CoexistenceNoiseEngine:
    """Computes Raman dark-count noise from WDM classical channels (§5.4).

    Implements the ``NoiseContributor`` protocol: its ``ptm(ctx)`` method
    returns the diagonal Pauli Transfer Matrix contribution from spontaneous
    Raman scattering photons that register as false detector clicks.

    Args:
        profile: Raman cross-section table for the fiber type.
        fiber: Physical parameters of the fiber span.
        control_plane: ``AsynchronousControlPlane`` at runtime; typed ``Any``
            to avoid the circular import between physics and control_plane
            packages (§3.3).  Reserved for future WDM-load queries.

    Raises:
        KeyError: ``deregister_channel`` raises if the ID is not found.
    """

    def __init__(
        self,
        profile: RamanProfile,
        fiber: FiberParams,
        control_plane: Any,  # AsynchronousControlPlane; Any avoids circular import
    ) -> None:
        self._profile = profile
        self._fiber = fiber
        self._control_plane = control_plane
        self._channels: dict[str, ClassicalChannelSpec] = {}

    def register_channel(self, spec: ClassicalChannelSpec) -> None:
        """Add a classical WDM channel to the noise model.

        Args:
            spec: Channel specification to register.
        """
        self._channels[spec.channel_id] = spec

    def deregister_channel(self, channel_id: str) -> None:
        """Remove a classical channel from the noise model.

        Args:
            channel_id: ID of the channel to remove.

        Raises:
            KeyError: If ``channel_id`` is not registered.
        """
        del self._channels[channel_id]

    def raman_rate(self, link_id: str, lambda_q_nm: float, t: float) -> float:
        """Compute the total spontaneous Raman photon rate at the quantum detector.

        Implements §5.4 for all registered classical channels:

        ``P_fwd = Pc · β · Δλ · L · exp(-α·L)``
        ``P_bwd = Pc · β · Δλ · (1 - exp(-2·α·L)) / (2·α)``
        ``rate_c = (P_fwd + P_bwd) · η_det · T_opt / (h · ν_q)``

        Args:
            link_id: Fiber link identifier (reserved for per-link WDM queries).
            lambda_q_nm: Quantum channel wavelength in nm.
            t: Current simulation time in seconds.

        Returns:
            Total Raman photon arrival rate in Hz.
        """
        alpha = self._fiber.alpha
        length = self._fiber.length_km
        nu_q = _C_MPS / (lambda_q_nm * 1e-9)
        h_nu_q = _H_JS * nu_q

        total_rate = 0.0
        for spec in self._channels.values():
            pc_w = spec.launch_power_mw * 1e-3
            beta = self._profile.beta(spec.lambda_c_nm, lambda_q_nm)

            p_fwd = pc_w * beta * _DELTA_LAMBDA_NM * length * math.exp(-alpha * length)
            p_bwd = (
                pc_w
                * beta
                * _DELTA_LAMBDA_NM
                * (1.0 - math.exp(-2.0 * alpha * length))
                / (2.0 * alpha)
            )

            rate_c = (p_fwd + p_bwd) * self._fiber.eta_detector * self._fiber.t_opt / h_nu_q
            total_rate += rate_c

        return total_rate

    def effective_dark_prob(
        self,
        link_id: str,
        lambda_q_nm: float,
        gate_width: float,
        t: float,
    ) -> float:
        """Compute the total dark-count probability per gate including Raman.

        Dual-detector model (§5, §9): each detector contributes p_dc independently,
        so the dark-count floor doubles.  The Raman click probability enters once
        (it is already the combined probability over both detectors):

        ``p = 2·p_dc + (1 − exp(−r_Raman · τ_gate))``   clamped to [0, 1].

        Args:
            link_id: Fiber link identifier.
            lambda_q_nm: Quantum channel wavelength in nm.
            gate_width: Gate duration in seconds.
            t: Current simulation time in seconds.

        Returns:
            Click probability in [0.0, 1.0].
        """
        rate = self.raman_rate(link_id, lambda_q_nm, t)
        p = 2.0 * self._fiber.p_dc + (1.0 - math.exp(-rate * gate_width))
        return float(min(max(p, 0.0), 1.0))

    def ptm(self, ctx: OpContext) -> np.ndarray:
        """Return the diagonal PTM contribution from Raman dark counts.

        Models Raman false clicks as symmetric X and Z errors on the qubit
        (photon detection collapses the state):

        ``px = p/2,  py = 0,  pz = p/2``

        ``ctx.lambda_q`` is expected in SI metres; ``× 1e9`` converts to nm.

        Args:
            ctx: Current operation context.  ``lambda_q`` must be in metres.

        Returns:
            Length-4 diagonal PTM ``[1, λx, λy, λz]``.
        """
        p = self.effective_dark_prob(
            ctx.link_id, ctx.lambda_q * 1e9, ctx.gate_width, ctx.t
        )
        # Half the dark-click probability on X and Z; py = 0.
        half_p = min(p / 2.0, 0.499)
        # Derive λy from λxz to keep the three λ values internally consistent
        # and avoid floating-point cancellation in validate_ptm's py recovery.
        lxz = 1.0 - 2.0 * half_p   # λx = λz = 1 - p
        ly = 2.0 * lxz - 1.0        # λy = 1 - 2p, consistent with lxz
        return np.array([1.0, lxz, ly, lxz], dtype=np.float64)
