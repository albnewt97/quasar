"""Spontaneous Raman Scattering (SpRS) noise model for WDM co-existence.

Implements §5.4: forward/backward Raman power, total photon rate, and the
resulting PTM contribution to the quantum channel.  No I/O, no Qt, no state
shared with other engines.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from qndt.core.context import OpContext
from qndt.physics.channels import depolarising_ptm

# Physical constants
_H_JS: float = 6.626e-34   # Planck constant [J·s]
_K_B: float = 1.381e-23    # Boltzmann constant [J/K]
_C_MPS: float = 3.0e8       # Speed of light   [m/s]
_T_DEFAULT_K: float = 300.0  # Default temperature [K]

# Default per-channel filter bandwidth used in the Raman integral
_DELTA_LAMBDA_NM: float = 1.0   # [nm]

# ---------------------------------------------------------------------------
# Normalized silica Raman gain spectrum shape  [LITERATURE-GROUNDED]
# ---------------------------------------------------------------------------
# g(|Δν|): tabulated from the canonical spontaneous-Raman spectrum of silica
# fiber in G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed. (Academic Press,
# 2019), Fig. 8.1.  The primary vibrational band of amorphous SiO₂ peaks at
# ≈ 13.2 THz (440 cm⁻¹); values are normalized so that g(13.2 THz) = 1.
# Values beyond 45 THz are taken as zero; intermediate values are linearly
# interpolated.  [LITERATURE-GROUNDED]
_G_FREQ_THZ: np.ndarray = np.array(
    [0.0, 3.0, 6.0, 9.0, 11.0, 12.5, 13.2, 14.0, 15.0, 18.0,
     21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0],
    dtype=np.float64,
)
_G_NORM: np.ndarray = np.array(
    [0.00, 0.02, 0.07, 0.20, 0.55, 0.92, 1.00, 0.95, 0.85, 0.55,
     0.35, 0.22, 0.13, 0.08, 0.05, 0.03, 0.015, 0.005, 0.0],
    dtype=np.float64,
)

# Calibration: β(1310 nm → 1550 nm) = 4.0 × 10⁻¹¹ 1/(km·nm)
# Eraerds et al., New J. Phys. 12, 063027 (2010), Table 1.
# [TAG: absolute scale calibrated]
_ERAERDS_LAMBDA_C_NM: float = 1310.0
_ERAERDS_LAMBDA_Q_NM: float = 1550.0
_ERAERDS_BETA_CAL: float = 4.0e-11  # 1/(km·nm)


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
    """Raman cross-section ρ(Δν) for spontaneous Raman scattering in silica fiber.

    Implements the frequency-offset profile from Whitepaper §5 eq (16):

    .. code-block:: text

        Δν       = ν_cl − ν_q  [Hz]  (> 0 Stokes, < 0 anti-Stokes)
        ρ(Δν)    = ρ_peak · g(|Δν|) · A(Δν, T)

    where:

    - ``g(|Δν|)``: normalized silica Raman gain spectrum shape (peak ≈ 13.2 THz);
      tabulated from Agrawal, *Nonlinear Fiber Optics*, 6th ed. (2019), Fig. 8.1.
      [LITERATURE-GROUNDED]
    - ``A(Δν, T)``: Bose–Einstein thermal asymmetry factor
      (Stokes) ``A = n(|Δν|,T) + 1``, (anti-Stokes) ``A = n(|Δν|,T)``
      where ``n(|Δν|,T) = 1/(exp(h|Δν|/kT) − 1)``.
      (Boyd, *Nonlinear Optics*, 4th ed. (2020), §10.2.)
    - ``ρ_peak`` [1/(km·nm)]: absolute scale; calibrated so that at the
      1310 nm → 1550 nm Stokes offset (Δν ≈ 35.4 THz, T = 300 K) the model
      reproduces β ≈ 4×10⁻¹¹ 1/(km·nm) from Eraerds et al. (2010).
      [TAG: absolute scale calibrated]

    The public interface ``beta(lambda_c_nm, lambda_q_nm) -> float`` is preserved
    for backward compatibility with ``CoexistenceNoiseEngine``.

    Args:
        rho_peak: Peak cross-section ρ_peak in [1/(km·nm)].
        temperature_k: Fiber temperature for Bose–Einstein factor; default 300 K.

    Raises:
        ValueError: If ``rho_peak`` ≤ 0 or ``temperature_k`` ≤ 0.
    """

    def __init__(self, rho_peak: float, temperature_k: float = _T_DEFAULT_K) -> None:
        if rho_peak <= 0.0:
            raise ValueError(f"rho_peak must be > 0; got {rho_peak}")
        if temperature_k <= 0.0:
            raise ValueError(f"temperature_k must be > 0; got {temperature_k}")
        self._rho_peak = rho_peak
        self._temperature_k = temperature_k

    @property
    def rho_peak(self) -> float:
        """Absolute scale ρ_peak [1/(km·nm)]. [TAG: absolute scale calibrated]"""
        return self._rho_peak

    @property
    def temperature_k(self) -> float:
        """Fiber temperature used for Bose–Einstein factor [K]."""
        return self._temperature_k

    def _gain_shape(self, abs_delta_nu_hz: float) -> float:
        """Interpolate normalized Raman gain profile g at |Δν|.

        Args:
            abs_delta_nu_hz: Frequency offset magnitude |Δν| in Hz.

        Returns:
            Normalized gain g(|Δν|) ∈ [0, 1].
        """
        return float(np.interp(abs_delta_nu_hz / 1e12, _G_FREQ_THZ, _G_NORM))

    def _bose_einstein_n(self, abs_delta_nu_hz: float) -> float:
        """Bose–Einstein mean occupation n(|Δν|, T) = 1/(exp(h|Δν|/kT) − 1).

        Args:
            abs_delta_nu_hz: Frequency offset magnitude |Δν| in Hz.

        Returns:
            Occupation number n ≥ 0.
        """
        if abs_delta_nu_hz < 1e9:
            return 1e6  # near-DC limit; n → ∞
        x = _H_JS * abs_delta_nu_hz / (_K_B * self._temperature_k)
        return 1.0 / math.expm1(x)

    def beta(self, lambda_c_nm: float, lambda_q_nm: float) -> float:
        """Return ρ(Δν) in [1/(km·nm)] for classical/quantum wavelength pair.

        Computes the SpRS cross-section at frequency offset
        ``Δν = ν_cl − ν_q = c/λ_cl − c/λ_q``:

        ``ρ(Δν) = ρ_peak · g(|Δν|) · A(Δν, T)``

        Stokes (λ_cl < λ_q, Δν > 0): A = n(|Δν|, T) + 1
        anti-Stokes (λ_cl > λ_q, Δν < 0): A = n(|Δν|, T)

        Args:
            lambda_c_nm: Classical pump wavelength in nm.
            lambda_q_nm: Quantum signal wavelength in nm.

        Returns:
            Raman cross-section ρ(Δν) in [1/(km·nm)].
        """
        nu_cl = _C_MPS / (lambda_c_nm * 1e-9)
        nu_q = _C_MPS / (lambda_q_nm * 1e-9)
        delta_nu = nu_cl - nu_q

        g = self._gain_shape(abs(delta_nu))
        n = self._bose_einstein_n(abs(delta_nu))
        a = (n + 1.0) if delta_nu > 0.0 else n

        return self._rho_peak * g * a

    @classmethod
    def smf28_default(cls) -> "RamanProfile":
        """SMF-28 Raman profile calibrated to Eraerds et al. (2010).

        Calibrates ρ_peak so that β(1310 nm, 1550 nm) = 4×10⁻¹¹ 1/(km·nm),
        matching Eraerds et al., *New J. Phys.* **12**, 063027 (2010), which
        reports SpRS noise power of order 10⁻¹⁴ W/nm for a 1 mW pump at 25 km
        over SMF-28 — implying β ≈ 4×10⁻¹¹ 1/(km·nm).

        The 1310 → 1550 nm pair gives Δν ≈ 35.4 THz (Stokes), which sits on
        the falling side of the silica Raman peak at ~13.2 THz.  Also cited:
        da Silva et al., *J. Lightwave Technol.* **32**, 2332 (2014) [ref 13].

        [TAG: absolute scale calibrated]

        Returns:
            ``RamanProfile`` calibrated to reproduce the Eraerds measurement.
        """
        # Bootstrap: unit-peak profile to read the profile value at cal. point
        unit = cls(rho_peak=1.0, temperature_k=_T_DEFAULT_K)
        beta_at_unit = unit.beta(_ERAERDS_LAMBDA_C_NM, _ERAERDS_LAMBDA_Q_NM)
        rho_peak_cal = _ERAERDS_BETA_CAL / beta_at_unit
        return cls(rho_peak=rho_peak_cal, temperature_k=_T_DEFAULT_K)


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

    The live-load path (primary): when a control plane is wired,
    ``raman_rate()`` calls ``self._control_plane.current_load(link_id, t)``
    (duck-typed; no import of ``control_plane`` package to avoid the circular
    dependency — ``control_plane.load`` imports ``ClassicalChannelSpec`` from
    this module).  The returned ``active_channels`` list is iterated in place of
    the static ``self._channels`` dict.

    Static fallback: if no control plane is wired, or if the live load has no
    active channels for the queried link, ``self._channels`` (populated via
    ``register_channel()``) is used instead.  This preserves standalone and
    test use without a control plane.

    Args:
        profile: Raman cross-section profile for the fiber type.
        fiber: Physical parameters of the fiber span.
        control_plane: ``AsynchronousControlPlane`` at runtime; typed ``Any``
            to avoid the circular import between physics and control_plane
            packages (§3.3).

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

        Implements §5.4.  Channel source (live vs. static):

        - **Live path** (CP-managed links): when a control plane is wired
          **and** ``self._control_plane.manages_link(link_id)`` is ``True``
          (the link was activated at least once in the ``WDMLoadTracker``),
          ``current_load().active_channels`` is iterated directly.  An empty
          list means all channels are off → rate is **0**, not a fallback.
        - **Static path** (unmanaged links): if no control plane is wired, or
          the link was never activated (``manages_link`` → ``False``), iterates
          ``self._channels`` (populated via ``register_channel()``).

        **Rule**: a link is owned by **either** static ``register_channel`` **or**
        the CP schedule, not both.  Once the CP manages a link, static channels
        for that link are permanently bypassed.

        Per-channel §5.4 formula unchanged:

        ``P_fwd = Pc · ρ(Δν) · Δλ · L · exp(-α·L)``
        ``P_bwd = Pc · ρ(Δν) · Δλ · (1 - exp(-2·α·L)) / (2·α)``
        ``rate_c = (P_fwd + P_bwd) · η_det · T_opt / (h · ν_q)``

        Args:
            link_id: Fiber link identifier; used to query per-link WDM load.
            lambda_q_nm: Quantum channel wavelength in nm.
            t: Current simulation time in seconds.

        Returns:
            Total Raman photon arrival rate in Hz.
        """
        alpha = self._fiber.alpha
        length = self._fiber.length_km
        nu_q = _C_MPS / (lambda_q_nm * 1e-9)
        h_nu_q = _H_JS * nu_q

        # B2 semantics: live path when CP manages link; static dict otherwise.
        channels: Iterable[ClassicalChannelSpec]
        if self._control_plane is not None and self._control_plane.manages_link(link_id):
            load = self._control_plane.current_load(link_id, t)
            channels = load.active_channels  # empty list → rate 0; no static fallback
        else:
            channels = self._channels.values()

        total_rate = 0.0
        for spec in channels:
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

        A Raman false click is a spontaneously scattered photon that carries no
        information about the transmitted qubit state — the registered detection
        event corresponds to a maximally mixed state ρ → I/2, i.e. the
        **depolarising channel**:

        .. code-block:: text

            ρ → (1−p)ρ + p·I/2    ↔    px = py = pz = p/4
            λ = [1, 1−p, 1−p, 1−p]

        **Isotropic:** An uncorrelated noise photon has no preferred Pauli axis,
        so px = py = pz = p/4 is the only physically consistent assignment.  The
        previous symmetric form (px = pz = p/2, py = 0) gave the correct X/Z QBER
        but arbitrarily excluded Y-errors; depolarising is the principled form.

        **pz-only REJECTED:** A pure-Z assignment gives λz = 1 (zero Z-basis
        error), contradicting the e₀ = ½ random-bit dark-count model in
        ``key_rate.py``.  A pz-only error requires a phase/interferometric
        receiver; the generic η_d / p_dc + e₀ = ½ model is not one.

        Note: λx and λz are identical to the old symmetric model, so the X- and
        Z-basis QBER contributions are unchanged by this switch.

        ``ctx.lambda_q`` is expected in SI metres; ``× 1e9`` converts to nm.

        Args:
            ctx: Current operation context.  ``lambda_q`` must be in metres.

        Returns:
            Length-4 diagonal PTM ``[1, 1−p, 1−p, 1−p]``.
        """
        p = self.effective_dark_prob(
            ctx.link_id, ctx.lambda_q * 1e9, ctx.gate_width, ctx.t
        )
        p = min(p, 1.0)  # effective_dark_prob already clamps; guard against fp drift
        return depolarising_ptm(p)
