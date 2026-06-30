"""Physics regression tests for the Raman SpRS co-existence noise engine.

All tests are marked physics_regression.

Model: ρ(Δν) = ρ_peak · g(|Δν|) · A(Δν, T)
  g(|Δν|): normalized silica Raman gain spectrum (Agrawal NLFO Fig. 8.1)
  A(Δν, T): Bose–Einstein factor — (n+1) Stokes, n anti-Stokes
  ρ_peak: calibrated so β(1310→1550 nm) = 4×10⁻¹¹ 1/(km·nm) [Eraerds 2010]

References
----------
Eraerds et al., New J. Phys. 12, 063027 (2010)  [ref 11]
da Silva et al., J. Lightwave Technol. 32, 2332 (2014)  [ref 13]
Agrawal, Nonlinear Fiber Optics, 6th ed. (2019)  [Raman gain spectrum shape]
Boyd, Nonlinear Optics, 4th ed. (2020)  [Bose–Einstein Stokes/anti-Stokes]

Parameter notes
---------------
* ``ctx.lambda_q`` in ``OpContext`` is in SI metres; ``CoexistenceNoiseEngine``
  converts to nm internally via ``lambda_q * 1e9``.
* The calibration point β(1310, 1550) = 4.0×10⁻¹¹ 1/(km·nm) is enforced
  exactly by construction in ``smf28_default()``.
* The old lookup-table values at non-calibration wavelength pairs are NOT
  reproduced within 30% by the new model — the old table was not physically
  consistent (Stokes and anti-Stokes at 35 THz had nearly equal values,
  contradicting the Bose–Einstein asymmetry by ~3 orders of magnitude).
  The new model is physically correct; the old table was a placeholder.
"""
from __future__ import annotations

import numpy as np
import pytest

from qndt.control_plane.load import WDMLoadTracker
from qndt.core.context import OpContext
from qndt.physics.channels import validate_ptm
from qndt.physics.raman import (
    _C_MPS,
    _ERAERDS_BETA_CAL,
    _ERAERDS_LAMBDA_C_NM,
    _ERAERDS_LAMBDA_Q_NM,
    _G_FREQ_THZ,
    _G_NORM,
    ClassicalChannelSpec,
    CoexistenceNoiseEngine,
    FiberParams,
    RamanProfile,
)

pytestmark = pytest.mark.physics_regression

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_PROFILE = RamanProfile.smf28_default()


def _make_fiber(
    length_km: float = 50.0,
    attenuation_db_per_km: float = 0.2,
    eta_detector: float = 0.1,
    t_opt: float = 0.1,
    p_dc: float = 1e-5,
) -> FiberParams:
    return FiberParams(
        length_km=length_km,
        attenuation_db_per_km=attenuation_db_per_km,
        eta_detector=eta_detector,
        t_opt=t_opt,
        p_dc=p_dc,
    )


def _make_engine(fiber: FiberParams | None = None) -> CoexistenceNoiseEngine:
    return CoexistenceNoiseEngine(
        profile=_PROFILE,
        fiber=fiber or _make_fiber(),
        control_plane=None,
    )


# ---------------------------------------------------------------------------
# Structural / interface tests (unchanged from previous version)
# ---------------------------------------------------------------------------


def test_raman_rate_zero_power() -> None:
    """ClassicalChannelSpec must raise ValueError for launch_power_mw <= 0."""
    with pytest.raises(ValueError, match="launch_power_mw"):
        ClassicalChannelSpec(channel_id="c0", lambda_c_nm=1310.0, launch_power_mw=0.0)


def test_raman_rate_single_channel() -> None:
    """Single 1mW/1310nm channel in 50km SMF-28 must give rate in [1e2, 1e8] Hz."""
    engine = _make_engine()
    engine.register_channel(
        ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1.0)
    )
    rate = engine.raman_rate("link_test", 1550.0, t=0.0)
    assert 1e2 <= rate <= 1e8, f"raman_rate out of physically plausible range: {rate:.3e} Hz"


def test_raman_increases_with_power() -> None:
    """Doubling launch_power_mw must double raman_rate exactly (linear in Pc)."""
    fiber = _make_fiber()
    engine_1x = _make_engine(fiber)
    engine_2x = _make_engine(fiber)
    engine_1x.register_channel(
        ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1.0)
    )
    engine_2x.register_channel(
        ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=2.0)
    )
    rate_1x = engine_1x.raman_rate("link_test", 1550.0, t=0.0)
    rate_2x = engine_2x.raman_rate("link_test", 1550.0, t=0.0)
    assert rate_2x == pytest.approx(2.0 * rate_1x, rel=1e-10)


def test_raman_increases_with_channels() -> None:
    """Two identical channels must give exactly 2× the rate of one channel."""
    fiber = _make_fiber()
    engine_1 = _make_engine(fiber)
    engine_2 = _make_engine(fiber)
    spec = ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1.0)
    engine_1.register_channel(spec)
    engine_2.register_channel(spec)
    engine_2.register_channel(
        ClassicalChannelSpec(channel_id="c2", lambda_c_nm=1310.0, launch_power_mw=1.0)
    )
    rate_1 = engine_1.raman_rate("link_test", 1550.0, t=0.0)
    rate_2 = engine_2.raman_rate("link_test", 1550.0, t=0.0)
    assert rate_2 == pytest.approx(2.0 * rate_1, rel=1e-10)


def test_dark_prob_floor() -> None:
    """Engine with no channels registered must return exactly 2·p_dc.

    Dual-detector model: both detectors contribute independently, doubling the
    dark-count floor.  With no channels raman_rate=0, so 1−exp(0)=0 and the
    result is exactly 2·p_dc.
    """
    p_dc = 1e-5
    fiber = _make_fiber(p_dc=p_dc)
    engine = _make_engine(fiber)
    result = engine.effective_dark_prob("link_test", 1550.0, gate_width=1e-9, t=0.0)
    assert result == pytest.approx(2.0 * p_dc, rel=1e-12)


def test_dark_prob_clamped() -> None:
    """effective_dark_prob must never exceed 1.0, even at absurdly high power."""
    engine = _make_engine()
    engine.register_channel(
        ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1e6)
    )
    result = engine.effective_dark_prob("link_test", 1550.0, gate_width=1e-9, t=0.0)
    assert result <= 1.0
    assert result == pytest.approx(1.0, abs=1e-12)


def test_ptm_is_valid_pauli_channel() -> None:
    """ptm(ctx) output must pass validate_ptm() for a realistic operating point."""
    fiber = _make_fiber()
    engine = _make_engine(fiber)
    engine.register_channel(
        ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1.0)
    )
    ctx = OpContext(
        link_id="link_test",
        node_id=None,
        t=0.0,
        lambda_q=1550e-9,
        gate_width=1e-9,
    )
    result = engine.ptm(ctx)
    assert validate_ptm(result), f"ptm() returned invalid Pauli channel PTM: {result}"


# ---------------------------------------------------------------------------
# PTM structure: depolarising channel  px = py = pz = p/4
# ---------------------------------------------------------------------------


def test_ptm_depolarizing_eigenvalues() -> None:
    """ptm() must return [1, 1-p, 1-p, 1-p] — isotropic depolarising structure.

    A Raman false click is a maximally-mixed noise photon (I/2) with no
    preferred Pauli axis: px = py = pz = p/4 → λ = [1, 1-p, 1-p, 1-p].

    Note: λx and λz are identical to the old symmetric (px=pz=p/2, py=0)
    model, so X- and Z-basis QBER contributions are UNCHANGED by this switch.
    Only λy changes from 1-2p (old) to 1-p (new/depolarising).
    """
    fiber = _make_fiber()
    engine = _make_engine(fiber)
    engine.register_channel(
        ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1.0)
    )
    ctx = OpContext(
        link_id="link_test",
        node_id=None,
        t=0.0,
        lambda_q=1550e-9,
        gate_width=1e-9,
    )
    result = engine.ptm(ctx)
    p = engine.effective_dark_prob("link_test", 1550.0, gate_width=1e-9, t=0.0)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(1.0 - p, rel=1e-10), (
        f"λx={result[1]:.8f} ≠ 1-p={1-p:.8f}"
    )
    assert result[2] == pytest.approx(1.0 - p, rel=1e-10), (
        f"λy={result[2]:.8f} ≠ 1-p={1-p:.8f}"
    )
    assert result[3] == pytest.approx(1.0 - p, rel=1e-10), (
        f"λz={result[3]:.8f} ≠ 1-p={1-p:.8f}"
    )


def test_ptm_isotropy() -> None:
    """ptm() eigenvalues must satisfy λx = λy = λz (isotropy).

    Raman noise photons are uncorrelated with the qubit state and have no
    preferred Pauli axis.  Isotropy (px = py = pz) is the only physically
    correct assignment; any axis-specific model (e.g. pz-only) would require
    a structured receiver mechanism absent from the generic dark-count model.
    """
    engine = _make_engine()
    engine.register_channel(
        ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1.0)
    )
    ctx = OpContext(
        link_id="link_test",
        node_id=None,
        t=0.0,
        lambda_q=1550e-9,
        gate_width=1e-9,
    )
    result = engine.ptm(ctx)
    assert result[1] == pytest.approx(result[2], rel=1e-10), (
        f"λx={result[1]:.8f} ≠ λy={result[2]:.8f}: PTM is not isotropic"
    )
    assert result[2] == pytest.approx(result[3], rel=1e-10), (
        f"λy={result[2]:.8f} ≠ λz={result[3]:.8f}: PTM is not isotropic"
    )


def test_ptm_identity_limit() -> None:
    """As p → 0, ptm() must approach the identity [1, 1, 1, 1].

    With no classical channels and negligible p_dc, effective_dark_prob ≈ 0
    and depolarising_ptm(0) equals the identity channel.
    """
    fiber = _make_fiber(p_dc=1e-15)
    engine = _make_engine(fiber)
    ctx = OpContext(
        link_id="link_test",
        node_id=None,
        t=0.0,
        lambda_q=1550e-9,
        gate_width=1e-9,
    )
    result = engine.ptm(ctx)
    assert result == pytest.approx(np.array([1.0, 1.0, 1.0, 1.0]), abs=1e-12)


# ---------------------------------------------------------------------------
# RamanProfile validation
# ---------------------------------------------------------------------------


def test_raman_profile_invalid_rho_peak() -> None:
    """RamanProfile must raise ValueError for rho_peak <= 0."""
    with pytest.raises(ValueError, match="rho_peak"):
        RamanProfile(rho_peak=0.0)


def test_raman_profile_invalid_temperature() -> None:
    """RamanProfile must raise ValueError for temperature_k <= 0."""
    with pytest.raises(ValueError, match="temperature_k"):
        RamanProfile(rho_peak=1.0e-9, temperature_k=-10.0)


# ---------------------------------------------------------------------------
# Physics regression (a): profile peaks near 13.2 THz  [LITERATURE-GROUNDED]
# ---------------------------------------------------------------------------


def test_profile_peaks_near_13_2_thz() -> None:
    """Stokes cross-section ρ(Δν) must peak near 13.2 THz.

    The primary vibrational band of amorphous SiO₂ is at ~440 cm⁻¹ ≈ 13.2 THz
    (Agrawal, Nonlinear Fiber Optics, 6th ed., Fig. 8.1).
    """
    profile = RamanProfile.smf28_default()
    lambda_q_nm = 1550.0
    nu_q = _C_MPS / (lambda_q_nm * 1e-9)

    # Sweep Δν from 3 to 40 THz (Stokes), 200 points
    delta_nu_thz = np.linspace(3.0, 40.0, 200)
    nu_cl_arr = nu_q + delta_nu_thz * 1e12
    lambda_c_arr = _C_MPS / nu_cl_arr * 1e9  # nm

    betas = np.array([profile.beta(float(lc), lambda_q_nm) for lc in lambda_c_arr])
    peak_idx = int(np.argmax(betas))
    peak_delta_nu = float(delta_nu_thz[peak_idx])

    assert 11.0 <= peak_delta_nu <= 15.5, (
        f"Profile peak at {peak_delta_nu:.2f} THz; expected near 13.2 THz "
        f"(silica Raman resonance, Agrawal NLFO Fig. 8.1)"
    )


# ---------------------------------------------------------------------------
# Physics regression (b): Stokes > anti-Stokes at equal |Δν|
# ---------------------------------------------------------------------------


def test_stokes_exceeds_anti_stokes() -> None:
    """Stokes cross-section must exceed anti-Stokes at the same |Δν|.

    At T = 300 K and |Δν| = 35.4 THz, the Bose–Einstein factor gives
    n ≈ 3.5×10⁻³, so Stokes ∝ (n+1) ≈ 1.003 and anti-Stokes ∝ n ≈ 0.0035 —
    a ratio of ~287.  Physical origin: Boyd, Nonlinear Optics §10.2.
    """
    profile = RamanProfile.smf28_default()
    # 1310→1550: Δν ≈ +35.4 THz (Stokes)
    beta_stokes = profile.beta(1310.0, 1550.0)
    # 1550→1310: Δν ≈ −35.4 THz (anti-Stokes); same |Δν|, same g value
    beta_anti = profile.beta(1550.0, 1310.0)

    assert beta_stokes > beta_anti, (
        f"Stokes ({beta_stokes:.3e}) must exceed anti-Stokes ({beta_anti:.3e}) "
        f"at equal |Δν| ≈ 35.4 THz"
    )
    # At this large offset the ratio should be >> 10 (n ≈ 3.5e-3, so (n+1)/n ≈ 287)
    assert beta_stokes / beta_anti > 50, (
        f"Stokes/anti-Stokes ratio = {beta_stokes/beta_anti:.1f}; expected >> 50 "
        f"at |Δν| ≈ 35.4 THz, T = 300 K"
    )


def test_stokes_anti_stokes_near_peak() -> None:
    """Stokes/anti-Stokes asymmetry must be smaller near the Raman peak than far from it.

    At |Δν| ≈ 13 THz, n is larger (kT / h|Δν| ≈ 0.47), so (n+1)/n is smaller
    compared to |Δν| ≈ 35 THz where kT << h|Δν|.
    """
    profile = RamanProfile.smf28_default()
    lambda_q = 1550.0
    nu_q = _C_MPS / (lambda_q * 1e-9)

    # Near peak: |Δν| ≈ 13.2 THz
    nu_near_peak = nu_q + 13.2e12
    lc_near_peak = _C_MPS / nu_near_peak * 1e9
    beta_s_near = profile.beta(lc_near_peak, lambda_q)
    lc_anti_near = _C_MPS / (nu_q - 13.2e12) * 1e9
    beta_as_near = profile.beta(lc_anti_near, lambda_q)
    ratio_near = beta_s_near / beta_as_near

    # Far from peak: |Δν| ≈ 35.4 THz (calibration point)
    ratio_far = profile.beta(1310.0, 1550.0) / profile.beta(1550.0, 1310.0)

    assert ratio_near < ratio_far, (
        f"Stokes/anti-Stokes ratio near peak ({ratio_near:.1f}) should be "
        f"less than far from peak ({ratio_far:.1f}): n is larger at smaller |Δν|"
    )


# ---------------------------------------------------------------------------
# Physics regression (c): monotonic falloff beyond the Raman peak
# ---------------------------------------------------------------------------


def test_monotonic_falloff_beyond_peak() -> None:
    """Stokes cross-section must fall monotonically for |Δν| > 14 THz.

    The normalized silica Raman gain g(|Δν|) is monotonically decreasing
    beyond the peak; since A(Δν > 0) = n+1 ≈ 1 for large Δν, ρ(Δν) also
    falls monotonically.
    """
    profile = RamanProfile.smf28_default()
    lambda_q_nm = 1550.0
    nu_q = _C_MPS / (lambda_q_nm * 1e-9)

    # 20 points from 14 to 44 THz (all on falling side of peak)
    delta_nu_thz = np.linspace(14.0, 44.0, 20)
    nu_cl_arr = nu_q + delta_nu_thz * 1e12
    lambda_c_arr = _C_MPS / nu_cl_arr * 1e9
    betas = [profile.beta(float(lc), lambda_q_nm) for lc in lambda_c_arr]

    for i in range(len(betas) - 1):
        assert betas[i] >= betas[i + 1], (
            f"Cross-section not monotonically falling: "
            f"β(Δν={delta_nu_thz[i]:.1f} THz)={betas[i]:.3e} < "
            f"β(Δν={delta_nu_thz[i+1]:.1f} THz)={betas[i+1]:.3e}"
        )


# ---------------------------------------------------------------------------
# Physics regression (d): magnitude calibrated to Eraerds et al. (2010)
# ---------------------------------------------------------------------------


def test_eraerds_calibration_magnitude() -> None:
    """β(1310 nm, 1550 nm) must equal 4×10⁻¹¹ 1/(km·nm) by calibration construction.

    Reference: Eraerds et al., New J. Phys. 12, 063027 (2010).
    SpRS noise power ~10⁻¹⁴ W/nm for 1 mW pump at 25 km → β ≈ 4×10⁻¹¹.
    Also: da Silva et al., J. Lightwave Technol. 32, 2332 (2014) [ref 13].
    """
    profile = RamanProfile.smf28_default()
    beta_cal = profile.beta(_ERAERDS_LAMBDA_C_NM, _ERAERDS_LAMBDA_Q_NM)
    assert beta_cal == pytest.approx(_ERAERDS_BETA_CAL, rel=1e-4), (
        f"Calibration point β(1310,1550) = {beta_cal:.4e}; "
        f"expected {_ERAERDS_BETA_CAL:.4e} 1/(km·nm)"
    )


def test_eraerds_limit() -> None:
    """Raman rate for 25 km SMF-28, 1 mW at 1310 nm, 1550 nm quantum, must be in [1e3, 1e6] Hz.

    Reference:
        Eraerds et al., New J. Phys. 12, 063027 (2010) — Table 1.
        Reports spontaneous Raman count rates of order 1–100 kHz for similar
        fiber configurations with SNSPD detectors and narrow-band filtering.
        β(1310→1550) = 4.0×10⁻¹¹ 1/(km·nm) is the Eraerds-anchored value;
        ρ_peak is calibrated to reproduce this exactly.

    Parameter derivation:
        α(1310 nm) = 0.35/4.343 ≈ 0.0806 km⁻¹
        P_fwd + P_bwd ≈ 3.7×10⁻¹³ W  (25 km, 1 mW, β = 4.0×10⁻¹¹)
        With η_det=0.5, t_opt=0.1: rate ≈ 1.47×10⁵ Hz  ∈ [1×10³, 1×10⁶].
    """
    fiber = FiberParams(
        length_km=25.0,
        attenuation_db_per_km=0.35,
        eta_detector=0.5,
        t_opt=0.1,
        p_dc=1e-6,
    )
    engine = CoexistenceNoiseEngine(
        profile=RamanProfile.smf28_default(),
        fiber=fiber,
        control_plane=None,
    )
    engine.register_channel(
        ClassicalChannelSpec(channel_id="c1310", lambda_c_nm=1310.0, launch_power_mw=1.0)
    )
    rate = engine.raman_rate("link_eraerds", 1550.0, t=0.0)

    assert 1e3 <= rate <= 1e6, (
        f"Eraerds limit test: raman_rate = {rate:.3e} Hz, "
        f"expected [1e3, 1e6] Hz (Eraerds et al., New J. Phys. 12, 063027 (2010))"
    )

    # Verify against hand calculation using calibrated β directly
    import math as _math
    alpha = fiber.alpha
    L = fiber.length_km
    beta = RamanProfile.smf28_default().beta(1310.0, 1550.0)
    pc_w = 1.0e-3
    p_fwd = pc_w * beta * 1.0 * L * _math.exp(-alpha * L)
    p_bwd = pc_w * beta * 1.0 * (1.0 - _math.exp(-2.0 * alpha * L)) / (2.0 * alpha)
    nu_q = 3e8 / (1550.0e-9)
    expected = (p_fwd + p_bwd) * fiber.eta_detector * fiber.t_opt / (6.626e-34 * nu_q)
    assert rate == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Physics regression (e): continuity with old Eraerds anchor
# ---------------------------------------------------------------------------


def test_continuity_eraerds_anchor() -> None:
    """New model exactly reproduces the Eraerds-anchored calibration point.

    The 1310 nm → 1550 nm Stokes pair is the single anchor from Eraerds (2010)
    and is reproduced by construction (ρ_peak is calibrated to this point).
    Other old lookup-table entries are NOT reproduced within 30% because the
    old table was not physically consistent: it assigned nearly equal values to
    Stokes and anti-Stokes at the same |Δν| = 35 THz, contradicting the
    Bose–Einstein asymmetry by ~3 orders of magnitude at T = 300 K.
    """
    profile = RamanProfile.smf28_default()
    # Calibration anchor — must match exactly
    assert profile.beta(1310.0, 1550.0) == pytest.approx(4.0e-11, rel=1e-3)

    # Other old points: new model is physically motivated; values differ
    # substantially from the unphysical old lookup table.  We only require that
    # results are in [1e-13, 1e-8] 1/(km·nm) — a physically plausible range.
    old_pairs = [(1310, 1490), (1310, 1610), (1550, 1310), (1550, 1450), (1550, 1650)]
    for lc, lq in old_pairs:
        b = profile.beta(float(lc), float(lq))
        assert 1e-13 <= b <= 1e-8, (
            f"β({lc},{lq}) = {b:.2e} outside plausible range [1e-13, 1e-8] 1/(km·nm)"
        )


# ---------------------------------------------------------------------------
# Physics regression: ρ_peak property and temperature param
# ---------------------------------------------------------------------------


def test_smf28_rho_peak_reasonable() -> None:
    """ρ_peak for SMF-28 default must be in a physically plausible range.

    Given β_cal = 4×10⁻¹¹ and g(35.4 THz) ≈ 0.034, ρ_peak ≈ 1.2×10⁻⁹.
    Bounds [1e-11, 1e-7] are generous but exclude obviously wrong calibrations.
    """
    profile = RamanProfile.smf28_default()
    assert 1e-11 <= profile.rho_peak <= 1e-7, (
        f"ρ_peak = {profile.rho_peak:.3e}; expected in [1e-11, 1e-7] 1/(km·nm)"
    )


def test_temperature_affects_anti_stokes() -> None:
    """Increasing temperature must increase the anti-Stokes cross-section.

    n(|Δν|, T) = 1/(exp(h|Δν|/kT) − 1) is monotonically increasing in T,
    so ρ_AS = ρ_peak · g · n also increases with T.
    """
    profile_cold = RamanProfile(rho_peak=1.0e-9, temperature_k=100.0)
    profile_warm = RamanProfile(rho_peak=1.0e-9, temperature_k=500.0)
    # anti-Stokes: 1550 nm classical → 1310 nm quantum
    b_cold = profile_cold.beta(1550.0, 1310.0)
    b_warm = profile_warm.beta(1550.0, 1310.0)
    assert b_warm > b_cold, (
        f"Anti-Stokes cross-section must increase with T: "
        f"cold={b_cold:.3e}, warm={b_warm:.3e}"
    )


def test_temperature_minimally_affects_stokes_at_large_offset() -> None:
    """Stokes cross-section at large |Δν| is nearly temperature-independent.

    For |Δν| >> kT/h (here 35 THz >> 6.2 THz at 300 K), n ≈ 0, so
    A_Stokes = n+1 ≈ 1 — very weakly dependent on T.
    """
    profile_cold = RamanProfile(rho_peak=1.0e-9, temperature_k=100.0)
    profile_warm = RamanProfile(rho_peak=1.0e-9, temperature_k=500.0)
    b_cold = profile_cold.beta(1310.0, 1550.0)
    b_warm = profile_warm.beta(1310.0, 1550.0)
    # Stokes ratio should be much closer to 1 than anti-Stokes ratio
    stokes_ratio = b_warm / b_cold
    assert 0.9 <= stokes_ratio <= 1.5, (
        f"Stokes cross-section should be nearly T-independent at 35 THz; "
        f"warm/cold = {stokes_ratio:.3f}"
    )


# ---------------------------------------------------------------------------
# Live control-plane integration: dynamic WDM load  (§3.3 F2)
# ---------------------------------------------------------------------------
# These tests verify raman_rate() sources channels from the control plane
# when wired, with fallback to static self._channels when the live load is empty.
#
# A minimal duck-typed control plane: a WDMLoadTracker wrapped to expose
# the same current_load() signature as AsynchronousControlPlane — no extra
# imports needed because raman.py calls by duck typing.

_LINK = "link_live"
_LAMBDA_Q = 1550.0


class _MockControlPlane:
    """Minimal duck-typed stand-in for AsynchronousControlPlane in live-CP tests."""

    def __init__(self, tracker: WDMLoadTracker) -> None:
        self._tracker = tracker

    def current_load(self, link_id: str, t: float) -> object:
        return self._tracker.current_load(link_id, t)

    def manages_link(self, link_id: str) -> bool:
        return self._tracker.manages_link(link_id)


def _make_live_engine(
    tracker: WDMLoadTracker,
    fiber: FiberParams | None = None,
) -> CoexistenceNoiseEngine:
    """Engine wired to a live WDMLoadTracker via the duck-typed mock control plane."""
    return CoexistenceNoiseEngine(
        profile=_PROFILE,
        fiber=fiber or _make_fiber(),
        control_plane=_MockControlPlane(tracker),
    )


_SPEC_1310 = ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1.0)


def test_live_cp_second_identical_channel_doubles_rate() -> None:
    """Activating a 2nd identical channel must exactly double raman_rate (§3.3 F2-a).

    The §5.4 formula is linear in Pc; two equal channels give 2× the single-channel
    rate.  This is the analogue of the static test_raman_increases_with_channels
    but sourced from the live control plane.
    """
    tracker = WDMLoadTracker()
    engine = _make_live_engine(tracker)

    tracker.activate(_LINK, _SPEC_1310)
    rate_1ch = engine.raman_rate(_LINK, _LAMBDA_Q, t=0.0)

    tracker.activate(
        _LINK, ClassicalChannelSpec(channel_id="c2", lambda_c_nm=1310.0, launch_power_mw=1.0)
    )
    rate_2ch = engine.raman_rate(_LINK, _LAMBDA_Q, t=0.0)

    assert rate_1ch > 0.0, "Single-channel Raman rate must be positive"
    assert rate_2ch == pytest.approx(2.0 * rate_1ch, rel=1e-10), (
        f"Two identical live channels must give exactly 2× the rate: "
        f"1ch={rate_1ch:.4e}, 2ch={rate_2ch:.4e}"
    )


def test_live_cp_power_increase_raises_rate() -> None:
    """Doubling a channel's live power via update_power must double raman_rate (§3.3 F2-b).

    update_power feeds into current_load().active_channels[*].launch_power_mw,
    which raman_rate() now reads from the live path.  Rate is linear in Pc.
    """
    tracker = WDMLoadTracker()
    engine = _make_live_engine(tracker)

    tracker.activate(_LINK, _SPEC_1310)
    rate_1mw = engine.raman_rate(_LINK, _LAMBDA_Q, t=0.0)

    tracker.update_power(_LINK, "c1", power_mw=2.0)
    rate_2mw = engine.raman_rate(_LINK, _LAMBDA_Q, t=0.0)

    assert rate_2mw == pytest.approx(2.0 * rate_1mw, rel=1e-10), (
        f"Doubling launch power must double Raman rate: 1mW={rate_1mw:.4e}, 2mW={rate_2mw:.4e}"
    )


def test_live_cp_deactivate_drops_rate_to_fallback() -> None:
    """Deactivating the live channel on a managed link drops rate to 0 (§3.3 B2).

    After deactivate the link remains CP-managed (manages_link → True), so
    raman_rate() uses the live path with an empty channel list → rate = 0.
    No static fallback is consulted.  (With no static channels registered,
    rate = 0 regardless; the behaviour change matters when static channels exist —
    see test_b2_managed_all_off_gives_zero_not_static.)
    """
    tracker = WDMLoadTracker()
    engine = _make_live_engine(tracker)
    # No static channels — engine._channels is empty

    tracker.activate(_LINK, _SPEC_1310)
    rate_active = engine.raman_rate(_LINK, _LAMBDA_Q, t=0.0)
    assert rate_active > 0.0, "Rate must be positive while live channel is active"

    tracker.deactivate(_LINK, "c1")
    rate_deactivated = engine.raman_rate(_LINK, _LAMBDA_Q, t=0.0)
    assert rate_deactivated == pytest.approx(0.0, abs=1e-30), (
        f"Rate must be 0 when live channel deactivated and no static fallback: {rate_deactivated}"
    )


def test_live_cp_empty_live_load_falls_back_to_static() -> None:
    """Empty live load (unknown link) must fall back to static self._channels (§3.3 F2-d).

    WDMLoadTracker.current_load() returns ClassicalLoad with empty active_channels
    for an unknown link.  raman_rate() then falls back to self._channels, which
    contains the statically-registered spec → rate matches the static-only value.
    """
    tracker = WDMLoadTracker()
    # No channels activated for _LINK in the tracker (unknown link → empty live load)

    engine = _make_live_engine(tracker)
    # Register the channel statically as fallback
    engine.register_channel(_SPEC_1310)

    rate_live_fallback = engine.raman_rate(_LINK, _LAMBDA_Q, t=0.0)

    # Verify it equals the rate a fully-static engine produces
    static_engine = _make_engine()
    static_engine.register_channel(_SPEC_1310)
    rate_static = static_engine.raman_rate(_LINK, _LAMBDA_Q, t=0.0)

    assert rate_live_fallback == pytest.approx(rate_static, rel=1e-10), (
        f"Live-fallback rate {rate_live_fallback:.4e} must equal static rate {rate_static:.4e}"
    )


def test_b2_managed_all_off_gives_zero_not_static() -> None:
    """CP-managed link with all channels off must give rate = 0, not static fallback (§3.3 B2).

    After activate+deactivate the link key persists in WDMLoadTracker._active
    (manages_link → True).  raman_rate() uses the live path → empty channels
    → rate = 0, even though a static channel is registered in self._channels.
    This test would FAIL with the old ``or`` fallback (B2 regression test).
    """
    tracker = WDMLoadTracker()
    engine = _make_live_engine(tracker)
    # Register a static channel — old code would fall back to this, new code must not.
    engine.register_channel(_SPEC_1310)

    tracker.activate(_LINK, _SPEC_1310)
    tracker.deactivate(_LINK, "c1")
    # Link is still managed (key in _active), but has zero active channels.
    assert tracker.manages_link(_LINK), "Link must remain managed after deactivate"

    rate = engine.raman_rate(_LINK, _LAMBDA_Q, t=0.0)
    assert rate == pytest.approx(0.0, abs=1e-30), (
        f"CP-managed link with all channels off must give rate = 0 (B2 semantics), "
        f"not static fallback; got {rate:.4e}"
    )


def test_b2_unmanaged_link_uses_static() -> None:
    """Unmanaged link (never activated in CP) must fall back to static channels (§3.3 B2).

    manages_link → False (no activate call) → raman_rate() uses self._channels,
    giving a nonzero rate from the static registration.
    """
    tracker = WDMLoadTracker()
    engine = _make_live_engine(tracker)
    # Static channel only — no CP activation for _LINK.
    engine.register_channel(_SPEC_1310)
    assert not tracker.manages_link(_LINK), "Link must be unmanaged with no activate calls"

    rate = engine.raman_rate(_LINK, _LAMBDA_Q, t=0.0)
    assert rate > 0.0, (
        f"Unmanaged link must use static self._channels, giving rate > 0; got {rate:.4e}"
    )


# ---------------------------------------------------------------------------
# Gain-shape sanity checks
# ---------------------------------------------------------------------------


def test_gain_shape_normalized() -> None:
    """Normalized gain g(|Δν|) must equal 1.0 at the peak frequency (13.2 THz)."""
    profile = RamanProfile(rho_peak=1.0)
    g_at_peak = profile._gain_shape(13.2e12)
    assert g_at_peak == pytest.approx(1.0, rel=1e-6)


def test_gain_shape_monotone_profile() -> None:
    """g values from the profile table must equal the tabulated reference values."""
    profile = RamanProfile(rho_peak=1.0)
    for freq_thz, g_expected in zip(_G_FREQ_THZ, _G_NORM):
        g_computed = profile._gain_shape(freq_thz * 1e12)
        assert g_computed == pytest.approx(float(g_expected), rel=1e-6), (
            f"g({freq_thz} THz) = {g_computed:.4f}; expected {g_expected:.4f}"
        )
