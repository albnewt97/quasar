"""Physics regression tests for the Raman SpRS co-existence noise engine.

All tests are marked physics_regression.  The Eraerds limit test additionally
cites the primary reference for the SMF-28 SpRS measurements.

Parameter notes
---------------
* ``ctx.lambda_q`` in ``OpContext`` is in SI metres; ``CoexistenceNoiseEngine``
  converts to nm internally via ``lambda_q * 1e9``.
* β(1310, 1550) = 4.0e-11 [1/(km·nm)] from ``smf28_default()`` (corrected to
  match Eraerds et al. 2010; an earlier table value of 3.2e-8 was off by
  ~3 orders of magnitude and produced unphysical GHz-scale dark click rates).
* The Eraerds test uses η_det=0.5, t_opt=0.1 which gives
  raman_rate ≈ 1.47e5 Hz for 25 km / 1 mW / 1310 nm → 1550 nm.
"""
from __future__ import annotations

import math

import pytest

from qndt.core.context import OpContext
from qndt.physics.channels import validate_ptm
from qndt.physics.raman import (
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
        control_plane=None,  # not used in these tests
    )


# ---------------------------------------------------------------------------
# test_raman_rate_zero_power
# ---------------------------------------------------------------------------


def test_raman_rate_zero_power() -> None:
    """ClassicalChannelSpec must raise ValueError for launch_power_mw <= 0."""
    with pytest.raises(ValueError, match="launch_power_mw"):
        ClassicalChannelSpec(channel_id="c0", lambda_c_nm=1310.0, launch_power_mw=0.0)


# ---------------------------------------------------------------------------
# test_raman_rate_single_channel
# ---------------------------------------------------------------------------


def test_raman_rate_single_channel() -> None:
    """Single 1mW/1310nm channel in 50km SMF-28 must give rate in [1e2, 1e8] Hz."""
    engine = _make_engine()
    engine.register_channel(
        ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1.0)
    )
    rate = engine.raman_rate("link_test", 1550.0, t=0.0)
    assert 1e2 <= rate <= 1e8, f"raman_rate out of physically plausible range: {rate:.3e} Hz"


# ---------------------------------------------------------------------------
# test_raman_increases_with_power
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# test_raman_increases_with_channels
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# test_dark_prob_floor
# ---------------------------------------------------------------------------


def test_dark_prob_floor() -> None:
    """Engine with no channels registered must return exactly 2·p_dc.

    Dual-detector model (§5, §9): both detectors contribute independently,
    doubling the dark-count floor.  The Raman term (1-exp(-r·τ)) enters once.
    With no channels raman_rate=0 so 1-exp(0)=0 and result == 2·p_dc exactly.
    """
    p_dc = 1e-5
    fiber = _make_fiber(p_dc=p_dc)
    engine = _make_engine(fiber)
    result = engine.effective_dark_prob("link_test", 1550.0, gate_width=1e-9, t=0.0)
    assert result == pytest.approx(2.0 * p_dc, rel=1e-12)


# ---------------------------------------------------------------------------
# test_dark_prob_clamped
# ---------------------------------------------------------------------------


def test_dark_prob_clamped() -> None:
    """effective_dark_prob must never exceed 1.0, even at absurdly high power."""
    engine = _make_engine()
    engine.register_channel(
        ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1e6)
    )
    result = engine.effective_dark_prob("link_test", 1550.0, gate_width=1e-9, t=0.0)
    assert result <= 1.0
    # With this power, rate·gate >> 1, so p_click ≈ 1.0 (fully saturated).
    assert result == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# test_ptm_is_valid_pauli_channel
# ---------------------------------------------------------------------------


def test_ptm_is_valid_pauli_channel() -> None:
    """ptm(ctx) output must pass validate_ptm() for a realistic operating point."""
    fiber = _make_fiber()
    engine = _make_engine(fiber)
    engine.register_channel(
        ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1.0)
    )
    # ctx.lambda_q in metres → engine multiplies by 1e9 to get nm
    ctx = OpContext(
        link_id="link_test",
        node_id=None,
        t=0.0,
        lambda_q=1550e-9,   # 1550 nm expressed in SI metres
        gate_width=1e-9,
    )
    result = engine.ptm(ctx)
    assert validate_ptm(result), f"ptm() returned invalid Pauli channel PTM: {result}"


# ---------------------------------------------------------------------------
# test_smf28_default_profile
# ---------------------------------------------------------------------------


def test_smf28_default_profile() -> None:
    """SMF-28 default β(1310, 1550) must be in the physically plausible range.

    Per Eraerds et al., New J. Phys. 12, 063027 (2010): SpRS noise power of
    order 1e-14 W/nm for 1mW pump at 25km implies beta ~= 4e-11 1/(km*nm).
    """
    b = RamanProfile.smf28_default().beta(1310.0, 1550.0)
    assert 1e-12 <= b <= 1e-9, f"β(1310, 1550) = {b:.2e} outside [1e-12, 1e-9] 1/(km·nm)"


# ---------------------------------------------------------------------------
# test_eraerds_limit  (Eraerds et al., New J. Phys. 12, 063027 (2010))
# ---------------------------------------------------------------------------


def test_eraerds_limit() -> None:
    """Raman rate for 25km SMF-28, 1mW at 1310nm, 1550nm quantum, must be in [1e3, 1e6] Hz.

    Reference:
        Eraerds et al., New J. Phys. 12, 063027 (2010) — Table 1.
        Reports spontaneous Raman count rates of order 1–100 kHz for similar
        fiber configurations with SNSPD detectors and narrow-band filtering.

    Parameter derivation:
        α(1310nm) = 0.35/4.343 ≈ 0.0806 km⁻¹
        P_fwd + P_bwd ≈ 3.7e-13 W  (25 km, 1 mW, β=4.0e-11)
        With η_det=0.5, t_opt=0.1: rate ≈ 1.47e5 Hz  ∈ [1e3, 1e6].
    """
    # 1310 nm classical attenuation in SMF-28 is ~0.35 dB/km
    fiber = FiberParams(
        length_km=25.0,
        attenuation_db_per_km=0.35,
        eta_detector=0.5,    # 50% SNSPD efficiency
        t_opt=0.1,           # 10% — bandpass filter + coupling losses
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

    # Verify the computed value is in the experimentally observed range.
    assert 1e3 <= rate <= 1e6, (
        f"Eraerds limit test failed: raman_rate = {rate:.3e} Hz, "
        f"expected in [1e3, 1e6] Hz "
        f"(Eraerds et al., New J. Phys. 12, 063027 (2010))"
    )

    # Sanity check the exact computed value against hand-calculation.
    alpha = fiber.alpha
    L = fiber.length_km
    beta = RamanProfile.smf28_default().beta(1310.0, 1550.0)
    pc_w = 1.0 * 1e-3
    p_fwd = pc_w * beta * 1.0 * L * math.exp(-alpha * L)
    p_bwd = pc_w * beta * 1.0 * (1.0 - math.exp(-2.0 * alpha * L)) / (2.0 * alpha)
    nu_q = 3e8 / (1550.0 * 1e-9)
    expected = (p_fwd + p_bwd) * fiber.eta_detector * fiber.t_opt / (6.626e-34 * nu_q)
    assert rate == pytest.approx(expected, rel=1e-6)
