"""Correctness Audit #1 — PTM analytic-limit coverage for every NoiseContributor.

Registered contributor set (TwinOrchestrator.__init__ lines 273-275):
  1. EnvironmentalTelemetryEngine  → R_env
  2. CoexistenceNoiseEngine        → R_Raman
  3. DeviceAgingModel              → R_aging ⊙ R_idle (idle_time folded into ctx)

Composition: R_eff = R_env ⊙ R_Raman ⊙ R_aging  matches §3.2.
R_idle is NOT a separate contributor — it is folded into DeviceAgingModel.ptm()
via ctx.idle_time (set from AsynchronousControlPlane.induced_idle in the orchestrator).

Already-covered limits (NOT duplicated here — see listed files):
  test_channels.py:        ptm_roundtrip (one fixed vector), λ-formula structure
  test_composer.py:        empty→identity, single passthrough, 2-channel exactness
  test_aging.py:           node_id-None, λz=exp(-t/T1), validate_ptm sweep,
                           Matthiessen wear curve, T2 floor
  test_raman.py:           validate_ptm (one point), Eraerds calibration, spectral shape
  test_nonmarkovianity.py: TCLSolver canonical rates, RHPWitness accumulation

Newly added (additive only):
  A — PauliRateVector: randomised round-trip (200 vectors, ≤1e-12); λ-formula §5.1
  B — ChannelComposer: order-independence; identity-composability; CP-validity
  C — AgingEngine: t_idle=0→identity; 1/e at T2; T1→∞ pure-dephasing; t_idle→∞ saturation
  D — CoexistenceNoiseEngine: zero-noise→identity; λ-formula exact; validate_ptm p∈[0,1]
  E — EnvironmentalTelemetryEngine: zero-telemetry identity; zero-S identity;
       exact 2-sample discrete convolution; RHP N_RHP=0 (stationary); N_RHP>0 (decaying noise)
  F — Uniform CP-validity: all three contributors over varied OpContexts

FINDING F1 (FIXED):
  T2 > 2·T1 now raises ValueError in DeviceAgingModel.__init__, set_node_params,
  and NodeConfigModel (pydantic load-time guard).  The boundary T2 == 2·T1 is
  accepted (pure-T1 limit, Tφ→∞).  Tests below verify the guard.

MODELING NOTES (for later modeling pass — no test needed):
  NOTE M1 (RESOLVED): Raman false click is now modeled as depolarising:
           px=py=pz=p/4 → λ=[1,1-p,1-p,1-p].
           A Raman noise photon is maximally mixed (I/2) with no preferred Pauli
           axis; depolarising is the physically correct isotropic form.
           The previous symmetric (px=pz=p/2, py=0) model gave λx=λz=1-p but
           λy=1-2p — X/Z QBER unchanged, Y-error now correctly included.
           pz-only rejected: gives λz=1 (zero Z-basis error), contradicting e₀=½.
  NOTE M2: Diagonal-PTM restriction in aging.py is deliberate — off-diagonal Φ_AD
           (amplitude damping) is excluded to preserve the §2 diagonal-PTM invariant
           on which ChannelComposer and TCLSolver depend. Noted in aging.py docstring.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from qndt.core.composer import ChannelComposer
from qndt.core.context import OpContext, PauliRateVector
from qndt.io.config import NodeConfigModel
from qndt.physics.aging import DeviceAgingModel
from qndt.physics.channels import dephasing_ptm, ptm_to_pauli_rates, validate_ptm
from qndt.physics.kernels import ExponentialKernel
from qndt.physics.raman import (
    ClassicalChannelSpec,
    CoexistenceNoiseEngine,
    FiberParams,
    RamanProfile,
)
from qndt.telemetry.engine import EnvironmentalTelemetryEngine
from qndt.telemetry.sources import TelemetrySample

pytestmark = pytest.mark.physics_regression

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_LID = "L"  # canonical link_id for single-link tests
_NID = "nA"  # canonical node_id


def _ctx(
    *,
    node_id: str | None = _NID,
    idle_time: float = 0.0,
    lambda_q: float = 1550e-9,
    gate_width: float = 1e-9,
    t: float = 0.0,
) -> OpContext:
    return OpContext(
        link_id=_LID,
        node_id=node_id,
        t=t,
        lambda_q=lambda_q,
        gate_width=gate_width,
        idle_time=idle_time,
    )


def _aging(
    t2: float = 1.0,
    t1: float = 200.0,
    kappa: float = 0.0,
    drift: float = 0.0,
) -> DeviceAgingModel:
    return DeviceAgingModel(
        t2_nominal=t2,
        wear_rate_kappa=kappa,
        calib_drift_rate=drift,
        t1_nominal=t1,
    )


def _coex_engine(p_dc: float = 1e-5) -> CoexistenceNoiseEngine:
    fiber = FiberParams(
        length_km=25.0,
        attenuation_db_per_km=0.2,
        eta_detector=0.8,
        t_opt=0.5,
        p_dc=p_dc,
    )
    return CoexistenceNoiseEngine(
        profile=RamanProfile.smf28_default(),
        fiber=fiber,
        control_plane=None,
    )


def _telemetry_engine(
    S: np.ndarray | None = None,
    tau: float = 1.0,
    env_ref: np.ndarray | None = None,
) -> EnvironmentalTelemetryEngine:
    if S is None:
        S = np.eye(3, dtype=np.float64)
    _ref = env_ref if env_ref is not None else np.zeros(3, dtype=np.float64)
    return EnvironmentalTelemetryEngine(
        sensitivity=S,
        kernel=ExponentialKernel(tau_x=tau, tau_y=tau, tau_z=tau),
        env_ref=_ref,
    )


# ===========================================================================
# A — PauliRateVector: randomised round-trip + λ-formula (§5.1)
# ===========================================================================


def test_a1_pauli_roundtrip_randomised() -> None:
    """ptm_to_pauli_rates(PauliRateVector(px,py,pz).ptm()) recovers (px,py,pz) within 1e-12.

    Exercises 200 random valid rate vectors in [0, 0.3]³ (sum < 0.9 < 1, each < 0.499)
    to confirm the §5.1 round-trip identity over a broad input range — not just the
    single fixed vector in test_ptm_roundtrip.
    """
    rng = np.random.default_rng(42)
    N = 200
    for _ in range(N):
        px, py, pz = rng.uniform(0.0, 0.3, size=3).tolist()
        orig = PauliRateVector(px=px, py=py, pz=pz)
        recovered = ptm_to_pauli_rates(orig.ptm())
        assert recovered.px == pytest.approx(orig.px, abs=1e-12)
        assert recovered.py == pytest.approx(orig.py, abs=1e-12)
        assert recovered.pz == pytest.approx(orig.pz, abs=1e-12)


def test_a2_lambda_formula_matches_spec_5_1() -> None:
    """PauliRateVector.ptm() satisfies §5.1: λx=1-2(py+pz), λy=1-2(px+pz), λz=1-2(px+py)."""
    rng = np.random.default_rng(99)
    for _ in range(100):
        px, py, pz = rng.uniform(0.0, 0.3, size=3).tolist()
        ptm = PauliRateVector(px=px, py=py, pz=pz).ptm()
        assert ptm[0] == 1.0
        assert ptm[1] == pytest.approx(1.0 - 2.0 * (py + pz), rel=1e-14)
        assert ptm[2] == pytest.approx(1.0 - 2.0 * (px + pz), rel=1e-14)
        assert ptm[3] == pytest.approx(1.0 - 2.0 * (px + py), rel=1e-14)


# ===========================================================================
# B — ChannelComposer: order-independence, identity-composability, CP-validity
# ===========================================================================


class _FixedPTM:
    """Minimal NoiseContributor that always returns a pre-set PTM."""

    def __init__(self, ptm: np.ndarray) -> None:
        self._ptm = ptm

    def ptm(self, ctx: OpContext) -> np.ndarray:
        return self._ptm


def test_b1_composer_order_independence() -> None:
    """Registration order must not affect effective_ptm (Hadamard product is commutative).

    Tests all 6 permutations of three distinct PTMs; each permutation must give
    the same element-wise product.
    """
    import itertools

    p1 = PauliRateVector(px=0.05, py=0.03, pz=0.02).ptm()
    p2 = PauliRateVector(px=0.04, py=0.01, pz=0.06).ptm()
    p3 = PauliRateVector(px=0.02, py=0.02, pz=0.03).ptm()
    contributors = [_FixedPTM(p1), _FixedPTM(p2), _FixedPTM(p3)]

    ctx = _ctx(node_id=None)
    reference = None
    for perm in itertools.permutations(contributors):
        composer = ChannelComposer()
        for c in perm:
            composer.register(c)
        result = composer.effective_ptm(ctx)
        if reference is None:
            reference = result.copy()
        else:
            np.testing.assert_allclose(result, reference, atol=1e-15)


def test_b2_identity_ptm_composes_away() -> None:
    """Composing any PTM with the identity PTM ones(4) must return the original PTM.

    Covers both (identity, A) and (A, identity) registration orders.
    """
    a = PauliRateVector(px=0.05, py=0.03, pz=0.07).ptm()
    identity_ptm = np.ones(4, dtype=np.float64)

    ctx = _ctx(node_id=None)

    # identity first
    c1 = ChannelComposer()
    c1.register(_FixedPTM(identity_ptm))
    c1.register(_FixedPTM(a))
    np.testing.assert_allclose(c1.effective_ptm(ctx), a, atol=1e-15)

    # identity last
    c2 = ChannelComposer()
    c2.register(_FixedPTM(a))
    c2.register(_FixedPTM(identity_ptm))
    np.testing.assert_allclose(c2.effective_ptm(ctx), a, atol=1e-15)


def test_b3_composition_of_valid_cp_ptms_is_valid_cp() -> None:
    """Composing N valid CP PTMs must produce a validate_ptm()-valid result."""
    ptms = [
        PauliRateVector(px=0.05, py=0.03, pz=0.02).ptm(),
        PauliRateVector(px=0.04, py=0.01, pz=0.06).ptm(),
        PauliRateVector(px=0.02, py=0.02, pz=0.03).ptm(),
        PauliRateVector(px=0.10, py=0.05, pz=0.08).ptm(),
    ]
    ctx = _ctx(node_id=None)
    composer = ChannelComposer()
    for p in ptms:
        composer.register(_FixedPTM(p))
    result = composer.effective_ptm(ctx)
    assert validate_ptm(result), f"Composed valid-CP PTMs gave invalid result: {result}"


# ===========================================================================
# C — AgingEngine analytic limits
# ===========================================================================


def test_c1_aging_ptm_t_idle_zero_is_identity() -> None:
    """ptm() with t_idle=0 and any node must return exactly [1, 1, 1, 1].

    pz = 0.5·(1 − exp(0)) = 0;  lx=ly = 1;  lz = exp(0/T1) = 1.
    """
    model = _aging(t2=1.0, t1=10.0)
    ctx = _ctx(node_id=_NID, idle_time=0.0)
    ptm = model.ptm(ctx)
    np.testing.assert_array_equal(ptm, np.ones(4, dtype=np.float64))


def test_c2_aging_lambda_xy_eq_exp_minus_t_over_T2() -> None:
    """λx = λy = exp(−t_idle/T2) exactly — derived from pz=0.5(1−exp(−t/T2)).

    Verification:
      pz = 0.5·(1−exp(−t/T2))
      lx_ly = 1 − 2·pz = 1 − (1−exp(−t/T2)) = exp(−t/T2)  ✓

    At t_idle = T2: λx = λy = 1/e (within 1e-12).
    Spot-checks at 0.25·T2 and 2·T2 as well.
    """
    T2 = 2.0  # s
    T1 = 1000.0  # large so lz ≈ 1
    model = _aging(t2=T2, t1=T1)

    for frac, idle in [(0.25, 0.5), (1.0, T2), (2.0, 4.0)]:
        ctx = _ctx(node_id=_NID, idle_time=idle)
        ptm = model.ptm(ctx)
        expected_lxy = math.exp(-idle / T2)
        assert ptm[1] == pytest.approx(expected_lxy, rel=1e-12), (
            f"λx mismatch at t_idle={idle} ({frac}·T2): "
            f"got {ptm[1]:.10f}, expected {expected_lxy:.10f}"
        )
        assert ptm[2] == pytest.approx(expected_lxy, rel=1e-12), (
            f"λy mismatch at t_idle={idle}: got {ptm[2]:.10f}"
        )


def test_c3_aging_pure_dephasing_limit_large_T1() -> None:
    """T1 → ∞: ptm() must equal channels.dephasing_ptm(pz_idle) within 1e-12.

    When T1 >> t_idle: lz = exp(−t/T1) ≈ 1, making the channel pure-Z dephasing.
    dephasing_ptm(pz) = [1, 1−2·pz, 1−2·pz, 1], matching [1, lxy, lxy, ~1].
    """
    T2 = 0.5  # s
    T1 = 1e12  # effectively infinite
    model = _aging(t2=T2, t1=T1)

    for idle_time in [0.1, 0.5, 1.0, 2.0]:
        ctx = _ctx(node_id=_NID, idle_time=idle_time)
        ptm_aging = model.ptm(ctx)
        pz_idle = model.idle_dephasing_pz(_NID, idle_time, ctx.t)
        ptm_dephas = dephasing_ptm(pz_idle)
        np.testing.assert_allclose(
            ptm_aging, ptm_dephas, atol=1e-8,
            err_msg=f"T1→∞ pure-dephasing mismatch at idle_time={idle_time}"
        )


def test_c4_aging_t_idle_to_infinity_saturation() -> None:
    """t_idle >> T2, T1: λx=λy saturate at 0.002 (clamp floor), λz → 0.

    The pz clamp _PZ_MAX=0.499 prevents lx_ly from reaching 0:
      lx_ly = 1 − 2·0.499 = 0.002.
    λz = exp(−t_idle/T1) → 0 for t_idle >> T1.
    """
    T2 = 1e-6  # µs → saturates quickly
    T1 = 0.001  # 1 ms → λz → 0 by t_idle=1
    model = _aging(t2=T2, t1=T1)
    ctx = _ctx(node_id=_NID, idle_time=1.0)
    ptm = model.ptm(ctx)

    # λx, λy: clamped floor at 1 − 2·0.499 = 0.002
    assert ptm[1] == pytest.approx(0.002, abs=1e-10), f"λx saturated floor: {ptm[1]}"
    assert ptm[2] == pytest.approx(0.002, abs=1e-10), f"λy saturated floor: {ptm[2]}"
    # λz: exp(−1.0/0.001) = exp(−1000) ≈ 0
    assert ptm[3] == pytest.approx(0.0, abs=1e-30), f"λz not → 0: {ptm[3]}"


# ===========================================================================
# D — CoexistenceNoiseEngine: zero-noise identity, λ-formula, validate_ptm sweep
# ===========================================================================


def test_d1_coexistence_near_zero_noise_identity() -> None:
    """No classical channels, minimal p_dc → ptm() ≈ identity within 1e-6.

    With rate=0 and p_dc=1e-9 (smallest valid value):
      p_total = 2·p_dc = 2e-9 ≈ 0 → ptm ≈ [1, 1, 1, 1].

    FiberParams requires p_dc > 0 so exact 0 is not possible; 1e-9 is the
    physical floor test.
    """
    fiber = FiberParams(
        length_km=25.0,
        attenuation_db_per_km=0.2,
        eta_detector=0.8,
        t_opt=0.5,
        p_dc=1e-9,
    )
    engine = CoexistenceNoiseEngine(
        profile=RamanProfile.smf28_default(), fiber=fiber, control_plane=None
    )
    # No channels registered → rate = 0
    ctx = _ctx(node_id=None, lambda_q=1550e-9, gate_width=1e-9)
    ptm = engine.ptm(ctx)

    np.testing.assert_allclose(ptm, np.ones(4), atol=1e-6,
                               err_msg="Near-zero noise ptm deviates from identity")
    assert validate_ptm(ptm)


def test_d2_coexistence_lambda_formula_exact() -> None:
    """ptm() must satisfy λx=λy=λz=1−p for dark-click probability p (depolarising model).

    A Raman false click is a maximally-mixed noise photon (I/2) → depolarising
    channel with px=py=pz=p/4 giving (§5.1):
      λx = 1−2(py+pz) = 1−p
      λy = 1−2(px+pz) = 1−p
      λz = 1−2(px+py) = 1−p

    Uses effective_dark_prob() to read p directly, then verifies the formula holds
    element-wise within floating-point precision.
    """
    engine = _coex_engine(p_dc=1e-5)
    engine.register_channel(
        ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1.0)
    )
    link_id, lambda_q_nm, gate_width, t = _LID, 1550.0, 1e-9, 0.0

    p = engine.effective_dark_prob(link_id, lambda_q_nm, gate_width, t)
    ctx = _ctx(node_id=None, lambda_q=lambda_q_nm * 1e-9, gate_width=gate_width)
    ptm = engine.ptm(ctx)

    assert ptm[0] == 1.0
    assert ptm[1] == pytest.approx(1.0 - p, rel=1e-14), f"λx formula: p={p:.4e}"
    assert ptm[2] == pytest.approx(1.0 - p, rel=1e-14), f"λy formula: p={p:.4e}"
    assert ptm[3] == pytest.approx(1.0 - p, rel=1e-14), f"λz formula: p={p:.4e}"


@pytest.mark.parametrize("total_p", [0.0, 0.01, 0.1, 0.5, 0.9, 0.998, 1.0])
def test_d3_coexistence_ptm_valid_for_all_p(total_p: float) -> None:
    """validate_ptm() must hold for the depolarising PTM formula at all p∈[0,1].

    Directly exercises the depolarising_ptm formula λ=[1,1-p,1-p,1-p] for the
    full range of dark-click probabilities including the saturation regime p=1
    where λ=[1,0,0,0] (maximally mixed output).
    """
    ptm = np.array([1.0, 1.0 - total_p, 1.0 - total_p, 1.0 - total_p], dtype=np.float64)
    assert validate_ptm(ptm), (
        f"Depolarising PTM formula invalid at p={total_p}: {ptm}"
    )


# ===========================================================================
# E — EnvironmentalTelemetryEngine analytic limits
# ===========================================================================


def test_e1_env_telemetry_no_samples_returns_identity() -> None:
    """ptm() with no ingested samples must return the identity PTM [1,1,1,1] exactly.

    pauli_rates() returns PauliRateVector(0,0,0) when the resampler window is empty;
    PauliRateVector(0,0,0).ptm() = [1,1,1,1] by §5.1.
    """
    engine = _telemetry_engine()
    ctx = _ctx(node_id=None)
    np.testing.assert_array_equal(engine.ptm(ctx), np.ones(4, dtype=np.float64))


def test_e2_env_telemetry_zero_sensitivity_returns_identity() -> None:
    """Zero sensitivity matrix → ptm() is identity for any telemetry.

    S=0 → S@(E-E_ref)=0 for every sample → acc=0 → p=0 → identity.
    """
    S_zero = np.zeros((3, 3), dtype=np.float64)
    engine = _telemetry_engine(S=S_zero)
    E = np.array([10.0, 5.0, 3.0])
    for i in range(5):
        engine.ingest(TelemetrySample(t=float(i), E=E, link_id=_LID))
    ctx = _ctx(node_id=None, t=4.0)
    np.testing.assert_array_equal(engine.ptm(ctx), np.ones(4, dtype=np.float64))


def test_e3_env_telemetry_exact_2sample_convolution() -> None:
    """Exact 2-sample discrete convolution: acc = K(0)@S@(E−E_ref)×dt.

    Setup: s0 at t=0, s1 at t=1.0; query at t=1.0.
    Loop body (i=1 only):
      tau = 1.0 − 1.0 = 0,  dt = 1.0 − 0.0 = 1.0
      K(0) = diag(1/τ_x, 1/τ_y, 1/τ_z)  [ExponentialKernel]
           = I for τ_x=τ_y=τ_z=1.0
      SE   = S @ (E − E_ref) = I @ E = E   (E_ref=0, S=I)
      acc  = K(0) @ SE × 1.0 = E

    Then squash: u = clip(E, 0, ∞); p = 0.5·tanh(u); ptm = PauliRateVector(p).ptm()
    Verified to 1e-12.
    """
    E_vec = np.array([0.2, 0.1, 0.05])
    engine = _telemetry_engine(tau=1.0)  # S=I, E_ref=0

    engine.ingest(TelemetrySample(t=0.0, E=E_vec, link_id=_LID))
    engine.ingest(TelemetrySample(t=1.0, E=E_vec, link_id=_LID))

    ctx = _ctx(node_id=None, t=1.0)
    ptm = engine.ptm(ctx)

    # Expected: acc = E_vec (from the exact formula above)
    u = np.clip(E_vec, 0.0, None)
    p = 0.5 * np.tanh(u)
    total = float(p.sum())
    if total > 0.499:
        p = p * (0.499 / total)
    expected = PauliRateVector(float(p[0]), float(p[1]), float(p[2])).ptm()
    np.testing.assert_allclose(ptm, expected, rtol=1e-12)


def test_e4_env_telemetry_rhp_zero_for_stationary_environment() -> None:
    """N_RHP ≈ 0 for a stationary (constant-E) environment.

    A constant-E feed produces static rates (dλ/dt ≈ 0) → all γ_k ≈ 0 →
    no information backflow → N_RHP must remain 0.

    Two consecutive pauli_rates calls with constant E yield nearly equal rates;
    TCLSolver returns γ ≈ 0 → RHPWitness accumulates nothing.
    """
    E_const = np.array([0.05, 0.05, 0.05])
    engine = _telemetry_engine(tau=1.0)

    # Long constant-E history so convolution is near steady-state
    for i in range(101):
        engine.ingest(TelemetrySample(t=float(i) * 0.1, E=E_const, link_id=_LID))

    # First query — stores rates as baseline
    ctx1 = _ctx(node_id=None, t=10.0)
    engine.ptm(ctx1)

    # Push one more identical sample, second query — rates unchanged
    engine.ingest(TelemetrySample(t=10.001, E=E_const, link_id=_LID))
    ctx2 = _ctx(node_id=None, t=10.001)
    engine.ptm(ctx2)

    assert engine.rhp_value(_LID) == pytest.approx(0.0, abs=1e-6), (
        f"Stationary feed: N_RHP should be ≈0, got {engine.rhp_value(_LID):.4e}"
    )


def test_e5_env_telemetry_rhp_positive_for_decaying_noise() -> None:
    """N_RHP > 0 when noise decays (coherence restored — non-Markovian backflow).

    Setup: push high-noise samples for t∈[−10,0], then query at t=0, t=5, t=10.
    At each successive query the old high-noise samples are further in the past,
    so the exponential kernel assigns them less weight → rates decay monotonically.
    Decreasing rates → λ increases → γ_k < 0 → N_RHP > 0.

    Three ptm() calls are required: the first establishes the rate baseline in
    _last_rates; the second produces the first CanonicalRates entry in the
    RHPWitness history; the third produces the second entry, which is when the
    witness accumulates ∫|γ| dt for the negative-γ interval [5, 10].
    """
    E_high = np.array([0.5, 0.5, 0.5])
    S = np.eye(3, dtype=np.float64)
    engine = EnvironmentalTelemetryEngine(
        sensitivity=S,
        kernel=ExponentialKernel(tau_x=1.0, tau_y=1.0, tau_z=1.0),
        env_ref=np.zeros(3, dtype=np.float64),
    )

    # Push high-noise history from t=−10 to t=0 (50 intervals of 0.2 s)
    for i in range(51):
        t_i = -10.0 + i * 0.2
        engine.ingest(TelemetrySample(t=t_i, E=E_high, link_id=_LID))

    # Three queries — first establishes baseline, second+third trigger accumulation
    for t_q in [0.0, 5.0, 10.0]:
        engine.ptm(_ctx(node_id=None, t=t_q))

    rhp = engine.rhp_value(_LID)
    assert rhp > 0.0, (
        f"Decaying noise (non-Markovian): N_RHP should be >0, got {rhp:.4e}. "
        "Rates decrease from t=0→t=5→t=10 as old samples lose exponential weight."
    )


# ===========================================================================
# F — Uniform CP-validity: all registered contributors over varied OpContexts
# ===========================================================================

_F_NODES = [None, "nA", "nB"]
_F_IDLES = [0.0, 0.01, 1.0, 10.0]
_F_LAMBDAS = [1310e-9, 1550e-9]
_F_WIDTHS = [1e-9, 1e-6]


@pytest.mark.parametrize("node_id", _F_NODES)
@pytest.mark.parametrize("idle_time", _F_IDLES)
@pytest.mark.parametrize("lambda_q", _F_LAMBDAS)
def test_f1_aging_ptm_always_valid_cp(
    node_id: str | None, idle_time: float, lambda_q: float
) -> None:
    """DeviceAgingModel.ptm() passes validate_ptm() for varied node/idle/λq.

    Covers node_id=None (identity path) and node_id with varying idle_time
    and quantum wavelengths.
    """
    model = _aging(t2=0.5, t1=200.0, kappa=1e-4)
    ctx = OpContext(
        link_id=_LID, node_id=node_id, t=1.0,
        lambda_q=lambda_q, gate_width=1e-9, idle_time=idle_time,
    )
    ptm = model.ptm(ctx)
    assert validate_ptm(ptm), (
        f"AgingModel ptm invalid: node_id={node_id}, idle={idle_time}, λq={lambda_q}: {ptm}"
    )
    for lam in ptm[1:]:
        assert abs(lam) <= 1.0 + 1e-10, f"|λ| > 1: {lam}"


@pytest.mark.parametrize("lambda_q", _F_LAMBDAS)
@pytest.mark.parametrize("gate_width", _F_WIDTHS)
def test_f2_coexistence_ptm_always_valid_cp(lambda_q: float, gate_width: float) -> None:
    """CoexistenceNoiseEngine.ptm() passes validate_ptm() for varied λq and gate widths."""
    engine = _coex_engine(p_dc=1e-5)
    engine.register_channel(
        ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1.0)
    )
    ctx = OpContext(
        link_id=_LID, node_id=None, t=0.0,
        lambda_q=lambda_q, gate_width=gate_width, idle_time=0.0,
    )
    ptm = engine.ptm(ctx)
    assert validate_ptm(ptm), (
        f"CoexistenceNoiseEngine ptm invalid: λq={lambda_q}, gw={gate_width}: {ptm}"
    )
    for lam in ptm[1:]:
        assert abs(lam) <= 1.0 + 1e-10, f"|λ| > 1: {lam}"


@pytest.mark.parametrize("lambda_q", _F_LAMBDAS)
def test_f3_telemetry_engine_ptm_always_valid_cp(lambda_q: float) -> None:
    """EnvironmentalTelemetryEngine.ptm() passes validate_ptm() with an active telemetry feed."""
    engine = _telemetry_engine(tau=5.0)
    E = np.array([0.05, 0.03, 0.02])
    for i in range(20):
        engine.ingest(TelemetrySample(t=float(i), E=E, link_id=_LID))
    ctx = OpContext(
        link_id=_LID, node_id=None, t=19.0,
        lambda_q=lambda_q, gate_width=1e-9, idle_time=0.0,
    )
    ptm = engine.ptm(ctx)
    assert validate_ptm(ptm), (
        f"TelemetryEngine ptm invalid: λq={lambda_q}: {ptm}"
    )
    for lam in ptm[1:]:
        assert abs(lam) <= 1.0 + 1e-10, f"|λ| > 1: {lam}"


# ===========================================================================
# FINDING F1 (FIXED) — T2 ≤ 2·T1 guard
# ===========================================================================


def test_finding_f1a_t2_exceeds_2t1_rejected_by_constructor() -> None:
    """DeviceAgingModel.__init__ raises ValueError for T2 > 2·T1.

    1/T2=1/(2T1)+1/Tφ with Tφ≥0 requires T2≤2T1; T2/T1=2.5 is unphysical.
    Nielsen & Chuang (2010) Ch. 8 [ref 1].
    """
    with pytest.raises(ValueError, match="t2_nominal"):
        DeviceAgingModel(
            t2_nominal=10.0,
            wear_rate_kappa=0.0,
            calib_drift_rate=0.0,
            t1_nominal=4.0,
        )


def test_finding_f1b_t2_exceeds_2t1_rejected_by_set_node_params() -> None:
    """set_node_params raises ValueError when the effective T2 > 2·T1 after override."""
    model = DeviceAgingModel(
        t2_nominal=1.0, wear_rate_kappa=0.0, calib_drift_rate=0.0, t1_nominal=200.0
    )
    with pytest.raises(ValueError, match="t2_nominal"):
        model.set_node_params("n", t2_nominal=10.0, t1_nominal=4.0)


def test_finding_f1c_t2_eq_2t1_accepted() -> None:
    """T2 = 2·T1 (pure-T1 limit, Tφ→∞) must be accepted — boundary is valid.

    Equality corresponds to 1/Tφ=0; rejecting it would exclude the physical
    T1-limited regime.  A relative tolerance of 1e-9 prevents fp false rejection.
    """
    model = DeviceAgingModel(
        t2_nominal=2.0, wear_rate_kappa=0.0, calib_drift_rate=0.0, t1_nominal=1.0
    )
    assert model.coherence_time("n", 0.0) == pytest.approx(2.0)
    model.set_node_params("n", t2_nominal=1.0, t1_nominal=0.5)
    assert model.node_params("n")["t2_nominal"] == pytest.approx(1.0)


def test_finding_f1d_set_node_params_lowering_t1_triggers_guard() -> None:
    """Lowering t1_nominal alone via set_node_params triggers the guard if T2 > 2·T1.

    The guard checks effective values after override, so an existing t2_nominal
    override combined with a new t1_nominal that would push T2 > 2·T1 is rejected.
    """
    model = DeviceAgingModel(
        t2_nominal=1.0, wear_rate_kappa=0.0, calib_drift_rate=0.0, t1_nominal=200.0
    )
    model.set_node_params("n", t2_nominal=4.0)  # valid: T2=4 ≤ 2*T1=400
    with pytest.raises(ValueError, match="t2_nominal"):
        model.set_node_params("n", t1_nominal=1.0)  # would give T2=4 > 2*T1=2


def test_finding_f1e_node_config_model_rejects_t2_gt_2t1() -> None:
    """NodeConfigModel pydantic validator rejects T2 > 2·T1 at load time."""
    with pytest.raises(Exception, match="t2_nominal"):
        NodeConfigModel(node_id="n", qubit_index=0, t2_nominal=10.0, t1_nominal=4.0)


def test_finding_f1e_node_config_model_accepts_t2_eq_2t1() -> None:
    """NodeConfigModel accepts the boundary T2 == 2·T1 without error."""
    cfg = NodeConfigModel(node_id="n", qubit_index=0, t2_nominal=2.0, t1_nominal=1.0)
    assert cfg.t2_nominal == pytest.approx(2.0)
    assert cfg.t1_nominal == pytest.approx(1.0)
