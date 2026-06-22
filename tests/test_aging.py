"""Physics regression tests for DeviceAgingModel (§5.5).

Verifies exponential T2 decay, idle dephasing formula, the T2 floor,
gate overrotation drift, and PTM validity.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from qndt.core.context import OpContext
from qndt.physics.aging import DeviceAgingModel
from qndt.physics.channels import validate_ptm

pytestmark = pytest.mark.physics_regression

# ---------------------------------------------------------------------------
# Shared factory
# ---------------------------------------------------------------------------


def _model(
    t2: float = 1.0,
    nc: float = 1e6,
    drift: float = 0.0,
    eps0: float = 0.0,
) -> DeviceAgingModel:
    return DeviceAgingModel(
        t2_nominal=t2,
        wear_const_nc=nc,
        calib_drift_rate=drift,
        gate_overrotation_0=eps0,
    )


# ---------------------------------------------------------------------------
# test_t2_nominal_at_zero_ops
# ---------------------------------------------------------------------------


def test_t2_nominal_at_zero_ops() -> None:
    """coherence_time returns t2_nominal when no ops have been registered."""
    model = _model(t2=1.0, nc=1e6)
    assert model.coherence_time("n", 0.0) == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# test_t2_monotonic_decrease
# ---------------------------------------------------------------------------


def test_t2_monotonic_decrease() -> None:
    """T2 strictly decreases as ops are registered (monotonic wear curve)."""
    model = _model(t2=1.0, nc=10.0)
    t2_values = []
    for i in range(5):
        model.register_op("n", "gate", t=float(i))
        t2_values.append(model.coherence_time("n", float(i)))
    assert all(t2_values[i] > t2_values[i + 1] for i in range(len(t2_values) - 1))


# ---------------------------------------------------------------------------
# test_t2_wear_curve
# ---------------------------------------------------------------------------


def test_t2_wear_curve() -> None:
    """After wear_const_nc operations, T2 must equal t2_nominal / e (within 1e-6).

    From §5.5: T2(Nc) = T2_0 · exp(-Nc / Nc) = T2_0 · exp(-1) = T2_0 / e.
    """
    nc = 1000
    model = _model(t2=2.0, nc=float(nc))
    for _ in range(nc):
        model.register_op("n", "gate", t=0.0)
    expected = 2.0 * math.exp(-1.0)
    assert model.coherence_time("n", 0.0) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# test_t2_floor
# ---------------------------------------------------------------------------


def test_t2_floor() -> None:
    """T2 never falls below 1e-9 s, even after extreme op counts.

    Uses Nc=1.0 and 100 ops so T2 = exp(-100) ≈ 3.7e-44 << 1e-9, verifying
    the floor is applied.
    """
    model = _model(t2=1.0, nc=1.0)
    for _ in range(100):
        model.register_op("n", "gate", t=0.0)
    assert model.coherence_time("n", 0.0) >= 1e-9


# ---------------------------------------------------------------------------
# test_idle_dephasing_zero_idle
# ---------------------------------------------------------------------------


def test_idle_dephasing_zero_idle() -> None:
    """idle_dephasing_pz with idle_time=0 must return exactly 0.0.

    exp(-0 / T2) = 1, so 0.5·(1-1) = 0.
    """
    model = _model(t2=1.0, nc=1e6)
    assert model.idle_dephasing_pz("n", 0.0, 0.0) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# test_idle_dephasing_long_idle
# ---------------------------------------------------------------------------


def test_idle_dephasing_long_idle() -> None:
    """idle_dephasing_pz with idle_time >> T2 must return the clamped max 0.499.

    pz → 0.5·(1 - 0) = 0.5, clamped to 0.499 per NoiseContributor constraint.
    Uses T2 = 1 µs and idle_time = 1 s (1e6 × T2).
    """
    model = _model(t2=1e-6, nc=1e9)
    result = model.idle_dephasing_pz("n", 1.0, 0.0)
    assert result == pytest.approx(0.499, rel=1e-6)


# ---------------------------------------------------------------------------
# test_ptm_none_node_id
# ---------------------------------------------------------------------------


def test_ptm_none_node_id() -> None:
    """ptm() with ctx.node_id=None must return the identity PTM np.ones(4)."""
    model = _model()
    ctx = OpContext(link_id="L", node_id=None, t=0.0, lambda_q=1550e-9, gate_width=1e-9)
    np.testing.assert_array_equal(model.ptm(ctx), np.ones(4, dtype=np.float64))


# ---------------------------------------------------------------------------
# test_ptm_is_valid
# ---------------------------------------------------------------------------


def test_ptm_is_valid() -> None:
    """ptm() must pass validate_ptm() for a range of idle times."""
    model = _model(t2=1e-3, nc=1e6)
    for idle_time in [0.0, 1e-6, 1e-3, 1.0]:
        ctx = OpContext(
            link_id="L",
            node_id="node_a",
            t=0.0,
            lambda_q=1550e-9,
            gate_width=1e-9,
            idle_time=idle_time,
        )
        ptm = model.ptm(ctx)
        assert validate_ptm(ptm), (
            f"ptm() returned invalid Pauli channel PTM at idle_time={idle_time}: {ptm}"
        )


# ---------------------------------------------------------------------------
# test_calib_drift
# ---------------------------------------------------------------------------


def test_calib_drift() -> None:
    """gate_overrotation must increase linearly at rate calib_drift_rate [rad/s].

    After registering the first op at t=0, the overrotation at t=10 must be
    gate_overrotation_0 + calib_drift_rate * 10 to within 1e-12.
    """
    drift = 0.01   # rad/s
    eps0 = 0.05    # rad initial overrotation
    model = _model(drift=drift, eps0=eps0)
    model.register_op("n", "gate", t=0.0)

    eps_at_0 = model.gate_overrotation("n", 0.0)
    eps_at_10 = model.gate_overrotation("n", 10.0)

    assert eps_at_0 == pytest.approx(eps0, rel=1e-12)
    assert eps_at_10 - eps_at_0 == pytest.approx(drift * 10.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Per-node parameter override tests
# ---------------------------------------------------------------------------


def test_node_params_default_fallback() -> None:
    """node_params() for a node with no override returns all global defaults."""
    model = _model(t2=1.5, nc=2e6, drift=0.001, eps0=0.02)
    params = model.node_params("unseen")
    assert params["t2_nominal"] == pytest.approx(1.5)
    assert params["wear_const_nc"] == pytest.approx(2e6)
    assert params["calib_drift_rate"] == pytest.approx(0.001)
    assert params["gate_overrotation_0"] == pytest.approx(0.02)


def test_set_node_params_override() -> None:
    """set_node_params overrides one node; other nodes still see global defaults."""
    model = _model(t2=1.0, nc=1e6)
    model.set_node_params("nA", t2_nominal=2.0)
    assert model.node_params("nA")["t2_nominal"] == pytest.approx(2.0)
    assert model.node_params("nA")["wear_const_nc"] == pytest.approx(1e6)  # global
    assert model.node_params("nB")["t2_nominal"] == pytest.approx(1.0)    # global


def test_per_node_coherence_differs() -> None:
    """Two nodes with different wear constants diverge in T2 after equal op counts."""
    model = _model(t2=1.0, nc=1e6)
    model.set_node_params("fast_wear", wear_const_nc=10.0)
    for _ in range(20):
        model.register_op("slow_wear", "gate", t=0.0)
        model.register_op("fast_wear", "gate", t=0.0)
    t2_slow = model.coherence_time("slow_wear", 0.0)
    t2_fast = model.coherence_time("fast_wear", 0.0)
    assert t2_fast < t2_slow, (
        f"fast_wear (Nc=10) must have smaller T2 than slow_wear (Nc=1e6): "
        f"{t2_fast:.3e} >= {t2_slow:.3e}"
    )


def test_backward_compat_no_override() -> None:
    """Without set_node_params, coherence_time is identical to the hand-computed formula.

    This proves the global path is byte-for-byte unchanged: T2_0 * exp(-N/Nc).
    """
    t2_0, nc, n_ops = 2.0, 500.0, 250
    model = _model(t2=t2_0, nc=nc)
    for _ in range(n_ops):
        model.register_op("n", "gate", t=0.0)
    expected = t2_0 * math.exp(-n_ops / nc)
    assert model.coherence_time("n", 0.0) == pytest.approx(expected, rel=1e-12)


def test_set_node_params_validation() -> None:
    """set_node_params raises ValueError for non-positive t2_nominal or wear_const_nc."""
    model = _model()
    with pytest.raises(ValueError, match="t2_nominal"):
        model.set_node_params("n", t2_nominal=0.0)
    with pytest.raises(ValueError, match="wear_const_nc"):
        model.set_node_params("n", wear_const_nc=-1.0)
    with pytest.raises(ValueError, match="calib_drift_rate"):
        model.set_node_params("n", calib_drift_rate=-0.1)


def test_partial_override() -> None:
    """Overriding only wear_const_nc leaves t2_nominal at the global default."""
    model = _model(t2=1.5, nc=1e6)
    model.set_node_params("nX", wear_const_nc=500.0)
    params = model.node_params("nX")
    assert params["t2_nominal"] == pytest.approx(1.5)    # global fallback
    assert params["wear_const_nc"] == pytest.approx(500.0)  # override


# ---------------------------------------------------------------------------
# T1 longitudinal relaxation tests (Stage 3, §6)
# ---------------------------------------------------------------------------


def test_ptm_lz_equals_exp_minus_t_over_t1() -> None:
    """ptm()[3] (λz) must equal exp(-idle_time / T1) — Pauli-twirled T1 decay (§6)."""
    t1 = 5.0
    model = DeviceAgingModel(t2_nominal=1.0, wear_const_nc=1e6, calib_drift_rate=0.0,
                             t1_nominal=t1)
    for idle in [0.0, 0.5, 1.0, 5.0]:
        ctx = OpContext(
            link_id="L", node_id="node_a", t=0.0, lambda_q=1550e-9,
            gate_width=1e-9, idle_time=idle,
        )
        lz = model.ptm(ctx)[3]
        expected = math.exp(-idle / t1)
        assert lz == pytest.approx(expected, rel=1e-12), (
            f"λz mismatch at idle_time={idle}: got {lz:.6f}, expected {expected:.6f}"
        )


def test_ptm_lz_less_than_lx_when_t1_greater_than_t2() -> None:
    """λz must equal 1 at idle_time=0 and decrease slower than λx when T1 >> T2."""
    model = DeviceAgingModel(t2_nominal=0.1, wear_const_nc=1e6, calib_drift_rate=0.0,
                             t1_nominal=10.0)
    ctx = OpContext(
        link_id="L", node_id="n", t=0.0, lambda_q=1550e-9,
        gate_width=1e-9, idle_time=1.0,
    )
    ptm = model.ptm(ctx)
    lx, lz = float(ptm[1]), float(ptm[3])
    assert lz > lx, f"T1=10s >> T2=0.1s: λz ({lz:.4f}) must decay slower than λx ({lx:.4f})"


def test_ptm_t1_per_node_override() -> None:
    """Per-node t1_nominal override changes λz independently of the global default."""
    model = _model(t2=1.0, nc=1e6)
    model.set_node_params("nFast", t1_nominal=1.0)
    model.set_node_params("nSlow", t1_nominal=100.0)

    idle = 0.5
    ctx_fast = OpContext(link_id="L", node_id="nFast", t=0.0, lambda_q=1550e-9,
                         gate_width=1e-9, idle_time=idle)
    ctx_slow = OpContext(link_id="L", node_id="nSlow", t=0.0, lambda_q=1550e-9,
                         gate_width=1e-9, idle_time=idle)
    lz_fast = model.ptm(ctx_fast)[3]
    lz_slow = model.ptm(ctx_slow)[3]
    assert lz_fast < lz_slow, (
        f"Faster T1 decay: lz_fast ({lz_fast:.4f}) must be < lz_slow ({lz_slow:.4f})"
    )
    assert lz_fast == pytest.approx(math.exp(-idle / 1.0), rel=1e-12)
    assert lz_slow == pytest.approx(math.exp(-idle / 100.0), rel=1e-12)


def test_ptm_t1_validation() -> None:
    """DeviceAgingModel must raise ValueError for t1_nominal <= 0."""
    with pytest.raises(ValueError, match="t1_nominal"):
        DeviceAgingModel(t2_nominal=1.0, wear_const_nc=1e6, calib_drift_rate=0.0,
                         t1_nominal=0.0)
    with pytest.raises(ValueError, match="t1_nominal"):
        DeviceAgingModel(t2_nominal=1.0, wear_const_nc=1e6, calib_drift_rate=0.0,
                         t1_nominal=-5.0)
    model = _model()
    with pytest.raises(ValueError, match="t1_nominal"):
        model.set_node_params("n", t1_nominal=0.0)
