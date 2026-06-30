"""Integration tests for TwinOrchestrator and the scenario config round-trip.

Verifies the full §4.1 data-flow pipeline (telemetry -> resampler -> engine ->
composer -> tracker -> results) end to end, plus the classical-quantum
coupling points (induced idle, Raman/WDM) and the JSON scenario format.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from qndt.control_plane.async_plane import AsynchronousControlPlane
from qndt.control_plane.load import WDMLoadTracker
from qndt.control_plane.routing import NetworkGraph
from qndt.core.orchestrator import (
    LinkConfig,
    NodeConfig,
    SimulationConfig,
    TwinOrchestrator,
    _scale_ptm_eigenvalues,
)
from qndt.io.config import (
    FiberParamsModel,
    KernelModel,
    LinkConfigModel,
    NodeConfigModel,
    ScenarioConfig,
    WDMScheduleEventModel,
    validate_sensitivity_matrix,
)
from qndt.physics.aging import DeviceAgingModel
from qndt.physics.kernels import ExponentialKernel, LorentzianKernel
from qndt.physics.raman import (
    ClassicalChannelSpec,
    CoexistenceNoiseEngine,
    FiberParams,
    RamanProfile,
)
from qndt.quantum.tracker import TensorStateTracker
from qndt.telemetry.engine import EnvironmentalTelemetryEngine

_LINK = LinkConfig(
    link_id="link_0",
    source_node="node_a",
    dest_node="node_b",
    lambda_q_nm=1550.0,
    gate_width_s=1e-9,
    qubit_index=0,
)
_NODE_A = NodeConfig(node_id="node_a", qubit_index=0)
_NODE_B = NodeConfig(node_id="node_b", qubit_index=1)


def test_single_step_returns_results() -> None:
    """A single step() returns exactly one result per configured link."""
    orch = TwinOrchestrator.build_simple(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=1.0,
        dt_s=0.1,
    )
    results = orch.step()
    assert len(results) == 1
    assert results[0].link_id == "link_0"
    assert results[0].t == 0.0


def test_run_advances_clock() -> None:
    """run(steps=5) with dt_s=0.1 advances the clock to 0.5s."""
    orch = TwinOrchestrator.build_simple(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=1.0,
        dt_s=0.1,
    )
    orch.run(steps=5)
    assert abs(orch.current_t() - 0.5) < 1e-9


def test_fidelity_degrades_with_noise() -> None:
    """Strong sensitivity + fast kernel decay drives fidelity down over time."""
    orch = TwinOrchestrator.build_simple(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=2.0,
        dt_s=0.1,
        sensitivity=np.eye(3, 3) * 0.5,
        kernel=ExponentialKernel(tau_x=1.0, tau_y=1.0, tau_z=1.0),
    )
    orch.run(steps=20)
    link_results = orch.results_for_link("link_0")
    assert link_results[-1].fidelity <= link_results[0].fidelity


def test_qber_in_valid_range() -> None:
    """qber is always clamped to [0.0, 0.5] regardless of noise level."""
    orch = TwinOrchestrator.build_simple(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=1.0,
        dt_s=0.1,
    )
    results = orch.run(steps=10)
    assert all(0.0 <= r.qber <= 0.5 for r in results)


def test_raman_rate_reported() -> None:
    """An active classical channel produces a positive Raman photon rate."""
    graph = NetworkGraph()
    graph.add_node("node_a")
    graph.add_node("node_b")
    graph.add_link("link_0", "node_a", "node_b")
    control_plane = AsynchronousControlPlane(graph=graph, load_tracker=WDMLoadTracker())

    fiber = FiberParams(
        length_km=25.0,
        attenuation_db_per_km=0.2,
        eta_detector=0.8,
        t_opt=0.5,
        p_dc=1e-5,
    )
    coexistence_engine = CoexistenceNoiseEngine(
        profile=RamanProfile.smf28_default(), fiber=fiber, control_plane=control_plane
    )
    coexistence_engine.register_channel(
        ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=5.0)
    )

    telemetry_engine = EnvironmentalTelemetryEngine(
        sensitivity=np.eye(3, 3) * 0.001,
        kernel=ExponentialKernel(tau_x=30.0, tau_y=30.0, tau_z=120.0),
    )
    aging_model = DeviceAgingModel(t2_nominal=1.0, wear_rate_kappa=0.0, calib_drift_rate=1e-6)
    tracker = TensorStateTracker(n_sites=1)

    config = SimulationConfig(links=(_LINK,), nodes=(_NODE_A,), duration_s=1.0, dt_s=0.1)
    orch = TwinOrchestrator(
        config=config,
        telemetry_engine=telemetry_engine,
        coexistence_engine=coexistence_engine,
        aging_model=aging_model,
        control_plane=control_plane,
        tracker=tracker,
    )

    results = orch.step()
    assert results[0].raman_rate_hz > 0.0


def test_induced_idle_affects_results() -> None:
    """A routed packet through a node produces a positive induced idle time."""
    orch = TwinOrchestrator.build_simple(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=1.0,
        dt_s=0.1,
    )
    orch._control_plane.route_packet("pkt_0", "node_a", "node_b", 0.0)
    results = orch.step()
    assert results[0].induced_idle_s > 0.0


def test_results_for_link_filter() -> None:
    """results_for_link() returns only results matching the requested link."""
    link_1 = LinkConfig(
        link_id="link_1",
        source_node="node_b",
        dest_node="node_c",
        lambda_q_nm=1550.0,
        gate_width_s=1e-9,
        qubit_index=2,
    )
    node_c = NodeConfig(node_id="node_c", qubit_index=2)
    orch = TwinOrchestrator.build_simple(
        n_qubits=3,
        link_configs=[_LINK, link_1],
        node_configs=[_NODE_A, _NODE_B, node_c],
        duration_s=1.0,
        dt_s=0.1,
    )
    orch.run(steps=5)
    link0_results = orch.results_for_link("link_0")
    link1_results = orch.results_for_link("link_1")
    assert len(link0_results) == 5
    assert len(link1_results) == 5
    assert all(r.link_id == "link_0" for r in link0_results)
    assert all(r.link_id == "link_1" for r in link1_results)


def test_reset_clears_results() -> None:
    """reset() clears the result log and rewinds the clock to zero."""
    orch = TwinOrchestrator.build_simple(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=1.0,
        dt_s=0.1,
    )
    orch.run(steps=5)
    assert len(orch.results()) == 5
    orch.reset()
    assert orch.results() == []
    assert orch.current_t() == 0.0


def test_qber_timeseries_length() -> None:
    """qber_timeseries() returns one (t, qber) pair per step."""
    orch = TwinOrchestrator.build_simple(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=1.0,
        dt_s=0.1,
    )
    orch.run(steps=7)
    ts = orch.qber_timeseries("link_0")
    assert len(ts) == 7
    assert all(len(pt) == 2 for pt in ts)


def test_scenario_config_roundtrip(tmp_path) -> None:
    """A ScenarioConfig survives a to_json_file -> from_json_file round-trip."""
    scenario = ScenarioConfig(
        scenario_name="Test Scenario",
        nodes=[NodeConfigModel(node_id="node_a", qubit_index=0)],
        links=[
            LinkConfigModel(
                link_id="link_0",
                source_node="node_a",
                dest_node="node_b",
                qubit_index=0,
            )
        ],
        kernel=KernelModel(type="exponential", tau_x=30.0, tau_y=30.0, tau_z=120.0),
        duration_s=5.0,
        dt_s=0.2,
    )
    path = tmp_path / "scenario.json"
    scenario.to_json_file(str(path))

    loaded = ScenarioConfig.from_json_file(str(path))
    assert loaded.scenario_name == "Test Scenario"

    sim_config = loaded.to_simulation_config()
    assert sim_config.links[0].link_id == "link_0"
    assert sim_config.duration_s == 5.0
    assert sim_config.dt_s == 0.2


# ---------------------------------------------------------------------------
# Step 1 — Regression tests: five wiring chains produce concrete deltas
# ---------------------------------------------------------------------------

# Panel-default sensitivity (matches S_SMF28_DEFAULT and TelemetryPanel defaults).
_S_PANEL_DEFAULT = np.array(
    [[0.0, 0.001, 0.0005],
     [0.0, 0.001, 0.0],
     [0.002, 0.0, 0.0005]],
    dtype=np.float64,
)

# Custom scenario used by tests 4 and 5: Lorentzian kernel + longer fiber span.
_CUSTOM_SCENARIO = ScenarioConfig(
    scenario_name="Custom",
    nodes=[
        NodeConfigModel(node_id="node_a", qubit_index=0),
        NodeConfigModel(node_id="node_b", qubit_index=1),
    ],
    links=[
        LinkConfigModel(
            link_id="link_0",
            source_node="node_a",
            dest_node="node_b",
            qubit_index=0,
            fiber=FiberParamsModel(
                length_km=50.0,
                attenuation_db_per_km=0.3,
                eta_detector=0.7,
                t_opt=0.4,
                p_dc=2e-5,
            ),
        )
    ],
    kernel=KernelModel(type="lorentzian", gamma=0.1, omega_0=1.0),
    duration_s=1.0,
    dt_s=0.1,
)


def test_kernel_shape_produces_different_qber() -> None:
    """Unit-area kernels differ in non-Markovian behaviour due to temporal shape.

    Both kernels integrate to 1 over [0,∞) (verified numerically via trapezoidal
    rule). The physical discriminant is the RHP non-Markovianity witness N_RHP:
    the Lorentzian's oscillatory sign changes produce negative canonical rates
    (TCL inversion), accumulating a large N_RHP; the monotone exponential stays
    essentially Markovian.  Both kernels are unit-area — the difference is shape.
    """
    tau_arr = np.linspace(0.0, 500.0, 5001)

    exp_k = ExponentialKernel(tau_x=30.0, tau_y=30.0, tau_z=120.0)
    lor_k = LorentzianKernel(gamma=0.1, omega_0=1.0)

    # (a) Verify unit area for both kernels (trapezoidal rule).
    exp_vals = np.array([exp_k.eval(t)[0, 0] for t in tau_arr])
    lor_vals = np.array([lor_k.eval(t)[0, 0] for t in tau_arr])
    exp_area = float(np.trapezoid(exp_vals, tau_arr))
    lor_area = float(np.trapezoid(lor_vals, tau_arr))
    assert abs(exp_area - 1.0) < 0.02, f"Exponential kernel area={exp_area:.4f}, expected 1"
    assert abs(lor_area - 1.0) < 0.05, f"Lorentzian kernel area={lor_area:.4f}, expected 1"

    # (b) Shape: Lorentzian oscillates below zero at τ=π; exponential stays non-negative.
    assert lor_k.eval(np.pi)[0, 0] < 0.0
    assert exp_k.eval(1000.0)[0, 0] >= 0.0

    # (c) Shape drives non-Markovian behaviour: after 50 steps (5 s) the Lorentzian
    #     N_RHP is orders of magnitude larger than the essentially-Markovian exponential.
    #     50 steps covers ~0.8 of the Lorentzian's first oscillation period (2π/ω₀ ≈ 6.3 s),
    #     which is sufficient to observe the sign-change accumulation.  With the ΔT
    #     temperature-coupling fix, the thermal pedestal is gone and only seismic/wind
    #     fluctuations drive the signal; 3 steps produced near-zero witness values in
    #     the new (small-signal) regime.  50 steps remain well within the test budget.
    common = dict(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=10.0,
        dt_s=0.1,
        sensitivity=_S_PANEL_DEFAULT,
    )
    exp_orch = TwinOrchestrator.build_simple(**common, kernel=exp_k)
    lor_orch = TwinOrchestrator.build_simple(**common, kernel=lor_k)
    for _ in range(50):
        exp_r = exp_orch.step()[0]
        lor_r = lor_orch.step()[0]

    assert lor_r.rhp_witness > 0.05, (
        f"Lorentzian shape must accumulate substantial N_RHP: {lor_r.rhp_witness:.3e}"
    )
    assert lor_r.rhp_witness > exp_r.rhp_witness * 100, (
        f"Lorentzian N_RHP ({lor_r.rhp_witness:.3e}) must far exceed "
        f"exponential ({exp_r.rhp_witness:.3e})"
    )


def test_high_sensitivity_raises_qber() -> None:
    """A 100× larger sensitivity matrix produces a significantly higher QBER."""
    common = dict(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=1.0,
        dt_s=0.1,
    )
    low_qber = TwinOrchestrator.build_simple(
        **common, sensitivity=_S_PANEL_DEFAULT
    ).step()[0].qber
    high_qber = TwinOrchestrator.build_simple(
        **common, sensitivity=_S_PANEL_DEFAULT * 100.0
    ).step()[0].qber
    assert high_qber > low_qber * 5, (
        f"100× sensitivity should raise QBER substantially: "
        f"low={low_qber:.6f} high={high_qber:.6f}"
    )


def test_wdm_channel_via_build_simple_raises_raman() -> None:
    """A WDM channel registered through build_simple() produces a nonzero Raman rate."""
    no_wdm = TwinOrchestrator.build_simple(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=1.0,
        dt_s=0.1,
    ).step()[0]
    with_wdm = TwinOrchestrator.build_simple(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=1.0,
        dt_s=0.1,
        wdm_channels=[
            ClassicalChannelSpec(channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=10.0)
        ],
    ).step()[0]
    assert no_wdm.raman_rate_hz == 0.0
    assert with_wdm.raman_rate_hz > 0.0, (
        f"WDM channel should produce positive Raman rate, got {with_wdm.raman_rate_hz}"
    )


def test_build_orchestrator_custom_differs_from_default() -> None:
    """ScenarioConfig.build_orchestrator() with custom fiber/kernel yields a different QBER."""
    custom_qber = _CUSTOM_SCENARIO.build_orchestrator().step()[0].qber
    default_qber = TwinOrchestrator.build_simple(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=1.0,
        dt_s=0.1,
    ).step()[0].qber
    # τ_qubit-scaled baseline: custom (50 km, τ≈245 µs) ≈ 5.0e-9 < default (25 km, τ≈122 µs)
    # ≈ 9.2e-9; gap ≈ 4.2e-9.  QBER no longer scales with dt — both values are now
    # ~1000× smaller than the old dt=0.1 s figures (ratio ≈ τ_qubit/dt).
    # The Lorentzian kernel's spectral profile and the 50 km τ_qubit still yield a
    # lower QBER than the 25 km Exponential scenario; direction is preserved.
    # Threshold = 5e-10 ≈ 12% of the gap, giving ~8× safety margin.
    assert custom_qber < default_qber, (
        f"Lorentzian 50km scenario must have lower QBER than Exponential 25km: "
        f"custom={custom_qber:.3e} default={default_qber:.3e}"
    )
    assert abs(custom_qber - default_qber) > 5e-10, (
        f"QBER gap ({abs(custom_qber - default_qber):.3e}) below calibrated "
        f"threshold 5e-10 (τ_qubit-scaled baseline: ~4.2e-9)"
    )


def test_scenario_json_roundtrip_qber(tmp_path) -> None:
    """JSON round-trip of a custom scenario reproduces bit-identical QBER."""
    path = tmp_path / "custom.json"
    _CUSTOM_SCENARIO.to_json_file(str(path))
    loaded = ScenarioConfig.from_json_file(str(path))

    qber_original = _CUSTOM_SCENARIO.build_orchestrator().step()[0].qber
    qber_loaded = loaded.build_orchestrator().step()[0].qber
    assert qber_original == qber_loaded, (
        f"JSON round-trip should reproduce identical QBER: "
        f"original={qber_original} loaded={qber_loaded}"
    )


# ---------------------------------------------------------------------------
# Step 2 — Serialisation: sensitivity and coexistence_channels survive round-trip
# ---------------------------------------------------------------------------


def test_sensitivity_field_survives_serialization_roundtrip(tmp_path) -> None:
    """ScenarioConfig.sensitivity is preserved through to_json_file / from_json_file."""
    custom_s = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    scenario = ScenarioConfig(sensitivity=custom_s)
    path = tmp_path / "sens.json"
    scenario.to_json_file(str(path))
    loaded = ScenarioConfig.from_json_file(str(path))
    assert loaded.sensitivity == custom_s


def test_coexistence_channels_field_survives_serialization_roundtrip(tmp_path) -> None:
    """ScenarioConfig.coexistence_channels is preserved through to_json_file / from_json_file."""
    channels = [
        {"channel_id": "c1", "lambda_c_nm": 1310.0, "launch_power_mw": 5.0, "active": True}
    ]
    scenario = ScenarioConfig(coexistence_channels=channels)
    path = tmp_path / "co.json"
    scenario.to_json_file(str(path))
    loaded = ScenarioConfig.from_json_file(str(path))
    assert loaded.coexistence_channels == channels


# ---------------------------------------------------------------------------
# Step 5 — Sensitivity shape guard: invalid input raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_sensitivity_wrong_rows_raises() -> None:
    """validate_sensitivity_matrix raises ValueError when row count ≠ 3."""
    import pytest
    with pytest.raises(ValueError, match="3 rows"):
        validate_sensitivity_matrix([[0.1, 0.2], [0.3, 0.4]])


def test_invalid_sensitivity_ragged_raises() -> None:
    """validate_sensitivity_matrix raises ValueError for a ragged matrix."""
    import pytest
    with pytest.raises(ValueError, match="ragged"):
        validate_sensitivity_matrix([[0.1, 0.2], [0.3], [0.4, 0.5]])


def test_wrong_column_count_sensitivity_raises() -> None:
    """validate_sensitivity_matrix raises ValueError for a uniform-but-wrong column width.

    A (3, 2) matrix is not ragged and passes the row-count check, so it
    would previously pass validation and only crash at ``S @ E`` runtime.
    The column-count guard must catch it before that.
    """
    import pytest
    with pytest.raises(ValueError, match=r"\d+ columns"):
        validate_sensitivity_matrix([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])


def test_zero_column_sensitivity_raises() -> None:
    """validate_sensitivity_matrix raises ValueError for a (3, 0) matrix.

    A (3, 0) matrix has the right row count and is not ragged, so it would
    silently pass the row/ragged checks — but it is always unusable because
    S @ E requires M ≥ 1 columns.
    """
    import pytest
    with pytest.raises(ValueError, match="column"):
        validate_sensitivity_matrix([[], [], []])


def test_invalid_sensitivity_via_build_orchestrator(tmp_path) -> None:
    """build_orchestrator() raises ValueError when sensitivity has the wrong shape."""
    import pytest
    scenario = ScenarioConfig(
        nodes=[NodeConfigModel(node_id="node_a", qubit_index=0)],
        links=[LinkConfigModel(
            link_id="link_0", source_node="node_a", dest_node="node_b", qubit_index=0
        )],
        sensitivity=[[0.1, 0.2], [0.3, 0.4]],  # only 2 rows — invalid
    )
    with pytest.raises(ValueError, match="3 rows"):
        scenario.build_orchestrator()


# ---------------------------------------------------------------------------
# Bug-fix regression tests (Task 6 — Steps 2, 2b, 5)
# ---------------------------------------------------------------------------

def test_live_sources_populated() -> None:
    """build_simple() creates a live telemetry iterator for every configured link.

    Step 2b fix: the per-link live source keeps the resampler window current so
    is_stale() never fires during a long run.
    """
    orch = TwinOrchestrator.build_simple(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=10.0,
        dt_s=1.0,
    )
    assert "link_0" in orch._live_sources


def test_telemetry_not_stale_after_long_run() -> None:
    """Resampler is_stale() stays False after 15 steps of 1-second increments.

    Without the live-source fix the last prewarm sample is at t'=-1 (prewarm
    ends at t=0).  After 11 steps (11s elapsed) the gap exceeds max_gap_s=10s
    and is_stale() becomes True, cutting off the telemetry feed to the viewer.
    """
    orch = TwinOrchestrator.build_simple(
        n_qubits=2,
        link_configs=[_LINK],
        node_configs=[_NODE_A, _NODE_B],
        duration_s=20.0,
        dt_s=1.0,
    )
    for _ in range(15):
        orch.step()
    assert not orch._telemetry_engine.resampler.is_stale("link_0")


def test_scale_ptm_eigenvalues_markovian_dt_invariant() -> None:
    """Generator scaling is exactly dt-invariant for Markovian (all Γ_i ≥ 0) channels.

    Uses a symmetric depolarising channel λ = [0.5, 0.5, 0.5], which has
    Γ_X = Γ_Y = Γ_Z = ln(2)/2 ≈ 0.347 > 0.  Composing 10 steps of dt=0.1s
    must reproduce the direct 1.0s step to machine precision — the generator
    formula exp(−γ·dt) is exact when no CP clamping is needed.

    The SMF-28 sensitivity is NOT used here: its implied Γ_Y ≈ −0.010/s
    (non-Markovian) forces CP clamping at sub-dt values, breaking algebraic
    dt-invariance by design — that is documented behaviour, not a bug.
    """
    eigs_1s = np.array([0.5, 0.5, 0.5])
    direct = _scale_ptm_eigenvalues(eigs_1s, 1.0)
    composed = np.ones(3)
    for _ in range(10):
        composed *= _scale_ptm_eigenvalues(eigs_1s, 0.1)
    np.testing.assert_allclose(
        composed, direct, rtol=1e-10,
        err_msg="Generator scaling is not algebraically dt-invariant for Markovian channel",
    )


# Balanced sensitivity: X and Y rows equal → lx = ly → Γ_Y ≈ 0 → Markovian.
# SMF-28 default has slightly different X/Y rows (Γ_Y ≈ −0.010/s) which causes
# CP clamping at sub-dt and breaks full-simulation dt-invariance.  This matrix
# is used only in the dt-convergence test where Markovian behaviour is required.
_S_BALANCED = np.array(
    [[0.0, 0.001, 0.0], [0.0, 0.001, 0.0], [0.002, 0.0, 0.0]], dtype=float
)


def test_dt_convergence_markovian_channel() -> None:
    """QBER is dt-invariant for a Markovian channel (τ_qubit-scaled, not dt-scaled).

    Since QBER is now computed from the effective PTM scaled to τ_qubit
    (LinkConfig.qubit_exposure_s) rather than dt_s, the per-step QBER no longer
    grows with dt.  Both orchestrators share the same _LINK (same qubit_exposure_s)
    and the same t=0 effective PTM (identical prewarm, no live sources), so
    QBER(dt=1.0) ≈ QBER(dt=0.1) to machine precision.

    _S_BALANCED has equal X/Y sensitivity rows → lx=ly → Γ_Y=0 → Markovian:
    no CP clamping is needed and the generator formula is algebraically exact.
    """
    def _build(dt: float) -> TwinOrchestrator:
        o = TwinOrchestrator.build_simple(
            n_qubits=2,
            link_configs=[_LINK],
            node_configs=[_NODE_A, _NODE_B],
            duration_s=5.0,
            dt_s=dt,
            sensitivity=_S_BALANCED,
        )
        o._live_sources = {}
        return o

    orch_coarse = _build(1.0)
    orch_fine = _build(0.1)

    # QBER is τ_qubit-scaled: both orchestrators use the same qubit_exposure_s,
    # so QBER is identical regardless of the numerical step size.
    qber_coarse = orch_coarse.step()[0].qber   # dt = 1.0 s
    qber_fine = orch_fine.step()[0].qber       # dt = 0.1 s

    rel_diff = abs(qber_coarse - qber_fine) / max(qber_coarse, qber_fine, 1e-12)
    assert rel_diff < 0.01, (
        f"QBER must be dt-invariant (τ_qubit-scaled, §5.7): "
        f"dt=1.0 QBER={qber_coarse:.3e}, "
        f"dt=0.1 QBER={qber_fine:.3e}, "
        f"rel_diff={rel_diff:.4f}"
    )


def test_qber_dt_invariant() -> None:
    """QBER(dt) == QBER(dt/2) within tolerance (§5.7 τ_qubit physical scaling).

    Builds two orchestrators with dt and dt/2, both using the same _LINK
    (same qubit_exposure_s) and the same t=0 effective PTM (no live sources).
    QBER must agree to < 1% regardless of the numerical timestep.

    Uses _S_BALANCED (Markovian) so no CP clamping breaks the exact semigroup
    identity that underlies the τ_qubit-scaling formula.
    """
    def _build(dt: float) -> TwinOrchestrator:
        o = TwinOrchestrator.build_simple(
            n_qubits=2,
            link_configs=[_LINK],
            node_configs=[_NODE_A, _NODE_B],
            duration_s=5.0,
            dt_s=dt,
            sensitivity=_S_BALANCED,
        )
        o._live_sources = {}
        return o

    qber_dt = _build(0.2).step()[0].qber
    qber_half = _build(0.1).step()[0].qber

    rel_diff = abs(qber_dt - qber_half) / max(qber_dt, qber_half, 1e-12)
    assert rel_diff < 0.01, (
        f"QBER must be dt-invariant (§5.7): "
        f"QBER(dt=0.2)={qber_dt:.6e}, QBER(dt=0.1)={qber_half:.6e}, "
        f"rel_diff={rel_diff:.4f}"
    )


# ---------------------------------------------------------------------------
# Step 3 — Temperature-origin invariance (pins the ΔT bug closed permanently)
# ---------------------------------------------------------------------------

def test_temperature_origin_invariance() -> None:
    """Pauli rates are invariant to a uniform shift of the temperature origin.

    Regression test for the ΔT coupling fix: driving the convolution on
    absolute T (old code) makes the model sensitive to the Celsius vs Kelvin
    choice of origin.  With the ΔT fix (``E − env_ref``), a +shift to all
    temperatures AND the same +shift to ``env_ref[0]`` must produce identical
    Pauli rates.

    This test FAILS against the pre-fix absolute-T code because the Celsius
    and Kelvin runs differ by S[pz,T]×shift in every convolution term.
    It PASSES after the fix because ``(E+shift) − (E_ref+shift) = E − E_ref``.
    """
    from qndt.telemetry.engine import EnvironmentalTelemetryEngine
    from qndt.telemetry.sources import TelemetrySample

    kernel = ExponentialKernel(tau_x=30.0, tau_y=30.0, tau_z=120.0)
    S = _S_PANEL_DEFAULT

    T_ref = 20.0          # Celsius operating point
    shift = 273.15        # shift to probe origin invariance (Kelvin offset)

    # Build a deterministic sample sequence in "Celsius" units.
    rng = np.random.default_rng(7)
    n_samples = 15
    samples_celsius = [
        TelemetrySample(
            t=float(i),
            E=np.array([
                T_ref + 5.0 * math.sin(2 * math.pi * i / 3600.0) + float(rng.normal(0, 0.1)),
                float(rng.normal(0, 0.001)),
                abs(float(rng.normal(0, 0.1))),
            ]),
            link_id="link_inv",
        )
        for i in range(n_samples)
    ]

    # Same physical temperatures expressed in shifted units (+273.15).
    samples_shifted = [
        TelemetrySample(
            t=s.t,
            E=np.array([s.E[0] + shift, s.E[1], s.E[2]]),
            link_id="link_inv",
        )
        for s in samples_celsius
    ]

    query_t = float(n_samples - 1)

    # Engine A: Celsius temperatures, reference at T_ref.
    engine_c = EnvironmentalTelemetryEngine(
        sensitivity=S, kernel=kernel,
        env_ref=np.array([T_ref, 0.0, 0.0]),
    )
    for s in samples_celsius:
        engine_c.ingest(s)
    rates_c = engine_c.pauli_rates("link_inv", query_t)

    # Engine B: shifted temperatures, reference shifted by the same amount.
    engine_s = EnvironmentalTelemetryEngine(
        sensitivity=S, kernel=kernel,
        env_ref=np.array([T_ref + shift, 0.0, 0.0]),
    )
    for s in samples_shifted:
        engine_s.ingest(s)
    rates_s = engine_s.pauli_rates("link_inv", query_t)

    np.testing.assert_allclose(
        [rates_c.px, rates_c.py, rates_c.pz],
        [rates_s.px, rates_s.py, rates_s.pz],
        rtol=1e-10,
        err_msg=(
            "Pauli rates must be invariant to a uniform shift of the "
            "temperature origin when env_ref is shifted by the same amount."
        ),
    )


# ---------------------------------------------------------------------------
# B1 — WDM load schedule integration test (§3.3)
# ---------------------------------------------------------------------------

def test_wdm_schedule_raman_varies_over_run() -> None:
    """A WDM schedule causes Raman rate to change between steps (§3.3 B1).

    Schedule: activate at t=2 → deactivate at t=4.
    - Before t=2: link is unmanaged (no activate yet) → static dict empty → rate 0.
    - At t=2: activate fires before link loop → link managed, channel active → rate > 0.
    - At t=4: deactivate fires → link managed, channel off → rate 0 (B2 semantics).

    Uses ScenarioConfig.build_orchestrator() to exercise the full config→engine path.
    """
    scenario = ScenarioConfig(
        scenario_name="B1 schedule test",
        nodes=[
            NodeConfigModel(node_id="A", qubit_index=0),
            NodeConfigModel(node_id="B", qubit_index=1),
        ],
        links=[
            LinkConfigModel(
                link_id="l0",
                source_node="A",
                dest_node="B",
                qubit_index=0,
            )
        ],
        wdm_schedule=[
            WDMScheduleEventModel(
                t=2.0, link_id="l0", action="activate",
                channel_id="c1", lambda_c_nm=1310.0, launch_power_mw=1.0,
            ),
            WDMScheduleEventModel(
                t=4.0, link_id="l0", action="deactivate",
                channel_id="c1",
            ),
        ],
        duration_s=6.0,
        dt_s=1.0,
    )
    orch = scenario.build_orchestrator()
    # Disable live telemetry sources so each step is deterministic.
    orch._live_sources = {}

    r0 = orch.step()[0]  # self._t=0 before step: no event (t_event=2 > 0)
    r1 = orch.step()[0]  # self._t=1: still no event
    r2 = orch.step()[0]  # self._t=2: activate fires → channel active → rate > 0
    r3 = orch.step()[0]  # self._t=3: channel still active
    r4 = orch.step()[0]  # self._t=4: deactivate fires → managed + empty → rate 0
    r5 = orch.step()[0]  # self._t=5: channel still off

    assert r0.raman_rate_hz == pytest.approx(0.0, abs=1e-30), (
        f"t=0 (before schedule): expected rate=0, got {r0.raman_rate_hz:.4e}"
    )
    assert r1.raman_rate_hz == pytest.approx(0.0, abs=1e-30), (
        f"t=1 (before schedule): expected rate=0, got {r1.raman_rate_hz:.4e}"
    )
    assert r2.raman_rate_hz > 0.0, (
        f"t=2 (activation): expected rate > 0, got {r2.raman_rate_hz:.4e}"
    )
    assert r3.raman_rate_hz > 0.0, (
        f"t=3 (channel active): expected rate > 0, got {r3.raman_rate_hz:.4e}"
    )
    assert r4.raman_rate_hz == pytest.approx(0.0, abs=1e-30), (
        f"t=4 (deactivation): expected rate=0 (B2 managed+empty), got {r4.raman_rate_hz:.4e}"
    )
    assert r5.raman_rate_hz == pytest.approx(0.0, abs=1e-30), (
        f"t=5 (channel off): expected rate=0, got {r5.raman_rate_hz:.4e}"
    )
