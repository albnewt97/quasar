"""Tests for the simulation engine <-> GUI wiring (§4.2).

Run headless via ``QT_QPA_PLATFORM=offscreen`` (see conftest.py qapp_env).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from pytestqt.qtbot import QtBot

from qndt.core.orchestrator import LinkConfig, NodeConfig, TwinOrchestrator
from qndt.gui.main_window import QuasarMainWindow
from qndt.gui.simulation_controller import SimulationController
from qndt.gui.simulation_runner import SimulationRunner, SimulationSignals
from qndt.io.config import KernelModel, LinkConfigModel, NodeConfigModel, ScenarioConfig


def _build_test_orchestrator() -> TwinOrchestrator:
    node_configs = [
        NodeConfig(node_id="Alice", qubit_index=0),
        NodeConfig(node_id="Bob", qubit_index=1),
    ]
    link_configs = [
        LinkConfig(
            link_id="link_01",
            source_node="Alice",
            dest_node="Bob",
            lambda_q_nm=1550.0,
            gate_width_s=1e-9,
            qubit_index=0,
        ),
    ]
    return TwinOrchestrator.build_simple(
        n_qubits=2,
        link_configs=link_configs,
        node_configs=node_configs,
        duration_s=1.0,
        dt_s=0.1,
    )


def test_simulation_runner_creates() -> None:
    """SimulationRunner() constructs without starting the thread."""
    orchestrator = _build_test_orchestrator()
    signals = SimulationSignals()
    runner = SimulationRunner(orchestrator, signals)
    assert not runner.isRunning()


def test_simulation_runner_single_step(qtbot: QtBot) -> None:
    """Starting the runner emits step_completed within 5 seconds."""
    orchestrator = _build_test_orchestrator()
    signals = SimulationSignals()
    runner = SimulationRunner(orchestrator, signals)

    with qtbot.waitSignal(signals.step_completed, timeout=5000):
        runner.start()

    runner.stop()
    runner.wait()


def test_simulation_controller_creates(qtbot: QtBot) -> None:
    """SimulationController() constructs from a fresh main window."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    controller = SimulationController(window, window.dashboard, window._topology_model)
    assert controller is not None


def test_reset_clears_clock(qtbot: QtBot) -> None:
    """_on_reset() restores the status bar clock label to t = 0.000 s."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    controller = SimulationController(window, window.dashboard, window._topology_model)

    window.update_clock(5.0)
    controller._on_reset()

    assert "0.000" in window._clock_label.text()


# ---------------------------------------------------------------------------
# Bug 2 regression: per-node aging config wiring tests
# ---------------------------------------------------------------------------


def test_node_config_change_stored(qtbot: QtBot) -> None:
    """Emitting node_config_changed stores values in controller._node_aging_overrides."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    controller = SimulationController(window, window.dashboard, window._topology_model)

    window._canvas.node_config_changed.emit(
        "n1", {"t2_nominal": 0.5, "wear_rate_kappa": 5e5, "node_type": "memory_node"}
    )

    assert "n1" in controller._node_aging_overrides
    assert controller._node_aging_overrides["n1"]["t2_nominal"] == pytest.approx(0.5)
    assert controller._node_aging_overrides["n1"]["wear_rate_kappa"] == pytest.approx(5e5)


def test_node_override_applied_on_build(qtbot: QtBot) -> None:
    """Per-node aging override values are used for dashboard.add_node on build."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    controller = SimulationController(window, window.dashboard, window._topology_model)

    window._topology_model.add_node("Alice", 0.0, 0.0, node_type="memory_node")
    window._topology_model.add_node("Bob", 200.0, 0.0, node_type="memory_node")
    window._topology_model.add_link("link_01", "Alice", "Bob")

    window._canvas.node_config_changed.emit(
        "Alice", {"t2_nominal": 0.3, "wear_rate_kappa": 2e5, "node_type": "memory_node"}
    )

    controller._build_orchestrator()

    assert "Alice" in window.dashboard.aging_plot._node_params
    t2, nc = window.dashboard.aging_plot._node_params["Alice"]
    assert t2 == pytest.approx(0.3)
    assert nc == pytest.approx(2e5)


def test_controller_pushes_telemetry(qtbot: QtBot) -> None:
    """_on_step_completed calls telemetry_viewer.ingest_sample for the selected link."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    window._topology_model.add_node("Alice", 0.0, 0.0, node_type="memory_node")
    window._topology_model.add_node("Bob", 200.0, 0.0, node_type="memory_node")
    window._topology_model.add_link("link_01", "Alice", "Bob")
    controller = SimulationController(window, window.dashboard, window._topology_model)
    orchestrator = controller._build_orchestrator()
    assert orchestrator is not None
    controller._orchestrator = orchestrator

    window._telemetry_viewer.update_link_list(["link_01"])
    window._telemetry_viewer._link_combo.setCurrentText("link_01")

    with patch.object(window._telemetry_viewer, "ingest_sample") as mock_ingest:
        # t=0.5 is within the prewarm window; gap from last prewarm sample is ~1.5s < 10s
        controller._on_step_completed(0.5, "link_01", 0.05, 0.9, 1e-4, 0.01, 0.0, 100.0)
        assert mock_ingest.called


def test_controller_updates_nonmarkov(qtbot: QtBot) -> None:
    """_on_step_completed calls dashboard.update_nonmarkov when canonical rates exist."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    window._topology_model.add_node("Alice", 0.0, 0.0, node_type="memory_node")
    window._topology_model.add_node("Bob", 200.0, 0.0, node_type="memory_node")
    window._topology_model.add_link("link_01", "Alice", "Bob")
    controller = SimulationController(window, window.dashboard, window._topology_model)
    orchestrator = controller._build_orchestrator()
    assert orchestrator is not None
    controller._orchestrator = orchestrator

    # Two pauli_rates calls populate _last_canonical_rates for link_01
    orchestrator._telemetry_engine.pauli_rates("link_01", 0.0)
    orchestrator._telemetry_engine.pauli_rates("link_01", 0.1)
    assert orchestrator._telemetry_engine.latest_canonical_rates("link_01") is not None

    with patch.object(window.dashboard, "update_nonmarkov") as mock_nm:
        controller._on_step_completed(0.1, "link_01", 0.05, 0.9, 1e-4, 0.01, 0.0, 100.0)
        assert mock_nm.called


def test_controller_updates_key_rate(qtbot: QtBot) -> None:
    """_on_step_completed calls update_key_rate AND update_key_rate_display."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    window._topology_model.add_node("Alice", 0.0, 0.0, node_type="memory_node")
    window._topology_model.add_node("Bob", 200.0, 0.0, node_type="memory_node")
    window._topology_model.add_link("link_01", "Alice", "Bob")
    controller = SimulationController(window, window.dashboard, window._topology_model)
    orchestrator = controller._build_orchestrator()
    assert orchestrator is not None
    controller._orchestrator = orchestrator

    with patch.object(window.dashboard, "update_key_rate") as mock_kr, \
         patch.object(window._channel_panel, "update_key_rate_display") as mock_panel:
        controller._on_step_completed(0.1, "link_01", 0.05, 0.9, 1e-4, 0.01, 0.0, 100.0)
        assert mock_kr.called
        assert mock_panel.called


def test_engine_has_canonical_rates(qtbot: QtBot) -> None:
    """EnvironmentalTelemetryEngine.latest_canonical_rates() returns non-None after 2+ steps."""
    orch = _build_test_orchestrator()
    orch.step()
    orch.step()
    cr = orch._telemetry_engine.latest_canonical_rates("link_01")
    assert cr is not None
    assert hasattr(cr, "gamma_x")
    assert hasattr(cr, "gamma_y")
    assert hasattr(cr, "gamma_z")


def test_per_node_override_reaches_model(qtbot: QtBot) -> None:
    """Per-node aging override flows all the way into the orchestrator's DeviceAgingModel."""
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    controller = SimulationController(window, window.dashboard, window._topology_model)

    window._topology_model.add_node("Alice", 0.0, 0.0, node_type="memory_node")
    window._topology_model.add_node("Bob", 200.0, 0.0, node_type="memory_node")
    window._topology_model.add_link("link_01", "Alice", "Bob")

    window._canvas.node_config_changed.emit(
        "Alice", {"t2_nominal": 0.4, "wear_rate_kappa": 3e5, "node_type": "memory_node"}
    )

    orchestrator = controller._build_orchestrator()
    assert orchestrator is not None

    params = orchestrator._aging_model.node_params("Alice")
    assert params["t2_nominal"] == pytest.approx(0.4)
    assert params["wear_rate_kappa"] == pytest.approx(3e5)


# ---------------------------------------------------------------------------
# Step 3 — Load-then-Apply clobber regression tests
# ---------------------------------------------------------------------------


def test_load_then_apply_preserves_fiber_and_sim_params(tmp_path, qtbot: QtBot) -> None:
    """Loading a scenario and clicking Apply on the channel panel keeps the loaded fiber values.

    Without the ChannelPanel.load_config() call in _load_scenario, the channel
    panel would still show its defaults after load.  Clicking Apply would emit
    those defaults, overwriting length_km, duration_s, chi_max, etc.
    """
    from qndt.io.config import FiberParamsModel

    scenario = ScenarioConfig(
        scenario_name="Fiber Test",
        nodes=[
            NodeConfigModel(node_id="Alice", qubit_index=0),
            NodeConfigModel(node_id="Bob", qubit_index=1),
        ],
        links=[
            LinkConfigModel(
                link_id="link_01",
                source_node="Alice",
                dest_node="Bob",
                qubit_index=0,
                fiber=FiberParamsModel(
                    length_km=50.0,  # default 25.0
                    attenuation_db_per_km=0.3,  # default 0.2
                    eta_detector=0.6,  # default 0.8
                ),
            )
        ],
        kernel=KernelModel(type="exponential"),
        duration_s=30.0,  # default 10.0
        dt_s=0.5,  # default 0.1
        chi_max=8,  # default 4
    )
    path = tmp_path / "fiber.json"
    scenario.to_json_file(str(path))

    window = QuasarMainWindow()
    qtbot.addWidget(window)
    controller = SimulationController(window, window.dashboard, window._topology_model)
    controller._load_scenario(str(path))

    # Simulate the user clicking Apply on the channel panel without any edits.
    controller._on_config_changed("channel", window._channel_panel.get_config())

    assert controller._current_fiber_config.get("length_km") == pytest.approx(50.0)
    assert controller._current_fiber_config.get("attenuation_db_per_km") == pytest.approx(0.3)
    assert controller._current_fiber_config.get("eta_detector") == pytest.approx(0.6)
    assert controller._current_scenario.duration_s == pytest.approx(30.0)
    assert controller._current_scenario.dt_s == pytest.approx(0.5)
    assert controller._current_scenario.chi_max == 8


def test_load_then_apply_preserves_kernel(tmp_path, qtbot: QtBot) -> None:
    """Loading a Lorentzian scenario then clicking Apply on the telemetry panel keeps Lorentzian.

    Without the load_config() fix, the telemetry panel would still show the
    default Exponential kernel after scenario load, so clicking Apply would
    silently overwrite the loaded kernel with Exponential.
    """
    scenario = ScenarioConfig(
        scenario_name="Lorentzian Test",
        nodes=[
            NodeConfigModel(node_id="Alice", qubit_index=0),
            NodeConfigModel(node_id="Bob", qubit_index=1),
        ],
        links=[
            LinkConfigModel(
                link_id="link_01",
                source_node="Alice",
                dest_node="Bob",
                qubit_index=0,
            )
        ],
        kernel=KernelModel(type="lorentzian", gamma=0.05, omega_0=2.0),
    )
    path = tmp_path / "lorentzian.json"
    scenario.to_json_file(str(path))

    window = QuasarMainWindow()
    qtbot.addWidget(window)
    controller = SimulationController(window, window.dashboard, window._topology_model)

    controller._load_scenario(str(path))

    # Simulate the user clicking Apply on the telemetry panel without any edits.
    controller._on_config_changed("telemetry", window._telemetry_panel.get_config())

    assert controller._current_scenario.kernel.type == "lorentzian"
    assert controller._current_scenario.kernel.gamma == pytest.approx(0.05, rel=0.01)


def test_load_then_apply_preserves_sensitivity(tmp_path, qtbot: QtBot) -> None:
    """Loading a scenario with a non-default sensitivity keeps it after telemetry Apply.

    _load_scenario() calls telemetry_panel.load_config(sensitivity=...) so the
    panel spinboxes reflect the loaded matrix.  A subsequent Apply must read
    those spinboxes and write the loaded values back into _current_scenario —
    NOT the panel's hardwired SMF-28 default.
    """
    custom_s = [
        [0.0, 0.005, 0.0025],
        [0.0, 0.005, 0.0],
        [0.01, 0.0, 0.0025],
    ]
    scenario = ScenarioConfig(
        scenario_name="Sensitivity Test",
        nodes=[
            NodeConfigModel(node_id="Alice", qubit_index=0),
            NodeConfigModel(node_id="Bob", qubit_index=1),
        ],
        links=[
            LinkConfigModel(
                link_id="link_01",
                source_node="Alice",
                dest_node="Bob",
                qubit_index=0,
            )
        ],
        sensitivity=custom_s,
    )
    path = tmp_path / "sensitivity.json"
    scenario.to_json_file(str(path))

    window = QuasarMainWindow()
    qtbot.addWidget(window)
    controller = SimulationController(window, window.dashboard, window._topology_model)
    controller._load_scenario(str(path))

    # Simulate Apply on the telemetry panel without any user edits.
    controller._on_config_changed("telemetry", window._telemetry_panel.get_config())

    loaded_s = controller._current_scenario.sensitivity
    assert loaded_s is not None, "sensitivity was None after load+Apply"
    for row_i, row in enumerate(custom_s):
        for col_i, val in enumerate(row):
            assert loaded_s[row_i][col_i] == pytest.approx(val, abs=1e-6), (
                f"sensitivity[{row_i}][{col_i}] clobbered: "
                f"expected {val}, got {loaded_s[row_i][col_i]}"
            )


# ---------------------------------------------------------------------------
# Step 2 (GUI path) — _save_scenario must persist sensitivity and coexistence_channels
# ---------------------------------------------------------------------------


def test_wrong_column_count_surfaces_dialog_not_crash(qtbot: QtBot) -> None:
    """A correct-rows / wrong-cols sensitivity matrix surfaces a dialog via _build_orchestrator.

    A (3, 2) matrix passes row-count and ragged checks but fails the column-count
    guard.  Without the try/except in _build_orchestrator this ValueError would
    propagate uncaught; with it the controller shows QMessageBox.critical and
    returns None instead of crashing at S @ E.
    """
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    controller = SimulationController(window, window.dashboard, window._topology_model)

    window._topology_model.add_node("Alice", 0.0, 0.0, node_type="memory_node")
    window._topology_model.add_node("Bob", 200.0, 0.0, node_type="memory_node")
    window._topology_model.add_link("link_01", "Alice", "Bob")

    # Correct row count, uniform width, but wrong column count (2 instead of 3).
    controller._current_scenario.sensitivity = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    with patch("qndt.gui.simulation_controller.QMessageBox") as mock_mb:
        result = controller._build_orchestrator()

    assert result is None
    mock_mb.critical.assert_called_once()


def test_malformed_sensitivity_surfaces_dialog_not_crash(qtbot: QtBot) -> None:
    """A malformed sensitivity matrix shows a QMessageBox.critical and returns None.

    Without the try/except in _build_orchestrator the ValueError propagates
    as an uncaught exception and the application crashes instead of surfacing
    a user-readable error.
    """
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    controller = SimulationController(window, window.dashboard, window._topology_model)

    window._topology_model.add_node("Alice", 0.0, 0.0, node_type="memory_node")
    window._topology_model.add_node("Bob", 200.0, 0.0, node_type="memory_node")
    window._topology_model.add_link("link_01", "Alice", "Bob")

    # Only 2 rows instead of the required 3
    controller._current_scenario.sensitivity = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    with patch("qndt.gui.simulation_controller.QMessageBox") as mock_mb:
        result = controller._build_orchestrator()

    assert result is None
    mock_mb.critical.assert_called_once()


def test_save_preserves_sensitivity_and_coexistence(tmp_path, qtbot: QtBot) -> None:
    """_save_scenario includes sensitivity and coexistence_channels in the serialised JSON.

    Without the fix, _save_scenario constructed a new ScenarioConfig without
    passing these two fields, so both were silently dropped on save.
    """
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    controller = SimulationController(window, window.dashboard, window._topology_model)

    custom_s = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    custom_channels = [
        {"channel_id": "c1", "lambda_c_nm": 1310.0, "launch_power_mw": 5.0, "active": True}
    ]
    controller._current_scenario.sensitivity = custom_s
    controller._current_scenario.coexistence_channels = custom_channels

    path = str(tmp_path / "saved.json")
    controller._save_scenario(path)

    loaded = ScenarioConfig.from_json_file(path)
    assert loaded.sensitivity == custom_s
    assert loaded.coexistence_channels == custom_channels


# ---------------------------------------------------------------------------
# Bug-fix regression tests (Task 6 — Steps 1, 4)
# ---------------------------------------------------------------------------

def test_duration_applied_without_explicit_apply(qtbot: QtBot) -> None:
    """_build_orchestrator() picks up duration from the panel without clicking Apply.

    _on_value_changed() only sets _dirty=True; it does NOT emit config_changed.
    The panel-sync call at the top of _build_orchestrator() is required to push
    the current spinbox value into _current_scenario.duration_s before build.
    """
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    controller = SimulationController(window, window.dashboard, window._topology_model)

    window._topology_model.add_node("Alice", 0.0, 0.0, node_type="memory_node")
    window._topology_model.add_node("Bob", 200.0, 0.0, node_type="memory_node")
    window._topology_model.add_link("link_01", "Alice", "Bob")

    # Change duration in the spinbox but do NOT click Apply.
    window._channel_panel._duration_s.setValue(30.0)

    orchestrator = controller._build_orchestrator()

    assert orchestrator is not None
    assert orchestrator._config.duration_s == 30.0


def test_n_classical_reflects_channel_count(qtbot: QtBot) -> None:
    """_on_step_completed passes the actual active WDM channel count to update_raman.

    Previously n_classical was hardcoded to 1 in _on_step_completed, so the
    Raman noise plot always showed "1 classical channel" regardless of how many
    channels were configured in the coexistence panel.
    """
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    controller = SimulationController(window, window.dashboard, window._topology_model)

    controller._current_scenario.coexistence_channels = [
        {"channel_id": "c1", "lambda_c_nm": 1310.0, "launch_power_mw": 5.0, "active": True},
        {"channel_id": "c2", "lambda_c_nm": 1330.0, "launch_power_mw": 5.0, "active": True},
    ]

    with patch.object(controller._dashboard, "update_raman") as mock_raman:
        controller._on_step_completed(
            t=1.0,
            link_id="link_01",
            qber=0.03,
            fidelity=0.95,
            raman_rate=100.0,
            rhp_witness=0.0,
            induced_idle=0.0,
            skr_bps=1e6,
        )

    mock_raman.assert_called_once_with("link_01", 1.0, 100.0, 2)


def test_live_step_populates_all_plots(qtbot: QtBot) -> None:
    """3 real orchestrator steps with _on_step_completed must leave data in every panel.

    This test drives the full pipeline without mocks to catch silent regressions
    where a method is called but data never reaches the rolling buffers.
    Non-Markovian requires 2+ steps (canonical rates unavailable on step 1).
    Telemetry is only pushed for the link selected in the combo.
    """
    window = QuasarMainWindow()
    qtbot.addWidget(window)
    window._topology_model.add_node("Alice", 0.0, 0.0, node_type="memory_node")
    window._topology_model.add_node("Bob", 200.0, 0.0, node_type="memory_node")
    window._topology_model.add_link("link_01", "Alice", "Bob")

    controller = SimulationController(window, window.dashboard, window._topology_model)
    orchestrator = controller._build_orchestrator()
    assert orchestrator is not None
    controller._orchestrator = orchestrator

    window._telemetry_viewer.update_link_list(["link_01"])
    window._telemetry_viewer._link_combo.setCurrentText("link_01")

    for _ in range(3):
        for result in orchestrator.step():
            controller._on_step_completed(
                result.t,
                result.link_id,
                result.qber,
                result.fidelity,
                result.raman_rate_hz,
                result.rhp_witness,
                result.induced_idle_s,
                result.secret_key_rate_bps,
            )

    dash = window.dashboard
    assert len(dash._qber_plot._xs["link_01"]) > 0, "qber_plot has no data"
    assert len(dash._fidelity_plot._xs["link_01"]) > 0, "fidelity_plot has no data"
    assert len(dash._key_rate_plot._xs["link_01"]) > 0, "key_rate_plot has no data"
    # Non-Markovian: data arrives from step 2 onwards (canonical rates need 2 pauli_rates calls).
    assert len(dash._nonmarkov_plot._witness_xs.get("link_01", [])) > 0, (
        "nonmarkov_plot has no data after 3 steps"
    )
    assert len(window._telemetry_viewer._xs["temp"]) > 0, "telemetry viewer has no data"
