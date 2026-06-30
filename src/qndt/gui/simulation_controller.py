"""SimulationController: mediator between the GUI and SimulationRunner (§4.2).

Lives in the Qt main thread and owns the ``SimulationRunner`` worker
thread.  All cross-thread traffic flows through ``SimulationSignals``;
this controller never touches ``TwinOrchestrator`` state from outside the
worker thread while it is running.
"""
from __future__ import annotations

import logging
from typing import cast

import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QMessageBox

from qndt.core.orchestrator import LinkConfig, NodeConfig, TwinOrchestrator
from qndt.gui.dashboard.dashboard_window import DashboardWindow
from qndt.gui.main_window import QuasarMainWindow
from qndt.gui.simulation_runner import SimulationRunner, SimulationSignals
from qndt.gui.topology.topology_model import TopologyModel
from qndt.io.config import (
    FiberParamsModel,
    KernelModel,
    LinkConfigModel,
    NodeConfigModel,
    ScenarioConfig,
    validate_sensitivity_matrix,
)
from qndt.physics.raman import ClassicalChannelSpec, FiberParams
from qndt.telemetry.sources import TelemetrySample

_log = logging.getLogger(__name__)

_DEFAULT_LAMBDA_Q_NM = 1550.0
_DEFAULT_GATE_WIDTH_S = 1e-9
_NODE_LAYOUT_SPACING = 150.0

_FIBER_KEYS: frozenset[str] = frozenset(
    {"lambda_q_nm", "gate_width_s", "length_km", "attenuation_db_per_km",
     "eta_detector", "t_opt", "p_dc"}
)
_SIM_KEYS: frozenset[str] = frozenset({"duration_s", "dt_s", "chi_max"})


def _parse_wdm_channels(raw: list[dict[str, object]]) -> list[ClassicalChannelSpec]:
    """Convert a list of raw channel dicts to ``ClassicalChannelSpec`` instances.

    Skips entries with ``active=False`` or invalid values.
    """
    specs: list[ClassicalChannelSpec] = []
    for ch in raw:
        if not ch.get("active", True):
            continue
        try:
            specs.append(
                ClassicalChannelSpec(
                    channel_id=str(ch.get("channel_id", "")),
                    lambda_c_nm=float(ch["lambda_c_nm"]),  # type: ignore[arg-type]
                    launch_power_mw=float(ch["launch_power_mw"]),  # type: ignore[arg-type]
                )
            )
        except (KeyError, ValueError, TypeError):
            pass
    return specs


class SimulationController(QObject):
    """Mediates between the main window, dashboard, and SimulationRunner."""

    def __init__(
        self,
        main_window: QuasarMainWindow,
        dashboard: DashboardWindow,
        topology_model: TopologyModel,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._main_window = main_window
        self._dashboard = dashboard
        self._topology_model = topology_model
        self._runner: SimulationRunner | None = None
        self._signals: SimulationSignals | None = None
        self._orchestrator: TwinOrchestrator | None = None
        self._current_scenario: ScenarioConfig = ScenarioConfig()
        self._node_aging_overrides: dict[str, dict[str, float]] = {}
        # Fiber/channel params from the channel panel (set by Apply click or JSON load).
        # All values are float; chi_max goes to _current_scenario, not here.
        self._current_fiber_config: dict[str, float] = {}

        main_window.simulation_run_requested.connect(self._on_run)
        main_window.simulation_step_requested.connect(self._on_step)
        main_window.simulation_pause_requested.connect(self._on_pause)
        main_window.simulation_reset_requested.connect(self._on_reset)
        main_window.scenario_new_requested.connect(self._on_new_scenario)
        main_window.scenario_open_requested.connect(self._load_scenario)
        topology_model.topology_changed.connect(self._on_topology_changed)
        main_window._canvas.node_config_changed.connect(self._on_node_config_changed)

        main_window._channel_panel.config_changed.connect(self._on_config_changed)
        main_window._coexistence_panel.config_changed.connect(self._on_config_changed)
        main_window._aging_panel.config_changed.connect(self._on_config_changed)
        main_window._control_plane_panel.config_changed.connect(self._on_config_changed)
        main_window._telemetry_panel.config_changed.connect(self._on_config_changed)

    # ------------------------------------------------------------------
    # Orchestrator construction
    # ------------------------------------------------------------------

    def _build_orchestrator(self) -> TwinOrchestrator | None:
        # Sync duration_s/dt_s/chi_max from the channel panel into _current_scenario
        # so the user does not need to click Apply before pressing Run.
        # _on_value_changed() only sets _dirty=True without emitting, so without
        # this call a spinbox change would be silently ignored.
        # The coexistence panel already commits on every table change; telemetry
        # panel changes (kernel/sensitivity) require explicit Apply.
        self._on_config_changed("channel", self._main_window._channel_panel.get_config())

        nodes = self._topology_model.to_scenario_nodes()
        links = self._topology_model.to_scenario_links()

        if not nodes or not links:
            QMessageBox.warning(
                self._main_window,
                "Cannot Run Simulation",
                "Add at least one node and one link to the topology before running.",
            )
            return None

        node_configs = [
            NodeConfig(
                node_id=str(node["node_id"]), qubit_index=cast(int, node["qubit_index"])
            )
            for node in nodes
        ]

        # --- Resolve per-link lambda / gate width ---
        # Prefer the channel panel's value (_current_fiber_config); fall back to
        # the first scenario link's value; finally use the module-level defaults.
        _sc_link0 = self._current_scenario.links[0] if self._current_scenario.links else None
        lambda_q_nm: float = self._current_fiber_config.get(
            "lambda_q_nm",
            _sc_link0.lambda_q_nm if _sc_link0 else _DEFAULT_LAMBDA_Q_NM,
        )
        gate_width_s: float = self._current_fiber_config.get(
            "gate_width_s",
            _sc_link0.gate_width_s if _sc_link0 else _DEFAULT_GATE_WIDTH_S,
        )

        link_configs = [
            LinkConfig(
                link_id=str(link["link_id"]),
                source_node=str(link["source_node"]),
                dest_node=str(link["dest_node"]),
                lambda_q_nm=lambda_q_nm,
                gate_width_s=gate_width_s,
                qubit_index=cast(int, link["qubit_index"]),
            )
            for link in links
        ]

        # --- Resolve fiber params ---
        fiber: FiberParams | None = None
        if self._current_fiber_config:
            fiber = FiberParams(
                length_km=self._current_fiber_config.get("length_km", 25.0),
                attenuation_db_per_km=self._current_fiber_config.get(
                    "attenuation_db_per_km", 0.2
                ),
                eta_detector=self._current_fiber_config.get("eta_detector", 0.8),
                t_opt=self._current_fiber_config.get("t_opt", 0.5),
                p_dc=self._current_fiber_config.get("p_dc", 1e-5),
            )
        elif _sc_link0 is not None:
            fiber = _sc_link0.fiber.to_fiber_params()

        # --- Resolve kernel ---
        kernel = self._current_scenario.kernel.to_kernel()

        # --- Resolve sensitivity matrix ---
        sensitivity: np.ndarray | None = None
        if self._current_scenario.sensitivity is not None:
            try:
                sensitivity = validate_sensitivity_matrix(self._current_scenario.sensitivity)
            except ValueError as exc:
                QMessageBox.critical(
                    self._main_window,
                    "Invalid Sensitivity Matrix",
                    f"The sensitivity matrix is malformed and cannot be used:\n\n{exc}\n\n"
                    "Fix the values in the Telemetry panel and try again.",
                )
                return None

        # --- Resolve WDM channels ---
        # Global coexistence_channels take precedence; fall back to per-link channels.
        raw_channels: list[dict[str, object]] = (
            self._current_scenario.coexistence_channels
            or [ch for sl in self._current_scenario.links for ch in sl.classical_channels]
        )
        wdm_channels = _parse_wdm_channels(raw_channels) or None

        # STEP 5: one-time resolved-config dump so changes are visible in the log.
        _log.info(
            "Resolved physics config before build:\n"
            "  kernel       = %s\n"
            "  sensitivity  = %s\n"
            "  fiber        = %r\n"
            "  WDM channels = %d\n"
            "  lambda_q_nm  = %.1f nm, gate_width_s = %g s\n"
            "  duration_s   = %.2f  dt_s = %g  chi_max = %d",
            self._current_scenario.kernel,
            "None (SMF-28 default)" if sensitivity is None else str(sensitivity.shape),
            fiber,
            len(wdm_channels) if wdm_channels else 0,
            lambda_q_nm,
            gate_width_s,
            self._current_scenario.duration_s,
            self._current_scenario.dt_s,
            self._current_scenario.chi_max,
        )

        orchestrator = TwinOrchestrator.build_simple(
            n_qubits=len(node_configs),
            link_configs=link_configs,
            node_configs=node_configs,
            duration_s=self._current_scenario.duration_s,
            dt_s=self._current_scenario.dt_s,
            chi_max=self._current_scenario.chi_max,
            sensitivity=sensitivity,
            kernel=kernel,
            fiber=fiber,
            wdm_channels=wdm_channels,
            node_aging_overrides=self._node_aging_overrides or None,
        )

        self._dashboard.clear_all()
        for link in link_configs:
            self._dashboard.add_link(link.link_id)
        for node in node_configs:
            override = self._node_aging_overrides.get(node.node_id, {})
            t2 = override.get("t2_nominal", 1.0)
            kappa = override.get("wear_rate_kappa", 1e-4)
            self._dashboard.add_node(node.node_id, t2, kappa)

        return orchestrator

    # ------------------------------------------------------------------
    # Simulation control slots
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        if self._runner is not None and self._runner.isRunning():
            return

        orchestrator = self._build_orchestrator()
        if orchestrator is None:
            return
        self._orchestrator = orchestrator

        link_ids = [link.link_id for link in orchestrator._config.links]
        self._main_window._telemetry_viewer.update_link_list(link_ids)
        self._main_window._telemetry_viewer.set_source_label("Synthetic")

        signals = SimulationSignals()
        signals.step_completed.connect(self._on_step_completed)
        signals.node_updated.connect(self._on_node_updated)
        signals.simulation_finished.connect(self._on_simulation_finished)
        signals.simulation_error.connect(self._on_simulation_error)
        signals.clock_tick.connect(self._main_window.update_clock)
        signals.status_changed.connect(self._main_window.set_simulation_status)
        self._signals = signals

        self._runner = SimulationRunner(self._orchestrator, signals)
        self._main_window._canvas.start_animation()
        self._runner.start()

    def _on_step(self) -> None:
        if self._runner is None or not self._runner.isRunning():
            self._on_run()
        if self._runner is not None:
            self._runner.step_once()

    def _on_pause(self) -> None:
        if self._runner is None or not self._runner.isRunning():
            return
        if self._runner._paused:
            self._runner.resume()
        else:
            self._runner.pause()

    def _on_reset(self) -> None:
        if self._runner is not None:
            self._runner.stop()
            self._runner.wait()
            self._runner = None
        self._signals = None
        if self._orchestrator is not None:
            self._orchestrator.reset()
        self._dashboard.clear_all()
        self._main_window.update_clock(0.0)
        self._main_window.set_simulation_status("IDLE")
        self._main_window._canvas.stop_animation()

    # ------------------------------------------------------------------
    # Worker signal slots
    # ------------------------------------------------------------------

    def _on_step_completed(
        self,
        t: float,
        link_id: str,
        qber: float,
        fidelity: float,
        raman_rate: float,
        rhp_witness: float,
        induced_idle: float,
        skr_bps: float = 0.0,
    ) -> None:
        self._dashboard.update_qber(link_id, t, qber)
        self._dashboard.update_fidelity(link_id, t, fidelity)
        n_active = sum(
            1 for ch in self._current_scenario.coexistence_channels
            if ch.get("active", True)
        )
        self._dashboard.update_raman(link_id, t, raman_rate, n_active)
        self._dashboard.update_heatmap({link_id: fidelity})
        self._dashboard.update_key_rate(link_id, t, qber, skr_bps)
        self._main_window._canvas.update_link_fidelity(link_id, fidelity)
        self._main_window._control_plane_panel.update_link_traffic(
            link_id, min(raman_rate / 1e6, 1.0), induced_idle
        )

        if self._orchestrator is not None:
            kr_result = self._orchestrator._kr_calc.calculate(qber)
            self._main_window._channel_panel.update_key_rate_display(kr_result)

        if self._orchestrator is not None:
            cr = self._orchestrator._telemetry_engine.latest_canonical_rates(link_id)
            if cr is not None:
                self._dashboard.update_nonmarkov(
                    link_id, t, rhp_witness, cr.gamma_x, cr.gamma_y, cr.gamma_z
                )

        if self._orchestrator is not None:
            selected = self._main_window._telemetry_viewer._link_combo.currentText()
            if not selected or link_id == selected:
                resampler = self._orchestrator._telemetry_engine.resampler
                if not resampler.is_stale(link_id):
                    try:
                        sample = TelemetrySample(
                            t=t, E=resampler.at(link_id, t), link_id=link_id
                        )
                        self._main_window._telemetry_viewer.ingest_sample(sample)
                    except KeyError:
                        pass

    def _on_node_updated(
        self, node_id: str, op_count: int, t2_current: float, overrotation: float, t: float
    ) -> None:
        self._dashboard.update_aging(node_id, op_count, t2_current, overrotation, t)
        self._main_window._canvas.update_node_fidelity(node_id, min(t2_current, 1.0))
        self._main_window._aging_panel.update_node_status(node_id, op_count, t2_current)

    def _on_simulation_finished(self) -> None:
        self._main_window._canvas.stop_animation()
        self._main_window.statusBar().showMessage("Simulation complete.")

    def _on_simulation_error(self, message: str) -> None:
        self._main_window._canvas.stop_animation()
        QMessageBox.critical(self._main_window, "Simulation Error", message)

    # ------------------------------------------------------------------
    # Topology / config / scenario slots
    # ------------------------------------------------------------------

    def _on_topology_changed(self) -> None:
        if self._runner is not None and self._runner.isRunning():
            self._runner.stop()
            self._runner.wait()
            self._runner = None
            self._signals = None
        self._orchestrator = None

    def _on_node_config_changed(self, node_id: str, config: dict[str, object]) -> None:
        """Store per-node aging override from NodePropertiesDialog. Applied on next run."""
        self._node_aging_overrides[node_id] = {
            "t2_nominal": float(config.get("t2_nominal", 1.0)),  # type: ignore[arg-type]
            "wear_rate_kappa": float(config.get("wear_rate_kappa", 1e-4)),  # type: ignore[arg-type]
        }
        self._main_window.statusBar().showMessage(
            f"Aging config updated for {node_id!r} (applies on next run)", 3000
        )

    def _on_config_changed(self, engine_id: str, params: dict[str, object]) -> None:
        """Route panel config changes into ``_current_scenario`` / ``_current_fiber_config``.

        Routing rules:
        - ``"channel"``: fiber/link keys → ``_current_fiber_config``; simulation
          keys (duration_s, dt_s, chi_max) → ``_current_scenario`` via setattr.
        - ``"telemetry"``: ``kernel`` dict → ``_current_scenario.kernel``
          (validated through ``KernelModel``); ``sensitivity`` list →
          ``_current_scenario.sensitivity``.
        - ``"coexistence"``: ``channels`` list → ``_current_scenario.coexistence_channels``.
        - all others: top-level ``ScenarioConfig`` fields matched via ``hasattr``.
        """
        if engine_id == "channel":
            # Fiber / link params go to _current_fiber_config (all floats).
            for key in _FIBER_KEYS:
                if key in params:
                    try:
                        self._current_fiber_config[key] = float(params[key])  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        pass
            # Simulation params (top-level ScenarioConfig fields) go directly.
            for key in _SIM_KEYS:
                if key in params and hasattr(self._current_scenario, key):
                    try:
                        setattr(self._current_scenario, key, params[key])
                    except (TypeError, ValueError):
                        pass

        elif engine_id == "telemetry":
            kdict = params.get("kernel")
            if isinstance(kdict, dict):
                try:
                    self._current_scenario.kernel = KernelModel.model_validate(kdict)
                except Exception:
                    pass
            slist = params.get("sensitivity")
            if isinstance(slist, list):
                try:
                    self._current_scenario.sensitivity = [
                        [float(v) for v in row] for row in slist
                    ]
                except (TypeError, ValueError):
                    pass

        elif engine_id == "coexistence":
            channels = params.get("channels")
            if isinstance(channels, list):
                self._current_scenario.coexistence_channels = list(channels)

        else:
            # aging, control_plane: standard top-level field routing.
            for key, value in params.items():
                if hasattr(self._current_scenario, key):
                    try:
                        setattr(self._current_scenario, key, value)
                    except (TypeError, ValueError):
                        pass

        _log.debug("Engine %r config updated: %r", engine_id, params)

    def _on_new_scenario(self) -> None:
        self._on_reset()
        self._topology_model.clear()
        self._current_scenario = ScenarioConfig()
        self._current_fiber_config = {}
        self._dashboard.clear_all()

    # ------------------------------------------------------------------
    # Scenario persistence
    # ------------------------------------------------------------------

    def _save_scenario(self, path: str) -> None:
        """Serialise the current topology and panel configuration to ``path``."""
        aging_cfg = self._main_window._aging_panel.get_config()
        channel_cfg = self._main_window._channel_panel.get_config()
        coexistence_channels = self._main_window._coexistence_panel.get_channels()

        nodes = [
            NodeConfigModel(
                node_id=str(node["node_id"]),
                qubit_index=cast(int, node["qubit_index"]),
                t2_nominal=cast(float, aging_cfg["t2_nominal"]),
                wear_rate_kappa=cast(float, aging_cfg["wear_rate_kappa"]),
                calib_drift_rate=cast(float, aging_cfg["drift_rate_kappa"]),
            )
            for node in self._topology_model.to_scenario_nodes()
        ]
        links = [
            LinkConfigModel(
                link_id=str(link["link_id"]),
                source_node=str(link["source_node"]),
                dest_node=str(link["dest_node"]),
                lambda_q_nm=cast(float, channel_cfg["lambda_q_nm"]),
                gate_width_s=cast(float, channel_cfg["gate_width_s"]),
                qubit_index=cast(int, link["qubit_index"]),
                fiber=FiberParamsModel(
                    length_km=cast(float, channel_cfg["length_km"]),
                    attenuation_db_per_km=cast(float, channel_cfg["attenuation_db_per_km"]),
                    eta_detector=cast(float, channel_cfg["eta_detector"]),
                    t_opt=cast(float, channel_cfg["t_opt"]),
                    p_dc=cast(float, channel_cfg["p_dc"]),
                ),
                classical_channels=coexistence_channels,
            )
            for link in self._topology_model.to_scenario_links()
        ]

        scenario = ScenarioConfig(
            scenario_name=self._current_scenario.scenario_name,
            nodes=nodes,
            links=links,
            kernel=self._current_scenario.kernel,
            sensitivity=self._current_scenario.sensitivity,
            coexistence_channels=self._current_scenario.coexistence_channels,
            duration_s=self._current_scenario.duration_s,
            dt_s=self._current_scenario.dt_s,
            chi_max=self._current_scenario.chi_max,
            kappa_max=self._current_scenario.kappa_max,
        )
        scenario.to_json_file(path)
        self._current_scenario = scenario

    def _load_scenario(self, path: str) -> None:
        """Load a scenario from ``path``, rebuilding topology and panels."""
        scenario = ScenarioConfig.from_json_file(path)
        self._current_scenario = scenario

        self._topology_model.clear()
        for i, node in enumerate(scenario.nodes):
            self._topology_model.add_node(node.node_id, i * _NODE_LAYOUT_SPACING, 0.0)
        for link in scenario.links:
            self._topology_model.add_link(link.link_id, link.source_node, link.dest_node)

        if scenario.nodes:
            first_node = scenario.nodes[0]
            self._main_window._aging_panel._t2_nominal.setValue(first_node.t2_nominal)
            self._main_window._aging_panel._wear_rate_kappa.setValue(first_node.wear_rate_kappa)
            self._main_window._aging_panel._drift_rate.setValue(first_node.calib_drift_rate)
        if scenario.links:
            first_link = scenario.links[0]
            # _current_fiber_config stores only floats (chi_max excluded).
            self._current_fiber_config = {
                "lambda_q_nm": first_link.lambda_q_nm,
                "gate_width_s": first_link.gate_width_s,
                "length_km": first_link.fiber.length_km,
                "attenuation_db_per_km": first_link.fiber.attenuation_db_per_km,
                "eta_detector": first_link.fiber.eta_detector,
                "t_opt": first_link.fiber.t_opt,
                "p_dc": first_link.fiber.p_dc,
                "duration_s": scenario.duration_s,
                "dt_s": scenario.dt_s,
            }
            # Pass all values including chi_max to the panel for display.
            panel_cfg: dict[str, float | int] = {
                **self._current_fiber_config,
                "chi_max": scenario.chi_max,
            }
            self._main_window._channel_panel.load_config(panel_cfg)

        # Load kernel/sensitivity into the telemetry panel so Apply preserves them.
        telemetry_cfg: dict[str, object] = {"kernel": scenario.kernel.model_dump()}
        if scenario.sensitivity is not None:
            telemetry_cfg["sensitivity"] = scenario.sensitivity
        self._main_window._telemetry_panel.load_config(telemetry_cfg)

        # Load classical WDM channels into the coexistence panel.
        channels_to_load: list[dict[str, object]] = scenario.coexistence_channels or [
            ch for link in scenario.links for ch in link.classical_channels
        ]
        if channels_to_load:
            self._main_window._coexistence_panel.load_config(channels_to_load)

        self._dashboard.clear_all()
