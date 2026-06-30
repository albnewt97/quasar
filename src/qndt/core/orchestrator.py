"""TwinOrchestrator: the master event kernel (§4.1, §3.3).

Owns no physics and no quantum state.  Wires telemetry, coexistence, aging,
and control-plane engines together through ChannelComposer, then drives
TensorStateTracker through the resulting effective PTM at each simulation
step.  This is the single place the §4.1 data-flow pipeline is assembled:

    induced_idle (control_plane)
        -> OpContext
        -> ChannelComposer.effective_ptm(ctx)  [telemetry, coexistence, aging]
        -> tracker.apply_channel(...)
        -> aging_model.register_op(...)
        -> raman_rate / rhp_witness / qber
        -> SimulationEvent published on the NoiseBus
        -> SimulationResult
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from qndt.control_plane.async_plane import AsynchronousControlPlane, JitterModel
from qndt.control_plane.load import WDMLoadTracker
from qndt.control_plane.routing import NetworkGraph
from qndt.core.bus import NoiseBus, SimulationEvent
from qndt.core.composer import ChannelComposer
from qndt.core.context import OpContext
from qndt.physics.aging import DeviceAgingModel
from qndt.physics.kernels import ExponentialKernel, MemoryKernel
from qndt.physics.key_rate import BB84KeyRateCalculator, KeyRateParams
from qndt.physics.raman import (
    ClassicalChannelSpec,
    CoexistenceNoiseEngine,
    FiberParams,
    RamanProfile,
)
from qndt.quantum.tracker import TensorStateTracker
from qndt.telemetry.calibration import S_SMF28_DEFAULT
from qndt.telemetry.engine import EnvironmentalTelemetryEngine
from qndt.telemetry.sources import SyntheticTelemetrySource, TelemetrySample

_log = logging.getLogger(__name__)

# §5.2: Pauli rates are squashed into [0, 0.5); QBER is clamped to the same bound.
_QBER_MAX: float = 0.5
# Tolerance for the CP-floor clamp (absorbs IEEE 754 cancellation near boundary).
_CP_CLAMP_TOL: float = 1e-9
# Speed of light (vacuum) and SMF-28 group index at 1550 nm.
# Used to compute the default per-qubit propagation delay τ_qubit = L·n_g/c.
# Source: Saleh & Teich, "Fundamentals of Photonics", §8.2.
_C_LIGHT: float = 2.998e8   # m/s
_N_GROUP: float = 1.468     # SMF-28 group refractive index at 1550 nm


class SimulationStepError(RuntimeError):
    """Raised by :meth:`TwinOrchestrator.step` when a PTM cannot be scaled safely.

    Caught by :class:`SimulationRunner` and surfaced as a ``QMessageBox.critical``
    dialog via the ``simulation_error`` signal path — same as sensitivity
    validation errors.
    """


def _scale_ptm_eigenvalues(eigenvalues_1s: np.ndarray, dt_s: float) -> np.ndarray:
    """Scale 1-second PTM diagonal eigenvalues to ``dt_s`` seconds.

    Primary formula — generator (semigroup) scaling::

        λ_i(dt) = exp(−γ_i · dt)   where   γ_i = −ln(λ_i)

    This satisfies ``(λ_i(dt))^(1/dt) = λ_i`` exactly, making it
    dt-invariant whenever all implied Lindblad rates are non-negative
    (i.e., the channel belongs to a valid Markovian semigroup).

    **Guard**: raises :class:`SimulationStepError` if any ``λ_i ≤ 0``.
    A zero or negative eigenvalue means the decoherence has saturated on
    that axis so that ``ln(λ_i)`` is undefined.  The caller must reduce
    ``dt_s`` or lower the sensitivity matrix.

    **CP safety**: after generator scaling the Pauli rates are clamped to
    ``[0, 0.25]``.  Environmental-convolution channels are non-Markovian;
    their implied Lindblad rates can be slightly negative (most common on
    the *y*-axis when ``px ≈ py`` but ``pz`` is large).  Clamping is the
    physical floor — a fully-decohered axis cannot decohere further.
    A ``WARNING`` is logged on every affected step; this is not silent.

    Args:
        eigenvalues_1s: Raw PTM eigenvalues ``[λx, λy, λz]`` for a 1-second step.
        dt_s: Target step size in seconds.

    Returns:
        Scaled eigenvalues ``[λx(dt), λy(dt), λz(dt)]``.

    Raises:
        SimulationStepError: If any eigenvalue ≤ 0 (decoherence saturation).
    """
    nonpos = [(ax, float(v)) for ax, v in zip("xyz", eigenvalues_1s) if v <= 0.0]
    if nonpos:
        raise SimulationStepError(
            f"PTM eigenvalue(s) ≤ 0: {nonpos} — dt is too coarse for these "
            "decoherence rates.  Reduce dt_s or lower the sensitivity matrix."
        )

    # Generator scaling: λ_i(dt) = λ_i^dt  (exp(−γ_i·dt) with γ_i = −ln(λ_i))
    gamma = -np.log(eigenvalues_1s)
    scaled: np.ndarray = np.exp(-gamma * dt_s)

    # CP safety: clamp negative Pauli rates to the [0, 0.25] physical floor.
    # Non-Markovian channels (e.g. 15× sensitivity) have a negative implied
    # Γ_Y, producing p_y < 0 after generator scaling — the channel lies
    # outside the Lindblad-semigroup polytope and must be projected back.
    lx, ly, lz = scaled
    px = (1.0 + lx - ly - lz) / 4.0
    py = (1.0 - lx + ly - lz) / 4.0
    pz = (1.0 - lx - ly + lz) / 4.0
    if px < 0.0 or py < 0.0 or pz < 0.0:
        # Warn only when the deviation is large enough to be a real physical issue
        # (e.g. genuinely non-Markovian channels), not for near-zero IEEE 754 noise.
        if px < -_CP_CLAMP_TOL or py < -_CP_CLAMP_TOL or pz < -_CP_CLAMP_TOL:
            _log.warning(
                "Generator-scaled PTM outside CP polytope at dt=%.3fs "
                "(px=%.2e py=%.2e pz=%.2e) — clamping to [0, 0.25].  "
                "Implied Lindblad rate(s) negative; channel is non-Markovian.  "
                "Reduce sensitivity or dt for a dt-invariant regime.",
                dt_s, px, py, pz,
            )
        px = min(max(px, 0.0), 0.25)
        py = min(max(py, 0.0), 0.25)
        pz = min(max(pz, 0.0), 0.25)
        scaled = np.array([
            1.0 - 2.0 * (py + pz),
            1.0 - 2.0 * (px + pz),
            1.0 - 2.0 * (px + py),
        ], dtype=np.float64)
    return scaled


@dataclass(frozen=True, slots=True)
class LinkConfig:
    """Static configuration of a single quantum fiber link.

    Args:
        link_id: Unique link identifier.
        source_node: Node id at the transmitting end (memory side).
        dest_node: Node id at the receiving end.
        lambda_q_nm: Quantum channel wavelength in nm.
        gate_width_s: Quantum gate duration in seconds.
        qubit_index: Index of the qubit in TensorStateTracker representing
            this link's quantum state.
        qubit_exposure_s: Physical per-qubit exposure time τ_qubit [s] used
            to scale the effective PTM for QBER estimation (§5.7).  Must be
            > 0.  For fly-by / point-to-point links set to the fiber
            propagation delay L·n_g/c; for quantum-memory / repeater nodes
            override with the memory hold time so that aging strongly drives
            QBER.  Default: propagation delay of a 25 km SMF-28 span.
    """

    link_id: str
    source_node: str
    dest_node: str
    lambda_q_nm: float
    gate_width_s: float
    qubit_index: int
    qubit_exposure_s: float = 25e3 * _N_GROUP / _C_LIGHT


@dataclass(frozen=True, slots=True)
class NodeConfig:
    """Static configuration of a quantum memory node.

    Args:
        node_id: Unique node identifier.
        qubit_index: Index of the qubit in TensorStateTracker at this node.
    """

    node_id: str
    qubit_index: int


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Full static configuration for a TwinOrchestrator run.

    Args:
        links: All fiber links in the topology.
        nodes: All quantum memory nodes in the topology.
        duration_s: Total simulated duration in seconds.
        dt_s: Simulation time step in seconds.
    """

    links: tuple[LinkConfig, ...]
    nodes: tuple[NodeConfig, ...]
    duration_s: float
    dt_s: float = 0.1


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Immutable record of one link's state at one simulation step.

    Args:
        t: Simulation time of this result [s].
        link_id: Fiber link this result belongs to.
        qber: Estimated quantum bit error rate, clamped to [0.0, 0.5].
        fidelity: State quality metric returned by tracker.apply_channel
            (global purity after the channel was applied).
        raman_rate_hz: Total spontaneous Raman photon rate at this time [Hz].
        rhp_witness: Current accumulated RHP non-Markovianity witness value.
        induced_idle_s: Classical-signalling-induced memory hold time [s].
        ptm_eigenvalues: Copy of the effective PTM's [λx, λy, λz] eigenvalues.
        secret_key_rate_bps: BB84 secret key rate [bits/sec] from GLLP formula.
        key_rate_positive: ``True`` if ``secret_key_rate_bps > 0`` (link is secure).
        security_margin: ``qber_threshold − qber``; positive means operating safely
            below the security cutoff.
        qber_threshold: QBER value at which the secret key rate reaches zero for
            the current ``KeyRateParams``.

    Notes:
        ``ptm_eigenvalues`` is stored as a fresh copy via ``object.__setattr__``
        in ``__post_init__`` because ``frozen=True`` prevents normal assignment.
    """

    t: float
    link_id: str
    qber: float
    fidelity: float
    raman_rate_hz: float
    rhp_witness: float
    induced_idle_s: float
    ptm_eigenvalues: np.ndarray
    secret_key_rate_bps: float = 0.0
    key_rate_positive: bool = False
    security_margin: float = 0.0
    qber_threshold: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ptm_eigenvalues", np.array(self.ptm_eigenvalues, dtype=np.float64)
        )


@dataclass(frozen=True, slots=True)
class WDMScheduleEntry:
    """A single timed WDM channel event applied by TwinOrchestrator during step().

    Applied at the start of the step whose simulation time equals or exceeds
    ``t`` (i.e., when ``self._t >= t``), before the link physics loop runs.
    Events are applied in ascending ``t`` order.

    Args:
        t: Simulation time [s] at which this event fires.
        link_id: Fiber link to apply the event on.
        action: One of ``"activate"``, ``"deactivate"``, or ``"update_power"``.
        channel_id: Classical channel identifier.
        lambda_c_nm: Centre wavelength [nm]; required for ``"activate"``.
        launch_power_mw: Launch power [mW]; required for ``"activate"`` and
            ``"update_power"``.  Ignored for ``"deactivate"``.
    """

    t: float
    link_id: str
    action: str
    channel_id: str
    lambda_c_nm: float = 0.0
    launch_power_mw: float = 0.0


class TwinOrchestrator:
    """Master event kernel wiring all physics engines to the quantum tracker.

    Holds no physics and no quantum state of its own (§3.3, §3.4).  Each call
    to ``step()`` advances every configured link through exactly the §4.1
    pipeline once, then advances the simulation clock.

    Args:
        config: Static simulation configuration (links, nodes, timing).
        telemetry_engine: Non-Markovian environmental noise engine.
        coexistence_engine: Raman/WDM co-existence noise engine.
        aging_model: Device aging / idle-dephasing noise engine.
        control_plane: Classical control plane (induced idle, WDM load).
        tracker: Sole owner of quantum state.
        bus: Event bus for publishing simulation step events.  A fresh
            ``NoiseBus()`` is created if ``None``.
        key_rate_params: BB84 key rate calculator parameters.  Defaults to
            ``KeyRateParams()`` (standard WCP BB84 with f_ec=1.16).
        wdm_schedule: Optional timed WDM channel events applied in-order during
            ``step()`` (§3.3 B1).  Sorted by ``t`` at construction.
    """

    def __init__(
        self,
        config: SimulationConfig,
        telemetry_engine: EnvironmentalTelemetryEngine,
        coexistence_engine: CoexistenceNoiseEngine,
        aging_model: DeviceAgingModel,
        control_plane: AsynchronousControlPlane,
        tracker: TensorStateTracker,
        bus: NoiseBus | None = None,
        key_rate_params: KeyRateParams | None = None,
        wdm_schedule: list[WDMScheduleEntry] | None = None,
    ) -> None:
        self._config = config
        self._telemetry_engine = telemetry_engine
        self._coexistence_engine = coexistence_engine
        self._aging_model = aging_model
        self._control_plane = control_plane
        self._tracker = tracker
        self._bus: NoiseBus = bus if bus is not None else NoiseBus()

        self._composer = ChannelComposer()
        self._composer.register(telemetry_engine)
        self._composer.register(coexistence_engine)
        self._composer.register(aging_model)

        self._kr_calc = BB84KeyRateCalculator(key_rate_params or KeyRateParams())

        self._results: list[SimulationResult] = []
        self._t: float = 0.0
        # Populated by build_simple(); stepped in step() to keep telemetry fresh.
        self._live_sources: dict[str, Iterator[TelemetrySample]] = {}

        # WDM schedule: sorted by t, processed in-order during step().
        self._wdm_schedule: list[WDMScheduleEntry] = sorted(
            wdm_schedule or [], key=lambda ev: ev.t
        )
        self._schedule_idx: int = 0

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def step(self, dt: float | None = None) -> list[SimulationResult]:
        """Advance every configured link through one simulation step.

        Args:
            dt: Time increment for this step [s]; defaults to ``config.dt_s``.

        Returns:
            The ``SimulationResult`` entries produced by this step only.
        """
        step_results: list[SimulationResult] = []
        dt_s = dt if dt is not None else self._config.dt_s

        # Apply WDM schedule events due at the current time (§3.3 B1).
        # Events fire when event.t <= self._t, before the link physics loop,
        # so Raman/QBER reflect the new WDM state within the same step.
        while self._schedule_idx < len(self._wdm_schedule):
            ev = self._wdm_schedule[self._schedule_idx]
            if ev.t > self._t:
                break
            self._apply_wdm_event(ev)
            self._schedule_idx += 1

        # Ingest one live telemetry sample per link so the convolution window
        # stays current throughout the run and is_stale() never fires.
        for _link in self._config.links:
            _src = self._live_sources.get(_link.link_id)
            if _src is not None:
                _live_sample = next(_src, None)
                if _live_sample is not None:
                    self._telemetry_engine.ingest(
                        TelemetrySample(
                            t=self._t, E=_live_sample.E, link_id=_link.link_id
                        )
                    )

        for link in self._config.links:
            induced_idle = self._control_plane.induced_idle(link.source_node, self._t)

            ctx = OpContext(
                link_id=link.link_id,
                node_id=link.source_node,
                t=self._t,
                lambda_q=link.lambda_q_nm * 1e-9,
                gate_width=link.gate_width_s,
                idle_time=induced_idle,
            )
            effective_ptm = self._composer.effective_ptm(ctx)

            # Generator (semigroup) scaling: λ_i(dt) = λ_i^dt = exp(−γ_i·dt).
            # Guard: raises SimulationStepError if any eigenvalue ≤ 0.
            # CP safety: clamps Pauli rates for non-Markovian channels.
            # See _scale_ptm_eigenvalues for the full rationale.
            scaled_eigenvalues = _scale_ptm_eigenvalues(effective_ptm[1:], dt_s)
            scaled_ptm = np.concatenate([[effective_ptm[0]], scaled_eigenvalues])

            fidelity = self._tracker.apply_channel(
                link.qubit_index,
                scaled_ptm,
                t=self._t,
                link_id=link.link_id,
                node_id=link.source_node,
            )
            # D = ∫u dt (whitepaper §6, Eq 18): u≈1 while the memory is
            # in-service, so each step contributes dt_s of wear time.
            self._aging_model.register_op(
                link.source_node, "channel", self._t, op_duration_s=dt_s
            )

            raman_rate = self._coexistence_engine.raman_rate(
                link.link_id, link.lambda_q_nm, self._t
            )
            rhp = self._telemetry_engine.rhp_value(link.link_id)

            # Exact per-qubit BB84 QBER for a Pauli channel (Nielsen & Chuang
            # §12 / GLLP).  For each basis:
            #   QBER_Z = (1 − λz) / 2   (X or Y errors flip Z-basis bits)
            #   QBER_X = (1 − λx) / 2   (Z or Y errors flip X-basis bits)
            #   QBER   = (QBER_Z + QBER_X) / 2 = (2 − λx − λz) / 4
            # λy does not enter QBER directly.
            #
            # QBER is scaled to τ_qubit (link.qubit_exposure_s), NOT to dt_s.
            # For fly-by links τ_qubit = propagation delay L·n_g/c; repeater
            # memory nodes override with the hold time.  dt-scaled eigenvalues
            # (scaled_ptm) continue to drive state evolution and aging only.
            # `fidelity` (= Tr(ρ²) from the tracker) is kept as a separate
            # diagnostic for the fidelity plot; it does NOT drive QBER here.
            _qber_eigs = _scale_ptm_eigenvalues(effective_ptm[1:], link.qubit_exposure_s)
            lx_q, lz_q = float(_qber_eigs[0]), float(_qber_eigs[2])
            qber = min(max(0.0, (2.0 - lx_q - lz_q) / 4.0), _QBER_MAX)

            kr = self._kr_calc.calculate(qber)

            self._bus.publish(
                SimulationEvent(
                    kind="simulation_step",
                    t=self._t,
                    source_id=link.link_id,
                    payload={"qber": qber, "fidelity": fidelity},
                )
            )

            result = SimulationResult(
                t=self._t,
                link_id=link.link_id,
                qber=qber,
                fidelity=fidelity,
                raman_rate_hz=raman_rate,
                rhp_witness=rhp,
                induced_idle_s=induced_idle,
                ptm_eigenvalues=scaled_ptm[1:],
                secret_key_rate_bps=kr.secret_key_rate_bps,
                key_rate_positive=kr.is_positive,
                security_margin=kr.security_margin,
                qber_threshold=kr.qber_threshold,
            )
            self._results.append(result)
            step_results.append(result)

        self._t += dt if dt is not None else self._config.dt_s
        return step_results

    def run(self, steps: int | None = None) -> list[SimulationResult]:
        """Run the simulation for a number of steps.

        Args:
            steps: Number of steps to run; defaults to
                ``round(config.duration_s / config.dt_s)``.

        Returns:
            All accumulated results since the last ``reset()``.
        """
        n_steps = (
            steps
            if steps is not None
            else int(round(self._config.duration_s / self._config.dt_s))
        )
        for _ in range(n_steps):
            self.step()
        return self.results()

    def _apply_wdm_event(self, ev: WDMScheduleEntry) -> None:
        """Dispatch a single WDM schedule event to the control plane.

        Args:
            ev: The schedule event to apply.
        """
        if ev.action == "activate":
            spec = ClassicalChannelSpec(
                channel_id=ev.channel_id,
                lambda_c_nm=ev.lambda_c_nm,
                launch_power_mw=ev.launch_power_mw,
            )
            self._control_plane.activate_channel(ev.link_id, spec)
        elif ev.action == "deactivate":
            self._control_plane.deactivate_channel(ev.link_id, ev.channel_id)
        elif ev.action == "update_power":
            self._control_plane.update_channel_power(
                ev.link_id, ev.channel_id, ev.launch_power_mw
            )
        else:
            _log.warning(
                "Unknown WDM schedule action %r for link %r at t=%.3fs; skipped.",
                ev.action, ev.link_id, ev.t,
            )

    def reset(self) -> None:
        """Reset the simulation clock, result log, and quantum state."""
        self._t = 0.0
        self._results.clear()
        self._tracker.reset()
        self._schedule_idx = 0

    def reconfigure(self, config: SimulationConfig) -> None:
        """Replace the static simulation configuration without resetting state.

        Args:
            config: New ``SimulationConfig`` to use for subsequent steps.
        """
        self._config = config

    def current_t(self) -> float:
        """Return the current simulation time [s]."""
        return self._t

    # ------------------------------------------------------------------
    # Result access
    # ------------------------------------------------------------------

    def results(self) -> list[SimulationResult]:
        """Return a copy of all accumulated results."""
        return list(self._results)

    def results_for_link(self, link_id: str) -> list[SimulationResult]:
        """Return accumulated results filtered to a single link.

        Args:
            link_id: Fiber link identifier.
        """
        return [r for r in self._results if r.link_id == link_id]

    def qber_timeseries(self, link_id: str) -> list[tuple[float, float]]:
        """Return ``(t, qber)`` pairs for a single link in chronological order."""
        return [(r.t, r.qber) for r in self.results_for_link(link_id)]

    def fidelity_timeseries(self, link_id: str) -> list[tuple[float, float]]:
        """Return ``(t, fidelity)`` pairs for a single link in chronological order."""
        return [(r.t, r.fidelity) for r in self.results_for_link(link_id)]

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def build_simple(
        cls,
        n_qubits: int,
        link_configs: list[LinkConfig],
        node_configs: list[NodeConfig],
        duration_s: float = 10.0,
        dt_s: float = 0.1,
        chi_max: int = 4,
        sensitivity: np.ndarray | None = None,
        kernel: MemoryKernel | None = None,
        fiber: FiberParams | None = None,
        key_rate_params: KeyRateParams | None = None,
        node_aging_overrides: dict[str, dict[str, float]] | None = None,
        wdm_channels: list[ClassicalChannelSpec] | None = None,
        wdm_schedule: list[WDMScheduleEntry] | None = None,
    ) -> TwinOrchestrator:
        """Build a fully-wired TwinOrchestrator with sensible defaults.

        Constructs a synthetic telemetry source, default kernel/fiber/Raman
        profile, an empty WDM load tracker, a network graph matching the
        supplied configs, and a fresh TensorStateTracker.  Pre-ingests 60s
        of synthetic telemetry per link so the convolution window is warm
        before the first ``step()``.

        Args:
            n_qubits: Number of qubits to allocate in TensorStateTracker.
            link_configs: Fiber links to wire into the topology.
            node_configs: Quantum memory nodes to wire into the topology.
            duration_s: Default simulation duration [s].
            dt_s: Default simulation time step [s].
            chi_max: Maximum MPDO bond dimension for the tracker.
            sensitivity: Telemetry sensitivity matrix S, shape (3, M).
                Defaults to ``S_SMF28_DEFAULT`` (§5.3 illustrative SMF-28 defaults — uncalibrated).
            kernel: Memory kernel for the telemetry engine.  Defaults to
                ``ExponentialKernel(tau_x=30, tau_y=30, tau_z=120)``.
            fiber: Fiber physical parameters.  Defaults to a 25 km SMF-28 span.
            key_rate_params: BB84 key rate calculator parameters.  Defaults to
                ``KeyRateParams()`` (standard WCP BB84 with f_ec=1.16).
            node_aging_overrides: Per-node aging parameter overrides.
            wdm_channels: Classical WDM channels to register with the
                ``CoexistenceNoiseEngine`` at construction time.  Each entry is
                passed to ``coexistence_engine.register_channel()``.  ``None``
                leaves the engine with no classical channels (zero Raman noise).
            wdm_schedule: Timed WDM channel events applied during ``step()``
                (§3.3 B1).  Passed through to :class:`TwinOrchestrator`.

        Returns:
            A ready-to-step ``TwinOrchestrator``.
        """
        if sensitivity is None:
            # §5.3: px/py have zero thermal coupling (only seismic/wind drive
            # bit-flip errors); an isotropic eye(3) placeholder would wrongly
            # couple temperature into px/py and overstate bit-flip noise.
            sensitivity = S_SMF28_DEFAULT
        if kernel is None:
            kernel = ExponentialKernel(tau_x=30.0, tau_y=30.0, tau_z=120.0)
        if fiber is None:
            fiber = FiberParams(
                length_km=25.0,
                attenuation_db_per_km=0.2,
                eta_detector=0.8,
                t_opt=0.5,
                p_dc=1e-5,
            )

        telemetry_engine = EnvironmentalTelemetryEngine(
            sensitivity=sensitivity, kernel=kernel
        )

        graph = NetworkGraph()
        seen_nodes: set[str] = set()
        for node in node_configs:
            graph.add_node(node.node_id)
            seen_nodes.add(node.node_id)
        for link in link_configs:
            if link.source_node not in seen_nodes:
                graph.add_node(link.source_node)
                seen_nodes.add(link.source_node)
            if link.dest_node not in seen_nodes:
                graph.add_node(link.dest_node)
                seen_nodes.add(link.dest_node)
        for link in link_configs:
            graph.add_link(link.link_id, link.source_node, link.dest_node)

        load_tracker = WDMLoadTracker()
        control_plane = AsynchronousControlPlane(
            graph=graph, load_tracker=load_tracker, jitter_model=JitterModel()
        )

        coexistence_engine = CoexistenceNoiseEngine(
            profile=RamanProfile.smf28_default(),
            fiber=fiber,
            control_plane=control_plane,
        )
        if wdm_channels:
            for _spec in wdm_channels:
                coexistence_engine.register_channel(_spec)

        aging_model = DeviceAgingModel(
            t2_nominal=1.0, wear_rate_kappa=1e-4, calib_drift_rate=1e-6
        )
        if node_aging_overrides:
            for _node_id, _params in node_aging_overrides.items():
                aging_model.set_node_params(_node_id, **_params)

        tracker = TensorStateTracker(n_sites=n_qubits, chi_max=chi_max)

        config = SimulationConfig(
            links=tuple(link_configs),
            nodes=tuple(node_configs),
            duration_s=duration_s,
            dt_s=dt_s,
        )

        orchestrator = cls(
            config=config,
            telemetry_engine=telemetry_engine,
            coexistence_engine=coexistence_engine,
            aging_model=aging_model,
            control_plane=control_plane,
            tracker=tracker,
            key_rate_params=key_rate_params,
            wdm_schedule=wdm_schedule,
        )

        # Pre-warm the convolution window with 60s of *history before t=0*,
        # not samples dated [0, 60). SyntheticTelemetrySource always starts
        # at t=0, so its raw output is shifted back by -60s here; otherwise
        # the causal window() (t_i <= query_t) would contain only the single
        # t=0 sample at the first step() call, making telemetry_engine
        # contribute zero noise on step 1 despite the "warm" pre-ingestion.
        _PREWARM_S = 60.0
        for link in link_configs:
            source = SyntheticTelemetrySource(
                link_id=link.link_id, duration_s=_PREWARM_S, dt_s=1.0
            )
            for sample in source:
                telemetry_engine.ingest(
                    TelemetrySample(
                        t=sample.t - _PREWARM_S, E=sample.E, link_id=sample.link_id
                    )
                )

        # Create per-link live sources so step() can keep the telemetry window
        # fresh throughout the run.  Seed 43 gives independent samples from the
        # prewarm (seed 42); duration margin avoids running out before the sim ends.
        live_srcs: dict[str, Iterator[TelemetrySample]] = {}
        for link in link_configs:
            live_src = SyntheticTelemetrySource(
                link_id=link.link_id,
                duration_s=duration_s + _PREWARM_S + 1.0,
                dt_s=dt_s,
                seed=43,
            )
            live_srcs[link.link_id] = iter(live_src)
        orchestrator._live_sources = live_srcs

        return orchestrator
