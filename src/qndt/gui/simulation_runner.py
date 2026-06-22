"""SimulationRunner: drives TwinOrchestrator in a background QThread (§4.2).

The simulation runs in a QThread; the GUI runs in the Qt main thread.
Cross-thread communication happens exclusively via ``SimulationSignals`` --
no shared mutable state is read or written across the thread boundary.
``SimulationSignals`` is a standalone ``QObject`` (not mixed into the
``QThread`` subclass) specifically so it can be created and connected to
in the main thread *before* the worker thread is started.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal

from qndt.core.orchestrator import TwinOrchestrator

_PAUSE_POLL_MS = 50
_STEP_POLL_MS = 10


class SimulationSignals(QObject):
    """Cross-thread signals emitted by ``SimulationRunner``.

    Signals:
        step_completed: ``(t, link_id, qber, fidelity, raman_rate,
            rhp_witness, induced_idle, skr_bps)`` for one link's step result.
        node_updated: ``(node_id, op_count, t2_current, overrotation, t)``
            for one node's aging state.
        simulation_finished: Emitted once the run reaches ``duration_s``.
        simulation_error: Emitted with an error message if ``step()`` raises.
        clock_tick: Emitted with the current simulation time after each step.
        status_changed: Emitted with one of "RUNNING"/"PAUSED"/"IDLE"/"ERROR".
    """

    step_completed = Signal(float, str, float, float, float, float, float, float)
    node_updated = Signal(str, int, float, float, float)
    simulation_finished = Signal()
    simulation_error = Signal(str)
    clock_tick = Signal(float)
    status_changed = Signal(str)


class SimulationRunner(QThread):
    """Runs a ``TwinOrchestrator`` to completion on a background thread."""

    def __init__(
        self,
        orchestrator: TwinOrchestrator,
        signals: SimulationSignals,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._orchestrator = orchestrator
        self._signals = signals
        self._running = False
        self._paused = False
        self._step_mode = False

    def run(self) -> None:
        """QThread entry point: step the orchestrator until duration_s."""
        self._running = True
        self._signals.status_changed.emit("RUNNING")

        try:
            while self._running:
                if self._paused:
                    self.msleep(_PAUSE_POLL_MS)
                    continue

                results = self._orchestrator.step()
                for result in results:
                    self._signals.step_completed.emit(
                        result.t,
                        result.link_id,
                        result.qber,
                        result.fidelity,
                        result.raman_rate_hz,
                        result.rhp_witness,
                        result.induced_idle_s,
                        result.secret_key_rate_bps,
                    )
                    self._signals.clock_tick.emit(result.t)

                for node_cfg in self._orchestrator._config.nodes:
                    t = self._orchestrator.current_t()
                    op_count = self._orchestrator._aging_model.op_count(node_cfg.node_id)
                    t2_current = self._orchestrator._aging_model.coherence_time(
                        node_cfg.node_id, t
                    )
                    overrotation = self._orchestrator._aging_model.gate_overrotation(
                        node_cfg.node_id, t
                    )
                    self._signals.node_updated.emit(
                        node_cfg.node_id, op_count, t2_current, overrotation, t
                    )

                if self._step_mode:
                    self._paused = True
                    self._step_mode = False

                if self._orchestrator.current_t() >= self._orchestrator._config.duration_s:
                    self._running = False

                self.msleep(_STEP_POLL_MS)
        except Exception as exc:
            self._signals.simulation_error.emit(str(exc))
            self._signals.status_changed.emit("ERROR")
            return

        self._signals.simulation_finished.emit()
        self._signals.status_changed.emit("IDLE")

    def pause(self) -> None:
        """Pause the run loop without stopping the thread."""
        self._paused = True
        self._signals.status_changed.emit("PAUSED")

    def resume(self) -> None:
        """Resume a paused run loop."""
        self._paused = False
        self._signals.status_changed.emit("RUNNING")

    def step_once(self) -> None:
        """Unpause for exactly one step, then re-pause automatically."""
        self._step_mode = True
        self._paused = False

    def stop(self) -> None:
        """Stop the run loop; the thread exits at the top of its next check."""
        self._running = False
        self._paused = False
