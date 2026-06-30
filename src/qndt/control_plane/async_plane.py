"""Asynchronous control plane: classical packet routing with latency and congestion.

The class is named 'Asynchronous' because it operates on its own clock,
independently of the quantum simulation clock (§3.3).  Implementation is
fully synchronous Python — no asyncio, no anyio, no threads.

This is the classical→quantum coupling point (docs/architecture.md §4.1):
- induced_idle() feeds DeviceAgingModel with memory hold times
- current_load() feeds CoexistenceNoiseEngine with WDM traffic levels
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qndt.control_plane.load import ClassicalLoad, WDMLoadTracker
from qndt.control_plane.routing import (
    LoopDetector,
    NetworkGraph,
    RouteNotFoundError,
    RoutingLoop,
)


@dataclass(frozen=True, slots=True)
class PacketResult:
    """Record of a single packet routing attempt.

    Args:
        packet_id: Caller-supplied unique identifier for this packet.
        source: Origin node.
        dest: Destination node.
        route: Node path taken (empty if delivery failed before routing).
        t: Simulation time when the packet was submitted [s].
        latency_s: Total end-to-end latency [s].
        delivered: ``True`` if the packet reached its destination.
        drop_reason: Human-readable drop cause, or ``None`` if delivered.
    """

    packet_id: str
    source: str
    dest: str
    route: list[str]
    t: float
    latency_s: float
    delivered: bool
    drop_reason: str | None


@dataclass(frozen=True, slots=True)
class JitterModel:
    """Per-hop latency model with Gaussian jitter and congestion scaling.

    A seeded :class:`numpy.random.Generator` is created in ``__post_init__``
    so that results are reproducible given the same ``seed``.

    Args:
        base_latency_s: Per-hop base latency at zero utilisation [s].
        jitter_std_s: Standard deviation of Gaussian jitter [s].
        congestion_factor: Latency multiplier at 100 % utilisation.
        seed: RNG seed.
    """

    base_latency_s: float = 1e-3
    jitter_std_s: float = 1e-4
    congestion_factor: float = 2.0
    seed: int = 42
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        # Replace the default-factory RNG with a seeded one.
        # object.__setattr__ bypasses the frozen guard (same trick as TelemetrySample).
        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))

    def sample_hop_latency(self, utilisation: float) -> float:
        """Sample a single hop latency given the current link utilisation.

        ``mean = base_latency_s · (1 + congestion_factor · utilisation)``

        Args:
            utilisation: WDM link utilisation fraction in ``[0, 1]``.

        Returns:
            Non-negative latency sample in seconds.
        """
        mean = self.base_latency_s * (1.0 + self.congestion_factor * utilisation)
        sample = float(self._rng.normal(mean, self.jitter_std_s))
        return max(0.0, sample)


class AsynchronousControlPlane:
    """Classical network simulation layer — the classical→quantum coupling point.

    Routes packets over :class:`NetworkGraph`, accumulates congestion history,
    and exposes ``induced_idle()`` / ``current_load()`` to the quantum engines
    (see docs/architecture.md §3.3 and §4.1).

    All methods are synchronous.  The 'Asynchronous' in the name refers to
    this layer operating on a clock independent of the quantum simulation.

    Args:
        graph: Network topology.
        load_tracker: Live WDM occupancy tracker.
        jitter_model: Latency + jitter model.  Defaults to ``JitterModel()``.
    """

    def __init__(
        self,
        graph: NetworkGraph,
        load_tracker: WDMLoadTracker,
        jitter_model: JitterModel | None = None,
    ) -> None:
        self._graph = graph
        self._load_tracker = load_tracker
        self._jitter_model: JitterModel = (
            jitter_model if jitter_model is not None else JitterModel()
        )
        self._packet_log: list[PacketResult] = []
        self._induced_idle: dict[str, float] = {}
        self._congestion_history: dict[str, list[tuple[float, float]]] = {}

    def route_packet(
        self,
        packet_id: str,
        source: str,
        dest: str,
        t: float,
    ) -> PacketResult:
        """Route a packet from ``source`` to ``dest`` and record the result.

        1. Computes the shortest path via Dijkstra.
        2. Checks for routing loops (defensive; Dijkstra paths are loop-free).
        3. Samples per-hop latency from the jitter model using current WDM load.
        4. Returns a :class:`PacketResult` and appends it to ``_packet_log``.

        Args:
            packet_id: Caller-supplied packet identifier.
            source: Origin node.
            dest: Destination node.
            t: Current simulation time [s].

        Returns:
            ``PacketResult`` with ``delivered=True`` on success, or
            ``delivered=False`` with a ``drop_reason`` on failure.
        """
        try:
            node_path = self._graph.shortest_path(source, dest)
            LoopDetector().check(node_path)
        except RouteNotFoundError:
            result = PacketResult(
                packet_id=packet_id,
                source=source,
                dest=dest,
                route=[],
                t=t,
                latency_s=0.0,
                delivered=False,
                drop_reason="no_route",
            )
            self._packet_log.append(result)
            return result
        except RoutingLoop as exc:
            result = PacketResult(
                packet_id=packet_id,
                source=source,
                dest=dest,
                route=exc.path,
                t=t,
                latency_s=0.0,
                delivered=False,
                drop_reason="routing_loop",
            )
            self._packet_log.append(result)
            return result

        total_latency = 0.0
        for i in range(len(node_path) - 1):
            n1, n2 = node_path[i], node_path[i + 1]
            link_id = self._graph.link_between(n1, n2) or ""
            load = self._load_tracker.current_load(link_id, t)
            total_latency += self._jitter_model.sample_hop_latency(load.utilisation)

        result = PacketResult(
            packet_id=packet_id,
            source=source,
            dest=dest,
            route=node_path,
            t=t,
            latency_s=total_latency,
            delivered=True,
            drop_reason=None,
        )
        self._packet_log.append(result)
        return result

    def current_load(self, link_id: str, t: float) -> ClassicalLoad:
        """Return WDM load for a link and record the utilisation for history.

        Args:
            link_id: Fiber link identifier.
            t: Simulation time [s].

        Returns:
            ``ClassicalLoad`` snapshot from the ``WDMLoadTracker``.
        """
        load = self._load_tracker.current_load(link_id, t)
        self._congestion_history.setdefault(link_id, []).append(
            (t, load.utilisation)
        )
        return load

    def induced_idle(self, node_id: str, t: float) -> float:  # noqa: ARG002
        """Classical signaling latency imposed on ``node_id`` as a hold time.

        This is the key classical→quantum coupling point (§4.1): high
        congestion means packets take longer, forcing quantum memories to
        hold entanglement longer while waiting for classical acknowledgements.

        Algorithm:
          Collect all PacketResults where ``node_id`` appears in ``result.route``.
          Average the latency of the most recent 10 such packets.
          Clamp to ``[0.0, 1.0]`` s.

        Args:
            node_id: Quantum memory node identifier.
            t: Simulation time (reserved for future time-windowed filtering).

        Returns:
            Induced idle time in seconds; ``0.0`` if no packets recorded.
        """
        relevant = [r for r in self._packet_log if node_id in r.route]
        if not relevant:
            return 0.0
        last_n = relevant[-10:]
        avg = sum(r.latency_s for r in last_n) / len(last_n)
        return min(max(avg, 0.0), 1.0)

    def jitter(self, node_id: str, t: float) -> float:  # noqa: ARG002
        """Standard deviation of recent per-packet latencies through ``node_id``.

        Args:
            node_id: Node identifier.
            t: Simulation time (reserved for future time-windowed filtering).

        Returns:
            Latency std deviation [s]; ``0.0`` if fewer than 2 packets.
        """
        relevant = [r for r in self._packet_log if node_id in r.route]
        if len(relevant) < 2:
            return 0.0
        latencies = [r.latency_s for r in relevant[-10:]]
        return float(np.std(latencies))

    def manages_link(self, link_id: str) -> bool:
        """Return True if ``link_id`` is managed by the WDM load tracker.

        Delegates to :meth:`WDMLoadTracker.manages_link`.  Used by
        :class:`~qndt.physics.raman.CoexistenceNoiseEngine` to distinguish
        "CP-managed, all channels off → Raman 0" from "unmanaged → static dict".

        Args:
            link_id: Fiber link identifier.

        Returns:
            ``True`` if the link was ever activated; ``False`` otherwise.
        """
        return self._load_tracker.manages_link(link_id)

    def activate_channel(self, link_id: str, spec: Any) -> None:
        """Activate a classical WDM channel on a link via the load tracker.

        ``spec`` is a ``ClassicalChannelSpec`` (from ``qndt.physics.raman``).
        Typed as ``Any`` here to avoid an ``AsynchronousControlPlane →
        CoexistenceNoiseEngine`` import dependency (§3.3 Law 3).

        Args:
            link_id: Fiber link identifier.
            spec: ``ClassicalChannelSpec`` channel specification.
        """
        self._load_tracker.activate(link_id, spec)

    def deactivate_channel(self, link_id: str, channel_id: str) -> None:
        """Deactivate a classical WDM channel on a link via the load tracker.

        Args:
            link_id: Fiber link identifier.
            channel_id: Channel to remove.
        """
        self._load_tracker.deactivate(link_id, channel_id)

    def update_channel_power(
        self, link_id: str, channel_id: str, power_mw: float
    ) -> None:
        """Update launch power for an active channel via the load tracker.

        Args:
            link_id: Fiber link identifier.
            channel_id: Channel to update.
            power_mw: New launch power in mW.

        Raises:
            KeyError: If the channel is not active on the link.
        """
        self._load_tracker.update_power(link_id, channel_id, power_mw)

    def congestion_timeseries(self, link_id: str) -> list[tuple[float, float]]:
        """Return the full congestion history for a link.

        Args:
            link_id: Fiber link identifier.

        Returns:
            List of ``(t, utilisation)`` tuples in recording order.
        """
        return list(self._congestion_history.get(link_id, []))

    def packet_log(self) -> list[PacketResult]:
        """Return a copy of the packet log.

        Returns:
            Shallow copy of ``_packet_log``.
        """
        return list(self._packet_log)

    def clear_log(self) -> None:
        """Clear packet log, induced idle cache, and congestion history."""
        self._packet_log.clear()
        self._induced_idle.clear()
        self._congestion_history.clear()

    @staticmethod
    def _nan_safe_std(values: list[float]) -> float:
        """Return std dev, or 0.0 for lists with < 2 elements."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)
