"""Integration tests for the classical control plane layer.

Covers NetworkGraph, LoopDetector, WDMLoadTracker, JitterModel, and
AsynchronousControlPlane including the classical→quantum coupling points
induced_idle() and current_load().
"""
from __future__ import annotations

import numpy as np
import pytest

from qndt.control_plane.async_plane import (
    AsynchronousControlPlane,
    JitterModel,
)
from qndt.control_plane.load import WDMLoadTracker
from qndt.control_plane.routing import (
    LoopDetector,
    NetworkGraph,
    RouteNotFoundError,
    RoutingLoop,
)
from qndt.physics.raman import ClassicalChannelSpec

# ---------------------------------------------------------------------------
# NetworkGraph tests
# ---------------------------------------------------------------------------


def test_network_graph_add_remove_nodes() -> None:
    """add_node / remove_node maintain nodes() list correctly."""
    g = NetworkGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_node("C")
    assert set(g.nodes()) == {"A", "B", "C"}
    g.remove_node("B")
    assert set(g.nodes()) == {"A", "C"}
    assert "B" not in g.nodes()


def test_network_graph_add_link_invalid_node() -> None:
    """add_link with an unknown source or dest raises ValueError."""
    g = NetworkGraph()
    g.add_node("A")
    with pytest.raises(ValueError, match="B"):
        g.add_link("l1", "A", "B")


def test_dijkstra_simple() -> None:
    """shortest_path on a 3-node linear graph returns the only path."""
    g = NetworkGraph()
    for n in ("A", "B", "C"):
        g.add_node(n)
    g.add_link("l1", "A", "B")
    g.add_link("l2", "B", "C")
    assert g.shortest_path("A", "C") == ["A", "B", "C"]


def test_dijkstra_no_route() -> None:
    """shortest_path raises RouteNotFoundError for disconnected nodes."""
    g = NetworkGraph()
    g.add_node("A")
    g.add_node("B")
    with pytest.raises(RouteNotFoundError) as exc_info:
        g.shortest_path("A", "B")
    assert "A" in str(exc_info.value)
    assert "B" in str(exc_info.value)


def test_dijkstra_prefers_low_weight() -> None:
    """Dijkstra returns the minimum-weight path, not the shortest hop count."""
    g = NetworkGraph()
    for n in ("A", "B", "C"):
        g.add_node(n)
    g.add_link("direct", "A", "B", weight=10.0)
    g.add_link("ac", "A", "C", weight=0.1)
    g.add_link("cb", "C", "B", weight=0.1)
    path = g.shortest_path("A", "B")
    # Weight via A→C→B = 0.2 < 10.0 via A→B direct
    assert path == ["A", "C", "B"]


# ---------------------------------------------------------------------------
# LoopDetector tests
# ---------------------------------------------------------------------------


def test_loop_detector_clean() -> None:
    """A path with no repeated nodes does not raise."""
    LoopDetector().check(["A", "B", "C"])  # must not raise


def test_loop_detector_detects() -> None:
    """A path with a repeated node raises RoutingLoop."""
    with pytest.raises(RoutingLoop) as exc_info:
        LoopDetector().check(["A", "B", "A", "C"])
    assert "A" in str(exc_info.value)


# ---------------------------------------------------------------------------
# WDMLoadTracker tests
# ---------------------------------------------------------------------------


def test_wdm_load_tracker_empty() -> None:
    """current_load on an unknown link returns zero power and zero utilisation."""
    tracker = WDMLoadTracker()
    load = tracker.current_load("nonexistent_link", 0.0)
    assert load.total_power_mw == pytest.approx(0.0)
    assert load.utilisation == pytest.approx(0.0)
    assert load.active_channels == []


def test_wdm_load_tracker_activate() -> None:
    """activate() causes current_load to reflect the channel and its power."""
    tracker = WDMLoadTracker()
    spec = ClassicalChannelSpec(channel_id="ch1", lambda_c_nm=1310.0, launch_power_mw=2.0)
    tracker.activate("link1", spec)
    load = tracker.current_load("link1", 1.0)
    assert len(load.active_channels) == 1
    assert load.total_power_mw == pytest.approx(2.0)
    assert load.active_channels[0].channel_id == "ch1"


def test_wdm_load_tracker_utilisation() -> None:
    """8 channels on a 80-capacity link gives utilisation = 0.1."""
    tracker = WDMLoadTracker(capacity_channels=80)
    for i in range(8):
        spec = ClassicalChannelSpec(
            channel_id=f"ch{i}",
            lambda_c_nm=1310.0 + i,
            launch_power_mw=1.0,
        )
        tracker.activate("link1", spec)
    load = tracker.current_load("link1", 0.0)
    assert load.utilisation == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# JitterModel tests
# ---------------------------------------------------------------------------


def test_jitter_model_positive() -> None:
    """sample_hop_latency always returns a non-negative value."""
    jm = JitterModel(seed=7)
    for _ in range(200):
        assert jm.sample_hop_latency(0.5) >= 0.0


def test_jitter_model_congestion() -> None:
    """High utilisation produces higher mean latency than low utilisation."""
    jm_high = JitterModel(seed=0)
    jm_low = JitterModel(seed=0)

    samples_high = [jm_high.sample_hop_latency(0.99) for _ in range(200)]
    samples_low = [jm_low.sample_hop_latency(0.01) for _ in range(200)]

    assert np.mean(samples_high) > np.mean(samples_low)


# ---------------------------------------------------------------------------
# AsynchronousControlPlane tests
# ---------------------------------------------------------------------------


def _make_cp(
    nodes: list[str] | None = None,
    links: list[tuple[str, str, str]] | None = None,
) -> tuple[AsynchronousControlPlane, NetworkGraph, WDMLoadTracker]:
    """Helper: build a control plane with a small topology."""
    g = NetworkGraph()
    for n in (nodes or ["A", "B", "C"]):
        g.add_node(n)
    for lid, src, dst in (links or [("l_ab", "A", "B"), ("l_bc", "B", "C")]):
        g.add_link(lid, src, dst)
    tracker = WDMLoadTracker()
    cp = AsynchronousControlPlane(graph=g, load_tracker=tracker)
    return cp, g, tracker


def test_route_packet_delivered() -> None:
    """route_packet on a connected graph returns delivered=True."""
    cp, _, _ = _make_cp()
    result = cp.route_packet("pkt1", "A", "C", t=0.0)
    assert result.delivered is True
    assert result.drop_reason is None
    assert result.route == ["A", "B", "C"]


def test_route_packet_no_route() -> None:
    """route_packet on a disconnected graph returns delivered=False, drop_reason='no_route'."""
    g = NetworkGraph()
    g.add_node("X")
    g.add_node("Y")
    # No link between X and Y
    tracker = WDMLoadTracker()
    cp = AsynchronousControlPlane(graph=g, load_tracker=tracker)
    result = cp.route_packet("p1", "X", "Y", t=0.0)
    assert result.delivered is False
    assert result.drop_reason == "no_route"
    assert result.latency_s == pytest.approx(0.0)


def test_induced_idle_zero_no_packets() -> None:
    """induced_idle returns 0.0 before any packets have been routed."""
    cp, _, _ = _make_cp()
    assert cp.induced_idle("A", 0.0) == pytest.approx(0.0)


def test_induced_idle_nonzero_after_routing() -> None:
    """After routing packets through a node, induced_idle returns a positive value."""
    cp, _, _ = _make_cp()
    for i in range(5):
        cp.route_packet(f"p{i}", "A", "C", t=float(i))
    idle = cp.induced_idle("A", 5.0)
    # JitterModel default: base_latency_s=1e-3 per hop, 2 hops A→B→C → ~2ms
    assert idle > 0.0


def test_congestion_history_recorded() -> None:
    """current_load() records (t, utilisation) entries in congestion_timeseries."""
    cp, _, _ = _make_cp()
    cp.current_load("l_ab", 1.0)
    cp.current_load("l_ab", 2.0)
    history = cp.congestion_timeseries("l_ab")
    assert len(history) == 2
    assert history[0][0] == pytest.approx(1.0)
    assert history[1][0] == pytest.approx(2.0)
    # No channels active → utilisation = 0
    assert history[0][1] == pytest.approx(0.0)
