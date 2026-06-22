"""Network topology graph and routing utilities for the classical control plane.

Pure data structures — no Qt, no asyncio, no anyio.  Provides Dijkstra
shortest-path routing over the quantum network topology.
"""
from __future__ import annotations

import heapq
from typing import Any


class RouteNotFoundError(Exception):
    """Raised when Dijkstra finds no path between two nodes.

    Args:
        source: Source node identifier.
        destination: Destination node identifier.
    """

    def __init__(self, source: str, destination: str) -> None:
        self.source = source
        self.destination = destination
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"No route from {self.source} to {self.destination}"


class RoutingLoop(Exception):
    """Raised when a proposed path contains a repeated node.

    Args:
        path: The path in which the loop was detected.
    """

    def __init__(self, path: list[str]) -> None:
        self.path = path
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"Routing loop detected: {' -> '.join(self.path)}"


class NetworkGraph:
    """Undirected weighted graph of quantum network nodes and fiber links.

    This is a pure data structure — no Qt, no async, no physics.  It is the
    authoritative model of the network topology; Qt rendering items must never
    duplicate this state (§3.3).

    Internal state:
        _nodes:  node_id → attribute dict (arbitrary key/value pairs).
        _links:  link_id → attribute dict; always contains 'source', 'dest', 'weight'.
        _adj:    node → neighbour → link_id  (both directions for undirected links).
    """

    def __init__(self) -> None:
        # Any used: attribute bags for arbitrary caller-supplied metadata.
        self._nodes: dict[str, Any] = {}
        self._links: dict[str, Any] = {}
        self._adj: dict[str, dict[str, str]] = {}

    def add_node(self, node_id: str, **attrs: object) -> None:
        """Register a node in the graph.

        Args:
            node_id: Unique node identifier.
            **attrs: Arbitrary metadata (type, position, T2 nominal, …).
        """
        self._nodes[node_id] = dict(attrs)
        self._adj.setdefault(node_id, {})

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all of its incident links.

        Args:
            node_id: Node to remove.  No-op if not present.
        """
        incident = [
            lid
            for lid, data in self._links.items()
            if data["source"] == node_id or data["dest"] == node_id
        ]
        for lid in incident:
            self.remove_link(lid)
        self._nodes.pop(node_id, None)
        self._adj.pop(node_id, None)

    def add_link(
        self,
        link_id: str,
        source: str,
        dest: str,
        weight: float = 1.0,
        **attrs: object,
    ) -> None:
        """Add an undirected link between two existing nodes.

        Args:
            link_id: Unique link identifier.
            source: Source node identifier.
            dest: Destination node identifier.
            weight: Routing cost (lower = preferred by Dijkstra).
            **attrs: Arbitrary metadata (fiber length, wavelength, …).

        Raises:
            ValueError: If ``source`` or ``dest`` is not in the graph.
        """
        if source not in self._nodes:
            raise ValueError(f"Source node {source!r} not in graph")
        if dest not in self._nodes:
            raise ValueError(f"Destination node {dest!r} not in graph")
        self._links[link_id] = {
            "source": source, "dest": dest, "weight": weight, **attrs
        }
        self._adj.setdefault(source, {})[dest] = link_id
        self._adj.setdefault(dest, {})[source] = link_id

    def remove_link(self, link_id: str) -> None:
        """Remove a link from the graph.

        Args:
            link_id: Link to remove.  No-op if not present.
        """
        if link_id not in self._links:
            return
        data = self._links.pop(link_id)
        src: str = data["source"]
        dst: str = data["dest"]
        self._adj.get(src, {}).pop(dst, None)
        self._adj.get(dst, {}).pop(src, None)

    def nodes(self) -> list[str]:
        """Return all node identifiers in insertion order.

        Returns:
            List of node IDs.
        """
        return list(self._nodes.keys())

    def links(self) -> list[str]:
        """Return all link identifiers in insertion order.

        Returns:
            List of link IDs.
        """
        return list(self._links.keys())

    def link_between(self, n1: str, n2: str) -> str | None:
        """Return the link ID connecting n1 and n2, or None.

        Args:
            n1: First node.
            n2: Second node.

        Returns:
            Link identifier or ``None``.
        """
        return self._adj.get(n1, {}).get(n2)

    def shortest_path(self, source: str, dest: str) -> list[str]:
        """Dijkstra's shortest path between two nodes.

        Args:
            source: Start node.
            dest: End node.

        Returns:
            List of node IDs from ``source`` to ``dest`` inclusive.

        Raises:
            ValueError: If ``source`` or ``dest`` is not in the graph.
            RouteNotFoundError: If no path exists.
        """
        if source not in self._nodes:
            raise ValueError(f"Source node {source!r} not in graph")
        if dest not in self._nodes:
            raise ValueError(f"Destination node {dest!r} not in graph")

        dist: dict[str, float] = {source: 0.0}
        prev: dict[str, str] = {}
        heap: list[tuple[float, str]] = [(0.0, source)]
        visited: set[str] = set()

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == dest:
                break
            for v, lid in self._adj.get(u, {}).items():
                if v in visited:
                    continue
                w: float = float(self._links[lid]["weight"])
                new_d = d + w
                if new_d < dist.get(v, float("inf")):
                    dist[v] = new_d
                    prev[v] = u
                    heapq.heappush(heap, (new_d, v))

        if dest not in dist:
            raise RouteNotFoundError(source=source, destination=dest)

        path: list[str] = []
        node = dest
        while node in prev:
            path.append(node)
            node = prev[node]
        path.append(source)
        path.reverse()
        return path

    def link_path(self, node_path: list[str]) -> list[str]:
        """Convert a node path to the list of traversed link IDs.

        Args:
            node_path: Ordered list of node IDs.

        Returns:
            List of link IDs in traversal order.

        Raises:
            ValueError: If any consecutive pair has no connecting link.
        """
        result: list[str] = []
        for i in range(len(node_path) - 1):
            lid = self.link_between(node_path[i], node_path[i + 1])
            if lid is None:
                raise ValueError(
                    f"No link between {node_path[i]!r} and {node_path[i + 1]!r}"
                )
            result.append(lid)
        return result


class LoopDetector:
    """Stateless detector for routing loops in a proposed node path."""

    def check(self, path: list[str]) -> None:
        """Raise RoutingLoop if any node appears more than once in path.

        Args:
            path: Proposed routing path as a list of node IDs.

        Raises:
            RoutingLoop: If a node is repeated.
        """
        seen: set[str] = set()
        for node in path:
            if node in seen:
                raise RoutingLoop(path=path)
            seen.add(node)
