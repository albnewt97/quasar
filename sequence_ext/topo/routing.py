# sequence_ext/topo/routing.py
"""
Routing utilities
=================

Provides simple routing algorithms for small- to medium-scale topologies.

Features
--------
- Shortest path search (Dijkstra) using custom edge weights.
- K-shortest paths (Yen's algorithm).
- Path cost computation using geometry + edge weights.
- Resilience analysis helpers (remove edges/nodes, recompute routes).

For larger or production-grade graphs, consider using NetworkX directly.
This module wraps common patterns with QUASAR-specific defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
import heapq

from .geometry import Node, Edge, path_length


# -----------------------------------------------------------------------------
# Graph container
# -----------------------------------------------------------------------------
@dataclass
class Graph:
    nodes: Dict[str, Node]
    edges: List[Edge]

    def neighbors(self, u: str) -> List[Tuple[str, float]]:
        """Return neighbors and edge lengths for node u."""
        out = []
        for e in self.edges:
            if e.src == u:
                out.append((e.dst, e.length if e.length is not None else 1.0))
            elif e.dst == u:
                out.append((e.src, e.length if e.length is not None else 1.0))
        return out


# -----------------------------------------------------------------------------
# Shortest path (Dijkstra)
# -----------------------------------------------------------------------------
def shortest_path(graph: Graph, src: str, dst: str) -> Tuple[float, List[str]]:
    """
    Dijkstra shortest path.

    Returns
    -------
    (cost, path)
    """
    q: List[Tuple[float, str, List[str]]] = [(0.0, src, [src])]
    visited = set()

    while q:
        cost, u, path = heapq.heappop(q)
        if u == dst:
            return cost, path
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph.neighbors(u):
            if v not in visited:
                heapq.heappush(q, (cost + w, v, path + [v]))

    raise ValueError(f"No path found {src} -> {dst}")


# -----------------------------------------------------------------------------
# K-shortest paths (Yen's algorithm, simple version)
# -----------------------------------------------------------------------------
def k_shortest_paths(graph: Graph, src: str, dst: str, k: int = 3) -> List[Tuple[float, List[str]]]:
    """
    Return up to k-shortest paths using a simplified Yen's algorithm.
    """
    cost, path = shortest_path(graph, src, dst)
    A = [(cost, path)]
    B: List[Tuple[float, List[str]]] = []

    for _ in range(1, k):
        for i in range(len(path) - 1):
            spur_node = path[i]
            root_path = path[: i + 1]

            # Remove edges from graph that would create duplicate root_path
            removed = []
            for c, p in A:
                if len(p) > i and p[: i + 1] == root_path:
                    u, v = p[i], p[i + 1]
                    for e in graph.edges:
                        if (e.src == u and e.dst == v) or (e.src == v and e.dst == u):
                            removed.append(e)
                            graph.edges.remove(e)

            try:
                spur_cost, spur_path = shortest_path(graph, spur_node, dst)
                total_path = root_path[:-1] + spur_path
                total_cost = sum(
                    e.length if e.length is not None else 1.0
                    for e in graph.edges
                    if (e.src, e.dst) in zip(total_path, total_path[1:])
                    or (e.dst, e.src) in zip(total_path, total_path[1:])
                )
                B.append((total_cost, total_path))
            except ValueError:
                pass
            finally:
                graph.edges.extend(removed)

        if not B:
            break
        B.sort(key=lambda x: x[0])
        A.append(B.pop(0))
        cost, path = A[-1]

    return A


__all__ = ["Graph", "shortest_path", "k_shortest_paths"]
