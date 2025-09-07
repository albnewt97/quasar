# sequence_ext/topo/__init__.py
"""
Topology utilities
==================

Geometry and routing helpers for network modeling in QUASAR.

Modules
-------
- geometry.py : Node/Edge primitives, Euclidean and haversine distances, path length.
- routing.py  : Simple routing algorithms (shortest path, k-shortest paths).

Public API
----------
- Node, Edge
- euclidean_m, haversine_km, path_length
- Graph, shortest_path, k_shortest_paths
"""

from .geometry import Node, Edge, euclidean_m, haversine_km, path_length
from .routing import Graph, shortest_path, k_shortest_paths

__all__ = [
    "Node",
    "Edge",
    "euclidean_m",
    "haversine_km",
    "path_length",
    "Graph",
    "shortest_path",
    "k_shortest_paths",
]
