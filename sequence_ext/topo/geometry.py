# sequence_ext/topo/geometry.py
"""
Topology geometry utilities
===========================

Lightweight geometric primitives and helpers for network/topology modeling.

Features
--------
- Node with Cartesian coordinates (meters) or geographic lat/lon (degrees).
- Edge with length cache and optional attributes.
- Distance utilities (Euclidean, haversine).
- Path length computation with safe checks.

Design principles
-----------------
- No hidden globals; all functions are pure.
- Typed dataclasses; slots for memory efficiency.
- Resilient to bad inputs (clear exceptions).

Note
----
For large-scale graphs, prefer using NetworkX for routing, then feed paths here
to compute accurate lengths with either Cartesian or geographic metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import math


# -----------------------------------------------------------------------------
# Primitives
# -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Node:
    """
    Network node.

    Parameters
    ----------
    name : str
        Unique identifier.
    xy_m : Tuple[float, float] | None
        Cartesian coordinates in meters (x, y). Use for city-scale fiber maps.
    latlon_deg : Tuple[float, float] | None
        Geographic coordinates in degrees (lat, lon). Use for country-scale maps.

    Exactly one of `xy_m` or `latlon_deg` should be provided.
    """

    name: str
    xy_m: Optional[Tuple[float, float]] = None
    latlon_deg: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        if (self.xy_m is None) == (self.latlon_deg is None):
            raise ValueError("Provide exactly one of xy_m or latlon_deg")

    def is_geo(self) -> bool:
        return self.latlon_deg is not None


@dataclass(frozen=True, slots=True)
class Edge:
    """Topology edge linking two nodes by name."""
    src: str
    dst: str
    # Optional cached length in meters or kilometers (depends on metric used).
    length: Optional[float] = None
    # Free-form attributes (attenuation overrides, fiber type, etc.) can be attached externally.


# -----------------------------------------------------------------------------
# Distance utilities
# -----------------------------------------------------------------------------
def euclidean_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance in meters between two (x, y)."""
    dx, dy = a[0] - b[0], a[1] - b[1]
    return math.hypot(dx, dy)


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance in kilometers between two (lat, lon) in degrees."""
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    s = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(s))


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
def path_length(
    nodes_by_name: dict[str, Node],
    path: Iterable[str],
    *,
    metric: str | None = None,
) -> float:
    """
    Compute total path length.

    Parameters
    ----------
    nodes_by_name : dict[str, Node]
        Mapping from node name to Node.
    path : Iterable[str]
        Ordered list of node names.
    metric : {"euclidean_m", "haversine_km"} | None
        If None, inferred from node coordinate type.

    Returns
    -------
    float
        Total length (meters for euclidean_m, kilometers for haversine_km).
    """
    seq: List[str] = list(path)
    if len(seq) < 2:
        return 0.0

    # Infer metric if not provided
    if metric is None:
        first = nodes_by_name[seq[0]]
        metric = "haversine_km" if first.is_geo() else "euclidean_m"

    total = 0.0
    for i in range(len(seq) - 1):
        a = nodes_by_name.get(seq[i])
        b = nodes_by_name.get(seq[i + 1])
        if a is None or b is None:
            raise KeyError(f"Unknown node in path at segment {seq[i]} -> {seq[i+1]}")

        if metric == "euclidean_m":
            if a.xy_m is None or b.xy_m is None:
                raise ValueError("All nodes in path must have xy_m for euclidean_m")
            total += euclidean_m(a.xy_m, b.xy_m)
        elif metric == "haversine_km":
            if a.latlon_deg is None or b.latlon_deg is None:
                raise ValueError("All nodes in path must have latlon_deg for haversine_km")
            total += haversine_km(a.latlon_deg, b.latlon_deg)
        else:
            raise ValueError("metric must be 'euclidean_m' or 'haversine_km'")

    return total


__all__ = ["Node", "Edge", "euclidean_m", "haversine_km", "path_length"]
