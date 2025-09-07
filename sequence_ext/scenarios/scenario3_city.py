# sequence_ext/scenarios/scenario3_city.py
"""
Scenario 3 – Urban city network
===============================

Models an MDI-QKD deployment across a metropolitan area.

Features
--------
- Multiple nodes with fiber interconnects (mesh/star topology).
- Routing and utilization metrics.
- Baseline for resilience analysis (link failures, rerouting).

Purpose
-------
Demonstrate performance of MDI-QKD over realistic city-scale fiber networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import networkx as nx

from ..orchestrator.orchestrator import Orchestrator
from ..io.logging import logger
from ..io.metrics import MetricFrame


@dataclass
class Scenario3City:
    pulse_rate_hz: int
    duration_s: float
    output_dir: str

    def run(self) -> Path:
        logger.info("Running Scenario 3: city fiber network")

        # Build a simple city topology (toy example)
        G = nx.Graph()
        nodes = ["A", "B", "C", "D", "E"]
        G.add_nodes_from(nodes)
        edges = [("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"), ("B", "E")]
        G.add_edges_from(edges)

        # Select a path for A–E communication
        path = nx.shortest_path(G, "A", "E")
        logger.info("Selected routing path: {}", path)

        # Orchestrator run
        orch = Orchestrator(self.pulse_rate_hz, self.duration_s, Path(self.output_dir))
        mf: MetricFrame = orch.run()

        # Annotate network table with chosen path
        mf.network["path"] = "->".join(path)

        return orch.write(mf, fmt="parquet")
