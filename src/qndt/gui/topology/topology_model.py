"""TopologyModel: Qt bridge between NetworkGraph and the topology canvas (§7.2).

The model owns no graph data of its own beyond what ``NetworkGraph`` already
holds plus the Qt-only concern of node screen position.  ``NetworkGraph``
(``qndt.control_plane.routing``) remains the authoritative topology model;
this class only renders signals on top of it for the Qt canvas to consume
(§3.6 GUI Isolation Law — Qt types are confined to ``qndt.gui``).
"""
from __future__ import annotations

from typing import cast

from PySide6.QtCore import QObject, Signal

from qndt.control_plane.routing import NetworkGraph


class TopologyModel(QObject):
    """Qt-signal wrapper around a ``NetworkGraph`` plus node screen positions.

    Signals:
        node_added: Emitted with the new node's id.
        node_removed: Emitted with the removed node's id.
        link_added: Emitted with the new link's id.
        link_removed: Emitted with the removed link's id.
        topology_changed: Emitted after any add/remove/position change.
    """

    node_added = Signal(str)
    node_removed = Signal(str)
    link_added = Signal(str)
    link_removed = Signal(str)
    topology_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._graph = NetworkGraph()
        self._node_positions: dict[str, tuple[float, float]] = {}

    def add_node(
        self,
        node_id: str,
        x: float,
        y: float,
        node_type: str = "memory_node",
        **attrs: object,
    ) -> None:
        """Register a node in the graph at the given scene position.

        Args:
            node_id: Unique node identifier.
            x: Scene x-coordinate.
            y: Scene y-coordinate.
            node_type: Node role (memory_node, bsm_node, source_node, detector).
            **attrs: Arbitrary additional node metadata.
        """
        self._graph.add_node(node_id, type=node_type, **attrs)
        self._node_positions[node_id] = (x, y)
        self.node_added.emit(node_id)
        self.topology_changed.emit()

    def remove_node(self, node_id: str) -> None:
        """Remove a node (and its incident links) from the graph.

        Args:
            node_id: Node to remove.
        """
        self._graph.remove_node(node_id)
        self._node_positions.pop(node_id, None)
        self.node_removed.emit(node_id)
        self.topology_changed.emit()

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
            source: Source node id.
            dest: Destination node id.
            weight: Routing cost.
            **attrs: Arbitrary additional link metadata.
        """
        self._graph.add_link(link_id, source, dest, weight=weight, **attrs)
        self.link_added.emit(link_id)
        self.topology_changed.emit()

    def remove_link(self, link_id: str) -> None:
        """Remove a link from the graph.

        Args:
            link_id: Link to remove.
        """
        self._graph.remove_link(link_id)
        self.link_removed.emit(link_id)
        self.topology_changed.emit()

    def clear(self) -> None:
        """Remove every link and node, leaving an empty topology.

        Implemented via ``remove_link``/``remove_node`` (rather than
        resetting ``_graph`` directly) so ``link_removed``/``node_removed``
        fire for each entry -- keeping any connected canvas in sync.
        """
        for link_id in list(self.link_ids()):
            self.remove_link(link_id)
        for node_id in list(self.node_ids()):
            self.remove_node(node_id)

    def node_position(self, node_id: str) -> tuple[float, float]:
        """Return the stored scene position of a node.

        Args:
            node_id: Node to look up.

        Raises:
            KeyError: If ``node_id`` has no stored position.
        """
        return self._node_positions[node_id]

    def set_node_position(self, node_id: str, x: float, y: float) -> None:
        """Update a node's stored scene position.

        Args:
            node_id: Node to update.
            x: New scene x-coordinate.
            y: New scene y-coordinate.
        """
        self._node_positions[node_id] = (x, y)
        self.topology_changed.emit()

    def graph(self) -> NetworkGraph:
        """Return the internal ``NetworkGraph`` (read-only reference)."""
        return self._graph

    def node_ids(self) -> list[str]:
        """Return all node identifiers."""
        return self._graph.nodes()

    def link_ids(self) -> list[str]:
        """Return all link identifiers."""
        return self._graph.links()

    def link_data(self, link_id: str) -> dict[str, object]:
        """Return the raw attribute dict for a link.

        Args:
            link_id: Link to look up.
        """
        return cast(dict[str, object], self._graph._links[link_id])  # noqa: SLF001

    def to_scenario_nodes(self) -> list[dict[str, object]]:
        """Return node dicts suitable for ``ScenarioConfig`` construction."""
        return [
            {
                "node_id": node_id,
                "qubit_index": i,
                "x": self._node_positions[node_id][0],
                "y": self._node_positions[node_id][1],
            }
            for i, node_id in enumerate(self.node_ids())
        ]

    def to_scenario_links(self) -> list[dict[str, object]]:
        """Return link dicts suitable for ``ScenarioConfig`` construction."""
        nodes = self.node_ids()
        result: list[dict[str, object]] = []
        for link_id in self.link_ids():
            data = self.link_data(link_id)
            source = str(data["source"])
            result.append(
                {
                    "link_id": link_id,
                    "source_node": source,
                    "dest_node": str(data["dest"]),
                    "qubit_index": nodes.index(source),
                }
            )
        return result
