"""TopologyCanvas: the main network editor canvas (§7.2).

Renders ``TopologyModel`` (which wraps ``NetworkGraph``).  The canvas never
owns graph data, only the visual representation of it (§3.6 GUI Isolation
Law — Qt types are confined to ``qndt.gui``).
"""
from __future__ import annotations

from uuid import uuid4

import networkx as nx
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QMenu,
    QWidget,
)

from qndt.gui.topology.link_item import FiberLinkItem
from qndt.gui.topology.node_item import QuantumNodeItem
from qndt.gui.topology.topology_model import TopologyModel

_GRID_SPACING: float = 30.0
_GRID_DOT_COLOUR = QColor("#21262D")
_GRID_DOT_RADIUS: float = 1.0

_MIN_SCALE: float = 0.2
_MAX_SCALE: float = 5.0


class TopologyCanvas(QGraphicsView):
    """Interactive network editor: drag nodes, draw links, animate state.

    Signals:
        node_selected: Emitted with the node id when a node is selected.
        link_selected: Emitted with the link id when a link is selected.
        node_double_clicked: Emitted with the node id for property dialogs.
        topology_changed: Forwarded from the underlying ``TopologyModel``.
        add_node_requested: Emitted with scene ``(x, y)`` on empty-space
            double-click, requesting that a new node be created there.

    Args:
        model: The ``TopologyModel`` to render.
        parent: Optional parent widget.
        read_only: When ``True``, disables mouse editing (node/link
            creation, context menus, link-draw mode); the animation
            timer still runs. Used to embed the canvas read-only inside
            ``NetworkHeatmap``.
    """

    node_selected = Signal(str)
    link_selected = Signal(str)
    node_double_clicked = Signal(str)
    node_config_changed = Signal(str, dict)
    topology_changed = Signal()
    add_node_requested = Signal(float, float)

    def __init__(
        self,
        model: TopologyModel,
        parent: QWidget | None = None,
        read_only: bool = False,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._read_only = read_only
        self._node_items: dict[str, QuantumNodeItem] = {}
        self._link_items: dict[str, FiberLinkItem] = {}
        self._drawing_link: bool = False
        self._link_source: str | None = None
        self._pending_node_type: str | None = None
        self._node_counter: int = 0
        self._mouse_scene_pos: QPointF | None = None
        self._simulating: bool = False
        self._scale_factor: float = 1.0

        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QBrush(QColor("#0D1117")))
        self.setScene(self._scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate)
        self._anim_timer.start(50)

        self.setAcceptDrops(True)
        # Allow FiberLinkItem delete × button to call back into the model.
        self._scene._delete_link_requested = self._model.remove_link  # type: ignore[attr-defined]

        self._model.node_added.connect(self._on_node_added)
        self._model.node_removed.connect(self._on_node_removed)
        self._model.link_added.connect(self._on_link_added)
        self._model.link_removed.connect(self._on_link_removed)
        self._model.topology_changed.connect(self.topology_changed.emit)

    def drawBackground(self, painter: QPainter, rect: QRectF | QRect) -> None:
        super().drawBackground(painter, rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_GRID_DOT_COLOUR)

        left = int(rect.left()) - (int(rect.left()) % int(_GRID_SPACING))
        top = int(rect.top()) - (int(rect.top()) % int(_GRID_SPACING))

        x = float(left)
        while x < rect.right():
            y = float(top)
            while y < rect.bottom():
                painter.drawEllipse(
                    QPointF(x, y), _GRID_DOT_RADIUS, _GRID_DOT_RADIUS
                )
                y += _GRID_SPACING
            x += _GRID_SPACING

    # ------------------------------------------------------------------
    # Model signal handlers
    # ------------------------------------------------------------------

    def _on_node_added(self, node_id: str) -> None:
        graph_attrs = self._model.graph()._nodes.get(node_id, {})  # noqa
        node_type = str(graph_attrs.get("type", "memory_node"))
        item = QuantumNodeItem(node_id, node_type)
        item._config_changed_callback = self._on_node_config_from_item
        x, y = self._model.node_position(node_id)
        item.setPos(x, y)
        self._scene.addItem(item)
        self._node_items[node_id] = item

    def _on_node_removed(self, node_id: str) -> None:
        item = self._node_items.pop(node_id, None)
        if item is not None:
            self._scene.removeItem(item)
        stale_links = [
            link_id
            for link_id, link_item in self._link_items.items()
            if link_item.source_item.node_id() == node_id
            or link_item.dest_item.node_id() == node_id
        ]
        for link_id in stale_links:
            link_item = self._link_items.pop(link_id)
            self._scene.removeItem(link_item)

    def _on_link_added(self, link_id: str) -> None:
        data = self._model.link_data(link_id)
        source_item = self._node_items.get(str(data["source"]))
        dest_item = self._node_items.get(str(data["dest"]))
        if source_item is None or dest_item is None:
            return
        link_item = FiberLinkItem(link_id, source_item, dest_item)
        self._scene.addItem(link_item)
        self._link_items[link_id] = link_item

    def _on_link_removed(self, link_id: str) -> None:
        item = self._link_items.pop(link_id, None)
        if item is not None:
            self._scene.removeItem(item)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _animate(self) -> None:
        if not self._simulating:
            return
        for link_item in self._link_items.values():
            link_item.advance(0.05)

    def start_animation(self) -> None:
        """Begin animating entanglement particles on every link."""
        self._simulating = True

    def stop_animation(self) -> None:
        """Stop animating entanglement particles."""
        self._simulating = False

    # ------------------------------------------------------------------
    # Live state updates
    # ------------------------------------------------------------------

    def update_node_fidelity(self, node_id: str, fidelity: float) -> None:
        """Push a live fidelity reading to a node's visual item.

        Args:
            node_id: Node to update.
            fidelity: New fidelity value in ``[0, 1]``.
        """
        item = self._node_items.get(node_id)
        if item is not None:
            item.update_fidelity(fidelity)

    def update_link_fidelity(self, link_id: str, fidelity: float) -> None:
        """Push a live fidelity reading to a link's visual item.

        Args:
            link_id: Link to update.
            fidelity: New fidelity value in ``[0, 1]``.
        """
        item = self._link_items.get(link_id)
        if item is not None:
            item.update_fidelity(fidelity)

    def update_link_raman(self, link_id: str, active: bool) -> None:
        """Toggle a link's classical-traffic (Raman coexistence) indicator.

        Args:
            link_id: Link to update.
            active: ``True`` if a classical WDM channel is currently active.
        """
        item = self._link_items.get(link_id)
        if item is not None:
            item.update_raman_active(active)

    # ------------------------------------------------------------------
    # View interaction
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        new_scale = self._scale_factor * factor
        if new_scale < _MIN_SCALE or new_scale > _MAX_SCALE:
            return
        self._scale_factor = new_scale
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._read_only:
            super().mouseDoubleClickEvent(event)
            return
        item = self.itemAt(event.position().toPoint())
        if item is None:
            pos = self.mapToScene(event.position().toPoint())
            self.add_node_requested.emit(pos.x(), pos.y())
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._read_only:
            super().mousePressEvent(event)
            return
        item = self.itemAt(event.position().toPoint())

        # Click-to-place: pending node type from palette
        if (
            self._pending_node_type is not None
            and event.button() == Qt.MouseButton.LeftButton
            and item is None
        ):
            pos = self.mapToScene(event.position().toPoint())
            self._node_counter += 1
            node_id = f"node_{self._node_counter}"
            self._model.add_node(
                node_id, pos.x(), pos.y(), node_type=self._pending_node_type
            )
            event.accept()
            return  # stay in placement mode until Escape

        if event.button() == Qt.MouseButton.RightButton:
            if isinstance(item, QuantumNodeItem):
                self._show_node_context_menu(item, event.globalPosition().toPoint())
                return
            if isinstance(item, FiberLinkItem):
                self._show_link_context_menu(item, event.globalPosition().toPoint())
                return

        if event.button() == Qt.MouseButton.LeftButton and isinstance(item, QuantumNodeItem):
            if self._drawing_link and self._link_source is not None:
                # --- complete link: destination node was clicked ---
                dest_id = item.node_id()
                if dest_id != self._link_source:
                    link_id = f"link_{uuid4().hex[:8]}"
                    source_id = self._link_source
                    self._cancel_link_draw()
                    self._model.add_link(link_id, source_id, dest_id)
                    event.accept()
                    return
                # clicked the source again — cancel draw mode
                self._cancel_link_draw()
                event.accept()
                return
            else:
                # --- initiate link: source node was clicked ---
                self.begin_link_draw(item.node_id())
                event.accept()
                return

        super().mousePressEvent(event)

    def _show_node_context_menu(self, item: QuantumNodeItem, global_pos: QPoint) -> None:
        menu = QMenu(self)
        draw_link_action = menu.addAction("Draw Link From Here")
        menu.addSeparator()
        remove_action = menu.addAction("Remove Node")
        source_action = menu.addAction("Set as Source Node")
        bsm_action = menu.addAction("Set as BSM Node")
        memory_action = menu.addAction("Set as Memory Node")
        detector_action = menu.addAction("Set as Detector")

        node_id = item.node_id()
        draw_link_action.triggered.connect(lambda: self.begin_link_draw(node_id))
        remove_action.triggered.connect(lambda: self._model.remove_node(node_id))
        source_action.triggered.connect(
            lambda: self._retype_node(node_id, "source_node")
        )
        bsm_action.triggered.connect(lambda: self._retype_node(node_id, "bsm_node"))
        memory_action.triggered.connect(
            lambda: self._retype_node(node_id, "memory_node")
        )
        detector_action.triggered.connect(
            lambda: self._retype_node(node_id, "detector")
        )
        menu.exec(global_pos)

    def _show_link_context_menu(self, item: FiberLinkItem, global_pos: QPoint) -> None:
        menu = QMenu(self)
        remove_action = menu.addAction("Remove Link")
        toggle_raman_action = menu.addAction("Toggle Raman Coexistence")

        link_id = item.link_id
        remove_action.triggered.connect(lambda: self._model.remove_link(link_id))
        toggle_raman_action.triggered.connect(
            lambda: self.update_link_raman(link_id, not item._raman_active)
        )
        menu.exec(global_pos)

    def _retype_node(self, node_id: str, node_type: str) -> None:
        x, y = self._model.node_position(node_id)
        self._model.remove_node(node_id)
        self._model.add_node(node_id, x, y, node_type=node_type)

    def begin_link_draw(self, source_node_id: str) -> None:
        """Enter link-drawing mode: the next clicked node completes the link.

        Args:
            source_node_id: Node id the new link will originate from.
        """
        if self._read_only:
            return
        # Cancel any active draw first (in case of re-entry from context menu)
        if self._drawing_link:
            self._cancel_link_draw()
        self._drawing_link = True
        self._link_source = source_node_id
        self.setCursor(Qt.CursorShape.CrossCursor)
        source_item = self._node_items.get(source_node_id)
        if source_item is not None:
            source_item.set_link_source(True)

    def _cancel_link_draw(self) -> None:
        """Exit link-drawing mode without creating a link, restoring visuals."""
        if self._link_source is not None:
            source_item = self._node_items.get(self._link_source)
            if source_item is not None:
                source_item.set_link_source(False)
        self._drawing_link = False
        self._link_source = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.viewport().update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Cancel link-drawing or pending-placement mode on Escape.

        Args:
            event: Key event.
        """
        if event.key() == Qt.Key.Key_Escape:
            if self._drawing_link:
                self._cancel_link_draw()
            if self._pending_node_type is not None:
                self._cancel_pending_node()
            return
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Track cursor position in scene coords for ghost and rubber-band previews.

        Args:
            event: Mouse move event.
        """
        self._mouse_scene_pos = self.mapToScene(event.position().toPoint())
        if self._drawing_link or self._pending_node_type is not None:
            self.viewport().update()
        super().mouseMoveEvent(event)

    def drawForeground(self, painter: QPainter, rect: QRectF | QRect) -> None:
        """Ghost node preview and rubber-band line, drawn in scene coordinates.

        Both previews respect ``_mouse_scene_pos``; if it is ``None`` (cursor
        has not entered the viewport yet, or has left it) nothing is drawn.

        Args:
            painter: Scene-coordinate painter.
            rect: Exposed region in scene coordinates (unused).
        """
        super().drawForeground(painter, rect)
        painter.save()

        # Ghost node preview during placement mode
        if self._pending_node_type is not None and self._mouse_scene_pos is not None:
            colour = QuantumNodeItem.NODE_COLOURS.get(self._pending_node_type, "#58A6FF")
            painter.setPen(QPen(QColor(colour), 2.0, Qt.PenStyle.DashLine))
            fill = QColor(colour)
            fill.setAlpha(40)
            painter.setBrush(QBrush(fill))
            painter.drawEllipse(self._mouse_scene_pos, 22.0, 22.0)

        # Rubber-band line preview during link-drawing mode
        if (self._drawing_link and self._link_source is not None
                and self._mouse_scene_pos is not None):
            source_item = self._node_items.get(self._link_source)
            if source_item is not None:
                painter.setPen(QPen(QColor("#58A6FF"), 1.5, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawLine(source_item.scenePos(), self._mouse_scene_pos)

        painter.restore()

    def fit_view(self) -> None:
        """Fit the view to show the entire topology."""
        self.fitInView(
            self._scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

    def add_demo_topology(self) -> None:
        """Populate the model with a default 4-node linear demo topology."""
        self._model.add_node("Alice", 100.0, 200.0, node_type="source_node")
        self._model.add_node("Repeater_1", 300.0, 200.0, node_type="memory_node")
        self._model.add_node("Repeater_2", 500.0, 200.0, node_type="memory_node")
        self._model.add_node("Bob", 700.0, 200.0, node_type="detector")
        self._model.add_link("link_01", "Alice", "Repeater_1")
        self._model.add_link("link_02", "Repeater_1", "Repeater_2")
        self._model.add_link("link_03", "Repeater_2", "Bob")

    # ------------------------------------------------------------------
    # Palette placement modes
    # ------------------------------------------------------------------

    def set_pending_node_type(self, node_type: str) -> None:
        """Enter click-to-place mode for ``node_type``.

        Empty string or an empty call clears placement mode.

        Args:
            node_type: Type to place on next empty-canvas click;
                pass ``""`` to cancel.
        """
        if not node_type:
            self._cancel_pending_node()
            return
        self._pending_node_type = node_type
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.viewport().update()

    def _cancel_pending_node(self) -> None:
        """Exit click-to-place mode, restoring the default cursor."""
        self._pending_node_type = None
        self._mouse_scene_pos = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.viewport().update()

    def leaveEvent(self, event: object) -> None:
        """Clear ghost preview when cursor exits the canvas viewport."""
        if self._pending_node_type is not None or self._drawing_link:
            self._mouse_scene_pos = None
            self.viewport().update()
        if event is not None:
            super().leaveEvent(event)  # type: ignore[arg-type]

    def _on_node_config_from_item(self, node_id: str, config: dict[str, object]) -> None:
        """Forward a node config change accepted in NodePropertiesDialog."""
        self.node_config_changed.emit(node_id, config)

    # ------------------------------------------------------------------
    # Drag-and-drop (from NodePalette)
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        node_type = event.mimeData().text().strip()
        if not node_type:
            event.ignore()
            return
        pos = self.mapToScene(event.position().toPoint())
        self._node_counter += 1
        node_id = f"node_{self._node_counter}"
        self._model.add_node(node_id, pos.x(), pos.y(), node_type=node_type)
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # Layout and topology operations
    # ------------------------------------------------------------------

    def auto_layout(self) -> None:
        """Reposition all nodes using a NetworkX spring layout."""
        node_ids = self._model.node_ids()
        if not node_ids:
            return

        g: nx.Graph = nx.Graph()
        g.add_nodes_from(node_ids)
        for link_id in self._model.link_ids():
            data = self._model.link_data(link_id)
            g.add_edge(str(data["source"]), str(data["dest"]))

        pos: dict[str, tuple[float, float]] = nx.spring_layout(g, seed=42)

        scene_rect = self._scene.sceneRect()
        if not scene_rect.isValid() or scene_rect.width() < 100:
            scene_rect.setRect(-400.0, -300.0, 800.0, 600.0)
        cx = scene_rect.center().x()
        cy = scene_rect.center().y()
        margin = 80.0
        w = (scene_rect.width() / 2.0) - margin
        h = (scene_rect.height() / 2.0) - margin

        for node_id, (nx_x, nx_y) in pos.items():
            x = cx + nx_x * w
            y = cy + nx_y * h
            self._model.set_node_position(node_id, x, y)
            item = self._node_items.get(node_id)
            if item is not None:
                item.setPos(x, y)

        self.viewport().update()

    def clear_all(self) -> None:
        """Remove every node and link, leaving an empty topology."""
        self._model.clear()
