"""Tests for TopologyModel, TopologyCanvas, QuantumNodeItem, FiberLinkItem (§7.2).

Run headless via ``QT_QPA_PLATFORM=offscreen`` (see conftest.py qapp_env).
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent
from pytestqt.qtbot import QtBot

from qndt.gui.topology.canvas import TopologyCanvas
from qndt.gui.topology.node_item import NodePropertiesDialog
from qndt.gui.topology.node_palette import NodePalette
from qndt.gui.topology.topology_model import TopologyModel


def test_topology_model_add_node() -> None:
    """add_node() registers the node in node_ids()."""
    model = TopologyModel()
    model.add_node("node_a", 0.0, 0.0)
    assert "node_a" in model.node_ids()


def test_topology_model_add_link() -> None:
    """add_link() between two existing nodes registers it in link_ids()."""
    model = TopologyModel()
    model.add_node("node_a", 0.0, 0.0)
    model.add_node("node_b", 100.0, 0.0)
    model.add_link("link_ab", "node_a", "node_b")
    assert "link_ab" in model.link_ids()


def test_topology_model_remove_node() -> None:
    """remove_node() drops the node from node_ids()."""
    model = TopologyModel()
    model.add_node("node_a", 0.0, 0.0)
    model.remove_node("node_a")
    assert "node_a" not in model.node_ids()


def test_topology_model_signals(qtbot: QtBot) -> None:
    """add_node() emits topology_changed."""
    model = TopologyModel()
    with qtbot.waitSignal(model.topology_changed, timeout=1000):
        model.add_node("node_a", 0.0, 0.0)


def test_canvas_creates_node_item(qtbot: QtBot) -> None:
    """Adding a node to the model creates a corresponding canvas item."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    model.add_node("node_a", 10.0, 20.0)
    assert "node_a" in canvas._node_items


def test_canvas_creates_link_item(qtbot: QtBot) -> None:
    """Adding a link to the model creates a corresponding canvas item."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    model.add_node("node_a", 0.0, 0.0)
    model.add_node("node_b", 100.0, 0.0)
    model.add_link("link_ab", "node_a", "node_b")
    assert "link_ab" in canvas._link_items


def test_canvas_demo_topology(qtbot: QtBot) -> None:
    """add_demo_topology() populates the model with 4 nodes and 3 links."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    canvas.add_demo_topology()
    assert len(model.node_ids()) == 4
    assert len(model.link_ids()) == 3


def test_node_item_fidelity_update(qtbot: QtBot) -> None:
    """update_fidelity() on a node item does not raise."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    model.add_node("node_a", 0.0, 0.0)
    canvas._node_items["node_a"].update_fidelity(0.3)


def test_link_item_advance(qtbot: QtBot) -> None:
    """advance() updates the link item's particle offset."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    model.add_node("node_a", 0.0, 0.0)
    model.add_node("node_b", 100.0, 0.0)
    model.add_link("link_ab", "node_a", "node_b")
    link_item = canvas._link_items["link_ab"]
    before = link_item._particle_offset
    link_item.advance(0.1)
    assert link_item._particle_offset != before


# ---------------------------------------------------------------------------
# Link-drawing interaction tests
# ---------------------------------------------------------------------------


def test_link_draw_mode_activates_on_node_click(qtbot: QtBot) -> None:
    """Left-clicking a node enters link-drawing mode and marks it as source."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    canvas.show()
    canvas.resize(800, 600)
    model.add_node("Alice", 100.0, 200.0)
    model.add_node("Bob", 400.0, 200.0)

    alice_item = canvas._node_items["Alice"]
    alice_vp = canvas.mapFromScene(alice_item.scenePos())
    qtbot.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=alice_vp)

    assert canvas._drawing_link is True
    assert canvas._link_source == "Alice"
    assert alice_item._is_link_source is True


def test_link_draw_completes_on_second_click(qtbot: QtBot) -> None:
    """Clicking a destination node while in draw mode creates the link and resets state."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    canvas.show()
    canvas.resize(800, 600)
    model.add_node("Alice", 100.0, 200.0)
    model.add_node("Bob", 400.0, 200.0)

    canvas.begin_link_draw("Alice")
    assert canvas._drawing_link is True

    bob_item = canvas._node_items["Bob"]
    bob_vp = canvas.mapFromScene(bob_item.scenePos())
    qtbot.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=bob_vp)

    assert canvas._drawing_link is False
    assert canvas._link_source is None
    assert len(model.link_ids()) == 1


def test_escape_cancels_link_draw(qtbot: QtBot) -> None:
    """Pressing Escape while in link-draw mode exits without creating a link."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    model.add_node("Alice", 100.0, 200.0)

    canvas.begin_link_draw("Alice")
    assert canvas._drawing_link is True
    assert canvas._node_items["Alice"]._is_link_source is True

    key_event = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Escape,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.keyPressEvent(key_event)

    assert canvas._drawing_link is False
    assert canvas._link_source is None
    assert canvas._node_items["Alice"]._is_link_source is False


def test_right_click_menu_has_draw_link_option(qtbot: QtBot) -> None:
    """begin_link_draw() (called by the 'Draw Link From Here' menu item) works cleanly."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    model.add_node("Alice", 100.0, 200.0)

    canvas.begin_link_draw("Alice")
    assert canvas._drawing_link is True
    assert canvas._link_source == "Alice"

    canvas._drawing_link = False
    canvas._link_source = None


# ---------------------------------------------------------------------------
# NodePalette tests
# ---------------------------------------------------------------------------


def test_palette_creates(qtbot: QtBot) -> None:
    """NodePalette() creates and has 4 type buttons."""
    palette = NodePalette()
    qtbot.addWidget(palette)
    assert len(palette._buttons) == 4
    for nt in ("source_node", "memory_node", "bsm_node", "detector"):
        assert nt in palette._buttons


def test_palette_emits_type(qtbot: QtBot) -> None:
    """Clicking a palette button emits node_type_selected with that type."""
    palette = NodePalette()
    qtbot.addWidget(palette)
    received: list[str] = []
    palette.node_type_selected.connect(received.append)

    with qtbot.waitSignal(palette.node_type_selected, timeout=1000):
        palette._buttons["source_node"].click()

    assert received and received[-1] == "source_node"


def test_palette_set_active_type(qtbot: QtBot) -> None:
    """set_active_type() marks the correct button checked."""
    palette = NodePalette()
    qtbot.addWidget(palette)

    palette.set_active_type("bsm_node")
    assert palette._buttons["bsm_node"].isChecked()
    assert not palette._buttons["memory_node"].isChecked()

    palette.set_active_type(None)
    for btn in palette._buttons.values():
        assert not btn.isChecked()


# ---------------------------------------------------------------------------
# Canvas placement mode tests
# ---------------------------------------------------------------------------


def test_canvas_pending_placement(qtbot: QtBot) -> None:
    """set_pending_node_type then click on empty canvas places a node of that type."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    canvas.show()
    canvas.resize(800, 600)

    canvas.set_pending_node_type("bsm_node")
    assert canvas._pending_node_type == "bsm_node"

    # Simulate a left-click on empty canvas space
    before = len(model.node_ids())
    qtbot.mouseClick(
        canvas.viewport(), Qt.MouseButton.LeftButton, pos=canvas.viewport().rect().center()
    )
    after = len(model.node_ids())
    assert after == before + 1

    # Node should be of the correct type
    new_id = model.node_ids()[-1]
    graph_attrs = model.graph()._nodes.get(new_id, {})
    assert graph_attrs.get("type") == "bsm_node"

    # Mode persists (still pending) until Escape
    assert canvas._pending_node_type == "bsm_node"

    key_event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
    )
    canvas.keyPressEvent(key_event)
    assert canvas._pending_node_type is None


def test_auto_layout_runs(qtbot: QtBot) -> None:
    """auto_layout() repositions nodes (positions change from initial values)."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    canvas.show()
    canvas.resize(800, 600)
    canvas.add_demo_topology()

    before = {nid: model.node_position(nid) for nid in model.node_ids()}
    canvas.auto_layout()
    after = {nid: model.node_position(nid) for nid in model.node_ids()}

    assert before != after, "auto_layout() must change at least one node position"


def test_clear_all(qtbot: QtBot) -> None:
    """clear_all() leaves 0 nodes and 0 links in the model."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    canvas.add_demo_topology()
    assert len(model.node_ids()) > 0

    canvas.clear_all()
    assert len(model.node_ids()) == 0
    assert len(model.link_ids()) == 0


def test_node_properties_dialog_creates(qtbot: QtBot) -> None:
    """NodePropertiesDialog for a node creates without error."""
    dialog = NodePropertiesDialog("Alice", "memory_node")
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Node Properties"
    cfg = dialog.get_config()
    assert cfg["node_type"] == "memory_node"
    assert "t2_nominal" in cfg


# ---------------------------------------------------------------------------
# Ghost-preview bug regression tests (BUG 1)
# ---------------------------------------------------------------------------


def test_ghost_not_added_as_scene_item(qtbot: QtBot) -> None:
    """Ghost preview in placement mode must not add items to the QGraphicsScene."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    canvas.show()
    canvas.resize(800, 600)

    before = len(canvas._scene.items())
    canvas.set_pending_node_type("memory_node")
    qtbot.mouseMove(canvas.viewport(), pos=QPoint(200, 200))

    assert len(canvas._scene.items()) == before, (
        "set_pending_node_type + mouse move must not add ghost as a scene item"
    )
    assert canvas._mouse_scene_pos is not None, (
        "mouseMoveEvent must update _mouse_scene_pos"
    )


def test_placement_mode_clears_on_escape(qtbot: QtBot) -> None:
    """Escape in placement mode clears both _pending_node_type and _mouse_scene_pos."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)

    canvas.set_pending_node_type("bsm_node")
    canvas._mouse_scene_pos = QPointF(100.0, 100.0)

    key_event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
    )
    canvas.keyPressEvent(key_event)

    assert canvas._pending_node_type is None
    assert canvas._mouse_scene_pos is None


def test_leave_event_clears_ghost(qtbot: QtBot) -> None:
    """leaveEvent while in placement mode clears _mouse_scene_pos."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)

    canvas.set_pending_node_type("detector")
    canvas._mouse_scene_pos = QPointF(50.0, 50.0)
    assert canvas._mouse_scene_pos is not None

    canvas.leaveEvent(None)

    assert canvas._mouse_scene_pos is None


def test_drag_drop_mime(qtbot: QtBot) -> None:
    """dropEvent with a valid node-type mime string places a node."""
    model = TopologyModel()
    canvas = TopologyCanvas(model)
    qtbot.addWidget(canvas)
    canvas.show()
    canvas.resize(800, 600)

    before = len(model.node_ids())

    mime = QMimeData()
    mime.setText("detector")

    # Call dropEvent indirectly via the canvas's dropEvent handler
    class _FakeDropEvent:
        def mimeData(self) -> QMimeData:
            return mime

        def position(self) -> QPointF:  # type: ignore[override]
            return QPointF(400.0, 300.0)

        def acceptProposedAction(self) -> None:
            pass

        def ignore(self) -> None:
            pass

    canvas.dropEvent(_FakeDropEvent())  # type: ignore[arg-type]
    assert len(model.node_ids()) == before + 1
    new_id = model.node_ids()[-1]
    assert model.graph()._nodes[new_id].get("type") == "detector"
