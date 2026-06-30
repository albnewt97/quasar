"""QuantumNodeItem and NodePropertiesDialog: visual node for topology canvas (§7.2).

Purely a rendering surface; node identity and graph membership live in
``TopologyModel`` / ``NetworkGraph``, never here (§3.6 GUI Isolation Law).
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QTransform
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsItem,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QLabel,
    QLineEdit,
    QStyleOptionGraphicsItem,
    QVBoxLayout,
    QWidget,
)

_GREEN = QColor("#3FB950")
_AMBER = QColor("#D29922")
_RED = QColor("#F85149")
_CYAN = QColor("#58A6FF")
_TEXT_LABEL = QColor("#8B949E")


def _lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    """Linearly interpolate between two colours.

    Args:
        c1: Colour at ``t=0``.
        c2: Colour at ``t=1``.
        t: Interpolation fraction, clamped to ``[0, 1]``.
    """
    t = max(0.0, min(1.0, t))
    r = round(c1.red() + (c2.red() - c1.red()) * t)
    g = round(c1.green() + (c2.green() - c1.green()) * t)
    b = round(c1.blue() + (c2.blue() - c1.blue()) * t)
    return QColor(r, g, b)


def _ring_colour(fidelity: float) -> QColor:
    """Map fidelity [0,1] → green/amber/red arc colour."""
    if fidelity >= 0.75:
        return _lerp_color(_AMBER, _GREEN, (fidelity - 0.75) / 0.25)
    if fidelity >= 0.5:
        return _lerp_color(_RED, _AMBER, (fidelity - 0.5) / 0.25)
    return _RED


class QuantumNodeItem(QGraphicsItem):
    """Draggable, selectable visual node on the topology canvas.

    Flat design: 15% alpha fill, solid-colour 2px border, fidelity progress
    ring, hover scale (×1.08) with "+" connection handle, cyan selection ring.

    Args:
        node_id: Unique node identifier (label text and tooltip).
        node_type: One of ``memory_node``, ``bsm_node``, ``source_node``,
            ``detector``; selects fill colour and icon glyph.
        parent: Optional parent graphics item.
    """

    RADIUS: float = 24.0
    NODE_COLOURS: dict[str, str] = {
        "memory_node": "#58A6FF",
        "bsm_node": "#BC8CFF",
        "source_node": "#3FB950",
        "detector": "#D29922",
    }
    _NODE_ICONS: dict[str, str] = {
        "memory_node": "M",
        "bsm_node": "B",
        "source_node": "S",
        "detector": "D",
    }

    def __init__(
        self,
        node_id: str,
        node_type: str = "memory_node",
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self._node_id = node_id
        self._node_type = node_type
        self._fidelity: float = 1.0
        self._hovered: bool = False
        self._is_link_source: bool = False
        self._config_changed_callback: Callable[[str, dict[str, object]], None] | None = None

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip(node_id)

    def boundingRect(self) -> QRectF:
        pad = self.RADIUS + 14.0  # room for fidelity ring + hover handle
        return QRectF(-pad, -pad, pad * 2.0, pad * 2.0 + 20.0)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        r = self.RADIUS
        colour = QColor(self.NODE_COLOURS.get(self._node_type, "#58A6FF"))
        selected = self.isSelected()

        # Flat fill: 15% opacity
        fill = QColor(colour)
        fill.setAlpha(38)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(fill))
        painter.drawEllipse(QRectF(-r, -r, r * 2, r * 2))

        # Border: cyan 2px when selected, else type-colour
        if selected:
            border_col: QColor = QColor(_CYAN)
            border_w = 2.0
        else:
            border_col = QColor(colour)
            border_col.setAlpha(220 if self._hovered else 180)
            border_w = 2.0
        painter.setPen(QPen(border_col, border_w))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRectF(-r, -r, r * 2, r * 2))

        # Fidelity progress ring (arc proportional to fidelity)
        if self._fidelity < 0.999:
            ring_r = r + 5.0
            ring_pen = QPen(_ring_colour(self._fidelity), 3.0)
            ring_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            span = -round(self._fidelity * 5760)  # clockwise from top
            painter.drawArc(
                QRectF(-ring_r, -ring_r, ring_r * 2, ring_r * 2), 1440, span
            )

        # Link-source dashed ring
        if self._is_link_source:
            ring_r2 = r + 10.0
            painter.setPen(QPen(_CYAN, 2.0, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(-ring_r2, -ring_r2, ring_r2 * 2, ring_r2 * 2))

        # Icon glyph
        icon_font = QFont()
        icon_font.setBold(True)
        icon_font.setPixelSize(12)
        painter.setFont(icon_font)
        painter.setPen(QPen(colour))
        painter.drawText(
            QRectF(-r, -r, r * 2, r * 2),
            Qt.AlignmentFlag.AlignCenter,
            self._NODE_ICONS.get(self._node_type, "?"),
        )

        # Label below node
        label_text = self._node_id if len(self._node_id) <= 12 else self._node_id[:12]
        label_font = QFont()
        label_font.setPixelSize(10)
        painter.setFont(label_font)
        painter.setPen(QPen(_TEXT_LABEL))
        painter.drawText(
            QRectF(-60.0, r + 6.0, 120.0, 16.0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            label_text,
        )

        # Hover "+" connection handle at right edge
        if self._hovered and not self._is_link_source:
            hx = r + 6.0
            handle_c = QColor(_CYAN)
            handle_c.setAlpha(200)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(handle_c))
            painter.drawEllipse(QRectF(hx - 6, -6, 12, 12))
            painter.setPen(QPen(QColor("#0D1117"), 1.5))
            painter.drawLine(int(hx - 3), 0, int(hx + 3), 0)
            painter.drawLine(int(hx), -3, int(hx), 3)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self._hovered = True
        self.setTransform(QTransform().scale(1.08, 1.08))
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self._hovered = False
        self.setTransform(QTransform())
        self.update()
        super().hoverLeaveEvent(event)

    def itemChange(
        self, change: QGraphicsItem.GraphicsItemChange, value: object
    ) -> object:
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if self.scene() is not None:
                self.scene().update()
        return super().itemChange(change, value)

    def set_link_source(self, active: bool) -> None:
        """Toggle the dashed-ring indicator marking this node as the link-draw source.

        Args:
            active: ``True`` to show the ring; ``False`` to remove it.
        """
        self._is_link_source = active
        self.update()

    def update_fidelity(self, fidelity: float) -> None:
        """Update live fidelity driving the progress ring colour.

        Args:
            fidelity: New fidelity in ``[0, 1]``; out-of-range values clamp.
        """
        self._fidelity = max(0.0, min(1.0, fidelity))
        self.update()

    def node_id(self) -> str:
        """Return this item's node identifier."""
        return self._node_id

    def node_type(self) -> str:
        """Return this item's node type."""
        return self._node_type

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSelected(True)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        dialog = NodePropertiesDialog(self._node_id, self._node_type)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            if self._config_changed_callback is not None:
                self._config_changed_callback(self._node_id, config)
        super().mouseDoubleClickEvent(event)


class NodePropertiesDialog(QDialog):
    """Properties dialog for a single quantum node.

    Shows an editable label, type selector, and T2/wear parameters for
    memory nodes.  The controller handler for applying changes to
    ``TwinOrchestrator`` is a stub — connect ``accepted`` externally.

    Args:
        node_id: Current node identifier (pre-populates label field).
        node_type: Current node type (pre-selects dropdown).
        parent: Optional parent widget.
    """

    _TYPES: list[str] = ["memory_node", "bsm_node", "source_node", "detector"]

    def __init__(
        self,
        node_id: str,
        node_type: str = "memory_node",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Node Properties")
        self.setMinimumWidth(300)

        outer = QVBoxLayout(self)
        form = QFormLayout()
        outer.addLayout(form)

        self._label_edit = QLineEdit(node_id)
        form.addRow("Label:", self._label_edit)

        self._type_combo = QComboBox()
        self._type_combo.addItems(self._TYPES)
        idx = self._TYPES.index(node_type) if node_type in self._TYPES else 0
        self._type_combo.setCurrentIndex(idx)
        form.addRow("Type:", self._type_combo)
        self._type_combo.currentTextChanged.connect(self._on_type_changed)

        self._t2_label = QLabel("T2 nominal:")
        self._t2_spin = QDoubleSpinBox()
        self._t2_spin.setRange(1e-6, 100.0)
        self._t2_spin.setValue(1.0)
        self._t2_spin.setSuffix(" s")
        form.addRow(self._t2_label, self._t2_spin)

        self._wear_label = QLabel("κ [s⁻²]:")
        self._wear_spin = QDoubleSpinBox()
        self._wear_spin.setRange(0.0, 10.0)
        self._wear_spin.setValue(1e-4)
        self._wear_spin.setDecimals(6)
        form.addRow(self._wear_label, self._wear_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._on_type_changed(node_type)

    def _on_type_changed(self, node_type: str) -> None:
        mem = node_type == "memory_node"
        self._t2_label.setVisible(mem)
        self._t2_spin.setVisible(mem)
        self._wear_label.setVisible(mem)
        self._wear_spin.setVisible(mem)

    def get_config(self) -> dict[str, object]:
        """Return current dialog state as a config dict.

        Returns:
            Dict with ``label``, ``node_type``, and optionally
            ``t2_nominal`` / ``wear_rate_kappa`` for memory nodes.
        """
        cfg: dict[str, object] = {
            "label": self._label_edit.text(),
            "node_type": self._type_combo.currentText(),
        }
        if self._type_combo.currentText() == "memory_node":
            cfg["t2_nominal"] = self._t2_spin.value()
            cfg["wear_rate_kappa"] = self._wear_spin.value()
        return cfg
