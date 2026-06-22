"""FiberLinkItem: bezier fiber link with badge chip and hover delete (§7.2).

Purely a rendering surface; link identity and graph membership live in
``TopologyModel`` / ``NetworkGraph``, never here (§3.6 GUI Isolation Law).
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

from qndt.gui.topology.node_item import QuantumNodeItem, _lerp_color

_GREEN = QColor("#3FB950")
_RED = QColor("#F85149")
_AMBER = QColor("#D29922")
_CYAN = QColor("#58A6FF")
_TEXT_LABEL = QColor("#8B949E")
_BADGE_BG = QColor("#1F2630")

_PAD: float = 24.0
_DELETE_R: float = 7.0


class FiberLinkItem(QGraphicsItem):
    """Bezier link between two ``QuantumNodeItem``s.

    Features: bezier curve (S-curve when height differs), fidelity colour,
    label badge chip at midpoint, Raman parallel indicator, animated particle,
    hover thicker line and delete "×" affordance.

    Args:
        link_id: Unique link identifier.
        source_item: Node item at the transmitting end.
        dest_item: Node item at the receiving end.
        parent: Optional parent graphics item.
    """

    def __init__(
        self,
        link_id: str,
        source_item: QuantumNodeItem,
        dest_item: QuantumNodeItem,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self._link_id = link_id
        self._source = source_item
        self._dest = dest_item
        self._fidelity: float = 1.0
        self._raman_active: bool = False
        self._particle_offset: float = 0.0
        self._hovered: bool = False

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(-1.0)
        self.setToolTip(link_id)

    @property
    def link_id(self) -> str:
        """Unique link identifier."""
        return self._link_id

    @property
    def source_item(self) -> QuantumNodeItem:
        """Node item at the transmitting end."""
        return self._source

    @property
    def dest_item(self) -> QuantumNodeItem:
        """Node item at the receiving end."""
        return self._dest

    def _bezier_path(self, src: QPointF, dst: QPointF) -> QPainterPath:
        dx = dst.x() - src.x()
        dy = dst.y() - src.y()
        path = QPainterPath(src)
        if abs(dy) < 4.0:
            path.lineTo(dst)
        else:
            cp1 = QPointF(src.x() + dx / 3.0, src.y())
            cp2 = QPointF(dst.x() - dx / 3.0, dst.y())
            path.cubicTo(cp1, cp2, dst)
        return path

    def boundingRect(self) -> QRectF:
        src = self._source.scenePos()
        dst = self._dest.scenePos()
        return QRectF(src, dst).normalized().adjusted(-_PAD, -_PAD, _PAD, _PAD)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        src = self._source.scenePos()
        dst = self._dest.scenePos()
        mid = QPointF((src.x() + dst.x()) / 2.0, (src.y() + dst.y()) / 2.0)

        path = self._bezier_path(src, dst)
        line_colour = _lerp_color(_RED, _GREEN, self._fidelity)
        line_width = 3.5 if (self.isSelected() or self._hovered) else 2.0
        painter.setPen(QPen(line_colour, line_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        # Raman coexistence: parallel amber line
        if self._raman_active:
            dx = dst.x() - src.x()
            dy = dst.y() - src.y()
            length = math.hypot(dx, dy)
            if length > 0:
                ox = -dy / length * 4.0
                oy = dx / length * 4.0
                raman_path = QPainterPath(QPointF(src.x() + ox, src.y() + oy))
                raman_path.lineTo(QPointF(dst.x() + ox, dst.y() + oy))
                painter.setOpacity(0.5)
                painter.setPen(QPen(_AMBER, 2.0))
                painter.drawPath(raman_path)
                painter.setOpacity(1.0)

        # Animated particle (cubic bezier formula)
        t = self._particle_offset
        dx2 = dst.x() - src.x()
        cp1x = src.x() + dx2 / 3.0
        cp2x = dst.x() - dx2 / 3.0
        s = 1.0 - t
        px = s**3 * src.x() + 3*s**2*t*cp1x + 3*s*t**2*cp2x + t**3 * dst.x()
        py = s**3 * src.y() + 3*s**2*t*src.y() + 3*s*t**2*dst.y() + t**3 * dst.y()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_CYAN)
        painter.drawEllipse(QPointF(px, py), 4.0, 4.0)

        # Label badge chip
        label_text = self._link_id if len(self._link_id) <= 10 else self._link_id[:10]
        badge_font = QFont()
        badge_font.setPixelSize(9)
        painter.setFont(badge_font)
        fm = painter.fontMetrics()
        text_w = fm.horizontalAdvance(label_text)
        badge_w = text_w + 10
        badge_h = 14
        badge_y_off = -14.0 if self._hovered else 0.0
        badge_rect = QRectF(
            mid.x() - badge_w / 2.0,
            mid.y() - badge_h / 2.0 + badge_y_off,
            badge_w,
            badge_h,
        )
        badge_bg = QColor(_BADGE_BG)
        badge_bg.setAlpha(200)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(badge_bg))
        painter.drawRoundedRect(badge_rect, 3, 3)
        painter.setPen(QPen(_TEXT_LABEL))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, label_text)

        # Delete "×" affordance at midpoint on hover
        if self._hovered:
            del_c = QColor(_RED)
            del_c.setAlpha(210)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(del_c))
            del_rect = QRectF(
                mid.x() - _DELETE_R, mid.y() - _DELETE_R,
                _DELETE_R * 2, _DELETE_R * 2,
            )
            painter.drawEllipse(del_rect)
            painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
            d = _DELETE_R - 3.0
            painter.drawLine(
                QPointF(mid.x() - d, mid.y() - d),
                QPointF(mid.x() + d, mid.y() + d),
            )
            painter.drawLine(
                QPointF(mid.x() + d, mid.y() - d),
                QPointF(mid.x() - d, mid.y() + d),
            )

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._hovered:
            src = self._source.scenePos()
            dst = self._dest.scenePos()
            mid = QPointF((src.x() + dst.x()) / 2.0, (src.y() + dst.y()) / 2.0)
            click = event.scenePos()
            if math.hypot(click.x() - mid.x(), click.y() - mid.y()) <= _DELETE_R * 1.5:
                scene = self.scene()
                if scene is not None and hasattr(scene, "_delete_link_requested"):
                    scene._delete_link_requested(self._link_id)
                event.accept()
                return
        super().mousePressEvent(event)

    def advance(self, dt: float) -> None:
        """Advance the animated entanglement particle along the link.

        Args:
            dt: Elapsed time fraction driving the particle's progress.
        """
        self._particle_offset = (self._particle_offset + dt * 0.3) % 1.0
        self.update()

    def update_fidelity(self, fidelity: float) -> None:
        """Update the live fidelity driving the line colour.

        Args:
            fidelity: New fidelity in ``[0, 1]``; clamped.
        """
        self._fidelity = max(0.0, min(1.0, fidelity))
        self.update()

    def update_raman_active(self, active: bool) -> None:
        """Toggle the classical-traffic (Raman coexistence) indicator.

        Args:
            active: ``True`` if a classical WDM channel is currently active.
        """
        self._raman_active = active
        self.update()
