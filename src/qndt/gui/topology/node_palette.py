"""NodePalette: vertical sidebar for selecting and dragging quantum node types (§7.2)."""
from __future__ import annotations

from PySide6.QtCore import QMimeData, QPoint, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QDrag, QFont, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_PALETTE_ENTRIES: tuple[tuple[str, str, str], ...] = (
    ("source_node", "S", "#3FB950"),
    ("memory_node", "M", "#58A6FF"),
    ("bsm_node", "B", "#BC8CFF"),
    ("detector", "D", "#D29922"),
)

_LABEL_MAP: dict[str, str] = {
    "source_node": "Source",
    "memory_node": "Memory",
    "bsm_node": "BSM",
    "detector": "Detect",
}

_BTN_SIZE: int = 48


class _PaletteButton(QPushButton):
    """Single palette button: glyph + label chip, click-to-select, drag-to-place."""

    def __init__(
        self,
        node_type: str,
        glyph: str,
        colour: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._node_type = node_type
        self._glyph = glyph
        self._colour = QColor(colour)
        self._drag_start: QPoint | None = None

        self.setFixedSize(_BTN_SIZE, _BTN_SIZE)
        self.setCheckable(True)
        self.setObjectName("palette_button")
        self.setToolTip(_LABEL_MAP.get(node_type, node_type))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        checked = self.isChecked()
        hovered = self.underMouse()

        bg = QColor(self._colour)
        bg.setAlpha(80 if checked else (50 if hovered else 38))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(0, 0, _BTN_SIZE, _BTN_SIZE, 10, 10)

        border = QColor(self._colour)
        border.setAlpha(255 if checked else (200 if hovered else 140))
        painter.setPen(QPen(border, 2 if checked else 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(1, 1, _BTN_SIZE - 2, _BTN_SIZE - 2, 10, 10)

        glyph_font = QFont()
        glyph_font.setBold(True)
        glyph_font.setPixelSize(16)
        painter.setFont(glyph_font)
        painter.setPen(QPen(QColor(self._colour)))
        painter.drawText(
            0, 0, _BTN_SIZE, _BTN_SIZE - 12,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            self._glyph,
        )

        lbl_font = QFont()
        lbl_font.setPixelSize(9)
        painter.setFont(lbl_font)
        painter.setPen(QPen(QColor("#8B949E")))
        painter.drawText(
            0, _BTN_SIZE - 13, _BTN_SIZE, 13,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            _LABEL_MAP.get(self._node_type, self._node_type)[:6],
        )
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and self._drag_start is not None
        ):
            dist = (event.position().toPoint() - self._drag_start).manhattanLength()
            if dist >= QApplication.startDragDistance():
                self._start_drag()
        super().mouseMoveEvent(event)

    def _start_drag(self) -> None:
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(self._node_type)
        drag.setMimeData(mime)
        pixmap = QPixmap(self.size())
        self.render(pixmap)
        drag.setPixmap(pixmap)
        drag.setHotSpot(pixmap.rect().center())
        drag.exec(Qt.DropAction.CopyAction)


class NodePalette(QWidget):
    """Vertical sidebar palette with one 48×48 button per node type.

    Signals:
        node_type_selected: Emits the selected node type string when a button
            is activated; emits an empty string when deselected.
    """

    node_type_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(72)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setObjectName("node_palette")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        header = QLabel("Nodes")
        header.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        header.setObjectName("palette_header")
        layout.addWidget(header)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._buttons: dict[str, _PaletteButton] = {}

        for node_type, glyph, colour in _PALETTE_ENTRIES:
            btn = _PaletteButton(node_type, glyph, colour, self)
            self._button_group.addButton(btn)
            self._buttons[node_type] = btn
            layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
            btn.clicked.connect(
                lambda checked, nt=node_type: self._on_clicked(nt, checked)
            )

        layout.addStretch()

    def _on_clicked(self, node_type: str, checked: bool) -> None:
        self.node_type_selected.emit(node_type if checked else "")

    def set_active_type(self, node_type: str | None) -> None:
        """Highlight the button for ``node_type``; pass ``None`` to clear all.

        Args:
            node_type: Type to activate, or ``None`` to deselect every button.
        """
        self._button_group.setExclusive(False)
        for nt, btn in self._buttons.items():
            btn.setChecked(nt == node_type if node_type else False)
        self._button_group.setExclusive(True)
