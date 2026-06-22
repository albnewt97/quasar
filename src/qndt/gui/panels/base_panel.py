"""BasePanel: shared base class for all parameter panels (§7.3, §4.2).

Each panel owns a pydantic model for validation.  Invalid inputs are
rejected here -- they never reach an engine.  ``config_changed`` only
fires after the panel's own values have validated successfully.
"""
from __future__ import annotations

from pydantic import BaseModel, ValidationError
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


class BasePanel(QWidget):
    """Shared base for all parameter panels.

    Signals:
        config_changed: Emitted with ``(engine_id, params_dict)`` once a
            panel's pending values have validated successfully.
        validation_error: Emitted with ``(field_name, error_message)`` for
            each field that failed validation.
    """

    config_changed = Signal(str, dict)
    validation_error = Signal(str, str)

    def __init__(self, engine_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._engine_id = engine_id
        self._dirty: bool = False
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)

    def _make_section_label(self, text: str) -> QLabel:
        """Build a section header label styled via the ``section_header`` QSS rule.

        Args:
            text: Section title, upper-cased for display.
        """
        label = QLabel(text.upper())
        label.setObjectName("section_header")
        return label

    def _make_field_row(
        self, label_text: str, widget: QWidget, unit: str = ""
    ) -> QHBoxLayout:
        """Build a labelled field row: ``[label | widget | unit]``.

        Args:
            label_text: Field name shown at fixed width on the left.
            widget: The input widget, stretched to fill remaining space.
            unit: Optional unit suffix shown at fixed width on the right.
        """
        row = QHBoxLayout()
        name_label = QLabel(label_text)
        name_label.setFixedWidth(100)
        row.addWidget(name_label)
        row.addWidget(widget, 1)
        unit_label = QLabel(unit)
        unit_label.setFixedWidth(50)
        row.addWidget(unit_label)
        return row

    def _emit_if_valid(self, params: dict[str, object]) -> None:
        """Validate ``params`` against this panel's pydantic model and emit.

        Args:
            params: Candidate parameter values for ``self._engine_id``.
        """
        model_cls = self._config_model()
        try:
            model_cls.model_validate(params)
        except ValidationError as exc:
            for error in exc.errors():
                field_name = ".".join(str(loc) for loc in error["loc"])
                self.validation_error.emit(field_name, error["msg"])
            return
        self.config_changed.emit(self._engine_id, params)

    def _config_model(self) -> type[BaseModel]:
        """Return the pydantic model used to validate this panel's config.

        Subclasses must override this to enable ``_emit_if_valid``.
        """
        raise NotImplementedError

    def _show_field_error(self, widget: QWidget, message: str) -> None:
        """Mark a field widget as invalid with a red border and tooltip.

        Args:
            widget: The offending input widget.
            message: Human-readable validation error.
        """
        widget.setStyleSheet("border: 1px solid #F85149;")
        widget.setToolTip(message)

    def _clear_field_error(self, widget: QWidget) -> None:
        """Clear a previously set field error.

        Args:
            widget: The input widget to restore.
        """
        widget.setStyleSheet("")
        widget.setToolTip("")

    def get_config(self) -> dict[str, object]:
        """Return the panel's current values as a dict.

        Subclasses must override this.
        """
        raise NotImplementedError
