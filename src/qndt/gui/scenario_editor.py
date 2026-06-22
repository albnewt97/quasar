"""ScenarioEditor: modal dialog for saving/loading scenario JSON files (§7.3, §12).

Edits the top-level simulation parameters of a ``ScenarioConfig`` and shows
a live JSON preview; topology nodes/links/kernel from the supplied config
are carried through unmodified since this dialog has no view onto the
topology canvas.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from qndt.io.config import ScenarioConfig


def _make_section_label(text: str) -> QLabel:
    """Build an upper-cased section header label."""
    label = QLabel(text.upper())
    label.setObjectName("section_header")
    return label


class ScenarioEditor(QDialog):
    """Modal dialog for saving/loading scenario JSON files.

    Signals:
        scenario_saved: Emitted with the destination path after a successful save.
        scenario_loaded: Emitted with the source path after a successful load.
    """

    scenario_saved = Signal(str)
    scenario_loaded = Signal(str)

    def __init__(self, current_config: ScenarioConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scenario Editor")
        self.setModal(True)
        self._config = current_config
        self._loaded_path: str | None = None

        layout = QVBoxLayout(self)

        layout.addWidget(_make_section_label("Scenario Metadata"))
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit(current_config.scenario_name)
        self._name_edit.textChanged.connect(self._update_json_preview)
        name_row.addWidget(self._name_edit)
        layout.addLayout(name_row)

        layout.addWidget(_make_section_label("Simulation Parameters"))
        grid = QGridLayout()

        grid.addWidget(QLabel("Duration (s):"), 0, 0)
        self._duration = QDoubleSpinBox()
        self._duration.setRange(0.1, 86400.0)
        self._duration.setValue(current_config.duration_s)
        self._duration.valueChanged.connect(self._update_json_preview)
        grid.addWidget(self._duration, 0, 1)

        grid.addWidget(QLabel("Time step (s):"), 1, 0)
        self._dt = QDoubleSpinBox()
        self._dt.setRange(1e-3, 10.0)
        self._dt.setDecimals(4)
        self._dt.setValue(current_config.dt_s)
        self._dt.valueChanged.connect(self._update_json_preview)
        grid.addWidget(self._dt, 1, 1)

        grid.addWidget(QLabel("χ_max:"), 2, 0)
        self._chi_max = QSpinBox()
        self._chi_max.setRange(1, 256)
        self._chi_max.setValue(current_config.chi_max)
        self._chi_max.valueChanged.connect(self._update_json_preview)
        grid.addWidget(self._chi_max, 2, 1)

        grid.addWidget(QLabel("κ_max:"), 3, 0)
        self._kappa_max = QSpinBox()
        self._kappa_max.setRange(1, 64)
        self._kappa_max.setValue(current_config.kappa_max)
        self._kappa_max.valueChanged.connect(self._update_json_preview)
        grid.addWidget(self._kappa_max, 3, 1)

        layout.addLayout(grid)

        layout.addWidget(_make_section_label("Scenario JSON"))
        self._json_preview = QPlainTextEdit()
        self._json_preview.setReadOnly(True)
        self._json_preview.setStyleSheet("font-family: 'Courier New', monospace;")
        layout.addWidget(self._json_preview)

        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(self._copy_to_clipboard)
        layout.addWidget(copy_button)

        button_box = QDialogButtonBox()
        save_as_button = button_box.addButton(
            "Save As...", QDialogButtonBox.ButtonRole.ActionRole
        )
        save_as_button.clicked.connect(self._save_as)
        load_button = button_box.addButton("Load...", QDialogButtonBox.ButtonRole.ActionRole)
        load_button.clicked.connect(self._load)
        close_button = button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        close_button.clicked.connect(self.reject)
        layout.addWidget(button_box)

        self._update_json_preview()

    def _save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Scenario", "", "JSON Files (*.json)")
        if not path:
            return
        config = self.get_config()
        config.to_json_file(path)
        self._config = config
        self._loaded_path = path
        self.scenario_saved.emit(path)
        QMessageBox.information(self, "Scenario Saved", "Scenario saved.")

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Scenario", "", "JSON Files (*.json)")
        if not path:
            return
        config = ScenarioConfig.from_json_file(path)
        self._config = config
        self._loaded_path = path
        self._name_edit.setText(config.scenario_name)
        self._duration.setValue(config.duration_s)
        self._dt.setValue(config.dt_s)
        self._chi_max.setValue(config.chi_max)
        self._kappa_max.setValue(config.kappa_max)
        self._update_json_preview()
        self.scenario_loaded.emit(path)

    def _update_json_preview(self) -> None:
        self._json_preview.setPlainText(self.get_config().model_dump_json(indent=2))

    def _copy_to_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._json_preview.toPlainText())

    def get_config(self) -> ScenarioConfig:
        """Return a ``ScenarioConfig`` built from the current field values."""
        return ScenarioConfig(
            scenario_name=self._name_edit.text(),
            nodes=self._config.nodes,
            links=self._config.links,
            kernel=self._config.kernel,
            duration_s=self._duration.value(),
            dt_s=self._dt.value(),
            chi_max=self._chi_max.value(),
            kappa_max=self._kappa_max.value(),
        )
