"""QuasarMainWindow: top-level dockable application shell (§7, §4.2, §11 prompt 11).

This module and everything under ``qndt.gui`` is the only place in the
codebase permitted to import PySide6 (§3.6 GUI Isolation Law).  The window
itself owns no physics and no quantum state -- it is a pure consumer that
emits Qt Signals for the caller (eventually ``TwinOrchestrator``) to act on.
"""
from __future__ import annotations

from uuid import uuid4

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from qndt.gui.dashboard.dashboard_window import DashboardWindow
from qndt.gui.panels.aging_panel import AgingPanel
from qndt.gui.panels.channel_panel import ChannelPanel
from qndt.gui.panels.coexistence_panel import CoexistencePanel
from qndt.gui.panels.control_plane_panel import ControlPlanePanel
from qndt.gui.panels.telemetry_panel import TelemetryPanel
from qndt.gui.scenario_editor import ScenarioEditor
from qndt.gui.telemetry_viewer import TelemetryViewer
from qndt.gui.topology.canvas import TopologyCanvas
from qndt.gui.topology.node_palette import NodePalette
from qndt.gui.topology.topology_model import TopologyModel
from qndt.io.config import ScenarioConfig


class QuasarMainWindow(QMainWindow):
    """Top-level application window: dock layout, menu bar, status bar.

    Signals:
        simulation_run_requested: Emitted when the user requests sim start.
        simulation_step_requested: Emitted when the user requests a single
            simulation step.
        simulation_pause_requested: Emitted when the user requests sim pause.
        simulation_stop_requested: Emitted when the user requests sim stop.
        simulation_reset_requested: Emitted when the user requests sim reset.
        scenario_new_requested: Emitted when the user chooses File > New
            Scenario.
        scenario_open_requested: Emitted with a file path when the user
            chooses File > Open Scenario.
        scenario_save_requested: Emitted with a file path when the user
            chooses File > Save Scenario.
        layout_reset_requested: Emitted when the user chooses View > Reset
            Layout.
    """

    simulation_run_requested = Signal()
    simulation_step_requested = Signal()
    simulation_pause_requested = Signal()
    simulation_stop_requested = Signal()
    simulation_reset_requested = Signal()
    scenario_new_requested = Signal()
    scenario_open_requested = Signal(str)
    scenario_save_requested = Signal(str)
    layout_reset_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Quasar — Quantum Network Digital Twin")
        self.resize(1600, 1000)
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowNestedDocks
        )

        self._build_central_widget()
        self._build_docks()
        self._build_menu_bar()
        self._build_status_bar()
        self._connect_menu_actions()
        self._canvas.add_demo_topology()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_central_widget(self) -> None:
        self._topology_model = TopologyModel(self)
        self._canvas = TopologyCanvas(self._topology_model, self)
        self._canvas.add_node_requested.connect(self._on_add_node_requested)
        self._canvas.topology_changed.connect(self._on_topology_changed)

        # Node palette sidebar
        self._node_palette = NodePalette(self)
        self._node_palette.node_type_selected.connect(self._canvas.set_pending_node_type)

        # Canvas toolbar strip
        canvas_toolbar = self._build_canvas_toolbar()

        # Assemble: toolbar above canvas
        canvas_panel = QWidget(self)
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        canvas_layout.addWidget(canvas_toolbar)
        canvas_layout.addWidget(self._canvas)

        # Central widget: [palette | canvas panel]
        central = QWidget(self)
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self._node_palette)
        central_layout.addWidget(canvas_panel, stretch=1)
        self.setCentralWidget(central)

    def _build_canvas_toolbar(self) -> QWidget:
        bar = QWidget(self)
        bar.setObjectName("canvas_toolbar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        def _btn(label: str, tip: str) -> QPushButton:
            b = QPushButton(label, bar)
            b.setToolTip(tip)
            b.setObjectName("canvas_toolbar_btn")
            b.setFixedHeight(24)
            return b

        btn_select = _btn("Select", "Select mode (Esc)")
        btn_select.clicked.connect(lambda: self._canvas.set_pending_node_type(""))
        layout.addWidget(btn_select)

        btn_link = _btn("Link", "Draw a fiber link (click source then dest node)")
        btn_link.clicked.connect(lambda: self._canvas._cancel_pending_node())
        layout.addWidget(btn_link)

        btn_delete = _btn("Delete", "Remove selected items")
        btn_delete.clicked.connect(self._on_delete_selected)
        layout.addWidget(btn_delete)

        sep = QWidget(bar)
        sep.setFixedWidth(1)
        sep.setObjectName("canvas_toolbar_sep")
        layout.addWidget(sep)

        btn_layout = _btn("Auto Layout", "Spring-layout all nodes")
        btn_layout.clicked.connect(self._canvas.auto_layout)
        layout.addWidget(btn_layout)

        btn_fit = _btn("Fit View", "Fit canvas to topology")
        btn_fit.clicked.connect(self._canvas.fit_view)
        layout.addWidget(btn_fit)

        btn_clear = _btn("Clear All", "Remove every node and link")
        btn_clear.clicked.connect(self._on_clear_all)
        layout.addWidget(btn_clear)

        layout.addStretch()
        return bar

    def _on_delete_selected(self) -> None:
        for item in list(self._canvas._scene.selectedItems()):
            from qndt.gui.topology.link_item import FiberLinkItem
            from qndt.gui.topology.node_item import QuantumNodeItem
            if isinstance(item, QuantumNodeItem):
                self._topology_model.remove_node(item.node_id())
            elif isinstance(item, FiberLinkItem):
                self._topology_model.remove_link(item.link_id)

    def _on_clear_all(self) -> None:
        self._canvas.clear_all()
        self._node_palette.set_active_type(None)

    def _build_docks(self) -> None:
        self._parameter_dock = QDockWidget("Parameters", self)
        self._parameter_dock.setObjectName("parameter_dock")
        parameter_tabs = QTabWidget()

        self._control_plane_panel = ControlPlanePanel()
        self._channel_panel = ChannelPanel()
        self._coexistence_panel = CoexistencePanel()
        self._telemetry_panel = TelemetryPanel()
        self._aging_panel = AgingPanel()

        telemetry_scroll = QScrollArea()
        telemetry_scroll.setWidgetResizable(True)
        telemetry_scroll.setWidget(self._telemetry_panel)

        parameter_tabs.addTab(self._control_plane_panel, "Network")
        parameter_tabs.addTab(self._channel_panel, "Channel")
        parameter_tabs.addTab(self._coexistence_panel, "Noise")
        parameter_tabs.addTab(telemetry_scroll, "Telemetry")
        parameter_tabs.addTab(self._aging_panel, "Aging")

        for panel in (
            self._control_plane_panel,
            self._channel_panel,
            self._coexistence_panel,
            self._telemetry_panel,
            self._aging_panel,
        ):
            panel.config_changed.connect(self._on_config_changed)

        self._parameter_dock.setWidget(parameter_tabs)
        self._parameter_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._parameter_dock)
        self._parameter_dock.setMinimumWidth(320)

        self._dashboard_dock = QDockWidget("Dashboard", self)
        self._dashboard_dock.setObjectName("dashboard_dock")
        self._dashboard = DashboardWindow(self._topology_model, self)
        self._dashboard_dock.setWidget(self._dashboard)
        self._dashboard_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dashboard_dock)
        self._dashboard_dock.setMinimumWidth(420)

        self._telemetry_dock = QDockWidget("Telemetry Viewer", self)
        self._telemetry_dock.setObjectName("telemetry_dock")
        self._telemetry_viewer = TelemetryViewer(self)
        self._telemetry_dock.setWidget(self._telemetry_viewer)
        self._telemetry_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._telemetry_dock)
        self._telemetry_dock.setMinimumHeight(180)

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        self._action_new_scenario = file_menu.addAction("&New Scenario")
        self._action_new_scenario.setShortcut(QKeySequence("Ctrl+N"))
        self._action_open_scenario = file_menu.addAction("&Open Scenario...")
        self._action_open_scenario.setShortcut(QKeySequence("Ctrl+O"))
        self._action_save_scenario = file_menu.addAction("&Save Scenario")
        self._action_save_scenario.setShortcut(QKeySequence("Ctrl+S"))
        self._action_save_scenario_as = file_menu.addAction("Save Scenario &As...")
        self._action_save_scenario_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        file_menu.addSeparator()
        self._action_exit = file_menu.addAction("E&xit")
        self._action_exit.setShortcut(QKeySequence("Ctrl+Q"))

        sim_menu = menu_bar.addMenu("&Simulation")
        self._action_sim_start = sim_menu.addAction("&Start")
        self._action_sim_start.setShortcut(QKeySequence("F5"))
        self._action_sim_step = sim_menu.addAction("Step &Once")
        self._action_sim_step.setShortcut(QKeySequence("F7"))
        self._action_sim_pause = sim_menu.addAction("&Pause")
        self._action_sim_pause.setShortcut(QKeySequence("F6"))
        self._action_sim_stop = sim_menu.addAction("S&top")
        self._action_sim_stop.setShortcut(QKeySequence("Shift+F5"))
        self._action_sim_reset = sim_menu.addAction("&Reset")
        self._action_sim_reset.setShortcut(QKeySequence("Ctrl+R"))

        view_menu = menu_bar.addMenu("&View")
        self._action_toggle_parameters = self._parameter_dock.toggleViewAction()
        self._action_toggle_parameters.setText("&Parameters Dock")
        view_menu.addAction(self._action_toggle_parameters)
        self._action_toggle_dashboard = self._dashboard_dock.toggleViewAction()
        self._action_toggle_dashboard.setText("&Dashboard Dock")
        view_menu.addAction(self._action_toggle_dashboard)
        self._action_toggle_telemetry = self._telemetry_dock.toggleViewAction()
        self._action_toggle_telemetry.setText("&Telemetry Dock")
        view_menu.addAction(self._action_toggle_telemetry)
        view_menu.addSeparator()
        self._action_reset_layout = view_menu.addAction("Reset &Layout")

        help_menu = menu_bar.addMenu("&Help")
        self._action_about = help_menu.addAction("&About Quasar")

    def _build_status_bar(self) -> None:
        status_bar = self.statusBar()
        self._status_label = QLabel("Idle")
        self._topology_label = QLabel("Nodes: 0 | Links: 0")
        self._clock_label = QLabel("t = 0.000 s")
        status_bar.addWidget(self._status_label)
        status_bar.addPermanentWidget(self._topology_label)
        status_bar.addPermanentWidget(self._clock_label)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_menu_actions(self) -> None:
        self._action_exit.triggered.connect(self.close)
        self._action_new_scenario.triggered.connect(self.scenario_new_requested.emit)
        self._action_open_scenario.triggered.connect(self._on_open_scenario)
        self._action_save_scenario_as.triggered.connect(self._on_save_scenario_as)
        self._action_sim_start.triggered.connect(self.simulation_run_requested.emit)
        self._action_sim_step.triggered.connect(self.simulation_step_requested.emit)
        self._action_sim_pause.triggered.connect(self.simulation_pause_requested.emit)
        self._action_sim_stop.triggered.connect(self.simulation_stop_requested.emit)
        self._action_sim_reset.triggered.connect(self.simulation_reset_requested.emit)
        self._action_reset_layout.triggered.connect(self._reset_layout)
        self._action_reset_layout.triggered.connect(self.layout_reset_requested.emit)

    def _on_add_node_requested(self, x: float, y: float) -> None:
        self._topology_model.add_node(f"node_{uuid4().hex[:6]}", x, y, node_type="memory_node")

    def _on_topology_changed(self) -> None:
        n_nodes = len(self._topology_model.node_ids())
        n_links = len(self._topology_model.link_ids())
        self._topology_label.setText(f"Nodes: {n_nodes} | Links: {n_links}")

    def _on_config_changed(self, engine_id: str, params: dict[str, object]) -> None:
        # TODO(prompt 14): wire this through to TwinOrchestrator.update_engine_config.
        print(f"Config changed: {engine_id} = {params}")

    def _on_open_scenario(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Scenario", "", "JSON Files (*.json)"
        )
        if path:
            self.scenario_open_requested.emit(path)

    def _on_save_scenario_as(self) -> None:
        dialog = ScenarioEditor(ScenarioConfig(), self)
        dialog.exec()

    @property
    def dashboard(self) -> DashboardWindow:
        """The dashboard window hosting all live simulation plots."""
        return self._dashboard

    @property
    def telemetry_viewer(self) -> TelemetryViewer:
        """The live telemetry viewer widget."""
        return self._telemetry_viewer

    # ------------------------------------------------------------------
    # Public state updates
    # ------------------------------------------------------------------

    def set_simulation_status(self, status: str) -> None:
        """Update the status bar's status label.

        Args:
            status: Human-readable simulation status (e.g. "Idle",
                "Running", "Paused", "Stopped").
        """
        self._status_label.setText(status)

    def update_clock(self, t: float) -> None:
        """Update the status bar's simulation clock label.

        Args:
            t: Current simulation time [s].
        """
        self._clock_label.setText(f"t = {t:.3f} s")

    def _reset_layout(self) -> None:
        """Restore the default dock positions and sizes."""
        self._parameter_dock.setFloating(False)
        self._dashboard_dock.setFloating(False)
        self._telemetry_dock.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._parameter_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dashboard_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._telemetry_dock)
        self._parameter_dock.show()
        self._dashboard_dock.show()
        self._telemetry_dock.show()
