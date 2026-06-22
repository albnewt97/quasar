"""Pydantic v2 scenario configuration models (§12 JSON scenario format).

Converts the user-facing JSON scenario format into the frozen dataclasses
consumed by the simulation engine.  This is the only module where the
JSON scenario format and the engine's typed configs meet.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field

from qndt.core.orchestrator import LinkConfig, NodeConfig, SimulationConfig
from qndt.physics.kernels import (
    ExponentialKernel,
    GaussianKernel,
    LorentzianKernel,
    MemoryKernel,
)
from qndt.physics.raman import ClassicalChannelSpec, FiberParams

if TYPE_CHECKING:
    from qndt.core.orchestrator import TwinOrchestrator


# Expected shape of the sensitivity matrix S (rows × cols).
# Rows: one per Pauli axis (p_x, p_y, p_z) — a physics invariant.
# Cols: one per environmental dimension (T °C, seismic m/s², wind N) — matches the
#       synthetic telemetry source and S_SMF28_DEFAULT in telemetry/calibration.py.
_SENSITIVITY_N_ROWS: int = 3
_SENSITIVITY_N_COLS: int = 3


def validate_sensitivity_matrix(slist: list[list[float]]) -> np.ndarray:
    """Validate and convert a sensitivity list to a ({rows}×{cols}) ndarray.

    Checks, in order:
    1. Exactly ``_SENSITIVITY_N_ROWS`` rows.
    2. At least one column (M ≥ 1).
    3. All rows the same length (not ragged).
    4. Exactly ``_SENSITIVITY_N_COLS`` columns.
    5. All values numeric.

    Args:
        slist: A ``_SENSITIVITY_N_ROWS``-row list of equal-length numeric lists.

    Returns:
        A float64 ndarray of shape
        (``_SENSITIVITY_N_ROWS``, ``_SENSITIVITY_N_COLS``).

    Raises:
        ValueError: On wrong row count, empty columns, ragged rows, wrong
            column count, or non-numeric values.
    """
    if len(slist) != _SENSITIVITY_N_ROWS:
        raise ValueError(
            f"Sensitivity matrix must have exactly {_SENSITIVITY_N_ROWS} rows "
            f"(p_x, p_y, p_z); got {len(slist)} row(s)"
        )
    n_cols = len(slist[0]) if slist[0] is not None else 0
    if n_cols == 0:
        raise ValueError(
            "Sensitivity matrix must have at least 1 column (M ≥ 1); "
            "all rows are empty"
        )
    # Ragged check before column-count check so ragged input gets a clear message.
    for i, row in enumerate(slist):
        if len(row) != n_cols:
            raise ValueError(
                f"Sensitivity matrix is ragged: row 0 has {n_cols} column(s), "
                f"row {i} has {len(row)}"
            )
    if n_cols != _SENSITIVITY_N_COLS:
        raise ValueError(
            f"Sensitivity matrix must have exactly {_SENSITIVITY_N_COLS} columns "
            f"(one per environmental dimension T/seismic/wind); got {n_cols}"
        )
    try:
        return np.array(slist, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Sensitivity matrix contains non-numeric values: {exc}"
        ) from exc


class FiberParamsModel(BaseModel):
    """Pydantic model for fiber span physical parameters (§9.3).

    Args:
        length_km: Fiber span length in km.
        attenuation_db_per_km: Power loss coefficient in dB/km.
        eta_detector: Single-photon detector efficiency in (0, 1].
        t_opt: Optical transmission of filter + coupling in (0, 1].
        p_dc: Intrinsic dark-count probability per gate in (0, 1].
    """

    length_km: float = 25.0
    attenuation_db_per_km: float = 0.2
    eta_detector: float = 0.8
    t_opt: float = 0.5
    p_dc: float = 1e-5

    def to_fiber_params(self) -> FiberParams:
        """Convert to the frozen ``FiberParams`` dataclass used by physics."""
        return FiberParams(
            length_km=self.length_km,
            attenuation_db_per_km=self.attenuation_db_per_km,
            eta_detector=self.eta_detector,
            t_opt=self.t_opt,
            p_dc=self.p_dc,
        )


class LinkConfigModel(BaseModel):
    """Pydantic model for a single fiber link (§12 scenario format).

    Args:
        link_id: Unique link identifier.
        source_node: Node id at the transmitting end.
        dest_node: Node id at the receiving end.
        lambda_q_nm: Quantum channel wavelength in nm.
        gate_width_s: Quantum gate duration in seconds.
        qubit_index: Index of the qubit representing this link's state.
        fiber: Fiber physical parameters.
        classical_channels: Raw WDM channel specs (engine-format dicts).
    """

    link_id: str
    source_node: str
    dest_node: str
    lambda_q_nm: float = 1550.0
    gate_width_s: float = 1e-9
    qubit_index: int
    fiber: FiberParamsModel = Field(default_factory=FiberParamsModel)
    classical_channels: list[dict[str, object]] = Field(default_factory=list)


class NodeConfigModel(BaseModel):
    """Pydantic model for a single quantum memory node (§12 scenario format).

    Args:
        node_id: Unique node identifier.
        qubit_index: Index of the qubit representing this node's memory.
        t2_nominal: Initial T2 coherence time at zero operations [s].
        wear_const_nc: Characteristic operation count for 1/e T2 decay.
        calib_drift_rate: Gate overrotation drift rate [rad/s].
    """

    node_id: str
    qubit_index: int
    t2_nominal: float = 1.0
    wear_const_nc: float = 1e6
    calib_drift_rate: float = 1e-6


class KernelModel(BaseModel):
    """Pydantic model selecting and parameterising a memory kernel (§12).

    Args:
        type: Kernel family to construct.
        tau_x: X-axis decay time constant [s] (exponential kernel).
        tau_y: Y-axis decay time constant [s] (exponential kernel).
        tau_z: Z-axis decay time constant [s] (exponential kernel).
        gamma: Damping half-width [Hz] (Lorentzian kernel).
        omega_0: Resonance frequency [Hz] (Lorentzian kernel).
        sigma: Gaussian width [s] (Gaussian kernel).
    """

    type: Literal["exponential", "lorentzian", "gaussian"] = "exponential"
    tau_x: float = 30.0
    tau_y: float = 30.0
    tau_z: float = 120.0
    gamma: float = 0.1
    omega_0: float = 1.0
    sigma: float = 10.0

    def to_kernel(self) -> MemoryKernel:
        """Construct the concrete ``MemoryKernel`` selected by ``type``.

        Raises:
            ValueError: If ``type`` is not a recognised kernel family.
        """
        if self.type == "exponential":
            return ExponentialKernel(
                tau_x=self.tau_x, tau_y=self.tau_y, tau_z=self.tau_z
            )
        if self.type == "lorentzian":
            return LorentzianKernel(gamma=self.gamma, omega_0=self.omega_0)
        if self.type == "gaussian":
            return GaussianKernel(sigma=self.sigma)
        raise ValueError(f"Unknown kernel type: {self.type!r}")


class ScenarioConfig(BaseModel):
    """Top-level scenario configuration (§12 JSON scenario format).

    Args:
        scenario_name: Human-readable scenario name.
        nodes: Quantum memory nodes in the topology.
        links: Fiber links in the topology.
        kernel: Memory kernel selection for the telemetry engine.
        sensitivity: 3×M sensitivity matrix S overriding ``S_SMF28_DEFAULT``
            (§5.3).  ``None`` means use the illustrative SMF-28 default (uncalibrated).
        coexistence_channels: Global list of active classical WDM channels
            (applied to all links).  When non-empty, overrides per-link
            ``classical_channels``.  Each entry must have ``channel_id``,
            ``lambda_c_nm``, ``launch_power_mw``, and ``active`` keys.
        duration_s: Total simulated duration [s].
        dt_s: Simulation time step [s].
        chi_max: Maximum MPDO bond dimension.
        kappa_max: Maximum Kraus bond dimension.
    """

    scenario_name: str = "Unnamed Scenario"
    nodes: list[NodeConfigModel] = Field(default_factory=list)
    links: list[LinkConfigModel] = Field(default_factory=list)
    kernel: KernelModel = Field(default_factory=KernelModel)
    sensitivity: list[list[float]] | None = None
    coexistence_channels: list[dict[str, object]] = Field(default_factory=list)
    duration_s: float = 10.0
    dt_s: float = 0.1
    chi_max: int = 4
    kappa_max: int = 8

    def to_simulation_config(self) -> SimulationConfig:
        """Convert to the frozen ``SimulationConfig`` consumed by the engine."""
        links = tuple(
            LinkConfig(
                link_id=link.link_id,
                source_node=link.source_node,
                dest_node=link.dest_node,
                lambda_q_nm=link.lambda_q_nm,
                gate_width_s=link.gate_width_s,
                qubit_index=link.qubit_index,
            )
            for link in self.links
        )
        nodes = tuple(
            NodeConfig(node_id=node.node_id, qubit_index=node.qubit_index)
            for node in self.nodes
        )
        return SimulationConfig(
            links=links, nodes=nodes, duration_s=self.duration_s, dt_s=self.dt_s
        )

    @classmethod
    def from_json_file(cls, path: str) -> ScenarioConfig:
        """Load and validate a ``ScenarioConfig`` from a JSON file.

        Args:
            path: Filesystem path to the scenario JSON file.
        """
        with open(path) as fh:
            data = json.load(fh)
        return cls.model_validate(data)

    def to_json_file(self, path: str) -> None:
        """Serialise this scenario to a JSON file.

        Args:
            path: Destination filesystem path.
        """
        with open(path, "w") as fh:
            fh.write(self.model_dump_json(indent=2))

    def build_orchestrator(self) -> TwinOrchestrator:
        """Build a fully-wired ``TwinOrchestrator`` from this scenario.

        Reads kernel type and parameters, sensitivity matrix (or falls back to
        ``S_SMF28_DEFAULT``), fiber params from the first link, and classical
        WDM channels from ``coexistence_channels`` when non-empty, otherwise
        from per-link ``classical_channels``.

        Returns:
            A ready-to-step ``TwinOrchestrator``.

        Raises:
            ValueError: If ``self.nodes`` is empty.
        """
        from qndt.core.orchestrator import TwinOrchestrator  # local: avoids circular import

        if not self.nodes:
            raise ValueError("ScenarioConfig.build_orchestrator() requires at least one node.")

        n_qubits = max(node.qubit_index for node in self.nodes) + 1
        link_configs = [
            LinkConfig(
                link_id=link.link_id,
                source_node=link.source_node,
                dest_node=link.dest_node,
                lambda_q_nm=link.lambda_q_nm,
                gate_width_s=link.gate_width_s,
                qubit_index=link.qubit_index,
            )
            for link in self.links
        ]
        node_configs = [
            NodeConfig(node_id=node.node_id, qubit_index=node.qubit_index)
            for node in self.nodes
        ]
        node_aging: dict[str, dict[str, float]] = {
            node.node_id: {
                "t2_nominal": node.t2_nominal,
                "wear_const_nc": node.wear_const_nc,
            }
            for node in self.nodes
        }

        kernel = self.kernel.to_kernel()
        sensitivity: np.ndarray | None = (
            validate_sensitivity_matrix(self.sensitivity)
            if self.sensitivity is not None
            else None
        )
        fiber: FiberParams | None = (
            self.links[0].fiber.to_fiber_params() if self.links else None
        )

        raw_channels: list[dict[str, object]] = self.coexistence_channels or [
            ch for link in self.links for ch in link.classical_channels
        ]
        wdm: list[ClassicalChannelSpec] = []
        for ch in raw_channels:
            if not ch.get("active", True):
                continue
            try:
                wdm.append(
                    ClassicalChannelSpec(
                        channel_id=str(ch.get("channel_id", "")),
                        lambda_c_nm=float(ch["lambda_c_nm"]),  # type: ignore[arg-type]
                        launch_power_mw=float(ch["launch_power_mw"]),  # type: ignore[arg-type]
                    )
                )
            except (KeyError, ValueError, TypeError):
                pass

        return TwinOrchestrator.build_simple(
            n_qubits=n_qubits,
            link_configs=link_configs,
            node_configs=node_configs,
            duration_s=self.duration_s,
            dt_s=self.dt_s,
            chi_max=self.chi_max,
            sensitivity=sensitivity,
            kernel=kernel,
            fiber=fiber,
            wdm_channels=wdm or None,
            node_aging_overrides=node_aging or None,
        )
