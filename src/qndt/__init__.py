"""qndt — Quantum Network Digital Twin.

Public API surface for the core simulation engine.
"""
from __future__ import annotations

from qndt.control_plane.async_plane import (
    AsynchronousControlPlane,
    JitterModel,
    PacketResult,
)
from qndt.control_plane.load import ClassicalLoad, WDMLoadTracker
from qndt.control_plane.routing import (
    LoopDetector,
    NetworkGraph,
    RouteNotFoundError,
    RoutingLoop,
)
from qndt.core.bus import NoiseBus, SimulationEvent
from qndt.core.composer import ChannelComposer, NoiseContributor
from qndt.core.context import OpContext, PauliRateVector
from qndt.io.adapters import AdapterRegistry, DataFrameAdapter
from qndt.physics.aging import DeviceAgingModel
from qndt.physics.channels import (
    compose_ptms,
    dephasing_ptm,
    depolarising_ptm,
    ptm_fidelity,
    ptm_to_pauli_rates,
    validate_ptm,
)
from qndt.physics.kernels import (
    ExponentialKernel,
    GaussianKernel,
    LorentzianKernel,
    MemoryKernel,
)
from qndt.physics.master_equation import CanonicalRates, RHPWitness, TCLSolver
from qndt.physics.raman import (
    ClassicalChannelSpec,
    CoexistenceNoiseEngine,
    FiberParams,
    RamanProfile,
)
from qndt.quantum.backends.quimb_adapter import MPDOConfig, QuimbAdapter
from qndt.quantum.tracker import SimulationStep, TensorStateTracker
from qndt.telemetry.calibration import (
    CalibrationDataset,
    SensitivityFitter,
    smf28_calibration,
)
from qndt.telemetry.engine import EnvironmentalTelemetryEngine
from qndt.telemetry.resampler import TelemetryResampler
from qndt.telemetry.sources import (
    CSVReplaySource,
    JSONStreamSource,
    SyntheticTelemetrySource,
    TelemetrySample,
    TelemetrySource,
)

__version__ = "0.2.0"

__all__ = [
    # core
    "ChannelComposer",
    "NoiseBus",
    "NoiseContributor",
    "OpContext",
    "PauliRateVector",
    "SimulationEvent",
    # physics.channels
    "compose_ptms",
    "dephasing_ptm",
    "depolarising_ptm",
    "ptm_fidelity",
    "ptm_to_pauli_rates",
    "validate_ptm",
    # physics.kernels
    "ExponentialKernel",
    "GaussianKernel",
    "LorentzianKernel",
    "MemoryKernel",
    # physics.raman
    "ClassicalChannelSpec",
    "CoexistenceNoiseEngine",
    "FiberParams",
    "RamanProfile",
    # physics.master_equation
    "CanonicalRates",
    "RHPWitness",
    "TCLSolver",
    # physics.aging
    "DeviceAgingModel",
    # control_plane
    "AsynchronousControlPlane",
    "ClassicalLoad",
    "JitterModel",
    "LoopDetector",
    "NetworkGraph",
    "PacketResult",
    "RouteNotFoundError",
    "RoutingLoop",
    "WDMLoadTracker",
    # telemetry
    "CSVReplaySource",
    "EnvironmentalTelemetryEngine",
    "JSONStreamSource",
    "SyntheticTelemetrySource",
    "TelemetryResampler",
    "TelemetrySample",
    "TelemetrySource",
    # telemetry.calibration
    "CalibrationDataset",
    "SensitivityFitter",
    "smf28_calibration",
    # io.adapters
    "AdapterRegistry",
    "DataFrameAdapter",
    # quantum
    "MPDOConfig",
    "QuimbAdapter",
    "SimulationStep",
    "TensorStateTracker",
]
