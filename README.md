# Quasar — Quantum Network Digital Twin

**A non-Markovian, telemetry-driven simulator for quantum internet deployment**

[![CI](https://github.com/albnewt97/quasar/actions/workflows/ci.yml/badge.svg)](https://github.com/albnewt97/quasar/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-200%2B-brightgreen.svg)](tests/)

**Full physics model & governing equations (§§2–8, eqs 1–20, 16 references) → [docs/physics/README.md](docs/physics/README.md)**

---

## Overview

Quasar is a production-grade Quantum Network Digital Twin (QNDT) simulator that models quantum communication infrastructure as a time-varying, telemetry-driven cyber-physical system. Rather than assigning a fixed depolarising rate at initialisation, Quasar composes the effective noise channel on every fiber link at each simulation step from four independently-evolving contributors: a live environmental telemetry stream convolved through a non-Markovian memory kernel, spontaneous Raman scattering from co-existing classical WDM traffic, exponential device aging driven by cumulative operation count, and classical control-plane congestion feeding back directly into qubit dephasing and Raman noise floor. The result is a simulator whose noise model evolves continuously over the operational lifetime of the network, not one frozen at setup time.

The physics is backed by a tensor-network (MPDO/LPDO) state tracker with bounded bond dimension rather than a dense 2ⁿ density matrix. Channel composition follows the Hadamard product of diagonal Pauli Transfer Matrices — an operation that is exact for Pauli channels and O(1) in the number of noise contributors regardless of how many engines are active. An online RHP non-Markovianity witness is computed continuously and surfaced in the dashboard with sign-change markers, giving the operator a live view of information backflow on each link.

Quasar targets research groups and engineering teams deploying or planning quantum repeater networks who need a simulator that reflects the conditions a real fiber link experiences: temperature fluctuations, seismic vibration, wind-induced stress, co-existing classical traffic, incremental memory degradation, and classical-signalling latency. It is categorically distinct from NetSquid, SeQUeNCe, and QuISP in five structural ways described below.

---

## Key features

- **Non-Markovian telemetry-driven channels**: environmental data (temperature, seismic acceleration, wind force) is convolved through a configurable memory kernel (exponential or Lorentzian) to produce time-varying Pauli rate vectors; an online RHP witness detects and quantifies information backflow.
- **Classical WDM co-existence engine**: spontaneous Raman scattering from classical channels propagating on the same fiber is computed from the SpRS profile of SMF-28, producing a wavelength-, power-, and direction-dependent dark-count contribution that feeds into the effective PTM.
- **Stateful device aging**: each quantum memory node tracks its cumulative operation count; T2 degrades as T₂(N) = T₂₀ · exp(−N/Nₒ) and gate overrotation drifts at a calibrated rate; per-node parameters are configurable independently.
- **Classical–quantum control-plane coupling**: classical packet routing congestion and jitter produce an `induced_idle` signal that simultaneously increases T2 dephasing on the affected memory and raises the Raman noise floor on the shared fiber — the two layers are not independent.
- **Tensor-network reduced-order backend**: quantum state is tracked as an MPDO/LPDO with bounded bond dimension χ_max and Kraus rank κ_max; bond growth is reported as a live quality-of-service signal rather than being hidden as an implementation detail.

---

## Installation

**Prerequisites**: Python 3.10 or later, pip.

```bash
git clone https://github.com/albnewt97/quasar.git
cd quasar
pip install -e ".[dev]"
```

Runtime dependencies (installed automatically): NumPy, SciPy, quimb, anyio, pydantic, PySide6, pyqtgraph, NetworkX, pandas, httpx, matplotlib.

---

## Requirements

*All versions are sourced directly from `pyproject.toml`; no separate `requirements.txt` exists.*

**Python**: `>=3.10`

**Runtime dependencies** (minimum versions):

| Package | Minimum version | Purpose |
|---|---|---|
| `numpy` | 1.26 | PTM algebra, kernel convolution |
| `scipy` | 1.12 | Signal processing, SpRS integration |
| `quimb` | 1.7 | MPDO/LPDO tensor-network state tracker |
| `anyio` | 4.3 | Async telemetry ingestion |
| `pydantic` | 2.6 | Scenario config and IO validation |
| `PySide6` | 6.7 | GUI (Qt6 bindings) |
| `pyqtgraph` | 0.13 | Live dashboard plots |
| `networkx` | 3.3 | Topology graph and Dijkstra routing |
| `pandas` | 2.2 | CSV telemetry ingestion |
| `httpx` | 0.27 | JSON stream telemetry source |
| `matplotlib` | 3.8 | Static example plots (`coexistence_sweep.py`) |

**Platform notes**: PySide6 requires a display or a virtual framebuffer. On headless Linux servers, install the required Qt system libraries and set `QT_QPA_PLATFORM=offscreen` before running tests or `quasar-sim` (the CI workflow installs `libgl1-mesa-glx`, `libxcb-*`, and related packages for reference). The physics engine and `quasar-sim` CLI run without a display; only the `quasar` GUI entrypoint requires one.

**Install**:

```bash
git clone https://github.com/albnewt97/quasar.git
cd quasar
pip install -e ".[dev]"   # includes pytest, ruff, mypy, hypothesis
```

---

## Quick start

The following runs a headless simulation of a two-node quantum link with synthetic telemetry and prints QBER, fidelity, and secret key rate at each step. No GUI is required.

```python
from qndt.core.orchestrator import LinkConfig, NodeConfig, TwinOrchestrator
from qndt.physics.key_rate import BB84KeyRateCalculator, KeyRateParams

# Define nodes and links
node_configs = [
    NodeConfig(node_id="Alice", qubit_index=0),
    NodeConfig(node_id="Bob",   qubit_index=1),
]
link_configs = [
    LinkConfig(
        link_id="alice_bob",
        source_node="Alice",
        dest_node="Bob",
        lambda_q_nm=1550.0,
        gate_width_s=1e-9,
        qubit_index=0,
    ),
]

# Build a fully-wired simulator with sensible defaults
orchestrator = TwinOrchestrator.build_simple(
    n_qubits=2,
    link_configs=link_configs,
    node_configs=node_configs,
    duration_s=5.0,
    dt_s=0.1,
)

kr_calc = BB84KeyRateCalculator(KeyRateParams())

print(f"{'t':>6}  {'QBER':>8}  {'Fidelity':>10}  {'SKR (bps)':>12}  Secure")
for _ in range(50):
    for result in orchestrator.step():
        kr = kr_calc.calculate(result.qber)
        print(
            f"{result.t:6.2f}  {result.qber:8.4f}  {result.fidelity:10.4f}  "
            f"{kr.secret_key_rate_bps:12.3e}  {kr.is_positive}"
        )
```

See [`examples/live_fiber_twin.py`](examples/live_fiber_twin.py) for a fuller four-node repeater chain with WDM co-existence, classical packet routing, and summary statistics, and [`examples/coexistence_sweep.py`](examples/coexistence_sweep.py) for a QBER-vs-classical-channel-count parameter sweep.

---

## Architecture

```
TELEMETRY SOURCE (CSV replay / live JSON / MQTT)
        │  ingest(TelemetrySample)
        ▼
  TelemetryResampler  ─────────────────────────────────────┐
        │  .window(link_id, t)                              │
        ▼                                                   │
  EnvironmentalTelemetryEngine                              │
  [S · K(τ) convolution → PauliRateVector → PTM]           │
        │  .ptm(ctx)                                        │
        ▼                                                   │
  ────────────────────────────────                          │
  ChannelComposer.effective_ptm(ctx)  ◄── DeviceAgingModel.ptm(ctx)
        ▲                             ◄── CoexistenceNoiseEngine.ptm(ctx)
        │                                        │
        │                            AsynchronousControlPlane
        │                            .induced_idle() / WDM load
        │
  TwinOrchestrator (master event kernel)
        │  tracker.apply_channel(qubit, ptm)
        ▼
  TensorStateTracker (MPDO/LPDO, bond dim χ)
        │  .fidelity() / .measure()
        ▼
  SimulationResult  →  NoiseBus  →  GUI / headless output
```

**Module summary**

| Package | Responsibility |
|---|---|
| `qndt.core` | `OpContext`, `ChannelComposer` (Hadamard PTM product), `NoiseBus`, `TwinOrchestrator` |
| `qndt.physics` | PTM algebra, memory kernels, Raman SpRS, device aging, BB84 key rate |
| `qndt.telemetry` | Telemetry sources (CSV, JSON stream), resampler, environmental engine, calibration |
| `qndt.control_plane` | Dijkstra routing, WDM load tracker, jitter model, `AsynchronousControlPlane` |
| `qndt.quantum` | `TensorStateTracker` (quimb MPDO/LPDO), protocol helpers |
| `qndt.io` | Pydantic scenario config, external adapter registry |
| `qndt.gui` | PySide6/pyqtgraph application — topology editor, parameter panels, live dashboard |

The GUI layer imports from `qndt.*` but nothing in `qndt.*` (outside `gui/`) imports from `qndt.gui.*`. Qt types never appear in the physics, telemetry, control-plane, or quantum packages.

---

## Running the GUI

```bash
quasar                                      # launch the interactive GUI
quasar-sim --scenario examples/metro_demo.json       # headless simulation
```

**First-run example** — for a visually active first run, load the bundled illustrative scenario:

1. Launch the GUI: `quasar`
2. **File → Load Scenario** → select `examples/metro_demo.json`
3. Click **Run** — all dashboard plots show non-trivial dynamics within seconds

> The sensitivity matrix in `metro_demo.json` is scaled for visual clarity, **not** calibrated against any measured fiber. It is an illustrative example only.

The GUI provides:

- **Topology editor**: drag-and-drop node placement (`memory_node`, `bsm_node`, `source_node`, `detector`); click-to-draw fiber links; spring auto-layout; right-click context menus. Double-click a node to configure per-node aging parameters.
- **Parameter docks**: channel wavelength and fiber parameters; telemetry source (CSV replay or live URL); Raman WDM channel table; device aging T2/Nₒ/drift; control-plane routing and jitter settings.
- **Live dashboard**: QBER per link (with computed BB84 security threshold ~9.8%, using GLLP with f_ec = 1.16), fidelity per node (with F < 0.5 shaded), Raman dark-count rate vs. time, RHP non-Markovianity witness (with backflow sign-change markers), T2 aging curves with theoretical overlay, network heatmap (link colour = live fidelity). All plots support crosshair hover readout, zoom/pan, legend toggle, and PNG export. The default S (sensitivity) matrix is an illustrative uncalibrated starting point; replace it with measured values for a calibrated twin.
- **Scenario editor**: save and reload simulation configurations as JSON; scenario diff view.

---

## Documentation

The architecture reference, physics equations, inviolable design laws, and GUI specification live in [`docs/architecture.md`](docs/architecture.md). This is the primary reference for contributors.

Full documentation is also built with MkDocs Material:

```bash
pip install mkdocs-material
mkdocs serve          # http://localhost:8000
```

Source in [`docs/`](docs/). Physics derivations with LaTeX are in [`docs/physics/`](docs/physics/).

---

## Testing

```bash
# Full test suite (215 tests)
QT_QPA_PLATFORM=offscreen pytest tests/ -v

# Physics regression suite only
QT_QPA_PLATFORM=offscreen pytest tests/ -m physics_regression -v --tb=short

# With coverage report
QT_QPA_PLATFORM=offscreen pytest tests/ --cov=src/qndt --cov-report=term-missing
```

The `physics_regression` marker covers analytic-limit checks: PTM Hadamard composition exactness, Raman SpRS vs. Eraerds et al. 2010, RHP witness sign correctness, T2 wear-curve formula, and BB84 GLLP threshold. GUI tests run headless via the `offscreen` Qt platform plugin.

---

## Citation

If you use Quasar in research, please cite it. The CITATION.cff file contains full metadata; the BibTeX entry is:

```bibtex
@software{quasar_qndt,
  title    = {Quasar: A Telemetry-Driven Quantum Network Digital Twin},
  author   = {Quasar Contributors},
  year     = {2026},
  license  = {Apache-2.0},
  url      = {https://github.com/quasar-qndt/quasar}
}
```

---

## Acknowledgements

Quasar was developed with AI-assisted programming (Anthropic Claude) under human direction and review. All physics was validated against the cited literature. Key references: Nielsen & Chuang (2010) for T2 dephasing; Eraerds et al., *New J. Phys.* 12, 063027 (2010) for the SpRS Raman model; Breuer & Petruccione (2002) and Rivas, Huelga & Plenio, *Rep. Prog. Phys.* 77, 094001 (2014) for the non-Markovian formalism and RHP witness; Lo, Ma & Chen, *Science* 308, 1911 (2005) for the GLLP BB84 key rate.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the `NoiseContributor` extension pattern, code style, and the PR checklist.

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
