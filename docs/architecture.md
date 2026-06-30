# Quasar Architecture

## Quantum Network Digital Twin — Simulator for Non-Markovian, Telemetry-Driven Quantum Internet Deployment

---

## 1. Project Identity

| Field | Value |
|---|---|
| Project name | Quasar |
| Python package | `qndt` |
| Source root | `src/qndt/` |
| GUI framework | PySide6 (Qt6) + pyqtgraph |
| Min Python | 3.10 |
| License | Apache-2.0 |

Quasar is a production-grade Quantum Network Digital Twin simulator. It models real-world quantum communication infrastructure as a time-varying, telemetry-driven cyber-physical system — not as an idealised textbook channel. It is categorically distinct from NetSquid, SeQUeNCe, and QuISP in five ways that are encoded structurally in the architecture (see §6).

---

## 2. Full Directory Layout

```
Quasar/
├── CLAUDE.md                        ← minimal tooling pointer (bulk here)
├── pyproject.toml                   ← PEP 621, single source of truth for deps
├── README.md
├── CONTRIBUTING.md
├── CITATION.cff
├── LICENSE
│
├── .github/
│   └── workflows/
│       ├── ci.yml                   ← ruff + mypy + pytest + coverage
│       └── physics-regression.yml  ← analytic limit regression suite
│
├── src/
│   └── qndt/
│       ├── __init__.py              ← exports version, public API surface
│       │
│       ├── core/                    ← protocol layer, zero physics here
│       │   ├── __init__.py
│       │   ├── context.py           ← OpContext, PauliRateVector (frozen dataclasses)
│       │   ├── composer.py          ← NoiseContributor protocol + ChannelComposer
│       │   ├── bus.py               ← pub/sub NoiseBus for inter-engine events
│       │   └── orchestrator.py      ← TwinOrchestrator, dual-clock event kernel
│       │
│       ├── physics/                 ← all equations, no I/O, no Qt
│       │   ├── __init__.py
│       │   ├── channels.py          ← PTM algebra, Pauli channel composition
│       │   ├── kernels.py           ← MemoryKernel base + Exponential/Lorentzian impls
│       │   ├── master_equation.py   ← TCL solver, RHP non-Markovianity witness
│       │   ├── raman.py             ← SpRS profile, fwd/bwd power, dark-count map
│       │   └── aging.py             ← DeviceAgingModel, duty-cycle wear, T2 degradation
│       │
│       ├── telemetry/               ← environmental data ingestion
│       │   ├── __init__.py
│       │   ├── sources.py           ← TelemetrySource protocol, CSVReplaySource, JSONStreamSource
│       │   ├── resampler.py         ← clock reconciliation, dropout/stale handling
│       │   ├── engine.py            ← EnvironmentalTelemetryEngine (kernel convolution + RHP)
│       │   └── calibration.py       ← sensitivity-matrix fitting; illustrative SMF-28 defaults (uncalibrated)
│       │
│       ├── control_plane/           ← classical network simulation (async)
│       │   ├── __init__.py
│       │   ├── async_plane.py       ← AsynchronousControlPlane (anyio), routing, jitter
│       │   ├── routing.py           ← Dijkstra + loop detection + retransmit logic
│       │   └── load.py              ← WDM occupancy tracker → CoexistenceNoiseEngine feed
│       │
│       ├── quantum/                 ← state tracking, protocols
│       │   ├── __init__.py
│       │   ├── tracker.py           ← TensorStateTracker (quimb MPDO/LPDO backend)
│       │   ├── backends/
│       │   │   ├── __init__.py
│       │   │   └── quimb_adapter.py ← concrete quimb binding, chi/kappa truncation
│       │   └── protocols.py         ← BB84, E91, entanglement swapping, purification
│       │
│       ├── io/                      ← config parsing, serialisation
│       │   ├── __init__.py
│       │   ├── config.py            ← pydantic v2 scenario config models
│       │   └── adapters.py          ← external telemetry format adapters (MQTT, REST)
│       │
│       └── gui/                     ← PySide6 application (imports qndt.* but never vice-versa)
│           ├── __init__.py
│           ├── app.py               ← QApplication entry point, main window assembly
│           ├── main_window.py       ← QMainWindow, dockable panel layout
│           │
│           ├── topology/            ← network graph editor
│           │   ├── __init__.py
│           │   ├── canvas.py        ← TopologyCanvas (QGraphicsScene/View)
│           │   ├── node_item.py     ← QuantumNodeItem (QGraphicsItem, drag/drop)
│           │   ├── link_item.py     ← FiberLinkItem (QGraphicsPathItem, animated)
│           │   └── topology_model.py← Qt model wrapping NetworkGraph (no Qt in NetworkGraph)
│           │
│           ├── panels/              ← parameter input forms
│           │   ├── __init__.py
│           │   ├── telemetry_panel.py   ← CSV/JSON source selector, sensitivity matrix editor
│           │   ├── channel_panel.py     ← per-link: wavelength, fiber params, kernel chooser
│           │   ├── coexistence_panel.py ← WDM channel config, Raman cross-section table
│           │   ├── aging_panel.py       ← T2 nominal, wear constant, calibration drift
│           │   └── control_plane_panel.py ← topology routing config, jitter model params
│           │
│           ├── dashboard/           ← live simulation visualisation
│           │   ├── __init__.py
│           │   ├── dashboard_window.py  ← tabbed dashboard container
│           │   ├── qber_plot.py         ← real-time QBER per link (pyqtgraph PlotWidget)
│           │   ├── fidelity_plot.py     ← per-node memory process fidelity decay curves
│           │   ├── raman_plot.py        ← Raman noise floor vs classical traffic load
│           │   ├── nonmarkov_plot.py    ← RHP witness timeline, sign-change highlighting
│           │   ├── aging_plot.py        ← T2 degradation per node over operational time
│           │   └── network_heatmap.py   ← topology overlay: link colour = live fidelity
│           │
│           ├── telemetry_viewer.py  ← live scrolling telemetry feed (temperature, seismic, wind)
│           ├── scenario_editor.py   ← save/load scenario JSON, scenario diff tool
│           └── styles/
│               ├── dark.qss         ← Qt stylesheet: dark scientific instrument aesthetic
│               └── icons/           ← SVG icons for nodes, fiber types, protocol badges
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  ← shared fixtures (mock telemetry, small topology)
│   ├── test_composer.py             ← PTM Hadamard composition exactness
│   ├── test_channels.py             ← PTM == direct Kraus on pure Pauli channels
│   ├── test_raman.py                ← SpRS vs Eraerds et al. analytic limit
│   ├── test_nonmarkovianity.py      ← RHP witness sign correctness
│   ├── test_resampler.py            ← dropout/stale/interp edge cases
│   ├── test_aging.py                ← T2 wear curve monotonicity
│   ├── test_orchestrator.py         ← full single-link integration: telemetry → QBER
│   └── test_gui/
│       ├── conftest.py              ← QApplication fixture (headless via offscreen platform)
│       ├── test_canvas.py           ← node add/remove, link draw
│       └── test_panels.py           ← parameter binding round-trip
│
├── benchmarks/
│   ├── chi_vs_fidelity.py           ← MPDO bond-dim scaling
│   └── throughput.py                ← events/sec vs network size
│
├── examples/
│   ├── live_fiber_twin.py           ← 4-node network, CSV temperature feed, live QBER
│   ├── coexistence_sweep.py         ← QBER vs classical channel count
│   └── nonmarkovian_demo.py         ← RHP witness sign-flip visualisation
│
└── docs/
    ├── mkdocs.yml
    ├── architecture.md              ← this file
    ├── physics/                     ← LaTeX derivations rendered via mkdocs-material
    └── api/                         ← auto-generated from docstrings
```

---

## 3. Inviolable Architecture Laws

These rules encode the physical and software architecture decisions made at design time. Every contributor is expected to preserve them.

### 3.1 The NoiseContributor Protocol (Core Law)

Every physics engine that contributes noise implements **one and only one method**:

```python
class NoiseContributor(Protocol):
    def ptm(self, ctx: OpContext) -> np.ndarray: ...
    # Returns: diagonal length-4 Pauli Transfer Matrix [1, λx, λy, λz]
```

No engine implements noise contribution any other way. Not Kraus operators. Not density matrix perturbation. Not direct state mutation. Only PTM.

### 3.2 Composition Law

Channel composition is **always** the Hadamard (element-wise) product of diagonal PTMs:

```
R_eff = R_env ⊙ R_Raman ⊙ R_aging ⊙ R_idle
```

This is exact for Pauli channels and O(1) regardless of contributor count.
`ChannelComposer.effective_ptm()` is the **only site where composition is orchestrated**:
it calls each contributor's `ptm(ctx)` and delegates the element-wise algebra to
`channels.compose_ptms()`.  `channels.compose_ptms()` is a reusable PTM algebra
primitive (available to tests and future utilities), but it may only be *driven* from
`effective_ptm()` in production code — never called directly by one engine on another's
output.  No engine calls another engine's `ptm()` method directly.

### 3.3 Reference Graph (Allowed Inter-Engine Dependencies)

The graph distinguishes two kinds of import:

- **Runtime data edges** — an engine holds a reference to another engine and calls its
  methods during the simulation loop.  These are the edges the law *constrains*.
- **Assembly-time imports** — `TwinOrchestrator` is the assembly root; it necessarily
  imports all concrete engine types in order to instantiate and wire them.  These are
  *expected and documented* here, not constrained.

**Runtime data edges (constrained set):**

```
EnvironmentalTelemetryEngine    (no runtime upstream deps)
CoexistenceNoiseEngine     ──►  AsynchronousControlPlane  (runtime: raman_rate() calls
                                  manages_link() + current_load() for live WDM channel set;
                                  §3.3 B2: manages_link() used as live/static discriminant)
DeviceAgingModel               (no runtime upstream deps)
AsynchronousControlPlane        (no runtime upstream deps; activate_channel/deactivate_channel/
                                  update_channel_power typed Any to avoid reverse-direction import)
TensorStateTracker              (no runtime upstream deps)
ChannelComposer            ──►  [all NoiseContributors]   (via Protocol; no concrete-type imports)
```

**WDM load schedule (§3.3 B1):**

`TwinOrchestrator.step()` applies any due `WDMScheduleEntry` events before the link physics
loop.  Events are sorted by `t` at construction; fired when `self._t >= event.t`.  Each
event calls the appropriate delegation method on `AsynchronousControlPlane`, which forwards
to `WDMLoadTracker`.  The schedule is supplied via `ScenarioConfig.wdm_schedule` (a list of
`WDMScheduleEventModel`) or directly via `TwinOrchestrator.build_simple(wdm_schedule=...)`.

**B2 fallback rule:**

A fiber link is owned by **either** static `register_channel()` **or** the CP schedule —
not both.  Once any schedule event activates a link, `manages_link(link_id)` returns `True`
for the lifetime of the tracker, and `raman_rate()` uses the live (possibly empty) path
exclusively, ignoring any static channels.  Mixing static and scheduled channels on the same
link produces unmanaged semantics only until the first `activate` call.

No engine-to-engine runtime data edge other than the above may exist.

**Assembly-time imports (TwinOrchestrator only):**

`TwinOrchestrator` is the simulation assembly root.  It imports — at module level, to
instantiate and register — every concrete engine type:

```
TwinOrchestrator  ──►  ChannelComposer, TensorStateTracker, AsynchronousControlPlane
                  ──►  EnvironmentalTelemetryEngine, CoexistenceNoiseEngine, DeviceAgingModel
                  ──►  (value-objects/utilities: WDMLoadTracker, NetworkGraph,
                         BB84KeyRateCalculator, MemoryKernel, SyntheticTelemetrySource, …)
```

The GUI layer (`qndt.gui.*`) may import anything from `qndt.*`. Nothing in `qndt.*`
(outside `gui/`) may import from `qndt.gui.*`.

### 3.4 State Ownership Law

`TensorStateTracker` is the **only** class that stores quantum state. No other class holds a density matrix, state vector, or MPDO. If you find yourself storing quantum state elsewhere, you are wrong.

### 3.5 Telemetry Path Law

Environmental data must flow through exactly this path:

```
TelemetrySource → TelemetryResampler → EnvironmentalTelemetryEngine → ptm(ctx)
```

No engine reads raw telemetry directly. No engine polls a file or network socket directly.

### 3.6 GUI Isolation Law

The GUI is a consumer of the simulation engine, never a controller of physics. Qt types (`QObject`, `Signal`, etc.) never appear in `core/`, `physics/`, `telemetry/`, `control_plane/`, `quantum/`, or `io/`. The `lambda_q` field in `OpContext` is stored in SI metres (not nanometres); this is the single unit-conversion point between engine inputs and the nm-based physics literature. Data flows from engine → GUI via Qt Signals connected to `TwinOrchestrator` callbacks. The simulation runs in a `QThread` (or `anyio` task); the GUI runs in the Qt main thread. Cross-thread communication via `Signal`/`Slot` only.

---

## 4. Data Flow Architecture

### 4.1 Simulation Plane (runs in QThread / anyio task)

```
TELEMETRY SOURCE (CSV / live JSON / MQTT)
        │  ingest(TelemetrySample)
        ▼
  TelemetryResampler  ──────────────────────────────────┐
        │  .window(link_id, t)                           │
        ▼                                                │
  EnvironmentalTelemetryEngine                           │
  [S matrix × K(τ) convolution → PauliRateVector]       │
        │  .ptm(ctx)                                     │
        ▼                                                │
  ────────────────────────────────                       │
  ChannelComposer.effective_ptm(ctx)  ◄── DeviceAgingModel.ptm(ctx)
        ▲                             ◄── CoexistenceNoiseEngine.ptm(ctx)
        │                                       │
        │                            AsynchronousControlPlane
        │                            .current_load() / .induced_idle()
        │
  TwinOrchestrator (master event kernel)
        │  tracker.apply_channel(qubit, ptm)
        ▼
  TensorStateTracker (MPDO/LPDO, bond-dim χ)
        │  .fidelity() / .measure()
        ▼
  Simulation results (QBER, process fidelity, RHP witness, bond dims)
```

### 4.2 GUI Plane (Qt main thread)

```
TwinOrchestrator
  │  emits SimulationStep(t, results) signal
  ▼
DashboardWindow
  ├── QberPlot.update(link_id, qber)
  ├── FidelityPlot.update(node_id, fidelity)
  ├── RamanPlot.update(link_id, raman_rate)
  ├── NonMarkovPlot.update(link_id, rhp_witness)
  ├── AgingPlot.update(node_id, t2_current)
  └── NetworkHeatmap.update(topology_fidelity_map)

TopologyCanvas
  │  emits TopologyChanged(graph) signal
  ▼
TwinOrchestrator.reconfigure(graph)

ParameterPanels
  │  emit ConfigChanged(engine_id, params) signal
  ▼
TwinOrchestrator.update_engine_config(engine_id, params)
```

---

## 5. Physics Reference (Key Equations)

### 5.1 Pauli Transfer Matrix (PTM)

For a Pauli channel $\mathcal{E}(\rho) = p_I\rho + p_x\sigma_x\rho\sigma_x + p_y\sigma_y\rho\sigma_y + p_z\sigma_z\rho\sigma_z$:

```
R = diag(1, λx, λy, λz)
λx = 1 - 2(py + pz)
λy = 1 - 2(px + pz)
λz = 1 - 2(px + py)
```

Composition of two Pauli channels: `R_eff = R1 ⊙ R2` (element-wise product). This is exact, not approximate.

The process fidelity reported in the dashboard and returned from `TensorStateTracker.fidelity()` is the overlap with the target state (typically the maximally entangled Bell state), not the average gate fidelity.

### 5.2 Non-Markovian Pauli Rate Vector

```
p⃗(t) = Φ( ∫_{-∞}^{t} K(t-t') · S · E⃗(t') dt' )

K(τ) ∈ R^{3×3}  — memory kernel matrix
S ∈ R^{3×M}     — sensitivity matrix (env axis → Pauli axis)
E⃗(t) ∈ R^M     — environmental state vector [T, a_seis, F_wind, ...]
Φ               — squashing map: p_i ∈ [0, 0.5), Σp_i ≤ 1
```

Discretised: `acc += K.eval(t - t'_k) @ (S @ E'_k) * dt_k` over buffered samples.

### 5.3 SMF-28 Sensitivity Matrix (default values)

```
          T      a_seis   F_wind
S =  [ 0.0,   high,    med  ]   ← px (mode coupling: seismic, wind)
     [ 0.0,   high,    0.0  ]   ← py (mode coupling: seismic only)
     [ high,  low,     med  ]   ← pz (dephasing: thermal + slow SOP)
```

Illustrative default values live in `src/qndt/telemetry/calibration.py`. These are physically-motivated but uncalibrated; see the comments in that module for details.

### 5.4 Raman Dark Count Probability (per gate)

The SpRS cross-section uses a frequency-offset profile ρ(Δν) calibrated to the silica Raman
gain spectrum (whitepaper eq 16; Agrawal NLFO Fig. 8.1; Eraerds 2010 calibration anchor):

```
Δν           = ν_cl − ν_q  [Hz]   (> 0 Stokes, < 0 anti-Stokes)
ρ(Δν)        = ρ_peak · g(|Δν|) · A(Δν, T)

g(|Δν|)      normalised silica Raman gain shape; peak ≈ 13.2 THz, falls to 0 at ~45 THz
A(Δν, T)     Stokes: n(|Δν|, T) + 1;  anti-Stokes: n(|Δν|, T)
n(|Δν|, T)   1 / (exp(h|Δν| / kT) − 1)   [Bose–Einstein]
ρ_peak       ≈ 1.19×10⁻⁹ 1/(km·nm)   [calibrated to Eraerds β(1310→1550) = 4×10⁻¹¹]

P_fwd(λq, t) = Pc(t) · ρ(Δν) · Δλq · L · exp(-αL)
P_bwd(λq, t) = Pc(t) · ρ(Δν) · Δλq · (1 - exp(-2αL)) / (2α)

r_Raman_tot(t) = Σ_{c ∈ active(t)}  (P_fwd + P_bwd) · η_det · T_opt / (h·νq)

active(t)  = channel source selected by manages_link() (§3.3 B2 semantics):
             • CP-managed path (manages_link(link_id) → True):
               AsynchronousControlPlane.current_load(link_id, t).active_channels
               Empty list means all channels off → rate = 0.  No static fallback.
             • Static path (manages_link → False, i.e., link never activated in CP):
               CoexistenceNoiseEngine._channels (populated via register_channel()).
             Rule: a link is owned by static register_channel OR the CP schedule,
             not both.  Once any schedule event activates a link, CP-managed semantics
             apply for the lifetime of that WDMLoadTracker instance.

p_click_noise(t) = p_dc + (1 - exp(-r_Raman_tot · τ_gate))
                 ≈ p_dc + r_Raman_tot · τ_gate   [for small argument]
```

### 5.5 Device Aging

Matthiessen's rule applied to the decoherence rate (whitepaper eq 18); rate-additivity
of independent dephasing channels — Nielsen & Chuang (2010), Ch. 8.  The κD term is a
calibrated phenomenological wear coefficient, not a derived constant.

```
1/T2(D) = 1/T2_0 + κ·D               [Matthiessen rule, eq 18; D = Σ Δt_op [s], κ [s⁻²]]
ε(t)    = ε_0 + κ_drift·t            [gate overrotation drift; κ_drift [rad/s] ≠ κ]
pz_idle = 0.5 · (1 - exp(-t_idle / T2(D)))
T2 ≤ 2·T1                            [enforced: 1/T2=1/(2T1)+1/Tφ, Tφ≥0; equality = pure-T1 limit]
```

### 5.6 RHP Non-Markovianity Witness

```
N_RHP = ∫_{γk(t) < 0} |γk(t)| dt
```

Computed online. A positive value means information backflow is occurring on this link. Surfaced in `NonMarkovPlot` with sign-change markers.

### 5.7 BB84 QBER (exact Pauli channel)

For a Pauli channel with diagonal PTM `[1, λx, λy, λz]`, the per-qubit BB84 QBER is exact
(Nielsen & Chuang §12 / GLLP):

```
QBER_Z = (1 − λz) / 2     Z-basis: X or Y error flips the bit
QBER_X = (1 − λx) / 2     X-basis: Z or Y error flips the bit
QBER   = (QBER_Z + QBER_X) / 2 = (2 − λx − λz) / 4
```

`λy` does not enter QBER directly (Y errors contribute equally to Z- and X-basis errors,
captured already by the λx and λz terms).

**Physical exposure time `τ_qubit`.**  `TwinOrchestrator.step()` computes QBER from the
composed effective PTM eigenvalues scaled to the per-qubit physical exposure time `τ_qubit`
(`LinkConfig.qubit_exposure_s`) — **not** to the numerical timestep `dt_s`.  QBER is therefore
dt-invariant: `QBER(dt) = QBER(dt/2)` to machine precision for a Markovian channel.

The dt-scaled eigenvalues (`scaled_ptm`) continue to drive `TensorStateTracker.apply_channel`
(state evolution) and the aging duty-cycle accumulator `D = ∫u dt`.  Only the QBER estimate
uses the τ_qubit scale.

**`τ_qubit` regimes**

| Regime | `qubit_exposure_s` | Physical meaning | QBER aging sensitivity |
|---|---|---|---|
| Fly-by / point-to-point | propagation delay `L·n_g/c` | Time for a photon to traverse the fiber | Weak (τ_qubit ≪ T₂) |
| Quantum memory / repeater | memory hold time | Time the qubit is stored before measurement | Strong (τ_qubit ~ T₂) |

**Default**: `qubit_exposure_s` is omitted from the scenario JSON → propagation delay
`L·n_g/c` with `n_g = 1.468` (SMF-28 at 1550 nm) and `c = 2.998×10⁸ m/s`.  Override
per-link in the scenario JSON for memory-node scenarios.

`fidelity` (= global state purity `Tr(ρ²)` returned by `TensorStateTracker.apply_channel`) is
reported as a separate diagnostic for the FidelityPlot and does **not** drive the QBER calculation.

---

## 6. What Makes Quasar Distinct

These are the five structural novelties. Every implementation decision preserves them.

| # | Novelty | What other tools do | What Quasar does |
|---|---------|--------------------|--------------------|
| 1 | **Non-Markovian telemetry** | Static depolarising rate set at init | Memory-kernel convolution over live env stream; online RHP witness |
| 2 | **Co-existence engine** | Quantum channel simulated in isolation | SpRS Raman cross-talk from classical WDM channels → dark count rate → PTM contribution |
| 3 | **Device aging** | Fixed T2/gate fidelity | T2 degrades with accumulated duty cycle via Matthiessen's rule (eq 18); gate overrotation drifts |
| 4 | **Classical↔quantum coupling** | Ideal control plane, free of cost | Classical congestion/jitter → `induced_idle` → T2 dephasing AND higher Raman noise |
| 5 | **Tensor-network backend** | Dense 2^{2n} global density matrix | MPDO/LPDO with χ_max, κ_max truncation; bond dims reported as QoS signal |

---

## 7. GUI Architecture Reference

### 7.1 Technology Choices (Rationale)

| Component | Choice | Rationale |
|---|---|---|
| GUI framework | PySide6 (Qt6) | Native on Win/Mac/Linux; QGraphicsScene for graph canvas; no browser process |
| Real-time plots | pyqtgraph | Purpose-built for scientific streaming data; 10–100× faster than matplotlib for live updates |
| Graph layout | NetworkX (logical) + QGraphicsScene (visual) | Clean separation: NetworkX owns the graph topology model; Qt owns only rendering |
| Threading | QThread + anyio | Simulation runs in QThread; async I/O (telemetry sockets) via anyio inside that thread |
| Styling | QSS dark theme | Dark scientific instrument aesthetic; readable on all monitor types used in labs |

### 7.2 TopologyCanvas Behaviour

- Nodes: drag to position, double-click to open `NodePropertiesDialog`
- Links: click source node → click destination node to draw fiber link
- Right-click node: context menu → Draw Link From Here / Remove Node / Set as Source Node / Set as BSM Node / Set as Memory Node / Set as Detector
- Right-click link: context menu → Remove Link / Toggle Raman Coexistence
- During simulation: nodes pulse with colour encoding memory process fidelity (green → red)
- During simulation: links animate with particle flow rate proportional to entanglement generation rate
- Topology exported to `NetworkGraph` model on every change; never stored only in Qt items

### 7.3 Parameter Panels Binding Pattern

Each panel owns a `ConfigModel` (pydantic). When the user changes a field:
1. Panel validates via pydantic → shows inline error if invalid
2. Panel emits `config_changed: Signal(str, dict)` with `(engine_id, params_dict)`
3. `TwinOrchestrator` slot receives, calls `engine.reconfigure(params)`
4. Engine marks its cache dirty; next `ptm(ctx)` call uses new params

This means parameter changes take effect at the next quantum event, not mid-operation.

### 7.4 Dashboard Plots Specification

| Plot | X-axis | Y-axis | Update rate | Special |
|---|---|---|---|---|
| `QberPlot` | Sim time [s] | QBER [0–1] per link | Every entanglement attempt | Threshold line at 11% (BB84 security bound) |
| `FidelityPlot` | Sim time [s] | Process fidelity [0–1] per node | Every memory op | Shaded region: F < 0.5 (below classical) |
| `RamanPlot` | Sim time [s] | Dark count rate [Hz] | Every WDM change | Overlaid: intrinsic dark count baseline |
| `NonMarkovPlot` | Sim time [s] | RHP witness value | Continuous | Sign-change markers; negative = backflow |
| `AgingPlot` | Accumulated duty cycle D [s] | T2 [s] per node | Every register_op | Matthiessen curve overlay T2(D)=1/(1/T2_0+κD) |
| `NetworkHeatmap` | Topology graph | Link colour = live fidelity | 1 Hz refresh | Exported as PNG on demand |

### 7.5 Telemetry Viewer

Scrolling live view of the raw environmental stream:
- Temperature [°C] — line plot, last 60 s
- Seismic acceleration [m/s²] — line plot, last 60 s
- Wind force [N] — line plot, last 60 s
- Source indicator: CSV (replay speed multiplier slider) or Live (URL + poll rate)
- Stale data alert: yellow border if last sample > `max_gap_s` ago

---

## 8. Code Quality Standards

### 8.1 Type Annotations
- All public functions and methods: full type annotations
- `mypy --strict` must pass with zero errors
- No `Any` except at serialisation boundaries (clearly marked with `# type: ignore[misc]` + comment)

### 8.2 Dataclasses
- Value objects: `@dataclass(frozen=True, slots=True)`
- Mutable engine state: `@dataclass(slots=True)` without `frozen`
- No plain dicts for structured data — always a typed dataclass or pydantic model

### 8.3 Protocols
- All extension points use `@runtime_checkable` Protocol
- Never use ABCs for extension points that cross module boundaries
- `isinstance(obj, NoiseContributor)` must always work at runtime

### 8.4 Docstrings
- Google style throughout
- Every public class: one-line summary + Args/Returns/Raises
- Physics equations in docstrings use LaTeX in backtick blocks: `` `λx = 1 - 2(py + pz)` ``

### 8.5 Testing Requirements
- Every physics engine: regression test against known analytic limit
- Every PTM composition path: test that result equals direct Kraus calculation
- GUI tests: headless via `QT_QPA_PLATFORM=offscreen`
- Minimum coverage: 85% on `physics/`, `core/`, `telemetry/`; 60% on `gui/`

### 8.6 Naming Conventions
- Physics scalars: match paper notation (`lambda_q`, `p_dc`, `chi_max`)
- Engine methods: verb_noun (`pauli_rates`, `induced_idle`, `register_op`)
- Qt slots: `on_signal_name` (`on_config_changed`, `on_topology_changed`)
- Qt signals: noun_verb_ed or noun_changed (`config_changed`, `simulation_stepped`)

---

## 9. pyproject.toml Specification

```toml
[project]
name = "qndt"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "quimb>=1.7",
    "anyio>=4.3",
    "pydantic>=2.6",
    "PySide6>=6.7",
    "pyqtgraph>=0.13",
    "networkx>=3.3",
    "pandas>=2.2",         # CSV telemetry ingestion
    "httpx>=0.27",         # JSON stream source
]

[project.optional-dependencies]
dev = [
    "pytest>=8.1",
    "pytest-cov>=5.0",
    "pytest-qt>=4.4",
    "pytest-anyio>=0.0.0",
    "hypothesis>=6.100",
    "ruff>=0.4",
    "mypy>=1.9",
]

[project.scripts]
quasar = "qndt.gui.app:main"        # launches the GUI
quasar-sim = "qndt.core.orchestrator:cli_main"  # headless sim

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
strict = true
python_version = "3.10"
```

---

## 10. Build & Run Commands

```bash
# One-time setup (from Quasar/ root)
pip install -e ".[dev]"

# Launch GUI
quasar

# Headless simulation (no Qt)
quasar-sim --scenario examples/live_fiber_twin.json

# Quality checks (run before every commit)
ruff check src/ tests/
mypy src/qndt/
QT_QPA_PLATFORM=offscreen pytest tests/ -v --cov=src/qndt --cov-report=term-missing

# Run only physics regression suite
QT_QPA_PLATFORM=offscreen pytest tests/ -m physics_regression -v

# Run GUI tests headless
QT_QPA_PLATFORM=offscreen pytest tests/test_gui/ -v
```

---

## 11. Scenario Config Format (JSON)

When saving/loading a simulation scenario:

```json
{
  "scenario_name": "Tokyo-Osaka Backbone",
  "nodes": [
    {"id": "tokyo_repeater_1", "type": "memory_node",
     "t2_nominal": 1.0, "wear_rate_kappa": 1e-4, "calib_drift_rate": 1e-6}
  ],
  "links": [
    {"id": "link_01", "source": "tokyo_repeater_1", "dest": "osaka_bsm_1",
     "lambda_q_nm": 1550.0, "fiber_length_km": 120.0,
     "attenuation_dB_per_km": 0.2,
     "classical_channels": [
       {"lambda_c_nm": 1310.0, "launch_power_mW": 1.0}
     ]}
  ],
  "telemetry": {
    "source_type": "csv",
    "path": "data/tokyo_fiber_temp_2024.csv",
    "columns": {"t": 0, "temperature_C": 1, "seismic_ms2": 2, "wind_N": 3},
    "link_id": "link_01",
    "speedup": 100.0
  },
  "kernel": {
    "type": "exponential",
    "tau_x": 30.0, "tau_y": 30.0, "tau_z": 120.0
  },
  "simulation": {
    "chi_max": 64,
    "kappa_max": 8,
    "protocol": "bb84",
    "duration_s": 3600.0
  }
}
```
