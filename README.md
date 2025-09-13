# QUASAR — QUAntum Simulation for Authentication & Relay

> **Industrial-grade quantum network simulator** for pre-deployment characterization: device-faithful physics, protocol & network layers, and **integrated finite-key security** with one-click scientific reports.

<p align="left">
  <a href="#"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <a href="#"><img alt="Status" src="https://img.shields.io/badge/status-early%20access-orange.svg"></a>
  <a href="#"><img alt="CI" src="https://img.shields.io/badge/CI-passing-green.svg"></a>
</p>

---

## TL;DR

* **Security-first**: Built-in **finite-key** (DV) + composable ε bookkeeping and authentication.
* **Device-faithful + scalable**: Detailed photonic hardware models *and* network/routing at scale.
* **Calibration-ready**: Import lab measurements, infer parameters, propagate uncertainty.
* **Reproducible**: Deterministic seeds, provenance manifests, Parquet outputs, golden-scenario CI.
* **Auto-reports**: Publication-quality figures and a PDF/HTML report with assumptions & audit logs.

---

## Table of Contents

1. [What is QUASAR?](#what-is-quasar)
2. [Key Differentiators](#key-differentiators)
3. [Feature Overview](#feature-overview)
4. [Architecture](#architecture)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Configuration (YAML Schema)](#configuration-yaml-schema)
8. [Examples](#examples)
9. [CLI Reference](#cli-reference)
10. [Python API](#python-api)
11. [Metrics & Outputs](#metrics--outputs)
12. [Security Layer](#security-layer)
13. [Validation & Reproducibility](#validation--reproducibility)
14. [Performance & Scaling](#performance--scaling)
15. [Visualization & Reporting](#visualization--reporting)
16. [Plugin SDK & Extensibility](#plugin-sdk--extensibility)
17. [Calibration & Digital Twin](#calibration--digital-twin)
18. [Classical Integration (ns-3/SDN)](#classical-integration-ns-3sdn)
19. [Distributed Execution & Cloud](#distributed-execution--cloud)
20. [Repository Structure](#repository-structure)
21. [Contributing](#contributing)
22. [Roadmap](#roadmap)
23. [Versioning & Releases](#versioning--releases)
24. [License](#license)
25. [Citation](#citation)
26. [FAQ](#faq)
27. [Glossary](#glossary)
28. [Acknowledgements](#acknowledgements)

---

## What is QUASAR?

**QUASAR** is a discrete-event **quantum network** simulator that models end-to-end performance from **physical components** up through **protocols**, **networking**, and **security**, producing **scientific-grade reports** to characterize proposed networks before deployment.

* Built for: **network architects**, **protocol designers**, **systems engineers**, and **researchers**.
* Use cases: link budgets, topology comparison, protocol evaluation (e.g., **BB84/MDI-QKD/E91**), routing/scheduling policies, security margin estimation, and calibration-informed **digital twins**.

---

## Key Differentiators

* **Integrated finite-key security** (DV) and **composable ε** bookkeeping **inside** the simulator (not in spreadsheets after the fact).
* **Device-faithful models** (sources, channels, detectors, BSMs, memories, impairments, drift) **plus** large-scale network routing.
* **Calibration-ready & uncertainty-aware**: ingest lab data, infer parameters, and propagate uncertainty to metrics and reports.
* **Reproducible & auditable**: deterministic seeds, provenance manifests, golden scenarios, and auto-generated, publication-quality reports.

---

## Feature Overview

### Core & Reproducibility

* Discrete-event kernel (ps/ns precision), named RNG streams, deterministic seeds.
* Scenario **YAML DSL**, checkpoints & forking, provenance manifests.
* Results in **Parquet/Arrow** with queryable metadata.

### Physical Layer (Device-Faithful)

* **Equipment**: SPDC/SFWM/SPS/WCP, modulators (phase/pol/intensity), interferometers, **BSM**, **SPAD/SNSPD**, time-taggers, frequency converters, **quantum memories**.
* **Channels**: fiber (attenuation, dispersion, PMD, temp drift), **free-space** (turbulence, pointing jitter, weather), co-propagating classical noise (Raman/ASE).
* **Impairments**: loss, dark counts, afterpulsing, timing jitter, pol/phase drift (OU/random walk), spectral mismatch, HOM visibility reductions, decoherence (T1/T2), crosstalk, deadtime/saturation.

### Protocol Layer

* **DV QKD**: **BB84 (+decoy)**, SARG04, DPS, COW; **MDI-QKD**; **E91** (entanglement-based).
* Entanglement generation, **swapping**, **purification**; time/wavelength multiplexing.
* Information reconciliation (Cascade, LDPC), privacy amplification (Toeplitz/SHA), sync & clock discipline.

### Network Layer

* Topologies: line, star/MDI hub, ring, mesh, hierarchical, satellite ↔ ground.
* Routing: shortest, k-shortest, reliability/rate-aware, memory-aware; multi-commodity flows.
* Scheduling: time-slot & wavelength planners; queueing (FIFO, priority, deadlines).
* Traffic models: session arrivals, key-demand traces, app-mix profiles (VPN, OT/SCADA, DC interconnect).

### Security Layer

* **Finite-key** analytics for DV protocols with selectable proof variants; parameter estimation; leakage accounting; **ε\_total** bookkeeping; **authentication** (UHF-MAC/Wegman–Carter).
* Toggleable **attack/side-channel** library: PNS, detector blinding, time-shift, Trojan-horse, saturation; LO manipulation (CV roadmap).

### Metrics & Viz

* Multi-layer metrics (Physical, Protocol, Network, Security) with **uncertainties**.
* Topology graphs, timelines, detector rasters, sweep heatmaps, violins/CDFs; animations.
* **One-click scientific report** (PDF/HTML) with abstract, methods, assumptions, CI bands, and appendices.

### Optional Modules (Roadmap)

* **CV-QKD** (GMCS, β-reconciliation, finite-size), **CV-MDI-QKD**.
* Satellite & advanced FSO models; **Hardware-in-the-Loop** (HIL) v2.
* GPU-accelerated kernels; ns-3/SDN deep bridge; digital twin auto-calibration.

---

## Architecture

```
CLI / Python API
  ├─ Orchestrator & Report/IO
  └─ Simulation Engine (discrete-event core; timeline; RNG)
        ├─ Physical Layer (equipment, channels, noise/defects)
        ├─ Protocol Layer (BB84, MDI, E91, swaps, purification)
        ├─ Network Layer (topology, routing, scheduling, traffic)
        ├─ Security Layer (finite-key, ε, auth, attack models)
        └─ Metrics & Observers → Viz & Report
```

* Packages: `quasar.core`, `quasar.models`, `quasar.protocols`, `quasar.network`, `quasar.security`, `quasar.metrics`, `quasar.io`, `quasar.viz`, `quasar.services`.
* Results flow: **Config → Run/Ensemble → Parquet + JSON → Viz → Report (PDF/HTML)**.

---

## Installation

```bash
# Clone your repo
git clone https://github.com/<your-org>/quasar.git
cd quasar

# Create a virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Install (editable) with dev extras
pip install -e ".[dev]"
```

**Python:** 3.10+
**Core deps (typical):** `numpy`, `scipy`, `pandas`, `pyarrow`, `pyyaml`, `networkx`, `matplotlib`, `plotly`, `rich`, `typer`
**Optional:** `ray` or `dask` (distributed), `fastapi`/`uvicorn` (service), `numba`/`jax`/CUDA (accel), `graphviz`, `tqdm`

---

## Quick Start

1. Run a demo **MDI-QKD** scenario:

```bash
python -m quasar.cli simulate examples/mdi_hub_demo.yaml --runs 8 --out results/mdi_demo --md
```

2. Explore outputs:

* `results/mdi_demo/metrics.json` (summary)
* `results/mdi_demo/report.md` (quick report)
* `results/mdi_demo/` (Parquet/JSON files for analysis)

3. Generate figures or a full report (see [Visualization & Reporting](#visualization--reporting)).

---

## Configuration (YAML Schema)

A scenario YAML fully describes nodes, links, equipment, protocols, sweeps, and run controls.

```yaml
meta:
  name: "mdi_hub_demo"    # run name
  seed: 12345             # global seed for reproducibility
  runs: 64                # ensemble size (optional; CLI can override)
  duration: 0.5s          # sim time (supports s/ms/us/ns/ps)
  out_dir: results/mdi
  tick: 0.01ns            # (optional) scheduler quantum

nodes:
  - id: Alice
    role: sender | receiver | mdi_relay | repeater | memory_node | gateway
    equipment:
      source: { type: wcp | spdc | sps | sfwm, mu: 0.3, rep_rate: 1e9, jitter_ps: 25 }
      encoder: { type: phase | pol | intensity, bandwidth_GHz: 10 }
      detectors:
        - { type: spad | snspd, eta: 0.85, dark_hz: 100, jitter_ps: 50, deadtime_ns: 60 }
      bsm: { visibility: 0.96 }
      memory: { T1_ms: 10, T2_ms: 5, read_eff: 0.7, write_eff: 0.7, multiplex: 8 }
      clock: { drift_ppm: 2, resync_s: 10 }

links:
  - a: Alice
    b: Charlie
    channel: { type: fiber | fso, L_km: 25, alpha_db_per_km: 0.2, D_ps_nm_km: 17, pmd_ps_sqrt_km: 0.1 }
  - a: Bob
    b: Charlie
    channel: { type: fiber, L_km: 25, alpha_db_per_km: 0.2 }

protocol:
  type: bb84 | mdi_qkd | e91
  sifting: { basis_prob_Z: 0.8 }
  decoy: { mu_signal: 0.4, mu_decoy: 0.1, p_signal: 0.8 }
  ec: { f: 1.15 }                       # reconciliation inefficiency
  pa: { security_epsilon: 1e-10 }       # target composable epsilon

sweep:
  mu: [0.1, 0.2, 0.3, 0.4]
  L_km: [10, 25, 40]

# Optional: attacks (toggle side-channel models)
attacks:
  pns: { enabled: true }
  time_shift: { enabled: false }

# Optional: calibration imports (CSV/JSON)
calibration:
  detectors: path/to/detector_specs.csv
  channels: path/to/link_loss.csv
```

**Notes:**

* Units: SI base internally; human-friendly shorthands allowed in YAML (e.g., `0.5s`, `25km`).
* Any omitted parameter uses the model’s default (documented per equipment class).

---

## Examples

* `examples/mdi_hub_demo.yaml`: MDI-QKD with two senders and a central BSM relay.
* `examples/bb84_decoy_sweep.yaml`: BB84 with decoy-state analysis and a distance sweep.
* `examples/satellite_fso_stub.yaml`: Free-space/satellite placeholder with turbulence knobs.

Run:

```bash
quasar simulate examples/bb84_decoy_sweep.yaml --runs 64 --out results/bb84_sweep --pdf
```

---

## CLI Reference

```bash
# Help
quasar --help

# Simulate a single scenario
quasar simulate path/to/topology.yaml --runs 64 --out out/run1 --pdf --md

# Validate a config against JSONSchema (helpful errors)
quasar validate path/to/topology.yaml

# Parameter sweep (Cartesian grid from CLI)
quasar sweep path/to/topology.yaml --grid mu=0.05:0.5:0.05 L_km=10,25,40 --runs 32

# Plot summaries (write figures to out/plots/)
quasar plot out/run1 --what keyrate,qber,hom,utilization --pdf

# Generate a standalone report from results (if re-rendering)
quasar report out/run1 --pdf --html
```

**Common flags**

* `--runs N`: ensemble size (repeatability with different seeds per run).
* `--out PATH`: output directory (results, figures, report).
* `--pdf` / `--html` / `--md`: report formats to render.

---

## Python API

```python
import quasar as qs

sim = qs.Simulator.from_yaml("examples/mdi_hub_demo.yaml")
res = sim.run(ensemble=64)

# Access aggregates and runs
print(res["aggregates"]["avg_qber"])
print(res["runs"][0]["sifted_rate_bps"])

# Save a report
from quasar.io.report import Report
Report(res).to_markdown("results/mdi_demo/report.md")
```

---

## Metrics & Outputs

**Directory layout (per run):**

```
results/<name>/
  ├─ manifest.json            # provenance (config snapshot, seeds, versions, env)
  ├─ metrics.json             # top-level aggregates
  ├─ metrics.parquet          # per-run & aggregated metrics
  ├─ events/                  # (optional) event traces
  ├─ figs/                    # generated figures/plots
  └─ report/                  # rendered reports (PDF/HTML/MD)
```

**Typical metrics (non-exhaustive):**

* **Physical**: transmittance η, click/arrival histograms, HOM visibility, jitter, dark fraction, deadtime loss, spectral overlap.
* **Protocol**: QBER (basis-conditioned), yields, decoy gains, BSM success probability, entanglement fidelity.
* **Network**: end-to-end throughput, latency distributions, blocking probability, path reliability, resource utilization.
* **Security**: **finite-key SKR** (lower bound), ε\_total, authentication failure bounds, leakage budgets.

**Uncertainty:** Ensemble runs compute mean/variance and confidence intervals. Sensitivity analysis is available (Sobol/gradient-free).

---

## Security Layer

* **Finite-key** for DV protocols (BB84/MDI/E91 variants): parameter estimation, error correction leakage accounting, privacy amplification, and composable ε bookkeeping.
* **Authentication**: UHF-MAC/Wegman–Carter; tag length and key consumption budgeting.
* **Attacks/side-channels** (toggleable): PNS, detector blinding, time-shift, Trojan-horse, saturation (CV/LO roadmap).
* **Outputs include**: explicit **assumptions**, ε\_total targets and achieved values, and machine-readable **audit logs**.

> ⚠️ **Note:** QUASAR provides well-documented implementations of standard finite-key analyses. Users are responsible for selecting assumptions/proofs appropriate for their deployment and verifying compliance with applicable standards.

---

## Validation & Reproducibility

* **Analytical baselines**: link loss laws; Poisson click statistics; HOM visibility; idealized key-rate formulas for sanity.
* **Cross-tool sanity**: reproduce selected public micro-scenarios where licensing permits.
* **Golden scenarios & CI**: regression-guarded test cases with threshold alarms.
* **Provenance**: every run bundles seeds, config, versions, OS/platform info, and hashes of core modules.

---

## Performance & Scaling

* Vectorized hot paths for click/time-binning; optional **Numba/JAX/C++** kernels.
* **Distributed ensembles**: Ray/Dask/SLURM backends; job isolation; resumable runs.
* Streaming aggregation to keep memory bounded; Parquet for efficient post-hoc analytics.
* Optional GPU acceleration for Monte Carlo hot paths (roadmap).

---

## Visualization & Reporting

* **Figures**: topology with edge annotations; detector click rasters; event timelines; sweep heatmaps (e.g., SKR vs μ & distance); violin/CDFs with CI bands.
* **Animations**: time-evolving paths, entanglement distribution, queue occupancy.
* **Reports** (PDF/HTML/MD): abstract; methods; configuration snapshot; figures; tables; assumptions; seeds; version hashes; references; and appendices.

Generate directly:

```bash
quasar report results/mdi_demo --pdf --html
```

---

## Plugin SDK & Extensibility

QUASAR supports **plugins** for equipment, noise models, protocols, routing policies, estimators, and attacks.

**Base interfaces (sketch):**

```python
from dataclasses import dataclass
from typing import Protocol, Any

@dataclass
class EquipmentModel:
    id: str
    def emit_events(self, t0: float, duration: float, rng) -> list[Any]:
        ...

class NoiseModel(Protocol):
    name: str
    def sample(self, t: float, ctx: dict) -> Any: ...
```

Register plugins via `entry_points` or a `plugins/` directory. See `quasar/plugins/README.md`.

---

## Calibration & Digital Twin

* Import lab measurements (CSV/JSON) for detectors, sources, channels.
* Perform **Bayesian parameter inference** (roadmap) to fit models to telemetry and **propagate uncertainty** into metrics.
* Track **drifts** and re-fit on new data; generate **what-if** comparisons vs calibrated baselines.

---

## Classical Integration (ns-3/SDN)

* Export key-delivery traces; emulate control-plane latencies.
* Optional **ns-3 bridge** to co-simulate classical routing, failover, and management plane behavior.
* SDN/NMS stubs (REST/NetConf northbound) for policy experiments.

---

## Distributed Execution & Cloud

* Submit ensembles/sweeps to Ray/Dask clusters or SLURM.
* Container images (CPU/GPU) with pinned dependencies for reproducibility.
* K8s job templates; artifact collection to object storage; provenance preserved.

---

## Repository Structure

```
quasar/
├─ quasar/                         # Python package
│  ├─ __init__.py
│  ├─ cli.py                       # CLI entrypoints
│  ├─ core/                        # Discrete-event engine & fundamentals
│  │  ├─ __init__.py
│  │  ├─ events.py
│  │  ├─ simulator.py
│  │  ├─ rng.py
│  │  └─ timeline.py
│  ├─ models/                      # Equipment & channel models (+ noise)
│  │  ├─ __init__.py
│  │  ├─ sources/                  # SPDC/SFWM/SPS/WCP
│  │  ├─ detectors/                # SPAD/SNSPD
│  │  ├─ channels/                 # fiber, FSO, converters
│  │  ├─ memories/                 # quantum memories
│  │  └─ noise/                    # loss, jitter, afterpulsing, drift, etc.
│  ├─ protocols/                   # BB84, MDI-QKD, E91, swaps, purification
│  │  ├─ __init__.py
│  │  ├─ bb84.py
│  │  ├─ mdi_qkd.py
│  │  └─ e91.py
│  ├─ network/                     # topology, routing, scheduling, traffic
│  │  ├─ __init__.py
│  │  ├─ topology.py
│  │  ├─ routing.py
│  │  ├─ scheduling.py
│  │  └─ traffic.py
│  ├─ security/                    # finite-key, ε bookkeeping, auth, attacks
│  │  ├─ __init__.py
│  │  ├─ finite_key.py
│  │  ├─ composability.py
│  │  ├─ authentication.py
│  │  └─ attacks/
│  ├─ metrics/                     # observers, aggregations, CI/uncertainty
│  │  ├─ __init__.py
│  │  ├─ definitions.py
│  │  └─ aggregators.py
│  ├─ io/                          # configs, schemas, persistence, reports
│  │  ├─ __init__.py
│  │  ├─ config.py                 # YAML loader + JSONSchema validation
│  │  ├─ report.py                 # PDF/HTML builder
│  │  ├─ parquet.py
│  │  └─ schemas/                  # JSONSchemas for YAML & results
│  ├─ viz/                         # plots and animations
│  │  ├─ __init__.py
│  │  ├─ plots.py
│  │  └─ animations.py
│  ├─ plugins/                     # external plugin entry points
│  │  └─ README.md
│  └─ services/                    # optional REST service (FastAPI)
│     ├─ __init__.py
│     ├─ api.py
│     └─ workers.py
│
├─ examples/                       # runnable configs & notebooks
│  ├─ mdi_hub_demo.yaml
│  ├─ bb84_decoy_sweep.yaml
│  ├─ satellite_fso_stub.yaml
│  └─ notebooks/
│     └─ exploring_results.ipynb
│
├─ docs/                           # user & dev docs (mkdocs or sphinx)
│  ├─ index.md
│  ├─ user-guide/
│  ├─ dev-guide/
│  └─ api/
│
├─ tests/                          # pytest suites
│  ├─ unit/
│  │  ├─ test_loss.py
│  │  ├─ test_clicks.py
│  │  └─ test_finite_key.py
│  ├─ integration/
│  │  └─ test_mdi_pipeline.py
│  └─ golden/                      # regression “golden” scenarios
│     └─ mdi_small.yaml
│
├─ benchmarks/                     # performance and scaling scenarios
│  └─ large_mesh_100nodes.yaml
│
├─ data/                           # (optional) calibration inputs, small fixtures
│  └─ calibration/
│     └─ example_detector.csv
│
├─ scripts/                        # helper scripts
│  ├─ run_demo.sh
│  └─ sweep_grid.py
│
├─ .github/workflows/              # CI/CD
│  ├─ ci.yml
│  └─ release.yml
│
├─ pyproject.toml
├─ README.md
├─ CONTRIBUTING.md
├─ CODE_OF_CONDUCT.md
├─ LICENSE
├─ CITATION.cff
├─ CHANGELOG.md
├─ .gitignore
├─ .pre-commit-config.yaml
├─ Makefile                        # or justfile
└─ mkdocs.yml                      # if using mkdocs
```

---

## Contributing

We welcome contributions! Please read **CONTRIBUTING.md** for details.

* **Setup**: `pip install -e ".[dev]"` and `pre-commit install`
* **Style**: `black`, `ruff`, `isort`, `mypy` (type hints encouraged)
* **Tests**: `pytest -q`; golden-scenario tests must pass
* **Commits**: Conventional Commits (e.g., `feat:`, `fix:`, `docs:`)
* **PRs**: Include tests and docs for new features; keep public APIs typed & documented

---

## Roadmap

* **M1 (MVP)**: DV (BB84+decoy, MDI-QKD), finite-key, physical kernels, calibration import, report v1
* **M2**: Routing/scheduling, traffic models, ns-3 bridge (basic)
* **M3**: Distributed runs, dashboards, Bayesian optimization
* **M4**: Service & ops (REST, RBAC, provenance catalog)
* **M5**: CV-QKD finite-size, Satellite/FSO Pro, HIL v2, GPU kernels

**Optional modules (extended):** CV-MDI, digital twin auto-calibration, attack “red team” pack, vendor library with datasheet importers, multi-tenant ops.

---

## Versioning & Releases

* Semantic Versioning (MAJOR.MINOR.PATCH).
* Release artifacts: source distribution, wheels, Docker images (CPU/GPU).
* Changelog managed in **CHANGELOG.md**; tags signed if possible.

---

## License

Apache-2.0. See **LICENSE** for details.

---

## Citation

If you use QUASAR in academic work:

```
@software{quasar2025,
  title        = {QUASAR: Quantum Simulation for Authentication and Relay},
  author       = {QUASAR Contributors},
  year         = {2025},
  url          = {https://github.com/<your-org>/quasar}
}
```

---

## FAQ

**Q: Is QUASAR a universal QC simulator?**
A: No. QUASAR focuses on **quantum networking** and QKD. Gate-model QC is out-of-scope except where needed for networking.

**Q: Can I use real hardware with QUASAR?**
A: Yes—via **Hardware-in-the-Loop (HIL)** shims (roadmap v2 for device control).

**Q: Does QUASAR support CV-QKD?**
A: CV finite-size support is on the roadmap. DV finite-key is supported first.

**Q: How reproducible are results?**
A: Deterministic seeds per component, provenance manifests, and golden-scenario CI make runs reproducible across environments.

---

## Glossary

* **BSM**: Bell-State Measurement
* **CV/DV**: Continuous/Discrete Variable
* **EC/PA**: Error Correction / Privacy Amplification
* **HIL**: Hardware in the Loop
* **HOM**: Hong–Ou–Mandel (interference visibility)
* **MDI-QKD**: Measurement-Device-Independent QKD
* **QBER**: Quantum Bit Error Rate
* **SKR**: Secret Key Rate (finite-key lower bound)

---

## Acknowledgements

Thanks to the broader quantum networking community for foundational ideas in discrete-event simulation, protocol design, and security analyses. QUASAR aims to integrate best practices into a cohesive, reproducible, and extensible toolchain.
