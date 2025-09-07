# QUASAR – QUAntum Simulation for Authentication & Relay

**Tagline:** *Extensible simulator for MDI-QKD networks with fiber, free-space, and realistic physics.*

---
## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e .
python scenario1_static.py        # works as-is today


## Description

**QUASAR** is a research-grade simulation framework for **Measurement-Device-Independent Quantum Key Distribution (MDI-QKD)** networks, built on top of the [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe) simulator.

It enables researchers and engineers to:

* Build and simulate **fiber and free-space quantum networks** with static and moving nodes.
* Model realistic physics: fiber dispersion, Raman noise, detector afterpulsing, atmospheric turbulence, weather effects.
* Orchestrate **MDI-QKD protocols** with Charlie as the untrusted relay.
* Collect metrics across **four layers**:

  * **Physical:** detectors, BSM outcomes, visibility, losses.
  * **Network:** topology, routing, resource utilization.
  * **Protocol:** sifting, QBER, error correction, throughput, latency.
  * **Security:** privacy amplification, key rate, composable bounds.
* Explore multiple **scenarios**:

  1. Static fiber-only networks (equidistant and uneven).
  2. Free-space moving-node networks with photon delay controllers.
  3. City-wide networks (e.g., London) mixing fiber backbones and mobile free-space links.
  4. UK-wide networks with optimization for Charlie placement, best path selection, and resilience.
* Visualize results via an **interactive dashboard**:

  * Topology diagrams and animations of moving nodes.
  * Dynamic plots for QBER, key rate, coincidence histograms, fading distributions.
  * Weather and turbulence effects on free-space channels.
  * Optimization overlays showing best paths and redundancy.

---

## Features

* Event-driven simulation (via SeQUeNCe kernel).
* Extendable **Physics Profiles** for custom channels, detectors, and sources.
* Weather simulation mode (fog, rain, haze, turbulence).
* Hybrid routing: mix fiber and free-space hops with handovers.
* Resilience and QoS testing (failures, jitter, variance in key rate).
* Security stress tests (eavesdropper-induced noise).
* Parameter sweeps with animated visualizations.
* Automated report generation (CSV/Parquet + PDF/HTML).

---

## Repository Structure (planned)

```
quasar/
├─ sequence_ext/
│  ├─ orchestrator/      # MDI-QKD logic & controllers
│  ├─ scenarios/         # Scenario 1–4 modules
│  ├─ topo/              # Topology & geometry helpers
│  ├─ io/                # Logging & structured outputs
│  └─ viz/               # Plotting, animations, dashboard API
├─ dashboard/            # Streamlit dashboard
├─ configs/              # YAML configs & physics presets
├─ scripts/              # CLI tools
├─ tests/                # Unit & integration tests
├─ data/runs/            # Simulation outputs
├─ .gitignore
├─ LICENSE (Apache 2.0)
├─ pyproject.toml / requirements.txt
└─ README.md
```

---

## Quick Start (planned)

```bash
# Clone repository
git clone https://github.com/<your-org>/quasar.git
cd quasar

# Install dependencies
pip install -r requirements.txt

# Run a baseline Scenario 1a (static equidistant fiber)
python scripts/run_scenario.py --config configs/scenario1_equidistant.yaml

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Scenarios

1. **Static Nodes**: equidistant and uneven fiber.
2. **Moving Node (Free-Space)**: one mobile Alice or Bob, photon delay controller, weather dynamics.
3. **City-wide (London)**: multiple Charlies, hybrid fiber/free-space routing.
4. **UK-wide**: optimized topology, best path selection, redundancy testing.

---

## License

This project is licensed under the **Apache 2.0 License** – see [LICENSE](LICENSE) for details.

---

## Acknowledgements

* Built on the [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe) simulator.
* Inspired by ongoing research in **quantum networks** and **MDI-QKD**.

---

## Citing QUASAR and SeQUeNCe

If you use **QUASAR** in your research, please cite it as:

```
@misc{quasar2025,
  title        = {QUASAR – QUAntum Simulation for Authentication \& Relay},
  author       = {Naman Srivastava},
  year         = {2025},
  howpublished = {\url{https://github.com/albnewt97/quasar}},
  note         = {Simulator for Measurement-Device-Independent Quantum Key Distribution (MDI-QKD) networks}
}
```

Please also cite **SeQUeNCe**, which QUASAR builds upon:

```
@misc{sequence-toolbox,
  title        = {SeQUeNCe: Simulator for QUantum NEtworks and Communication},
  author       = {SeQUeNCe Developers},
  howpublished = {\url{https://github.com/sequence-toolbox/SeQUeNCe}},
  year         = {2020--2025}
}
```
