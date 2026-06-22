# Quasar — Quantum Network Digital Twin

Quasar models real-world quantum communication infrastructure as a time-varying, telemetry-driven cyber-physical system, not as an idealised textbook channel.

## The Five Novelties

1. **Non-Markovian telemetry** — the effective noise channel is a live memory-kernel convolution over a real environmental stream (temperature, seismic acceleration, wind force), with an online Rivas-Huelga-Plenio (RHP) witness certifying information backflow, instead of a static depolarising rate fixed at init.
2. **Raman co-existence engine** — spontaneous Raman scattering from co-propagating classical WDM channels is converted into a dark-count rate and folded into the channel's PTM, instead of simulating the quantum channel in isolation.
3. **Device aging** — coherence time T2 decays exponentially with cumulative operation count and gate overrotation drifts linearly with elapsed time, instead of using a fixed T2/gate fidelity for the whole run.
4. **Classical↔quantum coupling** — classical control-plane congestion and jitter produce an `induced_idle` time that simultaneously raises T2 dephasing and the Raman noise floor, instead of treating the control plane as ideal and free.
5. **Tensor-network state backend** — quantum state is tracked through an MPDO/LPDO representation with bounded bond dimension (χ_max, κ_max), with bond growth itself reported as a live quality-of-service signal, instead of a dense 2^{2n} global density matrix.

## Quick Install

```bash
git clone https://github.com/quasar-qndt/quasar.git
cd quasar
pip install -e ".[dev]"
quasar
```

## Documentation

- [Physics](physics/channel_model.md) — the PTM channel model, non-Markovian dynamics, Raman co-existence, and device aging equations behind every noise contributor.
- [API Reference](api/core.md) — generated reference for `core`, `physics`, `telemetry`, `quantum`, and `gui`.
- [Examples](examples/live_fiber_twin.md) — runnable scripts demonstrating a live fiber twin and a co-existence sweep.
