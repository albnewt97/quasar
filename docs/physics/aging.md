# Device Aging

## Coherence Time Wear Model

```math
T_2(N) = T_2^{(0)}\,e^{-N/N_c}
```

`N` is the cumulative number of operations (gates, channel applications, measurements) a quantum memory node has performed since the start of the run, and `N_c` is the characteristic wear constant — the operation count at which `T_2` has decayed to `1/e` of its nominal value `T_2^{(0)}`. Physically, this models the cumulative effect of repeated gate-pulse heating cycles, mechanical and thermal stress on the device substrate, and gradual drift away from the calibration point each operation assumed — none of which reverse themselves between operations, so the decay is monotonic in `N` rather than in elapsed wall-clock time.

## Gate Overrotation Drift

```math
\varepsilon(t) = \varepsilon_0 + \kappa\,t
```

Independently of coherence decay, every gate carries a small systematic miscalibration `ε(t)` that grows *linearly* with the time elapsed since the node's last calibration — `κ` is the drift rate. This models the simple fact that a calibration captures the device's parameters at one instant, and the device continues to drift away from that snapshot the longer it operates before being recalibrated; it is a slow, deterministic process distinct from the stochastic decoherence captured by `T_2(N)`.

## Idle Dephasing from Classical Hold Times

```math
p_z^{\mathrm{idle}} = \tfrac{1}{2}
  \!\left(1 - e^{-t_{\mathrm{idle}}/T_2(N)}\right)
```

This equation is where the classical and quantum layers of the digital twin couple together (novelty #4, §6). `t_idle` is not a fixed parameter — it is `induced_idle`, the waiting time a quantum memory is forced to hold state because the **classical control plane** has not yet delivered the signalling needed to proceed (route computation, acknowledgment, retransmission after a dropped packet). Whatever congestion or jitter the `AsynchronousControlPlane` produces on the classical side is fed directly into this equation, converting classical network delay into a literal dephasing probability on the qubit it forces to wait — and a longer wait against an already-degraded `T2(N)` produces disproportionately more dephasing, so the two effects compound.

## Interpreting the Aging Dashboard

The `AgingPlot` panel tracks `T2(N)` against cumulative operation count `N`, with an exponential fit overlay so the configured `wear_const_nc` and observed decay can be visually cross-checked.

- **What "Critical" T2 means operationally**: there is no single universal threshold, but as `T2(N)` falls toward the timescale of `induced_idle` waits the node is actually experiencing, idle dephasing `p_z^{idle}` rapidly approaches its ceiling of 0.5 (fully dephased) — a node is operationally critical once its current `T2` is no longer comfortably larger than the control plane's typical induced-idle durations on links touching that node.
- **When to schedule recalibration**: recalibration resets `gate_overrotation_0` and the elapsed-time clock used by `ε(t)`, but it does **not** reset cumulative operation count `N` or restore `T2(N)` — wear is permanent in this model. Recalibration is therefore worth scheduling when `ε(t)`'s linear drift has grown large relative to the gate-error budget, independent of where `T2(N)` happens to be.
- **Effect of `wear_const_nc`**: a larger `Nc` means the node tolerates more cumulative operations before `T2` meaningfully decays — physically, a more robust device or a less aggressive duty cycle. Lowering `Nc` in a scenario config is the direct way to simulate a node under heavier operational stress without changing anything else about the topology.
