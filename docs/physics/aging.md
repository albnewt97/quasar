# Device Aging

## Coherence Time Wear Model

Quasar makes T₂ a function of accumulated operational duty cycle D(t) = ∫₀ᵗ u(t′) dt′, where u(t′) ∈ [0,1] is the instantaneous node utilisation. The wear law is Matthiessen's rule applied to the decoherence rate (whitepaper §6):

```math
\frac{1}{T_2\!\bigl(D\bigr)} = \frac{1}{T_2^{(0)}} + \kappa\,D(t),
\qquad \frac{1}{T_2} = \frac{1}{2\,T_1} + \frac{1}{T_\phi}
```

T₂⁽⁰⁾ is the as-calibrated coherence time, κ is the wear coefficient, and the second identity (standard T₁/T₂/T_φ relation) is exact. Wear acts on the pure-dephasing time T_φ; the longitudinal relaxation time T₁ is not subject to duty-cycle wear. The resulting dephasing eigenvalues are λ_x(t) = λ_y(t) = exp(−t_idle/T₂(D)) while λ_z is governed by T₁.

**Physical bound enforcement (T₂ ≤ 2T₁):** Because T_φ ≥ 0, the identity requires T₂ ≤ 2T₁ at all times. Equality (T₂ = 2T₁) is the pure-T₁ limit (T_φ → ∞). This bound is enforced at configuration load time by `NodeConfigModel` (pydantic validator) and at runtime by `DeviceAgingModel.__init__` and `set_node_params`. An unphysical T₂ > 2T₁ raises `ValueError` with the violated values. Reference: Nielsen & Chuang (2010) Ch. 8 [ref 1].

## Gate Overrotation Drift

> **[PHENOMENOLOGICAL MODEL]** — documented in whitepaper §6 as a deliberate code extension; calibrated engineering heuristic, not derived from a primary source. Code behaviour in `DeviceAgingModel.gate_overrotation` (`src/qndt/physics/aging.py`).

```math
\varepsilon(t) = \varepsilon_0 + \kappa_{\mathrm{drift}}\,t
```

Note: κ_drift here is the calibration drift rate [rad/s], distinct from the wear coefficient κ in the Matthiessen rule above. Every gate carries a small systematic miscalibration `ε(t)` that grows linearly with time elapsed since the node's last calibration. This models the fact that a calibration captures device parameters at one instant, and the device drifts from that snapshot over time — a slow, deterministic process distinct from the stochastic decoherence captured by the T₂ wear model.

## Idle Dephasing from Classical Hold Times

```math
p_z^{\mathrm{idle}} = \tfrac{1}{2}
  \!\left(1 - e^{-t_{\mathrm{idle}}/T_2(D)}\right)
```

This is where the classical and quantum layers couple (whitepaper §7). `t_idle` is `induced_idle` — the waiting time a quantum memory is forced to hold state because the classical control plane has not yet delivered the signalling needed to proceed (route computation, acknowledgment, retransmission). Whatever congestion or jitter the `AsynchronousControlPlane` produces converts directly into a dephasing probability on the waiting qubit; a longer wait against an already-worn T₂(D) compounds both effects. The Z-error probability p_z maps to the transverse PTM eigenvalues via λ_x = λ_y = 1 − 2p_z (standard phase-flip eigenvalue relation).

## Interpreting the Aging Dashboard

The `AgingPlot` panel tracks `T2(D)` against accumulated duty cycle `D` [s], with the Matthiessen curve `T2(D) = 1/(1/T2_0 + κD)` overlaid so the configured `wear_rate_kappa` and observed decay can be visually cross-checked.

- **What "Critical" T2 means operationally**: there is no single universal threshold, but as `T2(D)` falls toward the timescale of `induced_idle` waits the node is actually experiencing, idle dephasing `p_z^{idle}` rapidly approaches its ceiling of 0.5 (fully dephased) — a node is operationally critical once its current `T2` is no longer comfortably larger than the control plane's typical induced-idle durations on links touching that node.
- **When to schedule recalibration**: recalibration resets `gate_overrotation_0` and the elapsed-time clock used by `ε(t)`, but it does **not** reset accumulated duty cycle `D` or restore `T2(D)` — wear is permanent in this model. Recalibration is therefore worth scheduling when `ε(t)`'s linear drift has grown large relative to the gate-error budget, independent of where `T2(D)` happens to be.
- **Effect of `wear_rate_kappa`**: a smaller κ means the node tolerates more accumulated duty cycle before `T2` meaningfully decays — physically, a more robust device. Raising κ in a scenario config directly simulates a node under heavier wear stress without changing anything else about the topology.
