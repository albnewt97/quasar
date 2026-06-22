# Channel Model

## Pauli Transfer Matrix Representation

Quasar represents every noise contributor as a diagonal Pauli Transfer Matrix (PTM) rather than as a set of Kraus operators. For a Pauli channel, the Kraus representation and the PTM carry the same information, but the PTM has three properties that make it the right computational primitive for a system that composes noise from many independent physical sources at every quantum event:

- **It is diagonal.** A Pauli channel acting on a single qubit is fully described by three numbers — the eigenvalues of the channel along the X, Y, and Z axes — rather than a set of 2×2 Kraus matrices.
- **Composition is a Hadamard product.** Two Pauli channels applied in sequence compose by element-wise multiplication of their diagonal PTMs. This is *exact*, not a first-order approximation, and it is O(1) regardless of how many noise contributors are composed (§3.2). Composing four engines' worth of noise costs the same as composing one.
- **It maps directly onto measurable quantities.** The PTM eigenvalues are exactly the contrast-decay rates an experimentalist measures via randomized benchmarking or process tomography — λx, λy, λz are observable, not abstract amplitudes.

```math
R = \mathrm{diag}(1, \lambda_x, \lambda_y, \lambda_z)
```

```math
\lambda_x = 1 - 2(p_y + p_z), \quad
\lambda_y = 1 - 2(p_x + p_z), \quad
\lambda_z = 1 - 2(p_x + p_y)
```

Every `NoiseContributor` in Quasar (§3.1) implements exactly one method, `ptm(ctx) -> np.ndarray`, returning this length-4 array. `ChannelComposer.effective_ptm()` is the only place the Hadamard product is taken — no engine ever calls another engine's `ptm()` directly (§3.3).

## Time-Varying Channel via Environmental Convolution

The channel is not static. Each engine's instantaneous Pauli rate vector is itself the output of a convolution between a memory kernel and the environment's history:

```math
\vec{p}(t) = \Phi\!\left(\int_{-\infty}^{t}
  \mathbf{K}(t-t')\,\mathbf{S}\,\vec{E}(t')\,dt'\right)
```

- **`E(t')`** — the environmental state vector at past time `t'` (e.g. `[temperature, seismic_acceleration, wind_force]`), supplied by a `TelemetrySource`.
- **`S`** — the sensitivity matrix, mapping each environmental axis onto each Pauli axis (rows `[px, py, pz]`, columns matching `E`'s components). See the SMF-28 defaults below.
- **`K(τ)`** — the memory kernel, a 3×3 matrix describing how strongly an environmental disturbance at lag `τ` in the past still influences the present Pauli rate.
- **`Φ`** — a squashing map clamping the result into the physically valid Pauli-rate region: each `p_i ∈ [0, 0.5)` with `Σp_i ≤ 1`.

This is non-Markovian by construction: the present Pauli rate vector `p(t)` depends on the *entire weighted history* of the environment, not just its instantaneous value at `t`. A Markovian channel would set `p(t) = Φ(S·E(t))` — no integral, no memory. The kernel `K(τ)` is what gives the environment a memory; its decay profile (exponential, oscillatory, or Gaussian) determines how long past disturbances continue to affect the channel and whether that influence can ever *increase* the apparent coherence again, which is the signature of genuine information backflow (see [Non-Markovian Dynamics](nonmarkovian.md)).

In practice the integral is evaluated as a discrete sum over buffered telemetry samples: `acc += K.eval(t - t'_k) @ (S @ E'_k) * dt_k`.

## SMF-28 Sensitivity Matrix Defaults

| Pauli axis | Temperature [°C] | Seismic [m/s²] | Wind [N] | Physical justification |
|---|---|---|---|---|
| px | 0.0 | 0.001 | 0.0005 | Mode coupling: micro-bends from seismic flexing and wind-induced fiber sway rotate polarization into orthogonal modes; temperature alone does not couple modes in SMF-28. |
| py | 0.0 | 0.001 | 0.0 | Mode coupling: seismic flexing is the dominant Y-axis driver; wind-induced sway is geometrically biased toward the X-axis coupling above and contributes negligibly here. |
| pz | 0.002 | 0.0 | 0.0005 | Dephasing: thermal expansion changes the fiber's optical path length, directly randomizing accumulated phase (dominant term); wind-induced tension changes contribute a smaller slow state-of-polarization drift; seismic events are too fast to imprint a net phase shift before averaging out. |

These values are the **illustrative, uncalibrated** defaults referenced throughout the codebase — defined once in `qndt.telemetry.calibration.S_SMF28_DEFAULT` (non-GUI) and mirrored in the GUI's `TelemetryPanel` defaults, per the GUI Isolation Law (§3.6). The per-entry justifications above are qualitative physical motivation only; no field measurement, literature citation, or formal derivation underpins the specific magnitudes. Additionally, the temperature channel couples to the **absolute Celsius value**, not to a ΔT, so the resulting p_z (and the demo QBER) depends on the temperature-scale origin rather than on a physical drift.

## Memory Kernel Implementations

**Exponential** — `K(τ) = diag(exp(-τ/τx), exp(-τ/τy), exp(-τ/τz))`. Monotonic decay from `K(0) = I₃`, governed by a separate time constant per Pauli axis. Use this for the common case where the environment's influence simply fades with a single characteristic timescale per axis and you have no reason to expect oscillatory backflow — it is the closest of the three to a "Markovian-limit" approximation while still carrying finite memory.

**Lorentzian** — `K(τ) = exp(-γ·τ)·cos(ω₀·τ)·I₃`. A damped oscillation: the cosine factor drives the kernel through zero and negative, which is exactly what produces sign-changing canonical rates and a positive RHP witness. Use this when modelling a resonant physical process — for example a mechanical or acoustic resonance in the fiber mount — where the environment doesn't just fade but actively oscillates and can transiently *return* coherence to the system.

**Gaussian** — `K(τ) = amplitude · exp(-τ²/(2σ²)) · I₃`. Peaked at `τ=0`, symmetric, always non-negative. Use this to model a single correlated bath fluctuation event (e.g. a localized thermal transient) with a well-defined characteristic width `σ` and no oscillatory component — it will never by itself generate information backflow, since it never goes negative.
