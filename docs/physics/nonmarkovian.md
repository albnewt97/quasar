# Non-Markovian Dynamics

## What Non-Markovian Means in a Quantum Channel

A **Markovian** channel is memoryless: the system's future evolution depends only on its present state, not on how it got there. In the Pauli-rate picture, a Markovian channel's instantaneous error rates are a fixed function of the instantaneous environment — once the environment settles, the channel's behaviour is fully determined by where it is *right now*.

A **non-Markovian** channel has memory: information that has leaked from the quantum system into its environment can flow back. Physically, this happens when the environment is not a featureless heat bath but has its own internal structure (a resonance, a finite correlation time, a slowly-relaxing degree of freedom) that can temporarily hand coherence back to the system instead of irreversibly absorbing it. This is the **information backflow** Quasar's telemetry-convolution model (see [Channel Model](channel_model.md)) is built to capture: because the present Pauli rate vector is a convolution over the environment's entire weighted history, the channel can transiently "remember" a past disturbance and partially undo its own decoherence.

## TCL Master Equation

Quasar derives instantaneous decay rates from the Time-Convolutionless (TCL) master equation:

```math
\dot\rho = -\frac{i}{\hbar}[H(t),\rho]
  + \sum_k \gamma_k(t)\!\left(L_k\rho L_k^\dagger
  - \tfrac12\{L_k^\dagger L_k,\rho\}\right)
```

The `L_k` are the Pauli jump operators (X, Y, Z) and `γ_k(t)` are their *canonical rates* at time `t`. For a genuinely Markovian process, every `γ_k(t)` is non-negative for all `t` — population only ever flows out of the system into the environment, never back. Quasar's `TCLSolver` computes `γ_k(t)` directly from finite differences of consecutive PTM eigenvalue snapshots and **never clamps a negative rate to zero**: a negative `γ_k(t)` is not a numerical artifact to be suppressed, it is the non-Markovian signature itself — a literal, sign-indefinite witness that the channel is momentarily returning coherence to the qubit.

## RHP Non-Markovianity Witness

Quasar computes the rate-based divisibility witness (whitepaper §4, Eq 13), which sums the total magnitude of canonical-rate negativity across all three Pauli channels:

```math
\mathcal{N}_{\mathrm{rate}}
  = \sum_k \int_{\gamma_k(t)<0} \bigl|\gamma_k(t)\bigr|\,\mathrm{d}t
```

The sum runs over k ∈ {x, y, z}; each integral runs only over time intervals where γ_k(t) < 0. This is the rate-based divisibility witness — not the full RHP entanglement-flow measure I^(E) — but equivalent in the sense that 𝒩_rate > 0 ⟺ ∃k,t: γ_k(t) < 0 ⟺ CP-divisibility fails ⟺ Markovian criterion violated. Its absolute magnitude is on a different scale from I^(E) and must not be compared with entanglement-based measures. `𝒩_rate = 0` is consistent with (but does not prove) Markovian dynamics; `𝒩_rate > 0` **certifies** information backflow occurred — a sufficient, constructive witness.

Quasar's `RHPWitness` computes this online, per link, as each new `CanonicalRates` snapshot arrives: it integrates any negative-γ contribution over the interval since the previous snapshot and adds it to a running total, also recording the timestamp of every positive-to-negative sign change. The code variable is `N_RHP`; its value equals 𝒩_rate.

## Reading the Non-Markovian Dashboard

The `NonMarkovPlot` panel plots the accumulated `N_RHP` value for a link against simulation time, continuously, with **sign-change markers** dropped at every timestamp where a canonical rate crossed from positive into negative — these markers are the moments backflow episodes began.

- **What triggers backflow in practice**: fast environmental transients are the usual cause — a rapid temperature swing, a seismic event, or a wind gust hitting the fiber faster than the memory kernel's characteristic decay time `τ`. A slowly-drifting environment, by contrast, tends to stay well within the Markovian-consistent (monotonic decay) regime even with a non-trivial kernel.
- **How to read the magnitude**: `N_RHP` is cumulative and monotonically non-decreasing by construction (it only ever adds non-negative contributions), so its absolute value is most useful as a *trend* — a link whose `N_RHP` is climbing steadily is experiencing recurring backflow events, while a flat `N_RHP` means the channel has settled into Markovian-consistent decay since the last sign change.
- **What a flat-zero witness means**: if `N_RHP` stays at exactly zero for the whole run, either the environment never produced a fast enough transient to overcome the kernel's smoothing, or the configured kernel (e.g. `GaussianKernel`, which is always non-negative) is structurally incapable of producing backflow regardless of the environment — check the kernel choice before concluding the physical channel itself is Markovian.
